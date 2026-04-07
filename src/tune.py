"""
Hyperparameter tuning for XGBoost CO2 and WUE models using Optuna.

Why Optuna?
  Optuna uses Tree-structured Parzen Estimators (TPE) — a Bayesian method
  that learns which regions of the parameter space are promising from previous
  trials, rather than sampling blindly like RandomizedSearchCV.

Why not tune Prophet?
  Prophet has very few meaningful knobs and its seasonality parameters are
  already set by data characteristics (span_days >= 365 for yearly).
  XGBoost is where the real gains are.

How to run:
    python3 -m src.tune                        # tune both targets, 40 trials each
    python3 -m src.tune --target co2 --trials 60
    python3 -m src.tune --target wue --trials 20 --history-days 180

Best params are saved to models/best_params.json.
train.py automatically loads them if the file exists.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# optuna and xgboost are imported lazily inside tune_target() so that
# load_best_params() — which only reads a JSON file — can be called from
# train.py without requiring optuna to be installed.
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore[assignment,misc]

from .data_loader import PROJECT_ROOT, load_dataset
from .preprocessing import (
    build_feature_matrix,
    feature_columns_for_frame,
    prepare_forecasting_frame,
    select_recent_history,
    split_time_train_test,
)

MODELS_DIR = PROJECT_ROOT / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"

TARGET_COLUMNS = {
    "co2": "carbon_intensity_kg_per_kwh",
    "wue": "WUE_total",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

def _suggest_params(trial: Any) -> dict[str, Any]:  # trial: optuna.Trial
    """
    Define the XGBoost hyperparameter search space.

    Each call to trial.suggest_* picks one value from the given range.
    Optuna learns across trials which values tend to produce lower RMSE.

    Key parameters and what they control:
      n_estimators      — how many trees; more trees = more capacity but slower
      learning_rate     — how much each tree corrects the previous ones
                          (lower = needs more trees but generalises better)
      max_depth         — how complex each individual tree is
                          (higher = more detail but risks overfitting)
      subsample         — fraction of rows sampled per tree (reduces overfitting)
      colsample_bytree  — fraction of features sampled per tree
      min_child_weight  — minimum data points required to split a leaf
                          (higher = more conservative, less overfit)
      reg_lambda        — L2 regularisation on leaf weights
      reg_alpha         — L1 regularisation (sparse feature selection)
      gamma             — minimum loss reduction required to make a split
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }


# ---------------------------------------------------------------------------
# Time-series cross-validation
# ---------------------------------------------------------------------------

def _time_series_cv_rmse(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    params: dict[str, Any],
    n_splits: int = 3,
    random_state: int = 42,
) -> float:
    """
    Walk-forward cross-validation: always train on past, validate on future.

    Why not random k-fold?
      Random k-fold shuffles data, letting the model see future observations
      during training. For time series this is data leakage — the resulting
      RMSE is optimistic and won't reflect real-world performance.

    Walk-forward splits the timeline into n_splits equal windows:
      Split 1:  train=[t0..t1]  val=[t1..t2]
      Split 2:  train=[t0..t2]  val=[t2..t3]  (expanding window)
      Split 3:  train=[t0..t3]  val=[t3..t4]

    Returns the mean RMSE across all validation windows.
    """
    sorted_times = sorted(timestamps.unique())
    n = len(sorted_times)
    fold_size = n // (n_splits + 1)

    rmse_scores: list[float] = []

    for split_idx in range(1, n_splits + 1):
        val_start = sorted_times[split_idx * fold_size]
        val_end = (
            sorted_times[min((split_idx + 1) * fold_size, n) - 1]
            if split_idx < n_splits
            else sorted_times[-1]
        )

        train_mask = timestamps < val_start
        val_mask = (timestamps >= val_start) & (timestamps <= val_end)

        if train_mask.sum() < 10 or val_mask.sum() < 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
            random_state=random_state,
            early_stopping_rounds=30,
            **params,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
        rmse_scores.append(rmse)

    return float(np.mean(rmse_scores)) if rmse_scores else float("inf")


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    n_splits: int,
    random_state: int,
):
    """
    Returns a closure Optuna calls for each trial.

    A closure captures X, y, timestamps from the outer scope so Optuna
    only needs to pass the trial object — keeping the interface clean.
    """
    def objective(trial: Any) -> float:
        params = _suggest_params(trial)
        return _time_series_cv_rmse(
            X, y, timestamps, params,
            n_splits=n_splits,
            random_state=random_state,
        )
    return objective


# ---------------------------------------------------------------------------
# Main tuning entry point
# ---------------------------------------------------------------------------

def tune_target(
    target: str,
    data_path: str | None = None,
    n_trials: int = 40,
    history_days: int = 365,
    n_splits: int = 3,
    random_state: int = 42,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter search for one target (co2 or wue).

    Parameters
    ----------
    target         : 'co2' or 'wue'
    data_path      : path to dataset (uses default if None)
    n_trials       : how many Optuna trials to run
    history_days   : how many days of history to use for tuning
    n_splits       : number of walk-forward CV folds
    random_state   : reproducibility seed
    timeout_seconds: stop after this many seconds (None = no limit)

    Returns
    -------
    dict with best_params, best_rmse, n_trials_completed
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(f"target must be 'co2' or 'wue', got: {target!r}")

    # Lazy import — only needed when actually running tuning, not when loading best params
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    if XGBRegressor is None:
        raise ImportError("xgboost is required for tuning.")

    target_column = TARGET_COLUMNS[target]
    log.info(f"\nTuning {target} model ({n_trials} trials, {n_splits}-fold walk-forward CV)")

    # Load and prepare data
    raw_df = load_dataset(data_path)
    frame = prepare_forecasting_frame(raw_df)
    history = select_recent_history(
        frame,
        history_days=history_days,
        end_time=pd.Timestamp(frame["timestamp"].max()),
    )

    feature_columns = feature_columns_for_frame(history)
    X, _ = build_feature_matrix(history, feature_columns=feature_columns)
    y = pd.to_numeric(history[target_column], errors="coerce").fillna(0.0)
    timestamps = history["timestamp"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Run Optuna study (minimise RMSE)
    study = _optuna.create_study(
        direction="minimize",
        sampler=_optuna.samplers.TPESampler(seed=random_state),
        pruner=_optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        _make_objective(X, y, timestamps, n_splits=n_splits, random_state=random_state),
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=True,
    )

    best = study.best_trial
    log.info(f"  Best RMSE: {best.value:.6f}")
    log.info(f"  Best params: {best.params}")

    return {
        "target": target,
        "best_rmse": float(best.value),
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
        "baseline_params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "gamma": 0.0,
        },
    }


def tune_all_targets(
    data_path: str | None = None,
    n_trials: int = 40,
    history_days: int = 365,
    n_splits: int = 3,
    random_state: int = 42,
    timeout_seconds: int | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """
    Tune both CO2 and WUE models, save best params to models/best_params.json.
    train.py automatically loads this file if it exists.
    """
    results: dict[str, Any] = {}

    for target in ("co2", "wue"):
        result = tune_target(
            target=target,
            data_path=data_path,
            n_trials=n_trials,
            history_days=history_days,
            n_splits=n_splits,
            random_state=random_state,
            timeout_seconds=timeout_seconds,
        )
        results[target] = result

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            target: {
                "best_params": res["best_params"],
                "best_rmse": res["best_rmse"],
                "n_trials": res["n_trials_completed"],
            }
            for target, res in results.items()
        }
        BEST_PARAMS_PATH.write_text(json.dumps(output, indent=2))
        log.info(f"\nSaved best params to: {BEST_PARAMS_PATH}")

    return results


def load_best_params(target: str) -> dict[str, Any] | None:
    """
    Load saved best params for a target, or return None if not found.
    Called by train.py to use tuned params when available.
    """
    if not BEST_PARAMS_PATH.exists():
        return None
    data = json.loads(BEST_PARAMS_PATH.read_text())
    return data.get(target, {}).get("best_params")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters with Optuna."
    )
    parser.add_argument(
        "--target",
        choices=["co2", "wue", "both"],
        default="both",
        help="Which model to tune (default: both)",
    )
    parser.add_argument(
        "--trials", type=int, default=40,
        help="Number of Optuna trials per target (default: 40)",
    )
    parser.add_argument(
        "--history-days", type=int, default=365,
        help="Days of history to use for tuning (default: 365)",
    )
    parser.add_argument(
        "--splits", type=int, default=3,
        help="Number of walk-forward CV folds (default: 3)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Stop each target after this many seconds (default: no limit)",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Path to dataset (uses project default if omitted)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save best params to models/best_params.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = ["co2", "wue"] if args.target == "both" else [args.target]

    all_results: dict[str, Any] = {}
    for target in targets:
        result = tune_target(
            target=target,
            data_path=args.data_path,
            n_trials=args.trials,
            history_days=args.history_days,
            n_splits=args.splits,
            random_state=42,
            timeout_seconds=args.timeout,
        )
        all_results[target] = result

    if not args.no_save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            t: {
                "best_params": r["best_params"],
                "best_rmse": r["best_rmse"],
                "n_trials": r["n_trials_completed"],
            }
            for t, r in all_results.items()
        }
        BEST_PARAMS_PATH.write_text(json.dumps(output, indent=2))
        log.info(f"\nSaved: {BEST_PARAMS_PATH}")

    log.info("\n=== Tuning Summary ===")
    for target, result in all_results.items():
        baseline_rmse_approx = "—"
        log.info(
            f"  {target.upper():3s}  best RMSE: {result['best_rmse']:.6f}"
            f"  ({result['n_trials_completed']} trials)"
        )
        log.info(f"  Best params: {json.dumps(result['best_params'], indent=4)}")

    log.info("\nRe-train with tuned params:")
    log.info("  python3 -m src.train")


if __name__ == "__main__":
    main()
