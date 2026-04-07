from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __package__:
    from .runtime import ensure_openmp_runtime
else:
    from runtime import ensure_openmp_runtime

ensure_openmp_runtime()

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - handled at runtime.
    Prophet = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled at runtime.
    XGBRegressor = None

if __package__:
    from .data_loader import PROJECT_ROOT, load_dataset
    from .evaluate import compare_to_baseline, regression_metrics
    from .preprocessing import (
        FutureFeatureProfiles,
        build_feature_matrix,
        build_future_feature_frame,
        build_future_feature_profiles,
        build_profile_feature_frame,
        feature_columns_for_frame,
        prepare_forecasting_frame,
        select_recent_history,
        split_time_train_test,
    )
    from .tune import load_best_params
    from .weather_loader import load_weather_dataset, merge_weather_features
else:
    from data_loader import PROJECT_ROOT, load_dataset
    from evaluate import compare_to_baseline, regression_metrics
    from preprocessing import (
        FutureFeatureProfiles,
        build_feature_matrix,
        build_future_feature_frame,
        build_future_feature_profiles,
        build_profile_feature_frame,
        feature_columns_for_frame,
        prepare_forecasting_frame,
        select_recent_history,
        split_time_train_test,
    )
    from tune import load_best_params
    from weather_loader import load_weather_dataset, merge_weather_features


MODELS_DIR = PROJECT_ROOT / "models"
FORECAST_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PROPHET_MODELS_DIR = MODELS_DIR / "prophet_models"
TARGET_SPECS = {
    "co2": {
        "target_column": "carbon_intensity_kg_per_kwh",
        "model_filename": "co2_model.pkl",
        "label": "CO2 intensity",
    },
    "wue": {
        "target_column": "WUE_total",
        "model_filename": "wue_model.pkl",
        "label": "WUE",
    },
}


def _require_optional_dependencies() -> None:
    missing: list[str] = []
    if Prophet is None:
        missing.append("prophet")
    if XGBRegressor is None:
        missing.append("xgboost")
    if missing:
        raise ImportError(
            "Missing optional training dependencies: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements.txt`."
        )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown_city"


def build_xgboost_regressor(
    random_state: int = 42,
    target_key: str | None = None,
) -> XGBRegressor:
    """
    Build an XGBRegressor, using tuned hyperparameters if available.

    If models/best_params.json exists (written by src/tune.py), the tuned
    params for target_key are loaded and override the defaults below.
    If not, the hand-tuned defaults are used — training still works.
    """
    _require_optional_dependencies()

    # Default hand-tuned params
    params: dict = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
    }

    # Override with tuned params if available for this target
    if target_key is not None:
        tuned = load_best_params(target_key)
        if tuned:
            params.update(tuned)

    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
        random_state=random_state,
        **params,
    )


def build_prophet_model(has_yearly_pattern: bool) -> Prophet:
    _require_optional_dependencies()
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=has_yearly_pattern,
        seasonality_mode="additive",
        uncertainty_samples=0,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model


def enrich_with_weather_if_available(
    df: pd.DataFrame,
    weather_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not weather_path:
        return df, None

    weather_df = load_weather_dataset(weather_path)
    return merge_weather_features(df, weather_df)


def fit_prophet_models(
    train_frame: pd.DataFrame,
    target_column: str,
) -> tuple[dict[str, Prophet], dict[str, float], float]:
    models: dict[str, Prophet] = {}
    city_medians = (
        train_frame.groupby("city")[target_column].median().dropna().to_dict()
    )
    global_median = float(train_frame[target_column].median())

    for city, city_frame in train_frame.groupby("city"):
        prophet_frame = (
            city_frame[["timestamp", target_column]]
            .rename(columns={"timestamp": "ds", target_column: "y"})
            .dropna()
            .sort_values("ds")
        )
        if len(prophet_frame) < 24:
            continue

        span_days = (
            (prophet_frame["ds"].max() - prophet_frame["ds"].min()).total_seconds()
            / 86400.0
        )
        model = build_prophet_model(has_yearly_pattern=span_days >= 365.0)
        try:
            model.fit(prophet_frame)
            models[str(city)] = model
        except Exception:
            continue

    return models, {str(key): float(value) for key, value in city_medians.items()}, global_median


def predict_with_prophet(
    frame: pd.DataFrame,
    prophet_models: dict[str, Prophet],
    city_fallbacks: dict[str, float],
    global_fallback: float,
) -> pd.Series:
    predictions = pd.Series(index=frame.index, dtype=float)
    for city, city_frame in frame.groupby("city"):
        model = prophet_models.get(str(city))
        if model is None:
            continue
        ds_frame = pd.DataFrame({"ds": pd.to_datetime(city_frame["timestamp"], errors="coerce")})
        forecast = model.predict(ds_frame)
        predictions.loc[city_frame.index] = pd.to_numeric(forecast["yhat"], errors="coerce").values

    for city, city_frame in frame.groupby("city"):
        fallback = city_fallbacks.get(str(city), global_fallback)
        predictions.loc[city_frame.index] = predictions.loc[city_frame.index].fillna(fallback)
    return predictions.fillna(global_fallback)


def _fit_xgboost_target(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int,
    target_key: str | None = None,
) -> XGBRegressor:
    model = build_xgboost_regressor(random_state=random_state, target_key=target_key)
    model.fit(features, target)
    return model


def _serialize_profiles(profiles: FutureFeatureProfiles) -> dict[str, Any]:
    return {
        "hourly_profiles": profiles.hourly_profiles,
        "city_profiles": profiles.city_profiles,
        "global_profile": profiles.global_profile,
        "profile_columns": profiles.profile_columns,
        "profile_history_days": profiles.profile_history_days,
    }


def save_prophet_model_group(
    prophet_models: dict[str, Prophet],
    target_key: str,
    base_dir: Path = PROPHET_MODELS_DIR,
) -> Path:
    target_dir = base_dir / target_key
    target_dir.mkdir(parents=True, exist_ok=True)
    for city, model in prophet_models.items():
        joblib.dump(model, target_dir / f"{_slugify(city)}.pkl")
    return target_dir


def _train_single_target(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    evaluation_feature_frame: pd.DataFrame,
    final_history_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    target_key: str,
    random_state: int,
    save_models: bool,
) -> dict[str, Any]:
    target_column = TARGET_SPECS[target_key]["target_column"]
    feature_columns = feature_columns_for_frame(train_frame)

    X_train, encoded_columns = build_feature_matrix(train_frame, feature_columns=feature_columns)
    X_test, _ = build_feature_matrix(
        evaluation_feature_frame,
        feature_columns=feature_columns,
        encoded_columns=encoded_columns,
    )

    y_train = pd.to_numeric(train_frame[target_column], errors="coerce")
    y_test = pd.to_numeric(test_frame[target_column], errors="coerce")

    prophet_models, city_fallbacks, global_fallback = fit_prophet_models(train_frame, target_column)
    prophet_train_pred = predict_with_prophet(train_frame, prophet_models, city_fallbacks, global_fallback)
    prophet_test_pred = predict_with_prophet(test_frame, prophet_models, city_fallbacks, global_fallback)

    direct_model = _fit_xgboost_target(X_train, y_train, random_state=random_state, target_key=target_key)
    direct_test_pred = pd.Series(direct_model.predict(X_test), index=test_frame.index, dtype=float)

    residual_target = y_train - prophet_train_pred
    residual_model = _fit_xgboost_target(X_train, residual_target, random_state=random_state, target_key=target_key)
    hybrid_test_pred = prophet_test_pred + pd.Series(
        residual_model.predict(X_test),
        index=test_frame.index,
        dtype=float,
    )

    metrics = {
        "prophet": regression_metrics(y_test, prophet_test_pred),
        "xgboost": regression_metrics(y_test, direct_test_pred),
        "hybrid": regression_metrics(y_test, hybrid_test_pred),
    }
    comparisons = {
        "xgboost_vs_prophet": compare_to_baseline(metrics["xgboost"], metrics["prophet"]),
        "hybrid_vs_prophet": compare_to_baseline(metrics["hybrid"], metrics["prophet"]),
    }

    final_feature_columns = feature_columns_for_frame(final_history_frame)
    X_final, final_encoded_columns = build_feature_matrix(
        final_history_frame,
        feature_columns=final_feature_columns,
    )
    y_final = pd.to_numeric(final_history_frame[target_column], errors="coerce")

    final_prophet_models, final_city_fallbacks, final_global_fallback = fit_prophet_models(
        final_history_frame,
        target_column,
    )
    final_prophet_history = predict_with_prophet(
        final_history_frame,
        final_prophet_models,
        final_city_fallbacks,
        final_global_fallback,
    )
    final_direct_model = _fit_xgboost_target(X_final, y_final, random_state=random_state, target_key=target_key)
    final_residual_model = _fit_xgboost_target(
        X_final,
        y_final - final_prophet_history,
        random_state=random_state,
        target_key=target_key,
    )

    X_future, _ = build_feature_matrix(
        future_frame,
        feature_columns=final_feature_columns,
        encoded_columns=final_encoded_columns,
    )
    future_prophet_pred = predict_with_prophet(
        future_frame,
        final_prophet_models,
        final_city_fallbacks,
        final_global_fallback,
    )
    future_direct_pred = pd.Series(final_direct_model.predict(X_future), index=future_frame.index, dtype=float)
    future_hybrid_pred = future_prophet_pred + pd.Series(
        final_residual_model.predict(X_future),
        index=future_frame.index,
        dtype=float,
    )

    model_path = MODELS_DIR / TARGET_SPECS[target_key]["model_filename"]
    prophet_dir = PROPHET_MODELS_DIR / target_key
    saved_model_path: str | None = None
    saved_prophet_dir: str | None = None
    if save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        prophet_dir.mkdir(parents=True, exist_ok=True)
        profiles = build_future_feature_profiles(final_history_frame)
        bundle = {
            "target_key": target_key,
            "target_column": target_column,
            "direct_xgboost_model": final_direct_model,
            "hybrid_residual_model": final_residual_model,
            "feature_columns": final_feature_columns,
            "encoded_columns": final_encoded_columns,
            "city_fallbacks": final_city_fallbacks,
            "global_fallback": final_global_fallback,
            "future_feature_profiles": _serialize_profiles(profiles),
            "last_training_timestamp": pd.Timestamp(final_history_frame["timestamp"].max()).isoformat(),
        }
        joblib.dump(bundle, model_path)
        save_prophet_model_group(final_prophet_models, target_key, base_dir=PROPHET_MODELS_DIR)
        saved_model_path = str(model_path)
        saved_prophet_dir = str(prophet_dir)

    forecast_frame = future_frame[["timestamp", "city"]].copy()
    forecast_frame[f"prophet_{target_key}"] = future_prophet_pred.values
    forecast_frame[f"xgboost_{target_key}"] = future_direct_pred.values
    forecast_frame[f"hybrid_{target_key}"] = future_hybrid_pred.values

    return {
        "metrics": metrics,
        "comparisons": comparisons,
        "forecast": forecast_frame,
        "saved_model_path": saved_model_path,
        "saved_prophet_dir": saved_prophet_dir,
    }


def train_forecasting_models(
    data_path: str | None = None,
    weather_path: str | None = None,
    history_days: int = 365,
    test_horizon_hours: int = 48,
    forecast_horizon_hours: int = 48,
    random_state: int = 42,
    save_models: bool = True,
    city_limit: int | None = None,
) -> dict[str, Any]:
    """Train XGBoost, Prophet, and hybrid forecasting models, then emit city forecasts."""
    _require_optional_dependencies()

    raw_df = load_dataset(data_path)
    enriched_df, weather_info = enrich_with_weather_if_available(raw_df, weather_path=weather_path)
    forecasting_frame = prepare_forecasting_frame(enriched_df, city_limit=city_limit)

    split = split_time_train_test(
        forecasting_frame,
        test_horizon_hours=test_horizon_hours,
        history_days=history_days,
    )
    final_history = select_recent_history(
        forecasting_frame,
        history_days=history_days,
        end_time=pd.Timestamp(forecasting_frame["timestamp"].max()),
    )
    future_profiles = build_future_feature_profiles(final_history)
    evaluation_feature_frame = build_profile_feature_frame(
        split.test[["timestamp", "city"]],
        profiles=build_future_feature_profiles(split.train),
    )
    future_frame = build_future_feature_frame(
        final_history,
        horizon_hours=forecast_horizon_hours,
        profiles=future_profiles,
    )

    target_results = {
        target_key: _train_single_target(
            train_frame=split.train,
            test_frame=split.test,
            evaluation_feature_frame=evaluation_feature_frame,
            final_history_frame=final_history,
            future_frame=future_frame,
            target_key=target_key,
            random_state=random_state,
            save_models=save_models,
        )
        for target_key in TARGET_SPECS
    }

    forecasts = (
        target_results["co2"]["forecast"]
        .merge(target_results["wue"]["forecast"], on=["timestamp", "city"], how="inner")
        .sort_values(["timestamp", "city"])
        .reset_index(drop=True)
    )

    FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_output_path = FORECAST_OUTPUT_DIR / f"city_forecasts_{int(forecast_horizon_hours)}h.csv"
    forecasts.to_csv(forecast_output_path, index=False)

    return {
        "weather": weather_info,
        "history_days": int(history_days),
        "test_horizon_hours": int(test_horizon_hours),
        "forecast_horizon_hours": int(forecast_horizon_hours),
        "split": {
            "train_start": pd.Timestamp(split.train["timestamp"].min()).isoformat(),
            "train_end": pd.Timestamp(split.train["timestamp"].max()).isoformat(),
            "test_start": pd.Timestamp(split.test_start).isoformat(),
            "test_end": pd.Timestamp(split.test_end).isoformat(),
        },
        "targets": {
            target_key: {
                "metrics": result["metrics"],
                "comparisons": result["comparisons"],
                "saved_model_path": result["saved_model_path"],
                "saved_prophet_dir": result["saved_prophet_dir"],
            }
            for target_key, result in target_results.items()
        },
        "forecast_path": str(forecast_output_path),
        "forecast_preview": forecasts.head(10),
    }


def train_models(
    data_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_models: bool = True,
) -> dict[str, Any]:
    """Backward-compatible wrapper that now delegates to the forecasting pipeline."""
    test_horizon_hours = max(24, int(round(test_size * 24 * 10)))
    return train_forecasting_models(
        data_path=data_path,
        history_days=365,
        test_horizon_hours=test_horizon_hours,
        forecast_horizon_hours=48,
        random_state=random_state,
        save_models=save_models,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost, Prophet, and hybrid forecasting models for CO2 and WUE."
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to the master dataset (.zip, .parquet, .csv, or .xlsx).",
    )
    parser.add_argument(
        "--weather-path",
        default=None,
        help="Optional path to external weather data for feature enrichment.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=365,
        help="How many trailing historical days to use for training.",
    )
    parser.add_argument(
        "--test-horizon-hours",
        type=int,
        default=48,
        help="How many final hours to reserve for evaluation.",
    )
    parser.add_argument(
        "--forecast-horizon-hours",
        type=int,
        default=48,
        help="How many future hours to forecast per city.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for XGBoost reproducibility.",
    )
    parser.add_argument(
        "--city-limit",
        type=int,
        default=None,
        help="Optional number of cities to train for faster smoke tests.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Disable saving trained model bundles and Prophet artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_forecasting_models(
        data_path=args.data_path,
        weather_path=args.weather_path,
        history_days=args.history_days,
        test_horizon_hours=args.test_horizon_hours,
        forecast_horizon_hours=args.forecast_horizon_hours,
        random_state=args.random_state,
        save_models=not args.no_save_models,
        city_limit=args.city_limit,
    )

    print("Forecast training complete")
    print(
        f"  Train window: {results['split']['train_start']} to {results['split']['train_end']}"
    )
    print(
        f"  Test window:  {results['split']['test_start']} to {results['split']['test_end']}"
    )
    for target_key, target_result in results["targets"].items():
        print(f"\n{TARGET_SPECS[target_key]['label']} models")
        for model_name, metrics in target_result["metrics"].items():
            print(
                f"  {model_name:>7} -> RMSE {metrics['rmse']:.4f}, "
                f"MAE {metrics['mae']:.4f}, R2 {metrics['r2']:.4f}"
            )
        print(
            "  Hybrid vs Prophet RMSE improvement: "
            f"{target_result['comparisons']['hybrid_vs_prophet']['rmse_improvement_pct']:.2f}%"
        )
        if not args.no_save_models:
            print(f"  Saved model bundle: {target_result['saved_model_path']}")
            print(f"  Saved Prophet models: {target_result['saved_prophet_dir']}")
    print(f"\nSaved forecasts to: {results['forecast_path']}")


if __name__ == "__main__":
    main()
