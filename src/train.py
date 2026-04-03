from __future__ import annotations

import argparse

import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from .data_loader import PROJECT_ROOT, load_dataset
    from .evaluate import print_metrics, regression_metrics
    from .preprocessing import build_modeling_frame
except ImportError:
    from data_loader import PROJECT_ROOT, load_dataset
    from evaluate import print_metrics, regression_metrics
    from preprocessing import build_modeling_frame


def build_random_forest_pipeline(random_state: int = 42) -> Pipeline:
    """Create the primary regression model pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Create a simple mean-prediction baseline pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def fit_regressor(model: Pipeline, X_train, y_train) -> Pipeline:
    """Fit a regression pipeline and return it for reuse."""
    model.fit(X_train, y_train)
    return model


def train_models(
    data_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_models: bool = True,
) -> dict[str, dict[str, float]]:
    """Train separate models for WUE and carbon intensity."""
    df = load_dataset(data_path)
    X, y_wue, y_carbon = build_modeling_frame(df)

    X_train, X_test, y_wue_train, y_wue_test, y_carbon_train, y_carbon_test = train_test_split(
        X,
        y_wue,
        y_carbon,
        test_size=test_size,
        random_state=random_state,
    )

    wue_model = fit_regressor(
        build_random_forest_pipeline(random_state=random_state), X_train, y_wue_train
    )
    carbon_model = fit_regressor(
        build_random_forest_pipeline(random_state=random_state), X_train, y_carbon_train
    )

    y_wue_pred = wue_model.predict(X_test)
    y_carbon_pred = carbon_model.predict(X_test)

    metrics = {
        "wue": regression_metrics(y_wue_test, y_wue_pred),
        "carbon_intensity": regression_metrics(y_carbon_test, y_carbon_pred),
    }

    if save_models:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(wue_model, models_dir / "wue_model.joblib")
        joblib.dump(carbon_model, models_dir / "carbon_model.joblib")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train capstone regression models.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to dataset (.xlsx). Defaults to data/sample_dataset.xlsx.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split and model reproducibility.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Disable saving trained models to models/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_models(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        save_models=not args.no_save_models,
    )

    print_metrics("WUE", metrics["wue"])
    print("")
    print_metrics("Carbon intensity", metrics["carbon_intensity"])


if __name__ == "__main__":
    main()
