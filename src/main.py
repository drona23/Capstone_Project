from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split

try:
    from .data_loader import load_dataset
    from .evaluate import (
        compare_to_baseline,
        print_baseline_comparison,
        print_metrics,
        regression_metrics,
    )
    from .preprocessing import build_modeling_frame
    from .train import (
        build_baseline_pipeline,
        build_random_forest_pipeline,
        fit_regressor,
    )
    from .weather_loader import load_weather_dataset, merge_weather_features
except ImportError:
    from data_loader import load_dataset
    from evaluate import (
        compare_to_baseline,
        print_baseline_comparison,
        print_metrics,
        regression_metrics,
    )
    from preprocessing import build_modeling_frame
    from train import build_baseline_pipeline, build_random_forest_pipeline, fit_regressor
    from weather_loader import load_weather_dataset, merge_weather_features


def load_data_step(data_path: str | None = None):
    """Load the raw dataset."""
    return load_dataset(data_path)


def preprocess_data_step(df):
    """Transform the raw dataset into features and targets."""
    return build_modeling_frame(df)


def enrich_with_weather_step(df, weather_path: str | None = None):
    """Optionally enrich the main dataset with weather features."""
    if not weather_path:
        return df, None

    weather_df = load_weather_dataset(weather_path)
    return merge_weather_features(df, weather_df)


def split_data_step(X, y_wue, y_carbon, test_size: float, random_state: int):
    """Split features and both targets into train/test partitions."""
    return train_test_split(
        X,
        y_wue,
        y_carbon,
        test_size=test_size,
        random_state=random_state,
    )


def train_target_step(X_train, y_train, random_state: int):
    """Train the baseline and primary model for a single target."""
    baseline_model = fit_regressor(build_baseline_pipeline(), X_train, y_train)
    trained_model = fit_regressor(
        build_random_forest_pipeline(random_state=random_state), X_train, y_train
    )
    return baseline_model, trained_model


def evaluate_target_step(baseline_model, trained_model, X_test, y_test):
    """Evaluate baseline and trained models and compute RMSE comparison."""
    baseline_predictions = baseline_model.predict(X_test)
    trained_predictions = trained_model.predict(X_test)

    baseline_metrics = regression_metrics(y_test, baseline_predictions)
    trained_metrics = regression_metrics(y_test, trained_predictions)
    comparison = compare_to_baseline(trained_metrics, baseline_metrics)

    return {
        "baseline": baseline_metrics,
        "model": trained_metrics,
        "comparison": comparison,
    }


def run_pipeline(
    data_path: str | None = None,
    weather_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Execute the full pipeline from loading through evaluation."""
    df = load_data_step(data_path)
    df, weather_info = enrich_with_weather_step(df, weather_path=weather_path)
    X, y_wue, y_carbon = preprocess_data_step(df)
    (
        X_train,
        X_test,
        y_wue_train,
        y_wue_test,
        y_carbon_train,
        y_carbon_test,
    ) = split_data_step(X, y_wue, y_carbon, test_size, random_state)

    wue_baseline, wue_model = train_target_step(X_train, y_wue_train, random_state)
    carbon_baseline, carbon_model = train_target_step(
        X_train, y_carbon_train, random_state
    )

    return {
        "weather": weather_info,
        "wue": evaluate_target_step(wue_baseline, wue_model, X_test, y_wue_test),
        "carbon_intensity": evaluate_target_step(
            carbon_baseline, carbon_model, X_test, y_carbon_test
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end capstone ML pipeline."
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to dataset (.xlsx). Defaults to data/sample_dataset.xlsx.",
    )
    parser.add_argument(
        "--weather-path",
        default=None,
        help="Optional path to the weather master CSV or its directory.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split and model reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        data_path=args.data_path,
        weather_path=args.weather_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if results["weather"] is not None:
        print("Weather enrichment")
        print(f"  Join strategy: {results['weather']['join_strategy']}")
        print(
            "  Matched rows: "
            f"{results['weather']['matched_rows']}/{results['weather']['total_rows']}"
        )
        print(f"  Coverage: {results['weather']['coverage']:.2%}")
        print("")

    print("WUE results")
    print_metrics("WUE model", results["wue"]["model"])
    print_baseline_comparison(
        "WUE", results["wue"]["model"], results["wue"]["baseline"]
    )
    print("")
    print("Carbon intensity results")
    print_metrics(
        "Carbon intensity model", results["carbon_intensity"]["model"]
    )
    print_baseline_comparison(
        "Carbon intensity",
        results["carbon_intensity"]["model"],
        results["carbon_intensity"]["baseline"],
    )


if __name__ == "__main__":
    main()
