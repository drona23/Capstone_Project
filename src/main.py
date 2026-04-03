from __future__ import annotations

import argparse

if __package__:
    from .train import TARGET_SPECS, train_forecasting_models
else:
    from train import TARGET_SPECS, train_forecasting_models


def run_pipeline(
    data_path: str | None = None,
    weather_path: str | None = None,
    history_days: int = 365,
    test_horizon_hours: int = 48,
    forecast_horizon_hours: int = 48,
    random_state: int = 42,
    save_models: bool = True,
    city_limit: int | None = None,
):
    """Execute the full forecasting pipeline from loading through future predictions."""
    return train_forecasting_models(
        data_path=data_path,
        weather_path=weather_path,
        history_days=history_days,
        test_horizon_hours=test_horizon_hours,
        forecast_horizon_hours=forecast_horizon_hours,
        random_state=random_state,
        save_models=save_models,
        city_limit=city_limit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hybrid forecasting pipeline for WUE and CO2 intensity."
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
        help="Number of trailing historical days to use for training.",
    )
    parser.add_argument(
        "--test-horizon-hours",
        type=int,
        default=48,
        help="Number of holdout hours used for time-based evaluation.",
    )
    parser.add_argument(
        "--forecast-horizon-hours",
        type=int,
        default=48,
        help="Number of future hours to predict for each city.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible XGBoost training.",
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
        help="Disable saving model bundles and Prophet artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        data_path=args.data_path,
        weather_path=args.weather_path,
        history_days=args.history_days,
        test_horizon_hours=args.test_horizon_hours,
        forecast_horizon_hours=args.forecast_horizon_hours,
        random_state=args.random_state,
        save_models=not args.no_save_models,
        city_limit=args.city_limit,
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

    print("Forecasting split")
    print(f"  Train start: {results['split']['train_start']}")
    print(f"  Train end:   {results['split']['train_end']}")
    print(f"  Test start:  {results['split']['test_start']}")
    print(f"  Test end:    {results['split']['test_end']}")

    for target_key, target_result in results["targets"].items():
        print(f"\n{TARGET_SPECS[target_key]['label']} results")
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
    print("Forecast preview")
    print(results["forecast_preview"].to_string(index=False))


if __name__ == "__main__":
    main()
