from __future__ import annotations

from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return RMSE, MAE, and R2 for a regression prediction."""
    return {
        "rmse": float(sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_to_baseline(
    model_metrics: dict[str, float], baseline_metrics: dict[str, float]
) -> dict[str, float]:
    """Compare model performance against a baseline using RMSE."""
    rmse_delta = baseline_metrics["rmse"] - model_metrics["rmse"]
    rmse_improvement_pct = (
        (rmse_delta / baseline_metrics["rmse"]) * 100
        if baseline_metrics["rmse"]
        else 0.0
    )
    return {
        "rmse_delta": float(rmse_delta),
        "rmse_improvement_pct": float(rmse_improvement_pct),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    """Pretty-print regression metrics."""
    print(f"{label} metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")


def print_baseline_comparison(
    label: str,
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> None:
    """Print RMSE comparison between the trained model and the baseline."""
    comparison = compare_to_baseline(model_metrics, baseline_metrics)
    direction = "better" if comparison["rmse_delta"] >= 0 else "worse"

    print(f"{label} RMSE comparison:")
    print(f"  Baseline RMSE: {baseline_metrics['rmse']:.4f}")
    print(f"  Model RMSE:    {model_metrics['rmse']:.4f}")
    print(f"  Difference:    {abs(comparison['rmse_delta']):.4f} ({direction} than baseline)")
    print(f"  Improvement:   {comparison['rmse_improvement_pct']:.2f}%")
