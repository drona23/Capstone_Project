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


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    """Pretty-print regression metrics."""
    print(f"{label} metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")

