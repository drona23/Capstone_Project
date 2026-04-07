"""SHAP feature importance for CO2 and WUE XGBoost models."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
# shap is imported lazily inside compute_shap() so the module can be imported
# even when shap is not installed (e.g., running unit tests without the full
# requirements.txt, or importing api.py to check routes).
# Lazy import pattern: `import shap` moves inside the function body.

from .data_loader import PROJECT_ROOT
from .preprocessing import (
    FutureFeatureProfiles,
    build_feature_matrix,
    build_profile_feature_frame,
)

MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_LABELS: dict[str, str] = {
    "temperature_c": "Temperature (°C)",
    "humidity_pct": "Humidity (%)",
    "wind_speed_ms": "Wind Speed (m/s)",
    "scarcity_index": "Water Scarcity",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
    "day_of_year": "Day of Year",
    "is_weekend": "Is Weekend",
    "hour_sin": "Hour (sin)",
    "hour_cos": "Hour (cos)",
    "day_of_week_sin": "Day of Week (sin)",
    "day_of_week_cos": "Day of Week (cos)",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
}


@lru_cache(maxsize=2)
def _load_bundle(target: str) -> dict[str, Any]:
    """Load and cache a model bundle. target is 'co2' or 'wue'."""
    filename = "co2_model.pkl" if target == "co2" else "wue_model.pkl"
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return joblib.load(path)


@lru_cache(maxsize=2)
def _load_explainer(target: str) -> tuple[Any, dict[str, Any]]:
    """Build and cache a TreeExplainer for the direct XGBoost model."""
    import shap  # lazy import — only loaded when SHAP is actually needed
    bundle = _load_bundle(target)
    explainer = shap.TreeExplainer(bundle["direct_xgboost_model"])
    return explainer, bundle


def compute_shap(
    target: str,
    city: str,
    time: str,
    top_n: int = 8,
) -> dict[str, Any]:
    """
    Compute SHAP feature importances for a single city at a given timestamp.

    Parameters
    ----------
    target : 'co2' or 'wue'
    city   : data center city name (must match training data)
    time   : ISO 8601 timestamp string
    top_n  : number of top features to return (by |shap_value|)

    Returns
    -------
    dict with keys:
        city, target, prediction, base_value, features (list of dicts)
    """
    if target not in ("co2", "wue"):
        raise ValueError(f"target must be 'co2' or 'wue', got: {target!r}")

    explainer, bundle = _load_explainer(target)

    # Reconstruct FutureFeatureProfiles from the serialized dict stored in the bundle
    serialized = bundle["future_feature_profiles"]
    profiles = FutureFeatureProfiles(
        hourly_profiles=pd.DataFrame(serialized["hourly_profiles"]),
        city_profiles=pd.DataFrame(serialized["city_profiles"]),
        global_profile=serialized["global_profile"],
        profile_columns=serialized["profile_columns"],
        profile_history_days=int(serialized["profile_history_days"]),
    )

    # Build feature row for this city + timestamp using stored historical profiles
    time_city = pd.DataFrame([{"timestamp": pd.Timestamp(time).floor("h"), "city": city}])
    feature_frame = build_profile_feature_frame(time_city, profiles)

    X, _ = build_feature_matrix(
        feature_frame,
        feature_columns=bundle["feature_columns"],
        encoded_columns=bundle["encoded_columns"],
    )

    shap_values = explainer.shap_values(X)  # shape (1, n_features)
    shap_row = np.array(shap_values[0]) if isinstance(shap_values, list) else shap_values[0]

    # Collect base features only (skip one-hot city columns — city identity is obvious)
    encoded_columns: list[str] = bundle["encoded_columns"]
    features: list[dict[str, Any]] = []
    for i, col in enumerate(encoded_columns):
        if col.startswith("city_"):
            continue
        features.append({
            "name": FEATURE_LABELS.get(col, col),
            "raw_name": col,
            "value": float(X.iloc[0][col]),
            "shap_value": float(shap_row[i]),
        })

    features.sort(key=lambda f: abs(f["shap_value"]), reverse=True)

    return {
        "city": city,
        "target": target,
        "prediction": float(bundle["direct_xgboost_model"].predict(X)[0]),
        "base_value": float(explainer.expected_value)
        if not isinstance(explainer.expected_value, np.ndarray)
        else float(explainer.expected_value[0]),
        "features": features[:top_n],
    }
