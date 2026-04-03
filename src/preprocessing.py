from __future__ import annotations

import numpy as np
import pandas as pd


FUEL_COLUMNS = [
    "COAL",
    "HYDRO",
    "NATURALGAS",
    "NUCLEAR",
    "OTHER",
    "PETROLEUM",
    "SOLAR",
    "WIND",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute ratios while protecting against divide-by-zero."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def build_modeling_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare features and targets for model training."""
    data = df.copy()

    if "TIMESTAMP" in data.columns:
        data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
        data["hour"] = data["TIMESTAMP"].dt.hour
        data["day_of_week"] = data["TIMESTAMP"].dt.dayofweek
        data["month"] = data["TIMESTAMP"].dt.month

    required = {"WUE_total", "co2_kg", "total_gen_kwh"}
    missing_required = sorted(required - set(data.columns))
    if missing_required:
        raise ValueError(
            "Dataset is missing required target columns: "
            + ", ".join(missing_required)
        )

    data["carbon_intensity_kg_per_kwh"] = _safe_ratio(
        data["co2_kg"], data["total_gen_kwh"]
    )

    if "total_gen_mwh" in data.columns:
        for fuel_col in FUEL_COLUMNS:
            if fuel_col in data.columns:
                data[f"{fuel_col}_share"] = _safe_ratio(
                    data[fuel_col], data["total_gen_mwh"]
                )

    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    excluded = {"co2_kg", "WUE_total", "carbon_intensity_kg_per_kwh"}
    feature_cols = [col for col in numeric_cols if col not in excluded]

    if not feature_cols:
        raise ValueError("No numeric feature columns available after preprocessing.")

    modeling_df = data[feature_cols + ["WUE_total", "carbon_intensity_kg_per_kwh"]].copy()
    modeling_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    modeling_df.dropna(subset=["WUE_total", "carbon_intensity_kg_per_kwh"], inplace=True)

    X = modeling_df[feature_cols]
    y_wue = modeling_df["WUE_total"]
    y_carbon = modeling_df["carbon_intensity_kg_per_kwh"]
    return X, y_wue, y_carbon

