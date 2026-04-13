"""
build_eu_dataset.py
-------------------
Merge ENTSO-E generation mix + Open-Meteo weather into the same column
schema as base_data_with_metrics.csv — so the existing ML pipeline
(train.py, scheduling.py, app_backend.py) can load EU data without
any code changes.

Sources:
  - ENTSO-E Transparency Platform (hourly generation per fuel type)
    https://transparency.entsoe.eu/
  - Open-Meteo historical archive (free, no key needed)
    https://open-meteo.com/

Input files (data/processed/):
  - entso_e_generation_mix.csv   ← from fetch_entso_e.py
  - eu_weather_historical.csv    ← from fetch_weather.py --region eu

Output:
  - data/eu_data_with_metrics.csv

Usage:
    python -m src.build_eu_dataset
    python -m src.build_eu_dataset --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loader import PROJECT_ROOT
except ImportError:
    from data_loader import PROJECT_ROOT

ENTSO_PATH  = PROJECT_ROOT / "data" / "processed" / "entso_e_generation_mix.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "eu_weather_historical.csv"
COORDS_PATH  = PROJECT_ROOT / "data" / "reference"  / "eu_city_coordinates.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "eu_data_with_metrics.csv"

# IPCC 2014 lifecycle emission factors (kg CO2 / kWh) for EU
# Source: IPCC AR5 WG3, Annex III, Table A.III.2
EMISSION_FACTORS = {
    "coal":    0.8200,
    "gas":     0.4900,
    "oil":     0.6500,
    "nuclear": 0.0120,
    "hydro":   0.0040,
    "solar":   0.0450,
    "wind":    0.0110,
    "biomass": 0.2300,
    "other":   0.4000,
}

# Map ENTSO-E bidding zone → city name (must match eu_city_coordinates.csv)
ZONE_CITY_MAP = {
    "IE":    "Dublin",
    "NL":    "Amsterdam",
    "DE-LU": "Frankfurt",
    "GB":    "London",
    "FR":    "Paris",
    "SE3":   "Stockholm",
    "FI":    "Helsinki",
    "ES":    "Madrid",
    "PL":    "Warsaw",
    "DK2":   "Copenhagen",
}

# EU drought stress approximation (0=no scarcity, 1=extreme scarcity)
# Based on EU JRC MARS drought data (2023 annual average)
CITY_WATER_SCARCITY = {
    "Dublin":     0.05,
    "Amsterdam":  0.12,
    "Frankfurt":  0.18,
    "London":     0.15,
    "Paris":      0.22,
    "Stockholm":  0.03,
    "Helsinki":   0.02,
    "Madrid":     0.68,
    "Warsaw":     0.35,
    "Copenhagen": 0.10,
}


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_entso(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    df["generation_mwh"] = pd.to_numeric(df["generation_mwh"], errors="coerce").fillna(0.0)
    return df


def _load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = "TIMESTAMP" if "TIMESTAMP" in df.columns else "timestamp"
    df["timestamp"] = pd.to_datetime(df[col]).dt.floor("h")
    city_col = "CITY" if "CITY" in df.columns else "city"
    df = df.rename(columns={city_col: "city"})
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    return df[["city", "timestamp"] + weather_cols]


# ─── CO₂ intensity ────────────────────────────────────────────────────────────

def _compute_co2_intensity(entso_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hourly CO₂ intensity per ENTSO-E zone.

    co2_intensity = Σ (generation_fuel_i × emission_factor_i) / total_generation

    This is the standard marginal/average intensity methodology used in
    EU academic carbon-aware computing papers (e.g. Wiesner et al., 2021).
    """
    df = entso_df.copy()
    df["co2_kg_per_row"] = df["fuel_type"].map(EMISSION_FACTORS) * df["generation_mwh"] * 1000

    hourly = (
        df.groupby(["timestamp", "zone"])
        .agg(
            total_gen_mwh=("generation_mwh", "sum"),
            total_co2_kg=("co2_kg_per_row", "sum"),
        )
        .reset_index()
    )
    hourly["total_gen_kwh"] = hourly["total_gen_mwh"] * 1000.0
    hourly["co2_intensity_kg_per_kwh"] = (
        hourly["total_co2_kg"]
        / hourly["total_gen_kwh"].replace(0, float("nan"))
    )
    hourly["co2_kg"] = hourly["total_co2_kg"]
    return hourly


# ─── WUE estimation ───────────────────────────────────────────────────────────

def _estimate_wue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate Water Usage Effectiveness (WUE) from ambient temperature.

    Same formula as build_dataset.py so models trained on US data
    can be compared apples-to-apples with EU data.

    WUE = 1.0 + 0.008 × max(0, T - 15°C)
    This is a simplified engineering model consistent with published
    EU data center water efficiency benchmarks (EU Code of Conduct 2022).
    """
    df = df.copy()
    temp_col = next(
        (c for c in df.columns if "temperature" in c.lower()),
        None,
    )
    if temp_col:
        excess_heat = np.maximum(0.0, df[temp_col].fillna(15.0) - 15.0)
        df["WUE_total"] = 1.0 + 0.008 * excess_heat
    else:
        df["WUE_total"] = 1.2  # EU Code of Conduct 2022 average
    return df


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_eu_dataset(
    entso_path: Path = ENTSO_PATH,
    weather_path: Path = WEATHER_PATH,
    coords_path: Path = COORDS_PATH,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Merge ENTSO-E + weather data into a dataset with the same schema
    as base_data_with_metrics.csv.
    """
    # Load inputs
    entso_df = _load_entso(entso_path)

    if start:
        entso_df = entso_df[entso_df["timestamp"] >= pd.Timestamp(start)]
    if end:
        entso_df = entso_df[entso_df["timestamp"] <= pd.Timestamp(end)]

    # Compute CO₂ intensity per zone per hour
    intensity_df = _compute_co2_intensity(entso_df)

    # Map zone → city
    intensity_df["CITY"] = intensity_df["zone"].map(ZONE_CITY_MAP)
    intensity_df = intensity_df.dropna(subset=["CITY"])

    # Rename columns to match US schema
    intensity_df = intensity_df.rename(columns={"timestamp": "TIMESTAMP"})

    # Merge weather if available
    if weather_path.exists():
        weather_df = _load_weather(weather_path)
        weather_df = weather_df.rename(columns={"city": "CITY", "timestamp": "TIMESTAMP"})
        merged = intensity_df.merge(weather_df, on=["TIMESTAMP", "CITY"], how="left")
    else:
        print(f"  [INFO] Weather file not found at {weather_path}; skipping weather merge.")
        merged = intensity_df

    # Estimate WUE
    merged = _estimate_wue(merged)

    # Add water scarcity index (0–500 scale to match US dsci_0_500 column)
    merged["dsci_0_500"] = merged["CITY"].map(
        {city: v * 500 for city, v in CITY_WATER_SCARCITY.items()}
    ).fillna(250.0)

    # Add region label for traceability
    merged["region"] = "EU"

    # Sort and return
    merged = merged.sort_values(["CITY", "TIMESTAMP"]).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EU dataset from ENTSO-E + weather data")
    parser.add_argument("--start", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    print("Building EU dataset...")
    for path, label in [(ENTSO_PATH, "ENTSO-E"), (COORDS_PATH, "EU coordinates")]:
        if not path.exists():
            print(f"  [ERROR] {label} file not found: {path}")
            print(f"  Run 'python -m src.fetch_entso_e --token YOUR_TOKEN' first.")
            return

    df = build_eu_dataset(start=args.start, end=args.end)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nEU dataset summary:")
    print(f"  Rows:   {len(df):,}")
    print(f"  Cities: {sorted(df['CITY'].unique().tolist())}")
    print(f"  Dates:  {df['TIMESTAMP'].min()} → {df['TIMESTAMP'].max()}")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
