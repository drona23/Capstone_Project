"""
build_dataset.py
----------------
Merge EIA generation mix + Open-Meteo weather into one unified dataset
that the existing ML pipeline (train.py, scheduling.py) can read directly.

Carbon intensity is computed from EIA hourly fuel mix + EPA eGRID emission
factors — the standard methodology used in peer-reviewed research on
grid carbon accounting. No paid API required.

Sources:
  - EPA eGRID 2022 emission factors (kg CO2/kWh per fuel type)
    https://www.epa.gov/egrid
  - EIA hourly generation by fuel type (free API)
    https://www.eia.gov/opendata/
  - Open-Meteo historical weather archive (free, no key)
    https://open-meteo.com/

Input files (data/processed/):
  - eia_generation_mix.csv      ← from fetch_eia.py
  - dc_weather_historical.csv   ← from fetch_weather.py

Output:
  - data/base_data_with_metrics.csv  (main dataset the pipeline expects)

Usage:
    python -m src.build_dataset
    python -m src.build_dataset --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from .data_loader import PROJECT_ROOT
except ImportError:
    from data_loader import PROJECT_ROOT

EIA_PATH     = PROJECT_ROOT / "data" / "processed" / "eia_generation_mix.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "dc_weather_historical.csv"
COORDS_PATH  = PROJECT_ROOT / "data" / "reference"  / "city_coordinates.csv"
OUTPUT_PATH  = PROJECT_ROOT / "data" / "base_data_with_metrics.csv"

# ---------------------------------------------------------------------------
# EPA eGRID 2022 emission factors — kg CO2 per kWh generated
# Source: EPA eGRID Summary Tables 2022 (Table 1, national averages by fuel)
# https://www.epa.gov/egrid/download-data
# ---------------------------------------------------------------------------
EGRID_EMISSION_FACTORS = {
    "coal":         0.9552,   # kg CO2/kWh — highest emitter
    "natural_gas":  0.4031,   # kg CO2/kWh — cleaner than coal, still significant
    "petroleum":    0.6815,   # kg CO2/kWh
    "nuclear":      0.0055,   # kg CO2/kWh — lifecycle emissions only
    "hydro":        0.0067,   # kg CO2/kWh — lifecycle emissions only
    "solar":        0.0200,   # kg CO2/kWh — lifecycle emissions only
    "wind":         0.0073,   # kg CO2/kWh — lifecycle emissions only
    "other":        0.3500,   # kg CO2/kWh — conservative estimate for mixed
}

# BA region for each of the 28 DC locations
DC_BA_MAP = {
    "Abilene":            "ERCOT",
    "Shackelford County": "ERCOT",
    "Dona Ana County":    "WACM",
    "Lordstown":          "PJM",
    "Milam County":       "ERCOT",
    "Port Washington":    "MISO",
    "Mount Pleasant":     "MISO",
    "Atlanta":            "SOCO",
    "Wilbarger County":   "ERCOT",
    "Armstrong County":   "ERCOT",
    "Haskell County":     "ERCOT",
    "Pine Island":        "MISO",
    "Richland Parish":    "MISO",
    "Beaver Dam":         "MISO",
    "El Paso":            "EPE",
    "Bowling Green":      "PJM",
    "Carlisle":           "MISO",
    "Northern Indiana":   "MISO",
    "Memphis":            "MISO",
    "Lancaster":          "PJM",
    "Kenilworth":         "PJM",
    "West Texas":         "ERCOT",
    "Dallas":             "ERCOT",
    "Albany":             "NYIS",
    "Little Rock":        "MISO",
    "Meridian":           "MISO",
    "Rocky Mount":        "PJM",
    "Powhatan County":    "PJM",
}


def _load_eia(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).dt.floor("h")
    return df


def _load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["TIMESTAMP"]).dt.floor("h")
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    return df[["city", "timestamp"] + weather_cols]


def _compute_co2_intensity(eia_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hourly CO2 intensity (kg CO2/kWh) from EIA fuel mix shares
    and EPA eGRID emission factors.

    co2_intensity = Σ (share_fuel_i × emission_factor_i)

    This is the standard average emissions intensity approach used in
    academic carbon-aware computing research.
    """
    df = eia_df.copy()

    df["co2_intensity_kg_per_kwh"] = sum(
        df.get(f"{fuel}_share", pd.Series(0.0, index=df.index)) * factor
        for fuel, factor in EGRID_EMISSION_FACTORS.items()
    )

    # co2_kg = intensity × total generation converted to kWh
    df["total_gen_kwh"] = df["total_mwh"] * 1000.0
    df["co2_kg"] = df["co2_intensity_kg_per_kwh"] * df["total_gen_kwh"]

    return df


def _estimate_wue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate Water Usage Effectiveness from ambient temperature and humidity.

    WUE increases in hot/dry conditions (more evaporative cooling needed).
    Range: 1.0 (excellent) – 2.5 (poor).

    Derived from ASHRAE TC 9.9 guidelines and Green Grid WUE benchmarks.
    """
    temp = df.get("weather_temp", pd.Series(20.0, index=df.index))
    rhum = df.get("weather_rhum", pd.Series(50.0, index=df.index))

    wue = (
        1.2
        + 0.015 * (temp - 20).clip(lower=0)
        - 0.003 * (rhum - 50).clip(lower=0)
    ).clip(lower=1.0, upper=2.5)

    df["WUE_total"] = wue.round(4)
    return df


def build(start: str | None, end: str | None) -> pd.DataFrame:
    print("Loading source files...")

    for label, path in [("EIA mix", EIA_PATH), ("Weather", WEATHER_PATH)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} file not found: {path}\n"
                f"Run the corresponding fetch script first:\n"
                f"  EIA:     python -m src.fetch_eia --api-key YOUR_KEY\n"
                f"  Weather: python -m src.fetch_weather"
            )

    eia_df     = _load_eia(EIA_PATH)
    weather_df = _load_weather(WEATHER_PATH)
    coords_df  = pd.read_csv(COORDS_PATH)

    print(f"  EIA:     {len(eia_df):,} rows  ({eia_df['ba_region'].nunique()} regions)")
    print(f"  Weather: {len(weather_df):,} rows  ({weather_df['city'].nunique()} cities)")

    # --- Step 1: compute CO2 intensity from EIA fuel mix ---
    print("\nComputing CO2 intensity from EIA fuel mix + EPA eGRID factors...")
    eia_df = _compute_co2_intensity(eia_df)
    print(f"  CO2 range: {eia_df['co2_intensity_kg_per_kwh'].min():.4f} – "
          f"{eia_df['co2_intensity_kg_per_kwh'].max():.4f} kg/kWh")

    # --- Step 2: attach BA region to each DC city ---
    ba_map_df = pd.DataFrame(
        {"city": list(DC_BA_MAP.keys()), "ba_region": list(DC_BA_MAP.values())}
    )
    weather_df = weather_df.merge(ba_map_df, on="city", how="left")

    # --- Step 3: join EIA data onto weather by (ba_region, timestamp) ---
    share_cols = [c for c in eia_df.columns if c.endswith("_share")]
    eia_cols   = ["ba_region", "timestamp", "total_mwh", "total_gen_kwh",
                  "co2_kg", "co2_intensity_kg_per_kwh"] + share_cols

    merged = weather_df.merge(eia_df[eia_cols], on=["ba_region", "timestamp"], how="left")

    # --- Step 4: estimate WUE from weather ---
    print("Estimating WUE from temperature and humidity...")
    merged = _estimate_wue(merged)

    # --- Step 5: attach coordinates and metadata ---
    merged = merged.merge(
        coords_df[["city", "latitude", "longitude", "capacity_mw", "category"]],
        on="city",
        how="left",
    )

    # --- Step 6: rename to match pipeline column expectations ---
    merged = merged.rename(columns={
        "city":                    "CITY",
        "timestamp":               "TIMESTAMP",
        "co2_intensity_kg_per_kwh": "co2_intensity",
    })

    # --- Step 7: optional date filter ---
    if start:
        merged = merged[merged["TIMESTAMP"] >= pd.Timestamp(start)]
    if end:
        merged = merged[merged["TIMESTAMP"] <= pd.Timestamp(end)]

    merged = merged.sort_values(["CITY", "TIMESTAMP"]).reset_index(drop=True)

    before = len(merged)
    merged = merged.dropna(subset=["co2_kg", "total_gen_kwh", "WUE_total"])
    dropped = before - len(merged)
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing core values")

    # Clip any negative CO2 values (data anomalies from near-zero generation hours)
    neg = (merged["co2_intensity"] < 0).sum()
    if neg:
        merged["co2_intensity"] = merged["co2_intensity"].clip(lower=0)
        merged["co2_kg"] = merged["co2_intensity"] * merged["total_gen_kwh"]
        print(f"  Clipped {neg} negative CO2 rows to 0")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified dataset from EIA + Weather (EPA eGRID emission factors)"
    )
    parser.add_argument("--start",  default=None, help="Filter start date YYYY-MM-DD")
    parser.add_argument("--end",    default=None, help="Filter end date YYYY-MM-DD")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    df = build(args.start, args.end)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDataset built successfully")
    print(f"  Rows:         {len(df):,}")
    print(f"  Cities:       {df['CITY'].nunique()}")
    print(f"  Date range:   {df['TIMESTAMP'].min()} → {df['TIMESTAMP'].max()}")
    print(f"  CO2 range:    {df['co2_kg'].min():.1f} – {df['co2_kg'].max():.1f} kg")
    print(f"  WUE range:    {df['WUE_total'].min():.2f} – {df['WUE_total'].max():.2f}")
    print(f"\nSaved → {output_path}")
    print("Ready to run:  python -m src.train")


if __name__ == "__main__":
    main()
