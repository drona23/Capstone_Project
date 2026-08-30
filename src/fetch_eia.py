"""
fetch_eia.py
------------
Fetch hourly electricity generation mix from the EIA (Energy Information
Administration) for the 7 grid regions covering your 28 AI data center locations.

EIA API is free to use. Register for a key at:
https://www.eia.gov/opendata/register.php  (instant approval)

Produces: data/processed/eia_generation_mix.csv

Columns per row:
  timestamp, ba_region, fuel_type, generation_mwh

Fuel types: COL (coal), NG (natural gas), NUC (nuclear),
            SUN (solar), WND (wind), WAT (hydro), OIL (petroleum), OTH (other)

Usage:
    python -m src.fetch_eia --api-key YOUR_KEY
    python -m src.fetch_eia --api-key YOUR_KEY --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from .data_loader import PROJECT_ROOT
except ImportError:
    from data_loader import PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "eia_generation_mix.csv"

EIA_BASE = "https://api.eia.gov/v2"
GEN_URL  = f"{EIA_BASE}/electricity/rto/fuel-type-data/data/"

# 7 grid regions covering all 28 DC locations
BA_REGIONS = ["MISO", "ERCO", "PJM", "WACM", "SOCO", "EPE", "NYIS"]

# EIA uses "ERCO" for ERCOT (different from WattTime which uses "ERCOT")
BA_RENAME = {"ERCO": "ERCOT"}

FUEL_TYPES = {
    "COL": "coal",
    "NG":  "natural_gas",
    "NUC": "nuclear",
    "SUN": "solar",
    "WND": "wind",
    "WAT": "hydro",
    "OIL": "petroleum",
    "OTH": "other",
}


def fetch_region_generation(
    api_key: str,
    ba: str,
    start: str,
    end: str,
    offset: int = 0,
) -> list[dict]:
    """Fetch one page of hourly generation data for a BA region."""
    params = {
        "api_key":          api_key,
        "frequency":        "hourly",
        "data[0]":          "value",
        "facets[respondent][]": ba,
        "start":            f"{start}T00",
        "end":              f"{end}T23",
        "sort[0][column]":  "period",
        "sort[0][direction]": "asc",
        "length":           5000,
        "offset":           offset,
    }

    resp = requests.get(GEN_URL, params=params, timeout=60)

    if resp.status_code == 429:
        print(" [rate limited, waiting 30s]", end="", flush=True)
        time.sleep(30)
        resp = requests.get(GEN_URL, params=params, timeout=60)

    if resp.status_code != 200:
        print(f" [WARN] EIA {ba}: {resp.status_code}: {resp.text[:120]}")
        return []

    return resp.json().get("response", {}).get("data", [])


def fetch_all_pages(api_key: str, ba: str, start: str, end: str) -> pd.DataFrame:
    """Page through EIA results (max 5000 per request) until exhausted."""
    all_rows = []
    offset = 0
    while True:
        rows = fetch_region_generation(api_key, ba, start, end, offset)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < 5000:
            break
        offset += 5000
        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def process_region(df: pd.DataFrame, ba: str) -> pd.DataFrame:
    """Clean and pivot raw EIA rows into wide format with one column per fuel type."""
    df = df.copy().rename(columns={"period": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["generation_mwh"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    fuel_column = next(
        (column for column in ("type", "fueltype", "type-name") if column in df.columns),
        None,
    )
    if fuel_column is None:
        raise ValueError("EIA response is missing a fuel type column.")
    df["fuel_code"] = df[fuel_column].astype(str).str.strip().str.upper()

    # Normalise fuel codes to our standard names
    label_aliases = {
        "COAL": "coal",
        "NATURAL GAS": "natural_gas",
        "NUCLEAR": "nuclear",
        "SOLAR": "solar",
        "WIND": "wind",
        "HYDRO": "hydro",
        "PETROLEUM": "petroleum",
        "OTHER": "other",
    }
    df["fuel"] = (
        df["fuel_code"].map(FUEL_TYPES).fillna(df["fuel_code"].map(label_aliases))
    ).fillna("other")

    # Pivot: one row per timestamp, one column per fuel
    pivoted = (
        df.groupby(["timestamp", "fuel"])["generation_mwh"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all expected fuel columns exist
    for fuel in FUEL_TYPES.values():
        if fuel not in pivoted.columns:
            pivoted[fuel] = 0.0

    pivoted["total_mwh"] = pivoted[[f for f in FUEL_TYPES.values()]].sum(axis=1)

    # Compute fuel share fractions from 0 to 1.
    for fuel in FUEL_TYPES.values():
        pivoted[f"{fuel}_share"] = (
            pivoted[fuel] / pivoted["total_mwh"].replace(0, float("nan"))
        ).fillna(0)

    canonical_ba = BA_RENAME.get(ba, ba)
    pivoted.insert(0, "ba_region", canonical_ba)

    return pivoted


def fetch_all(api_key: str, start: str, end: str) -> pd.DataFrame:
    all_frames: list[pd.DataFrame] = []

    print(f"Fetching EIA generation mix for {len(BA_REGIONS)} regions ({start} to {end})")
    print("Source: EIA Open Data API (free)\n")

    for ba in BA_REGIONS:
        print(f"  {ba}", end=" ... ", flush=True)
        raw = fetch_all_pages(api_key, ba, start, end)

        if raw.empty:
            print("no data")
            continue

        processed = process_region(raw, ba)
        all_frames.append(processed)
        fuel_columns = [column for column in processed if column.endswith("_share")]
        print(f"{len(processed):,} hourly rows | fuels: {fuel_columns}")
        time.sleep(0.5)

    if not all_frames:
        raise RuntimeError("No EIA data fetched. Check your API key and connectivity.")

    return pd.concat(all_frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EIA generation mix for 7 grid regions")
    parser.add_argument("--api-key", required=True, help="EIA API key (free at eia.gov/opendata)")
    parser.add_argument("--start",   default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output",  default=str(OUTPUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    df = fetch_all(args.api_key, args.start, args.end)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df):,} rows to {output_path}")
    print(f"Regions: {sorted(df['ba_region'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
