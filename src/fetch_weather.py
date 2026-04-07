"""
fetch_weather.py
----------------
Fetch historical hourly weather data from Open-Meteo (free, no API key)
for the 28 real AI data center locations defined in city_coordinates.csv.

Produces: data/processed/dc_weather_historical.csv

Usage:
    python -m src.fetch_weather                        # 2022-2023 (default)
    python -m src.fetch_weather --start 2019-01-01 --end 2023-12-31
    python -m src.fetch_weather --start 2024-01-01 --end 2024-12-31
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

COORDS_PATH = PROJECT_ROOT / "data" / "reference" / "city_coordinates.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dc_weather_historical.csv"

# Open-Meteo free historical archive endpoint — no API key required
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables to fetch (maps to weather_loader.py column names)
HOURLY_VARIABLES = [
    "temperature_2m",       # → weather_temp (°C)
    "relative_humidity_2m", # → weather_rhum (%)
    "wind_speed_10m",       # → weather_wspd (km/h)
    "precipitation",        # → weather_prcp (mm)
    "surface_pressure",     # → weather_pres (hPa)
    "wind_direction_10m",   # → weather_wdir (°)
    "dew_point_2m",         # → weather_dwpt (°C)
]

COLUMN_RENAME = {
    "temperature_2m": "weather_temp",
    "relative_humidity_2m": "weather_rhum",
    "wind_speed_10m": "weather_wspd",
    "precipitation": "weather_prcp",
    "surface_pressure": "weather_pres",
    "wind_direction_10m": "weather_wdir",
    "dew_point_2m": "weather_dwpt",
}


def fetch_city_weather(
    city: str,
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "America/Chicago",
        "wind_speed_unit": "kmh",
    }

    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [WARN] {city}: request failed — {exc}")
        return None

    hourly = data.get("hourly", {})
    if "time" not in hourly:
        print(f"  [WARN] {city}: no hourly data returned")
        return None

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "TIMESTAMP"})
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.rename(columns=COLUMN_RENAME)
    df.insert(0, "city", city)

    # Derived features (match weather_loader.py conventions)
    df["weather_temp_dewpoint_gap"] = df["weather_temp"] - df["weather_dwpt"]
    df["weather_rain_flag"] = (df["weather_prcp"].fillna(0) > 0).astype(int)
    df["weather_humidity_temp_index"] = df["weather_temp"] * df["weather_rhum"] / 100.0

    return df


def fetch_all(start_date: str, end_date: str) -> pd.DataFrame:
    coords = pd.read_csv(COORDS_PATH)
    all_frames: list[pd.DataFrame] = []

    print(f"Fetching weather for {len(coords)} locations ({start_date} → {end_date})")
    print("Source: Open-Meteo Historical Archive (free, no API key)\n")

    for _, row in coords.iterrows():
        city = row["city"]
        lat = float(row["latitude"])
        lng = float(row["longitude"])
        print(f"  → {city} ({lat:.3f}, {lng:.3f})", end=" ... ", flush=True)

        df = fetch_city_weather(city, lat, lng, start_date, end_date)
        if df is not None:
            all_frames.append(df)
            print(f"{len(df):,} rows")
        else:
            print("skipped")

        # Be polite to the free API
        time.sleep(0.4)

    if not all_frames:
        raise RuntimeError("No weather data fetched — check connectivity.")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical weather for AI DC locations")
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    df = fetch_all(args.start, args.end)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df):,} rows × {len(df.columns)} columns → {output_path}")
    print(f"Cities covered: {df['city'].nunique()}")
    print(f"Date range: {df['TIMESTAMP'].min()} → {df['TIMESTAMP'].max()}")


if __name__ == "__main__":
    main()
