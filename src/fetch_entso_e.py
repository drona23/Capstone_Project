"""
fetch_entso_e.py
----------------
Fetch hourly electricity generation mix from ENTSO-E Transparency Platform
for the 10 EU data center zones.

ENTSO-E API is free — register for a token at:
https://transparency.entsoe.eu/usrm/user/createPublicUser

Produces: data/processed/entso_e_generation_mix.csv

Columns per row:
  timestamp, zone, fuel_type, generation_mwh

Fuel types mapped from ENTSO-E psrType codes to common names:
  B01=biomass, B02=coal, B04=gas, B09=hydro, B10=hydro_pumped,
  B11=hydro_run, B16=solar, B18=wind_offshore, B19=wind_onshore,
  B13=nuclear, B17=other

Usage:
    python -m src.fetch_entso_e --token YOUR_TOKEN
    python -m src.fetch_entso_e --token YOUR_TOKEN --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

try:
    from .data_loader import PROJECT_ROOT
except ImportError:
    from data_loader import PROJECT_ROOT

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "entso_e_generation_mix.csv"

ENTSO_E_BASE = "https://web-api.tp.entsoe.eu/api"

# EIC codes for bidding zones covering the 10 EU DC cities
# Source: https://transparency.entsoe.eu/content/static_content/Static content/web api/Guide.html
ZONE_EIC = {
    "IE":    "10YIE-1001A00010",  # Ireland (Dublin)
    "NL":    "10YNL----------L",  # Netherlands (Amsterdam)
    "DE-LU": "10Y1001A1001A82H",  # Germany-Luxembourg (Frankfurt)
    "GB":    "10YGB----------A",  # Great Britain (London)
    "FR":    "10YFR-RTE------C",  # France (Paris)
    "SE3":   "10Y1001A1001A46L",  # Sweden SE3 (Stockholm)
    "FI":    "10YFI-1--------U",  # Finland (Helsinki)
    "ES":    "10YES-REE------0",  # Spain (Madrid)
    "PL":    "10YPL-AREA-----S",  # Poland (Warsaw)
    "DK2":   "10YDK-2--------M",  # Denmark East (Copenhagen)
}

# Map ENTSO-E psrType codes to readable fuel names
# Codes from: https://transparency.entsoe.eu/content/static_content/Static content/web api/Guide.html
PSR_TYPE_MAP = {
    "B01": "biomass",
    "B02": "coal",
    "B03": "gas",
    "B04": "gas",
    "B05": "coal",       # hard coal
    "B06": "oil",
    "B07": "oil",
    "B08": "oil",
    "B09": "hydro",      # hydro water reservoir
    "B10": "hydro",      # hydro pumped storage
    "B11": "hydro",      # hydro run-of-river
    "B12": "hydro",      # hydro marine
    "B13": "nuclear",
    "B14": "other",
    "B15": "other",
    "B16": "solar",
    "B17": "other",
    "B18": "wind",       # wind offshore
    "B19": "wind",       # wind onshore
    "B20": "other",
}

# EPA-style CO2 factors (kg CO2 per MWh) for ENTSO-E fuel types
# Source: IPCC 2014, European Environment Agency
CO2_KG_PER_MWH = {
    "coal":    820.0,
    "gas":     490.0,
    "oil":     650.0,
    "nuclear": 12.0,
    "hydro":   4.0,
    "solar":   45.0,
    "wind":    11.0,
    "biomass": 230.0,
    "other":   400.0,
}


def _parse_generation_xml(xml_text: str, zone: str) -> list[dict]:
    """Parse ENTSO-E ActualGenerationPerProductionType XML response."""
    root = ET.fromstring(xml_text)
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

    rows = []
    for ts_block in root.findall(".//ns:TimeSeries", ns):
        psr_el = ts_block.find(".//ns:psrType", ns)
        if psr_el is None:
            continue
        psr_type = psr_el.text or ""
        fuel = PSR_TYPE_MAP.get(psr_type, "other")

        period = ts_block.find("ns:Period", ns)
        if period is None:
            continue

        start_el = period.find("ns:timeInterval/ns:start", ns)
        resolution_el = period.find("ns:resolution", ns)
        if start_el is None or resolution_el is None:
            continue

        start_ts = pd.Timestamp(start_el.text, tz="UTC")
        resolution = resolution_el.text  # PT60M or PT15M

        if "60" in resolution:
            freq_h = 1
        elif "30" in resolution:
            freq_h = 0
        else:
            freq_h = 0  # 15-minute; skip sub-hourly

        if freq_h == 0:
            continue

        for point in period.findall("ns:Point", ns):
            pos_el = point.find("ns:position", ns)
            qty_el = point.find("ns:quantity", ns)
            if pos_el is None or qty_el is None:
                continue

            position = int(pos_el.text)
            try:
                generation_mwh = float(qty_el.text)
            except (TypeError, ValueError):
                continue

            hour_offset = position - 1
            timestamp = start_ts + pd.Timedelta(hours=hour_offset)
            rows.append(
                {
                    "timestamp": timestamp.tz_localize(None),
                    "zone": zone,
                    "fuel_type": fuel,
                    "generation_mwh": generation_mwh,
                }
            )
    return rows


def fetch_zone_generation(
    token: str,
    zone: str,
    eic: str,
    start: str,
    end: str,
) -> list[dict]:
    """Fetch hourly generation data for one ENTSO-E bidding zone."""
    params = {
        "securityToken": token,
        "documentType": "A75",          # Actual generation per production type
        "processType": "A16",           # Realised
        "in_Domain": eic,
        "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
        "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
    }

    try:
        response = requests.get(ENTSO_E_BASE, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [WARN] {zone}: request failed — {exc}")
        return []

    if "<html" in response.text.lower():
        print(f"  [WARN] {zone}: received HTML (likely rate-limited or auth error)")
        return []

    return _parse_generation_xml(response.text, zone)


def compute_co2_intensity(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Add co2_intensity_kg_per_kwh column to hourly generation totals."""
    gen_df = gen_df.copy()
    gen_df["co2_kg"] = gen_df["fuel_type"].map(CO2_KG_PER_MWH) * gen_df["generation_mwh"]

    hourly = (
        gen_df.groupby(["timestamp", "zone"])
        .agg(
            total_gen_mwh=("generation_mwh", "sum"),
            total_co2_kg=("co2_kg", "sum"),
        )
        .reset_index()
    )
    hourly["co2_intensity_kg_per_kwh"] = (
        hourly["total_co2_kg"] / (hourly["total_gen_mwh"] * 1000).replace(0, float("nan"))
    )
    return hourly


def fetch_all_zones(
    token: str,
    start: str = "2023-01-01",
    end: str = "2023-12-31",
    sleep_s: float = 1.0,
) -> pd.DataFrame:
    """Fetch generation data for all zones and return a combined DataFrame."""
    all_rows: list[dict] = []

    for zone, eic in ZONE_EIC.items():
        print(f"  Fetching {zone} ({eic})...")
        rows = fetch_zone_generation(token, zone, eic, start, end)
        print(f"    → {len(rows):,} rows")
        all_rows.extend(rows)
        time.sleep(sleep_s)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "zone", "fuel_type", "generation_mwh"])

    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EU generation data from ENTSO-E")
    parser.add_argument("--token", required=True, help="ENTSO-E API security token")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    print(f"Fetching ENTSO-E generation data: {args.start} → {args.end}")
    gen_df = fetch_all_zones(args.token, args.start, args.end)

    if gen_df.empty:
        print("No data retrieved. Check your token and date range.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gen_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(gen_df):,} rows → {output_path}")


if __name__ == "__main__":
    main()
