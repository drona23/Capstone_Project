"""
fetch_entso_e.py
----------------
Fetch hourly electricity generation mix from ENTSO-E Transparency Platform
for the 10 EU data center zones.

ENTSO-E API is free to use. Register for a token at:
https://transparency.entsoe.eu/usrm/user/createPublicUser

Produces: data/processed/entso_e_generation_mix.csv

Columns per row:
  timestamp, zone, fuel_type, generation_mwh

Fuel types mapped from ENTSO-E psrType codes to common names:
  B01=biomass, B02/B03/B05=coal, B04=gas, B06/B07=oil,
  B10/B11/B12/B13=hydro, B14=nuclear, B17=solar,
  B19/B20=wind, and remaining categories=other.

Usage:
    python -m src.fetch_entso_e --token YOUR_TOKEN
    python -m src.fetch_entso_e --token YOUR_TOKEN --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from defusedxml import ElementTree as ET

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
# Codes follow the ENTSO-E Energy Identification Codes list.
PSR_TYPE_MAP = {
    "B01": "biomass",
    "B02": "coal",
    "B03": "coal",
    "B04": "gas",
    "B05": "coal",
    "B06": "oil",
    "B07": "oil",
    "B08": "coal",
    "B09": "other",
    "B10": "hydro",
    "B11": "hydro",
    "B12": "hydro",
    "B13": "hydro",
    "B14": "nuclear",
    "B15": "other",
    "B16": "other",
    "B17": "solar",
    "B18": "other",
    "B19": "wind",
    "B20": "wind",
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
        resolution = resolution_el.text or ""
        resolution_minutes = {"PT60M": 60, "PT30M": 30, "PT15M": 15}.get(
            resolution
        )
        if resolution_minutes is None:
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

            timestamp = start_ts + pd.Timedelta(
                minutes=(position - 1) * resolution_minutes
            )
            rows.append(
                {
                    "timestamp": timestamp.floor("h").tz_localize(None),
                    "zone": zone,
                    "fuel_type": fuel,
                    "generation_mwh": generation_mwh
                    * (resolution_minutes / 60.0),
                }
            )
    if not rows:
        return []
    hourly = (
        pd.DataFrame(rows)
        .groupby(["timestamp", "zone", "fuel_type"], as_index=False)[
            "generation_mwh"
        ]
        .sum()
    )
    return hourly.to_dict(orient="records")


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
        print(f"  [WARN] {zone}: request failed: {exc}")
        return []

    if "<html" in response.text.lower():
        print(f"  [WARN] {zone}: received HTML (likely rate-limited or auth error)")
        return []

    return _parse_generation_xml(response.text, zone)


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
        print(f"    {len(rows):,} rows")
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

    print(f"Fetching ENTSO-E generation data: {args.start} to {args.end}")
    gen_df = fetch_all_zones(args.token, args.start, args.end)

    if gen_df.empty:
        print("No data retrieved. Check your token and date range.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gen_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(gen_df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
