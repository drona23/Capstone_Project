"""
cross_region_comparison.py
--------------------------
Compare scheduler performance on US vs EU grids using identical job batches.

Research question:
    Does carbon-aware workload routing provide greater benefit in a
    heterogeneous grid (EU, 40× CO2 spread) vs a homogeneous grid
    (US, 3× CO2 spread)?

Methodology:
    1. Sample a synthetic job batch representative of both regions.
    2. Schedule on US data (optimal + naive baselines).
    3. Schedule on EU data (optimal + naive baselines).
    4. Report: CO2 reduction %, water reduction %, coverage, and CO2 savings
       per job for both regions.

Output:
    - Console table
    - data/results/cross_region_comparison.csv

Usage:
    python -m src.cross_region_comparison
    python -m src.cross_region_comparison --jobs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .scheduling import (
        load_dc_config,
        prepare_environment_frame,
        schedule_jobs,
        schedule_jobs_naive,
        summarize_schedule,
    )
    from .data_loader import PROJECT_ROOT, load_dataset
except ImportError:
    from scheduling import (
        load_dc_config,
        prepare_environment_frame,
        schedule_jobs,
        schedule_jobs_naive,
        summarize_schedule,
    )
    from data_loader import PROJECT_ROOT, load_dataset

EU_DATA_PATH = PROJECT_ROOT / "data" / "eu_data_with_metrics.csv"
EU_DC_PATH   = PROJECT_ROOT / "data" / "templates" / "eu_dc_config.csv"
OUT_PATH     = PROJECT_ROOT / "data" / "results" / "cross_region_comparison.csv"


def _build_jobs(dc_df: pd.DataFrame, n: int, data_start: str, data_end: str, seed: int) -> pd.DataFrame:
    """Build a synthetic job batch whose origin cities come from dc_df."""
    rng = np.random.default_rng(seed)
    cities = dc_df["city"].tolist()
    return pd.DataFrame({
        "job_id":         [f"job_{i:04d}" for i in range(n)],
        "power_demand":   rng.choice([10.0, 20.0, 50.0, 100.0], n).tolist(),
        "duration_hours": rng.integers(2, 9, n).tolist(),
        "earliest_start": data_start,
        "deadline":       data_end,
        "origin_city":    rng.choice(cities, n).tolist(),
        "priority":       1,
    })


def _run_region(
    label: str,
    data_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict:
    """Schedule one region, return summary metrics for both optimal and naive."""
    env = prepare_environment_frame(data_df)

    # CO2 grid characteristics
    co2_values = env.groupby("city")["predicted_co2_intensity"].mean()
    co2_min  = co2_values.min()
    co2_max  = co2_values.max()
    co2_spread = co2_max / co2_min

    # Optimal schedule
    sched_opt   = schedule_jobs(env, jobs_df, dc_df, alpha=alpha, beta=beta, gamma=gamma)
    summary_opt = summarize_schedule(sched_opt)

    # Naive schedule (no spatial/temporal shifting)
    sched_naive   = schedule_jobs_naive(env, jobs_df, dc_df)
    summary_naive = summarize_schedule(sched_naive)

    opt_co2   = summary_opt["total_expected_co2_kg"]
    naive_co2 = summary_naive["total_expected_co2_kg"]
    opt_water   = summary_opt["total_expected_water_liters"]
    naive_water = summary_naive["total_expected_water_liters"]

    n_placed = int(summary_opt["scheduled_jobs"])
    co2_reduction_pct   = (1 - opt_co2   / naive_co2)   * 100 if naive_co2   > 0 else 0.0
    water_reduction_pct = (1 - opt_water / naive_water) * 100 if naive_water > 0 else 0.0
    co2_saved_per_job   = (naive_co2 - opt_co2) / n_placed if n_placed > 0 else 0.0

    # City allocation: fraction of jobs sent to top-3 cleanest cities
    scheduled_rows = sched_opt[sched_opt["scheduled"] == 1]
    top3_clean = co2_values.nsmallest(3).index.tolist()
    frac_to_clean = (
        scheduled_rows["assigned_city"].isin(top3_clean).sum() / n_placed
        if n_placed > 0 else 0.0
    )

    return {
        "region":               label,
        "n_cities":             len(dc_df),
        "co2_min_kg_kwh":       round(co2_min, 4),
        "co2_max_kg_kwh":       round(co2_max, 4),
        "co2_spread_x":         round(co2_spread, 1),
        "cleanest_city":        co2_values.idxmin(),
        "dirtiest_city":        co2_values.idxmax(),
        "jobs_total":           len(jobs_df),
        "jobs_scheduled":       n_placed,
        "coverage_pct":         round(summary_opt["coverage"] * 100, 1),
        "naive_co2_kg":         round(naive_co2, 2),
        "optimal_co2_kg":       round(opt_co2, 2),
        "co2_reduction_pct":    round(co2_reduction_pct, 1),
        "co2_saved_kg":         round(naive_co2 - opt_co2, 2),
        "co2_saved_per_job_kg": round(co2_saved_per_job, 2),
        "naive_water_l":        round(naive_water, 1),
        "optimal_water_l":      round(opt_water, 1),
        "water_reduction_pct":  round(water_reduction_pct, 1),
        "frac_to_top3_clean":   round(frac_to_clean * 100, 1),
        "top3_clean_cities":    ", ".join(top3_clean),
    }


def run_cross_region_comparison(
    n_jobs: int = 100,
    alpha: float = 0.7,
    beta: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    gamma = round(1 - alpha - beta, 4)

    # ── Load US data ──────────────────────────────────────────────────────────
    us_data = load_dataset()
    us_dc   = load_dc_config()
    # Use 6-month window matching both datasets
    us_window = us_data[
        (us_data["TIMESTAMP"] >= "2023-01-01") &
        (us_data["TIMESTAMP"] <  "2023-07-01")
    ].copy()
    us_jobs = _build_jobs(us_dc, n_jobs, "2023-01-01", "2023-06-30", seed)

    # ── Load EU data ──────────────────────────────────────────────────────────
    eu_data = pd.read_csv(EU_DATA_PATH)
    eu_dc   = pd.read_csv(EU_DC_PATH)
    eu_window = eu_data[
        (eu_data["TIMESTAMP"] >= "2023-01-01") &
        (eu_data["TIMESTAMP"] <  "2023-07-01")
    ].copy()
    eu_jobs = _build_jobs(eu_dc, n_jobs, "2023-01-01", "2023-06-30", seed)

    # ── Run both regions ──────────────────────────────────────────────────────
    us_result = _run_region("US", us_window, us_dc, us_jobs, alpha, beta, gamma)
    eu_result = _run_region("EU", eu_window, eu_dc, eu_jobs, alpha, beta, gamma)

    return pd.DataFrame([us_result, eu_result])


def _print_report(df: pd.DataFrame, n_jobs: int, alpha: float, beta: float) -> None:
    us = df[df["region"] == "US"].iloc[0]
    eu = df[df["region"] == "EU"].iloc[0]

    print("=" * 70)
    print(f"CROSS-REGION COMPARISON  —  {n_jobs} jobs  |  α={alpha}, β={beta}")
    print("=" * 70)

    print(f"\n{'GRID CHARACTERISTICS':}")
    print(f"  {'Metric':<30} {'US':>12} {'EU':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Cities':<30} {int(us['n_cities']):>12} {int(eu['n_cities']):>12}")
    print(f"  {'CO2 min (kg/kWh)':<30} {us['co2_min_kg_kwh']:>12.4f} {eu['co2_min_kg_kwh']:>12.4f}")
    print(f"  {'CO2 max (kg/kWh)':<30} {us['co2_max_kg_kwh']:>12.4f} {eu['co2_max_kg_kwh']:>12.4f}")
    print(f"  {'CO2 spread (max/min)':<30} {us['co2_spread_x']:>11.1f}× {eu['co2_spread_x']:>11.1f}×")
    print(f"  {'Cleanest city':<30} {us['cleanest_city']:>12} {eu['cleanest_city']:>12}")
    print(f"  {'Dirtiest city':<30} {us['dirtiest_city']:>12} {eu['dirtiest_city']:>12}")

    print(f"\n{'SCHEDULING RESULTS':}")
    print(f"  {'Metric':<30} {'US':>12} {'EU':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Jobs scheduled':<30} {int(us['jobs_scheduled']):>12} {int(eu['jobs_scheduled']):>12}")
    print(f"  {'Coverage':<30} {us['coverage_pct']:>11.1f}% {eu['coverage_pct']:>11.1f}%")
    print(f"  {'Naive CO2 (kg)':<30} {us['naive_co2_kg']:>12,.1f} {eu['naive_co2_kg']:>12,.1f}")
    print(f"  {'Optimal CO2 (kg)':<30} {us['optimal_co2_kg']:>12,.1f} {eu['optimal_co2_kg']:>12,.1f}")
    print(f"  {'CO2 reduction':<30} {us['co2_reduction_pct']:>11.1f}% {eu['co2_reduction_pct']:>11.1f}%")
    print(f"  {'CO2 saved per job (kg)':<30} {us['co2_saved_per_job_kg']:>12,.1f} {eu['co2_saved_per_job_kg']:>12,.1f}")
    print(f"  {'Water reduction':<30} {us['water_reduction_pct']:>11.1f}% {eu['water_reduction_pct']:>11.1f}%")
    print(f"  {'Jobs to top-3 clean cities':<30} {us['frac_to_top3_clean']:>11.1f}% {eu['frac_to_top3_clean']:>11.1f}%")

    print(f"\n{'KEY FINDING':}")
    eu_lift = eu['co2_reduction_pct'] - us['co2_reduction_pct']
    print(f"  EU routing saves {eu['co2_reduction_pct']:.1f}% CO2 vs {us['co2_reduction_pct']:.1f}% for US")
    print(f"  → Smart routing is {eu_lift:.1f} percentage points more impactful in the EU")
    print(f"  → EU CO2 spread ({eu['co2_spread_x']:.0f}×) vs US ({us['co2_spread_x']:.0f}×) explains the gap")
    print(f"  → Top-3 clean EU cities: {eu['top3_clean_cities']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="US vs EU cross-region scheduling comparison"
    )
    parser.add_argument("--jobs",  type=int,   default=100,  help="Jobs per region")
    parser.add_argument("--alpha", type=float, default=0.7,  help="Carbon weight")
    parser.add_argument("--beta",  type=float, default=0.2,  help="Water weight")
    parser.add_argument("--output", default=str(OUT_PATH),   help="Output CSV path")
    args = parser.parse_args()

    if not EU_DATA_PATH.exists():
        print(f"[ERROR] EU dataset not found: {EU_DATA_PATH}")
        print("  Run: python -m src.generate_eu_synthetic_data")
        print("  Then: python -m src.build_eu_dataset")
        return

    df = run_cross_region_comparison(
        n_jobs=args.jobs,
        alpha=args.alpha,
        beta=args.beta,
    )

    _print_report(df, args.jobs, args.alpha, args.beta)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
