"""
sensitivity_analysis.py
-----------------------
Grid search over scheduler weight parameters (α, β) to map the Pareto
frontier between carbon emissions and water usage.

The scheduler optimizes: score = α×co2 + β×water + γ×distance
where α + β + γ = 1.

This analysis answers:
  "How does operator preference (carbon vs water vs latency) affect outcomes?"

Methodology:
    For each (α, β) combination on a grid:
    1. Schedule the same job batch.
    2. Record total CO2 and total water.
    3. Plot/report the Pareto frontier.

Output:
    - Console table
    - data/results/sensitivity_analysis_{region}.csv
    - Pareto-optimal points highlighted

Usage:
    python -m src.sensitivity_analysis
    python -m src.sensitivity_analysis --region EU --jobs 50
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
        summarize_schedule,
    )
    from .data_loader import PROJECT_ROOT, load_dataset
except ImportError:
    from scheduling import (
        load_dc_config,
        prepare_environment_frame,
        schedule_jobs,
        summarize_schedule,
    )
    from data_loader import PROJECT_ROOT, load_dataset

EU_DATA_PATH = PROJECT_ROOT / "data" / "eu_data_with_metrics.csv"
EU_DC_PATH   = PROJECT_ROOT / "data" / "templates" / "eu_dc_config.csv"


def _build_jobs(dc_df: pd.DataFrame, n: int, data_start: str, data_end: str, seed: int = 42) -> pd.DataFrame:
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


def _is_pareto_optimal(df: pd.DataFrame) -> pd.Series:
    """
    Mark rows that are Pareto-optimal on (co2, water) — minimizing both.
    A point is dominated if another point has strictly lower CO2 AND water.
    """
    dominated = np.zeros(len(df), dtype=bool)
    co2    = df["total_co2_kg"].values
    water  = df["total_water_l"].values
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if co2[j] <= co2[i] and water[j] <= water[i] and (co2[j] < co2[i] or water[j] < water[i]):
                dominated[i] = True
                break
    return pd.Series(~dominated, name="pareto_optimal")


def run_sensitivity_analysis(
    data_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    alpha_values: list[float] | None = None,
    beta_values:  list[float] | None = None,
) -> pd.DataFrame:
    """
    Grid search over (α, β) weights. γ = 1 - α - β (distance penalty).
    Points where α + β > 1 are skipped.
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if beta_values is None:
        beta_values  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    env = prepare_environment_frame(data_df)
    rows = []

    for alpha in alpha_values:
        for beta in beta_values:
            gamma = round(1.0 - alpha - beta, 4)
            if gamma < -1e-6:
                continue  # invalid combination

            sched   = schedule_jobs(env, jobs_df, dc_df, alpha=alpha, beta=beta, gamma=gamma)
            summary = summarize_schedule(sched)

            rows.append({
                "alpha":          alpha,
                "beta":           beta,
                "gamma":          round(gamma, 4),
                "weight_label":   f"α={alpha:.1f} β={beta:.1f} γ={gamma:.1f}",
                "total_co2_kg":   round(summary["total_expected_co2_kg"], 2),
                "total_water_l":  round(summary["total_expected_water_liters"], 2),
                "jobs_scheduled": int(summary["scheduled_jobs"]),
                "coverage":       round(summary["coverage"], 3),
            })

    result = pd.DataFrame(rows)
    result["pareto_optimal"] = _is_pareto_optimal(result)
    return result


def _print_report(df: pd.DataFrame, region: str) -> None:
    print("=" * 75)
    print(f"SENSITIVITY ANALYSIS — {region} GRID")
    print(f"{'α (carbon)':>10} {'β (water)':>10} {'γ (dist)':>9} {'CO2 (kg)':>12} {'Water (L)':>11} {'Pareto':>7}")
    print("-" * 75)

    for _, row in df.sort_values(["alpha", "beta"]).iterrows():
        pareto = "★" if row["pareto_optimal"] else ""
        print(
            f"{row['alpha']:>10.1f} {row['beta']:>10.1f} {row['gamma']:>9.1f} "
            f"{row['total_co2_kg']:>12,.1f} {row['total_water_l']:>11,.1f} {pareto:>7}"
        )

    pareto_df = df[df["pareto_optimal"]].sort_values("total_co2_kg")
    print("=" * 75)
    print(f"\nPareto-optimal configurations ({len(pareto_df)} points):")
    print(f"  {'Config':<22} {'CO2 (kg)':>12} {'Water (L)':>11}")
    print(f"  {'-'*47}")
    for _, row in pareto_df.iterrows():
        print(f"  {row['weight_label']:<22} {row['total_co2_kg']:>12,.1f} {row['total_water_l']:>11,.1f}")

    min_co2   = df.loc[df["total_co2_kg"].idxmin()]
    min_water = df.loc[df["total_water_l"].idxmin()]
    print(f"\nExtremes:")
    print(f"  Min CO2:   {min_co2['total_co2_kg']:>10,.1f} kg  at α={min_co2['alpha']:.1f}, β={min_co2['beta']:.1f}")
    print(f"  Min water: {min_water['total_water_l']:>10,.1f} L   at α={min_water['alpha']:.1f}, β={min_water['beta']:.1f}")

    co2_range   = df["total_co2_kg"].max()   - df["total_co2_kg"].min()
    water_range = df["total_water_l"].max()  - df["total_water_l"].min()
    print(f"\nSensitivity range:")
    print(f"  CO2:   {co2_range:>10,.1f} kg  ({100*co2_range/df['total_co2_kg'].max():.1f}% of max)")
    print(f"  Water: {water_range:>10,.1f} L   ({100*water_range/df['total_water_l'].max():.1f}% of max)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: α/β weight grid search"
    )
    parser.add_argument("--region", choices=["US", "EU"], default="US")
    parser.add_argument("--jobs",   type=int,   default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.region == "EU":
        if not EU_DATA_PATH.exists():
            print(f"[ERROR] EU dataset not found: {EU_DATA_PATH}")
            return
        data_df = pd.read_csv(EU_DATA_PATH)
        dc_df   = pd.read_csv(EU_DC_PATH)
        data_df = data_df[
            (data_df["TIMESTAMP"] >= "2023-01-01") &
            (data_df["TIMESTAMP"] <  "2023-07-01")
        ].copy()
        jobs_df = _build_jobs(dc_df, args.jobs, "2023-01-01", "2023-06-30")
    else:
        data_df = load_dataset()
        dc_df   = load_dc_config()
        data_df = data_df[
            (data_df["TIMESTAMP"] >= "2023-01-01") &
            (data_df["TIMESTAMP"] <  "2023-07-01")
        ].copy()
        jobs_df = _build_jobs(dc_df, args.jobs, "2023-01-01", "2023-06-30")

    print(f"Sensitivity Analysis — {args.region} | {args.jobs} jobs | 11×11 weight grid")
    print(f"  Dataset: {len(data_df):,} rows, {dc_df['city'].nunique()} cities\n")

    df = run_sensitivity_analysis(data_df, dc_df, jobs_df)
    _print_report(df, args.region)

    out_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "results" / f"sensitivity_analysis_{args.region.lower()}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
