"""
forecast_robustness.py
----------------------
Evaluate how the carbon-aware scheduler degrades as CO2 forecast error
increases. Simulates the real-world scenario where a scheduler must act
on imperfect predictions (e.g., day-ahead forecasts vs realized values).

Methodology:
    1. Take realized hourly CO2 intensity as ground truth.
    2. Add Gaussian noise (σ = noise_level × mean) to simulate forecast error.
    3. Schedule jobs using the noisy forecast.
    4. Measure actual CO2 by applying ground-truth intensity to scheduled slots.
    5. Compare against the oracle (scheduled on perfect forecast).

Metric: "forecast regret" = (actual CO2 under noisy schedule) - (oracle CO2)
        expressed as percentage of oracle CO2.

Usage:
    python -m src.forecast_robustness
    python -m src.forecast_robustness --jobs 100 --region US
    python -m src.forecast_robustness --jobs 50 --region EU
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .scheduling import (
        prepare_environment_frame,
        prepare_scheduler_state,
        schedule_jobs,
        summarize_schedule,
        load_dc_config,
        load_jobs_dataset,
    )
    from .data_loader import PROJECT_ROOT, load_dataset
except ImportError:
    from scheduling import (
        prepare_environment_frame,
        prepare_scheduler_state,
        schedule_jobs,
        summarize_schedule,
        load_dc_config,
        load_jobs_dataset,
    )
    from data_loader import PROJECT_ROOT, load_dataset

# Noise levels to test: 0% = perfect forecast, 50% = very noisy
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50]

EU_DATA_PATH = PROJECT_ROOT / "data" / "eu_data_with_metrics.csv"
EU_DC_PATH   = PROJECT_ROOT / "data" / "templates" / "eu_dc_config.csv"


def _add_forecast_noise(
    env_df: pd.DataFrame,
    noise_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Perturb predicted_co2_intensity with multiplicative Gaussian noise.

    noise_level=0.10 means σ = 10% of each city's mean intensity.
    Clipped to [0, ∞) so intensity stays non-negative.
    """
    df = env_df.copy()
    if noise_level == 0.0:
        return df

    city_means = df.groupby("city")["predicted_co2_intensity"].transform("mean")
    sigma = noise_level * city_means
    noise = rng.normal(loc=0.0, scale=sigma)
    df["predicted_co2_intensity"] = np.maximum(0.0, df["predicted_co2_intensity"] + noise)
    return df


def _evaluate_schedule_against_truth(
    schedule_df: pd.DataFrame,
    truth_env: pd.DataFrame,
) -> float:
    """
    Given a schedule (assignments made on noisy forecast), compute the
    actual CO2 that would result using ground-truth intensities.

    Returns total actual CO2 in kg.
    """
    scheduled = schedule_df[schedule_df["scheduled"] == 1].copy()
    if scheduled.empty:
        return 0.0

    # Build a lookup: city × timestamp → true co2 intensity
    truth_lookup = truth_env.set_index(["city", "timestamp"])["predicted_co2_intensity"]

    actual_co2_total = 0.0
    for _, row in scheduled.iterrows():
        city = row["assigned_city"]
        slot = pd.Timestamp(row["scheduled_start"]).floor("h")
        power = float(row["power_demand"])
        duration = int(row["duration_hours"])

        # Sum true intensity over the duration window
        total_kwh = power * duration
        true_intensity = truth_lookup.get((city, slot), np.nan)
        if np.isnan(true_intensity):
            # Fall back to city mean if exact slot missing
            city_mask = truth_env["city"] == city
            true_intensity = truth_env.loc[city_mask, "predicted_co2_intensity"].mean()

        actual_co2_total += true_intensity * total_kwh

    return actual_co2_total


def run_forecast_robustness(
    data_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    noise_levels: list[float] = NOISE_LEVELS,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run forecast robustness experiment across noise levels.

    Returns a DataFrame with columns:
        noise_level, forecast_rmse, oracle_co2_kg, actual_co2_kg, regret_kg, regret_pct,
        jobs_scheduled, coverage
    """
    rng = np.random.default_rng(seed)

    # Ground-truth environment frame (noise=0)
    truth_env = prepare_environment_frame(data_df)
    truth_state = prepare_scheduler_state(truth_env, dc_df)

    rows = []
    for noise_level in noise_levels:
        # Create noisy forecast
        noisy_env = _add_forecast_noise(truth_env, noise_level, rng)

        # Schedule on noisy forecast
        noisy_state = prepare_scheduler_state(noisy_env, dc_df)
        schedule = schedule_jobs(
            noisy_env,
            jobs_df,
            dc_df,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            prepared_state=noisy_state,
        )
        if schedule.empty or "scheduled" not in schedule.columns:
            continue
        summary = summarize_schedule(schedule)

        # Measure actual CO2 using truth intensities
        actual_co2 = _evaluate_schedule_against_truth(schedule, truth_env)

        # Forecast RMSE relative to truth
        if noise_level == 0.0:
            oracle_co2 = actual_co2
            rmse = 0.0
        else:
            merged = truth_env.set_index(["city", "timestamp"])["predicted_co2_intensity"].rename("truth")
            noisy_indexed = noisy_env.set_index(["city", "timestamp"])["predicted_co2_intensity"]
            aligned = pd.concat([merged, noisy_indexed], axis=1).dropna()
            aligned.columns = ["truth", "forecast"]
            rmse = float(np.sqrt(((aligned["forecast"] - aligned["truth"]) ** 2).mean()))

        regret_kg = actual_co2 - oracle_co2
        regret_pct = (regret_kg / oracle_co2 * 100) if oracle_co2 > 0 else 0.0

        rows.append({
            "noise_level":     noise_level,
            "noise_pct":       f"{noise_level * 100:.0f}%",
            "forecast_rmse":   round(rmse, 6),
            "oracle_co2_kg":   round(oracle_co2, 2),
            "actual_co2_kg":   round(actual_co2, 2),
            "regret_kg":       round(regret_kg, 2),
            "regret_pct":      round(regret_pct, 2),
            "jobs_scheduled":  int(summary["scheduled_jobs"]),
            "coverage":        round(summary["coverage"], 3),
        })

    return pd.DataFrame(rows)


def _build_eu_jobs(dc_df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Build synthetic EU jobs matching EU city names."""
    rng = np.random.default_rng(seed)
    cities = dc_df["city"].tolist()
    return pd.DataFrame({
        "job_id":         [f"eu_job_{i}" for i in range(n)],
        "power_demand":   rng.choice([10.0, 20.0, 50.0, 100.0], n),
        "duration_hours": rng.integers(2, 9, n).tolist(),
        "earliest_start": "2023-06-01 00:00:00",
        "deadline":       "2023-06-30 00:00:00",
        "origin_city":    rng.choice(cities, n).tolist(),
        "priority":       1,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast robustness experiment for carbon-aware scheduler"
    )
    parser.add_argument(
        "--region",
        choices=["US", "EU"],
        default="US",
        help="Dataset region (default: US)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=100,
        help="Number of jobs to schedule (default: 100)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="Carbon weight (default: 0.7)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.2, help="Water weight (default: 0.2)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to CSV path (optional)",
    )
    args = parser.parse_args()

    print(f"Forecast Robustness Experiment")
    print(f"  Region: {args.region}")
    print(f"  Jobs:   {args.jobs}")
    print(f"  Weights: α={args.alpha}, β={args.beta}, γ={1 - args.alpha - args.beta:.2f}")
    print()

    if args.region == "EU":
        if not EU_DATA_PATH.exists():
            print(f"[ERROR] EU dataset not found: {EU_DATA_PATH}")
            print("  Run: python -m src.generate_eu_synthetic_data")
            print("  Then: python -m src.build_eu_dataset")
            return
        data_df = pd.read_csv(EU_DATA_PATH)
        dc_df   = pd.read_csv(EU_DC_PATH)
        jobs_df = _build_eu_jobs(dc_df, args.jobs)
        # Filter to a representative month
        data_df = data_df[
            (data_df["TIMESTAMP"] >= "2023-06-01") &
            (data_df["TIMESTAMP"] <  "2023-07-01")
        ].copy()
    else:
        data_df = load_dataset()
        dc_df   = load_dc_config()
        all_jobs = load_jobs_dataset()
        jobs_df = all_jobs.head(args.jobs).copy()
        # Shift job timestamps to align with dataset window
        data_start = pd.Timestamp(data_df["TIMESTAMP"].min())
        data_end   = pd.Timestamp(data_df["TIMESTAMP"].max())
        jobs_df["earliest_start"] = data_start
        jobs_df["deadline"]       = data_end
        data_df = data_df[
            (data_df["TIMESTAMP"] >= "2023-01-01") &
            (data_df["TIMESTAMP"] <  "2023-07-01")
        ].copy()

    print(f"  Dataset rows: {len(data_df):,}")
    print(f"  DCs:          {len(dc_df)}")
    print(f"  Jobs:         {len(jobs_df)}")
    print()

    results = run_forecast_robustness(
        data_df=data_df,
        dc_df=dc_df,
        jobs_df=jobs_df,
        alpha=args.alpha,
        beta=args.beta,
        gamma=round(1 - args.alpha - args.beta, 4),
    )

    # Print table
    print("=" * 75)
    print(f"{'Noise':>6} {'RMSE':>10} {'Oracle CO2':>12} {'Actual CO2':>12} {'Regret':>10} {'Regret%':>9} {'Coverage':>9}")
    print("-" * 75)
    oracle_co2 = results.loc[results["noise_level"] == 0.0, "oracle_co2_kg"].iloc[0]
    for _, row in results.iterrows():
        print(
            f"{row['noise_pct']:>6}  "
            f"{row['forecast_rmse']:>9.5f}  "
            f"{row['oracle_co2_kg']:>11.2f}  "
            f"{row['actual_co2_kg']:>11.2f}  "
            f"{row['regret_kg']:>9.2f}  "
            f"{row['regret_pct']:>8.1f}%  "
            f"{row['coverage']:>8.3f}"
        )
    print("=" * 75)

    # Robustness verdict
    p50_regret = results.loc[results["noise_level"] == 0.50, "regret_pct"].iloc[0]
    p20_regret = results.loc[results["noise_level"] == 0.20, "regret_pct"].iloc[0]
    print(f"\nKey findings:")
    print(f"  At 20% forecast error: +{p20_regret:.1f}% CO2 regret")
    print(f"  At 50% forecast error: +{p50_regret:.1f}% CO2 regret")
    if p20_regret < 10.0:
        print(f"  → Scheduler is ROBUST to typical day-ahead forecast errors (<10% regret at 20% noise)")
    else:
        print(f"  → Scheduler shows meaningful sensitivity to forecast error")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
