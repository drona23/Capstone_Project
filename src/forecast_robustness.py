"""Forecast-error stress test with paired jobs and full-window accounting."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .scheduling import prepare_environment_frame, schedule_jobs

DEFAULT_NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)


def add_forecast_noise(
    env_df: pd.DataFrame,
    noise_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add zero-mean Gaussian error scaled to each city's mean intensity."""
    if noise_level < 0:
        raise ValueError("noise_level cannot be negative.")
    noisy = prepare_environment_frame(env_df)
    if noise_level == 0:
        return noisy
    city_mean = noisy.groupby("city")[
        "carbon_intensity_kg_per_kwh"
    ].transform("mean")
    error = rng.normal(0.0, noise_level * city_mean.to_numpy())
    noisy["carbon_intensity_kg_per_kwh"] = np.maximum(
        0.0,
        noisy["carbon_intensity_kg_per_kwh"].to_numpy() + error,
    )
    return noisy


def evaluate_schedule_against_truth(
    schedule_df: pd.DataFrame,
    truth_env_df: pd.DataFrame,
    dc_df: pd.DataFrame,
) -> float:
    """Compute carbon from every execution hour using realized intensity and PUE."""
    scheduled = schedule_df[schedule_df["scheduled"].eq(1)]
    if scheduled.empty:
        return 0.0

    truth = prepare_environment_frame(truth_env_df).set_index(["city", "timestamp"])
    config = dc_df.copy().rename(columns={"max_power_per_hour": "max_power_kw"})
    pue_by_dc = config.set_index("dc_id")["pue"].astype(float).to_dict()
    total = 0.0
    for _, row in scheduled.iterrows():
        start = pd.to_datetime(row["scheduled_start"], utc=True).tz_convert(None)
        duration = int(row["duration_hours"])
        timestamps = pd.date_range(start, periods=duration, freq="h")
        keys = pd.MultiIndex.from_product(
            [[str(row["assigned_city"])], timestamps], names=["city", "timestamp"]
        )
        intensities = truth.reindex(keys)["carbon_intensity_kg_per_kwh"]
        if intensities.isna().any():
            raise ValueError(
                f"Missing realized carbon data for job {row['job_id']} execution window."
            )
        pue = float(pue_by_dc[str(row["assigned_dc_id"])])
        total += float(row["power_demand_kw"]) * pue * float(intensities.sum())
    return total


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, 0.0
    return mean, float(1.96 * array.std(ddof=1) / math.sqrt(len(array)))


def run_forecast_robustness(
    env_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    noise_levels: tuple[float, ...] | list[float] = DEFAULT_NOISE_LEVELS,
    repetitions: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare noisy carbon-only schedules with a perfect-forecast reference."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    truth = prepare_environment_frame(env_df)
    reference = schedule_jobs(truth, jobs_df, dc_df, alpha=1.0, beta=0.0, gamma=0.0)
    rows: list[dict[str, float]] = []

    for noise_level in noise_levels:
        rmse_values: list[float] = []
        delta_values: list[float] = []
        coverage_values: list[float] = []
        paired_counts: list[float] = []
        for repetition in range(repetitions):
            rng = np.random.default_rng(seed + 10_000 * repetition + int(noise_level * 1000))
            forecast = add_forecast_noise(truth, float(noise_level), rng)
            schedule = schedule_jobs(
                forecast, jobs_df, dc_df, alpha=1.0, beta=0.0, gamma=0.0
            )

            reference_ids = set(
                reference.loc[reference["scheduled"].eq(1), "job_id"].astype(str)
            )
            schedule_ids = set(
                schedule.loc[schedule["scheduled"].eq(1), "job_id"].astype(str)
            )
            common_ids = reference_ids & schedule_ids
            reference_paired = reference[
                reference["job_id"].astype(str).isin(common_ids)
            ]
            schedule_paired = schedule[schedule["job_id"].astype(str).isin(common_ids)]
            reference_carbon = evaluate_schedule_against_truth(
                reference_paired, truth, dc_df
            )
            actual_carbon = evaluate_schedule_against_truth(schedule_paired, truth, dc_df)
            delta = (
                100.0 * (actual_carbon - reference_carbon) / reference_carbon
                if reference_carbon > 0
                else 0.0
            )
            rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            forecast["carbon_intensity_kg_per_kwh"].to_numpy()
                            - truth["carbon_intensity_kg_per_kwh"].to_numpy()
                        )
                        ** 2
                    )
                )
            )
            rmse_values.append(rmse)
            delta_values.append(delta)
            coverage_values.append(float(schedule["scheduled"].mean()))
            paired_counts.append(float(len(common_ids)))

        rmse_mean, rmse_ci = _mean_ci95(rmse_values)
        delta_mean, delta_ci = _mean_ci95(delta_values)
        coverage_mean, coverage_ci = _mean_ci95(coverage_values)
        rows.append(
            {
                "noise_level": float(noise_level),
                "forecast_rmse_mean": rmse_mean,
                "forecast_rmse_ci95": rmse_ci,
                "carbon_delta_pct_mean": delta_mean,
                "carbon_delta_pct_ci95": delta_ci,
                "coverage_mean": coverage_mean,
                "coverage_ci95": coverage_ci,
                "paired_jobs_mean": float(np.mean(paired_counts)),
                "repetitions": float(repetitions),
            }
        )
    return pd.DataFrame(rows)
