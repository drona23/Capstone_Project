"""Deterministic benchmark data for offline and CI experiments.

The benchmark is synthetic by design. It encodes plausible hourly variation and
explicit carbon-water tradeoffs without representing a real operator or site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONE_PROFILES = {
    "Zone North": {
        "carbon_base": 0.18,
        "carbon_amplitude": 0.05,
        "carbon_phase": 4.0,
        "water_base": 1.10,
        "water_amplitude": 0.10,
        "pue": 1.12,
        "water_scarcity_index": 0.15,
    },
    "Zone West": {
        "carbon_base": 0.30,
        "carbon_amplitude": 0.12,
        "carbon_phase": 13.0,
        "water_base": 0.35,
        "water_amplitude": 0.05,
        "pue": 1.18,
        "water_scarcity_index": 0.80,
    },
    "Zone South": {
        "carbon_base": 0.48,
        "carbon_amplitude": 0.07,
        "carbon_phase": 18.0,
        "water_base": 0.45,
        "water_amplitude": 0.08,
        "pue": 1.20,
        "water_scarcity_index": 0.65,
    },
    "Zone East": {
        "carbon_base": 0.34,
        "carbon_amplitude": 0.06,
        "carbon_phase": 8.0,
        "water_base": 0.70,
        "water_amplitude": 0.07,
        "pue": 1.15,
        "water_scarcity_index": 0.30,
    },
}


def generate_environment(seed: int = 42, hours: int = 168) -> pd.DataFrame:
    """Generate one week of complete hourly environmental signals."""
    if hours < 48:
        raise ValueError("The benchmark requires at least 48 hours.")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=hours, freq="h")
    rows: list[dict[str, object]] = []
    hour_of_day = np.arange(hours) % 24
    slow_cycle = np.sin(2 * np.pi * np.arange(hours) / (24 * 7))

    for city, profile in ZONE_PROFILES.items():
        phase = float(profile["carbon_phase"])
        carbon = (
            float(profile["carbon_base"])
            + float(profile["carbon_amplitude"])
            * np.sin(2 * np.pi * (hour_of_day - phase) / 24)
            + 0.015 * slow_cycle
            + rng.normal(0, 0.008, hours)
        )
        water = (
            float(profile["water_base"])
            + float(profile["water_amplitude"])
            * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
            + rng.normal(0, 0.012, hours)
        )
        for timestamp, carbon_value, water_value in zip(
            timestamps, carbon, water, strict=True
        ):
            rows.append(
                {
                    "timestamp": timestamp,
                    "city": city,
                    "carbon_intensity_kg_per_kwh": max(0.01, float(carbon_value)),
                    "water_intensity_l_per_kwh": max(0.01, float(water_value)),
                }
            )
    return pd.DataFrame(rows)


def generate_dc_config(max_power_kw: float = 160.0) -> pd.DataFrame:
    """Generate four synthetic data center configurations."""
    return pd.DataFrame(
        [
            {
                "dc_id": f"dc_{index}",
                "city": city,
                "max_power_kw": float(max_power_kw),
                "pue": float(profile["pue"]),
                "water_usage_factor": 1.0,
                "water_scarcity_index": float(profile["water_scarcity_index"]),
            }
            for index, (city, profile) in enumerate(ZONE_PROFILES.items(), start=1)
        ]
    )


def generate_jobs(
    seed: int = 42,
    count: int = 180,
    hours: int = 168,
) -> pd.DataFrame:
    """Generate non-preemptive jobs with heterogeneous power, duration, and slack."""
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(seed)
    cities = list(ZONE_PROFILES)
    anchor = pd.Timestamp("2025-01-01")
    latest_arrival_hour = max(1, hours - 30)
    arrivals = rng.integers(0, latest_arrival_hour, count)
    durations = rng.integers(1, 7, count)
    slack = rng.integers(3, 25, count)
    deadlines = np.minimum(arrivals + durations + slack, hours)

    return pd.DataFrame(
        {
            "job_id": [f"job_{seed:03d}_{index:04d}" for index in range(count)],
            "origin_city": rng.choice(cities, count),
            "power_demand_kw": rng.integers(15, 46, count).astype(float),
            "duration_hours": durations.astype(int),
            "earliest_start": [
                anchor + pd.Timedelta(hours=int(hour)) for hour in arrivals
            ],
            "deadline": [
                anchor + pd.Timedelta(hours=int(hour)) for hour in deadlines
            ],
            "priority": rng.choice([0, 1, 2], count, p=[0.20, 0.55, 0.25]),
        }
    )


def generate_benchmark(
    seed: int = 42,
    jobs: int = 180,
    hours: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return environment, jobs, and data center frames for one seed."""
    return (
        generate_environment(seed=seed, hours=hours),
        generate_jobs(seed=seed, count=jobs, hours=hours),
        generate_dc_config(),
    )
