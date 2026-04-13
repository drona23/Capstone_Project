"""
Shared pytest fixtures.

Why a conftest.py?
  pytest auto-discovers this file and makes all fixtures here available
  to every test in the directory — no import needed.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal in-memory DataFrames (no file I/O, fast)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_jobs_df() -> pd.DataFrame:
    """Three jobs with the minimum columns the scheduler expects."""
    now = pd.Timestamp("2024-06-01 12:00")
    return pd.DataFrame([
        {
            "job_id": "job_a",
            "origin_city": "Dallas",
            "power_demand": 10.0,
            "duration_hours": 2,
            "earliest_start": now,
            "deadline": now + pd.Timedelta(hours=6),
            "priority": 0,
            "priority_label": "high",
        },
        {
            "job_id": "job_b",
            "origin_city": "Dallas",
            "power_demand": 5.0,
            "duration_hours": 1,
            "earliest_start": now,
            "deadline": now + pd.Timedelta(hours=4),
            "priority": 1,
            "priority_label": "medium",
        },
        {
            "job_id": "job_c",
            "origin_city": "Atlanta",
            "power_demand": 20.0,
            "duration_hours": 3,
            "earliest_start": now,
            "deadline": now + pd.Timedelta(hours=8),
            "priority": 2,
            "priority_label": "low",
        },
    ])


@pytest.fixture
def sample_dc_df() -> pd.DataFrame:
    """Two data centers with the minimum columns the scheduler expects."""
    return pd.DataFrame([
        {
            "dc_id": "dc_dallas_1",
            "city": "Dallas",
            "max_power_per_hour": 500.0,
            "pue": 1.2,
            "water_usage_factor": 1.0,
            "region_water_scarcity": 0.5,
            "is_new": 0,
        },
        {
            "dc_id": "dc_atlanta_1",
            "city": "Atlanta",
            "max_power_per_hour": 500.0,
            "pue": 1.1,
            "water_usage_factor": 0.9,
            "region_water_scarcity": 0.3,
            "is_new": 0,
        },
    ])


@pytest.fixture
def sample_env_df() -> pd.DataFrame:
    """Environment data (CO2, WUE, water scarcity) for two cities over 72 hours."""
    timestamps = pd.date_range("2024-06-01", periods=72, freq="h")
    data = []
    for i, ts in enumerate(timestamps):
        # Dallas: varies between 0.3 and 0.5 kg CO2/kWh
        dallas_co2 = 0.3 + 0.1 * (i % 24) / 24
        # Atlanta: varies between 0.2 and 0.4 kg CO2/kWh (generally cleaner)
        atlanta_co2 = 0.2 + 0.1 * (i % 24) / 24
        data.extend([
            {
                "TIMESTAMP": ts,
                "CITY": "Dallas",
                "co2_kg": dallas_co2 * 1000,  # Assuming 1000 kWh for this hour
                "total_gen_kwh": 1000.0,
                "WUE_total": 1.5,  # Water Usage Efficiency
                "dsci_0_500": 300.0,  # Water scarcity index
            },
            {
                "TIMESTAMP": ts,
                "CITY": "Atlanta",
                "co2_kg": atlanta_co2 * 1000,
                "total_gen_kwh": 1000.0,
                "WUE_total": 1.2,
                "dsci_0_500": 150.0,
            },
        ])
    return pd.DataFrame(data)


@pytest.fixture
def sample_schedule_df() -> pd.DataFrame:
    """A small schedule result DataFrame mimicking scheduler output."""
    return pd.DataFrame([
        {"job_id": "job_a", "scheduled": 1, "expected_co2_kg": 2.0, "expected_water_liters": 10.0},
        {"job_id": "job_b", "scheduled": 1, "expected_co2_kg": 1.0, "expected_water_liters": 5.0},
        {"job_id": "job_c", "scheduled": 0, "expected_co2_kg": 0.0, "expected_water_liters": 0.0},
    ])
