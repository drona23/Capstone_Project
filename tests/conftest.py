from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def environment() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=12, freq="h")
    rows = []
    for city, carbon, water in (
        ("Alpha", 0.20, 1.00),
        ("Beta", 0.50, 0.30),
    ):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    "city": city,
                    "carbon_intensity_kg_per_kwh": carbon + index * 0.01,
                    "water_intensity_l_per_kwh": water,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def dc_config() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dc_id": "dc_alpha",
                "city": "Alpha",
                "max_power_kw": 100.0,
                "pue": 1.10,
                "water_usage_factor": 1.0,
                "water_scarcity_index": 0.10,
            },
            {
                "dc_id": "dc_beta",
                "city": "Beta",
                "max_power_kw": 100.0,
                "pue": 1.20,
                "water_usage_factor": 1.0,
                "water_scarcity_index": 0.80,
            },
        ]
    )


@pytest.fixture
def jobs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": "job_1",
                "origin_city": "Alpha",
                "power_demand_kw": 20.0,
                "duration_hours": 2,
                "earliest_start": "2025-01-01 00:00:00+00:00",
                "deadline": "2025-01-01 05:00:00+00:00",
                "priority": 1,
            },
            {
                "job_id": "job_2",
                "origin_city": "Beta",
                "power_demand_kw": 30.0,
                "duration_hours": 3,
                "earliest_start": "2025-01-01 01:00:00+00:00",
                "deadline": "2025-01-01 08:00:00+00:00",
                "priority": 1,
            },
        ]
    )
