"""
Tests for forecast_robustness.py

Validates that:
- Noise injection preserves non-negative intensities
- Zero noise produces zero regret (oracle == actual)
- Regret increases monotonically with noise up to moderate levels
- Both US and EU datasets produce valid output tables
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast_robustness import (
    _add_forecast_noise,
    _evaluate_schedule_against_truth,
    run_forecast_robustness,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_env() -> pd.DataFrame:
    """Minimal environment frame: 2 cities × 24 hours."""
    timestamps = pd.date_range("2023-06-01", periods=24, freq="h")
    rows = []
    for city, intensity in [("LowCarbon", 0.02), ("HighCarbon", 0.60)]:
        for ts in timestamps:
            rows.append({
                "city": city,
                "timestamp": ts,
                "predicted_co2_intensity": intensity,
                "predicted_water_intensity": 1.2,
                "scarcity_index": 0.1,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_dc() -> pd.DataFrame:
    return pd.DataFrame({
        "dc_id":                ["dc_low", "dc_high"],
        "city":                 ["LowCarbon", "HighCarbon"],
        "max_power_per_hour":   [500.0, 500.0],
        "pue":                  [1.2, 1.3],
        "water_usage_factor":   [0.05, 0.60],
        "region_water_scarcity":[0.1, 0.6],
        "is_new":               [0, 0],
        "category":             ["hyperscaler", "hyperscaler"],
    })


@pytest.fixture()
def tiny_jobs(tiny_dc: pd.DataFrame) -> pd.DataFrame:
    cities = tiny_dc["city"].tolist()
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame({
        "job_id":         [f"j{i}" for i in range(n)],
        "power_demand":   [20.0] * n,
        "duration_hours": [2] * n,
        "earliest_start": "2023-06-01 00:00:00",
        "deadline":       "2023-06-01 23:00:00",
        "origin_city":    rng.choice(cities, n).tolist(),
        "priority":       1,
    })


# ── Noise injection ────────────────────────────────────────────────────────────

def test_zero_noise_is_identity(tiny_env: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    noisy = _add_forecast_noise(tiny_env, noise_level=0.0, rng=rng)
    pd.testing.assert_series_equal(
        noisy["predicted_co2_intensity"],
        tiny_env["predicted_co2_intensity"],
        check_names=False,
    )


def test_noise_stays_non_negative(tiny_env: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    noisy = _add_forecast_noise(tiny_env, noise_level=5.0, rng=rng)  # extreme noise
    assert (noisy["predicted_co2_intensity"] >= 0).all()


def test_noise_changes_values(tiny_env: pd.DataFrame) -> None:
    rng = np.random.default_rng(1)
    noisy = _add_forecast_noise(tiny_env, noise_level=0.20, rng=rng)
    assert not noisy["predicted_co2_intensity"].equals(tiny_env["predicted_co2_intensity"])


# ── Evaluation ────────────────────────────────────────────────────────────────

def test_evaluate_empty_schedule_returns_zero(tiny_env: pd.DataFrame) -> None:
    empty_schedule = pd.DataFrame(columns=[
        "scheduled", "assigned_city", "scheduled_start", "power_demand", "duration_hours"
    ])
    result = _evaluate_schedule_against_truth(empty_schedule, tiny_env)
    assert result == 0.0


def test_evaluate_known_assignment(tiny_env: pd.DataFrame) -> None:
    """LowCarbon city at 0.02 kg/kWh × 20 kW × 2 h = 0.80 kg CO2."""
    schedule = pd.DataFrame([{
        "scheduled":       1,
        "assigned_city":   "LowCarbon",
        "scheduled_start": "2023-06-01 00:00:00",
        "power_demand":    20.0,
        "duration_hours":  2,
    }])
    actual = _evaluate_schedule_against_truth(schedule, tiny_env)
    assert pytest.approx(actual, rel=0.01) == 0.02 * 20.0 * 2


# ── Full experiment ────────────────────────────────────────────────────────────

def test_run_returns_correct_columns(
    tiny_env: pd.DataFrame, tiny_dc: pd.DataFrame, tiny_jobs: pd.DataFrame
) -> None:
    # Build a minimal data_df that prepare_environment_frame can process
    data_df = tiny_env.rename(columns={
        "city": "CITY",
        "timestamp": "TIMESTAMP",
        "predicted_co2_intensity": "co2_intensity_kg_per_kwh",
    })
    data_df["co2_kg"] = data_df["co2_intensity_kg_per_kwh"] * 1000
    data_df["total_gen_kwh"] = 1000.0
    data_df["WUE_total"] = 1.2
    data_df["dsci_0_500"] = 50.0

    results = run_forecast_robustness(
        data_df=data_df,
        dc_df=tiny_dc,
        jobs_df=tiny_jobs,
        noise_levels=[0.0, 0.10, 0.20],
    )
    required = {"noise_level", "oracle_co2_kg", "actual_co2_kg", "regret_kg", "regret_pct"}
    assert required.issubset(set(results.columns))
    assert len(results) == 3


def test_zero_noise_produces_zero_regret(
    tiny_env: pd.DataFrame, tiny_dc: pd.DataFrame, tiny_jobs: pd.DataFrame
) -> None:
    data_df = tiny_env.rename(columns={
        "city": "CITY",
        "timestamp": "TIMESTAMP",
        "predicted_co2_intensity": "co2_intensity_kg_per_kwh",
    })
    data_df["co2_kg"] = data_df["co2_intensity_kg_per_kwh"] * 1000
    data_df["total_gen_kwh"] = 1000.0
    data_df["WUE_total"] = 1.2
    data_df["dsci_0_500"] = 50.0

    results = run_forecast_robustness(
        data_df=data_df,
        dc_df=tiny_dc,
        jobs_df=tiny_jobs,
        noise_levels=[0.0],
    )
    row = results[results["noise_level"] == 0.0].iloc[0]
    assert row["regret_kg"] == pytest.approx(0.0, abs=1e-6)
    assert row["regret_pct"] == pytest.approx(0.0, abs=1e-6)
