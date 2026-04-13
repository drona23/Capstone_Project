"""
Tests for cross_region_comparison.py and sensitivity_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cross_region_comparison import _build_jobs, _run_region
from src.sensitivity_analysis import (
    _build_jobs as sa_build_jobs,
    _is_pareto_optimal,
    run_sensitivity_analysis,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_dc() -> pd.DataFrame:
    return pd.DataFrame({
        "dc_id":                ["dc_low", "dc_mid", "dc_high"],
        "city":                 ["LowCarbon", "MidCarbon", "HighCarbon"],
        "max_power_per_hour":   [500.0, 500.0, 500.0],
        "pue":                  [1.1, 1.2, 1.3],
        "water_usage_factor":   [0.05, 0.30, 0.60],
        "region_water_scarcity":[0.05, 0.30, 0.60],
        "is_new":               [0, 0, 0],
        "category":             ["hyperscaler"] * 3,
    })


@pytest.fixture()
def tiny_data() -> pd.DataFrame:
    """24 hours × 3 cities with clearly different CO2 intensities."""
    timestamps = pd.date_range("2023-06-01", periods=24, freq="h")
    rows = []
    for city, co2, wue, dsci in [
        ("LowCarbon",  0.02, 1.1,  25.0),
        ("MidCarbon",  0.30, 1.2, 150.0),
        ("HighCarbon", 0.60, 1.3, 300.0),
    ]:
        for ts in timestamps:
            rows.append({
                "CITY":       city,
                "TIMESTAMP":  ts,
                "co2_kg":     co2 * 1000,
                "total_gen_kwh": 1000.0,
                "WUE_total":  wue,
                "dsci_0_500": dsci,
            })
    return pd.DataFrame(rows)


# ── _build_jobs ────────────────────────────────────────────────────────────────

def test_build_jobs_count(tiny_dc: pd.DataFrame) -> None:
    jobs = _build_jobs(tiny_dc, 30, "2023-06-01", "2023-06-30", seed=0)
    assert len(jobs) == 30


def test_build_jobs_required_columns(tiny_dc: pd.DataFrame) -> None:
    jobs = _build_jobs(tiny_dc, 10, "2023-06-01", "2023-06-30", seed=0)
    for col in ["job_id", "power_demand", "duration_hours", "origin_city", "priority"]:
        assert col in jobs.columns


def test_build_jobs_origin_cities_valid(tiny_dc: pd.DataFrame) -> None:
    jobs = _build_jobs(tiny_dc, 20, "2023-06-01", "2023-06-30", seed=0)
    valid_cities = set(tiny_dc["city"].tolist())
    assert set(jobs["origin_city"]).issubset(valid_cities)


# ── _run_region ────────────────────────────────────────────────────────────────

def test_run_region_returns_expected_keys(tiny_data: pd.DataFrame, tiny_dc: pd.DataFrame) -> None:
    jobs = _build_jobs(tiny_dc, 10, "2023-06-01", "2023-06-01 23:00:00", seed=0)
    result = _run_region("TEST", tiny_data, tiny_dc, jobs, alpha=0.7, beta=0.2, gamma=0.1)
    for key in ["region", "co2_reduction_pct", "optimal_co2_kg", "naive_co2_kg", "co2_spread_x"]:
        assert key in result


def test_run_region_optimal_le_naive(tiny_data: pd.DataFrame, tiny_dc: pd.DataFrame) -> None:
    """Optimal scheduler should never be worse than naive on CO2."""
    jobs = _build_jobs(tiny_dc, 15, "2023-06-01", "2023-06-01 23:00:00", seed=1)
    result = _run_region("TEST", tiny_data, tiny_dc, jobs, alpha=0.7, beta=0.2, gamma=0.1)
    assert result["optimal_co2_kg"] <= result["naive_co2_kg"] + 1e-6


def test_run_region_co2_spread_positive(tiny_data: pd.DataFrame, tiny_dc: pd.DataFrame) -> None:
    jobs = _build_jobs(tiny_dc, 5, "2023-06-01", "2023-06-01 23:00:00", seed=2)
    result = _run_region("TEST", tiny_data, tiny_dc, jobs, alpha=0.7, beta=0.2, gamma=0.1)
    assert result["co2_spread_x"] >= 1.0


# ── _is_pareto_optimal ─────────────────────────────────────────────────────────

def test_pareto_marks_dominated_correctly() -> None:
    df = pd.DataFrame({
        "total_co2_kg":  [100, 50, 200],
        "total_water_l": [200, 150, 100],
    })
    result = _is_pareto_optimal(df)
    # Row 0 (100, 200) is dominated by row 1 (50, 150) → not Pareto
    assert result.iloc[0] == False  # noqa: E712
    # Row 1 (50, 150) is Pareto — nothing strictly better on both
    assert result.iloc[1] == True   # noqa: E712
    # Row 2 (200, 100) is Pareto — best water
    assert result.iloc[2] == True   # noqa: E712


def test_pareto_single_row_is_optimal() -> None:
    df = pd.DataFrame({"total_co2_kg": [50.0], "total_water_l": [100.0]})
    result = _is_pareto_optimal(df)
    assert result.iloc[0] == True  # noqa: E712


# ── run_sensitivity_analysis ───────────────────────────────────────────────────

def test_sensitivity_output_shape(tiny_data: pd.DataFrame, tiny_dc: pd.DataFrame) -> None:
    jobs = sa_build_jobs(tiny_dc, 10, "2023-06-01", "2023-06-01 23:00:00")
    df = run_sensitivity_analysis(
        tiny_data, tiny_dc, jobs,
        alpha_values=[0.0, 0.5, 1.0],
        beta_values=[0.0, 0.5],
    )
    # Valid combinations: (0,0),(0,0.5),(0.5,0),(0.5,0.5),(1,0) = 5 rows
    assert len(df) == 5
    assert "pareto_optimal" in df.columns


def test_sensitivity_at_least_one_pareto(tiny_data: pd.DataFrame, tiny_dc: pd.DataFrame) -> None:
    jobs = sa_build_jobs(tiny_dc, 10, "2023-06-01", "2023-06-01 23:00:00")
    df = run_sensitivity_analysis(
        tiny_data, tiny_dc, jobs,
        alpha_values=[0.0, 0.5, 1.0],
        beta_values=[0.0, 0.5],
    )
    assert df["pareto_optimal"].any()
