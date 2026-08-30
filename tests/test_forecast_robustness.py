from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast_robustness import (
    add_forecast_noise,
    evaluate_schedule_against_truth,
    run_forecast_robustness,
)


def test_noise_is_nonnegative(environment) -> None:
    noisy = add_forecast_noise(environment, 4.0, np.random.default_rng(2))
    assert (noisy["carbon_intensity_kg_per_kwh"] >= 0).all()


def test_truth_evaluation_uses_full_window_and_pue(dc_config) -> None:
    truth = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "city": ["Alpha", "Alpha"],
            "carbon_intensity_kg_per_kwh": [0.1, 0.3],
            "water_intensity_l_per_kwh": [1.0, 1.0],
        }
    )
    schedule = pd.DataFrame(
        [
            {
                "job_id": "job",
                "scheduled": 1,
                "assigned_dc_id": "dc_alpha",
                "assigned_city": "Alpha",
                "scheduled_start": "2025-01-01 00:00:00",
                "power_demand_kw": 10.0,
                "duration_hours": 2,
            }
        ]
    )
    actual = evaluate_schedule_against_truth(schedule, truth, dc_config)
    assert actual == pytest.approx(10.0 * 1.1 * (0.1 + 0.3))


def test_zero_noise_matches_perfect_forecast(environment, dc_config, jobs) -> None:
    result = run_forecast_robustness(
        environment,
        dc_config,
        jobs,
        noise_levels=[0.0],
        repetitions=2,
    )
    assert result.loc[0, "carbon_delta_pct_mean"] == pytest.approx(0.0)
    assert result.loc[0, "forecast_rmse_mean"] == pytest.approx(0.0)
