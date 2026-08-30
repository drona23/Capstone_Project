from __future__ import annotations

import pandas as pd
import pytest

from src.scheduling import (
    build_job_record,
    paired_comparison,
    prepare_scheduler_state,
    schedule_jobs,
    score_candidates,
)


def test_build_job_record_normalizes_timezone() -> None:
    record = build_job_record(
        "Alpha",
        10,
        2,
        "2025-01-01 01:00:00+01:00",
        "2025-01-01 05:00:00+01:00",
    )
    assert record["earliest_start"] == pd.Timestamp("2025-01-01 00:00:00")


def test_scheduler_respects_deadline(environment, dc_config) -> None:
    jobs = pd.DataFrame(
        [
            {
                "job_id": "exact",
                "origin_city": "Alpha",
                "power_demand_kw": 10.0,
                "duration_hours": 2,
                "earliest_start": "2025-01-01 00:00:00",
                "deadline": "2025-01-01 02:00:00",
                "priority": 1,
            }
        ]
    )
    result = schedule_jobs(environment, jobs, dc_config)
    assert result.loc[0, "scheduled"] == 1
    assert result.loc[0, "scheduled_end"] <= pd.Timestamp(jobs.loc[0, "deadline"])


def test_impossible_deadline_is_not_scheduled(environment, dc_config) -> None:
    jobs = pd.DataFrame(
        [
            {
                "job_id": "late",
                "origin_city": "Alpha",
                "power_demand_kw": 10.0,
                "duration_hours": 3,
                "earliest_start": "2025-01-01 00:00:00",
                "deadline": "2025-01-01 02:00:00",
                "priority": 1,
            }
        ]
    )
    result = schedule_jobs(environment, jobs, dc_config)
    assert result.loc[0, "scheduled"] == 0
    assert result.loc[0, "failure_reason"] == "no_feasible_slot"


def test_capacity_is_enforced(environment, dc_config) -> None:
    constrained = dc_config.copy()
    constrained["max_power_kw"] = 10.0
    jobs = pd.DataFrame(
        [
            {
                "job_id": "too_large",
                "origin_city": "Alpha",
                "power_demand_kw": 11.0,
                "duration_hours": 1,
                "earliest_start": "2025-01-01",
                "deadline": "2025-01-01 03:00:00",
                "priority": 1,
            }
        ]
    )
    assert schedule_jobs(environment, jobs, constrained).loc[0, "scheduled"] == 0


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        score_candidates([], alpha=0.5, beta=0.5, gamma=0.5)


def test_paired_comparison_does_not_reward_missing_jobs() -> None:
    baseline = pd.DataFrame(
        [
            {
                "job_id": "shared",
                "power_demand_kw": 10.0,
                "duration_hours": 1,
                "scheduled": 1,
                "expected_co2_kg": 10.0,
                "expected_water_liters": 10.0,
                "scarcity_weighted_water_liters": 12.0,
                "relocated": 0,
            },
            {
                "job_id": "baseline_only",
                "power_demand_kw": 10.0,
                "duration_hours": 1,
                "scheduled": 1,
                "expected_co2_kg": 90.0,
                "expected_water_liters": 90.0,
                "scarcity_weighted_water_liters": 100.0,
                "relocated": 0,
            },
        ]
    )
    proposed = baseline.copy()
    proposed.loc[proposed["job_id"].eq("baseline_only"), "scheduled"] = 0
    proposed.loc[proposed["job_id"].eq("baseline_only"), "expected_co2_kg"] = float("nan")
    proposed.loc[proposed["job_id"].eq("baseline_only"), "expected_water_liters"] = float(
        "nan"
    )
    proposed.loc[
        proposed["job_id"].eq("baseline_only"), "scarcity_weighted_water_liters"
    ] = float("nan")
    result = paired_comparison(baseline, proposed)
    assert result["common_jobs"] == 1
    assert result["co2_reduction_pct"] == pytest.approx(0.0)
    assert result["proposed_coverage"] == pytest.approx(0.5)


def test_missing_city_environment_is_rejected(environment, dc_config) -> None:
    invalid = dc_config.copy()
    invalid.loc[0, "city"] = "Missing"
    with pytest.raises(ValueError, match="No environmental data"):
        prepare_scheduler_state(environment, invalid)
