"""
Unit tests for src/scheduling.py

These tests use only in-memory DataFrames — no file I/O, no models.
They verify the scheduling algorithm's core logic:
  - Job record construction
  - Candidate scoring (alpha/beta/gamma weights)
  - Schedule summarization
  - Baseline vs. optimized comparison
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.scheduling import (
    _candidate_score,
    build_job_record,
    compare_summaries,
    summarize_schedule,
)


# ---------------------------------------------------------------------------
# build_job_record
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildJobRecord:
    def test_returns_series(self):
        record = build_job_record(
            origin_city="Dallas",
            power_demand=10.0,
            duration_hours=2,
            earliest_start="2024-06-01 12:00",
            deadline="2024-06-01 18:00",
        )
        assert isinstance(record, pd.Series)

    def test_required_fields_present(self):
        record = build_job_record("Dallas", 10.0, 2, "2024-06-01", "2024-06-02")
        for field in ["job_id", "origin_city", "power_demand", "duration_hours",
                      "earliest_start", "deadline", "priority", "priority_label"]:
            assert field in record.index, f"Missing field: {field}"

    def test_timestamps_parsed(self):
        record = build_job_record("Dallas", 10.0, 2, "2024-06-01", "2024-06-02")
        assert isinstance(record["earliest_start"], pd.Timestamp)
        assert isinstance(record["deadline"], pd.Timestamp)

    def test_priority_label_mapping(self):
        high = build_job_record("Dallas", 5.0, 1, "2024-06-01", "2024-06-02", priority=0)
        med = build_job_record("Dallas", 5.0, 1, "2024-06-01", "2024-06-02", priority=1)
        low = build_job_record("Dallas", 5.0, 1, "2024-06-01", "2024-06-02", priority=2)
        assert high["priority_label"] == "high"
        assert med["priority_label"] == "medium"
        assert low["priority_label"] == "low"

    def test_custom_job_id(self):
        record = build_job_record("Dallas", 5.0, 1, "2024-06-01", "2024-06-02",
                                  job_id="test_123")
        assert record["job_id"] == "test_123"


# ---------------------------------------------------------------------------
# _candidate_score  — the heart of the scheduler
#
# score = alpha * (co2_mean * pue)
#       + beta  * (water_mean * water_usage_factor * region_water_scarcity)
#       + gamma * (0 if same city, else 1)
#
# TODO: Fill in the parametrize cases below.
# Each tuple is: (alpha, beta, gamma, origin_city, dc_city, description)
# Think about edge cases where a weight = 0, or same vs. different city.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("alpha, beta, gamma, origin_city, dc_city, check", [
    # Case 1: alpha=0 → CO2 component is computed but adds nothing to score
    (0.0, 1.0, 1.0, "Dallas", "Atlanta", "zero_alpha"),
    # Case 2: beta=0 → water component is computed but adds nothing to score
    (1.0, 0.0, 1.0, "Dallas", "Atlanta", "zero_beta"),
    # Case 3: gamma=0 → no latency penalty regardless of city distance
    (1.0, 1.0, 0.0, "Dallas", "Atlanta", "zero_gamma"),
    # Case 4: same city → latency penalty is 0 even with a high gamma
    (1.0, 1.0, 5.0, "Dallas", "Dallas", "same_city_no_latency"),
])
def test_candidate_score_weights(alpha, beta, gamma, origin_city, dc_city, check):
    """
    Verifies the alpha/beta/gamma knobs behave correctly.
    Each case isolates one dimension of the scoring formula.

    score = alpha * co2_component
          + beta  * water_component
          + gamma * (0 if same city else 1)
    """
    dc_record = {
        "city": dc_city,
        "pue": 1.2,
        "water_usage_factor": 1.0,
        "region_water_scarcity": 0.5,
    }
    co2_mean = 0.4   # co2_component = 0.4 * 1.2 = 0.48
    water_mean = 2.0  # water_component = 2.0 * 1.0 * 0.5 = 1.0

    score, co2_component, water_component = _candidate_score(
        co2_mean=co2_mean,
        water_mean=water_mean,
        dc_record=dc_record,
        origin_city=origin_city,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    latency_penalty = gamma if origin_city != dc_city else 0.0

    if check == "zero_alpha":
        # CO2 component is still computed (0.48), just not weighted in the score
        assert co2_component == pytest.approx(co2_mean * 1.2)
        assert score == pytest.approx(0.0 * co2_component + beta * water_component + latency_penalty)

    elif check == "zero_beta":
        # Water component is still computed (1.0), just not weighted in the score
        assert water_component == pytest.approx(water_mean * 1.0 * 0.5)
        assert score == pytest.approx(alpha * co2_component + 0.0 * water_component + latency_penalty)

    elif check == "zero_gamma":
        # No latency penalty — score is purely environmental even across cities
        assert score == pytest.approx(alpha * co2_component + beta * water_component)

    elif check == "same_city_no_latency":
        # Same city: latency term is structurally 0, even though gamma=5.0
        expected_without_latency = alpha * co2_component + beta * water_component
        assert score == pytest.approx(expected_without_latency)


# ---------------------------------------------------------------------------
# summarize_schedule
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSummarizeSchedule:
    def test_coverage_two_of_three(self, sample_schedule_df):
        result = summarize_schedule(sample_schedule_df)
        assert result["scheduled_jobs"] == pytest.approx(2.0)
        assert result["total_jobs"] == pytest.approx(3.0)
        assert result["coverage"] == pytest.approx(2 / 3)

    def test_total_co2_only_scheduled(self, sample_schedule_df):
        result = summarize_schedule(sample_schedule_df)
        # job_c is not scheduled → its 0.0 co2 doesn't contribute, sum is 3.0
        assert result["total_expected_co2_kg"] == pytest.approx(3.0)

    def test_total_water_only_scheduled(self, sample_schedule_df):
        result = summarize_schedule(sample_schedule_df)
        assert result["total_expected_water_liters"] == pytest.approx(15.0)

    def test_empty_schedule(self):
        empty = pd.DataFrame(columns=["scheduled", "expected_co2_kg", "expected_water_liters"])
        result = summarize_schedule(empty)
        assert result["coverage"] == pytest.approx(0.0)
        assert result["scheduled_jobs"] == pytest.approx(0.0)

    def test_all_unscheduled(self):
        df = pd.DataFrame([
            {"scheduled": 0, "expected_co2_kg": 0.0, "expected_water_liters": 0.0},
            {"scheduled": 0, "expected_co2_kg": 0.0, "expected_water_liters": 0.0},
        ])
        result = summarize_schedule(df)
        assert result["coverage"] == pytest.approx(0.0)
        assert result["scheduled_jobs"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compare_summaries
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompareSummaries:
    def _make_summary(self, co2: float, water: float, coverage: float, jobs: float):
        return {
            "total_expected_co2_kg": co2,
            "total_expected_water_liters": water,
            "coverage": coverage,
            "scheduled_jobs": jobs,
            "total_jobs": 10.0,
        }

    def test_co2_reduction_pct(self):
        baseline = self._make_summary(co2=100.0, water=50.0, coverage=0.8, jobs=8)
        optimized = self._make_summary(co2=80.0, water=50.0, coverage=0.8, jobs=8)
        result = compare_summaries(baseline, optimized)
        assert result["co2_reduction_pct"] == pytest.approx(20.0)

    def test_water_reduction_pct(self):
        baseline = self._make_summary(co2=100.0, water=200.0, coverage=0.8, jobs=8)
        optimized = self._make_summary(co2=100.0, water=150.0, coverage=0.8, jobs=8)
        result = compare_summaries(baseline, optimized)
        assert result["water_reduction_pct"] == pytest.approx(25.0)

    def test_zero_baseline_co2_no_division_error(self):
        """If baseline CO2 is 0, reduction should be 0.0 (not ZeroDivisionError)."""
        baseline = self._make_summary(co2=0.0, water=50.0, coverage=0.8, jobs=8)
        optimized = self._make_summary(co2=0.0, water=40.0, coverage=0.9, jobs=9)
        result = compare_summaries(baseline, optimized)
        assert result["co2_reduction_pct"] == pytest.approx(0.0)

    def test_coverage_delta(self):
        baseline = self._make_summary(co2=100.0, water=50.0, coverage=0.6, jobs=6)
        optimized = self._make_summary(co2=80.0, water=40.0, coverage=0.9, jobs=9)
        result = compare_summaries(baseline, optimized)
        assert result["coverage_delta_pct"] == pytest.approx(30.0)

    def test_scheduled_jobs_delta(self):
        baseline = self._make_summary(co2=100.0, water=50.0, coverage=0.6, jobs=6)
        optimized = self._make_summary(co2=80.0, water=40.0, coverage=0.9, jobs=9)
        result = compare_summaries(baseline, optimized)
        assert result["scheduled_jobs_delta"] == pytest.approx(3.0)
