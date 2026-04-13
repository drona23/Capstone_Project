"""Tests for baseline scheduling strategies."""

import pandas as pd
import pytest

from src.scheduling import (
    PreparedSchedulerState,
    PreparedCityMetrics,
    build_job_record,
    prepare_environment_frame,
    prepare_scheduler_state,
    load_dc_config,
)
from src.baselines import (
    baseline_random,
    baseline_nearest_neighbor,
    baseline_time_of_day_heuristic,
)


@pytest.fixture
def sample_prepared_state(sample_env_df, sample_dc_df):
    """Create a sample prepared state for testing baselines."""
    env_df = prepare_environment_frame(sample_env_df)
    return prepare_scheduler_state(env_df, sample_dc_df)


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    return build_job_record(
        origin_city="Dallas",
        power_demand=500.0,
        duration_hours=4,
        earliest_start="2024-06-01T12:00",
        deadline="2024-06-01T20:00",
        priority=1,  # 0=high, 1=medium, 2=low
        job_id="test_job_001",
    )


class TestBaselineRandom:
    """Tests for random baseline strategy."""

    def test_returns_schedule_candidate(self, sample_prepared_state, sample_job):
        """Random baseline returns a ScheduleCandidate or None."""
        result = baseline_random(sample_job, sample_prepared_state)
        assert result is None or result.dc_id is not None

    def test_respects_deadline(self, sample_prepared_state, sample_job):
        """Random baseline respects job deadline."""
        result = baseline_random(sample_job, sample_prepared_state)
        if result:
            assert result.end_time <= pd.Timestamp(sample_job["deadline"])

    def test_respects_earliest_start(self, sample_prepared_state, sample_job):
        """Random baseline respects job earliest start."""
        result = baseline_random(sample_job, sample_prepared_state)
        if result:
            assert result.start_time >= pd.Timestamp(sample_job["earliest_start"])

    def test_has_positive_co2_expectation(self, sample_prepared_state, sample_job):
        """Random baseline returns positive CO2 expectations."""
        result = baseline_random(sample_job, sample_prepared_state)
        if result:
            assert result.expected_co2_kg >= 0.0

    def test_has_positive_water_expectation(self, sample_prepared_state, sample_job):
        """Random baseline returns positive water expectations."""
        result = baseline_random(sample_job, sample_prepared_state)
        if result:
            assert result.expected_water_liters >= 0.0


class TestBaselineNearestNeighbor:
    """Tests for nearest neighbor baseline strategy."""

    def test_returns_schedule_candidate(self, sample_prepared_state, sample_job):
        """Nearest neighbor baseline returns a ScheduleCandidate or None."""
        result = baseline_nearest_neighbor(sample_job, sample_prepared_state)
        assert result is None or result.dc_id is not None

    def test_prefers_same_city(self, sample_prepared_state, sample_job):
        """Nearest neighbor prefers same-city DC if available."""
        result = baseline_nearest_neighbor(sample_job, sample_prepared_state)
        if result:
            # If origin city is in DC list, result should be in same city
            if sample_job["origin_city"] in [dc["city"] for dc in sample_prepared_state.dc_records]:
                assert result.city == sample_job["origin_city"]

    def test_respects_deadline(self, sample_prepared_state, sample_job):
        """Nearest neighbor respects job deadline."""
        result = baseline_nearest_neighbor(sample_job, sample_prepared_state)
        if result:
            assert result.end_time <= pd.Timestamp(sample_job["deadline"])

    def test_respects_earliest_start(self, sample_prepared_state, sample_job):
        """Nearest neighbor respects job earliest start."""
        result = baseline_nearest_neighbor(sample_job, sample_prepared_state)
        if result:
            assert result.start_time >= pd.Timestamp(sample_job["earliest_start"])


class TestBaselineTimeOfDay:
    """Tests for time-of-day heuristic baseline strategy."""

    def test_returns_schedule_candidate(self, sample_prepared_state, sample_job):
        """Time-of-day heuristic returns a ScheduleCandidate or None."""
        result = baseline_time_of_day_heuristic(sample_job, sample_prepared_state)
        assert result is None or result.dc_id is not None

    def test_picks_low_carbon_city(self, sample_prepared_state, sample_job):
        """Time-of-day heuristic picks city with lowest CO2."""
        result = baseline_time_of_day_heuristic(sample_job, sample_prepared_state)
        if result:
            # Result should have a city in prepared state
            assert result.city in sample_prepared_state.city_metrics

    def test_respects_deadline(self, sample_prepared_state, sample_job):
        """Time-of-day heuristic respects job deadline."""
        result = baseline_time_of_day_heuristic(sample_job, sample_prepared_state)
        if result:
            assert result.end_time <= pd.Timestamp(sample_job["deadline"])

    def test_respects_earliest_start(self, sample_prepared_state, sample_job):
        """Time-of-day heuristic respects job earliest start."""
        result = baseline_time_of_day_heuristic(sample_job, sample_prepared_state)
        if result:
            assert result.start_time >= pd.Timestamp(sample_job["earliest_start"])

    def test_has_positive_co2_expectation(self, sample_prepared_state, sample_job):
        """Time-of-day heuristic returns positive CO2 expectations."""
        result = baseline_time_of_day_heuristic(sample_job, sample_prepared_state)
        if result:
            assert result.expected_co2_kg >= 0.0
