"""Baseline scheduling strategies for comparison with optimized scheduler."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .scheduling import PreparedSchedulerState, ScheduleCandidate


@dataclass
class BaselineResult:
    """Aggregated result from a baseline strategy."""
    strategy_name: str
    jobs_scheduled: int
    total_co2_kg: float
    total_water_liters: float
    avg_co2_per_job: float
    avg_water_per_job: float
    jobs_failed: int


def baseline_random(
    job: pd.Series,
    prepared_state: PreparedSchedulerState,
) -> ScheduleCandidate | None:
    """Baseline 1: Random city selection.

    Picks any city uniformly at random and schedules at earliest available slot.
    No optimization for carbon or water.
    """
    if prepared_state.dc_records is None or len(prepared_state.dc_records) == 0:
        return None

    duration_hours = int(job["duration_hours"])
    earliest_start = pd.Timestamp(job["earliest_start"]).floor("h")
    deadline = pd.Timestamp(job["deadline"])

    if earliest_start >= deadline:
        return None

    # Pick random data center
    dc_record = random.choice(prepared_state.dc_records)
    city = str(dc_record["city"])
    dc_id = str(dc_record["dc_id"])

    # Find earliest available start time
    city_metrics = prepared_state.city_metrics.get(city)
    if city_metrics is None:
        return None

    start_idx = int(
        np.searchsorted(
            prepared_state.time_values,
            np.datetime64(earliest_start.to_datetime64(), "ns"),
            side="left"
        )
    )

    if start_idx + duration_hours >= len(prepared_state.time_index):
        return None

    end_idx = start_idx + duration_hours
    start_time = pd.Timestamp(prepared_state.time_index[start_idx])
    end_time = pd.Timestamp(prepared_state.time_index[end_idx])

    # Calculate averages
    co2_span = max(1, end_idx - start_idx)
    water_span = max(1, end_idx - start_idx)

    co2_mean = (city_metrics.co2_prefix[end_idx] - city_metrics.co2_prefix[start_idx]) / co2_span
    water_mean = (city_metrics.water_prefix[end_idx] - city_metrics.water_prefix[start_idx]) / water_span

    pue = float(dc_record.get("pue", 1.0))
    water_factor = float(dc_record.get("water_usage_factor", 1.0))
    scarcity = float(dc_record.get("region_water_scarcity", 1.0))

    # Total energy consumption: power_demand (kW) * duration_hours (h) = kWh
    power_demand = float(job.get("power_demand", 1.0))
    total_kwh = power_demand * duration_hours

    # Total CO2: intensity (kg/kWh) * total_kwh * PUE
    expected_co2 = co2_mean * total_kwh * pue
    # Total water: intensity * total_kwh * water_factor * scarcity
    expected_water = water_mean * total_kwh * water_factor * scarcity

    # Dummy score (not used for ranking, just placeholder)
    score = 0.0

    return ScheduleCandidate(
        dc_id=dc_id,
        city=city,
        start_time=start_time,
        end_time=end_time,
        score=score,
        expected_co2_kg=expected_co2,
        expected_water_liters=expected_water,
        start_index=start_idx,
        end_index=end_idx,
    )


def baseline_nearest_neighbor(
    job: pd.Series,
    prepared_state: PreparedSchedulerState,
) -> ScheduleCandidate | None:
    """Baseline 2: Nearest data center (current industry practice).

    Schedules to closest DC by location, earliest available slot.
    Ignores carbon and water completely.
    """
    origin_city = str(job["origin_city"])
    duration_hours = int(job["duration_hours"])
    earliest_start = pd.Timestamp(job["earliest_start"]).floor("h")
    deadline = pd.Timestamp(job["deadline"])

    if earliest_start >= deadline:
        return None

    # Find DC in same city, or pick first available if none exists
    same_city_dc = None
    for dc_record in prepared_state.dc_records:
        if str(dc_record["city"]) == origin_city:
            same_city_dc = dc_record
            break

    dc_record = same_city_dc or prepared_state.dc_records[0]
    city = str(dc_record["city"])
    dc_id = str(dc_record["dc_id"])

    city_metrics = prepared_state.city_metrics.get(city)
    if city_metrics is None:
        return None

    start_idx = int(
        np.searchsorted(
            prepared_state.time_values,
            np.datetime64(earliest_start.to_datetime64(), "ns"),
            side="left"
        )
    )

    if start_idx + duration_hours >= len(prepared_state.time_index):
        return None

    end_idx = start_idx + duration_hours
    start_time = pd.Timestamp(prepared_state.time_index[start_idx])
    end_time = pd.Timestamp(prepared_state.time_index[end_idx])

    # Calculate averages
    co2_span = max(1, end_idx - start_idx)
    water_span = max(1, end_idx - start_idx)

    co2_mean = (city_metrics.co2_prefix[end_idx] - city_metrics.co2_prefix[start_idx]) / co2_span
    water_mean = (city_metrics.water_prefix[end_idx] - city_metrics.water_prefix[start_idx]) / water_span

    pue = float(dc_record.get("pue", 1.0))
    water_factor = float(dc_record.get("water_usage_factor", 1.0))
    scarcity = float(dc_record.get("region_water_scarcity", 1.0))

    # Total energy consumption: power_demand (kW) * duration_hours (h) = kWh
    power_demand = float(job.get("power_demand", 1.0))
    total_kwh = power_demand * duration_hours

    # Total CO2: intensity (kg/kWh) * total_kwh * PUE
    expected_co2 = co2_mean * total_kwh * pue
    # Total water: intensity * total_kwh * water_factor * scarcity
    expected_water = water_mean * total_kwh * water_factor * scarcity

    return ScheduleCandidate(
        dc_id=dc_id,
        city=city,
        start_time=start_time,
        end_time=end_time,
        score=0.0,
        expected_co2_kg=expected_co2,
        expected_water_liters=expected_water,
        start_index=start_idx,
        end_index=end_idx,
    )


def baseline_time_of_day_heuristic(
    job: pd.Series,
    prepared_state: PreparedSchedulerState,
) -> ScheduleCandidate | None:
    """Baseline 3: Time-of-day heuristic (simple rule-based approach).

    Picks city with lowest CO2 at that hour globally.
    Still respects deadlines but ignores water.
    """
    duration_hours = int(job["duration_hours"])
    earliest_start = pd.Timestamp(job["earliest_start"]).floor("h")
    deadline = pd.Timestamp(job["deadline"])

    if earliest_start >= deadline:
        return None

    start_idx = int(
        np.searchsorted(
            prepared_state.time_values,
            np.datetime64(earliest_start.to_datetime64(), "ns"),
            side="left"
        )
    )

    if start_idx + duration_hours >= len(prepared_state.time_index):
        return None

    end_idx = start_idx + duration_hours

    # Find city with lowest average CO2 during this window
    best_city = None
    best_co2 = float("inf")

    for city, city_metrics in prepared_state.city_metrics.items():
        co2_span = max(1, end_idx - start_idx)
        co2_mean = (city_metrics.co2_prefix[end_idx] - city_metrics.co2_prefix[start_idx]) / co2_span

        if co2_mean < best_co2:
            best_co2 = co2_mean
            best_city = city

    if best_city is None:
        return None

    # Find a DC in best_city
    dc_record = None
    for record in prepared_state.dc_records:
        if str(record["city"]) == best_city:
            dc_record = record
            break

    if dc_record is None:
        return None

    city = best_city
    dc_id = str(dc_record["dc_id"])
    city_metrics = prepared_state.city_metrics[city]

    start_time = pd.Timestamp(prepared_state.time_index[start_idx])
    end_time = pd.Timestamp(prepared_state.time_index[end_idx])

    # Calculate metrics
    co2_span = max(1, end_idx - start_idx)
    water_span = max(1, end_idx - start_idx)

    co2_mean = (city_metrics.co2_prefix[end_idx] - city_metrics.co2_prefix[start_idx]) / co2_span
    water_mean = (city_metrics.water_prefix[end_idx] - city_metrics.water_prefix[start_idx]) / water_span

    pue = float(dc_record.get("pue", 1.0))
    water_factor = float(dc_record.get("water_usage_factor", 1.0))
    scarcity = float(dc_record.get("region_water_scarcity", 1.0))

    # Total energy consumption: power_demand (kW) * duration_hours (h) = kWh
    power_demand = float(job.get("power_demand", 1.0))
    duration_hours = int(job.get("duration_hours", 1))
    total_kwh = power_demand * duration_hours

    # Total CO2: intensity (kg/kWh) * total_kwh * PUE
    expected_co2 = co2_mean * total_kwh * pue
    # Total water: intensity * total_kwh * water_factor * scarcity
    expected_water = water_mean * total_kwh * water_factor * scarcity

    return ScheduleCandidate(
        dc_id=dc_id,
        city=city,
        start_time=start_time,
        end_time=end_time,
        score=0.0,
        expected_co2_kg=expected_co2,
        expected_water_liters=expected_water,
        start_index=start_idx,
        end_index=end_idx,
    )
