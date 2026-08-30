"""Core scheduler for reproducible carbon and water aware experiments.

All timestamps are interpreted as UTC. Capacity is expressed as IT power in kW.
Carbon intensity is kg CO2e per facility kWh. Water Usage Effectiveness is
liters per IT kWh, so PUE is applied to carbon but not to onsite water use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_JOB_COLUMNS = {
    "job_id",
    "origin_city",
    "power_demand_kw",
    "duration_hours",
    "earliest_start",
    "deadline",
}
REQUIRED_DC_COLUMNS = {
    "dc_id",
    "city",
    "max_power_kw",
    "pue",
    "water_usage_factor",
    "water_scarcity_index",
}


@dataclass(frozen=True)
class ScheduleCandidate:
    """One feasible placement for one non-preemptive job."""

    dc_id: str
    city: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_index: int
    end_index: int
    expected_co2_kg: float
    expected_water_liters: float
    scarcity_weighted_water_liters: float
    relocation_penalty: float
    score: float = np.nan


@dataclass(frozen=True)
class PreparedCityMetrics:
    carbon_prefix: np.ndarray
    water_prefix: np.ndarray
    valid_prefix: np.ndarray


@dataclass(frozen=True)
class PreparedSchedulerState:
    time_index: pd.DatetimeIndex
    time_values: np.ndarray
    city_metrics: dict[str, PreparedCityMetrics]
    dc_records: tuple[dict[str, float | str], ...]


CandidateSelector = Callable[[pd.Series, list[ScheduleCandidate]], ScheduleCandidate]


def _utc_naive_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return pd.Timestamp(timestamp).tz_convert(None)


def _utc_naive_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def build_job_record(
    origin_city: str,
    power_demand_kw: float,
    duration_hours: int,
    earliest_start: str | pd.Timestamp,
    deadline: str | pd.Timestamp,
    priority: int = 1,
    job_id: str = "job",
) -> pd.Series:
    """Build one validated job record."""
    record = pd.Series(
        {
            "job_id": str(job_id),
            "origin_city": str(origin_city),
            "power_demand_kw": float(power_demand_kw),
            "duration_hours": int(duration_hours),
            "earliest_start": _utc_naive_timestamp(earliest_start),
            "deadline": _utc_naive_timestamp(deadline),
            "priority": int(priority),
        }
    )
    if record["power_demand_kw"] <= 0 or record["duration_hours"] <= 0:
        raise ValueError("Power demand and duration must be positive.")
    if record["deadline"] <= record["earliest_start"]:
        raise ValueError("Deadline must be later than earliest_start.")
    return record


def load_jobs_dataset(path: str | Path) -> pd.DataFrame:
    """Load a canonical jobs CSV."""
    jobs = pd.read_csv(path)
    if "power_demand" in jobs.columns and "power_demand_kw" not in jobs.columns:
        jobs = jobs.rename(columns={"power_demand": "power_demand_kw"})
    _require_columns(jobs, REQUIRED_JOB_COLUMNS, "Jobs dataset")
    jobs["earliest_start"] = _utc_naive_series(jobs["earliest_start"])
    jobs["deadline"] = _utc_naive_series(jobs["deadline"])
    return jobs


def load_dc_config(path: str | Path) -> pd.DataFrame:
    """Load a data center configuration and normalize legacy column names."""
    config = pd.read_csv(path)
    config = config.rename(
        columns={
            "max_power_per_hour": "max_power_kw",
            "region_water_scarcity": "water_scarcity_index",
        }
    )
    _require_columns(config, REQUIRED_DC_COLUMNS, "Data center configuration")
    return config


def prepare_environment_frame(
    source_df: pd.DataFrame,
    city_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize source data to one environmental row per city and UTC hour."""
    data = source_df.copy()
    timestamp_col = next(
        (column for column in ("timestamp", "TIMESTAMP", "Timestamp") if column in data),
        None,
    )
    city_col = next((column for column in ("city", "CITY") if column in data), None)
    if timestamp_col is None or city_col is None:
        raise ValueError("Environment data requires timestamp and city columns.")

    data["timestamp"] = _utc_naive_series(data[timestamp_col]).dt.floor("h")
    data["city"] = data[city_col].astype(str).str.strip()

    if "carbon_intensity_kg_per_kwh" in data:
        carbon = pd.to_numeric(data["carbon_intensity_kg_per_kwh"], errors="coerce")
    elif {"co2_kg", "total_gen_kwh"}.issubset(data.columns):
        numerator = pd.to_numeric(data["co2_kg"], errors="coerce")
        denominator = pd.to_numeric(data["total_gen_kwh"], errors="coerce").replace(0, np.nan)
        carbon = numerator / denominator
    else:
        raise ValueError(
            "Environment data requires carbon_intensity_kg_per_kwh or "
            "co2_kg plus total_gen_kwh."
        )

    if "water_intensity_l_per_kwh" in data:
        water = pd.to_numeric(data["water_intensity_l_per_kwh"], errors="coerce")
    elif "WUE_total" in data:
        water = pd.to_numeric(data["WUE_total"], errors="coerce")
    else:
        raise ValueError(
            "Environment data requires water_intensity_l_per_kwh or WUE_total."
        )

    if "scarcity_index" in data:
        scarcity = pd.to_numeric(data["scarcity_index"], errors="coerce")
    elif "dsci_0_500" in data:
        scarcity = pd.to_numeric(data["dsci_0_500"], errors="coerce") / 500.0
    else:
        scarcity = pd.Series(0.0, index=data.index)

    normalized = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "city": data["city"],
            "carbon_intensity_kg_per_kwh": carbon,
            "water_intensity_l_per_kwh": water,
            "scarcity_index": scarcity,
        }
    )
    if city_subset is not None:
        normalized = normalized[normalized["city"].isin(city_subset)]

    normalized = normalized.dropna(
        subset=[
            "timestamp",
            "city",
            "carbon_intensity_kg_per_kwh",
            "water_intensity_l_per_kwh",
        ]
    )
    if (normalized["carbon_intensity_kg_per_kwh"] < 0).any():
        raise ValueError("Carbon intensity cannot be negative.")
    if (normalized["water_intensity_l_per_kwh"] < 0).any():
        raise ValueError("Water intensity cannot be negative.")
    if not normalized["scarcity_index"].dropna().between(0, 1).all():
        raise ValueError("scarcity_index must be between 0 and 1.")

    return (
        normalized.groupby(["timestamp", "city"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["timestamp", "city"])
        .reset_index(drop=True)
    )


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def prepare_scheduler_state(
    env_df: pd.DataFrame,
    dc_df: pd.DataFrame,
) -> PreparedSchedulerState:
    """Validate inputs and precompute hourly prefix sums."""
    env = prepare_environment_frame(env_df)
    config = dc_df.copy().rename(
        columns={
            "max_power_per_hour": "max_power_kw",
            "region_water_scarcity": "water_scarcity_index",
        }
    )
    _require_columns(config, REQUIRED_DC_COLUMNS, "Data center configuration")
    if config["dc_id"].duplicated().any():
        raise ValueError("dc_id values must be unique.")

    numeric_columns = [
        "max_power_kw",
        "pue",
        "water_usage_factor",
        "water_scarcity_index",
    ]
    for column in numeric_columns:
        config[column] = pd.to_numeric(config[column], errors="raise")
    if (config["max_power_kw"] <= 0).any():
        raise ValueError("max_power_kw must be positive.")
    if (config["pue"] < 1).any():
        raise ValueError("PUE must be at least 1.")
    if (config["water_usage_factor"] < 0).any():
        raise ValueError("water_usage_factor cannot be negative.")
    if not config["water_scarcity_index"].between(0, 1).all():
        raise ValueError("water_scarcity_index must be between 0 and 1.")

    missing_cities = sorted(set(config["city"].astype(str)) - set(env["city"]))
    if missing_cities:
        raise ValueError(
            "No environmental data for configured cities: " + ", ".join(missing_cities)
        )

    time_index = pd.date_range(
        env["timestamp"].min(), env["timestamp"].max(), freq="h"
    )
    city_metrics: dict[str, PreparedCityMetrics] = {}
    for city, city_frame in env.groupby("city"):
        aligned = city_frame.set_index("timestamp").sort_index().reindex(time_index)
        carbon = aligned["carbon_intensity_kg_per_kwh"].to_numpy(dtype=float)
        water = aligned["water_intensity_l_per_kwh"].to_numpy(dtype=float)
        valid = np.isfinite(carbon) & np.isfinite(water)
        city_metrics[str(city)] = PreparedCityMetrics(
            carbon_prefix=_prefix_sum(np.where(valid, carbon, 0.0)),
            water_prefix=_prefix_sum(np.where(valid, water, 0.0)),
            valid_prefix=_prefix_sum(valid.astype(float)),
        )

    records = tuple(config[list(REQUIRED_DC_COLUMNS)].to_dict(orient="records"))
    return PreparedSchedulerState(
        time_index=time_index,
        time_values=time_index.to_numpy(dtype="datetime64[ns]"),
        city_metrics=city_metrics,
        dc_records=records,
    )


def initialize_capacity_state(state: PreparedSchedulerState) -> dict[str, np.ndarray]:
    return {
        str(record["dc_id"]): np.zeros(len(state.time_index), dtype=np.float64)
        for record in state.dc_records
    }


def reserve_capacity(
    used_capacity: dict[str, np.ndarray],
    candidate: ScheduleCandidate,
    power_demand_kw: float,
) -> None:
    used_capacity[candidate.dc_id][candidate.start_index : candidate.end_index] += (
        power_demand_kw
    )


def _candidate_start_positions(
    state: PreparedSchedulerState,
    earliest_start: pd.Timestamp,
    latest_start: pd.Timestamp,
    duration_hours: int,
) -> np.ndarray:
    if duration_hours <= 0 or duration_hours > len(state.time_index):
        return np.empty(0, dtype=np.int32)
    lower = int(
        np.searchsorted(
            state.time_values,
            np.datetime64(earliest_start.floor("h"), "ns"),
            side="left",
        )
    )
    upper = int(
        np.searchsorted(
            state.time_values,
            np.datetime64(latest_start.floor("h"), "ns"),
            side="right",
        )
    ) - 1
    upper = min(upper, len(state.time_index) - duration_hours)
    if lower > upper:
        return np.empty(0, dtype=np.int32)
    return np.arange(lower, upper + 1, dtype=np.int32)


def _window_mean(prefix: np.ndarray, start: int, end: int) -> float:
    return float((prefix[end] - prefix[start]) / (end - start))


def enumerate_feasible_candidates(
    job: pd.Series,
    state: PreparedSchedulerState,
    used_capacity: dict[str, np.ndarray],
) -> list[ScheduleCandidate]:
    """Enumerate all deadline, data, and capacity feasible placements."""
    duration = int(job["duration_hours"])
    power = float(job["power_demand_kw"])
    earliest = _utc_naive_timestamp(job["earliest_start"]).floor("h")
    deadline = _utc_naive_timestamp(job["deadline"])
    latest_start = (deadline - pd.Timedelta(hours=duration)).floor("h")
    positions = _candidate_start_positions(state, earliest, latest_start, duration)
    if positions.size == 0 or power <= 0:
        return []

    candidates: list[ScheduleCandidate] = []
    for record in state.dc_records:
        city = str(record["city"])
        dc_id = str(record["dc_id"])
        metrics = state.city_metrics[city]
        max_power = float(record["max_power_kw"])
        for start in positions:
            start_index = int(start)
            end_index = start_index + duration
            valid_count = metrics.valid_prefix[end_index] - metrics.valid_prefix[start_index]
            if valid_count != duration:
                continue
            capacity_window = used_capacity[dc_id][start_index:end_index]
            if not np.all(capacity_window + power <= max_power + 1e-12):
                continue

            carbon_intensity = _window_mean(
                metrics.carbon_prefix, start_index, end_index
            )
            water_intensity = _window_mean(metrics.water_prefix, start_index, end_index)
            it_energy_kwh = power * duration
            expected_co2 = it_energy_kwh * float(record["pue"]) * carbon_intensity
            expected_water = (
                it_energy_kwh
                * water_intensity
                * float(record["water_usage_factor"])
            )
            scarcity_water = expected_water * (
                1.0 + float(record["water_scarcity_index"])
            )
            start_time = pd.Timestamp(state.time_index[start_index])
            candidates.append(
                ScheduleCandidate(
                    dc_id=dc_id,
                    city=city,
                    start_time=start_time,
                    end_time=start_time + pd.Timedelta(hours=duration),
                    start_index=start_index,
                    end_index=end_index,
                    expected_co2_kg=float(expected_co2),
                    expected_water_liters=float(expected_water),
                    scarcity_weighted_water_liters=float(scarcity_water),
                    relocation_penalty=float(str(job["origin_city"]) != city),
                )
            )
    return candidates


def _minmax(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    spread = float(values.max() - minimum)
    if spread <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / spread


def score_candidates(
    candidates: list[ScheduleCandidate],
    alpha: float,
    beta: float,
    gamma: float,
) -> list[ScheduleCandidate]:
    """Score candidates after per-job min-max normalization of physical impacts."""
    weights = np.asarray([alpha, beta, gamma], dtype=float)
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("alpha, beta, and gamma must be non-negative and sum to 1.")
    if not candidates:
        return []

    carbon = _minmax(np.asarray([item.expected_co2_kg for item in candidates]))
    water = _minmax(
        np.asarray([item.scarcity_weighted_water_liters for item in candidates])
    )
    scored = [
        replace(
            item,
            score=float(
                alpha * carbon[index]
                + beta * water[index]
                + gamma * item.relocation_penalty
            ),
        )
        for index, item in enumerate(candidates)
    ]
    return sorted(
        scored,
        key=lambda item: (item.score, item.start_time, item.dc_id),
    )


def _validate_jobs(jobs_df: pd.DataFrame) -> pd.DataFrame:
    jobs = jobs_df.copy()
    if "power_demand" in jobs and "power_demand_kw" not in jobs:
        jobs = jobs.rename(columns={"power_demand": "power_demand_kw"})
    _require_columns(jobs, REQUIRED_JOB_COLUMNS, "Jobs dataset")
    jobs["earliest_start"] = _utc_naive_series(jobs["earliest_start"])
    jobs["deadline"] = _utc_naive_series(jobs["deadline"])
    jobs["duration_hours"] = pd.to_numeric(jobs["duration_hours"], errors="raise").astype(int)
    jobs["power_demand_kw"] = pd.to_numeric(jobs["power_demand_kw"], errors="raise")
    jobs["priority"] = pd.to_numeric(jobs.get("priority", 1), errors="coerce").fillna(1)
    if jobs["job_id"].duplicated().any():
        raise ValueError("job_id values must be unique.")
    return jobs.sort_values(["priority", "earliest_start", "job_id"]).reset_index(drop=True)


def _schedule_row(job: pd.Series, candidate: ScheduleCandidate | None) -> dict[str, object]:
    row = job.to_dict()
    if candidate is None:
        return {
            **row,
            "assigned_dc_id": None,
            "assigned_city": None,
            "scheduled_start": pd.NaT,
            "scheduled_end": pd.NaT,
            "scheduler_score": np.nan,
            "expected_co2_kg": np.nan,
            "expected_water_liters": np.nan,
            "scarcity_weighted_water_liters": np.nan,
            "relocated": np.nan,
            "scheduled": 0,
            "failure_reason": "no_feasible_slot",
        }
    return {
        **row,
        "assigned_dc_id": candidate.dc_id,
        "assigned_city": candidate.city,
        "scheduled_start": candidate.start_time,
        "scheduled_end": candidate.end_time,
        "scheduler_score": candidate.score,
        "expected_co2_kg": candidate.expected_co2_kg,
        "expected_water_liters": candidate.expected_water_liters,
        "scarcity_weighted_water_liters": candidate.scarcity_weighted_water_liters,
        "relocated": int(candidate.relocation_penalty > 0),
        "scheduled": 1,
        "failure_reason": None,
    }


def schedule_with_policy(
    env_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    selector: CandidateSelector,
    alpha: float = 0.55,
    beta: float = 0.35,
    gamma: float = 0.10,
) -> pd.DataFrame:
    """Run a deterministic greedy policy with a caller-supplied selector."""
    state = prepare_scheduler_state(env_df, dc_df)
    used_capacity = initialize_capacity_state(state)
    rows: list[dict[str, object]] = []
    for _, job in _validate_jobs(jobs_df).iterrows():
        raw_candidates = enumerate_feasible_candidates(job, state, used_capacity)
        candidates = score_candidates(raw_candidates, alpha, beta, gamma)
        candidate = selector(job, candidates) if candidates else None
        if candidate is not None:
            reserve_capacity(used_capacity, candidate, float(job["power_demand_kw"]))
        rows.append(_schedule_row(job, candidate))
    return pd.DataFrame(rows)


def schedule_jobs(
    env_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    alpha: float = 0.55,
    beta: float = 0.35,
    gamma: float = 0.10,
) -> pd.DataFrame:
    """Run the normalized multi-objective greedy scheduler."""

    def best_score(
        _job: pd.Series, candidates: list[ScheduleCandidate]
    ) -> ScheduleCandidate:
        return candidates[0]

    return schedule_with_policy(
        env_df,
        jobs_df,
        dc_df,
        selector=best_score,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def summarize_schedule(schedule_df: pd.DataFrame) -> dict[str, float]:
    """Summarize coverage and physical impacts without hiding failed jobs."""
    if schedule_df.empty:
        return {
            "total_jobs": 0.0,
            "scheduled_jobs": 0.0,
            "coverage": 0.0,
            "total_co2_kg": 0.0,
            "total_water_liters": 0.0,
            "total_scarcity_weighted_water_liters": 0.0,
            "relocation_rate": 0.0,
        }
    scheduled = schedule_df["scheduled"].eq(1)
    return {
        "total_jobs": float(len(schedule_df)),
        "scheduled_jobs": float(scheduled.sum()),
        "coverage": float(scheduled.mean()),
        "total_co2_kg": float(schedule_df.loc[scheduled, "expected_co2_kg"].sum()),
        "total_water_liters": float(
            schedule_df.loc[scheduled, "expected_water_liters"].sum()
        ),
        "total_scarcity_weighted_water_liters": float(
            schedule_df.loc[scheduled, "scarcity_weighted_water_liters"].sum()
        ),
        "relocation_rate": float(schedule_df.loc[scheduled, "relocated"].mean())
        if scheduled.any()
        else 0.0,
    }


def paired_comparison(
    baseline_schedule: pd.DataFrame,
    proposed_schedule: pd.DataFrame,
) -> dict[str, float]:
    """Compare impacts only for jobs completed by both policies."""
    baseline_summary = summarize_schedule(baseline_schedule)
    proposed_summary = summarize_schedule(proposed_schedule)
    baseline = baseline_schedule[baseline_schedule["scheduled"].eq(1)][
        [
            "job_id",
            "power_demand_kw",
            "duration_hours",
            "expected_co2_kg",
            "expected_water_liters",
            "scarcity_weighted_water_liters",
        ]
    ]
    proposed = proposed_schedule[proposed_schedule["scheduled"].eq(1)][
        [
            "job_id",
            "expected_co2_kg",
            "expected_water_liters",
            "scarcity_weighted_water_liters",
        ]
    ]
    paired = baseline.merge(proposed, on="job_id", suffixes=("_baseline", "_proposed"))

    def reduction(column: str) -> float:
        old = float(paired[f"{column}_baseline"].sum())
        new = float(paired[f"{column}_proposed"].sum())
        return 100.0 * (old - new) / old if old > 0 else 0.0

    common_energy = (
        paired["power_demand_kw"] * paired["duration_hours"]
    ).sum() if not paired.empty else 0.0
    return {
        "baseline_coverage": baseline_summary["coverage"],
        "proposed_coverage": proposed_summary["coverage"],
        "coverage_gap_percentage_points": 100.0
        * (proposed_summary["coverage"] - baseline_summary["coverage"]),
        "common_jobs": float(len(paired)),
        "common_it_energy_kwh": float(common_energy),
        "co2_reduction_pct": reduction("expected_co2_kg"),
        "water_reduction_pct": reduction("expected_water_liters"),
        "scarcity_weighted_water_reduction_pct": reduction(
            "scarcity_weighted_water_liters"
        ),
    }
