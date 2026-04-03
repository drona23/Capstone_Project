from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .data_loader import PROJECT_ROOT, load_dataset
except ImportError:
    from data_loader import PROJECT_ROOT, load_dataset


DEFAULT_JOBS_PATH = PROJECT_ROOT / "data" / "processed" / "azure_jobs_sample.csv"
DEFAULT_DC_CONFIG_PATH = PROJECT_ROOT / "data" / "templates" / "dc_config_template.csv"

REQUIRED_JOB_COLUMNS = [
    "job_id",
    "origin_city",
    "power_demand",
    "duration_hours",
    "earliest_start",
    "deadline",
]
REQUIRED_DC_COLUMNS = [
    "dc_id",
    "city",
    "max_power_per_hour",
    "pue",
    "water_usage_factor",
    "region_water_scarcity",
    "is_new",
]


@dataclass
class ScheduleCandidate:
    dc_id: str
    city: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    score: float
    expected_co2_kg: float
    expected_water_liters: float
    start_index: int
    end_index: int


@dataclass
class PreparedCityMetrics:
    co2_prefix: np.ndarray
    water_prefix: np.ndarray
    valid_prefix: np.ndarray


@dataclass
class PreparedSchedulerState:
    time_index: pd.DatetimeIndex
    time_values: np.ndarray
    city_metrics: dict[str, PreparedCityMetrics]
    dc_records: list[dict[str, Any]]


@dataclass
class RouteFlow:
    route: str
    origin_city: str
    assigned_city: str
    jobs: int
    total_co2_kg: float
    total_water_liters: float
    avg_scheduler_score: float
    avg_latency_ms: float


def build_job_record(
    origin_city: str,
    power_demand: float,
    duration_hours: int,
    earliest_start: str | pd.Timestamp,
    deadline: str | pd.Timestamp,
    priority: int = 0,
    job_id: str = "simulated_job",
) -> pd.Series:
    """Construct a scheduler-compatible job record for interactive simulations."""
    priority_value = int(priority)
    priority_label = {0: "high", 1: "medium", 2: "low"}.get(priority_value, "low")
    return pd.Series(
        {
            "job_id": job_id,
            "origin_city": origin_city,
            "power_demand": float(power_demand),
            "duration_hours": int(duration_hours),
            "earliest_start": pd.Timestamp(earliest_start),
            "deadline": pd.Timestamp(deadline),
            "priority": priority_value,
            "priority_label": priority_label,
        }
    )


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_jobs_dataset(path: str | Path = DEFAULT_JOBS_PATH) -> pd.DataFrame:
    jobs_path = _resolve_path(path)
    jobs_df = pd.read_csv(
        jobs_path, parse_dates=["earliest_start", "deadline"], low_memory=False
    )
    missing = [col for col in REQUIRED_JOB_COLUMNS if col not in jobs_df.columns]
    if missing:
        raise ValueError(f"Jobs dataset is missing required columns: {', '.join(missing)}")
    return jobs_df


def load_dc_config(path: str | Path = DEFAULT_DC_CONFIG_PATH) -> pd.DataFrame:
    config_path = _resolve_path(path)
    dc_df = pd.read_csv(config_path, low_memory=False)
    missing = [col for col in REQUIRED_DC_COLUMNS if col not in dc_df.columns]
    if missing:
        raise ValueError(f"DC config is missing required columns: {', '.join(missing)}")
    return dc_df


def prepare_environment_frame(
    source_df: pd.DataFrame, city_subset: list[str] | None = None
) -> pd.DataFrame:
    """Create the city-hour environmental intensity frame used by the scheduler."""
    df = source_df.copy()

    timestamp_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else "Timestamp"
    city_col = "CITY" if "CITY" in df.columns else "city"
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce").dt.floor("h")

    for column in ["co2_kg", "total_gen_kwh", "WUE_total", "dsci_0_500"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if city_subset is not None:
        df = df[df[city_col].isin(city_subset)].copy()

    if "co2_kg" not in df.columns or "total_gen_kwh" not in df.columns:
        raise ValueError("Dataset must contain co2_kg and total_gen_kwh for scheduling.")
    if "WUE_total" not in df.columns:
        raise ValueError("Dataset must contain WUE_total for scheduling.")

    df["predicted_co2_intensity"] = df["co2_kg"] / df["total_gen_kwh"].replace(0, np.nan)
    df["predicted_water_intensity"] = df["WUE_total"]
    if "dsci_0_500" in df.columns:
        df["scarcity_index"] = df["dsci_0_500"] / 500.0
    else:
        df["scarcity_index"] = 1.0

    env_df = (
        df[[timestamp_col, city_col, "predicted_co2_intensity", "predicted_water_intensity", "scarcity_index"]]
        .dropna(subset=[timestamp_col, city_col, "predicted_co2_intensity", "predicted_water_intensity"])
        .groupby([timestamp_col, city_col], as_index=False)
        .agg(
            predicted_co2_intensity=("predicted_co2_intensity", "mean"),
            predicted_water_intensity=("predicted_water_intensity", "mean"),
            scarcity_index=("scarcity_index", "mean"),
        )
        .rename(columns={timestamp_col: "timestamp", city_col: "city"})
    )
    return env_df


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def prepare_scheduler_state(
    env_df: pd.DataFrame,
    dc_df: pd.DataFrame,
) -> PreparedSchedulerState:
    if env_df.empty:
        raise ValueError("Environment frame is empty; cannot prepare scheduler state.")

    time_index = pd.date_range(
        start=pd.Timestamp(env_df["timestamp"].min()).floor("h"),
        end=pd.Timestamp(env_df["timestamp"].max()).floor("h"),
        freq="h",
    )
    city_metrics: dict[str, PreparedCityMetrics] = {}

    for city, city_df in env_df.groupby("city"):
        aligned = city_df.set_index("timestamp").sort_index().reindex(time_index)
        co2 = pd.to_numeric(aligned["predicted_co2_intensity"], errors="coerce").to_numpy(dtype=float)
        water = pd.to_numeric(aligned["predicted_water_intensity"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(co2) & np.isfinite(water)
        city_metrics[str(city)] = PreparedCityMetrics(
            co2_prefix=_prefix_sum(np.where(valid, co2, 0.0)),
            water_prefix=_prefix_sum(np.where(valid, water, 0.0)),
            valid_prefix=_prefix_sum(valid.astype(float)),
        )

    dc_records: list[dict[str, Any]] = []
    for _, row in dc_df.iterrows():
        dc_records.append(
            {
                "dc_id": str(row["dc_id"]),
                "city": str(row["city"]),
                "max_power_per_hour": float(row["max_power_per_hour"]),
                "pue": float(row["pue"]),
                "water_usage_factor": float(row["water_usage_factor"]),
                "region_water_scarcity": float(row["region_water_scarcity"]),
            }
        )

    return PreparedSchedulerState(
        time_index=time_index,
        time_values=time_index.to_numpy(dtype="datetime64[ns]"),
        city_metrics=city_metrics,
        dc_records=dc_records,
    )


def _initialize_capacity_state(state: PreparedSchedulerState) -> dict[str, np.ndarray]:
    horizon = len(state.time_index)
    return {
        str(record["dc_id"]): np.zeros(horizon, dtype=np.float32)
        for record in state.dc_records
    }


def _candidate_start_positions(
    state: PreparedSchedulerState,
    earliest_start: pd.Timestamp,
    latest_start: pd.Timestamp,
    duration_hours: int,
) -> np.ndarray:
    if duration_hours <= 0 or duration_hours > len(state.time_index):
        return np.empty(0, dtype=np.int32)

    start_value = np.datetime64(pd.Timestamp(earliest_start).floor("h").to_datetime64(), "ns")
    end_value = np.datetime64(pd.Timestamp(latest_start).floor("h").to_datetime64(), "ns")
    lower = int(np.searchsorted(state.time_values, start_value, side="left"))
    upper = int(np.searchsorted(state.time_values, end_value, side="right")) - 1
    upper = min(upper, len(state.time_index) - duration_hours)
    if lower > upper:
        return np.empty(0, dtype=np.int32)
    return np.arange(lower, upper + 1, dtype=np.int32)


def _window_is_valid(city_metrics: PreparedCityMetrics, start_index: int, end_index: int) -> bool:
    valid_count = city_metrics.valid_prefix[end_index] - city_metrics.valid_prefix[start_index]
    return bool(valid_count >= (end_index - start_index))


def _window_mean(prefix: np.ndarray, start_index: int, end_index: int) -> float:
    span = max(1, end_index - start_index)
    return float((prefix[end_index] - prefix[start_index]) / span)


def _candidate_score(
    co2_mean: float,
    water_mean: float,
    dc_record: dict[str, Any],
    origin_city: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float, float]:
    co2_component = co2_mean * float(dc_record["pue"])
    water_component = (
        water_mean
        * float(dc_record["water_usage_factor"])
        * float(dc_record["region_water_scarcity"])
    )
    latency_penalty = gamma if origin_city != str(dc_record["city"]) else 0.0
    score = alpha * co2_component + beta * water_component + latency_penalty
    return float(score), float(co2_component), float(water_component)


def _capacity_available(
    used_capacity: dict[str, np.ndarray],
    dc_id: str,
    start_index: int,
    end_index: int,
    required_power: float,
    max_power_per_hour: float,
) -> bool:
    window = used_capacity[dc_id][start_index:end_index]
    if window.size == 0:
        return False
    return float(window.max()) + required_power <= max_power_per_hour


def _reserve_capacity(
    used_capacity: dict[str, np.ndarray],
    dc_id: str,
    start_index: int,
    end_index: int,
    required_power: float,
) -> None:
    used_capacity[dc_id][start_index:end_index] += required_power


def _enumerate_candidates(
    job: pd.Series,
    prepared_state: PreparedSchedulerState,
    used_capacity: dict[str, np.ndarray] | None,
    alpha: float,
    beta: float,
    gamma: float,
) -> list[ScheduleCandidate]:
    duration_hours = int(job["duration_hours"])
    earliest_start = pd.Timestamp(job["earliest_start"]).floor("h")
    latest_start = (pd.Timestamp(job["deadline"]) - pd.Timedelta(hours=duration_hours)).floor("h")
    required_power = float(job["power_demand"])

    if latest_start < earliest_start:
        return []

    start_positions = _candidate_start_positions(
        prepared_state,
        earliest_start,
        latest_start,
        duration_hours,
    )
    if start_positions.size == 0:
        return []

    capacity_state = used_capacity or _initialize_capacity_state(prepared_state)
    candidates: list[ScheduleCandidate] = []
    origin_city = str(job["origin_city"])

    for dc_record in prepared_state.dc_records:
        city = str(dc_record["city"])
        city_metrics = prepared_state.city_metrics.get(city)
        if city_metrics is None:
            continue

        dc_id = str(dc_record["dc_id"])
        for start_index in start_positions:
            end_index = int(start_index + duration_hours)
            if not _window_is_valid(city_metrics, int(start_index), end_index):
                continue
            if not _capacity_available(
                capacity_state,
                dc_id,
                int(start_index),
                end_index,
                required_power,
                float(dc_record["max_power_per_hour"]),
            ):
                continue

            co2_mean = _window_mean(city_metrics.co2_prefix, int(start_index), end_index)
            water_mean = _window_mean(city_metrics.water_prefix, int(start_index), end_index)
            score, co2_component, water_component = _candidate_score(
                co2_mean,
                water_mean,
                dc_record,
                origin_city,
                alpha,
                beta,
                gamma,
            )
            start_time = pd.Timestamp(prepared_state.time_index[int(start_index)])
            end_time = pd.Timestamp(prepared_state.time_index[end_index - 1]) + pd.Timedelta(hours=1)
            candidates.append(
                ScheduleCandidate(
                    dc_id=dc_id,
                    city=city,
                    start_time=start_time,
                    end_time=end_time,
                    score=score,
                    expected_co2_kg=required_power * co2_component * duration_hours,
                    expected_water_liters=required_power * water_component * duration_hours,
                    start_index=int(start_index),
                    end_index=end_index,
                )
            )

    return sorted(candidates, key=lambda candidate: candidate.score)


def _find_best_assignment(
    job: pd.Series,
    prepared_state: PreparedSchedulerState,
    used_capacity: dict[str, np.ndarray],
    alpha: float,
    beta: float,
    gamma: float,
) -> ScheduleCandidate | None:
    candidates = _enumerate_candidates(
        job=job,
        prepared_state=prepared_state,
        used_capacity=used_capacity,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    return candidates[0] if candidates else None


def rank_job_candidates(
    env_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    job: pd.Series,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.05,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return the best feasible routing options for a single job."""
    candidates_df = list_job_candidates(
        env_df=env_df,
        dc_df=dc_df,
        job=job,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    if candidates_df.empty:
        return candidates_df

    unique_candidates = (
        candidates_df.sort_values("scheduler_score")
        .drop_duplicates(subset=["assigned_city"], keep="first")
        .head(top_k)
        .reset_index(drop=True)
    )
    unique_candidates["rank"] = np.arange(1, len(unique_candidates) + 1)
    ordered_columns = [
        "rank",
        "job_id",
        "origin_city",
        "assigned_dc_id",
        "assigned_city",
        "scheduled_start",
        "scheduled_end",
        "scheduler_score",
        "expected_co2_kg",
        "expected_water_liters",
        "same_city",
    ]
    return unique_candidates[ordered_columns]


def list_job_candidates(
    env_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    job: pd.Series,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.05,
    prepared_state: PreparedSchedulerState | None = None,
) -> pd.DataFrame:
    """Return all feasible routing options for a single job."""
    state = prepared_state or prepare_scheduler_state(env_df, dc_df)
    candidates = _enumerate_candidates(
        job=job,
        prepared_state=state,
        used_capacity=_initialize_capacity_state(state),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    ranked_rows: list[dict[str, object]] = []
    for candidate in candidates:
        ranked_rows.append(
            {
                "job_id": job.get("job_id", "simulated_job"),
                "origin_city": str(job["origin_city"]),
                "assigned_dc_id": candidate.dc_id,
                "assigned_city": candidate.city,
                "scheduled_start": candidate.start_time,
                "scheduled_end": candidate.end_time,
                "scheduler_score": candidate.score,
                "expected_co2_kg": candidate.expected_co2_kg,
                "expected_water_liters": candidate.expected_water_liters,
                "same_city": int(str(job["origin_city"]) == candidate.city),
            }
        )
    return pd.DataFrame(ranked_rows)


def schedule_jobs(
    env_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.05,
    prepared_state: PreparedSchedulerState | None = None,
) -> pd.DataFrame:
    """Greedy multi-objective scheduler with hourly capacity constraints."""
    if jobs_df.empty:
        return jobs_df.copy()

    state = prepared_state or prepare_scheduler_state(env_df, dc_df)
    used_capacity = _initialize_capacity_state(state)

    jobs = jobs_df.copy()
    jobs["priority_sort"] = jobs["priority"].fillna(1)
    jobs = jobs.sort_values(["priority_sort", "earliest_start", "job_id"]).reset_index(drop=True)

    scheduled_rows: list[dict[str, object]] = []
    for _, job in jobs.iterrows():
        assignment = _find_best_assignment(
            job,
            state,
            used_capacity,
            alpha,
            beta,
            gamma,
        )

        if assignment is None:
            scheduled_rows.append(
                {
                    **job.to_dict(),
                    "assigned_dc_id": None,
                    "assigned_city": None,
                    "scheduled_start": pd.NaT,
                    "scheduled_end": pd.NaT,
                    "scheduler_score": np.nan,
                    "expected_co2_kg": np.nan,
                    "expected_water_liters": np.nan,
                    "scheduled": 0,
                }
            )
            continue

        _reserve_capacity(
            used_capacity,
            assignment.dc_id,
            assignment.start_index,
            assignment.end_index,
            float(job["power_demand"]),
        )
        scheduled_rows.append(
            {
                **job.to_dict(),
                "assigned_dc_id": assignment.dc_id,
                "assigned_city": assignment.city,
                "scheduled_start": assignment.start_time,
                "scheduled_end": assignment.end_time,
                "scheduler_score": assignment.score,
                "expected_co2_kg": assignment.expected_co2_kg,
                "expected_water_liters": assignment.expected_water_liters,
                "scheduled": 1,
            }
        )

    return pd.DataFrame(scheduled_rows).drop(columns=["priority_sort"], errors="ignore")


def schedule_jobs_naive(
    env_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    dc_df: pd.DataFrame,
    prepared_state: PreparedSchedulerState | None = None,
) -> pd.DataFrame:
    """Naive baseline: prefer same-city placement, otherwise first feasible DC."""
    naive_dc = dc_df.copy()
    return schedule_jobs(
        env_df,
        jobs_df,
        naive_dc,
        alpha=0.0,
        beta=0.0,
        gamma=0.20,
        prepared_state=prepared_state,
    )


def summarize_schedule(schedule_df: pd.DataFrame) -> dict[str, float]:
    scheduled_mask = schedule_df["scheduled"] == 1
    return {
        "scheduled_jobs": float(scheduled_mask.sum()),
        "total_jobs": float(len(schedule_df)),
        "coverage": float(scheduled_mask.mean()) if len(schedule_df) else 0.0,
        "total_expected_co2_kg": float(
            schedule_df.loc[scheduled_mask, "expected_co2_kg"].sum()
        ),
        "total_expected_water_liters": float(
            schedule_df.loc[scheduled_mask, "expected_water_liters"].sum()
        ),
    }


def compare_summaries(
    baseline: dict[str, float], optimized: dict[str, float]
) -> dict[str, float]:
    def reduction(old: float, new: float) -> float:
        if old == 0:
            return 0.0
        return ((old - new) / old) * 100.0

    return {
        "co2_reduction_pct": reduction(
            baseline["total_expected_co2_kg"], optimized["total_expected_co2_kg"]
        ),
        "water_reduction_pct": reduction(
            baseline["total_expected_water_liters"],
            optimized["total_expected_water_liters"],
        ),
        "coverage_delta_pct": (optimized["coverage"] - baseline["coverage"]) * 100.0,
        "scheduled_jobs_delta": optimized["scheduled_jobs"] - baseline["scheduled_jobs"],
    }


def run_scheduling_pipeline(
    data_path: str | Path,
    jobs_path: str | Path = DEFAULT_JOBS_PATH,
    dc_config_path: str | Path = DEFAULT_DC_CONFIG_PATH,
    job_limit: int | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.05,
    priority_filter: str | None = None,
) -> dict[str, object]:
    jobs_df = load_jobs_dataset(jobs_path)
    if priority_filter is not None and priority_filter != "all" and "priority_label" in jobs_df.columns:
        jobs_df = jobs_df[jobs_df["priority_label"].str.lower() == priority_filter.lower()].copy()
    if job_limit is not None:
        jobs_df = jobs_df.head(job_limit).copy()

    dc_df = load_dc_config(dc_config_path)
    data_df = load_dataset(data_path)
    env_df = prepare_environment_frame(data_df, city_subset=dc_df["city"].unique().tolist())
    prepared_state = prepare_scheduler_state(env_df, dc_df)

    optimized_schedule = schedule_jobs(
        env_df,
        jobs_df,
        dc_df,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        prepared_state=prepared_state,
    )
    baseline_schedule = schedule_jobs_naive(env_df, jobs_df, dc_df, prepared_state=prepared_state)

    optimized_summary = summarize_schedule(optimized_schedule)
    baseline_summary = summarize_schedule(baseline_schedule)
    comparison = compare_summaries(baseline_summary, optimized_summary)

    return {
        "environment": env_df,
        "optimized_schedule": optimized_schedule,
        "baseline_schedule": baseline_schedule,
        "optimized_summary": optimized_summary,
        "baseline_summary": baseline_summary,
        "comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first scheduling pipeline using environmental metrics and workload traces."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the master sustainability dataset (.zip, .parquet, .csv, or .xlsx).",
    )
    parser.add_argument(
        "--jobs-path",
        default=str(DEFAULT_JOBS_PATH),
        help="Path to the scheduler jobs CSV.",
    )
    parser.add_argument(
        "--dc-config-path",
        default=str(DEFAULT_DC_CONFIG_PATH),
        help="Path to the data-center configuration CSV.",
    )
    parser.add_argument(
        "--job-limit",
        type=int,
        default=100,
        help="Optional number of jobs to schedule for a quick experiment.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for carbon intensity in the scheduler objective.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight for water intensity in the scheduler objective.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.05,
        help="Latency penalty applied when routing away from the origin city.",
    )
    parser.add_argument(
        "--priority-filter",
        default="all",
        choices=["all", "high", "low"],
        help="Optional job-priority subset to schedule.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/schedule_results.csv",
        help="Where to write the optimized schedule CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_scheduling_pipeline(
        data_path=args.data_path,
        jobs_path=args.jobs_path,
        dc_config_path=args.dc_config_path,
        job_limit=args.job_limit,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        priority_filter=args.priority_filter,
    )

    output_path = _resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["optimized_schedule"].to_csv(output_path, index=False)

    print("Scheduling summary")
    print(
        f"  Jobs scheduled: {int(results['optimized_summary']['scheduled_jobs'])}/"
        f"{int(results['optimized_summary']['total_jobs'])}"
    )
    print(
        f"  Coverage: {results['optimized_summary']['coverage']:.2%}"
    )
    print(
        f"  Optimized CO2 (kg): {results['optimized_summary']['total_expected_co2_kg']:.2f}"
    )
    print(
        f"  Optimized water (L): {results['optimized_summary']['total_expected_water_liters']:.2f}"
    )
    print(
        f"  Baseline CO2 reduction: {results['comparison']['co2_reduction_pct']:.2f}%"
    )
    print(
        f"  Baseline water reduction: {results['comparison']['water_reduction_pct']:.2f}%"
    )
    print(f"  Saved optimized schedule to: {output_path}")


if __name__ == "__main__":
    main()
