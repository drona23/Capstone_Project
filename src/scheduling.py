from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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
    return pd.Series(
        {
            "job_id": job_id,
            "origin_city": origin_city,
            "power_demand": float(power_demand),
            "duration_hours": int(duration_hours),
            "earliest_start": pd.Timestamp(earliest_start),
            "deadline": pd.Timestamp(deadline),
            "priority": int(priority),
            "priority_label": "high" if int(priority) == 0 else "low",
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


def _build_environment_lookup(env_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    for city, city_df in env_df.groupby("city"):
        lookup[city] = city_df.set_index("timestamp").sort_index()
    return lookup


def _candidate_starts(
    time_index: pd.DatetimeIndex, earliest_start: pd.Timestamp, latest_start: pd.Timestamp
) -> list[pd.Timestamp]:
    mask = (time_index >= earliest_start) & (time_index <= latest_start)
    return list(time_index[mask])


def _candidate_score(
    metrics: pd.DataFrame,
    dc_row: pd.Series,
    origin_city: str,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float, float]:
    co2_component = (
        metrics["predicted_co2_intensity"].mean()
        * float(dc_row["pue"])
    )
    water_component = (
        metrics["predicted_water_intensity"].mean()
        * float(dc_row["water_usage_factor"])
        * float(dc_row["region_water_scarcity"])
    )
    latency_penalty = gamma if origin_city != dc_row["city"] else 0.0
    score = alpha * co2_component + beta * water_component + latency_penalty
    return float(score), float(co2_component), float(water_component)


def _capacity_available(
    used_capacity: dict[tuple[str, pd.Timestamp], float],
    dc_id: str,
    hours: pd.DatetimeIndex,
    required_power: float,
    max_power_per_hour: float,
) -> bool:
    for ts in hours:
        current = used_capacity.get((dc_id, ts), 0.0)
        if current + required_power > max_power_per_hour:
            return False
    return True


def _reserve_capacity(
    used_capacity: dict[tuple[str, pd.Timestamp], float],
    dc_id: str,
    hours: pd.DatetimeIndex,
    required_power: float,
) -> None:
    for ts in hours:
        used_capacity[(dc_id, ts)] = used_capacity.get((dc_id, ts), 0.0) + required_power


def _enumerate_candidates(
    job: pd.Series,
    dc_df: pd.DataFrame,
    env_lookup: dict[str, pd.DataFrame],
    time_index: pd.DatetimeIndex,
    used_capacity: dict[tuple[str, pd.Timestamp], float] | None,
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

    starts = _candidate_starts(time_index, earliest_start, latest_start)
    if not starts:
        return []

    candidates: list[ScheduleCandidate] = []
    capacity_state = used_capacity or {}
    for _, dc_row in dc_df.iterrows():
        city = dc_row["city"]
        city_env = env_lookup.get(city)
        if city_env is None:
            continue

        for start_time in starts:
            hours = pd.date_range(start=start_time, periods=duration_hours, freq="h")
            metrics = city_env.reindex(hours)
            if metrics[["predicted_co2_intensity", "predicted_water_intensity"]].isna().any().any():
                continue

            if not _capacity_available(
                capacity_state,
                str(dc_row["dc_id"]),
                hours,
                required_power,
                float(dc_row["max_power_per_hour"]),
            ):
                continue

            score, co2_component, water_component = _candidate_score(
                metrics,
                dc_row,
                str(job["origin_city"]),
                alpha,
                beta,
                gamma,
            )
            candidate = ScheduleCandidate(
                dc_id=str(dc_row["dc_id"]),
                city=city,
                start_time=start_time,
                end_time=hours[-1] + pd.Timedelta(hours=1),
                score=score,
                expected_co2_kg=required_power * co2_component * duration_hours,
                expected_water_liters=required_power * water_component * duration_hours,
            )
            candidates.append(candidate)

    return sorted(candidates, key=lambda candidate: candidate.score)


def _find_best_assignment(
    job: pd.Series,
    dc_df: pd.DataFrame,
    env_lookup: dict[str, pd.DataFrame],
    time_index: pd.DatetimeIndex,
    used_capacity: dict[tuple[str, pd.Timestamp], float],
    alpha: float,
    beta: float,
    gamma: float,
) -> ScheduleCandidate | None:
    candidates = _enumerate_candidates(
        job=job,
        dc_df=dc_df,
        env_lookup=env_lookup,
        time_index=time_index,
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
    env_lookup = _build_environment_lookup(env_df)
    time_index = pd.DatetimeIndex(sorted(env_df["timestamp"].unique()))
    candidates = _enumerate_candidates(
        job=job,
        dc_df=dc_df,
        env_lookup=env_lookup,
        time_index=time_index,
        used_capacity={},
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    unique_candidates: list[ScheduleCandidate] = []
    seen_cities: set[str] = set()
    for candidate in candidates:
        if candidate.city in seen_cities:
            continue
        unique_candidates.append(candidate)
        seen_cities.add(candidate.city)
        if len(unique_candidates) >= top_k:
            break

    ranked_rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(unique_candidates, start=1):
        ranked_rows.append(
            {
                "rank": rank,
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
) -> pd.DataFrame:
    """Greedy multi-objective scheduler with hourly capacity constraints."""
    env_lookup = _build_environment_lookup(env_df)
    time_index = pd.DatetimeIndex(sorted(env_df["timestamp"].unique()))
    used_capacity: dict[tuple[str, pd.Timestamp], float] = {}

    jobs = jobs_df.copy()
    jobs["priority_sort"] = jobs["priority"].fillna(1)
    jobs = jobs.sort_values(["priority_sort", "earliest_start", "job_id"]).reset_index(drop=True)

    scheduled_rows: list[dict[str, object]] = []
    for _, job in jobs.iterrows():
        assignment = _find_best_assignment(
            job,
            dc_df,
            env_lookup,
            time_index,
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

        hours = pd.date_range(
            start=assignment.start_time,
            end=assignment.end_time - pd.Timedelta(hours=1),
            freq="h",
        )
        _reserve_capacity(
            used_capacity,
            assignment.dc_id,
            hours,
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
) -> pd.DataFrame:
    """Naive baseline: prefer same-city placement, otherwise first feasible DC."""
    naive_dc = dc_df.copy()
    return schedule_jobs(env_df, jobs_df, naive_dc, alpha=0.0, beta=0.0, gamma=0.20)


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

    optimized_schedule = schedule_jobs(
        env_df,
        jobs_df,
        dc_df,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    baseline_schedule = schedule_jobs_naive(env_df, jobs_df, dc_df)

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
