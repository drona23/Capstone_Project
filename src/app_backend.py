from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import pandas as pd

try:
    from .data_loader import PROJECT_ROOT, load_dataset
    from .scheduling import (
        DEFAULT_DC_CONFIG_PATH,
        DEFAULT_JOBS_PATH,
        build_job_record,
        compare_summaries,
        list_job_candidates,
        load_dc_config,
        load_jobs_dataset,
        prepare_scheduler_state,
        prepare_environment_frame,
        schedule_jobs,
        schedule_jobs_naive,
        summarize_schedule,
    )
    from .baselines import (
        baseline_random,
        baseline_nearest_neighbor,
        baseline_time_of_day_heuristic,
    )
except ImportError:
    from data_loader import PROJECT_ROOT, load_dataset
    from scheduling import (
        DEFAULT_DC_CONFIG_PATH,
        DEFAULT_JOBS_PATH,
        build_job_record,
        compare_summaries,
        list_job_candidates,
        load_dc_config,
        load_jobs_dataset,
        prepare_scheduler_state,
        prepare_environment_frame,
        schedule_jobs,
        schedule_jobs_naive,
        summarize_schedule,
    )
    from baselines import (
        baseline_random,
        baseline_nearest_neighbor,
        baseline_time_of_day_heuristic,
    )


DEFAULT_COORDINATES_PATH = PROJECT_ROOT / "data" / "reference" / "city_coordinates.csv"
DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "base_data_with_metrics.parquet",
    PROJECT_ROOT / "data" / "base_data_with_metrics.csv",
    Path.home() / "Downloads" / "Archive 3.zip",
    PROJECT_ROOT / "data" / "sample_dataset.xlsx",
]
PATH_TYPES = ["balanced", "low_carbon", "low_water", "low_latency"]
PATH_TYPE_LABELS = {
    "balanced": "Balanced",
    "low_carbon": "Low Carbon",
    "low_water": "Low Water",
    "low_latency": "Low Latency",
}
PRIORITY_TO_CODE = {"high": 0, "medium": 1, "low": 2}
INTERACTIVE_HORIZON_HOURS = 24


def resolve_default_paths() -> dict[str, str]:
    data_path = SustainabilitySchedulingBackend._resolve_data_path(None)
    return {
        "data_path": str(data_path),
        "jobs_path": str(DEFAULT_JOBS_PATH),
        "dc_config_path": str(DEFAULT_DC_CONFIG_PATH),
        "coords_path": str(DEFAULT_COORDINATES_PATH),
    }


@dataclass(frozen=True)
class SchedulerInputs:
    origin_city: str
    priority: str
    latency_sensitivity: float
    workload_size: float
    alpha: float
    beta: float
    gamma: float
    start_time: pd.Timestamp
    top_k: int = 4


@dataclass(frozen=True)
class BatchSchedulerInputs:
    priority: str
    latency_sensitivity: float
    alpha: float
    beta: float
    gamma: float
    start_time: pd.Timestamp
    batch_size: int = 100


@dataclass(frozen=True)
class OverviewSnapshot:
    metrics: dict[str, float]
    nodes: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp


class SustainabilitySchedulingBackend:
    """Backend façade for API-backed scheduling simulations."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        jobs_path: str | Path | None = None,
        dc_config_path: str | Path | None = None,
        coords_path: str | Path | None = None,
    ) -> None:
        self.data_path = self._resolve_data_path(data_path)
        self.jobs_path = self._resolve_path(jobs_path, DEFAULT_JOBS_PATH)
        self.dc_config_path = self._resolve_path(dc_config_path, DEFAULT_DC_CONFIG_PATH)
        self.coords_path = self._resolve_path(coords_path, DEFAULT_COORDINATES_PATH)

        self.env_df = self._load_environment_frame(self.data_path, self.dc_config_path)
        self.jobs_df = load_jobs_dataset(self.jobs_path)
        self.dc_df = load_dc_config(self.dc_config_path)
        self.coords_df = self._load_coordinates(self.coords_path)
        self.dc_city_set = set(self.dc_df["city"].dropna().unique().tolist())
        self.cache_key = self._build_cache_key()

    @staticmethod
    def _resolve_path(path: str | Path | None, fallback: Path) -> Path:
        if path is None or str(path).strip() == "":
            return fallback

        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return PROJECT_ROOT / candidate

    @classmethod
    def _resolve_data_path(cls, data_path: str | Path | None) -> Path:
        if data_path is not None and str(data_path).strip() != "":
            return cls._resolve_path(data_path, DEFAULT_DATA_CANDIDATES[-1])

        for candidate in DEFAULT_DATA_CANDIDATES:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return DEFAULT_DATA_CANDIDATES[-1]

    @staticmethod
    def _load_coordinates(coords_path: Path) -> pd.DataFrame:
        coords_df = pd.read_csv(coords_path)
        coords_df["latitude"] = pd.to_numeric(coords_df["latitude"], errors="coerce")
        coords_df["longitude"] = pd.to_numeric(coords_df["longitude"], errors="coerce")
        coords_df = coords_df.dropna(subset=["city", "latitude", "longitude"]).copy()
        return coords_df

    @staticmethod
    def _load_environment_frame(data_path: Path, dc_config_path: Path) -> pd.DataFrame:
        data_df = load_dataset(data_path)
        return prepare_environment_frame(data_df)

    def _build_cache_key(self) -> str:
        parts: list[str] = []
        for path in [self.data_path, self.jobs_path, self.dc_config_path, self.coords_path]:
            resolved = Path(path)
            if resolved.exists():
                stat = resolved.stat()
                parts.append(f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}")
            else:
                parts.append(str(resolved))
        return "|".join(parts)

    def time_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.env_df["timestamp"].min(), self.env_df["timestamp"].max()

    def default_overview_window(self, window_days: int = 30) -> tuple[pd.Timestamp, pd.Timestamp]:
        env_start, env_end = self.time_bounds()
        jobs_start = self.jobs_df["earliest_start"].min()
        start = max(env_start, jobs_start.floor("h"))
        end = min(start + pd.Timedelta(days=window_days), env_end)
        return start, end

    def default_scheduler_timestamp(self) -> pd.Timestamp:
        start, _ = self.default_overview_window(window_days=1)
        return start

    def context(self) -> dict[str, Any]:
        duration_median = float(self.jobs_df["duration_hours"].median()) if "duration_hours" in self.jobs_df.columns else 1.0
        power_median = float(self.jobs_df["power_demand"].median()) if "power_demand" in self.jobs_df.columns else 100.0
        priority_counts = self.jobs_df.get("priority_label", pd.Series(dtype=str)).value_counts().to_dict()
        env_start, env_end = self.time_bounds()
        default_window_start, default_window_end = self.default_overview_window()
        return {
            "origin_cities": sorted(self.coords_df["city"].unique().tolist()),
            "default_origin_city": str(self.dc_df["city"].iloc[0]),
            "time_min": env_start.to_pydatetime(),
            "time_max": env_end.to_pydatetime(),
            "default_window_start": default_window_start.to_pydatetime(),
            "default_window_end": default_window_end.to_pydatetime(),
            "default_scheduler_timestamp": self.default_scheduler_timestamp().to_pydatetime(),
            "default_workload_size": round(power_median, 2),
            "default_duration_hours": max(1, int(round(duration_median))),
            "default_batch_size": 100,
            "batch_size_min": 50,
            "batch_size_max": 200,
            "priority_counts": {str(key): int(value) for key, value in priority_counts.items()},
        }

    @staticmethod
    def _filter_environment_window(
        env_df: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_time).floor("h")
        end = pd.Timestamp(end_time).floor("h")
        return env_df[(env_df["timestamp"] >= start) & (env_df["timestamp"] <= end)].copy()

    def _shift_trace_jobs_to_window(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        jobs = self.jobs_df.copy()
        if jobs.empty:
            return jobs

        offset = pd.Timestamp(start_time).floor("h") - jobs["earliest_start"].min().floor("h")
        jobs["earliest_start"] = jobs["earliest_start"] + offset
        jobs["deadline"] = jobs["deadline"] + offset

        window_hours = max(
            1,
            int(
                (
                    pd.Timestamp(end_time).floor("h")
                    - pd.Timestamp(start_time).floor("h")
                ).total_seconds()
                / 3600.0
            ),
        )
        interactive_limit = max(1, min(window_hours, INTERACTIVE_HORIZON_HOURS))
        raw_slack = (
            (jobs["deadline"] - jobs["earliest_start"]).dt.total_seconds() / 3600.0
            - pd.to_numeric(jobs["duration_hours"], errors="coerce")
        )
        jobs["duration_hours"] = (
            pd.to_numeric(jobs["duration_hours"], errors="coerce")
            .clip(lower=1, upper=interactive_limit)
            .round()
            .astype(int)
        )
        normalized_slack = raw_slack.clip(lower=1, upper=interactive_limit).round().astype(int)
        jobs["deadline"] = jobs["earliest_start"] + pd.to_timedelta(
            jobs["duration_hours"] + normalized_slack,
            unit="h",
        )

        return jobs[
            (jobs["earliest_start"] <= pd.Timestamp(end_time))
            & (jobs["deadline"] >= pd.Timestamp(start_time))
        ].copy()

    def _derive_job_shape(self, priority: str) -> tuple[int, int]:
        jobs = self.jobs_df.copy()
        if "priority_label" in jobs.columns:
            scoped = jobs[jobs["priority_label"].str.lower() == priority.lower()].copy()
            if scoped.empty and priority.lower() == "medium":
                scoped = jobs.copy()
            if not scoped.empty:
                jobs = scoped

        duration = int(
            min(
                INTERACTIVE_HORIZON_HOURS,
                max(
                    1,
                    round(
                        pd.to_numeric(jobs["duration_hours"], errors="coerce")
                        .clip(lower=1, upper=INTERACTIVE_HORIZON_HOURS)
                        .median()
                    ),
                ),
            )
        )
        slack_series = (
            (jobs["deadline"] - jobs["earliest_start"]).dt.total_seconds() / 3600.0
            - pd.to_numeric(jobs["duration_hours"], errors="coerce")
        )
        slack = (
            int(
                min(
                    INTERACTIVE_HORIZON_HOURS,
                    max(
                        1,
                        round(
                            slack_series.dropna()
                            .clip(lower=1, upper=INTERACTIVE_HORIZON_HOURS)
                            .median()
                        ),
                    ),
                )
            )
            if not slack_series.dropna().empty
            else 6
        )
        return duration, slack

    def _build_node_frame(self, env_window: pd.DataFrame) -> pd.DataFrame:
        if env_window.empty:
            return pd.DataFrame(
                columns=[
                    "city",
                    "avg_co2_intensity",
                    "avg_wue",
                    "avg_scarcity",
                    "records",
                    "latitude",
                    "longitude",
                    "state",
                    "zip",
                    "has_data_center",
                ]
            )

        nodes = (
            env_window.groupby("city", as_index=False)
            .agg(
                avg_co2_intensity=("predicted_co2_intensity", "mean"),
                avg_wue=("predicted_water_intensity", "mean"),
                avg_scarcity=("scarcity_index", "mean"),
                records=("timestamp", "size"),
            )
            .merge(self.coords_df, on="city", how="left")
            .dropna(subset=["latitude", "longitude"])
        )
        nodes["has_data_center"] = nodes["city"].isin(self.dc_city_set).astype(int)
        return nodes.sort_values("avg_co2_intensity", ascending=False).reset_index(drop=True)

    def build_overview_snapshot(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> OverviewSnapshot:
        env_window = self._filter_environment_window(self.env_df, start_time, end_time)
        jobs_window = self._shift_trace_jobs_to_window(start_time, end_time)
        nodes = self._build_node_frame(env_window)

        metrics = {
            "avg_co2_intensity": float(env_window["predicted_co2_intensity"].mean()) if not env_window.empty else 0.0,
            "avg_wue": float(env_window["predicted_water_intensity"].mean()) if not env_window.empty else 0.0,
            "total_jobs_processed": float(len(jobs_window)),
            "co2_reduction_pct": 0.0,
            "water_reduction_pct": 0.0,
            "coverage_pct": 0.0,
        }

        if not env_window.empty and not jobs_window.empty:
            optimized = schedule_jobs(env_window, jobs_window, self.dc_df, alpha=1.0, beta=1.0, gamma=1.0)
            baseline = schedule_jobs_naive(env_window, jobs_window, self.dc_df)
            optimized_summary = summarize_schedule(optimized)
            baseline_summary = summarize_schedule(baseline)
            comparison = compare_summaries(baseline_summary, optimized_summary)
            metrics.update(
                {
                    "total_jobs_processed": float(optimized_summary["total_jobs"]),
                    "co2_reduction_pct": float(comparison["co2_reduction_pct"]),
                    "water_reduction_pct": float(comparison["water_reduction_pct"]),
                    "coverage_pct": float(optimized_summary["coverage"] * 100.0),
                }
            )

        return OverviewSnapshot(
            metrics=metrics,
            nodes=nodes,
            start_time=pd.Timestamp(start_time),
            end_time=pd.Timestamp(end_time),
        )

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_km = 6371.0
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius_km * c

    @staticmethod
    def _minmax(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        if values.dropna().empty:
            return pd.Series([0.0] * len(values), index=values.index)
        low = values.min()
        high = values.max()
        if low == high:
            return pd.Series([0.0] * len(values), index=values.index)
        return (values - low) / (high - low)

    @staticmethod
    def _select_candidate(
        candidates: pd.DataFrame,
        sort_column: str,
        path_type: str,
        used_indices: set[int],
        used_cities: set[str],
    ) -> dict[str, Any] | None:
        if candidates.empty:
            return None
        ordered = candidates.sort_values(sort_column, ascending=True)
        for index, row in ordered.iterrows():
            candidate_city = str(row["assigned_city"])
            if index in used_indices or candidate_city in used_cities:
                continue
            used_indices.add(index)
            used_cities.add(candidate_city)
            payload = row.to_dict()
            payload["type"] = path_type
            return payload
        return None

    def _attach_route_metrics(self, candidates: pd.DataFrame, origin_city: str) -> pd.DataFrame:
        coords = self.coords_df[["city", "latitude", "longitude"]].copy()
        origin = coords.rename(
            columns={
                "city": "origin_city",
                "latitude": "origin_latitude",
                "longitude": "origin_longitude",
            }
        )
        destination = coords.rename(
            columns={
                "city": "assigned_city",
                "latitude": "destination_latitude",
                "longitude": "destination_longitude",
            }
        )
        enriched = candidates.merge(origin, on="origin_city", how="left").merge(
            destination, on="assigned_city", how="left"
        )
        enriched = enriched.dropna(
            subset=["origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude"]
        ).copy()

        latency_ms: list[float] = []
        for _, row in enriched.iterrows():
            distance_km = self._haversine_km(
                float(row["origin_latitude"]),
                float(row["origin_longitude"]),
                float(row["destination_latitude"]),
                float(row["destination_longitude"]),
            )
            latency_ms.append(5.0 + (distance_km / 200.0))
        enriched["latency"] = latency_ms

        return enriched

    @staticmethod
    def _batch_profile(priority: str) -> dict[str, int]:
        profiles = {
            "high": {"spread_hours": 4, "max_duration": 8, "max_slack": 2},
            "medium": {"spread_hours": 8, "max_duration": 12, "max_slack": 4},
            "low": {"spread_hours": 12, "max_duration": 16, "max_slack": 6},
        }
        return profiles.get(priority.lower(), profiles["medium"])

    def _build_batch_jobs(
        self,
        start_time: pd.Timestamp,
        batch_size: int,
        priority: str,
    ) -> pd.DataFrame:
        jobs = self.jobs_df.copy()
        if jobs.empty:
            return jobs

        profile = self._batch_profile(priority)
        sample_size = max(1, min(len(jobs), int(batch_size)))
        batch = (
            jobs.sort_values(["earliest_start", "job_id"])
            .head(sample_size)
            .reset_index(drop=True)
            .copy()
        )

        original_earliest = batch["earliest_start"].copy()
        original_slack = (
            (batch["deadline"] - batch["earliest_start"]).dt.total_seconds() / 3600.0
            - pd.to_numeric(batch["duration_hours"], errors="coerce")
        ).clip(lower=1)

        offset_hours = (
            (original_earliest.dt.floor("h") - original_earliest.min().floor("h"))
            .dt.total_seconds()
            .div(3600.0)
            .fillna(0)
            .astype(int)
        )
        batch["earliest_start"] = pd.Timestamp(start_time).floor("h") + pd.to_timedelta(
            offset_hours.mod(profile["spread_hours"]),
            unit="h",
        )

        duration_rank = pd.to_numeric(batch["duration_hours"], errors="coerce").rank(
            method="average",
            pct=True,
        )
        batch["duration_hours"] = (
            (duration_rank.fillna(0.5) * profile["max_duration"])
            .clip(lower=1, upper=profile["max_duration"])
            .round()
            .astype(int)
        )

        slack_rank = original_slack.rank(method="average", pct=True)
        slack_hours = (
            (slack_rank.fillna(0.5) * profile["max_slack"])
            .clip(lower=1, upper=profile["max_slack"])
            .round()
            .astype(int)
        )
        batch["deadline"] = batch["earliest_start"] + pd.to_timedelta(
            batch["duration_hours"] + slack_hours,
            unit="h",
        )

        priority_code = PRIORITY_TO_CODE.get(priority.lower(), 1)
        batch["priority"] = priority_code
        batch["priority_label"] = priority.lower()
        return batch

    def _attach_schedule_coordinates(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        if schedule_df.empty:
            return schedule_df.copy()

        coords = self.coords_df[["city", "latitude", "longitude"]].copy()
        origin = coords.rename(
            columns={
                "city": "origin_city",
                "latitude": "origin_latitude",
                "longitude": "origin_longitude",
            }
        )
        destination = coords.rename(
            columns={
                "city": "assigned_city",
                "latitude": "destination_latitude",
                "longitude": "destination_longitude",
            }
        )
        enriched = schedule_df.merge(origin, on="origin_city", how="left").merge(
            destination,
            on="assigned_city",
            how="left",
        )
        enriched = enriched.dropna(
            subset=["origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude"]
        ).copy()
        latency_ms: list[float] = []
        for _, row in enriched.iterrows():
            latency_ms.append(
                5.0
                + (
                    self._haversine_km(
                        float(row["origin_latitude"]),
                        float(row["origin_longitude"]),
                        float(row["destination_latitude"]),
                        float(row["destination_longitude"]),
                    )
                    / 200.0
                )
            )
        enriched["latency_ms"] = latency_ms
        return enriched

    def _aggregate_schedule_flows(
        self,
        schedule_df: pd.DataFrame,
        scenario: str,
    ) -> list[dict[str, Any]]:
        if schedule_df.empty:
            return []

        scheduled = self._attach_schedule_coordinates(schedule_df[schedule_df["scheduled"] == 1].copy())
        if scheduled.empty:
            return []

        flows = (
            scheduled.groupby(
                [
                    "origin_city",
                    "assigned_city",
                    "origin_latitude",
                    "origin_longitude",
                    "destination_latitude",
                    "destination_longitude",
                ],
                as_index=False,
            )
            .agg(
                jobs=("job_id", "count"),
                total_co2_kg=("expected_co2_kg", "sum"),
                total_water_liters=("expected_water_liters", "sum"),
                avg_scheduler_score=("scheduler_score", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
            )
            .sort_values(["jobs", "total_co2_kg"], ascending=[False, False])
            .reset_index(drop=True)
        )
        outputs: list[dict[str, Any]] = []
        for index, row in flows.iterrows():
            route = f"{row['origin_city']} -> {row['assigned_city']}"
            outputs.append(
                {
                    "flow_id": f"{scenario}:{index}:{row['origin_city']}:{row['assigned_city']}",
                    "route": route,
                    "origin_city": str(row["origin_city"]),
                    "assigned_city": str(row["assigned_city"]),
                    "origin_latitude": float(row["origin_latitude"]),
                    "origin_longitude": float(row["origin_longitude"]),
                    "destination_latitude": float(row["destination_latitude"]),
                    "destination_longitude": float(row["destination_longitude"]),
                    "jobs": int(row["jobs"]),
                    "co2": float(row["total_co2_kg"]),
                    "water_liters": float(row["total_water_liters"]),
                    "latency": float(row["avg_latency_ms"]),
                    "score": float(row["avg_scheduler_score"]),
                    "scenario": scenario,
                }
            )
        return outputs

    @staticmethod
    def _weighted_average_latency(flows: list[dict[str, Any]]) -> float:
        if not flows:
            return 0.0
        total_jobs = sum(int(flow.get("jobs", 0)) for flow in flows)
        if total_jobs <= 0:
            return 0.0
        weighted_latency = sum(
            float(flow.get("latency", 0.0)) * int(flow.get("jobs", 0))
            for flow in flows
        )
        return weighted_latency / total_jobs

    def build_batch_story(
        self,
        comparison: dict[str, float],
        optimized_flows: list[dict[str, Any]],
        baseline_flows: list[dict[str, Any]],
        batch_size: int,
    ) -> dict[str, str]:
        busiest_flow = optimized_flows[0] if optimized_flows else None
        if busiest_flow is None:
            return {
                "title": "No feasible batch schedule",
                "summary": "The current batch setup could not place any jobs inside the selected simulation window.",
            }

        baseline_latency = self._weighted_average_latency(baseline_flows)
        optimized_latency = self._weighted_average_latency(optimized_flows)
        latency_delta = optimized_latency - baseline_latency
        destination = busiest_flow["assigned_city"]
        summary = (
            f"The optimized batch sends the heaviest traffic toward {destination}, where the system finds a better "
            f"carbon and water profile for this {batch_size}-job window. Compared with the baseline schedule, "
            f"the batch changes CO2 by {float(comparison.get('co2_reduction_pct', 0.0)):+.1f}% and water by "
            f"{float(comparison.get('water_reduction_pct', 0.0)):+.1f}%, while average route latency moves "
            f"{latency_delta:+.1f} ms."
        )
        return {
            "title": f"Why {destination} absorbs the busiest flow",
            "summary": summary,
        }

    def run_batch_scheduler(self, inputs: BatchSchedulerInputs) -> dict[str, Any]:
        start_time = pd.Timestamp(inputs.start_time).floor("h")
        end_time = start_time + pd.Timedelta(hours=INTERACTIVE_HORIZON_HOURS)
        env_window = self._filter_environment_window(self.env_df, start_time, end_time)
        if env_window.empty:
            return {
                "time": start_time.isoformat(),
                "batch_size": int(inputs.batch_size),
                "priority": inputs.priority,
                "window_end": end_time.isoformat(),
                "nodes": [],
                "optimized_flows": [],
                "baseline_flows": [],
                "selected_flow": None,
                "optimized_summary": {
                    "scheduled_jobs": 0.0,
                    "total_jobs": float(inputs.batch_size),
                    "coverage": 0.0,
                    "total_expected_co2_kg": 0.0,
                    "total_expected_water_liters": 0.0,
                    "avg_latency_ms": 0.0,
                },
                "baseline_summary": {
                    "scheduled_jobs": 0.0,
                    "total_jobs": float(inputs.batch_size),
                    "coverage": 0.0,
                    "total_expected_co2_kg": 0.0,
                    "total_expected_water_liters": 0.0,
                    "avg_latency_ms": 0.0,
                },
                "comparison": {
                    "co2_reduction_pct": 0.0,
                    "water_reduction_pct": 0.0,
                    "coverage_delta_pct": 0.0,
                    "scheduled_jobs_delta": 0.0,
                    "latency_delta_ms": 0.0,
                },
                "insight": self.build_batch_story(
                    {
                        "co2_reduction_pct": 0.0,
                        "water_reduction_pct": 0.0,
                        "latency_delta_ms": 0.0,
                    },
                    [],
                    [],
                    int(inputs.batch_size),
                ),
            }

        jobs_window = self._build_batch_jobs(start_time, inputs.batch_size, inputs.priority)
        prepared_state = prepare_scheduler_state(env_window, self.dc_df)
        optimized = schedule_jobs(
            env_window,
            jobs_window,
            self.dc_df,
            alpha=float(inputs.alpha),
            beta=float(inputs.beta),
            gamma=float(inputs.gamma) * float(inputs.latency_sensitivity),
            prepared_state=prepared_state,
        )
        baseline = schedule_jobs_naive(
            env_window,
            jobs_window,
            self.dc_df,
            prepared_state=prepared_state,
        )

        optimized_flows = self._aggregate_schedule_flows(optimized, scenario="optimized")
        baseline_flows = self._aggregate_schedule_flows(baseline, scenario="baseline")
        optimized_summary = summarize_schedule(optimized)
        baseline_summary = summarize_schedule(baseline)
        optimized_summary["avg_latency_ms"] = self._weighted_average_latency(optimized_flows)
        baseline_summary["avg_latency_ms"] = self._weighted_average_latency(baseline_flows)

        comparison = compare_summaries(baseline_summary, optimized_summary)
        comparison["latency_delta_ms"] = (
            float(optimized_summary["avg_latency_ms"]) - float(baseline_summary["avg_latency_ms"])
        )

        nodes = self._build_node_frame(env_window)
        selected_flow = optimized_flows[0] if optimized_flows else None
        insight = self.build_batch_story(
            comparison,
            optimized_flows,
            baseline_flows,
            int(inputs.batch_size),
        )

        return {
            "time": start_time.isoformat(),
            "window_end": end_time.isoformat(),
            "batch_size": int(inputs.batch_size),
            "priority": inputs.priority,
            "nodes": nodes.fillna("").to_dict(orient="records"),
            "optimized_flows": optimized_flows,
            "baseline_flows": baseline_flows,
            "selected_flow": selected_flow,
            "optimized_summary": optimized_summary,
            "baseline_summary": baseline_summary,
            "comparison": comparison,
            "insight": insight,
        }

    def run_scheduler(self, inputs: SchedulerInputs) -> list[dict[str, Any]]:
        start_time = pd.Timestamp(inputs.start_time).floor("h")
        duration_hours, slack_hours = self._derive_job_shape(inputs.priority)
        deadline = start_time + pd.Timedelta(hours=duration_hours + slack_hours)
        priority_code = PRIORITY_TO_CODE.get(inputs.priority.lower(), 1)

        env_window = self._filter_environment_window(self.env_df, start_time, deadline)
        if env_window.empty:
            return []

        job = build_job_record(
            origin_city=inputs.origin_city,
            power_demand=float(inputs.workload_size),
            duration_hours=duration_hours,
            earliest_start=start_time,
            deadline=deadline,
            priority=priority_code,
            job_id="interactive_simulation_job",
        )
        candidates = list_job_candidates(
            env_df=env_window,
            dc_df=self.dc_df,
            job=job,
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
        )
        if candidates.empty:
            return []

        candidates = self._attach_route_metrics(candidates, inputs.origin_city)
        if candidates.empty:
            return []

        workload_energy_kwh = float(inputs.workload_size) * float(duration_hours)
        candidates["co2"] = pd.to_numeric(candidates["expected_co2_kg"], errors="coerce")
        candidates["wue"] = pd.to_numeric(candidates["expected_water_liters"], errors="coerce") / max(workload_energy_kwh, 1e-9)
        candidates["normalized_co2"] = self._minmax(candidates["co2"])
        candidates["normalized_wue"] = self._minmax(candidates["wue"])
        candidates["normalized_latency"] = self._minmax(candidates["latency"])
        candidates["score"] = (
            float(inputs.alpha) * candidates["normalized_co2"]
            + float(inputs.beta) * candidates["normalized_wue"]
            + float(inputs.gamma) * float(inputs.latency_sensitivity) * candidates["normalized_latency"]
        )
        candidates = candidates.sort_values("score", ascending=True)

        used_indices: set[int] = set()
        used_cities: set[str] = set()
        selected: list[dict[str, Any]] = []
        selectors = [
            ("score", "balanced"),
            ("co2", "low_carbon"),
            ("wue", "low_water"),
            ("latency", "low_latency"),
        ]
        for sort_column, path_type in selectors:
            candidate = self._select_candidate(
                candidates,
                sort_column,
                path_type,
                used_indices,
                used_cities,
            )
            if candidate is not None:
                selected.append(candidate)

        if not selected:
            return []

        outputs: list[dict[str, Any]] = []
        for candidate in selected[: inputs.top_k]:
            outputs.append(
                {
                    "path": [candidate["origin_city"], candidate["assigned_city"]],
                    "co2": float(candidate["co2"]),
                    "wue": float(candidate["wue"]),
                    "latency": float(candidate["latency"]),
                    "score": float(candidate["score"]),
                    "type": str(candidate["type"]),
                    "scheduled_start": pd.Timestamp(candidate["scheduled_start"]).isoformat(),
                    "scheduled_end": pd.Timestamp(candidate["scheduled_end"]).isoformat(),
                    "origin_city": str(candidate["origin_city"]),
                    "assigned_city": str(candidate["assigned_city"]),
                    "origin_latitude": float(candidate["origin_latitude"]),
                    "origin_longitude": float(candidate["origin_longitude"]),
                    "destination_latitude": float(candidate["destination_latitude"]),
                    "destination_longitude": float(candidate["destination_longitude"]),
                    "water_liters": float(candidate["expected_water_liters"]),
                }
            )
        return outputs

    def build_scheduler_nodes(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
        env_window = self._filter_environment_window(self.env_df, start_time, end_time)
        return self._build_node_frame(env_window)

    @staticmethod
    def explain_path_choice(
        selected_path: dict[str, Any],
        all_paths: list[dict[str, Any]],
    ) -> dict[str, str]:
        if not selected_path:
            return {
                "title": "No route selected",
                "summary": "Choose a candidate route to inspect its trade-offs.",
            }

        frame = pd.DataFrame(all_paths)
        best_co2 = frame["co2"].min() if not frame.empty else selected_path["co2"]
        best_wue = frame["wue"].min() if not frame.empty else selected_path["wue"]
        best_latency = frame["latency"].min() if not frame.empty else selected_path["latency"]

        def pct_delta(value: float, baseline: float) -> float:
            if baseline == 0:
                return 0.0
            return ((value - baseline) / baseline) * 100.0

        path_label = PATH_TYPE_LABELS.get(selected_path["type"], selected_path["type"].replace("_", " ").title())
        summary = (
            f"{path_label} routes {selected_path['path'][0]} to {selected_path['path'][-1]} "
            f"with score {selected_path['score']:.3f}. "
            f"Its carbon impact is {pct_delta(selected_path['co2'], best_co2):+.1f}% versus the lowest-carbon option, "
            f"water intensity is {pct_delta(selected_path['wue'], best_wue):+.1f}% versus the lowest-water option, "
            f"and latency is {pct_delta(selected_path['latency'], best_latency):+.1f}% versus the lowest-latency option."
        )

        if selected_path["type"] == "balanced":
            title = "Why the balanced route leads"
        elif selected_path["type"] == "low_carbon":
            title = "Why the low-carbon route wins"
        elif selected_path["type"] == "low_water":
            title = "Why the low-water route wins"
        else:
            title = "Why the low-latency route wins"

        return {"title": title, "summary": summary}

    @staticmethod
    def select_baseline_path(paths: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not paths:
            return None

        same_city_paths = [
            path for path in paths if path.get("path") and path["path"][0] == path["path"][-1]
        ]
        scoped = same_city_paths if same_city_paths else paths
        return min(
            scoped,
            key=lambda path: (
                float(path.get("latency", 0.0)),
                float(path.get("score", 0.0)),
            ),
        )

    @staticmethod
    def compare_paths(
        selected_path: dict[str, Any],
        baseline_path: dict[str, Any] | None,
    ) -> dict[str, float]:
        if not selected_path:
            return {
                "co2_reduction_pct": 0.0,
                "water_reduction_pct": 0.0,
                "latency_delta_ms": 0.0,
                "latency_delta_pct": 0.0,
            }

        baseline = baseline_path or selected_path

        def reduction(old: float, new: float) -> float:
            if old == 0:
                return 0.0
            return ((old - new) / old) * 100.0

        def delta_pct(new: float, old: float) -> float:
            if old == 0:
                return 0.0
            return ((new - old) / old) * 100.0

        selected_co2 = float(selected_path.get("co2", 0.0))
        selected_wue = float(selected_path.get("wue", 0.0))
        selected_latency = float(selected_path.get("latency", 0.0))
        baseline_co2 = float(baseline.get("co2", selected_co2))
        baseline_wue = float(baseline.get("wue", selected_wue))
        baseline_latency = float(baseline.get("latency", selected_latency))

        return {
            "co2_reduction_pct": reduction(baseline_co2, selected_co2),
            "water_reduction_pct": reduction(baseline_wue, selected_wue),
            "latency_delta_ms": selected_latency - baseline_latency,
            "latency_delta_pct": delta_pct(selected_latency, baseline_latency),
        }

    def run_baseline_comparison(
        self,
        start_time: pd.Timestamp,
        batch_size: int = 50,
        priority: str = "medium",
    ) -> dict[str, Any]:
        """Compare optimal scheduler vs three baselines on same batch."""
        from .scheduling import prepare_scheduler_state

        # Prepare state (self.env_df is already prepared)
        prepared_state = prepare_scheduler_state(self.env_df, self.dc_df)

        # Load and shift jobs
        end_time = start_time + pd.Timedelta(hours=48)
        jobs = self._shift_trace_jobs_to_window(start_time, end_time)

        # Limit batch
        jobs_batch = jobs.head(batch_size).copy()
        if jobs_batch.empty:
            return {
                "timestamp": start_time.isoformat(),
                "batch_size": 0,
                "results": [],
                "summary": "No jobs available for this time window",
            }

        # Run each strategy
        strategies = {
            "optimal": {
                "name": "Optimal (Your Method)",
                "co2_total": 0.0,
                "water_total": 0.0,
                "jobs_scheduled": 0,
            },
            "random": {
                "name": "Random Routing",
                "co2_total": 0.0,
                "water_total": 0.0,
                "jobs_scheduled": 0,
            },
            "nearest": {
                "name": "Nearest Data Center",
                "co2_total": 0.0,
                "water_total": 0.0,
                "jobs_scheduled": 0,
            },
            "time_of_day": {
                "name": "Time-of-Day Heuristic",
                "co2_total": 0.0,
                "water_total": 0.0,
                "jobs_scheduled": 0,
            },
        }

        # Run optimal scheduler
        for _, job in jobs_batch.iterrows():
            from .scheduling import list_job_candidates
            candidates_df = list_job_candidates(
                env_df=self.env_df,
                dc_df=self.dc_df,
                job=job,
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
                prepared_state=prepared_state,
            )
            if not candidates_df.empty:
                best_row = candidates_df.loc[candidates_df["scheduler_score"].idxmin()]
                strategies["optimal"]["co2_total"] += float(best_row["expected_co2_kg"])
                strategies["optimal"]["water_total"] += float(best_row["expected_water_liters"])
                strategies["optimal"]["jobs_scheduled"] += 1

        # Run random baseline
        for _, job in jobs_batch.iterrows():
            result = baseline_random(job, prepared_state)
            if result:
                strategies["random"]["co2_total"] += result.expected_co2_kg
                strategies["random"]["water_total"] += result.expected_water_liters
                strategies["random"]["jobs_scheduled"] += 1

        # Run nearest baseline
        for _, job in jobs_batch.iterrows():
            result = baseline_nearest_neighbor(job, prepared_state)
            if result:
                strategies["nearest"]["co2_total"] += result.expected_co2_kg
                strategies["nearest"]["water_total"] += result.expected_water_liters
                strategies["nearest"]["jobs_scheduled"] += 1

        # Run time-of-day baseline
        for _, job in jobs_batch.iterrows():
            result = baseline_time_of_day_heuristic(job, prepared_state)
            if result:
                strategies["time_of_day"]["co2_total"] += result.expected_co2_kg
                strategies["time_of_day"]["water_total"] += result.expected_water_liters
                strategies["time_of_day"]["jobs_scheduled"] += 1

        # Calculate reductions vs baselines
        optimal_co2 = strategies["optimal"]["co2_total"]
        optimal_water = strategies["optimal"]["water_total"]

        def calc_reduction(baseline_total: float, optimal_total: float) -> float:
            if baseline_total == 0:
                return 0.0
            return ((baseline_total - optimal_total) / baseline_total) * 100.0

        results = []
        for key, stats in strategies.items():
            result_item = {
                "strategy": stats["name"],
                "jobs_scheduled": stats["jobs_scheduled"],
                "total_co2_kg": round(stats["co2_total"], 2),
                "total_water_liters": round(stats["water_total"], 2),
                "avg_co2_per_job": (
                    round(stats["co2_total"] / stats["jobs_scheduled"], 4)
                    if stats["jobs_scheduled"] > 0
                    else 0.0
                ),
                "avg_water_per_job": (
                    round(stats["water_total"] / stats["jobs_scheduled"], 4)
                    if stats["jobs_scheduled"] > 0
                    else 0.0
                ),
            }

            # Add reduction metrics relative to each baseline
            if key != "optimal":
                result_item["co2_reduction_vs_optimal_pct"] = calc_reduction(
                    stats["co2_total"], optimal_co2
                )
                result_item["water_reduction_vs_optimal_pct"] = calc_reduction(
                    stats["water_total"], optimal_water
                )

            results.append(result_item)

        # Calculate averages
        avg_co2_random = (
            strategies["random"]["co2_total"] / strategies["random"]["jobs_scheduled"]
            if strategies["random"]["jobs_scheduled"] > 0
            else 0.0
        )
        avg_co2_optimal = (
            strategies["optimal"]["co2_total"] / strategies["optimal"]["jobs_scheduled"]
            if strategies["optimal"]["jobs_scheduled"] > 0
            else 0.0
        )

        co2_improvement = calc_reduction(avg_co2_random, avg_co2_optimal)

        return {
            "timestamp": start_time.isoformat(),
            "batch_size": int(batch_size),
            "results": results,
            "summary": {
                "total_jobs_in_batch": len(jobs_batch),
                "co2_improvement_vs_random_pct": round(co2_improvement, 2),
                "winner": "Optimal" if optimal_co2 <= strategies["random"]["co2_total"] else "Random",
            },
        }

    @classmethod
    def build_path_story(
        cls,
        selected_path: dict[str, Any],
        all_paths: list[dict[str, Any]],
        baseline_path: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        if not selected_path:
            return {
                "title": "No recommendation available",
                "summary": "The system could not generate a feasible route for the selected hour.",
            }

        comparison = cls.compare_paths(selected_path, baseline_path)
        selected_label = PATH_TYPE_LABELS.get(
            str(selected_path.get("type", "balanced")),
            str(selected_path.get("type", "balanced")).replace("_", " ").title(),
        )
        destination = selected_path["path"][-1]

        strongest_driver = max(
            [
                ("carbon", comparison["co2_reduction_pct"]),
                ("water", comparison["water_reduction_pct"]),
                ("latency", -comparison["latency_delta_pct"]),
            ],
            key=lambda item: item[1],
        )[0]

        if strongest_driver == "carbon":
            rationale = "lower carbon intensity at the destination"
        elif strongest_driver == "water":
            rationale = "lower water stress and better water efficiency"
        else:
            rationale = "a smaller latency penalty than the alternatives"

        summary = (
            f"The recommended route sends the workload to {destination} using the {selected_label.lower()} option. "
            f"Compared with the baseline route, it changes CO2 by {comparison['co2_reduction_pct']:+.1f}% "
            f"and water impact by {comparison['water_reduction_pct']:+.1f}%, with latency moving "
            f"{comparison['latency_delta_ms']:+.1f} ms. The recommendation is driven mainly by {rationale}."
        )
        return {
            "title": f"Why {destination} is recommended",
            "summary": summary,
        }


def run_scheduler(
    backend: SustainabilitySchedulingBackend,
    inputs: SchedulerInputs,
) -> list[dict[str, Any]]:
    """Module-level scheduler entry point for interactive simulation clients."""
    return backend.run_scheduler(inputs)
