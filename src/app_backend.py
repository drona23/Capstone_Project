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
        prepare_environment_frame,
        schedule_jobs,
        schedule_jobs_naive,
        summarize_schedule,
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
        prepare_environment_frame,
        schedule_jobs,
        schedule_jobs_naive,
        summarize_schedule,
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
class OverviewSnapshot:
    metrics: dict[str, float]
    nodes: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp


class SustainabilitySchedulingBackend:
    """Backend façade for the Streamlit simulator."""

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


def run_scheduler(
    backend: SustainabilitySchedulingBackend,
    inputs: SchedulerInputs,
) -> list[dict[str, Any]]:
    """Module-level scheduler entry point for the Streamlit app."""
    return backend.run_scheduler(inputs)
