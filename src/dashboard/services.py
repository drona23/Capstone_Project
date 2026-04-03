from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    from ..data_loader import load_dataset
    from ..scheduling import (
        build_job_record,
        load_dc_config,
        load_jobs_dataset,
        prepare_environment_frame,
        rank_job_candidates,
        run_scheduling_pipeline,
    )
except ImportError:
    from data_loader import load_dataset
    from scheduling import (
        build_job_record,
        load_dc_config,
        load_jobs_dataset,
        prepare_environment_frame,
        rank_job_candidates,
        run_scheduling_pipeline,
    )

from .config import (
    DEFAULT_COORDINATES_PATH,
    DEFAULT_DC_CONFIG_PATH,
    DEFAULT_JOBS_PATH,
    default_data_path,
    resolve_app_path,
)


def _detect_city_column(df: pd.DataFrame) -> str:
    if "CITY" in df.columns:
        return "CITY"
    if "city" in df.columns:
        return "city"
    raise ValueError("Dataset must include a city column named CITY or city.")


def _detect_timestamp_column(df: pd.DataFrame) -> str:
    for candidate in ["TIMESTAMP", "Timestamp", "timestamp"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Dataset must include a timestamp column.")


@st.cache_data(show_spinner=False)
def load_coordinate_frame(coords_path: str) -> pd.DataFrame:
    coords_df = pd.read_csv(coords_path)
    coords_df["latitude"] = pd.to_numeric(coords_df["latitude"], errors="coerce")
    coords_df["longitude"] = pd.to_numeric(coords_df["longitude"], errors="coerce")
    return coords_df.dropna(subset=["city", "latitude", "longitude"])


@st.cache_data(show_spinner="Loading master dataset...")
def load_master_frame(data_path: str) -> pd.DataFrame:
    return load_dataset(data_path)


@st.cache_data(show_spinner=False)
def load_scheduler_environment(data_path: str, dc_config_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_df = load_master_frame(data_path)
    dc_df = load_dc_config(dc_config_path)
    env_df = prepare_environment_frame(data_df, city_subset=dc_df["city"].dropna().unique().tolist())
    return env_df, dc_df


@st.cache_data(show_spinner=False)
def load_jobs_frame(jobs_path: str) -> pd.DataFrame:
    return load_jobs_dataset(jobs_path)


@st.cache_data(show_spinner=False)
def get_overview_payload(data_path: str, coords_path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = load_master_frame(data_path)
    coords_df = load_coordinate_frame(coords_path)

    city_col = _detect_city_column(raw_df)
    timestamp_col = _detect_timestamp_column(raw_df)
    data = raw_df.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")

    numeric_columns = ["co2_kg", "total_gen_kwh", "WUE_total", "liters", "dsci_0_500"]
    for column in numeric_columns:
        if column not in data.columns:
            data[column] = np.nan
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["carbon_intensity_kg_per_kwh"] = data["co2_kg"] / data["total_gen_kwh"].replace(0, np.nan)

    city_metrics = (
        data.groupby(city_col, as_index=False)
        .agg(
            avg_hourly_co2_kg=("co2_kg", "mean"),
            avg_hourly_liters=("liters", "mean"),
            avg_wue=("WUE_total", "mean"),
            avg_carbon_intensity=("carbon_intensity_kg_per_kwh", "mean"),
            avg_scarcity=("dsci_0_500", "mean"),
            records=(city_col, "size"),
        )
        .rename(columns={city_col: "city"})
    )

    city_metrics = city_metrics.merge(coords_df, on="city", how="left")
    city_metrics = city_metrics.dropna(subset=["latitude", "longitude"]).sort_values(
        "avg_hourly_co2_kg", ascending=False
    )

    timestamp_min = data[timestamp_col].min()
    timestamp_max = data[timestamp_col].max()
    summary = {
        "city_count": int(city_metrics["city"].nunique()),
        "region_count": int(data["EGRIDREGION"].nunique()) if "EGRIDREGION" in data.columns else 0,
        "mean_hourly_co2_kg": float(data["co2_kg"].mean()),
        "mean_hourly_liters": float(data["liters"].mean()) if "liters" in data.columns else float("nan"),
        "mean_wue": float(data["WUE_total"].mean()),
        "mean_carbon_intensity": float(data["carbon_intensity_kg_per_kwh"].mean()),
        "records": int(len(data)),
        "date_range": f"{timestamp_min:%Y-%m-%d} to {timestamp_max:%Y-%m-%d}",
    }
    return city_metrics, summary


@st.cache_data(show_spinner=False)
def get_scheduler_metadata(
    data_path: str,
    jobs_path: str,
    dc_config_path: str,
    coords_path: str,
) -> dict[str, object]:
    env_df, dc_df = load_scheduler_environment(data_path, dc_config_path)
    jobs_df = load_jobs_frame(jobs_path)
    coords_df = load_coordinate_frame(coords_path)

    priority_counts = (
        jobs_df["priority_label"].fillna("unknown").value_counts().to_dict()
        if "priority_label" in jobs_df.columns
        else {}
    )
    return {
        "origin_cities": sorted(coords_df["city"].unique().tolist()),
        "dc_cities": sorted(dc_df["city"].dropna().unique().tolist()),
        "timestamp_min": env_df["timestamp"].min().to_pydatetime(),
        "timestamp_max": env_df["timestamp"].max().to_pydatetime(),
        "job_counts": {str(key): int(value) for key, value in priority_counts.items()},
        "default_origin_city": str(dc_df["city"].iloc[0]),
    }


@st.cache_data(show_spinner=False)
def run_scheduler_batch(
    data_path: str,
    jobs_path: str,
    dc_config_path: str,
    priority_filter: str,
    job_limit: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict[str, object]:
    results = run_scheduling_pipeline(
        data_path=data_path,
        jobs_path=jobs_path,
        dc_config_path=dc_config_path,
        job_limit=job_limit,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        priority_filter=priority_filter,
    )

    optimized = results["optimized_schedule"].copy()
    optimized = optimized[optimized["scheduled"] == 1].copy()
    route_summary = pd.DataFrame()
    if not optimized.empty:
        route_summary = (
            optimized.groupby(["origin_city", "assigned_city"], as_index=False)
            .agg(
                jobs=("job_id", "count"),
                total_power=("power_demand", "sum"),
                total_expected_co2_kg=("expected_co2_kg", "sum"),
                total_expected_water_liters=("expected_water_liters", "sum"),
            )
            .sort_values(["jobs", "total_expected_co2_kg"], ascending=[False, False])
        )

    return {
        "optimized_summary": results["optimized_summary"],
        "baseline_summary": results["baseline_summary"],
        "comparison": results["comparison"],
        "route_summary": route_summary,
    }


@st.cache_data(show_spinner=False)
def run_route_simulation(
    data_path: str,
    dc_config_path: str,
    coords_path: str,
    origin_city: str,
    power_demand: float,
    duration_hours: int,
    earliest_start_iso: str,
    slack_hours: int,
    priority_code: int,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int,
) -> pd.DataFrame:
    env_df, dc_df = load_scheduler_environment(data_path, dc_config_path)
    coords_df = load_coordinate_frame(coords_path)

    earliest_start = pd.Timestamp(earliest_start_iso).floor("h")
    deadline = earliest_start + pd.Timedelta(hours=int(duration_hours) + int(slack_hours))
    job = build_job_record(
        origin_city=origin_city,
        power_demand=power_demand,
        duration_hours=duration_hours,
        earliest_start=earliest_start,
        deadline=deadline,
        priority=priority_code,
        job_id="ui_simulation_job",
    )
    candidates = rank_job_candidates(
        env_df=env_df,
        dc_df=dc_df,
        job=job,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        top_k=top_k,
    )
    if candidates.empty:
        return candidates

    origin_coords = coords_df.rename(
        columns={
            "city": "origin_city",
            "latitude": "origin_latitude",
            "longitude": "origin_longitude",
        }
    )[["origin_city", "origin_latitude", "origin_longitude"]]
    destination_coords = coords_df.rename(
        columns={
            "city": "assigned_city",
            "latitude": "destination_latitude",
            "longitude": "destination_longitude",
        }
    )[["assigned_city", "destination_latitude", "destination_longitude"]]

    return (
        candidates.merge(origin_coords, on="origin_city", how="left")
        .merge(destination_coords, on="assigned_city", how="left")
        .dropna(
            subset=[
                "origin_latitude",
                "origin_longitude",
                "destination_latitude",
                "destination_longitude",
            ]
        )
    )


def get_resolved_paths(
    data_path: str | None = None,
    jobs_path: str | None = None,
    dc_config_path: str | None = None,
    coords_path: str | None = None,
) -> dict[str, str]:
    resolved = {
        "data_path": str(resolve_app_path(data_path, default_data_path())),
        "jobs_path": str(resolve_app_path(jobs_path, DEFAULT_JOBS_PATH)),
        "dc_config_path": str(resolve_app_path(dc_config_path, DEFAULT_DC_CONFIG_PATH)),
        "coords_path": str(resolve_app_path(coords_path, DEFAULT_COORDINATES_PATH)),
    }
    return resolved
