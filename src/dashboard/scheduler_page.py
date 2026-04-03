from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import streamlit as st

from .components import KpiCard, inject_app_styles, render_kpi_cards, render_page_header
from .maps import build_routing_map
from .services import (
    get_resolved_paths,
    get_scheduler_metadata,
    run_route_simulation,
    run_scheduler_batch,
)


PRIORITY_OPTIONS = {
    "All": "all",
    "High": "high",
    "Low": "low",
}
ROUTE_PRIORITY_CODES = {
    "High": 0,
    "Standard": 1,
    "Low": 1,
}


def _format_percent(value: float) -> str:
    return f"{value:.2f}%"


def render_scheduler_page() -> None:
    inject_app_styles()
    defaults = get_resolved_paths()

    with st.sidebar:
        st.header("Simulation Sources")
        data_path = st.text_input("Master dataset", value=defaults["data_path"], key="scheduler_data_path")
        jobs_path = st.text_input("Jobs dataset", value=defaults["jobs_path"], key="scheduler_jobs_path")
        dc_config_path = st.text_input("DC config", value=defaults["dc_config_path"], key="scheduler_dc_config_path")
        coords_path = st.text_input("City coordinates", value=defaults["coords_path"], key="scheduler_coords_path")

    render_page_header(
        "Scheduler Simulator",
        "Tune routing weights, latency sensitivity, and workload priority to explore greener placement options.",
    )

    try:
        metadata = get_scheduler_metadata(data_path, jobs_path, dc_config_path, coords_path)
    except Exception as exc:
        st.error(f"Unable to load scheduler metadata: {exc}")
        return

    info_col, stats_col = st.columns([1.3, 1.0], gap="large")
    with info_col:
        st.caption(
            "Available time window: "
            f"{metadata['timestamp_min']:%Y-%m-%d %H:%M} to {metadata['timestamp_max']:%Y-%m-%d %H:%M}"
        )
    with stats_col:
        job_counts = metadata["job_counts"]
        st.caption(
            "Azure job sample counts: "
            f"high={job_counts.get('high', 0)}, low={job_counts.get('low', 0)}"
        )

    with st.form("scheduler_controls"):
        control_left, control_mid, control_right = st.columns(3, gap="large")
        with control_left:
            priority_focus = st.selectbox("Priority focus", list(PRIORITY_OPTIONS.keys()), index=0)
            latency_penalty = st.slider("Latency sensitivity", min_value=0.0, max_value=0.50, value=0.08, step=0.01)
            job_limit = st.slider("Batch job limit", min_value=10, max_value=250, value=80, step=10)
            route_priority = st.selectbox("Simulated route priority", list(ROUTE_PRIORITY_CODES.keys()), index=0)
        with control_mid:
            co2_weight = st.slider("CO2 weight", min_value=0.0, max_value=3.0, value=1.4, step=0.1)
            water_weight = st.slider("Water weight", min_value=0.0, max_value=3.0, value=1.1, step=0.1)
            power_demand = st.number_input("Simulated job power demand", min_value=10.0, max_value=600.0, value=140.0, step=10.0)
            duration_hours = st.slider("Simulated duration (hours)", min_value=1, max_value=24, value=6, step=1)
        with control_right:
            origin_city = st.selectbox(
                "Origin city",
                metadata["origin_cities"],
                index=metadata["origin_cities"].index(metadata["default_origin_city"]),
            )
            default_date = metadata["timestamp_min"].date()
            start_date = st.date_input("Earliest start date", value=default_date)
            start_hour = st.slider("Earliest start hour", min_value=0, max_value=23, value=12, step=1)
            slack_hours = st.slider("Deadline slack (hours)", min_value=1, max_value=24, value=8, step=1)
            top_k = st.slider("Routing paths to display", min_value=3, max_value=8, value=5, step=1)

        submitted = st.form_submit_button("Run simulator", use_container_width=True)

    if not submitted and "scheduler_has_run" not in st.session_state:
        submitted = True
    if not submitted:
        return
    st.session_state["scheduler_has_run"] = True

    earliest_start = datetime.combine(start_date, time(hour=start_hour))

    try:
        batch_results = run_scheduler_batch(
            data_path=data_path,
            jobs_path=jobs_path,
            dc_config_path=dc_config_path,
            priority_filter=PRIORITY_OPTIONS[priority_focus],
            job_limit=job_limit,
            alpha=co2_weight,
            beta=water_weight,
            gamma=latency_penalty,
        )
        route_results = run_route_simulation(
            data_path=data_path,
            dc_config_path=dc_config_path,
            coords_path=coords_path,
            origin_city=origin_city,
            power_demand=power_demand,
            duration_hours=duration_hours,
            earliest_start_iso=earliest_start.isoformat(),
            slack_hours=slack_hours,
            priority_code=ROUTE_PRIORITY_CODES[route_priority],
            alpha=co2_weight,
            beta=water_weight,
            gamma=latency_penalty,
            top_k=top_k,
        )
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return

    optimized_summary = batch_results["optimized_summary"]
    comparison = batch_results["comparison"]
    render_kpi_cards(
        [
            KpiCard(
                "Scheduled jobs",
                f"{int(optimized_summary['scheduled_jobs'])}/{int(optimized_summary['total_jobs'])}",
                f"Coverage: {optimized_summary['coverage']:.1%}",
            ),
            KpiCard(
                "CO2 reduction vs baseline",
                _format_percent(comparison["co2_reduction_pct"]),
                "Estimated reduction against naive same-city-first placement.",
            ),
            KpiCard(
                "Water reduction vs baseline",
                _format_percent(comparison["water_reduction_pct"]),
                "Weighted by site water intensity and scarcity assumptions.",
            ),
            KpiCard(
                "Active objective mix",
                f"CO2 {co2_weight:.1f} / Water {water_weight:.1f}",
                f"Latency penalty {latency_penalty:.2f} with {priority_focus.lower()}-priority batch focus.",
            ),
        ]
    )

    st.markdown("")
    map_col, table_col = st.columns([1.7, 1.0], gap="large")
    with map_col:
        st.subheader("Routing paths map")
        route_map = build_routing_map(route_results)
        if route_map is None:
            st.info("No feasible routing path was found for the current simulation inputs.")
        else:
            st.pydeck_chart(route_map, use_container_width=True)

    with table_col:
        st.subheader("Top routing options")
        if route_results.empty:
            st.write("No feasible routes matched the selected timing and capacity window.")
        else:
            st.dataframe(
                route_results[
                    [
                        "rank",
                        "assigned_city",
                        "scheduled_start",
                        "expected_co2_kg",
                        "expected_water_liters",
                        "same_city",
                    ]
                ].rename(
                    columns={
                        "rank": "Rank",
                        "assigned_city": "Destination",
                        "scheduled_start": "Start",
                        "expected_co2_kg": "CO2 (kg)",
                        "expected_water_liters": "Water (L)",
                        "same_city": "Same city",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("Batch routing summary")
    route_summary = batch_results["route_summary"]
    if isinstance(route_summary, pd.DataFrame) and not route_summary.empty:
        st.dataframe(
            route_summary.rename(
                columns={
                    "origin_city": "Origin",
                    "assigned_city": "Assigned city",
                    "jobs": "Jobs",
                    "total_power": "Total power",
                    "total_expected_co2_kg": "Total CO2 (kg)",
                    "total_expected_water_liters": "Total water (L)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("No batch routes were generated for the selected controls.")
