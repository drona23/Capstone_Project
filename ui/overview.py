from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_backend import OverviewSnapshot, SustainabilitySchedulingBackend
from utils.visualization import build_network_deck, render_kpi_cards

BACKEND_HASH = {SustainabilitySchedulingBackend: lambda backend: backend.cache_key}


@st.cache_data(show_spinner=False, hash_funcs=BACKEND_HASH)
def load_overview_snapshot(
    backend: SustainabilitySchedulingBackend,
    start_time_iso: str,
    end_time_iso: str,
) -> OverviewSnapshot:
    return backend.build_overview_snapshot(pd.Timestamp(start_time_iso), pd.Timestamp(end_time_iso))


def _format_metric(value: float, suffix: str = "", decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}{suffix}"


def render_overview_page(backend: SustainabilitySchedulingBackend) -> None:
    context = backend.context()
    st.title("Interactive Sustainability-Aware Workload Scheduling Simulator")
    st.markdown(
        '<div class="capstone-lead">System-level visibility across carbon intensity, water usage, and workload routing outcomes.</div>',
        unsafe_allow_html=True,
    )

    time_window = st.slider(
        "Time Window",
        min_value=context["time_min"],
        max_value=context["time_max"],
        value=(context["default_window_start"], context["default_window_end"]),
        format="YYYY-MM-DD HH:mm",
        help="Adjust the time window to update the map and system KPIs.",
    )
    start_time, end_time = time_window
    snapshot = load_overview_snapshot(backend, start_time.isoformat(), end_time.isoformat())

    render_kpi_cards(
        [
            {
                "label": "Average CO2 Intensity",
                "value": _format_metric(snapshot.metrics["avg_co2_intensity"], " kg/kWh", 4),
                "help": "Mean modeled carbon intensity in the selected window.",
            },
            {
                "label": "Average WUE",
                "value": _format_metric(snapshot.metrics["avg_wue"], " L/kWh", 3),
                "help": "Water-use effectiveness across the active node set.",
            },
            {
                "label": "Total Jobs Processed",
                "value": f"{int(snapshot.metrics['total_jobs_processed']):,}",
                "help": f"Coverage in the selected window: {snapshot.metrics['coverage_pct']:.1f}%.",
            },
            {
                "label": "CO2 Reduction vs Baseline",
                "value": _format_metric(snapshot.metrics["co2_reduction_pct"], "%", 2),
                "help": "Compared with a naive same-city-first routing baseline.",
            },
            {
                "label": "Water Reduction vs Baseline",
                "value": _format_metric(snapshot.metrics["water_reduction_pct"], "%", 2),
                "help": "Lower is better when routing around high-WUE nodes.",
            },
        ]
    )

    left_col, right_col = st.columns([1.8, 1.0], gap="large")
    with left_col:
        st.subheader("Carbon and water conditions map")
        deck = build_network_deck(snapshot.nodes)
        if deck is None:
            st.info("No node data is available for the selected time window.")
        else:
            st.pydeck_chart(deck, use_container_width=True)

    with right_col:
        st.subheader("Highest-impact nodes")
        st.dataframe(
            snapshot.nodes[
                [
                    "city",
                    "state",
                    "avg_co2_intensity",
                    "avg_wue",
                    "avg_scarcity",
                    "has_data_center",
                ]
            ]
            .head(12)
            .rename(
                columns={
                    "city": "City",
                    "state": "State",
                    "avg_co2_intensity": "CO2 intensity",
                    "avg_wue": "WUE",
                    "avg_scarcity": "Scarcity",
                    "has_data_center": "DC node",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Node detail")
    st.dataframe(
        snapshot.nodes.rename(
            columns={
                "city": "City",
                "state": "State",
                "zip": "ZIP",
                "avg_co2_intensity": "CO2 intensity",
                "avg_wue": "WUE",
                "avg_scarcity": "Scarcity",
                "records": "Records",
                "has_data_center": "DC node",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
