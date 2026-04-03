from __future__ import annotations

import pandas as pd
import streamlit as st

from .components import KpiCard, inject_app_styles, render_kpi_cards, render_page_header
from .maps import build_overview_map
from .services import get_overview_payload, get_resolved_paths


def _format_number(value: float, decimals: int = 1, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def render_overview_page() -> None:
    inject_app_styles()
    defaults = get_resolved_paths()
    with st.sidebar:
        st.header("Data Sources")
        data_path = st.text_input("Master dataset", value=defaults["data_path"], key="overview_data_path")
        coords_path = st.text_input("City coordinates", value=defaults["coords_path"], key="overview_coords_path")

    render_page_header(
        "Overview",
        "A city-level snapshot of carbon and water conditions across the research dataset.",
    )

    try:
        city_metrics, summary = get_overview_payload(data_path, coords_path)
    except Exception as exc:
        st.error(f"Unable to load overview data: {exc}")
        return

    render_kpi_cards(
        [
            KpiCard("Cities covered", f"{summary['city_count']}", f"Across {summary['region_count']} eGRID regions."),
            KpiCard("Avg hourly CO2", _format_number(summary["mean_hourly_co2_kg"], 0, " kg"), summary["date_range"]),
            KpiCard("Avg WUE", _format_number(summary["mean_wue"], 3), "Mean water-usage effectiveness across all hours."),
            KpiCard("Avg carbon intensity", _format_number(summary["mean_carbon_intensity"], 4), f"Built from {summary['records']:,} hourly records."),
        ]
    )

    st.markdown("")
    left_col, right_col = st.columns([1.7, 1.0], gap="large")
    with left_col:
        st.subheader("CO2 map")
        overview_map = build_overview_map(city_metrics)
        if overview_map is None:
            st.info("No city coordinates were available for the overview map.")
        else:
            st.pydeck_chart(overview_map, use_container_width=True)

    with right_col:
        st.subheader("Highest-impact cities")
        st.dataframe(
            city_metrics[
                [
                    "city",
                    "state",
                    "avg_hourly_co2_kg",
                    "avg_wue",
                    "avg_carbon_intensity",
                    "avg_scarcity",
                ]
            ]
            .head(12)
            .rename(
                columns={
                    "city": "City",
                    "state": "State",
                    "avg_hourly_co2_kg": "Avg hourly CO2 (kg)",
                    "avg_wue": "Avg WUE",
                    "avg_carbon_intensity": "Avg carbon intensity",
                    "avg_scarcity": "Avg scarcity",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("City detail table")
    st.dataframe(
        city_metrics.rename(
            columns={
                "city": "City",
                "state": "State",
                "zip": "ZIP",
                "avg_hourly_co2_kg": "Avg hourly CO2 (kg)",
                "avg_hourly_liters": "Avg hourly liters",
                "avg_wue": "Avg WUE",
                "avg_carbon_intensity": "Avg carbon intensity",
                "avg_scarcity": "Avg scarcity",
                "records": "Hourly records",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
