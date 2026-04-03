from __future__ import annotations

import json
import os
from datetime import datetime, time
from typing import Any

import pandas as pd
import requests
import streamlit as st

from utils.visualization import (
    build_network_deck,
    render_explanation_card,
    render_kpi_cards,
    render_page_header,
    render_panel_header,
    render_route_chip,
)

DEFAULT_API_URL = os.environ.get("SCHEDULER_API_URL", "http://127.0.0.1:8000")
API_TIMEOUT_SECONDS = 30


def _request_json(method: str, url: str, **kwargs) -> dict[str, Any]:
    response = requests.request(method, url, timeout=API_TIMEOUT_SECONDS, **kwargs)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def load_context(api_url: str) -> dict[str, Any]:
    return _request_json("GET", f"{api_url.rstrip('/')}/context")


@st.cache_data(show_spinner=False)
def run_simulation(api_url: str, payload_json: str) -> dict[str, Any]:
    payload = json.loads(payload_json)
    return _request_json("POST", f"{api_url.rstrip('/')}/simulate", json=payload)


def _default_simulation_state(context: dict[str, Any]) -> dict[str, Any]:
    default_time = pd.Timestamp(context["default_time"]).floor("h")
    return {
        "priority": "medium",
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "selected_date": default_time.date(),
        "selected_hour": int(default_time.hour),
        "time": default_time,
        "compare_mode": "Candidates",
    }


def _clamp_timestamp(
    selected_date: datetime.date,
    selected_hour: int,
    time_min: pd.Timestamp,
    time_max: pd.Timestamp,
) -> pd.Timestamp:
    selected_timestamp = pd.Timestamp(datetime.combine(selected_date, time(hour=int(selected_hour))))
    return min(max(selected_timestamp, time_min.floor("h")), time_max.floor("h"))


def _payload_from_state(state: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {
        "priority": state["priority"],
        "alpha": float(state["alpha"]),
        "beta": float(state["beta"]),
        "gamma": float(state["gamma"]),
        "time": pd.Timestamp(state["time"]).isoformat(),
        "latency_sensitivity": 0.55,
        "workload_size": float(context["default_workload_size"]),
        "origin_city": str(context["origin_city"]),
        "top_k": 4,
    }


def _best_path_label(path: dict[str, Any] | None) -> str:
    if not path:
        return "No feasible route"
    return path.get("route", " -> ".join(path.get("path", [])))

def _comparison_table(paths: list[dict[str, Any]]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()

    frame = pd.DataFrame(paths).copy()
    frame["Path"] = frame["type"].map(PATH_TYPE_LABELS)
    frame["Route"] = frame["route"]
    frame["CO2 (kg)"] = frame["co2"].map(lambda value: round(float(value), 2))
    frame["WUE (L/kWh)"] = frame["wue"].map(lambda value: round(float(value), 3))
    frame["Latency (ms)"] = frame["latency"].map(lambda value: round(float(value), 1))
    frame["Score"] = frame["score"].map(lambda value: round(float(value), 3))
    return frame[["Path", "Route", "CO2 (kg)", "WUE (L/kWh)", "Latency (ms)", "Score"]].head(3)


def render_scheduler_page(api_url: str = DEFAULT_API_URL) -> None:
    try:
        context = load_context(api_url)
    except Exception as exc:
        render_page_header(
            "Sustainability-Aware Workload Routing",
            "Live simulation view for routing decisions across carbon, water, and latency trade-offs.",
            "API unavailable",
        )
        st.error(
            "The simulator could not reach the FastAPI backend at "
            f"`{api_url}`. Start it with `uvicorn src.api:app --reload` and refresh. "
            f"Details: {exc}"
        )
        return

    time_min = pd.Timestamp(context["time_min"])
    time_max = pd.Timestamp(context["time_max"])

    if "simulation_state" not in st.session_state:
        st.session_state["simulation_state"] = _default_simulation_state(context)

    state = dict(st.session_state["simulation_state"])
    render_page_header(
        "Sustainability-Aware Workload Routing",
        "A presentation-ready simulation showing where the job goes, why it moves, and what that decision does to carbon and water impact.",
        f"Connected to {api_url}",
    )

    payload = _payload_from_state(state, context)
    simulation = run_simulation(api_url, json.dumps(payload, sort_keys=True))

    selected_path = simulation.get("selected_path")
    baseline_path = simulation.get("baseline_path")
    paths = simulation.get("paths", [])
    nodes = pd.DataFrame(simulation.get("nodes", []))

    map_shell = st.container()
    with map_shell:
        st.markdown('<div class="map-shell">', unsafe_allow_html=True)
        render_panel_header(
            "Simulation map",
            "Nodes show environmental conditions; arcs show the candidate routing options for the selected workload hour.",
        )
        compare_mode = state.get("compare_mode", "Candidates")
        comparison_mode = "baseline" if compare_mode == "Baseline vs Recommended" else "candidates"
        highlighted_path = selected_path

        deck = build_network_deck(
            nodes,
            paths=paths,
            selected_path_type=(highlighted_path or {}).get("type"),
            comparison_mode=comparison_mode,
            selected_path=highlighted_path,
            baseline_path=baseline_path,
        )
        if deck is None:
            st.info("No environmental node data is available for this simulation hour.")
        else:
            st.pydeck_chart(deck, use_container_width=True)
        st.caption(
            "Node color shows CO2 intensity from green to red. Node size reflects WUE. "
            "The soft halo highlights water scarcity. Candidate paths are color coded by objective."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([0.95, 1.35], gap="large")

    with bottom_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_panel_header(
            "Control panel",
            f"Incoming workload source: {context['origin_city']}. Adjust weights and rerun the routing simulation.",
        )
        with st.form("simulation_controls", clear_on_submit=False):
            priority = st.selectbox("Priority", ["low", "medium", "high"], index=["low", "medium", "high"].index(state["priority"]))
            selected_date = st.date_input(
                "Simulation date",
                value=state["selected_date"],
                min_value=time_min.date(),
                max_value=time_max.date(),
            )
            selected_hour = st.slider("Simulation hour", 0, 23, int(state["selected_hour"]), 1)
            alpha = st.slider("Carbon weight", 0.0, 2.0, float(state["alpha"]), 0.05)
            beta = st.slider("Water weight", 0.0, 2.0, float(state["beta"]), 0.05)
            gamma = st.slider("Latency weight", 0.0, 2.0, float(state["gamma"]), 0.05)
            compare_mode = st.segmented_control(
                "Map mode",
                options=["Candidates", "Baseline vs Recommended"],
                selection_mode="single",
                default=state.get("compare_mode", "Candidates"),
            )
            submitted = st.form_submit_button("Run simulation", use_container_width=True)

        if submitted:
            selected_time = _clamp_timestamp(selected_date, selected_hour, time_min, time_max)
            st.session_state["simulation_state"] = {
                "priority": priority,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "selected_date": selected_time.date(),
                "selected_hour": int(selected_time.hour),
                "time": selected_time,
                "compare_mode": compare_mode or "Candidates",
            }
            st.rerun()

        render_route_chip(
            f"Current simulation hour: {pd.Timestamp(state['time']):%Y-%m-%d %H:%M}"
        )
        st.caption(
            "The Run simulation action sends the current inputs to the FastAPI `/simulate` endpoint and redraws the map with the returned candidate routes."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_panel_header(
            "Result and insight",
            "The recommended route is compared against a naive baseline that prefers the local or lowest-latency path.",
        )

        if not paths or selected_path is None:
            st.warning("No feasible route was produced for this simulation hour.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        metrics = simulation.get("metrics", {})

        render_route_chip(f"Recommended route: {_best_path_label(selected_path)}")
        render_kpi_cards(
            [
                {
                    "label": "CO2 reduction",
                    "value": f"{float(metrics.get('co2_reduction_pct', 0.0)):+.1f}%",
                    "help": "Compared with the baseline routing choice.",
                },
                {
                    "label": "Water reduction",
                    "value": f"{float(metrics.get('water_reduction_pct', 0.0)):+.1f}%",
                    "help": "Lower water impact is better.",
                },
                {
                    "label": "Latency impact",
                    "value": f"{float(metrics.get('latency_delta_ms', 0.0)):+.1f} ms",
                    "help": "Positive means slower than baseline.",
                },
            ]
        )

        insight = simulation.get("insight", {})
        render_explanation_card(
            str(insight.get("title", "Recommendation")),
            str(insight.get("summary", "No explanation is available.")),
        )

        st.markdown("**Top candidate routes**")
        st.dataframe(
            _comparison_table(paths),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
