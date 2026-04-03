from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_backend import (
    PATH_TYPE_LABELS,
    SchedulerInputs,
    SustainabilitySchedulingBackend,
    run_scheduler,
)
from utils.visualization import (
    PATH_TYPE_LABELS as UI_PATH_LABELS,
    build_network_deck,
    render_explanation_card,
    render_kpi_cards,
)

BACKEND_HASH = {SustainabilitySchedulingBackend: lambda backend: backend.cache_key}


@st.cache_data(show_spinner=False, hash_funcs=BACKEND_HASH)
def load_scheduler_paths(
    backend: SustainabilitySchedulingBackend,
    origin_city: str,
    priority: str,
    latency_sensitivity: float,
    workload_size: float,
    alpha: float,
    beta: float,
    gamma: float,
    start_time_iso: str,
) -> list[dict[str, object]]:
    inputs = SchedulerInputs(
        origin_city=origin_city,
        priority=priority,
        latency_sensitivity=latency_sensitivity,
        workload_size=workload_size,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        start_time=pd.Timestamp(start_time_iso),
    )
    return run_scheduler(backend, inputs)


@st.cache_data(show_spinner=False, hash_funcs=BACKEND_HASH)
def load_scheduler_nodes(
    backend: SustainabilitySchedulingBackend,
    start_time_iso: str,
    end_time_iso: str,
) -> pd.DataFrame:
    return backend.build_scheduler_nodes(pd.Timestamp(start_time_iso), pd.Timestamp(end_time_iso))


def _safe_selection_index(selection_event, row_count: int) -> int:
    if row_count == 0:
        return 0
    if selection_event is None:
        return 0
    try:
        rows = selection_event.selection.rows
    except AttributeError:
        return 0
    if rows:
        return int(rows[0])
    return 0


def _format_path_table(paths: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(paths).copy()
    if frame.empty:
        return frame
    frame["Path"] = frame["type"].map(UI_PATH_LABELS)
    frame["Route"] = frame["path"].map(lambda path: " -> ".join(path))
    frame["CO2 (kg)"] = frame["co2"].map(lambda value: round(float(value), 2))
    frame["WUE (L/kWh)"] = frame["wue"].map(lambda value: round(float(value), 3))
    frame["Latency (ms)"] = frame["latency"].map(lambda value: round(float(value), 2))
    frame["Score"] = frame["score"].map(lambda value: round(float(value), 3))
    return frame[["Path", "Route", "CO2 (kg)", "WUE (L/kWh)", "Latency (ms)", "Score"]]


def render_scheduler_page(backend: SustainabilitySchedulingBackend) -> None:
    context = backend.context()
    st.title("Scheduler Simulator")
    st.markdown(
        '<div class="capstone-lead">Interactive multi-objective routing across carbon, water, and latency trade-offs.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Simulation reference hour derived from the workload trace: "
        f"{context['default_scheduler_timestamp']:%Y-%m-%d %H:%M}"
    )

    left_col, center_col, right_col = st.columns([0.95, 1.55, 1.05], gap="large")

    with left_col:
        st.subheader("Controls")
        with st.form("scheduler_controls"):
            origin_city = st.selectbox(
                "Origin city",
                context["origin_cities"],
                index=context["origin_cities"].index(context["default_origin_city"]),
            )
            priority = st.selectbox("Job priority", ["low", "medium", "high"], index=1)
            latency_sensitivity = st.slider("Latency sensitivity", 0.0, 1.0, 0.55, 0.05)
            workload_size = st.number_input(
                "Workload size",
                min_value=1.0,
                value=float(context["default_workload_size"]),
                step=10.0,
                help="Approximate power demand used to shape the simulated job.",
            )
            alpha = st.slider("Carbon weight (alpha)", 0.0, 2.0, 1.0, 0.05)
            beta = st.slider("Water weight (beta)", 0.0, 2.0, 1.0, 0.05)
            gamma = st.slider("Latency weight (gamma)", 0.0, 2.0, 1.0, 0.05)
            st.form_submit_button("Run Simulation", use_container_width=True)

    start_time = pd.Timestamp(context["default_scheduler_timestamp"])
    paths = load_scheduler_paths(
        backend,
        origin_city=origin_city,
        priority=priority,
        latency_sensitivity=latency_sensitivity,
        workload_size=float(workload_size),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        start_time_iso=start_time.isoformat(),
    )

    if not paths:
        with center_col:
            st.warning("No feasible routing paths were generated for the current simulation inputs.")
        with right_col:
            st.info("Try reducing the workload size or lowering latency sensitivity.")
        return

    selected_start = min(pd.Timestamp(path["scheduled_start"]) for path in paths)
    selected_end = max(pd.Timestamp(path["scheduled_end"]) for path in paths)
    nodes = load_scheduler_nodes(backend, selected_start.isoformat(), selected_end.isoformat())

    best_path = min(paths, key=lambda path: float(path["score"]))
    render_kpi_cards(
        [
            {
                "label": "Balanced route",
                "value": best_path["path"][-1],
                "help": f"Best combined score: {best_path['score']:.3f}.",
            },
            {
                "label": "Lowest CO2 option",
                "value": f"{min(path['co2'] for path in paths):,.1f} kg",
                "help": "Minimum carbon footprint across candidate routes.",
            },
            {
                "label": "Lowest WUE option",
                "value": f"{min(path['wue'] for path in paths):.3f} L/kWh",
                "help": "Best water-efficiency route among generated candidates.",
            },
            {
                "label": "Lowest latency option",
                "value": f"{min(path['latency'] for path in paths):.1f} ms",
                "help": "Fastest route before the latency weight is applied.",
            },
        ]
    )

    with right_col:
        st.subheader("Path comparison")
        table_frame = _format_path_table(paths)
        selection = st.dataframe(
            table_frame,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_index = _safe_selection_index(selection, len(paths))
        selected_path = paths[selected_index]
        explanation = backend.explain_path_choice(selected_path, paths)
        render_explanation_card(explanation["title"], explanation["summary"])
        tradeoff_df = pd.DataFrame(
            [
                {"Metric": "CO2 (kg)", "Selected": round(float(selected_path["co2"]), 2)},
                {"Metric": "WUE (L/kWh)", "Selected": round(float(selected_path["wue"]), 3)},
                {"Metric": "Latency (ms)", "Selected": round(float(selected_path["latency"]), 2)},
                {"Metric": "Score", "Selected": round(float(selected_path["score"]), 3)},
            ]
        )
        st.dataframe(tradeoff_df, hide_index=True, use_container_width=True)

    with center_col:
        st.subheader("Routing map")
        deck = build_network_deck(nodes, paths=paths, selected_path_type=selected_path["type"])
        if deck is None:
            st.info("No network map could be rendered for the current path set.")
        else:
            st.pydeck_chart(deck, use_container_width=True)
        st.caption(
            "Path colors: green = low carbon, blue = low water, yellow = balanced, red = low latency. "
            f"Selected path: {PATH_TYPE_LABELS.get(selected_path['type'], selected_path['type'])}."
        )
