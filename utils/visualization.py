from __future__ import annotations

from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

PATH_TYPE_COLORS = {
    "low_carbon": [44, 162, 95, 235],
    "low_water": [44, 127, 184, 235],
    "balanced": [221, 170, 32, 235],
    "low_latency": [203, 24, 29, 235],
    "baseline": [107, 114, 128, 190],
}
PATH_TYPE_LABELS = {
    "balanced": "Balanced",
    "low_carbon": "Low Carbon",
    "low_water": "Low Water",
    "low_latency": "Low Latency",
    "baseline": "Baseline",
}


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1rem;
            padding-bottom: 1.75rem;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(226, 239, 233, 0.90), transparent 24%),
                radial-gradient(circle at top right, rgba(225, 235, 245, 0.90), transparent 22%),
                linear-gradient(180deg, #f7faf8 0%, #eef4f3 100%);
        }
        .sim-header {
            display: flex;
            justify-content: space-between;
            align-items: end;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }
        .sim-title {
            color: #0f2a1e;
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 700;
            margin: 0;
        }
        .sim-lead {
            color: #4f645c;
            font-size: 1rem;
            margin-top: 0.35rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(31, 41, 55, 0.10);
            color: #224838;
            font-size: 0.85rem;
            white-space: nowrap;
        }
        .map-shell {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 26px;
            padding: 0.8rem 0.8rem 0.4rem 0.8rem;
            box-shadow: 0 16px 40px rgba(31, 41, 55, 0.07);
            margin-bottom: 1rem;
        }
        .panel-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 34px rgba(31, 41, 55, 0.06);
            height: 100%;
        }
        .panel-title {
            color: #10261f;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .panel-subtitle {
            color: #63756d;
            font-size: 0.9rem;
            margin-bottom: 0.9rem;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
            margin: 0.2rem 0 0.85rem 0;
        }
        .kpi-card {
            background: linear-gradient(180deg, rgba(246, 249, 248, 1) 0%, rgba(238, 244, 242, 1) 100%);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 18px;
            padding: 0.9rem 0.95rem;
        }
        .kpi-label {
            color: #5e6e67;
            font-size: 0.78rem;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            color: #10261f;
            font-size: 1.55rem;
            line-height: 1.05;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .kpi-help {
            color: #60736b;
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .story-card {
            background: linear-gradient(180deg, rgba(21, 78, 52, 0.95) 0%, rgba(16, 58, 42, 0.98) 100%);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            color: white;
            margin: 0.75rem 0 0.9rem 0;
        }
        .story-title {
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.82;
            margin-bottom: 0.45rem;
        }
        .story-copy {
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .route-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.38rem 0.65rem;
            background: rgba(242, 246, 244, 1);
            border: 1px solid rgba(31, 41, 55, 0.08);
            color: #123325;
            font-size: 0.84rem;
            margin-bottom: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, status_text: str) -> None:
    st.markdown(
        (
            '<div class="sim-header">'
            '<div>'
            f'<div class="sim-title">{title}</div>'
            f'<div class="sim-lead">{subtitle}</div>'
            '</div>'
            f'<div class="status-pill">{status_text}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f'<div class="panel-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="panel-title">{title}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def render_route_chip(text: str) -> None:
    st.markdown(f'<div class="route-chip">{text}</div>', unsafe_allow_html=True)


def _series_or_default(frame: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    for candidate in candidates:
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce")
    return pd.Series([default] * len(frame), index=frame.index, dtype=float)


def create_map_nodes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    nodes = df.copy()
    co2 = _series_or_default(nodes, ["avg_co2_intensity", "co2", "co2_intensity"])
    wue = _series_or_default(nodes, ["avg_wue", "wue"])
    scarcity = _series_or_default(nodes, ["avg_scarcity", "scarcity", "scarcity_index"])

    co2_min = float(co2.min()) if not co2.dropna().empty else 0.0
    co2_max = float(co2.max()) if not co2.dropna().empty else 1.0
    if co2_min == co2_max:
        co2_scaled = pd.Series([0.5] * len(nodes), index=nodes.index)
    else:
        co2_scaled = (co2 - co2_min) / (co2_max - co2_min)

    wue_min = float(wue.min()) if not wue.dropna().empty else 0.0
    wue_max = float(wue.max()) if not wue.dropna().empty else 1.0
    if wue_min == wue_max:
        wue_scaled = pd.Series([0.5] * len(nodes), index=nodes.index)
    else:
        wue_scaled = (wue - wue_min) / (wue_max - wue_min)

    nodes["fill_color"] = [
        [
            int(44 + (value * 190)),
            int(170 - (value * 115)),
            int(82 - (value * 48)),
            215,
        ]
        for value in co2_scaled.fillna(0.5)
    ]
    nodes["radius"] = [int(18000 + value * 42000) for value in wue_scaled.fillna(0.5)]
    nodes["scarcity_weight"] = scarcity.fillna(0.0).clip(lower=0.0)
    nodes["outline_color"] = [
        [19, 78, 55, 225] if int(value) == 1 else [255, 255, 255, 155]
        for value in pd.to_numeric(nodes.get("has_data_center", 0), errors="coerce").fillna(0)
    ]
    nodes["tooltip_title"] = nodes["city"].astype(str)
    nodes["tooltip_line1"] = co2.map(lambda value: f"CO2 intensity: {value:.3f} kg/kWh")
    nodes["tooltip_line2"] = wue.map(lambda value: f"WUE: {value:.3f} L/kWh")
    nodes["tooltip_line3"] = scarcity.map(lambda value: f"Scarcity index: {value:.3f}")
    return nodes


def _midpoint(value_a: float, value_b: float) -> float:
    return (float(value_a) + float(value_b)) / 2.0


def _candidate_paths_frame(
    paths: list[dict[str, Any]],
    selected_path_type: str | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(paths).copy()
    if frame.empty:
        return frame

    frame["path_color"] = frame["type"].map(PATH_TYPE_COLORS)
    frame["line_width"] = frame["type"].map(
        lambda path_type: 10 if path_type == selected_path_type else 5
    )
    frame["text_color"] = frame["path_color"]
    frame["marker_text"] = ">"
    frame["mid_latitude"] = frame.apply(
        lambda row: _midpoint(row["origin_latitude"], row["destination_latitude"]),
        axis=1,
    )
    frame["mid_longitude"] = frame.apply(
        lambda row: _midpoint(row["origin_longitude"], row["destination_longitude"]),
        axis=1,
    )
    frame["tooltip_title"] = frame["route"]
    frame["tooltip_line1"] = frame["type"].map(
        lambda value: f"Path type: {PATH_TYPE_LABELS.get(str(value), str(value))}"
    )
    frame["tooltip_line2"] = frame.apply(
        lambda row: f"CO2: {float(row['co2']):.2f} kg | WUE: {float(row['wue']):.3f} L/kWh",
        axis=1,
    )
    frame["tooltip_line3"] = frame["latency"].map(
        lambda value: f"Latency: {float(value):.1f} ms"
    )
    return frame


def _compare_paths_frame(
    selected_path: dict[str, Any] | None,
    baseline_path: dict[str, Any] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if baseline_path is not None:
        row = dict(baseline_path)
        row["type"] = "baseline"
        row["route"] = row.get("route", " -> ".join(row.get("path", [])))
        rows.append(row)
    if selected_path is not None:
        row = dict(selected_path)
        row["route"] = row.get("route", " -> ".join(row.get("path", [])))
        rows.append(row)
    return _candidate_paths_frame(rows, selected_path_type=selected_path.get("type") if selected_path else None)


def create_path_layers(
    paths: list[dict[str, Any]],
    selected_path_type: str | None = None,
    comparison_mode: str = "candidates",
    selected_path: dict[str, Any] | None = None,
    baseline_path: dict[str, Any] | None = None,
) -> list[pdk.Layer]:
    if comparison_mode == "baseline":
        path_frame = _compare_paths_frame(selected_path, baseline_path)
    else:
        path_frame = _candidate_paths_frame(paths, selected_path_type=selected_path_type)

    if path_frame.empty:
        return []

    return [
        pdk.Layer(
            "ArcLayer",
            data=path_frame,
            get_source_position="[origin_longitude, origin_latitude]",
            get_target_position="[destination_longitude, destination_latitude]",
            get_source_color="path_color",
            get_target_color="path_color",
            get_width="line_width",
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=path_frame,
            get_position="[mid_longitude, mid_latitude]",
            get_text="marker_text",
            get_color="text_color",
            get_size=20,
            get_angle=0,
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            pickable=False,
        ),
    ]


def build_network_deck(
    nodes_df: pd.DataFrame,
    paths: list[dict[str, Any]] | None = None,
    selected_path_type: str | None = None,
    comparison_mode: str = "candidates",
    selected_path: dict[str, Any] | None = None,
    baseline_path: dict[str, Any] | None = None,
) -> pdk.Deck | None:
    nodes = create_map_nodes(nodes_df)
    if nodes.empty:
        return None

    layers: list[pdk.Layer] = [
        pdk.Layer(
            "HeatmapLayer",
            data=nodes,
            get_position="[longitude, latitude]",
            get_weight="scarcity_weight",
            opacity=0.35,
            threshold=0.12,
            radius_pixels=70,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=nodes,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_line_color="outline_color",
            get_radius="radius",
            line_width_min_pixels=1.4,
            stroked=True,
            pickable=True,
        ),
    ]
    layers.extend(
        create_path_layers(
            paths or [],
            selected_path_type=selected_path_type,
            comparison_mode=comparison_mode,
            selected_path=selected_path,
            baseline_path=baseline_path,
        )
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=float(nodes["latitude"].mean()),
            longitude=float(nodes["longitude"].mean()),
            zoom=3.55,
            pitch=28 if paths else 8,
        ),
        tooltip={
            "html": (
                "<b>{tooltip_title}</b><br/>"
                "{tooltip_line1}<br/>"
                "{tooltip_line2}<br/>"
                "{tooltip_line3}"
            ),
            "style": {"backgroundColor": "#10261f", "color": "white", "fontSize": "12px"},
        },
        map_style="light",
        height=560,
    )


def render_kpi_cards(metrics: list[dict[str, str]]) -> None:
    cards = []
    for metric in metrics:
        cards.append(
            (
                '<div class="kpi-card">'
                f'<div class="kpi-label">{metric["label"]}</div>'
                f'<div class="kpi-value">{metric["value"]}</div>'
                f'<div class="kpi-help">{metric["help"]}</div>'
                "</div>"
            )
        )
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_explanation_card(title: str, summary: str) -> None:
    st.markdown(
        (
            '<div class="story-card">'
            f'<div class="story-title">{title}</div>'
            f'<div class="story-copy">{summary}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
