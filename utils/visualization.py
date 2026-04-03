from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

PATH_TYPE_COLORS = {
    "low_carbon": [49, 163, 84, 220],
    "low_water": [49, 130, 189, 220],
    "balanced": [221, 170, 32, 220],
    "low_latency": [203, 24, 29, 220],
}
PATH_TYPE_LABELS = {
    "balanced": "Balanced",
    "low_carbon": "Low Carbon",
    "low_water": "Low Water",
    "low_latency": "Low Latency",
}


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
            max-width: 1500px;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(229, 241, 249, 0.7), transparent 24%),
                        linear-gradient(180deg, #f5f8fb 0%, #eef3f7 100%);
        }
        .capstone-lead {
            color: #496070;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.9rem;
            margin: 0.4rem 0 1rem 0;
        }
        .kpi-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(44, 76, 109, 0.12);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 30px rgba(44, 76, 109, 0.08);
        }
        .kpi-label {
            color: #567086;
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            color: #0f2740;
            font-size: 1.7rem;
            line-height: 1.1;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .kpi-help {
            color: #596d7d;
            font-size: 0.88rem;
            line-height: 1.35;
        }
        .explanation-card {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(44, 76, 109, 0.12);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-top: 0.85rem;
        }
        .panel-card {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid rgba(44, 76, 109, 0.10);
            border-radius: 18px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_map_nodes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    nodes = df.copy()
    co2 = pd.to_numeric(nodes["avg_co2_intensity"], errors="coerce")
    wue = pd.to_numeric(nodes["avg_wue"], errors="coerce")

    co2_min = float(co2.min()) if not co2.dropna().empty else 0.0
    co2_max = float(co2.max()) if not co2.dropna().empty else 1.0
    if co2_min == co2_max:
        normalized_co2 = pd.Series([0.5] * len(nodes), index=nodes.index)
    else:
        normalized_co2 = (co2 - co2_min) / (co2_max - co2_min)

    wue_min = float(wue.min()) if not wue.dropna().empty else 0.0
    wue_max = float(wue.max()) if not wue.dropna().empty else 1.0
    if wue_min == wue_max:
        normalized_wue = pd.Series([0.5] * len(nodes), index=nodes.index)
    else:
        normalized_wue = (wue - wue_min) / (wue_max - wue_min)

    nodes["fill_color"] = [
        [
            int(40 + value * 195),
            int(168 - value * 120),
            int(68 - value * 40),
            210,
        ]
        for value in normalized_co2.fillna(0.5)
    ]
    nodes["radius"] = [int(22000 + value * 42000) for value in normalized_wue.fillna(0.5)]
    nodes["outline_color"] = [
        [16, 68, 104, 210] if int(has_dc) == 1 else [255, 255, 255, 150]
        for has_dc in nodes.get("has_data_center", pd.Series([0] * len(nodes)))
    ]
    return nodes


def create_path_layers(paths: list[dict[str, object]], selected_path_type: str | None = None) -> list[pdk.Layer]:
    if not paths:
        return []

    paths_df = pd.DataFrame(paths).copy()
    paths_df["city"] = paths_df["path"].map(lambda path: " -> ".join(path))
    paths_df["path_color"] = paths_df["type"].map(PATH_TYPE_COLORS)
    paths_df["line_width"] = paths_df["type"].map(
        lambda path_type: 9 if selected_path_type == path_type else 4
    )
    paths_df["arc_color"] = paths_df.apply(
        lambda row: row["path_color"][:3] + [245 if row["type"] == selected_path_type or selected_path_type is None else 110],
        axis=1,
    )

    return [
        pdk.Layer(
            "ArcLayer",
            data=paths_df,
            get_source_position="[origin_longitude, origin_latitude]",
            get_target_position="[destination_longitude, destination_latitude]",
            get_source_color=[52, 73, 94, 170],
            get_target_color="arc_color",
            get_width="line_width",
            pickable=True,
            auto_highlight=True,
        )
    ]


def build_network_deck(
    nodes_df: pd.DataFrame,
    paths: list[dict[str, object]] | None = None,
    selected_path_type: str | None = None,
) -> pdk.Deck | None:
    nodes = create_map_nodes(nodes_df)
    if nodes.empty:
        return None

    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes,
        get_position="[longitude, latitude]",
        get_fill_color="fill_color",
        get_line_color="outline_color",
        line_width_min_pixels=1,
        stroked=True,
        get_radius="radius",
        pickable=True,
    )

    layers = [node_layer]
    layers.extend(create_path_layers(paths or [], selected_path_type=selected_path_type))

    tooltip_html = (
        "<b>{city}</b><br/>"
        "CO2 intensity: {avg_co2_intensity}<br/>"
        "WUE: {avg_wue}<br/>"
        "Scarcity: {avg_scarcity}"
    )
    if paths:
        tooltip_html = (
            "<b>{city}</b><br/>"
            "CO2 intensity: {avg_co2_intensity}<br/>"
            "WUE: {avg_wue}<br/>"
            "Scarcity: {avg_scarcity}<br/><hr style='margin:0.25rem 0;border:none;border-top:1px solid rgba(255,255,255,0.2);'/>"
            "Path type: {type}<br/>"
            "CO2: {co2}<br/>"
            "Latency: {latency}"
        )

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=float(nodes["latitude"].mean()),
            longitude=float(nodes["longitude"].mean()),
            zoom=3.7,
            pitch=18 if paths else 0,
        ),
        tooltip={
            "html": tooltip_html,
            "style": {"backgroundColor": "#0f1724", "color": "white"},
        },
        map_style="light",
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
                '</div>'
            )
        )
    st.markdown('<div class="kpi-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_explanation_card(title: str, summary: str) -> None:
    st.markdown(
        (
            '<div class="explanation-card">'
            f'<div class="kpi-label">{title}</div>'
            f'<div class="kpi-help">{summary}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
