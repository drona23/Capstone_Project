from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk


def _scale_series(series: pd.Series, minimum: float, maximum: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    if values.dropna().empty:
        return pd.Series([minimum] * len(values), index=values.index)

    low = values.min()
    high = values.max()
    if pd.isna(low) or pd.isna(high) or low == high:
        return pd.Series([minimum + (maximum - minimum) / 2.0] * len(values), index=values.index)

    scaled = (values - low) / (high - low)
    return minimum + scaled * (maximum - minimum)


def _co2_color(value: float, low: float, high: float) -> list[int]:
    if pd.isna(value) or low == high:
        return [80, 140, 220, 180]
    normalized = float((value - low) / (high - low))
    normalized = min(max(normalized, 0.0), 1.0)
    red = int(70 + normalized * 170)
    green = int(170 - normalized * 110)
    blue = int(210 - normalized * 140)
    return [red, green, blue, 190]


def build_overview_map(city_metrics: pd.DataFrame) -> pdk.Deck | None:
    if city_metrics.empty:
        return None

    map_df = city_metrics.copy()
    low = float(map_df["avg_hourly_co2_kg"].min())
    high = float(map_df["avg_hourly_co2_kg"].max())
    map_df["fill_color"] = map_df["avg_hourly_co2_kg"].apply(lambda value: _co2_color(value, low, high))
    map_df["radius"] = _scale_series(map_df["avg_wue"], 22000, 52000)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_fill_color="fill_color",
        get_radius="radius",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 120],
        line_width_min_pixels=1,
    )

    view_state = pdk.ViewState(
        latitude=float(map_df["latitude"].mean()),
        longitude=float(map_df["longitude"].mean()),
        zoom=3.45,
        pitch=0,
    )
    tooltip = {
        "html": (
            "<b>{city}</b><br/>"
            "Avg hourly CO2: {avg_hourly_co2_kg} kg<br/>"
            "Avg WUE: {avg_wue}<br/>"
            "Avg carbon intensity: {avg_carbon_intensity}"
        ),
        "style": {"backgroundColor": "#09101d", "color": "white"},
    }
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")


def build_routing_map(routes_df: pd.DataFrame) -> pdk.Deck | None:
    if routes_df.empty:
        return None

    route_df = routes_df.copy().sort_values("rank")
    rank_scale = _scale_series(route_df["rank"].rsub(route_df["rank"].max() + 1), 90, 240)
    route_df["arc_color"] = [
        [40, int(value), 150, 210] for value in rank_scale.round().astype(int).tolist()
    ]
    route_df["line_width"] = _scale_series(route_df["rank"].rsub(route_df["rank"].max() + 1), 2, 8)

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=route_df,
        get_source_position="[origin_longitude, origin_latitude]",
        get_target_position="[destination_longitude, destination_latitude]",
        get_source_color=[32, 46, 66, 160],
        get_target_color="arc_color",
        get_width="line_width",
        pickable=True,
        auto_highlight=True,
    )

    origin_nodes = route_df[["origin_city", "origin_latitude", "origin_longitude"]].drop_duplicates().rename(
        columns={
            "origin_city": "city",
            "origin_latitude": "latitude",
            "origin_longitude": "longitude",
        }
    )
    origin_nodes["node_color"] = [[26, 54, 93, 230]] * len(origin_nodes)
    origin_nodes["radius"] = 32000

    destination_nodes = route_df[["assigned_city", "destination_latitude", "destination_longitude"]].drop_duplicates().rename(
        columns={
            "assigned_city": "city",
            "destination_latitude": "latitude",
            "destination_longitude": "longitude",
        }
    )
    destination_nodes["node_color"] = [[46, 164, 126, 220]] * len(destination_nodes)
    destination_nodes["radius"] = 26000

    node_df = pd.concat([origin_nodes, destination_nodes], ignore_index=True)
    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=node_df,
        get_position="[longitude, latitude]",
        get_fill_color="node_color",
        get_radius="radius",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 130],
        line_width_min_pixels=1,
    )

    all_lats = pd.concat([route_df["origin_latitude"], route_df["destination_latitude"]], ignore_index=True)
    all_lons = pd.concat([route_df["origin_longitude"], route_df["destination_longitude"]], ignore_index=True)
    tooltip = {
        "html": (
            "<b>Rank {rank}: {origin_city} -> {assigned_city}</b><br/>"
            "Start: {scheduled_start}<br/>"
            "CO2: {expected_co2_kg} kg<br/>"
            "Water: {expected_water_liters} L"
        ),
        "style": {"backgroundColor": "#09101d", "color": "white"},
    }
    return pdk.Deck(
        layers=[arc_layer, node_layer],
        initial_view_state=pdk.ViewState(
            latitude=float(all_lats.mean()),
            longitude=float(all_lons.mean()),
            zoom=3.7,
            pitch=20,
        ),
        tooltip=tooltip,
        map_style="light",
    )
