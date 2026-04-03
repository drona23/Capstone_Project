from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FUEL_COLUMNS = [
    "COAL",
    "HYDRO",
    "NATURALGAS",
    "NUCLEAR",
    "OTHER",
    "PETROLEUM",
    "SOLAR",
    "WIND",
]
WEATHER_CANDIDATES = {
    "temperature_c": ["temp_c", "weather_temp", "temp_c_from_water", "TEMPERATUREF"],
    "humidity_pct": ["rh_pct", "weather_rhum", "rh_pct_from_water", "HUMIDITY"],
    "wind_speed_ms": ["wspd_ms", "weather_wspd"],
}
BASE_FEATURE_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "scarcity_index",
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]
TARGET_COLUMNS = ["WUE_total", "carbon_intensity_kg_per_kwh"]


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class FutureFeatureProfiles:
    hourly_profiles: pd.DataFrame
    city_profiles: pd.DataFrame
    global_profile: dict[str, float]
    profile_columns: list[str]
    profile_history_days: int


def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["hour"] = enriched["timestamp"].dt.hour
    enriched["day_of_week"] = enriched["timestamp"].dt.dayofweek
    enriched["month"] = enriched["timestamp"].dt.month
    enriched["day_of_year"] = enriched["timestamp"].dt.dayofyear
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["hour"] / 24.0)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["hour"] / 24.0)
    enriched["day_of_week_sin"] = np.sin(2 * np.pi * enriched["day_of_week"] / 7.0)
    enriched["day_of_week_cos"] = np.cos(2 * np.pi * enriched["day_of_week"] / 7.0)
    enriched["month_sin"] = np.sin(2 * np.pi * enriched["month"] / 12.0)
    enriched["month_cos"] = np.cos(2 * np.pi * enriched["month"] / 12.0)
    return enriched


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def resolve_timestamp_column(df: pd.DataFrame) -> str:
    for candidate in ["TIMESTAMP", "Timestamp", "timestamp"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Dataset is missing a timestamp column.")


def resolve_city_column(df: pd.DataFrame) -> str:
    for candidate in ["CITY", "city"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Dataset is missing a city column.")


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _extract_weather_feature(df: pd.DataFrame, candidates: list[str], convert_fahrenheit: bool = False) -> pd.Series:
    for candidate in candidates:
        if candidate in df.columns:
            series = pd.to_numeric(df[candidate], errors="coerce")
            if convert_fahrenheit:
                return (series - 32.0) * (5.0 / 9.0)
            return series
    return pd.Series(np.nan, index=df.index, dtype=float)


def _generation_total_mwh(df: pd.DataFrame) -> pd.Series:
    if "total_gen_mwh" in df.columns:
        return pd.to_numeric(df["total_gen_mwh"], errors="coerce")
    if "total_gen_kwh" in df.columns:
        return pd.to_numeric(df["total_gen_kwh"], errors="coerce") / 1000.0

    available_fuels = [column for column in FUEL_COLUMNS if column in df.columns]
    if available_fuels:
        return df[available_fuels].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return pd.Series(np.nan, index=df.index, dtype=float)


def prepare_forecasting_frame(
    df: pd.DataFrame,
    city_limit: int | None = None,
) -> pd.DataFrame:
    """Build the canonical city-hour forecasting frame used by XGBoost and Prophet."""
    data = df.copy()
    timestamp_col = resolve_timestamp_column(data)
    city_col = resolve_city_column(data)

    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce").dt.floor("h")
    data[city_col] = data[city_col].astype(str).str.strip()

    numeric_candidates = [
        "WUE_total",
        "co2_kg",
        "total_gen_kwh",
        "total_gen_mwh",
        "dsci_0_500",
        *FUEL_COLUMNS,
        *sum(WEATHER_CANDIDATES.values(), []),
    ]
    _coerce_numeric_columns(data, numeric_candidates)

    required_targets = {"WUE_total", "co2_kg", "total_gen_kwh"}
    missing_targets = sorted(required_targets - set(data.columns))
    if missing_targets:
        raise ValueError(
            "Dataset is missing required target columns: " + ", ".join(missing_targets)
        )

    frame = pd.DataFrame(
        {
            "timestamp": data[timestamp_col],
            "city": data[city_col],
            "WUE_total": data["WUE_total"],
            "carbon_intensity_kg_per_kwh": _safe_ratio(data["co2_kg"], data["total_gen_kwh"]),
        }
    )

    frame["temperature_c"] = _extract_weather_feature(
        data,
        WEATHER_CANDIDATES["temperature_c"],
        convert_fahrenheit="TEMPERATUREF" in data.columns and "temp_c" not in data.columns,
    )
    frame["humidity_pct"] = _extract_weather_feature(data, WEATHER_CANDIDATES["humidity_pct"])
    frame["wind_speed_ms"] = _extract_weather_feature(data, WEATHER_CANDIDATES["wind_speed_ms"])

    scarcity = pd.to_numeric(data.get("dsci_0_500"), errors="coerce") if "dsci_0_500" in data.columns else pd.Series(np.nan, index=data.index)
    frame["scarcity_index"] = scarcity / 500.0

    generation_total = _generation_total_mwh(data)
    for fuel_column in FUEL_COLUMNS:
        if fuel_column in data.columns:
            frame[f"{fuel_column.lower()}_share"] = _safe_ratio(data[fuel_column], generation_total)
        else:
            frame[f"{fuel_column.lower()}_share"] = np.nan

    frame = _add_time_features(frame)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["timestamp", "city", *TARGET_COLUMNS]).sort_values(["timestamp", "city"])

    if city_limit is not None:
        selected_cities = sorted(frame["city"].dropna().unique().tolist())[: int(city_limit)]
        frame = frame[frame["city"].isin(selected_cities)].copy()

    return frame.reset_index(drop=True)


def feature_columns_for_frame(frame: pd.DataFrame) -> list[str]:
    fuel_share_columns = [f"{column.lower()}_share" for column in FUEL_COLUMNS]
    ordered = BASE_FEATURE_COLUMNS + fuel_share_columns
    return [column for column in ordered if column in frame.columns]


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    encoded_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create an aligned feature matrix with one-hot encoded cities."""
    if feature_columns is None:
        feature_columns = feature_columns_for_frame(frame)

    base = frame[["city", *feature_columns]].copy()
    encoded = pd.get_dummies(base, columns=["city"], prefix="city", dtype=float)

    if encoded_columns is None:
        encoded_columns = encoded.columns.tolist()
    else:
        missing = [column for column in encoded_columns if column not in encoded.columns]
        for column in missing:
            encoded[column] = 0.0
        extra = [column for column in encoded.columns if column not in encoded_columns]
        if extra:
            encoded = encoded.drop(columns=extra)
        encoded = encoded[encoded_columns]

    return encoded.astype(float), encoded_columns


def split_time_train_test(
    frame: pd.DataFrame,
    test_horizon_hours: int = 48,
    history_days: int = 365,
) -> TimeSplit:
    """Hold out the final horizon using a strict time-based split."""
    timestamps = pd.Index(sorted(frame["timestamp"].dropna().unique()))
    if len(timestamps) <= test_horizon_hours:
        raise ValueError("Not enough hourly observations for the requested test horizon.")

    test_timestamps = timestamps[-int(test_horizon_hours):]
    test_start = pd.Timestamp(test_timestamps[0])
    test_end = pd.Timestamp(test_timestamps[-1])
    history_start = test_start - pd.Timedelta(days=int(history_days))

    scoped = frame[(frame["timestamp"] >= history_start) & (frame["timestamp"] <= test_end)].copy()
    train = scoped[scoped["timestamp"] < test_start].copy()
    test = scoped[scoped["timestamp"] >= test_start].copy()

    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test partition.")

    return TimeSplit(train=train, test=test, test_start=test_start, test_end=test_end)


def select_recent_history(
    frame: pd.DataFrame,
    history_days: int = 365,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if end_time is None:
        end_time = pd.Timestamp(frame["timestamp"].max())
    history_start = pd.Timestamp(end_time) - pd.Timedelta(days=int(history_days))
    scoped = frame[(frame["timestamp"] >= history_start) & (frame["timestamp"] <= pd.Timestamp(end_time))].copy()
    if scoped.empty:
        raise ValueError("No history rows are available for the requested training window.")
    return scoped.sort_values(["timestamp", "city"]).reset_index(drop=True)


def build_future_feature_profiles(
    history_frame: pd.DataFrame,
    profile_history_days: int = 30,
) -> FutureFeatureProfiles:
    profile_columns = feature_columns_for_frame(history_frame)
    recent_start = pd.Timestamp(history_frame["timestamp"].max()) - pd.Timedelta(days=int(profile_history_days))
    recent = history_frame[history_frame["timestamp"] >= recent_start].copy()
    if recent.empty:
        recent = history_frame.copy()

    hourly_profiles = (
        recent.groupby(["city", "hour"], as_index=False)[profile_columns].median(numeric_only=True)
    )
    city_profiles = recent.groupby("city", as_index=False)[profile_columns].median(numeric_only=True)
    global_profile = recent[profile_columns].median(numeric_only=True).to_dict()
    return FutureFeatureProfiles(
        hourly_profiles=hourly_profiles,
        city_profiles=city_profiles,
        global_profile={key: float(value) for key, value in global_profile.items()},
        profile_columns=profile_columns,
        profile_history_days=int(profile_history_days),
    )


def build_profile_feature_frame(
    time_city_frame: pd.DataFrame,
    profiles: FutureFeatureProfiles,
) -> pd.DataFrame:
    """Populate exogenous features from historical profiles without using future observations."""
    if not {"timestamp", "city"}.issubset(time_city_frame.columns):
        raise ValueError("The profile feature frame requires `timestamp` and `city` columns.")

    future = time_city_frame[["timestamp", "city"]].copy()
    future["timestamp"] = pd.to_datetime(future["timestamp"], errors="coerce").dt.floor("h")
    future["city"] = future["city"].astype(str).str.strip()
    future = _add_time_features(future)

    future = future.merge(profiles.hourly_profiles, on=["city", "hour"], how="left")
    city_profiles = profiles.city_profiles.set_index("city")
    for column in profiles.profile_columns:
        if column not in future.columns:
            future[column] = np.nan
        future[column] = future[column].fillna(future["city"].map(city_profiles[column]))
        future[column] = future[column].fillna(profiles.global_profile.get(column, np.nan))

    return future.sort_values(["timestamp", "city"]).reset_index(drop=True)


def build_future_feature_frame(
    history_frame: pd.DataFrame,
    horizon_hours: int = 48,
    profiles: FutureFeatureProfiles | None = None,
) -> pd.DataFrame:
    """Project the next horizon of city-hour features from historical hourly profiles."""
    if profiles is None:
        profiles = build_future_feature_profiles(history_frame)

    last_timestamp = pd.Timestamp(history_frame["timestamp"].max()).floor("h")
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=int(horizon_hours),
        freq="h",
    )
    cities = sorted(history_frame["city"].dropna().unique().tolist())
    time_city_frame = pd.MultiIndex.from_product(
        [future_timestamps, cities],
        names=["timestamp", "city"],
    ).to_frame(index=False)
    return build_profile_feature_frame(time_city_frame, profiles=profiles)


def build_modeling_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Backward-compatible helper for legacy regression code paths."""
    frame = prepare_forecasting_frame(df)
    X, _ = build_feature_matrix(frame)
    return X, frame["WUE_total"], frame["carbon_intensity_kg_per_kwh"]
