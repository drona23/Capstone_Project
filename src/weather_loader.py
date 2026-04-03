from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .data_loader import PROJECT_ROOT
except ImportError:
    from data_loader import PROJECT_ROOT


DEFAULT_WEATHER_FILENAMES = [
    "meteostat_weather_master_2019_2023.csv",
    "meteostat_weather_master.csv",
]

WEATHER_VALUE_COLUMNS = [
    "temp",
    "dwpt",
    "rhum",
    "prcp",
    "snow",
    "wdir",
    "wspd",
    "wpgt",
    "pres",
    "tsun",
    "coco",
]

TIMESTAMP_COLUMN_CANDIDATES = ["TIMESTAMP", "Timestamp", "timestamp", "datetime"]
ZIP_COLUMN_CANDIDATES = ["ZIP", "zip", "Zip", "zip_code", "zipcode", "zip code"]
CITY_STATE_COLUMN_CANDIDATES = ["CityST", "city_state", "city_st", "CITYST"]
CITY_COLUMN_CANDIDATES = ["city", "City", "CITY"]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_weather_path(path: str | Path) -> Path:
    """Resolve a weather dataset path from a CSV file or directory."""
    candidate = _resolve_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Weather dataset path does not exist: {candidate}")

    if candidate.is_file():
        return candidate

    for filename in DEFAULT_WEATHER_FILENAMES:
        nested = candidate / filename
        if nested.exists():
            return nested

    raise FileNotFoundError(
        "Weather dataset directory does not contain a supported master CSV. "
        f"Expected one of: {', '.join(DEFAULT_WEATHER_FILENAMES)}"
    )


def _normalize_timestamp_hour(series: pd.Series) -> pd.Series:
    """Convert timestamps to local wall-clock hourly resolution."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"([+-]\d{2}:\d{2})$", "", regex=True)
        .str.replace("Z", "", regex=False)
    )
    return pd.to_datetime(cleaned, errors="coerce").dt.floor("h")


def _normalize_zip(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.extract(r"(\d+)")[0]
    return values.str.zfill(5)


def _normalize_city_state(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _normalize_city(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .str.lower()
    )


def _find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def load_weather_dataset(path: str | Path) -> pd.DataFrame:
    """Load the Meteostat weather dataset and compute prefixed features."""
    weather_path = resolve_weather_path(path)
    usecols = ["Timestamp", "LocationID", "LocationType", "CityST"] + WEATHER_VALUE_COLUMNS
    weather_df = pd.read_csv(weather_path, usecols=lambda col: col in usecols, low_memory=False)

    weather_df["weather_timestamp_hour"] = _normalize_timestamp_hour(weather_df["Timestamp"])
    weather_df["weather_zip"] = _normalize_zip(weather_df["LocationID"])
    weather_df["weather_city_state"] = _normalize_city_state(weather_df["CityST"])
    weather_df["weather_city"] = _normalize_city(weather_df["CityST"])

    for column in WEATHER_VALUE_COLUMNS:
        if column in weather_df.columns:
            weather_df[f"weather_{column}"] = pd.to_numeric(
                weather_df[column], errors="coerce"
            )

    weather_df["weather_temp_dewpoint_gap"] = (
        weather_df["weather_temp"] - weather_df["weather_dwpt"]
    )
    weather_df["weather_rain_flag"] = (
        weather_df["weather_prcp"].fillna(0).gt(0).astype(int)
    )
    weather_df["weather_humidity_temp_index"] = (
        weather_df["weather_temp"] * weather_df["weather_rhum"] / 100.0
    )

    feature_columns = [
        "weather_timestamp_hour",
        "weather_zip",
        "weather_city_state",
        "weather_city",
        "weather_temp",
        "weather_dwpt",
        "weather_rhum",
        "weather_prcp",
        "weather_wdir",
        "weather_wspd",
        "weather_pres",
        "weather_coco",
        "weather_temp_dewpoint_gap",
        "weather_rain_flag",
        "weather_humidity_temp_index",
    ]
    return weather_df[feature_columns]


def _prepare_main_merge_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    timestamp_column = _find_first_column(df, TIMESTAMP_COLUMN_CANDIDATES)
    if timestamp_column is None:
        raise ValueError(
            "Main dataset must contain a timestamp column such as TIMESTAMP or Timestamp."
        )

    prepared = df.copy()
    prepared["weather_timestamp_hour"] = _normalize_timestamp_hour(prepared[timestamp_column])

    zip_column = _find_first_column(prepared, ZIP_COLUMN_CANDIDATES)
    if zip_column is not None:
        prepared["weather_join_location"] = _normalize_zip(prepared[zip_column])
        return prepared, "zip"

    city_state_column = _find_first_column(prepared, CITY_STATE_COLUMN_CANDIDATES)
    if city_state_column is not None:
        prepared["weather_join_location"] = _normalize_city_state(
            prepared[city_state_column]
        )
        return prepared, "city_state"

    city_column = _find_first_column(prepared, CITY_COLUMN_CANDIDATES)
    if city_column is not None:
        prepared["weather_join_location"] = _normalize_city(prepared[city_column])
        return prepared, "city"

    raise ValueError(
        "Main dataset must contain a location column such as ZIP, CityST, or city to merge weather data."
    )


def merge_weather_features(
    df: pd.DataFrame, weather_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Merge weather features into the main dataset using timestamp and location."""
    prepared, join_strategy = _prepare_main_merge_keys(df)

    if join_strategy == "zip":
        weather_key = "weather_zip"
    elif join_strategy == "city_state":
        weather_key = "weather_city_state"
    else:
        weather_key = "weather_city"

    merged = prepared.merge(
        weather_df,
        how="left",
        left_on=["weather_timestamp_hour", "weather_join_location"],
        right_on=["weather_timestamp_hour", weather_key],
    )

    matched_rows = int(merged["weather_temp"].notna().sum())
    total_rows = int(len(merged))
    coverage = float(matched_rows / total_rows) if total_rows else 0.0

    merged.drop(
        columns=["weather_join_location", "weather_zip", "weather_city_state", "weather_city"],
        inplace=True,
        errors="ignore",
    )

    return merged, {
        "join_strategy": join_strategy,
        "matched_rows": matched_rows,
        "total_rows": total_rows,
        "coverage": coverage,
    }
