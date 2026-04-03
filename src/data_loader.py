from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "base_data_with_metrics.parquet",
    PROJECT_ROOT / "data" / "base_data_with_metrics.csv",
    PROJECT_ROOT / "data" / "sample_dataset.xlsx",
]
ZIP_MEMBER_CANDIDATES = [
    "base_data_with_metrics.parquet",
    "base_data_with_metrics.csv",
]


def _find_default_data_path() -> Path:
    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return DEFAULT_DATA_CANDIDATES[-1]


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Resolve a dataset path against the project root."""
    if path is None:
        return _find_default_data_path()

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _normalize_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["TIMESTAMP", "Timestamp", "timestamp"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _read_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer, low_memory=False)
    return _normalize_timestamp_columns(df)


def _read_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return _normalize_timestamp_columns(df)


def _read_parquet(path_or_buffer) -> pd.DataFrame:
    df = pd.read_parquet(path_or_buffer)
    return _normalize_timestamp_columns(df)


def _read_regular_file(data_path: Path) -> pd.DataFrame:
    suffix = data_path.suffix.lower()
    if suffix == ".xlsx":
        return _read_excel(data_path)
    if suffix == ".csv":
        return _read_csv(data_path)
    if suffix == ".parquet":
        return _read_parquet(data_path)

    raise ValueError(
        f"Unsupported dataset format: {data_path.suffix}. "
        "Supported formats are .xlsx, .csv, .parquet, and .zip."
    )


def _select_zip_member(zf: zipfile.ZipFile) -> str:
    members = [
        name
        for name in zf.namelist()
        if not name.endswith("/") and not name.startswith("__MACOSX/")
    ]

    for candidate in ZIP_MEMBER_CANDIDATES:
        if candidate in members:
            return candidate

    for member in members:
        if member.lower().endswith(".parquet"):
            return member

    for member in members:
        if member.lower().endswith(".csv"):
            return member

    raise FileNotFoundError(
        "No supported dataset file found inside ZIP archive. "
        "Expected a .parquet or .csv member."
    )


def _read_from_zip(data_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(data_path) as zf:
        member = _select_zip_member(zf)
        if member.lower().endswith(".parquet"):
            buffer = BytesIO(zf.read(member))
            return _read_parquet(buffer)
        with zf.open(member) as fh:
            return _read_csv(fh)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset from xlsx, csv, parquet, or zip."""
    data_path = resolve_data_path(path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place your dataset in data/ or pass --data-path."
        )

    if data_path.is_file() and data_path.stat().st_size == 0:
        raise ValueError(
            f"Dataset file exists but is empty at {data_path}. "
            "Replace it with a valid dataset file."
        )

    try:
        if data_path.suffix.lower() == ".zip":
            return _read_from_zip(data_path)
        return _read_regular_file(data_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset at {data_path}: {exc}") from exc
