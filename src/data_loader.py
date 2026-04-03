from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_dataset.xlsx"


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Resolve a dataset path against the project root."""
    if path is None:
        return DEFAULT_DATA_PATH

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset from Excel and return a DataFrame."""
    data_path = resolve_data_path(path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place your dataset at data/sample_dataset.xlsx or pass --data-path."
        )

    if data_path.stat().st_size == 0:
        raise ValueError(
            f"Dataset file exists but is empty at {data_path}. "
            "Replace it with a valid .xlsx file."
        )

    try:
        return pd.read_excel(data_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset at {data_path}: {exc}") from exc

