"""Small, explicit table loader used by optional data adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet table without developer-specific fallbacks."""
    source = resolve_project_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if source.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {source}")
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, low_memory=False)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    raise ValueError("Supported table formats are .csv and .parquet.")
