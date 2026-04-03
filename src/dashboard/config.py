from __future__ import annotations

from pathlib import Path

try:
    from ..data_loader import PROJECT_ROOT
except ImportError:
    from data_loader import PROJECT_ROOT


DEFAULT_COORDINATES_PATH = PROJECT_ROOT / "data" / "reference" / "city_coordinates.csv"
DEFAULT_JOBS_PATH = PROJECT_ROOT / "data" / "processed" / "azure_jobs_sample.csv"
DEFAULT_DC_CONFIG_PATH = PROJECT_ROOT / "data" / "templates" / "dc_config_template.csv"
DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "base_data_with_metrics.parquet",
    PROJECT_ROOT / "data" / "base_data_with_metrics.csv",
    Path.home() / "Downloads" / "Archive 3.zip",
    PROJECT_ROOT / "data" / "sample_dataset.xlsx",
]


def first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return candidates[-1]


def default_data_path() -> Path:
    return first_existing_path(DEFAULT_DATA_CANDIDATES)


def resolve_app_path(path: str | Path | None, fallback: Path) -> Path:
    if path is None or str(path).strip() == "":
        return fallback

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
