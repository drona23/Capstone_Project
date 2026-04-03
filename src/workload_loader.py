from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_loader import PROJECT_ROOT
except ImportError:
    from data_loader import PROJECT_ROOT


DEFAULT_AZURE_TRACE_PATH = PROJECT_ROOT / "data" / "external" / "azure_packing" / "packing_trace_zone_a_v1.sqlite"
DEFAULT_CITY_POOL = [
    "Ashburn",
    "Atlanta",
    "Austin",
    "Chicago",
    "Dallas",
    "Phoenix",
    "Portland",
    "Seattle",
]


def resolve_workload_trace_path(path: str | Path | None = None) -> Path:
    """Resolve a workload trace path against the project root."""
    if path is None:
        return DEFAULT_AZURE_TRACE_PATH

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_azure_vm_trace(
    path: str | Path | None = None,
    limit: int | None = None,
    only_completed: bool = True,
) -> pd.DataFrame:
    """Load VM requests joined with VM type resource definitions."""
    trace_path = resolve_workload_trace_path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Azure workload trace not found at {trace_path}")

    where_clause = "where vm.endtime is not null" if only_completed else ""
    limit_clause = f"limit {int(limit)}" if limit is not None else ""
    query = f"""
        select
            vm.vmId,
            vm.tenantId,
            vm.vmTypeId,
            vm.priority,
            vm.starttime,
            vm.endtime,
            vmType.machineId,
            vmType.core,
            vmType.memory,
            vmType.hdd,
            vmType.ssd,
            vmType.nic
        from vm
        left join (
            select
                vmTypeId,
                min(machineId) as machineId,
                avg(core) as core,
                avg(memory) as memory,
                avg(hdd) as hdd,
                avg(ssd) as ssd,
                avg(nic) as nic
            from vmType
            group by vmTypeId
        ) as vmType
            on vm.vmTypeId = vmType.vmTypeId
        {where_clause}
        order by vm.vmId
        {limit_clause}
    """

    with sqlite3.connect(trace_path) as conn:
        return pd.read_sql_query(query, conn)


def _map_city_from_tenant(series: pd.Series, city_pool: list[str]) -> pd.Series:
    """Deterministically assign a city to each tenant as a scheduler proxy."""
    tenant_numeric = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return tenant_numeric.map(lambda value: city_pool[value % len(city_pool)])


def build_jobs_from_azure_trace(
    trace_df: pd.DataFrame,
    start_datetime: str = "2025-01-01 00:00:00",
    city_pool: list[str] | None = None,
    power_scale: float = 100.0,
    slack_hours: int = 6,
) -> pd.DataFrame:
    """Convert Azure VM trace rows into a scheduler-friendly jobs table."""
    if city_pool is None:
        city_pool = DEFAULT_CITY_POOL

    jobs = trace_df.copy()
    jobs["job_id"] = jobs["vmId"].map(lambda value: f"azure_vm_{value}")

    # Azure start/end times are fractional days relative to trace collection.
    start_reference = jobs["starttime"].min()
    normalized_start_days = jobs["starttime"] - start_reference
    normalized_end_days = jobs["endtime"] - start_reference
    anchor = pd.Timestamp(start_datetime)
    jobs["earliest_start"] = anchor + pd.to_timedelta(normalized_start_days, unit="D")
    jobs["end_timestamp"] = anchor + pd.to_timedelta(normalized_end_days, unit="D")

    duration_hours = (
        (jobs["end_timestamp"] - jobs["earliest_start"]).dt.total_seconds() / 3600.0
    )
    jobs["duration_hours"] = np.ceil(duration_hours.clip(lower=1.0)).astype(int)

    jobs["origin_city"] = _map_city_from_tenant(jobs["tenantId"], city_pool)

    # Power demand is not provided directly, so we proxy it with normalized CPU.
    jobs["power_demand"] = (jobs["core"].fillna(0.05) * power_scale).round(2)

    jobs["deadline"] = jobs["earliest_start"] + pd.to_timedelta(
        jobs["duration_hours"] + slack_hours, unit="h"
    )
    jobs["priority_label"] = jobs["priority"].map({0: "high", 1: "low"}).fillna("unknown")

    selected_columns = [
        "job_id",
        "origin_city",
        "power_demand",
        "duration_hours",
        "earliest_start",
        "deadline",
        "priority",
        "priority_label",
        "tenantId",
        "vmTypeId",
        "machineId",
        "core",
        "memory",
        "ssd",
        "nic",
    ]
    return jobs[selected_columns]


def export_jobs_from_azure_trace(
    input_path: str | Path | None,
    output_path: str | Path,
    limit: int | None = None,
) -> Path:
    """Load Azure trace rows, convert them to jobs, and save a CSV."""
    trace_df = load_azure_vm_trace(input_path, limit=limit)
    jobs_df = build_jobs_from_azure_trace(trace_df)

    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    jobs_df.to_csv(destination, index=False)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Azure packing trace into a scheduler-ready jobs CSV."
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_AZURE_TRACE_PATH),
        help="Path to the Azure packing trace SQLite file.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/azure_jobs_sample.csv",
        help="Path to write the generated jobs CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Optional number of completed VM rows to convert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_jobs_from_azure_trace(
        input_path=args.input_path,
        output_path=args.output_path,
        limit=args.limit,
    )
    print(f"Wrote jobs dataset to {output_path}")


if __name__ == "__main__":
    main()
