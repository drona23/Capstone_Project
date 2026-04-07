from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .app_backend import (
    BatchSchedulerInputs,
    SchedulerInputs,
    SustainabilitySchedulingBackend,
    resolve_default_paths,
)
from .explain import compute_shap


class SimulationRequest(BaseModel):
    priority: str = Field(default="medium", pattern="^(low|medium|high)$")
    alpha: float = Field(default=1.0, ge=0.0, le=5.0)
    beta: float = Field(default=1.0, ge=0.0, le=5.0)
    gamma: float = Field(default=1.0, ge=0.0, le=5.0)
    time: str
    latency_sensitivity: float = Field(default=0.55, ge=0.0, le=1.0)
    workload_size: Optional[float] = Field(default=None, gt=0.0)
    origin_city: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=8)


class BatchSimulationRequest(BaseModel):
    priority: str = Field(default="medium", pattern="^(low|medium|high)$")
    alpha: float = Field(default=1.0, ge=0.0, le=5.0)
    beta: float = Field(default=1.0, ge=0.0, le=5.0)
    gamma: float = Field(default=1.0, ge=0.0, le=5.0)
    time: str
    latency_sensitivity: float = Field(default=0.55, ge=0.0, le=1.0)
    batch_size: int = Field(default=100, ge=50, le=200)


app = FastAPI(
    title="Sustainability-Aware Scheduling API",
    version="1.0.0",
    description="Simulation API for interactive workload routing demonstrations.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_backend() -> SustainabilitySchedulingBackend:
    defaults = resolve_default_paths()
    return SustainabilitySchedulingBackend(
        data_path=defaults["data_path"],
        jobs_path=defaults["jobs_path"],
        dc_config_path=defaults["dc_config_path"],
        coords_path=defaults["coords_path"],
    )


def _serialize_nodes(nodes: pd.DataFrame) -> list[dict[str, Any]]:
    if nodes.empty:
        return []

    frame = nodes.copy()
    for column in [
        "avg_co2_intensity",
        "avg_wue",
        "avg_scarcity",
        "records",
        "latitude",
        "longitude",
        "has_data_center",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.fillna("").to_dict(orient="records")


def _route_label(path: dict[str, Any]) -> str:
    return " -> ".join(path.get("path", []))


def _choose_selected_path(paths: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not paths:
        return None

    for path in paths:
        if str(path.get("type")) == "balanced":
            return path
    return min(paths, key=lambda item: float(item.get("score", 0.0)))


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/context")
def context() -> dict[str, Any]:
    backend = get_backend()
    app_context = backend.context()
    return {
        "origin_city": app_context["default_origin_city"],
        "time_min": pd.Timestamp(app_context["time_min"]).isoformat(),
        "time_max": pd.Timestamp(app_context["time_max"]).isoformat(),
        "default_time": pd.Timestamp(app_context["default_scheduler_timestamp"]).isoformat(),
        "default_workload_size": float(app_context["default_workload_size"]),
        "default_batch_size": int(app_context["default_batch_size"]),
        "batch_size_min": int(app_context["batch_size_min"]),
        "batch_size_max": int(app_context["batch_size_max"]),
        "priority_counts": app_context["priority_counts"],
    }


@app.post("/simulate")
def simulate(request: SimulationRequest) -> dict[str, Any]:
    backend = get_backend()
    app_context = backend.context()

    try:
        start_time = pd.Timestamp(request.time).floor("h")
    except Exception as exc:  # pragma: no cover - request validation guard.
        raise HTTPException(status_code=422, detail=f"Invalid simulation time: {exc}") from exc

    workload_size = (
        float(request.workload_size)
        if request.workload_size is not None
        else float(app_context["default_workload_size"])
    )
    origin_city = request.origin_city or str(app_context["default_origin_city"])

    paths = backend.run_scheduler(
        SchedulerInputs(
            origin_city=origin_city,
            priority=request.priority,
            latency_sensitivity=float(request.latency_sensitivity),
            workload_size=workload_size,
            alpha=float(request.alpha),
            beta=float(request.beta),
            gamma=float(request.gamma),
            start_time=start_time,
            top_k=int(request.top_k),
        )
    )

    if not paths:
        window_end = start_time + pd.Timedelta(hours=24)
        return {
            "time": start_time.isoformat(),
            "origin_city": origin_city,
            "paths": [],
            "selected_path": None,
            "baseline_path": None,
            "metrics": {
                "co2_reduction_pct": 0.0,
                "water_reduction_pct": 0.0,
                "latency_delta_ms": 0.0,
                "latency_delta_pct": 0.0,
            },
            "insight": backend.build_path_story({}, []),
            "nodes": _serialize_nodes(backend.build_scheduler_nodes(start_time, window_end)),
        }

    selected_path = _choose_selected_path(paths)
    baseline_path = backend.select_baseline_path(paths)
    metrics = backend.compare_paths(selected_path, baseline_path)
    insight = backend.build_path_story(selected_path, paths, baseline_path=baseline_path)

    window_start = min(pd.Timestamp(path["scheduled_start"]) for path in paths)
    window_end = max(pd.Timestamp(path["scheduled_end"]) for path in paths)
    nodes = backend.build_scheduler_nodes(window_start, window_end)

    enriched_paths = []
    for path in paths:
        enriched_path = dict(path)
        enriched_path["route"] = _route_label(path)
        enriched_paths.append(enriched_path)

    selected_payload = dict(selected_path)
    selected_payload["route"] = _route_label(selected_path)
    baseline_payload = dict(baseline_path) if baseline_path is not None else None
    if baseline_payload is not None:
        baseline_payload["route"] = _route_label(baseline_payload)

    return {
        "time": start_time.isoformat(),
        "origin_city": origin_city,
        "paths": enriched_paths,
        "selected_path": selected_payload,
        "baseline_path": baseline_payload,
        "metrics": metrics,
        "insight": insight,
        "nodes": _serialize_nodes(nodes),
    }


@app.post("/simulate-batch")
def simulate_batch(request: BatchSimulationRequest) -> dict[str, Any]:
    backend = get_backend()
    try:
        start_time = pd.Timestamp(request.time).floor("h")
    except Exception as exc:  # pragma: no cover - request validation guard.
        raise HTTPException(status_code=422, detail=f"Invalid simulation time: {exc}") from exc

    payload = backend.run_batch_scheduler(
        BatchSchedulerInputs(
            priority=request.priority,
            latency_sensitivity=float(request.latency_sensitivity),
            alpha=float(request.alpha),
            beta=float(request.beta),
            gamma=float(request.gamma),
            start_time=start_time,
            batch_size=int(request.batch_size),
        )
    )
    return payload


@app.get("/explain")
def explain(city: str, target: str, time: str) -> dict[str, Any]:
    """
    Return SHAP feature importances for a city's CO2 or WUE prediction.

    Query params:
        city   — data center city name
        target — 'co2' or 'wue'
        time   — ISO 8601 timestamp
    """
    if target not in ("co2", "wue"):
        raise HTTPException(status_code=422, detail="target must be 'co2' or 'wue'")
    try:
        return compute_shap(target=target, city=city, time=time)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc
