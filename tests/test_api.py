"""
API-level tests using FastAPI's TestClient.

TestClient wraps the ASGI app and makes real HTTP requests without starting
a server process. It's synchronous, fast, and great for CI.

Why test the API layer separately from the scheduler?
  The API converts HTTP request shapes into scheduler inputs and serializes
  the output. Bugs can exist here even when the scheduler itself is correct
  (e.g., missing fields, wrong status codes, validation rejects valid input).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_backend

# Clear lru_cache before each test so state doesn't leak between tests.
# lru_cache is module-level, so without clearing it the same backend
# instance is reused across the entire test session.
@pytest.fixture(autouse=True)
def clear_backend_cache():
    get_backend.cache_clear()
    yield
    get_backend.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data == {"status": "ok"}


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestContextEndpoint:
    def test_returns_200(self, client):
        response = client.get("/context")
        assert response.status_code == 200

    def test_required_fields_present(self, client):
        data = client.get("/context").json()
        for field in [
            "origin_city", "time_min", "time_max", "default_time",
            "default_workload_size", "default_batch_size",
            "batch_size_min", "batch_size_max",
        ]:
            assert field in data, f"Missing field: {field}"

    def test_batch_size_bounds_valid(self, client):
        data = client.get("/context").json()
        assert data["batch_size_min"] <= data["default_batch_size"] <= data["batch_size_max"]

    def test_time_range_valid(self, client):
        data = client.get("/context").json()
        assert data["time_min"] < data["time_max"]


# ---------------------------------------------------------------------------
# /simulate — validation guard
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestSimulateValidation:
    def test_invalid_priority_rejected(self, client):
        context = client.get("/context").json()
        response = client.post("/simulate", json={
            "priority": "urgent",       # not in low/medium/high
            "time": context["default_time"],
            "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
        })
        assert response.status_code == 422

    def test_alpha_out_of_range_rejected(self, client):
        context = client.get("/context").json()
        response = client.post("/simulate", json={
            "priority": "medium",
            "time": context["default_time"],
            "alpha": 99.0,   # > 5.0 max
            "beta": 1.0, "gamma": 1.0,
        })
        assert response.status_code == 422

    def test_valid_request_returns_200(self, client):
        context = client.get("/context").json()
        response = client.post("/simulate", json={
            "priority": "medium",
            "time": context["default_time"],
            "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
        })
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        context = client.get("/context").json()
        data = client.post("/simulate", json={
            "priority": "medium",
            "time": context["default_time"],
            "alpha": 1.0, "beta": 1.0, "gamma": 1.0,
        }).json()
        for field in ["time", "origin_city", "paths", "nodes", "metrics", "insight"]:
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# /explain — validation
# ---------------------------------------------------------------------------

@pytest.mark.api
class TestExplainEndpoint:
    def test_invalid_target_rejected(self, client):
        response = client.get("/explain", params={
            "city": "Dallas",
            "target": "energy",   # invalid
            "time": "2024-06-01T12:00",
        })
        assert response.status_code == 422

    def test_valid_co2_request_structure(self, client):
        """If models exist, response must have the right shape."""
        response = client.get("/explain", params={
            "city": "Dallas",
            "target": "co2",
            "time": "2024-06-01T12:00",
        })
        # 200 = models loaded; 503 = model file missing (acceptable in CI)
        assert response.status_code in (200, 503)
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "base_value" in data
            assert "features" in data
            assert isinstance(data["features"], list)

    def test_explain_features_have_shap_values(self, client):
        response = client.get("/explain", params={
            "city": "Dallas",
            "target": "co2",
            "time": "2024-06-01T12:00",
        })
        if response.status_code == 200:
            features = response.json()["features"]
            assert len(features) > 0
            for f in features:
                assert "name" in f
                assert "shap_value" in f
                assert isinstance(f["shap_value"], float)
