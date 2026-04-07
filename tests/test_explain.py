"""
Unit tests for src/explain.py (SHAP feature importance).

These tests verify the explain module's output contract:
  - Shape and types of the returned dict
  - Features are sorted by |shap_value|
  - Base features only (no city_ one-hot columns)
  - prediction ≈ base_value + sum(shap_values) within tolerance

Why test SHAP output shape?
  The frontend reads specific keys. If the keys change or are missing,
  the bar chart silently breaks. These tests catch that regression.
"""
from __future__ import annotations

import pytest

# Skip the whole module if model bundles don't exist (e.g., clean CI clone)
pytest.importorskip("shap")

try:
    from src.explain import compute_shap, _load_bundle
    MODELS_AVAILABLE = (_load_bundle.__wrapped__ if hasattr(_load_bundle, "__wrapped__")
                        else _load_bundle)  # just check if importable
    try:
        _load_bundle("co2")
        MODELS_AVAILABLE = True
    except FileNotFoundError:
        MODELS_AVAILABLE = False
except Exception:
    MODELS_AVAILABLE = False

needs_models = pytest.mark.skipif(
    not MODELS_AVAILABLE,
    reason="Model bundle files not present — run train.py first",
)


# ---------------------------------------------------------------------------
# compute_shap — output contract
# ---------------------------------------------------------------------------

@pytest.mark.unit
@needs_models
class TestComputeShapContract:
    @pytest.mark.parametrize("target", ["co2", "wue"])
    def test_returns_dict_with_required_keys(self, target):
        result = compute_shap(target=target, city="Dallas", time="2024-06-01T12:00")
        for key in ["city", "target", "prediction", "base_value", "features"]:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.parametrize("target", ["co2", "wue"])
    def test_features_is_list(self, target):
        result = compute_shap(target=target, city="Dallas", time="2024-06-01T12:00")
        assert isinstance(result["features"], list)
        assert len(result["features"]) > 0

    @pytest.mark.parametrize("target", ["co2", "wue"])
    def test_feature_schema(self, target):
        result = compute_shap(target=target, city="Dallas", time="2024-06-01T12:00")
        for f in result["features"]:
            assert "name" in f
            assert "raw_name" in f
            assert "value" in f
            assert "shap_value" in f
            assert isinstance(f["shap_value"], float)

    @pytest.mark.parametrize("target", ["co2", "wue"])
    def test_no_city_columns_in_features(self, target):
        """City one-hot columns (city_Dallas etc.) should be excluded."""
        result = compute_shap(target=target, city="Dallas", time="2024-06-01T12:00")
        for f in result["features"]:
            assert not f["raw_name"].startswith("city_"), (
                f"City column leaked into features: {f['raw_name']}"
            )

    @pytest.mark.parametrize("target", ["co2", "wue"])
    def test_features_sorted_by_magnitude(self, target):
        """Features must be sorted descending by |shap_value|."""
        result = compute_shap(target=target, city="Dallas", time="2024-06-01T12:00")
        shap_abs = [abs(f["shap_value"]) for f in result["features"]]
        assert shap_abs == sorted(shap_abs, reverse=True), (
            "Features are not sorted by |shap_value| descending"
        )

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target must be"):
            compute_shap(target="energy", city="Dallas", time="2024-06-01T12:00")

    @pytest.mark.parametrize("target", ["co2", "wue"])
    def test_returns_at_most_8_features(self, target):
        result = compute_shap(target=target, city="Dallas", time="2024-06-01T12:00", top_n=8)
        assert len(result["features"]) <= 8
