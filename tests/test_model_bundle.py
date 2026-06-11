"""
tests/test_model_bundle.py

Tests for Phase 2: ModelBundle, quantile ordering, model card, /model/info endpoint.
"""

from __future__ import annotations

import importlib
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from training.model_bundle import ModelBundle
from training.utils.features import FEATURE_COLS


# ---------------------------------------------------------------------------
# 1. Bundle save + load round-trip
# ---------------------------------------------------------------------------

def test_bundle_save_and_load(toy_bundle, tmp_path):
    path = tmp_path / "bundle.pkl"
    toy_bundle.save(path)
    loaded = ModelBundle.load(path)
    assert isinstance(loaded, ModelBundle)
    assert loaded.training_meta["rmse_kg_ha"] == toy_bundle.training_meta["rmse_kg_ha"]


# ---------------------------------------------------------------------------
# 2. Quantile ordering: lower <= mean <= upper on every sample
# ---------------------------------------------------------------------------

def test_quantile_ordering_never_crossed(toy_bundle):
    rng = np.random.default_rng(99)
    X = pd.DataFrame({col: rng.random(200) for col in FEATURE_COLS})
    mean, lower, upper = toy_bundle.predict(X)
    assert np.all(lower <= mean), "lower > mean on some samples"
    assert np.all(mean <= upper), "upper < mean on some samples"
    assert np.all(lower <= upper), "lower > upper on some samples (crossing)"


# ---------------------------------------------------------------------------
# 3. Feature importance dict contains all FEATURE_COLS
# ---------------------------------------------------------------------------

def test_feature_importance_covers_all_features(toy_bundle):
    fi = toy_bundle.feature_importance
    assert isinstance(fi, dict)
    assert set(fi.keys()) == set(FEATURE_COLS), (
        f"Missing keys: {set(FEATURE_COLS) - set(fi.keys())}"
    )
    # All importances should be non-negative
    assert all(v >= 0 for v in fi.values())


# ---------------------------------------------------------------------------
# 4. top_features returns sorted list
# ---------------------------------------------------------------------------

def test_top_features_sorted_descending(toy_bundle):
    top = toy_bundle.top_features(n=5)
    assert len(top) == 5
    scores = [item["importance"] for item in top]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 5. Training metadata fields present
# ---------------------------------------------------------------------------

def test_training_meta_required_fields(toy_bundle):
    meta = toy_bundle.training_meta
    required = [
        "trained_at", "rmse_kg_ha", "mae_kg_ha",
        "pi_coverage_pct", "confidence_method", "n_train", "n_test",
    ]
    for key in required:
        assert key in meta, f"Missing training_meta key: '{key}'"


# ---------------------------------------------------------------------------
# 6. Legacy pickle fallback: serving/model still loads and predicts
# ---------------------------------------------------------------------------

def test_legacy_pickle_fallback_loads_and_predicts(model_pkl_path, tmp_path):
    """
    When only model.pkl is present (no bundle), serving/model should
    load and return symmetric CI predictions without raising.
    """
    import serving.model
    importlib.reload(serving.model)
    loaded = serving.model.load_model()
    assert loaded, "Legacy model.pkl failed to load"
    assert not serving.model.using_bundle(), "Should NOT be using bundle"

    rng = np.random.default_rng(1)
    X = pd.DataFrame({col: rng.random(5) for col in FEATURE_COLS})
    mean, lower, upper = serving.model.predict(X)
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)


# ---------------------------------------------------------------------------
# 7. /model/info endpoint returns well-formed JSON
# ---------------------------------------------------------------------------

def test_model_info_endpoint(app):
    client = TestClient(app)
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_version" in body
    assert "confidence_method" in body
    assert "using_bundle" in body
    assert "top_features" in body
    assert isinstance(body["top_features"], list)
    # When bundle is loaded, confidence_method must NOT be legacy
    if body["using_bundle"]:
        assert body["confidence_method"] == "xgboost_quantile_regression"
        assert body["rmse_kg_ha"] is not None
        assert body["pi_coverage_pct"] is not None


# ---------------------------------------------------------------------------
# 8. /predict response includes confidence_method field
# ---------------------------------------------------------------------------

def test_predict_response_includes_confidence_method(app):
    client = TestClient(app)
    resp = client.post("/predict", json={
        "field_id": "F001",
        "event_timestamp": "2026-05-17T00:00:00",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "confidence_method" in body
    assert body["confidence_method"] in (
        "xgboost_quantile_regression", "legacy_symmetric_ci"
    )


# ---------------------------------------------------------------------------
# 9. /health response includes using_bundle flag
# ---------------------------------------------------------------------------

def test_health_includes_using_bundle(app):
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "using_bundle" in resp.json()
