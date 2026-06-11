"""
tests/test_model.py

Unit tests for serving/model.py.
Verifies that the model bundle loads correctly, that predictions
respect CI ordering (lower <= mean <= upper), and that feature
importance and metadata are complete and structurally correct.
"""

import importlib
import pickle

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from training.utils.features import FEATURE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_bundle(bundle: dict, tmp_path, monkeypatch):
    """Serialise bundle to disk and point env vars at it."""
    bp = tmp_path / "model_bundle.pkl"
    bp.write_bytes(pickle.dumps(bundle))
    monkeypatch.setenv("PICKLE_BUNDLE_PATH", str(bp))
    # Also set legacy path so the loader doesn't find a stale env var
    monkeypatch.setenv(
        "PICKLE_MODEL_PATH", str(tmp_path / "nonexistent_legacy.pkl")
    )
    return str(bp)


def _load_module(monkeypatch):
    """Reload serving.model so it picks up env-var changes made in this test."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "disabled")
    import serving.model as m
    importlib.reload(m)
    return m


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

class TestBundleLoading:
    def test_load_bundle_returns_true(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        result = m.load_model()
        assert result is True

    def test_is_loaded_after_load(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        assert m.is_loaded() is True

    def test_has_quantile_ci_with_bundle(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        meta = m.get_bundle_meta()
        assert meta["has_quantile_ci"] is True

    def test_model_version_is_string(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        assert isinstance(m.model_version(), str)
        assert m.model_version() != "not_loaded"

    def test_legacy_pkl_falls_back_gracefully(self, model_pkl_path, monkeypatch, toy_model_bytes, tmp_path):
        m = _load_module(monkeypatch)
        result = m.load_model()
        assert result is True
        assert m.is_loaded() is True
        # Legacy pickle has no quantile models
        meta = m.get_bundle_meta()
        assert meta["has_quantile_ci"] is False


# ---------------------------------------------------------------------------
# CI ordering: lower <= mean <= upper (must hold for every row)
# ---------------------------------------------------------------------------

class TestCIOrdering:
    def test_lower_le_mean_le_upper_bundle(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        rng = np.random.default_rng(7)
        X = pd.DataFrame(
            [{c: rng.random() for c in FEATURE_COLS} for _ in range(100)]
        )
        preds, lower, upper = m.predict(X)
        assert np.all(lower <= preds + 1e-6), (
            f"lower > mean in {(lower > preds + 1e-6).sum()} rows"
        )
        assert np.all(preds <= upper + 1e-6), (
            f"mean > upper in {(preds > upper + 1e-6).sum()} rows"
        )

    def test_lower_le_upper_always(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        rng = np.random.default_rng(8)
        X = pd.DataFrame(
            [{c: rng.random() for c in FEATURE_COLS} for _ in range(100)]
        )
        _, lower, upper = m.predict(X)
        assert np.all(lower <= upper + 1e-6)

    def test_ci_ordering_with_legacy_pkl_heuristic(self, model_pkl_path, monkeypatch):
        """
        Legacy pkl falls back to ±15% heuristic.
        CI ordering must still hold.
        """
        m = _load_module(monkeypatch)
        m.load_model()
        rng = np.random.default_rng(9)
        X = pd.DataFrame(
            [{c: rng.random() for c in FEATURE_COLS} for _ in range(50)]
        )
        preds, lower, upper = m.predict(X)
        assert np.all(lower <= preds + 1e-6)
        assert np.all(preds <= upper + 1e-6)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_importance_has_all_feature_cols(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        importance = m.get_feature_importance()
        assert set(importance.keys()) == set(FEATURE_COLS)

    def test_importance_values_are_nonnegative(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        importance = m.get_feature_importance()
        assert all(v >= 0 for v in importance.values())

    def test_importance_values_sum_positive(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        importance = m.get_feature_importance()
        assert sum(importance.values()) > 0


# ---------------------------------------------------------------------------
# Bundle metadata
# ---------------------------------------------------------------------------

class TestBundleMeta:
    def test_meta_contains_expected_keys(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        meta = m.get_bundle_meta()
        for key in ("rmse", "trained_at", "dataset_source", "feature_cols",
                    "has_quantile_ci", "model_version"):
            assert key in meta, f"Missing key: {key}"

    def test_rmse_is_positive(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        meta = m.get_bundle_meta()
        assert meta["rmse"] > 0

    def test_feature_cols_match_training(self, model_bundle_path, monkeypatch):
        m = _load_module(monkeypatch)
        m.load_model()
        meta = m.get_bundle_meta()
        assert set(meta["feature_cols"]) == set(FEATURE_COLS)
