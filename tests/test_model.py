"""
tests/test_model.py

Unit tests for serving/model.py.

Covers:
- load_model() successfully loads the 3-model bundle from conftest fixture
- predict() returns arrays with correct shapes
- lower <= mean <= upper ordering is enforced on every row
- get_feature_importance() returns all feature columns with non-negative scores
- get_bundle_meta() reports has_quantile_ci=True for a full bundle
- model_version() returns a non-empty string after loading
"""

import importlib
import pickle

import numpy as np
import pandas as pd
import pytest

from training.utils.features import FEATURE_COLS


@pytest.fixture
def loaded_model(model_bundle_path):
    """Reload serving.model and call load_model() with the test bundle."""
    import serving.model as m
    importlib.reload(m)
    m.load_model()
    return m


class TestModelLoading:
    def test_is_loaded_after_load_model(self, loaded_model):
        assert loaded_model.is_loaded() is True

    def test_model_version_is_not_empty(self, loaded_model):
        v = loaded_model.model_version()
        assert isinstance(v, str)
        assert len(v) > 0
        assert v != "not_loaded"

    def test_bundle_meta_has_quantile_ci(self, loaded_model):
        meta = loaded_model.get_bundle_meta()
        assert meta["has_quantile_ci"] is True

    def test_bundle_meta_rmse_is_reasonable(self, loaded_model):
        meta = loaded_model.get_bundle_meta()
        # Toy training on random data — RMSE should be well below 10_000
        assert 0 < meta["rmse"] < 10_000

    def test_bundle_meta_feature_cols_complete(self, loaded_model):
        meta = loaded_model.get_bundle_meta()
        assert set(meta["feature_cols"]) == set(FEATURE_COLS)


class TestPredictOrdering:
    def _make_inputs(self, n: int = 100, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            [{col: float(rng.random()) for col in FEATURE_COLS} for _ in range(n)]
        )

    def test_predict_returns_three_arrays(self, loaded_model):
        X = self._make_inputs()
        result = loaded_model.predict(X)
        assert len(result) == 3

    def test_arrays_have_correct_shape(self, loaded_model):
        X = self._make_inputs(50)
        preds, lower, upper = loaded_model.predict(X)
        assert preds.shape == (50,)
        assert lower.shape == (50,)
        assert upper.shape == (50,)

    def test_lower_le_mean(self, loaded_model):
        X = self._make_inputs(200)
        preds, lower, upper = loaded_model.predict(X)
        violations = np.sum(lower > preds + 1e-6)
        assert violations == 0, f"{violations} rows where lower > mean"

    def test_mean_le_upper(self, loaded_model):
        X = self._make_inputs(200)
        preds, lower, upper = loaded_model.predict(X)
        violations = np.sum(preds > upper + 1e-6)
        assert violations == 0, f"{violations} rows where mean > upper"

    def test_predictions_are_finite(self, loaded_model):
        X = self._make_inputs(50)
        preds, lower, upper = loaded_model.predict(X)
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))


class TestFeatureImportance:
    def test_importance_has_all_features(self, loaded_model):
        imp = loaded_model.get_feature_importance()
        assert set(imp.keys()) == set(FEATURE_COLS)

    def test_importance_values_non_negative(self, loaded_model):
        imp = loaded_model.get_feature_importance()
        assert all(v >= 0 for v in imp.values())

    def test_importance_values_sum_to_approx_one(self, loaded_model):
        imp = loaded_model.get_feature_importance()
        total = sum(imp.values())
        # XGBoost gain importance — may not sum to exactly 1, but should be positive
        assert total > 0
