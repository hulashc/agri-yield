"""
tests/test_ci_bounds.py

Baseline tests for confidence interval ordering and predict() output shape.
Phase 2 will add real quantile model tests once CI is model-derived.
"""

import numpy as np
import pandas as pd
import pytest

from training.utils.features import FEATURE_COLS


@pytest.fixture
def sample_feature_df():
    rng = np.random.default_rng(0)
    row = {col: float(rng.random()) for col in FEATURE_COLS}
    return pd.DataFrame([row])


def test_predict_output_is_triple(app, sample_feature_df):
    """predict() must return (preds, lower, upper) — a 3-tuple of arrays."""
    import serving.model as m
    result = m.predict(sample_feature_df)
    assert len(result) == 3
    preds, lower, upper = result
    assert len(preds) == 1
    assert len(lower) == 1
    assert len(upper) == 1


def test_predict_lower_less_than_upper(app, sample_feature_df):
    """lower_bound must always be strictly less than upper_bound."""
    import serving.model as m
    _, lower, upper = m.predict(sample_feature_df)
    assert lower[0] < upper[0], f"lower={lower[0]} is not < upper={upper[0]}"


def test_predict_mean_within_bounds(app, sample_feature_df):
    """Mean prediction must sit between lower and upper."""
    import serving.model as m
    preds, lower, upper = m.predict(sample_feature_df)
    assert lower[0] <= preds[0] <= upper[0], (
        f"mean={preds[0]} not in [{lower[0]}, {upper[0]}]"
    )


def test_predict_yields_positive(app, sample_feature_df):
    """Predictions must be positive kg/ha values — never negative."""
    import serving.model as m
    preds, lower, upper = m.predict(sample_feature_df)
    assert preds[0] > 0
    # Lower bound for Phase 1 (fixed fraction) may be positive too
    assert lower[0] >= 0


def test_predict_multiple_rows(app):
    """predict() must handle a batch of rows, not just one."""
    import serving.model as m
    rng = np.random.default_rng(99)
    rows = [{col: float(rng.random()) for col in FEATURE_COLS} for _ in range(5)]
    df = pd.DataFrame(rows)
    preds, lower, upper = m.predict(df)
    assert len(preds) == 5
    assert all(lower[i] < upper[i] for i in range(5))


def test_predict_handles_missing_columns(app):
    """predict() must fill missing columns with 0 rather than raising."""
    import serving.model as m
    # Only pass a subset of features
    df = pd.DataFrame([{"lat": 52.0, "lon": -1.5}])
    # Should not raise
    preds, lower, upper = m.predict(df)
    assert len(preds) == 1
