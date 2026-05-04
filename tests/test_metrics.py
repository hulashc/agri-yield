"""
Smoke tests for training metric utilities.
"""

import numpy as np
import pytest

from training.utils.metrics import compute_metrics


def test_compute_metrics_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    preds = np.array([100.0, 200.0, 300.0])
    m = compute_metrics(y, preds)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_compute_metrics_returns_all_keys():
    y = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.1, 2.1, 3.1])
    m = compute_metrics(y, preds)
    assert "rmse" in m
    assert "mae" in m
    assert "r2" in m


def test_compute_metrics_non_negative_rmse():
    y = np.random.rand(50) * 1000
    preds = np.random.rand(50) * 1000
    m = compute_metrics(y, preds)
    assert m["rmse"] >= 0
    assert m["mae"] >= 0
