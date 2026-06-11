"""
tests/test_drift.py

Unit tests for monitoring/psi_detector.py.

Covers:
- compute_psi maths (identical, different, small-sample)
- _append_to_buffer accumulation with in-memory fallback (REDIS_URL="")
- evaluate_drift return shape and green-before-min-samples behaviour
"""

import numpy as np
import pytest

# Force in-memory buffer for all drift tests (Redis not available in CI)
import os
os.environ.setdefault("REDIS_URL", "")

from monitoring.psi_detector import (
    MIN_CURRENT_SAMPLES,
    PSI_AMBER,
    PSI_RED,
    _append_to_buffer,
    compute_psi,
    evaluate_drift,
)


# ---------------------------------------------------------------------------
# compute_psi
# ---------------------------------------------------------------------------

class TestComputePsi:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(10, 2, 500)
        psi = compute_psi(ref, ref.copy())
        assert psi < 0.05, f"Expected PSI near 0 for identical dist, got {psi:.4f}"

    def test_very_different_distributions_high(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(10, 1, 500)
        cur = rng.normal(50, 1, 500)
        psi = compute_psi(ref, cur)
        assert psi > 0.5, f"Expected high PSI for shifted dist, got {psi:.4f}"

    def test_returns_zero_when_current_below_min_samples(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(10, 2, 500)
        cur = rng.normal(10, 2, MIN_CURRENT_SAMPLES - 1)
        psi = compute_psi(ref, cur)
        assert psi == 0.0

    def test_returns_zero_when_reference_too_small(self):
        cur = np.random.default_rng(3).normal(10, 2, 100)
        psi = compute_psi(np.array([1.0, 2.0, 3.0]), cur)  # ref has only 3 values
        assert psi == 0.0

    def test_output_is_non_negative_for_similar_dists(self):
        rng = np.random.default_rng(4)
        ref = rng.normal(15, 3, 500)
        cur = rng.normal(15.5, 3.1, 100)
        psi = compute_psi(ref, cur)
        # PSI can be very slightly negative due to float rounding — clamp expected
        assert psi >= -0.01


# ---------------------------------------------------------------------------
# _append_to_buffer (in-memory mode since REDIS_URL="")
# ---------------------------------------------------------------------------

class TestAppendToBuffer:
    def test_buffer_accumulates_values(self):
        buf = None
        field = "drift_test_accumulate"
        for i in range(35):
            buf = _append_to_buffer(field, "t2m_max_today", float(i))
        assert buf is not None
        assert len(buf) == 35

    def test_buffer_does_not_exceed_max_size(self):
        from monitoring.psi_detector import BUFFER_SIZE
        field = "drift_test_capped"
        for i in range(BUFFER_SIZE + 50):
            buf = _append_to_buffer(field, "rainfall_today_mm", float(i % 100))
        assert len(buf) <= BUFFER_SIZE

    def test_buffer_returns_ndarray(self):
        buf = _append_to_buffer("drift_test_dtype", "et0_today", 3.14)
        assert isinstance(buf, np.ndarray)
        assert buf.dtype == float


# ---------------------------------------------------------------------------
# evaluate_drift
# ---------------------------------------------------------------------------

class TestEvaluateDrift:
    def test_green_before_min_samples(self):
        # A brand-new field with 0 history should be green
        result = evaluate_drift(
            "brand_new_field_xyz",
            {"t2m_max_today": 15.0, "rainfall_today_mm": 2.0},
            reference_cache={},
        )
        assert result["drift_level"] == "green"
        assert result["max_psi"] == 0.0
        assert result["drift_warning"] is False

    def test_returns_expected_keys(self):
        result = evaluate_drift(
            "key_check_field",
            {"t2m_max_today": 12.0},
            reference_cache={},
        )
        assert set(result.keys()) == {"max_psi", "drift_warning", "drift_level", "psi_scores"}

    def test_drift_level_values_are_valid(self):
        result = evaluate_drift(
            "level_check_field",
            {"t2m_max_today": 20.0, "rainfall_today_mm": 5.0},
            reference_cache={},
        )
        assert result["drift_level"] in ("green", "amber", "red")

    def test_psi_scores_keyed_by_feature(self):
        result = evaluate_drift(
            "psi_keys_field",
            {"t2m_max_today": 18.0, "rainfall_today_mm": 1.0},
            reference_cache={},
        )
        for key in result["psi_scores"]:
            assert isinstance(key, str)
            assert isinstance(result["psi_scores"][key], float)

    def test_max_psi_matches_psi_scores(self):
        result = evaluate_drift(
            "max_psi_check",
            {"t2m_max_today": 25.0},
            reference_cache={},
        )
        if result["psi_scores"]:
            assert result["max_psi"] == max(result["psi_scores"].values())

    def test_warning_consistent_with_level(self):
        result = evaluate_drift(
            "warn_level_consistency",
            {"t2m_max_today": 10.0},
            reference_cache={},
        )
        if result["drift_level"] == "red":
            assert result["drift_warning"] is True
        else:
            assert result["drift_warning"] is False
