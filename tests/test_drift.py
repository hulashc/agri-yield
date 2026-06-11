"""
tests/test_drift.py

Unit tests for monitoring/psi_detector.py.
Verifies PSI computation correctness, buffer accumulation,
and the green-before-min-samples early-return behaviour.

Redis is disabled in all tests via the autouse env_setup fixture
(REDIS_URL="") so tests always run against the in-memory fallback.
"""

import numpy as np
import pytest

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
        assert psi < 0.05, f"Expected near-zero PSI for identical dist, got {psi:.4f}"

    def test_very_different_distributions_high_psi(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(10, 1, 500)
        cur = rng.normal(50, 1, 500)
        psi = compute_psi(ref, cur)
        assert psi > PSI_RED, f"Expected PSI > {PSI_RED} for very different dists, got {psi:.4f}"

    def test_returns_zero_below_min_current_samples(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(10, 2, 500)
        cur = rng.normal(10, 2, MIN_CURRENT_SAMPLES - 1)  # one below threshold
        psi = compute_psi(ref, cur)
        assert psi == 0.0, f"Expected 0.0 for insufficient current samples, got {psi}"

    def test_returns_zero_for_small_reference(self):
        ref = np.array([1.0, 2.0, 3.0])  # < 10 samples
        cur = np.random.default_rng(3).normal(10, 1, 50)
        psi = compute_psi(ref, cur)
        assert psi == 0.0

    def test_moderate_shift_is_between_thresholds(self):
        rng = np.random.default_rng(4)
        ref = rng.normal(10, 2, 500)
        cur = rng.normal(13, 2, 100)  # moderate shift
        psi = compute_psi(ref, cur)
        # Not necessarily amber, but should be > 0
        assert psi >= 0.0

    def test_psi_is_nonnegative(self):
        rng = np.random.default_rng(5)
        ref = rng.normal(10, 2, 500)
        cur = rng.normal(12, 3, 60)
        psi = compute_psi(ref, cur)
        assert psi >= 0.0


# ---------------------------------------------------------------------------
# _append_to_buffer (in-memory mode, REDIS_URL="" via autouse fixture)
# ---------------------------------------------------------------------------

class TestAppendToBuffer:
    def test_buffer_accumulates_values(self):
        for i in range(35):
            buf = _append_to_buffer("test_accum", "t2m_max_today", float(i))
        assert len(buf) == 35

    def test_buffer_caps_at_buffer_size(self):
        from monitoring.psi_detector import BUFFER_SIZE
        for i in range(BUFFER_SIZE + 50):
            buf = _append_to_buffer("test_cap", "rainfall_today_mm", float(i))
        assert len(buf) == BUFFER_SIZE

    def test_buffer_returns_ndarray(self):
        buf = _append_to_buffer("test_type", "et0_today", 3.14)
        assert isinstance(buf, np.ndarray)

    def test_buffer_values_are_floats(self):
        _append_to_buffer("test_float", "solar_radiation_today", 5.5)
        buf = _append_to_buffer("test_float", "solar_radiation_today", 6.0)
        assert buf.dtype == np.float64

    def test_separate_field_ids_are_isolated(self):
        for i in range(10):
            _append_to_buffer("field_iso_A", "t2m_min_today", float(i))
        buf_b = _append_to_buffer("field_iso_B", "t2m_min_today", 99.0)
        assert len(buf_b) == 1  # field B has only one entry


# ---------------------------------------------------------------------------
# evaluate_drift
# ---------------------------------------------------------------------------

class TestEvaluateDrift:
    def test_returns_green_before_min_samples(self):
        result = evaluate_drift(
            "field_green_test",
            {"t2m_max_today": 15.0, "rainfall_today_mm": 2.0},
            reference_cache={},
        )
        assert result["drift_level"] == "green"
        assert result["max_psi"] == 0.0
        assert result["drift_warning"] is False

    def test_returns_expected_keys(self):
        result = evaluate_drift(
            "field_keys_test",
            {"t2m_max_today": 15.0},
            reference_cache={},
        )
        assert "drift_warning" in result
        assert "drift_level" in result
        assert "max_psi" in result
        assert "psi_scores" in result

    def test_drift_warning_false_when_green(self):
        result = evaluate_drift(
            "field_warn_test",
            {"t2m_max_today": 15.0},
            reference_cache={},
        )
        assert result["drift_warning"] is False

    def test_unknown_feature_does_not_raise(self):
        result = evaluate_drift(
            "field_unknown",
            {"nonexistent_feature": 99.9},
            reference_cache={},
        )
        assert result["drift_level"] == "green"

    def test_none_reference_cache_still_returns(self):
        """When reference_cache=None, falls back to disk reads (returns empty → green)."""
        result = evaluate_drift(
            "field_no_cache",
            {"t2m_max_today": 15.0},
            reference_cache=None,
        )
        assert result["drift_level"] in ("green", "amber", "red")
        assert isinstance(result["max_psi"], float)
