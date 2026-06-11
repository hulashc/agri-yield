"""
monitoring/psi_detector.py

Population Stability Index (PSI) drift detector.
Compares live request feature distributions against a reference
distribution built from NASA POWER historical data.

PSI < 0.25  → green  (no action)
PSI 0.25–0.50 → amber  (log warning)
PSI > 0.50  → red    (trigger retraining)

Note: thresholds are intentionally wider than the classic 0.10/0.20
because training data uses synthetic/proxy values.

Drift buffer persistence:
  Live feature values are stored in Redis as capped lists, keyed by
  field_id and feature name. If Redis is unavailable the detector
  falls back to an in-memory dict (same behaviour as before this change,
  with the loss that history resets on container restart).
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.prometheus_metrics import DRIFT_LEVEL, DRIFT_WARNINGS_TOTAL, PSI_SCORE

logger = logging.getLogger(__name__)

PSI_AMBER = 0.25
PSI_RED = 0.50
MIN_CURRENT_SAMPLES = 30

MONITORED_FEATURES = [
    "rainfall_today_mm",
    "t2m_max_today",
    "t2m_min_today",
    "et0_today",
    "solar_radiation_today",
    "gdd_accumulation",
    "ndvi_latest",
]

BUFFER_SIZE = 500

# ── Redis buffer setup ────────────────────────────────────────────────────────
# If REDIS_URL is not set or Redis is unavailable, fall back to in-memory dict.
REDIS_URL = os.getenv("REDIS_URL", "")
_redis_client = None
_memory_buffer: dict = {}  # fallback only


def _get_redis():
    """Return a Redis client (or None if unavailable)."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_URL:
        return None
    try:
        import redis  # noqa: PLC0415
        client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1)
        client.ping()
        _redis_client = client
        logger.info("Drift buffer: connected to Redis at %s", REDIS_URL)
        return _redis_client
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — drift buffer will use in-memory fallback.", exc)
        return None


def _redis_key(field_id: str, feature: str) -> str:
    return f"drift:buffer:{field_id}:{feature}"


def _append_to_buffer(field_id: str, feature: str, value: float) -> np.ndarray:
    """
    Append a live value to the rolling buffer and return the current window.

    Tries Redis first (persistent across restarts).
    Falls back to an in-memory dict if Redis is unavailable.
    """
    r = _get_redis()

    if r is not None:
        key = _redis_key(field_id, feature)
        try:
            pipe = r.pipeline()
            pipe.rpush(key, value)
            pipe.ltrim(key, -BUFFER_SIZE, -1)  # keep only the last BUFFER_SIZE values
            pipe.lrange(key, 0, -1)
            results = pipe.execute()
            raw = results[2]  # lrange result
            return np.array([float(v) for v in raw])
        except Exception as exc:
            logger.debug("Redis buffer write failed (%s) — using memory fallback.", exc)
            # Drop through to in-memory

    # In-memory fallback
    _memory_buffer.setdefault(field_id, {}).setdefault(feature, [])
    buf = _memory_buffer[field_id][feature]
    buf.append(value)
    if len(buf) > BUFFER_SIZE:
        buf.pop(0)
    return np.array(buf)


# ── PSI computation ───────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) < 10 or len(current) < MIN_CURRENT_SAMPLES:
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    def safe_pct(arr: np.ndarray, bins: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(arr, bins=bins)
        pct = counts / len(arr)
        pct = np.where(pct == 0, 1e-4, pct)
        return pct

    ref_pct = safe_pct(reference, breakpoints)
    cur_pct = safe_pct(current, breakpoints)

    min_len = min(len(ref_pct), len(cur_pct))
    ref_pct = ref_pct[:min_len]
    cur_pct = cur_pct[:min_len]

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def load_reference_distribution(
    feature: str, data_root: str = "data/raw/nasa_power"
) -> np.ndarray:
    """
    Load the reference distribution for a single feature.
    Kept for backward compatibility — prefer reference_cache.py at startup.
    """
    root = Path(data_root)
    all_files = list(root.glob("*/[0-9][0-9][0-9][0-9].parquet"))

    nasa_col_map = {
        "rainfall_today_mm": "PRECTOTCORR",
        "t2m_max_today": "T2M_MAX",
        "t2m_min_today": "T2M_MIN",
        "et0_today": "EVPTRNS",
        "solar_radiation_today": "ALLSKY_SFC_SW_DWN",
    }

    col = nasa_col_map.get(feature)
    if col is None:
        return np.array([])

    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f, columns=["date", col])
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return np.array([])

    combined = pd.concat(dfs)
    return combined[col].dropna().values


# ── Main evaluation entry point ───────────────────────────────────────────────

def evaluate_drift(
    field_id: str,
    live_features: dict,
    reference_cache: dict | None = None,
) -> dict:
    """
    Run PSI for each monitored feature using a rolling buffer of live values.

    Buffer is Redis-backed (persistent) when REDIS_URL is set;
    falls back to in-memory otherwise.

    Returns green until MIN_CURRENT_SAMPLES requests have been seen per field.
    """
    psi_scores = {}

    for feature in MONITORED_FEATURES:
        val = live_features.get(feature)
        if val is None:
            continue

        current = _append_to_buffer(field_id, feature, float(val))

        if len(current) < MIN_CURRENT_SAMPLES:
            psi_scores[feature] = 0.0
            PSI_SCORE.labels(feature_name=feature).set(0.0)
            continue

        ref = (
            reference_cache.get(feature)
            if reference_cache
            else load_reference_distribution(feature)
        )
        if ref is None or len(ref) == 0:
            continue

        psi = compute_psi(ref, current)
        psi_scores[feature] = round(psi, 4)
        PSI_SCORE.labels(feature_name=feature).set(psi)

        if psi > PSI_RED:
            DRIFT_WARNINGS_TOTAL.labels(field_id=field_id, feature_name=feature).inc()
            logger.warning("DRIFT RED: %s/%s PSI=%.3f", field_id, feature, psi)
        elif psi > PSI_AMBER:
            logger.info("DRIFT AMBER: %s/%s PSI=%.3f", field_id, feature, psi)

    max_psi = max(psi_scores.values()) if psi_scores else 0.0

    if max_psi > PSI_RED:
        level = "red"
        drift_level_code = 2
    elif max_psi > PSI_AMBER:
        level = "amber"
        drift_level_code = 1
    else:
        level = "green"
        drift_level_code = 0

    DRIFT_LEVEL.labels(field_id=field_id).set(drift_level_code)

    return {
        "max_psi": round(max_psi, 4),
        "drift_warning": max_psi > PSI_RED,
        "drift_level": level,
        "psi_scores": psi_scores,
    }
