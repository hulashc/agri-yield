"""
monitoring/psi_detector.py

Population Stability Index (PSI) drift detector.
Compares live request feature distributions against a reference
distribution built from NASA POWER historical data.

PSI < 0.25  → green  (no action)
PSI 0.25-0.50 → amber  (log warning)
PSI > 0.50  → red    (trigger retraining)

Note: thresholds are intentionally wider than the classic 0.10/0.20
because training data uses synthetic/proxy values. PSI is only meaningful
once a rolling buffer of live requests is available (MIN_CURRENT_SAMPLES).

Buffer persistence:
  Live feature values are persisted in Redis (key prefix drift:buf:<field>:<feature>)
  so drift state survives container restarts and redeployments.
  If REDIS_URL is unset or Redis is unreachable, the detector falls back to an
  in-memory dict silently — monitoring still works, just resets on restart.

Performance note:
  load_reference_distribution() reads all NASA POWER Parquet files from disk.
  This is EXPENSIVE and must NOT be called on every request.
  The recommended pattern is to call _warm_reference_cache() at app startup
  (see serving/app.py lifespan) and always pass reference_cache=_REFERENCE_CACHE
  into evaluate_drift(). The fallback path (no cache) will log a warning.
"""

import logging
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from monitoring.prometheus_metrics import DRIFT_LEVEL, DRIFT_WARNINGS_TOTAL, PSI_SCORE

logger = logging.getLogger(__name__)

PSI_AMBER = 0.25
PSI_RED = 0.50

# Minimum number of live samples required before PSI is meaningful.
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

# ---------------------------------------------------------------------------
# Redis-backed rolling buffer
# ---------------------------------------------------------------------------

BUFFER_SIZE = 500
REDIS_KEY_PREFIX = "drift:buf:"
REDIS_TTL_SECONDS = 86400 * 7  # 7 days — keeps drift history across deploys

# In-memory fallback — used when Redis is unavailable
_live_buffer: dict[str, dict[str, list[float]]] = {}

# Cached Redis client — None means "not yet attempted" or "unavailable"
_redis_client = None
_redis_attempted = False


def _get_redis():
    """
    Return a connected Redis client, or None if Redis is not configured
    or is unreachable.

    The connection is attempted once and then cached.
    If Redis goes down mid-run, individual buffer writes will catch the
    exception and fall back to in-memory for that call.
    """
    global _redis_client, _redis_attempted
    if _redis_attempted:
        return _redis_client
    _redis_attempted = True
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        logger.info("REDIS_URL not set — drift buffer will use in-memory fallback.")
        return None
    try:
        import redis as redis_lib  # noqa: PLC0415
        client = redis_lib.from_url(redis_url, socket_connect_timeout=2, decode_responses=False)
        client.ping()
        _redis_client = client
        logger.info("Drift buffer: Redis connected (%s)", redis_url.split("@")[-1])
        return _redis_client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable (%s) — drift buffer will use in-memory fallback.", exc
        )
        return None


def _append_to_buffer(field_id: str, feature: str, value: float) -> NDArray[np.float64]:
    """
    Append a live value to the rolling buffer and return the current window.

    Persistence hierarchy:
    1. Redis (survives restarts) — used when REDIS_URL is set and reachable
    2. In-memory dict (resets on restart) — fallback when Redis is unavailable
    """
    r = _get_redis()
    key = f"{REDIS_KEY_PREFIX}{field_id}:{feature}"

    if r is not None:
        try:
            r.rpush(key, value)
            r.ltrim(key, -BUFFER_SIZE, -1)
            r.expire(key, REDIS_TTL_SECONDS)
            raw = r.lrange(key, 0, -1)
            return np.array([float(v) for v in raw], dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Redis buffer write failed for %s/%s (%s) — using in-memory.",
                field_id, feature, exc,
            )
            # Fall through to in-memory

    # In-memory fallback
    _live_buffer.setdefault(field_id, {}).setdefault(feature, [])
    buf = _live_buffer[field_id][feature]
    buf.append(value)
    if len(buf) > BUFFER_SIZE:
        buf.pop(0)
    return np.array(buf, dtype=float)


def compute_psi(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Sum( (actual% - expected%) * ln(actual% / expected%) )

    Args:
        reference: Array of reference distribution values (training period)
        current:   Array of current/live values
        n_bins:    Number of histogram bins

    Returns:
        PSI score (float). Higher = more drift. Returns 0.0 if either
        array has insufficient samples.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) < 10 or len(current) < MIN_CURRENT_SAMPLES:
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    def safe_pct(arr: NDArray[np.float64], bins: NDArray[np.float64]) -> NDArray[np.float64]:
        counts, _ = np.histogram(arr, bins=bins)
        pct = counts / len(arr)
        pct = np.where(pct == 0, 1e-4, pct)
        return np.asarray(pct, dtype=float)

    ref_pct = safe_pct(reference, breakpoints)
    cur_pct = safe_pct(current, breakpoints)

    min_len = min(len(ref_pct), len(cur_pct))
    ref_pct = ref_pct[:min_len]
    cur_pct = cur_pct[:min_len]

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def load_reference_distribution(
    feature: str, data_root: str = "data/raw/nasa_power"
) -> NDArray[np.float64]:
    """
    Load the reference distribution for a feature from saved Parquet files.

    NOTE: This is a disk-read operation. Call it at startup only via
    _warm_reference_cache() and never inside hot request paths.
    """
    root = Path(data_root)
    all_files = list(root.glob("*/[0-9][0-9][0-9][0-9].parquet"))

    nasa_col_map: dict[str, str] = {
        "rainfall_today_mm": "PRECTOTCORR",
        "t2m_max_today": "T2M_MAX",
        "t2m_min_today": "T2M_MIN",
        "et0_today": "EVPTRNS",
        "solar_radiation_today": "ALLSKY_SFC_SW_DWN",
    }

    col = nasa_col_map.get(feature)
    if col is None:
        return np.array([], dtype=float)

    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f, columns=["date", col])
            dfs.append(df)
        except Exception:  # noqa: BLE001
            continue

    if not dfs:
        return np.array([], dtype=float)

    combined = pd.concat(dfs)
    return np.asarray(combined[col].dropna().to_numpy(dtype=float))


def evaluate_drift(
    field_id: str,
    live_features: dict[str, object],
    reference_cache: dict[str, NDArray[np.float64]] | None = None,
) -> dict[str, object]:
    """
    Run PSI for each monitored feature using a rolling buffer of live values.
    Returns green until MIN_CURRENT_SAMPLES requests have been seen per field.

    The rolling buffer is persisted in Redis when available, so drift history
    survives container restarts and redeployments.

    Args:
        field_id: Unique field identifier for per-field buffer tracking.
        live_features: Dict of feature name to live value from the request.
        reference_cache: Pre-loaded reference distributions (dict of feature to np.ndarray).
                         Should be populated at app startup via _warm_reference_cache().
                         If None, falls back to per-call disk reads (slow — avoid in production).
    """
    if reference_cache is None:
        logger.warning(
            "evaluate_drift called without reference_cache — falling back to disk reads. "
            "This is slow. Pass the startup-warmed cache from serving/app.py instead."
        )

    psi_scores: dict[str, float] = {}

    for feature in MONITORED_FEATURES:
        val = live_features.get(feature)
        if val is None:
            continue

        current = _append_to_buffer(field_id, feature, float(val))  # type: ignore[arg-type]

        if len(current) < MIN_CURRENT_SAMPLES:
            psi_scores[feature] = 0.0
            PSI_SCORE.labels(feature_name=feature).set(0.0)
            continue

        if reference_cache is not None:
            ref = reference_cache.get(feature)
        else:
            ref = load_reference_distribution(feature)

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
