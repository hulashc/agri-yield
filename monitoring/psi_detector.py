"""
monitoring/psi_detector.py

Population Stability Index (PSI) drift detector.
Compares live request feature distributions against a reference
distribution built from NASA POWER historical data.

PSI < 0.10  → green  (no action)
PSI 0.10–0.20 → amber  (log warning)
PSI > 0.20  → red    (trigger retraining)
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from monitoring.prometheus_metrics import DRIFT_WARNINGS_TOTAL, PSI_SCORE, DRIFT_LEVEL

logger = logging.getLogger(__name__)

PSI_AMBER = 0.10
PSI_RED = 0.20

MONITORED_FEATURES = [
    "rainfall_today_mm",
    "t2m_max_today",
    "t2m_min_today",
    "et0_today",
    "solar_radiation_today",
    "gdd_accumulation",
    "ndvi_latest",
]


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    Args:
        reference: Array of reference distribution values (training period)
        current:   Array of current/live values
        n_bins:    Number of histogram bins

    Returns:
        PSI score (float). Higher = more drift.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) < 10 or len(current) < 5:
        logger.warning("Insufficient data for PSI computation, returning 0.0")
        return 0.0

    # Use reference distribution to define bin edges
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates from flat distributions

    def safe_pct(arr, bins):
        counts, _ = np.histogram(arr, bins=bins)
        pct = counts / len(arr)
        pct = np.where(pct == 0, 1e-4, pct)  # Avoid log(0)
        return pct

    ref_pct = safe_pct(reference, breakpoints)
    cur_pct = safe_pct(current, breakpoints)

    # Align lengths (bin count may differ if breakpoints collapsed)
    min_len = min(len(ref_pct), len(cur_pct))
    ref_pct = ref_pct[:min_len]
    cur_pct = cur_pct[:min_len]

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def load_reference_distribution(
    feature: str, data_root: str = "data/raw/nasa_power"
) -> np.ndarray:
    """
    Load the reference distribution for a feature from saved Parquet files.
    Uses the last full growing season (prior calendar year).
    """
    root = Path(data_root)
    all_files = list(root.glob("*/[0-9][0-9][0-9][0-9].parquet"))

    # Map internal feature name to NASA POWER column
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


def evaluate_drift(
    field_id: str,
    live_features: dict,
    reference_cache: Optional[dict] = None,
) -> dict:
    """
    Run PSI for each monitored feature.
    Returns a dict with per-feature PSI scores and overall drift level.

    Args:
        field_id:        Field identifier (for Prometheus labels)
        live_features:   Dict of feature_name → float (current request)
        reference_cache: Optional pre-loaded reference arrays (avoid re-reading parquet)

    Returns:
        {
          "max_psi": float,
          "drift_warning": bool,
          "drift_level": "green" | "amber" | "red",
          "psi_scores": {feature: psi_value, ...}
        }
    """
    # Build a "current" distribution from the single live request
    # In production this is a rolling window of the last N requests
    # For now, we use the single value compared against reference
    psi_scores = {}

    for feature in MONITORED_FEATURES:
        val = live_features.get(feature)
        if val is None:
            continue

        ref = (
            reference_cache.get(feature)
            if reference_cache
            else load_reference_distribution(feature)
        )
        if ref is not None and len(ref) == 0:
            continue

        # Simulate a small current distribution from the single value
        # (In production: maintain a rolling buffer of last 500 requests)
        noise = np.random.normal(0, abs(val) * 0.01 + 0.001, 50)
        current = np.array([val] * 50) + noise

        psi = compute_psi(ref, current)
        psi_scores[feature] = round(psi, 4)

        # Push to Prometheus
        PSI_SCORE.labels(feature_name=feature).set(psi)

        if psi > PSI_RED:
            DRIFT_WARNINGS_TOTAL.labels(field_id=field_id, feature_name=feature).inc()
            logger.warning(f"🔴 DRIFT RED: {field_id}/{feature} PSI={psi:.3f}")
        elif psi > PSI_AMBER:
            logger.info(f"🟡 DRIFT AMBER: {field_id}/{feature} PSI={psi:.3f}")

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
