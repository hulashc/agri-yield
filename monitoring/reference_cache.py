"""
monitoring/reference_cache.py

Loads all PSI reference distributions once at application startup.
This replaces the per-request load_reference_distribution() calls
that read all Parquet files on every /predict or /fields request.

Usage (in serving/app.py lifespan):

    from monitoring.reference_cache import load_all_reference_distributions
    reference_cache = load_all_reference_distributions()
    # pass reference_cache to evaluate_drift() calls
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MONITORED_FEATURES = [
    "rainfall_today_mm",
    "t2m_max_today",
    "t2m_min_today",
    "et0_today",
    "solar_radiation_today",
    "gdd_accumulation",
    "ndvi_latest",
]

NASA_COL_MAP = {
    "rainfall_today_mm": "PRECTOTCORR",
    "t2m_max_today": "T2M_MAX",
    "t2m_min_today": "T2M_MIN",
    "et0_today": "EVPTRNS",
    "solar_radiation_today": "ALLSKY_SFC_SW_DWN",
}


def load_all_reference_distributions(
    data_root: str = "data/raw/nasa_power",
) -> dict[str, np.ndarray]:
    """
    Load reference distributions for all monitored features in one pass.

    Reads each Parquet file only once and builds arrays for every feature
    simultaneously. Called once at startup — result is held in memory for
    the lifetime of the process.

    Returns:
        Dict mapping feature name → np.ndarray of reference values.
        Missing or unmapped features return empty arrays.
    """
    root = Path(data_root)
    all_files = list(root.glob("*/[0-9][0-9][0-9][0-9].parquet"))

    if not all_files:
        log.warning(
            "No NASA POWER Parquet files found at %s — PSI drift will be skipped.",
            data_root,
        )
        return {feature: np.array([]) for feature in MONITORED_FEATURES}

    # Identify which columns we actually need
    needed_cols = set(NASA_COL_MAP.values())

    all_dfs: list[pd.DataFrame] = []
    for filepath in all_files:
        try:
            df = pd.read_parquet(filepath, columns=list(needed_cols))
            all_dfs.append(df)
        except Exception as exc:
            log.debug("Skipping %s: %s", filepath, exc)
            continue

    cache: dict[str, np.ndarray] = {}

    if not all_dfs:
        log.warning("All Parquet files failed to load — returning empty reference cache.")
        return {feature: np.array([]) for feature in MONITORED_FEATURES}

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Reference cache: loaded %d rows from %d files.", len(combined), len(all_dfs))

    for feature in MONITORED_FEATURES:
        col = NASA_COL_MAP.get(feature)
        if col is None or col not in combined.columns:
            cache[feature] = np.array([])
            log.debug("Feature %s has no NASA column mapping — skipping.", feature)
        else:
            cache[feature] = combined[col].dropna().values
            log.debug("Feature %s: %d reference samples loaded.", feature, len(cache[feature]))

    return cache
