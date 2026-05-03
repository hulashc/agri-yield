from __future__ import annotations

import pandas as pd

PSI_THRESHOLD = 0.2
FEATURE_BREACH_RATIO = 0.30
RMSE_DEGRADATION_THRESHOLD = 0.10


def should_retrain_from_psi(psi_df: pd.DataFrame) -> tuple[bool, dict]:
    """
    psi_df columns expected:
    - feature_name
    - psi
    """
    if psi_df.empty:
        return False, {"reason": "no_psi_rows"}

    breached = psi_df[psi_df["psi"] > PSI_THRESHOLD]
    breach_ratio = len(breached) / len(psi_df)
    return breach_ratio > FEATURE_BREACH_RATIO, {
        "breached_feature_count": int(len(breached)),
        "total_feature_count": int(len(psi_df)),
        "breach_ratio": float(breach_ratio),
        "psi_threshold": PSI_THRESHOLD,
    }


def should_retrain_from_rmse(
    current_rmse: float, reference_rmse: float
) -> tuple[bool, dict]:
    if reference_rmse == 0:
        return False, {"reason": "reference_rmse_zero"}
    degradation = (current_rmse - reference_rmse) / reference_rmse
    return degradation > RMSE_DEGRADATION_THRESHOLD, {
        "current_rmse": float(current_rmse),
        "reference_rmse": float(reference_rmse),
        "degradation_ratio": float(degradation),
        "rmse_threshold": RMSE_DEGRADATION_THRESHOLD,
    }


def classify_drift(
    rolling_psi_breach: bool,
    season_psi_breach: bool,
    season_rmse_breach: bool,
) -> str:
    if season_rmse_breach:
        return "genuine_degradation"
    if season_psi_breach:
        return "seasonal_abnormality"
    if rolling_psi_breach and not season_psi_breach:
        return "short_term_shift_or_expected_seasonality"
    return "stable"
