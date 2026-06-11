"""
serving/model.py

Model loading + prediction for the agri-yield serving layer.

Load priority:
  1. MLflow registry (only if MLFLOW_TRACKING_URI != 'disabled')
  2. model_bundle.pkl  (3-model bundle with real quantile CIs)  <-- primary
  3. model.pkl         (legacy single model, symmetric ±10% CI fallback)

Bounds from model_bundle are statistically derived (XGBoost quantile regression).
The old CI_WIDTH env var is retained for backward compat but IGNORED when a
bundle is present. A warning is logged if CI_WIDTH is set and a bundle is found.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.utils.features import FEATURE_COLS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "disabled")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "agri-yield-xgb")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_CACHE_PATH = "/tmp/mlflow_model_cache"

_REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = os.getenv("BUNDLE_MODEL_PATH", str(_REPO_ROOT / "model_bundle.pkl"))
PICKLE_MODEL_PATH = os.getenv("PICKLE_MODEL_PATH", str(_REPO_ROOT / "model.pkl"))

# Legacy CI_WIDTH — only used if no bundle is loaded
_LEGACY_CI_WIDTH = float(os.getenv("CI_WIDTH", "0.10"))

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_bundle: Any = None          # ModelBundle instance when loaded
_legacy_model: Any = None    # raw XGBRegressor when only pickle available
_model_version: str = "not_loaded"
_using_bundle: bool = False

NON_FEATURE_COLS = ["field_id", "event_timestamp"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_from_mlflow() -> bool:
    """Try MLflow registry. Returns False immediately if MLFLOW_TRACKING_URI=disabled."""
    global _bundle, _legacy_model, _model_version, _using_bundle
    if MLFLOW_TRACKING_URI == "disabled":
        log.info("MLflow disabled via env — skipping.")
        return False
    try:
        import mlflow  # noqa: PLC0415
        import mlflow.xgboost  # noqa: PLC0415

        Path(MODEL_CACHE_PATH).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        _model_version = version.version
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        _legacy_model = mlflow.xgboost.load_model(model_uri, dst_path=MODEL_CACHE_PATH)
        _using_bundle = False
        log.info("Loaded model from MLflow v%s (no bundle — legacy CI will be used)", _model_version)
        return True
    except Exception as exc:
        log.info("MLflow unavailable (%s) — falling back.", exc)
        return False


def _load_from_bundle() -> bool:
    """Load ModelBundle (3-model pack with real quantile CIs)."""
    global _bundle, _model_version, _using_bundle
    bundle_path = Path(BUNDLE_PATH)
    log.info("Trying ModelBundle at: %s (exists=%s)", bundle_path, bundle_path.exists())
    try:
        from training.model_bundle import ModelBundle  # noqa: PLC0415
        _bundle = ModelBundle.load(bundle_path)
        _model_version = "bundle-" + _bundle.training_meta.get("trained_at", "unknown")[:10]
        _using_bundle = True
        ci_width_env = os.getenv("CI_WIDTH")
        if ci_width_env:
            log.warning(
                "CI_WIDTH env var is set (%s) but will be IGNORED — "
                "prediction intervals come from the quantile bundle.",
                ci_width_env,
            )
        log.info(
            "Loaded ModelBundle: RMSE=%.1f kg/ha | PI coverage=%.1f%% | version=%s",
            _bundle.training_meta.get("rmse_kg_ha", 0),
            _bundle.training_meta.get("pi_coverage_pct", 0),
            _model_version,
        )
        return True
    except Exception as exc:
        log.info("ModelBundle unavailable (%s) — trying legacy pickle.", exc)
        return False


def _load_from_pickle() -> bool:
    """Load legacy model.pkl. CI will be symmetric ±CI_WIDTH (fallback only)."""
    global _legacy_model, _model_version, _using_bundle
    pkl_path = Path(PICKLE_MODEL_PATH)
    log.info("Trying legacy pickle at: %s (exists=%s)", pkl_path, pkl_path.exists())
    try:
        with open(pkl_path, "rb") as f:
            _legacy_model = pickle.load(f)  # noqa: S301
        _model_version = "pkl-ci"
        _using_bundle = False
        log.warning(
            "Loaded legacy model.pkl — prediction intervals are SYMMETRIC \u00b1%.0f%% "
            "(not statistically derived). Deploy model_bundle.pkl for calibrated CIs.",
            _LEGACY_CI_WIDTH * 100,
        )
        return True
    except Exception as exc:
        log.warning("Could not load pickle from %s: %s", pkl_path, exc)
        return False


def load_model() -> bool:
    """Load priority: MLflow → ModelBundle → legacy pickle."""
    if _load_from_mlflow():
        return True
    if _load_from_bundle():
        return True
    if _load_from_pickle():
        return True
    log.warning("No model available — /predict will return 503.")
    return False


# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------

def is_loaded() -> bool:
    return _bundle is not None or _legacy_model is not None


def using_bundle() -> bool:
    """True when a ModelBundle is active (real quantile CIs)."""
    return _using_bundle


def get_bundle_meta() -> dict:
    """
    Return model card dict for the /model/info endpoint.
    Safe to call even when only legacy model is loaded.
    """
    if _bundle is not None:
        meta = dict(_bundle.training_meta)
        meta["model_version"] = _model_version
        meta["confidence_method"] = meta.get("confidence_method", "xgboost_quantile_regression")
        meta["top_features"] = _bundle.top_features(n=10)
        meta["using_bundle"] = True
        return meta
    return {
        "model_version": _model_version,
        "confidence_method": "legacy_symmetric_ci",
        "ci_width_pct": _LEGACY_CI_WIDTH * 100,
        "warning": "ModelBundle not loaded — CI bounds are not statistically derived.",
        "using_bundle": False,
    }


def model_version() -> str:
    return str(_model_version)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _prep_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature cols, coerce to numeric, fill missing, reorder."""
    df = feature_df.drop(columns=[c for c in NON_FEATURE_COLS if c in feature_df.columns])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLS]


def predict(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean_preds, lower_preds, upper_preds).

    When a ModelBundle is loaded: bounds are XGBoost quantile predictions.
    When only legacy pickle is loaded: bounds are symmetric \u00b1CI_WIDTH.
    """
    df = _prep_features(feature_df)

    if _using_bundle and _bundle is not None:
        return _bundle.predict(df)

    # Legacy fallback
    if _legacy_model is not None:
        preds = _legacy_model.predict(df)
        lower = preds * (1 - _LEGACY_CI_WIDTH)
        upper = preds * (1 + _LEGACY_CI_WIDTH)
        return preds, lower, upper

    raise RuntimeError("model_not_ready")
