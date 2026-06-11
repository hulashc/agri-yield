from __future__ import annotations

import logging
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from training.utils.features import FEATURE_COLS

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "disabled")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "agri-yield-xgb")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
# CI_WIDTH is used ONLY as a last-resort fallback when no bundle is available.
# Real prediction intervals come from quantile models in the bundle.
CI_WIDTH = float(os.getenv("CI_WIDTH", "0.15"))
MODEL_CACHE_PATH = "/tmp/mlflow_model_cache"

_REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = os.getenv("BUNDLE_MODEL_PATH", str(_REPO_ROOT / "model_bundle.pkl"))
PICKLE_MODEL_PATH = os.getenv("PICKLE_MODEL_PATH", str(_REPO_ROOT / "model.pkl"))

# ── Module-level state ────────────────────────────────────────────────────────
_model = None          # mean XGBRegressor (always set when loaded)
_lower_model = None    # 10th-percentile quantile model
_upper_model = None    # 90th-percentile quantile model
_model_version: str = "not_loaded"
_bundle_meta: dict = {}
_using_bundle: bool = False


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_bundle(path: str) -> bool:
    """Load mean + lower + upper models from model_bundle.pkl."""
    global _model, _lower_model, _upper_model, _model_version, _bundle_meta, _using_bundle
    p = Path(path)
    if not p.exists():
        log.info("Bundle not found at %s", p)
        return False
    try:
        with open(p, "rb") as f:
            bundle = pickle.load(f)  # noqa: S301
        _model = bundle["mean_model"]
        _lower_model = bundle.get("lower_model")
        _upper_model = bundle.get("upper_model")
        # Strip models from meta, keep scalar metadata only
        _bundle_meta = {
            k: v for k, v in bundle.items()
            if k not in ("mean_model", "lower_model", "upper_model")
        }
        _using_bundle = True
        _model_version = _bundle_meta.get("model_version", "bundle-ci")
        log.info(
            "Loaded model bundle — CI coverage %.1f%%, avg width %.0f kg/ha",
            bundle.get("coverage_80pct", 0) * 100,
            bundle.get("avg_interval_width_kg_ha", 0),
        )
        return True
    except Exception as exc:
        log.warning("Could not load bundle: %s", exc)
        return False


def _load_from_mlflow() -> bool:
    """Try MLflow registry. Returns False immediately if MLFLOW_TRACKING_URI=disabled."""
    global _model, _model_version, _using_bundle
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
        _model = mlflow.xgboost.load_model(model_uri, dst_path=MODEL_CACHE_PATH)
        _using_bundle = False
        log.info("Loaded mean model from MLflow v%s (no quantile bundle)", _model_version)
        return True
    except Exception as exc:
        log.info("MLflow unavailable (%s) — falling back.", exc)
        return False


def _load_from_pickle() -> bool:
    """Load model.pkl baked into the Docker image by CI (mean model only)."""
    global _model, _model_version, _using_bundle
    pkl_path = Path(PICKLE_MODEL_PATH)
    log.info("Trying pickle at: %s (exists=%s)", pkl_path, pkl_path.exists())
    try:
        with open(pkl_path, "rb") as f:
            _model = pickle.load(f)  # noqa: S301
        _model_version = "pkl-ci"
        _using_bundle = False
        log.info("Loaded mean model from pickle: %s", pkl_path)
        return True
    except Exception as exc:
        log.warning("Could not load pickle from %s: %s", pkl_path, exc)
        return False


def load_model() -> bool:
    """
    Load priority:
    1. model_bundle.pkl  — mean + quantile models (best: real CIs)
    2. MLflow registry   — mean model only (docker-compose local)
    3. model.pkl         — mean model only (CI fallback)
    """
    if _load_bundle(BUNDLE_PATH):
        return True
    if _load_from_mlflow():
        return True
    if _load_from_pickle():
        return True
    log.warning("No model available — /predict will return 503.")
    return False


def is_loaded() -> bool:
    return _model is not None


def using_bundle() -> bool:
    """True when model_bundle.pkl was successfully loaded (real quantile CIs active)."""
    return _using_bundle


def has_quantile_models() -> bool:
    """True when lower/upper quantile models are available."""
    return _lower_model is not None and _upper_model is not None


def get_model():
    if _model is None:
        raise RuntimeError("model_not_ready")
    return _model


def get_bundle_meta() -> dict:
    """Return metadata saved alongside the model bundle."""
    return _bundle_meta


NON_FEATURE_COLS = ["field_id", "event_timestamp"]


def _prepare_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Shared feature preparation for all three model calls."""
    df = feature_df.drop(columns=[c for c in NON_FEATURE_COLS if c in feature_df.columns])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLS]


def predict(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean_preds, lower_preds, upper_preds).

    When quantile models are loaded (model_bundle.pkl), lower/upper come from
    the 10th and 90th percentile XGBoost quantile regressors — statistically
    derived prediction intervals.

    When only the mean model is available (MLflow or plain pickle fallback),
    lower/upper fall back to mean ± CI_WIDTH fraction (legacy behaviour).
    """
    df = _prepare_features(feature_df)
    preds = _model.predict(df)

    if has_quantile_models():
        lower = _lower_model.predict(df)
        upper = _upper_model.predict(df)
        # Enforce monotonicity: quantile crossing correction
        lower = np.minimum(lower, preds)
        upper = np.maximum(upper, preds)
    else:
        log.debug("Quantile models not loaded — using CI_WIDTH=%.2f fallback", CI_WIDTH)
        lower = preds * (1 - CI_WIDTH)
        upper = preds * (1 + CI_WIDTH)

    return preds, lower, upper


def model_version() -> str:
    return str(_model_version)
