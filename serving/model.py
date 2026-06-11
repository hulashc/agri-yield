from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from training.utils.features import FEATURE_COLS

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "disabled")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "agri-yield-xgb")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_CACHE_PATH = "/tmp/mlflow_model_cache"

_REPO_ROOT = Path(__file__).resolve().parent.parent
PICKLE_BUNDLE_PATH = os.getenv("PICKLE_BUNDLE_PATH", str(_REPO_ROOT / "model_bundle.pkl"))
PICKLE_MODEL_PATH = os.getenv("PICKLE_MODEL_PATH", str(_REPO_ROOT / "model.pkl"))

# Bundle fields (populated by load_model)
_mean_model = None
_lower_model = None
_upper_model = None
_bundle_meta: dict = {}
_model_version: str = "not_loaded"


def _load_from_mlflow() -> bool:
    global _mean_model, _model_version
    if MLFLOW_TRACKING_URI == "disabled":
        log.info("MLflow disabled via env — skipping.")
        return False
    try:
        import mlflow
        import mlflow.xgboost

        Path(MODEL_CACHE_PATH).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        _model_version = version.version
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        _mean_model = mlflow.xgboost.load_model(model_uri, dst_path=MODEL_CACHE_PATH)
        log.info("Loaded mean model from MLflow v%s", _model_version)
        return True
    except Exception as exc:
        log.info("MLflow unavailable (%s) — falling back to bundle.", exc)
        return False


def _load_from_bundle() -> bool:
    """Load model_bundle.pkl produced by updated train_and_export.py."""
    global _mean_model, _lower_model, _upper_model, _bundle_meta, _model_version
    bundle_path = Path(PICKLE_BUNDLE_PATH)
    log.info("Trying bundle at: %s (exists=%s)", bundle_path, bundle_path.exists())
    try:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)  # noqa: S301
        _mean_model = bundle["mean"]
        _lower_model = bundle.get("lower")
        _upper_model = bundle.get("upper")
        _bundle_meta = {
            k: v for k, v in bundle.items() if k not in ("mean", "lower", "upper")
        }
        _model_version = f"bundle-{_bundle_meta.get('trained_at', 'unknown')[:10]}"
        log.info(
            "Loaded model bundle (RMSE=%.1f, source=%s, trained=%s)",
            _bundle_meta.get("rmse", 0),
            _bundle_meta.get("dataset_source", "?"),
            _bundle_meta.get("trained_at", "?")[:10],
        )
        return True
    except Exception as exc:
        log.warning("Could not load bundle from %s: %s", bundle_path, exc)
        return False


def _load_from_legacy_pickle() -> bool:
    """Fall back to plain model.pkl (mean model only) for backward compat."""
    global _mean_model, _model_version
    pkl_path = Path(PICKLE_MODEL_PATH)
    log.info("Trying legacy pickle at: %s (exists=%s)", pkl_path, pkl_path.exists())
    try:
        with open(pkl_path, "rb") as f:
            _mean_model = pickle.load(f)  # noqa: S301
        _model_version = "pkl-legacy"
        log.info("Loaded legacy pickle: %s", pkl_path)
        return True
    except Exception as exc:
        log.warning("Could not load legacy pickle from %s: %s", pkl_path, exc)
        return False


def load_model() -> bool:
    """Try MLflow → bundle → legacy pickle."""
    if _load_from_mlflow():
        return True
    if _load_from_bundle():
        return True
    if _load_from_legacy_pickle():
        return True
    log.warning("No model available — /predict will return 503.")
    return False


def is_loaded() -> bool:
    return _mean_model is not None


def get_model():
    if _mean_model is None:
        raise RuntimeError("model_not_ready")
    return _mean_model


def get_bundle_meta() -> dict:
    """Return training metadata for /model/info endpoint."""
    return {
        **_bundle_meta,
        "model_version": _model_version,
        "has_quantile_ci": _lower_model is not None and _upper_model is not None,
        "feature_cols": FEATURE_COLS,
    }


def get_feature_importance() -> dict:
    """Return feature importance from bundle or from live model if available."""
    if _bundle_meta.get("feature_importance"):
        return _bundle_meta["feature_importance"]
    if _mean_model is not None and hasattr(_mean_model, "feature_importances_"):
        return {
            feat: float(score)
            for feat, score in zip(FEATURE_COLS, _mean_model.feature_importances_, strict=False)
        }
    return {}


NON_FEATURE_COLS = ["field_id", "event_timestamp"]


def _prep_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.drop(columns=[c for c in NON_FEATURE_COLS if c in feature_df.columns])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLS]


def predict(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean_preds, lower_preds, upper_preds).

    If quantile models are available (model_bundle.pkl), CI bounds are
    statistically derived from q0.10 and q0.90 quantile regressors.
    If only the legacy mean model is loaded, falls back to a ±15% heuristic.

    Ordering is enforced: lower <= mean <= upper per row.
    """
    df = _prep_features(feature_df)
    preds = _mean_model.predict(df)

    if _lower_model is not None and _upper_model is not None:
        lower = _lower_model.predict(df)
        upper = _upper_model.predict(df)
    else:
        # Legacy fallback only — no quantile models loaded
        ci_width = float(os.getenv("CI_WIDTH", "0.15"))
        lower = preds * (1 - ci_width)
        upper = preds * (1 + ci_width)

    # Enforce ordering: lower <= mean <= upper
    lower = np.minimum(lower, preds)
    upper = np.maximum(upper, preds)

    return preds, lower, upper


def model_version() -> str:
    return str(_model_version)
