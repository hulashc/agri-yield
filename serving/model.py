from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from training.utils.features import FEATURE_COLS

log = logging.getLogger(__name__)

# MLflow config (used only if MLflow is reachable)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "agri-yield-xgb")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
CI_WIDTH = float(os.getenv("CI_WIDTH", "0.15"))
MODEL_CACHE_PATH = "/tmp/mlflow_model_cache"

# Pickle fallback — baked into Docker image by CI
PICKLE_MODEL_PATH = os.getenv("PICKLE_MODEL_PATH", "model.pkl")

_model = None
_model_version: str = "not_loaded"


def _load_from_mlflow() -> bool:
    global _model, _model_version
    # Skip MLflow entirely if URI is set to 'disabled'
    if MLFLOW_TRACKING_URI == "disabled":
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
        _model = mlflow.xgboost.load_model(model_uri, dst_path=MODEL_CACHE_PATH)
        log.info("Loaded model from MLflow v%s", _model_version)
        return True
    except Exception as exc:
        log.info("MLflow unavailable (%s) — trying pickle fallback.", exc)
        return False


def _load_from_pickle() -> bool:
    global _model, _model_version
    try:
        with open(PICKLE_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _model_version = "pkl-ci"
        log.info("Loaded model from pickle: %s", PICKLE_MODEL_PATH)
        return True
    except Exception as exc:
        log.warning("Could not load pickle model from %s: %s", PICKLE_MODEL_PATH, exc)
        return False


def load_model() -> bool:
    """Try MLflow first (local dev), fall back to pickle (production)."""
    if _load_from_mlflow():
        return True
    if _load_from_pickle():
        return True
    log.warning("No model available — /predict will return 503.")
    return False


def is_loaded() -> bool:
    return _model is not None


def get_model():
    if _model is None:
        raise RuntimeError("model_not_ready")
    return _model


NON_FEATURE_COLS = ["field_id", "event_timestamp"]


def predict(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = get_model()
    df = feature_df.drop(columns=[c for c in NON_FEATURE_COLS if c in feature_df.columns])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]
    preds = model.predict(df)
    lower = preds * (1 - CI_WIDTH)
    upper = preds * (1 + CI_WIDTH)
    return preds, lower, upper


def model_version() -> str:
    return str(_model_version)
