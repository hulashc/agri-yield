from __future__ import annotations

import logging
import os

import mlflow.xgboost
import numpy as np
import pandas as pd

from training.utils.features import FEATURE_COLS

log = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "agri-yield-xgb")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
CI_WIDTH = float(os.getenv("CI_WIDTH", "0.15"))

_model = None
_model_version: str = "not_loaded"


def load_model() -> bool:
    """
    Attempt to load the Production model from MLflow.
    Returns True if successful, False if no model exists yet.
    Does NOT raise — allows the API to start without a trained model.
    """
    global _model, _model_version
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        prod = [v for v in versions if v.current_stage == MODEL_STAGE]
        if not prod:
            log.warning(
                "No '%s' model in stage '%s'. "
                "Run training/train.py then training/promote.py to register one. "
                "API will start but /predict will return 503 until a model is available.",
                REGISTERED_MODEL_NAME,
                MODEL_STAGE,
            )
            return False
        latest = sorted(prod, key=lambda v: int(v.version))[-1]
        _model_version = latest.version
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        _model = mlflow.xgboost.load_model(model_uri)
        log.info("Loaded model '%s' version %s from MLflow.", REGISTERED_MODEL_NAME, _model_version)
        return True
    except Exception as exc:
        log.warning("Could not load model: %s. API starting without a model.", exc)
        return False


def is_loaded() -> bool:
    return _model is not None


def get_model():
    if _model is None:
        raise RuntimeError("model_not_ready")
    return _model


NON_FEATURE_COLS = ["field_id", "event_timestamp"]


def predict(
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
