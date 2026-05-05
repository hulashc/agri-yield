from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from training.utils.features import FEATURE_COLS

log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "agri-yield-xgb")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
CI_WIDTH = float(os.getenv("CI_WIDTH", "0.15"))
MODEL_CACHE_PATH = "/tmp/mlflow_model_cache"

_model = None
_model_version: str = "not_loaded"
_model_feature_cols: list[str] = []


def load_model() -> bool:
    global _model, _model_version, _model_feature_cols
    try:
        Path(MODEL_CACHE_PATH).mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        _model_version = version.version

        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        _model = mlflow.xgboost.load_model(model_uri, dst_path=MODEL_CACHE_PATH)

        # Derive the exact feature list the model was trained on
        booster = _model.get_booster()
        _model_feature_cols = booster.feature_names or FEATURE_COLS
        log.info(
            "Loaded model '%s' @%s (version %s) — %d features: %s",
            REGISTERED_MODEL_NAME, MODEL_ALIAS, _model_version,
            len(_model_feature_cols), _model_feature_cols,
        )
        return True
    except Exception as exc:
        log.warning(
            "Could not load model '%s' @%s: %s — "
            "API will start but /predict returns 503 until a model is available.",
            REGISTERED_MODEL_NAME, MODEL_ALIAS, exc,
        )
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

    # Use the exact feature columns the loaded model was trained on
    cols = _model_feature_cols if _model_feature_cols else FEATURE_COLS
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    df = df[cols]

    preds = model.predict(df)
    lower = preds * (1 - CI_WIDTH)
    upper = preds * (1 + CI_WIDTH)
    return preds, lower, upper


def model_version() -> str:
    return str(_model_version)
