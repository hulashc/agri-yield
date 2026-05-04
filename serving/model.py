from __future__ import annotations

import os

import mlflow.xgboost
import numpy as np
import pandas as pd

# Import the single source-of-truth feature list from training
from training.utils.features import FEATURE_COLS

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "agri-yield-xgb")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
CI_WIDTH = float(os.getenv("CI_WIDTH", "0.15"))  # ±15% confidence interval

_model = None
_model_version: str = "unknown"


def load_model() -> None:
    global _model, _model_version

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    prod = [v for v in versions if v.current_stage == MODEL_STAGE]
    if not prod:
        raise RuntimeError(f"No {MODEL_STAGE} model found for {REGISTERED_MODEL_NAME}")
    latest = sorted(prod, key=lambda v: int(v.version))[-1]
    _model_version = latest.version

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    _model = mlflow.xgboost.load_model(model_uri)


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")
    return _model


NON_FEATURE_COLS = ["field_id", "event_timestamp"]


def predict(
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = get_model()

    # Drop entity columns, cast to numeric
    df = feature_df.drop(
        columns=[c for c in NON_FEATURE_COLS if c in feature_df.columns]
    )
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Add missing columns with default 0, then select in exact training order
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]

    preds = model.predict(df)
    lower = preds * (1 - CI_WIDTH)
    upper = preds * (1 + CI_WIDTH)
    return preds, lower, upper


def model_version() -> str:
    return _model_version
