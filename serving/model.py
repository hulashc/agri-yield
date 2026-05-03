from __future__ import annotations

import os

import mlflow.xgboost
import numpy as np
import pandas as pd

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
    _model = mlflow.xgboost.load_model(f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}")


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")
    return _model


# Exact column order the model was trained on — must match train.py
FEATURE_COLS = [
    "crop_type",
    "soil_temp_mean",
    "soil_temp_std",
    "moisture_mean",
    "moisture_std",
    "ph_mean",
    "nitrogen_mean",
    "phosphorus_mean",
    "potassium_mean",
    "air_temp_mean",
    "precip_total",
    "humidity_mean",
    "wind_speed_mean",
    "latest_ndvi",
    "cloud_cover_pct",
    "ndvi_interpolated",
    "ndvi_proxied",
]

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
