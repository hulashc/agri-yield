"""
serving/app.py
FastAPI prediction endpoint for agri-yield.
"""

import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException

from ingestion.openmeteo_live import get_live_features
from monitoring.prometheus_metrics import (
    PREDICTION_CI_WIDTH,
    PREDICTION_LATENCY,
    PREDICTIONS_TOTAL,
    PREDICTION_YIELD_KG_HA,
    STALE_FEATURE_REQUESTS,
)
from monitoring.psi_detector import evaluate_drift
from serving.metrics import metrics_router
import serving.model as model_module
from serving.model import load_model
from serving.schemas import PredictRequest

app = FastAPI(title="Agri Yield API", version="0.1.0")
app.include_router(metrics_router)

# Load field metadata once at startup
_FIELDS_DF = pd.read_csv("data/seed/uk_fields.csv").set_index("field_id")


def get_field_meta(field_id: str) -> dict:  # type: ignore[type-arg]
    if field_id not in _FIELDS_DF.index:
        raise HTTPException(status_code=404, detail=f"Unknown field_id: {field_id}")
    return _FIELDS_DF.loc[field_id].to_dict()


load_model()


@app.get("/health")
def health() -> dict:  # type: ignore[type-arg]
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: PredictRequest) -> dict:  # type: ignore[type-arg]
    start = time.time()

    field_meta = get_field_meta(request.field_id)
    live = get_live_features(
        request.field_id,
        float(field_meta["lat"]),
        float(field_meta["lon"]),
    )

    if live.get("stale_features"):
        STALE_FEATURE_REQUESTS.labels(field_id=request.field_id).inc()

    features = {**field_meta, **live}

    drift_result = evaluate_drift(request.field_id, features)
    feat_df = pd.DataFrame([features])
    preds, lower, upper = model_module.predict(feat_df)
    prediction = {
        "mean": float(preds[0]),
        "lower": float(lower[0]),
        "upper": float(upper[0]),
    }

    latency = time.time() - start
    PREDICTION_LATENCY.observe(latency)
    PREDICTIONS_TOTAL.labels(
        crop_type=str(field_meta["crop_type"]),
        region=str(field_meta["region"]),
    ).inc()
    PREDICTION_YIELD_KG_HA.labels(crop_type=str(field_meta["crop_type"])).observe(
        prediction["mean"]
    )
    PREDICTION_CI_WIDTH.observe(prediction["upper"] - prediction["lower"])

    return {
        "field_id": request.field_id,
        "predicted_yield_kg_ha": prediction["mean"],
        "lower_bound": prediction["lower"],
        "upper_bound": prediction["upper"],
        "confidence_level": 0.80,
        "drift_warning": drift_result["drift_warning"],
        "drift_level": drift_result["drift_level"],
        "psi_score": drift_result["max_psi"],
        "stale_features": live.get("stale_features", False),
        "model_version": model_module.model_version(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
