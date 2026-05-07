"""
serving/app.py
FastAPI prediction endpoint for agri-yield.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

log = logging.getLogger(__name__)

# Support absolute path via env var (set in Dockerfile.prod) or fall back to relative
FIELDS_CSV_PATH = os.getenv("FIELDS_CSV_PATH", "data/seed/uk_fields.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and field metadata at startup. Failures are non-fatal."""
    global _FIELDS_DF
    try:
        _FIELDS_DF = pd.read_csv(FIELDS_CSV_PATH).set_index("field_id")
        log.info("Loaded %d fields from %s", len(_FIELDS_DF), FIELDS_CSV_PATH)
    except FileNotFoundError:
        log.warning("%s not found — /predict will return 503 until data is seeded.", FIELDS_CSV_PATH)
        _FIELDS_DF = pd.DataFrame()

    load_model()
    yield


app = FastAPI(title="Agri Yield API", version="0.1.0", lifespan=lifespan)

# CORS — allow requests from local dev, GitHub Pages, and any Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "https://hulashc.github.io",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(metrics_router)

_FIELDS_DF: pd.DataFrame = pd.DataFrame()


def get_field_meta(field_id: str) -> dict:  # type: ignore[type-arg]
    if _FIELDS_DF.empty or field_id not in _FIELDS_DF.index:
        raise HTTPException(status_code=404, detail=f"Unknown field_id: {field_id}")
    return _FIELDS_DF.loc[field_id].to_dict()


@app.get("/health")
def health() -> dict:  # type: ignore[type-arg]
    return {
        "status": "ok",
        "model_loaded": model_module.is_loaded(),
        "model_version": model_module.model_version(),
        "fields_loaded": not _FIELDS_DF.empty,
    }


@app.post("/predict")
async def predict(request: PredictRequest) -> dict:  # type: ignore[type-arg]
    if not model_module.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="No Production model available. Run training/train.py then training/promote.py first.",
        )

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
