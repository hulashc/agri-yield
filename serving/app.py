from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from serving.feast_client import fetch_online_features
from serving.health import run_health_checks
from serving.logging_sink import log_prediction
from serving.model import load_model, model_version, predict
from serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="agri-yield inference service", lifespan=lifespan)

# instrument() must be called before the app starts — outside lifespan
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    checks = run_health_checks()
    if not checks["healthy"]:
        raise HTTPException(status_code=503, detail=checks)
    return checks


@app.post("/predict", response_model=PredictResponse)
def predict_single(req: PredictRequest):
    features = fetch_online_features([req.field_id])
    preds, lower, upper = predict(features)
    result = PredictResponse(
        field_id=req.field_id,
        predicted_yield_kg_per_ha=float(preds[0]),
        lower_bound=float(lower[0]),
        upper_bound=float(upper[0]),
        model_version=str(model_version()),
    )
    log_prediction(
        req.field_id,
        result.predicted_yield_kg_per_ha,
        result.lower_bound,
        result.upper_bound,
        result.model_version,
    )
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    field_ids = [r.field_id for r in req.requests]
    features = fetch_online_features(field_ids)
    preds, lower, upper = predict(features)
    predictions = [
        PredictResponse(
            field_id=fid,
            predicted_yield_kg_per_ha=float(p),
            lower_bound=float(lb),
            upper_bound=float(u),
            model_version=str(model_version()),
        )
        for fid, p, lb, u in zip(field_ids, preds, lower, upper)
    ]
    for pr in predictions:
        log_prediction(
            pr.field_id,
            pr.predicted_yield_kg_per_ha,
            pr.lower_bound,
            pr.upper_bound,
            pr.model_version,
        )
    return BatchPredictResponse(predictions=predictions)
