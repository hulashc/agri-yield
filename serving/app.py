"""
serving/app.py
FastAPI prediction endpoint for agri-yield.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import serving.model as model_module
from ingestion.openmeteo_live import get_live_features
from monitoring.prometheus_metrics import (
    PREDICTION_CI_WIDTH,
    PREDICTION_LATENCY,
    PREDICTION_YIELD_KG_HA,
    PREDICTIONS_TOTAL,
    STALE_FEATURE_REQUESTS,
)
from monitoring.psi_detector import evaluate_drift
from monitoring.reference_cache import load_all_reference_distributions
from serving.explainability import get_global_importance, get_local_explanation
from serving.metrics import metrics_router
from serving.model import load_model
from serving.schemas import ExplainRequest, ExplainResponse, ModelInfoResponse, PredictRequest
from serving.version import BUILD_VERSION

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FIELDS_PATH = str(_REPO_ROOT / "data" / "seed" / "uk_fields.csv")
FIELDS_CSV_PATH = os.getenv("FIELDS_CSV_PATH", _DEFAULT_FIELDS_PATH)

# Semaphore limits concurrent Open-Meteo calls on Render's free 1-CPU instance.
_PREDICT_SEMAPHORE = asyncio.Semaphore(2)

_FIELDS_DF: pd.DataFrame = pd.DataFrame()

_STATIC_DIR = Path(__file__).resolve().parent / "static"


async def _startup_load():
    """Load CSV + model + reference cache in the background so the port binds immediately."""
    global _FIELDS_DF
    try:
        log.info("Loading fields from: %s", FIELDS_CSV_PATH)
        _FIELDS_DF = pd.read_csv(FIELDS_CSV_PATH).set_index("field_id")
        log.info("Loaded %d fields", len(_FIELDS_DF))
    except FileNotFoundError:
        log.warning("%s not found.", FIELDS_CSV_PATH)
        _FIELDS_DF = pd.DataFrame()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup_load()
    # Load all NASA POWER reference distributions once — avoids per-request Parquet scan.
    loop = asyncio.get_running_loop()
    app.state.reference_cache = await loop.run_in_executor(
        None, load_all_reference_distributions
    )
    log.info(
        "Reference cache loaded: %d features",
        sum(1 for v in app.state.reference_cache.values() if len(v) > 0),
    )
    yield


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "https://hulashc.github.io",
]
_extra_origin = os.getenv("EXTRA_CORS_ORIGIN", "").strip()
if _extra_origin:
    _ALLOWED_ORIGINS.append(_extra_origin)

app = FastAPI(title="Agri Yield API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(metrics_router)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def get_field_meta(field_id: str) -> dict:
    if _FIELDS_DF.empty or field_id not in _FIELDS_DF.index:
        raise HTTPException(status_code=404, detail=f"Unknown field_id: {field_id}")
    return _FIELDS_DF.loc[field_id].to_dict()


async def _predict_one(field_id: str, row: pd.Series, reference_cache: dict) -> dict:
    """Run a single field prediction — used by bulk /fields endpoint."""
    async with _PREDICT_SEMAPHORE:
        try:
            loop = asyncio.get_running_loop()
            live = await loop.run_in_executor(
                None,
                get_live_features,
                field_id,
                float(row["lat"]),
                float(row["lon"]),
            )
            features = {**row.to_dict(), **live}
            drift_result = evaluate_drift(field_id, features, reference_cache=reference_cache)
            feat_df = pd.DataFrame([features])
            preds, lower, upper = model_module.predict(feat_df)
            return {
                "field_id": field_id,
                "name": str(row.get("name", field_id)),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "crop_type": str(row.get("crop_type", "")),
                "region": str(row.get("region", "")),
                "area_ha": round(float(row.get("area_ha", 0)), 1),
                "soil_type": str(row.get("soil_type", "")),
                "predicted_yield_kg_ha": round(float(preds[0]), 1),
                "lower_bound": round(float(lower[0]), 1),
                "upper_bound": round(float(upper[0]), 1),
                "confidence_level": 0.80,
                "confidence_method": (
                    "xgboost_quantile_regression"
                    if model_module.using_bundle()
                    else "legacy_symmetric_ci"
                ),
                "drift_warning": drift_result["drift_warning"],
                "drift_level": drift_result["drift_level"],
                "stale_features": live.get("stale_features", False),
                "temp_c": live.get("t2m_max_today"),
                "rainfall_mm": live.get("rainfall_today_mm"),
                "solar_rad_mj_m2": live.get("solar_radiation_today"),
                "error": None,
            }
        except Exception as exc:
            log.warning("Prediction failed for %s: %s", field_id, exc)
            return {
                "field_id": field_id,
                "name": str(row.get("name", field_id)),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "crop_type": str(row.get("crop_type", "")),
                "region": str(row.get("region", "")),
                "area_ha": round(float(row.get("area_ha", 0)), 1),
                "soil_type": str(row.get("soil_type", "")),
                "predicted_yield_kg_ha": None,
                "lower_bound": None,
                "upper_bound": None,
                "confidence_level": None,
                "confidence_method": None,
                "drift_warning": False,
                "drift_level": "none",
                "stale_features": False,
                "temp_c": None,
                "rainfall_mm": None,
                "solar_rad_mj_m2": None,
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/fields")
async def bulk_fields(request: Request):
    """Return all fields with live predictions — powers the map UI."""
    if _FIELDS_DF.empty:
        raise HTTPException(
            status_code=503,
            detail="Fields not loaded yet — please retry in a moment.",
        )
    if not model_module.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet — please retry in a moment.",
        )

    reference_cache = getattr(request.app.state, "reference_cache", {})
    tasks = [
        _predict_one(fid, row, reference_cache)
        for fid, row in _FIELDS_DF.iterrows()
    ]
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=90.0)
    except TimeoutError:
        log.warning("/fields timed out after 90s")
        raise HTTPException(status_code=503, detail="Prediction timed out — please retry.")
    return {"fields": list(results), "model_version": model_module.model_version()}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """
    Return model card: training metadata, CI method, feature importance.
    Surfaces the quantile coverage, interval width, and top predictive features
    so judges and stakeholders can understand the model without code access.
    """
    if not model_module.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    meta = model_module.get_bundle_meta()
    return ModelInfoResponse(
        model_version=meta.get("model_version", model_module.model_version()),
        confidence_method=(
            "xgboost_quantile_regression"
            if model_module.using_bundle()
            else "legacy_symmetric_ci"
        ),
        using_bundle=model_module.using_bundle(),
        trained_at=meta.get("trained_at"),
        rmse_kg_ha=meta.get("holdout_rmse"),
        mae_kg_ha=meta.get("holdout_mae"),
        pi_coverage_pct=(
            round(meta["coverage_80pct"] * 100, 1)
            if meta.get("coverage_80pct") is not None
            else None
        ),
        interval_label=(
            f"Q{int(meta['quantile_lower'] * 100)}–Q{int(meta['quantile_upper'] * 100)}"
            if meta.get("quantile_lower") is not None
            else None
        ),
        split_policy=meta.get("split_policy", "temporal_train_test_split"),
        n_train=meta.get("n_train"),
        n_test=meta.get("n_test"),
        top_features=meta.get("top_features", get_global_importance(top_n=10)),
        warning=(
            None
            if model_module.using_bundle()
            else "Bundle not loaded — using legacy CI. Re-train to generate model_bundle.pkl."
        ),
    )


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(body: ExplainRequest, request: Request):
    """
    Return a local SHAP explanation for a single field prediction.

    Given a field_id, fetches live weather features, runs the mean model,
    and returns the top SHAP contributors that drove the prediction up or down
    from the model's baseline expectation.

    Use this endpoint to power explainability panels in the dashboard
    or to answer 'why is this field predicted to yield X kg/ha?'
    """
    if not model_module.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    field_meta = get_field_meta(body.field_id)

    async with _PREDICT_SEMAPHORE:
        loop = asyncio.get_running_loop()
        live = await loop.run_in_executor(
            None,
            get_live_features,
            body.field_id,
            float(field_meta["lat"]),
            float(field_meta["lon"]),
        )

    features = {**field_meta, **live}
    feat_df = pd.DataFrame([features])

    explanation = get_local_explanation(feat_df, top_n=body.top_n)

    return ExplainResponse(
        field_id=body.field_id,
        prediction=explanation.get("prediction", 0.0),
        base_value=explanation.get("base_value"),
        top_contributors=explanation.get("top_contributors", []),
        source=explanation.get("source", "unknown"),
        error=explanation.get("error"),
        model_version=model_module.model_version(),
        last_updated=datetime.now(UTC).isoformat(),
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def map_ui():
    """Serve the frontend SPA from serving/static/index.html."""
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="Frontend not built.")
    return FileResponse(str(index))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "build_version": BUILD_VERSION,
        "model_loaded": model_module.is_loaded(),
        "model_version": model_module.model_version(),
        "using_bundle": model_module.using_bundle(),
        "has_quantile_models": model_module.has_quantile_models(),
        "fields_loaded": not _FIELDS_DF.empty,
    }


@app.post("/predict")
async def predict(body: PredictRequest, request: Request) -> dict:
    if not model_module.is_loaded():
        raise HTTPException(
            status_code=503, detail="Model not loaded yet — please retry in a moment."
        )

    start = time.time()
    field_meta = get_field_meta(body.field_id)

    async with _PREDICT_SEMAPHORE:
        loop = asyncio.get_running_loop()
        live = await loop.run_in_executor(
            None,
            get_live_features,
            body.field_id,
            float(field_meta["lat"]),
            float(field_meta["lon"]),
        )

    if live.get("stale_features"):
        STALE_FEATURE_REQUESTS.labels(field_id=body.field_id).inc()

    features = {**field_meta, **live}
    reference_cache = getattr(request.app.state, "reference_cache", {})
    drift_result = evaluate_drift(body.field_id, features, reference_cache=reference_cache)
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
        "field_id": body.field_id,
        "predicted_yield_kg_ha": prediction["mean"],
        "lower_bound": prediction["lower"],
        "upper_bound": prediction["upper"],
        "confidence_level": 0.80,
        "confidence_method": (
            "xgboost_quantile_regression"
            if model_module.using_bundle()
            else "legacy_symmetric_ci"
        ),
        "drift_warning": drift_result["drift_warning"],
        "drift_level": drift_result["drift_level"],
        "psi_score": drift_result["max_psi"],
        "stale_features": live.get("stale_features", False),
        "temp_c": live.get("t2m_max_today"),
        "rainfall_mm": live.get("rainfall_today_mm"),
        "solar_rad_mj_m2": live.get("solar_radiation_today"),
        "model_version": model_module.model_version(),
        "last_updated": datetime.now(UTC).isoformat(),
    }
