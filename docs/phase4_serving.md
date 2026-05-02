# Phase 4 — Serving Layer (Weeks 7–8)

## Overview

Phase 4 exposes the trained XGBoost model as a production-grade inference service.
Requests fetch online features from Feast (Redis), run inference, return predictions
with confidence intervals, and log every prediction for downstream monitoring.
The service is containerised and deployed to Kubernetes via Helm with auto-scaling.

---

## Directory Structure

Create the following inside `agri-yield/`:

```
serving/
├── __init__.py
├── app.py                  # FastAPI application
├── model.py                # Model loader + inference
├── feast_client.py         # Online feature fetching
├── schemas.py              # Pydantic request/response models
├── health.py               # /health endpoint logic
├── logging_sink.py         # Prediction logger
Dockerfile                  # Multi-stage build
helm/
└── agri-yield-serving/
    ├── Chart.yaml
    ├── values.yaml
    └── templates/
        ├── deployment.yaml
        ├── service.yaml
        ├── hpa.yaml
        └── configmap.yaml
```

---

## Part 1 — Pydantic Schemas

**`serving/schemas.py`**

```python
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    field_id: str
    event_timestamp: str  # ISO-8601, e.g. "2024-06-01T00:00:00"


class PredictResponse(BaseModel):
    field_id: str
    predicted_yield_kg_per_ha: float
    lower_bound: float
    upper_bound: float
    model_version: str


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
```

---

## Part 2 — Feast Online Feature Client

**`serving/feast_client.py`**

This fetches features from Redis (online store) at inference time — sub-millisecond
latency compared to the offline Parquet path used during training.

```python
from __future__ import annotations

import pandas as pd
from feast import FeatureStore

FEAST_REPO_PATH = "features/feast_repo/feature_repo"

FEATURES = [
    "soil_features:soil_temp_mean",
    "soil_features:soil_temp_std",
    "soil_features:moisture_mean",
    "soil_features:moisture_std",
    "soil_features:ph_mean",
    "soil_features:nitrogen_mean",
    "soil_features:phosphorus_mean",
    "soil_features:potassium_mean",
    "weather_features:air_temp_mean",
    "weather_features:precip_total",
    "weather_features:humidity_mean",
    "weather_features:wind_speed_mean",
    "vegetation_features:latest_ndvi",
    "vegetation_features:cloud_cover_pct",
    "vegetation_features:ndvi_interpolated",
    "vegetation_features:ndvi_proxied",
]

_store: FeatureStore | None = None


def get_store() -> FeatureStore:
    global _store
    if _store is None:
        _store = FeatureStore(repo_path=FEAST_REPO_PATH)
    return _store


def fetch_online_features(field_ids: list[str]) -> pd.DataFrame:
    """Fetch latest online features for a list of field_ids from Redis."""
    store = get_store()
    entity_rows = [{"field_id": fid} for fid in field_ids]
    feature_vector = store.get_online_features(
        features=FEATURES,
        entity_rows=entity_rows,
    ).to_df()
    return feature_vector
```

---

## Part 3 — Model Loader

**`serving/model.py`**

Loads the Production model from MLflow at startup and exposes a predict function
that also computes a simple confidence interval using quantile estimation.

```python
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


def predict(
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (predictions, lower_bounds, upper_bounds)."""
    model = get_model()
    preds = model.predict(feature_df)
    lower = preds * (1 - CI_WIDTH)
    upper = preds * (1 + CI_WIDTH)
    return preds, lower, upper


def model_version() -> str:
    return _model_version
```

---

## Part 4 — Prediction Logger

**`serving/logging_sink.py`**

Logs every prediction to a JSON file (swap for BigQuery / Kafka in production).

```python
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "data/prediction_logs/predictions.jsonl"))


def log_prediction(
    field_id: str,
    predicted: float,
    lower: float,
    upper: float,
    model_version: str,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "field_id": field_id,
        "predicted_yield_kg_per_ha": predicted,
        "lower_bound": lower,
        "upper_bound": upper,
        "model_version": model_version,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
```

---

## Part 5 — Health Check

**`serving/health.py`**

Returns 200 when all systems are green, 503 when any check fails.
Kubernetes liveness/readiness probes hit this endpoint.

```python
from __future__ import annotations

import os
from datetime import UTC, datetime

import redis

from serving.model import get_model

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MAX_MATERIALIZATION_AGE_HOURS = int(os.getenv("MAX_MAT_AGE_HOURS", "25"))


def check_model() -> bool:
    try:
        get_model()
        return True
    except Exception:
        return False


def check_redis() -> bool:
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=1)
        return r.ping()
    except Exception:
        return False


def check_materialization_age() -> bool:
    """Check a Redis key written by the nightly materialization job."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=1)
        ts_bytes = r.get("feast:last_materialization_ts")
        if not ts_bytes:
            return False
        last_mat = datetime.fromisoformat(ts_bytes.decode())
        age_hours = (datetime.now(UTC) - last_mat).total_seconds() / 3600
        return age_hours < MAX_MATERIALIZATION_AGE_HOURS
    except Exception:
        return False


def run_health_checks() -> dict:
    checks = {
        "model_loaded": check_model(),
        "redis_connected": check_redis(),
        "materialization_fresh": check_materialization_age(),
    }
    checks["healthy"] = all(checks.values())
    return checks
```

---

## Part 6 — FastAPI Application

**`serving/app.py`**

```python
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

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
        model_version=model_version(),
    )
    log_prediction(
        req.field_id, result.predicted_yield_kg_per_ha,
        result.lower_bound, result.upper_bound, result.model_version,
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
            lower_bound=float(l),
            upper_bound=float(u),
            model_version=model_version(),
        )
        for fid, p, l, u in zip(field_ids, preds, lower, upper)
    ]
    for pr in predictions:
        log_prediction(
            pr.field_id, pr.predicted_yield_kg_per_ha,
            pr.lower_bound, pr.upper_bound, pr.model_version,
        )
    return BatchPredictResponse(predictions=predictions)
```

---

## Part 7 — Multi-Stage Dockerfile

**`Dockerfile`** (root of repo)

```dockerfile
# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY serving/ ./serving/
COPY features/feast_repo/ ./features/feast_repo/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and test locally:

```powershell
docker build -t agri-yield-serving:latest .
docker run -p 8000:8000 `
  -e REDIS_HOST=host.docker.internal `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  agri-yield-serving:latest
```

---

## Part 8 — Helm Chart

**`helm/agri-yield-serving/Chart.yaml`**

```yaml
apiVersion: v2
name: agri-yield-serving
description: Inference service for agri-yield XGBoost model
type: application
version: 0.1.0
appVersion: "1.0.0"
```

**`helm/agri-yield-serving/values.yaml`**

```yaml
replicaCount: 2

image:
  repository: agri-yield-serving
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetPredictionQueueDepth: 100   # custom metric via KEDA or Prometheus adapter

env:
  REDIS_HOST: redis-master
  REDIS_PORT: "6379"
  REGISTERED_MODEL_NAME: agri-yield-xgb
  MODEL_STAGE: Production
  CI_WIDTH: "0.15"
  MAX_MAT_AGE_HOURS: "25"
```

**`helm/agri-yield-serving/templates/deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: serving
          image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: {{ .Release.Name }}-config
          resources:
            requests:
              cpu: {{ .Values.resources.requests.cpu }}
              memory: {{ .Values.resources.requests.memory }}
            limits:
              cpu: {{ .Values.resources.limits.cpu }}
              memory: {{ .Values.resources.limits.memory }}
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 20
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
```

**`helm/agri-yield-serving/templates/hpa.yaml`**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ .Release.Name }}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ .Release.Name }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: External
      external:
        metric:
          name: prediction_queue_depth
        target:
          type: AverageValue
          averageValue: {{ .Values.autoscaling.targetPredictionQueueDepth }}
```

**`helm/agri-yield-serving/templates/configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-config
data:
  REDIS_HOST: {{ .Values.env.REDIS_HOST }}
  REDIS_PORT: {{ .Values.env.REDIS_PORT }}
  REGISTERED_MODEL_NAME: {{ .Values.env.REGISTERED_MODEL_NAME }}
  MODEL_STAGE: {{ .Values.env.MODEL_STAGE }}
  CI_WIDTH: {{ .Values.env.CI_WIDTH }}
  MAX_MAT_AGE_HOURS: {{ .Values.env.MAX_MAT_AGE_HOURS }}
```

---

## Part 9 — Nightly Feast Materialization

Run this as a cron job (or Prefect task) each night to push offline features into Redis.

```python
# orchestration/materialize.py
from datetime import UTC, datetime, timedelta

import redis
from feast import FeatureStore

REDIS_HOST = "localhost"
FEAST_REPO_PATH = "features/feast_repo/feature_repo"


def materialize() -> None:
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    store.materialize(start_date=start, end_date=end)

    # Stamp timestamp so /health can verify freshness
    r = redis.Redis(host=REDIS_HOST)
    r.set("feast:last_materialization_ts", end.isoformat())
    print(f"Materialization complete: {start} → {end}")


if __name__ == "__main__":
    materialize()
```

---

## Part 10 — Local Dev Run (No Kubernetes)

Test the full stack locally before containerising:

```powershell
# 1. Start Redis
docker run -d -p 6379:6379 redis:7

# 2. Run nightly materialization
uv run python -m orchestration.materialize

# 3. Start MLflow (if not already running)
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# 4. Start the serving API
uv run uvicorn serving.app:app --reload --port 8000

# 5. Test single prediction
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"field_id": "field_1", "event_timestamp": "2024-06-01T00:00:00"}'

# 6. Test health
curl http://localhost:8000/health
```

---

## Phase 4 Completion Checklist

- [ ] `serving/` module created with all 6 files
- [ ] `POST /predict` returns prediction + confidence interval
- [ ] `POST /predict/batch` handles up to 500 fields
- [ ] `/health` returns 503 when model, Redis, or materialization checks fail
- [ ] Prediction logs written to `data/prediction_logs/predictions.jsonl`
- [ ] Multi-stage Dockerfile builds and runs cleanly
- [ ] Helm chart deploys with HPA min=2, max=10
- [ ] Nightly materialization stamps `feast:last_materialization_ts` in Redis
- [ ] All files committed and pushed to GitHub
- [ ] `uv run python -m training.promote` still passes (regression check)
