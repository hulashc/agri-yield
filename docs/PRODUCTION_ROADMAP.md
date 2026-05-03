# Production Roadmap — `agri-yield`

> End-to-end checklist to take the agricultural yield prediction system from structured codebase to a live, monitored, publicly demonstrable MLOps product.

---

## Current State Audit

| Layer | Files Present | Status |
|---|---|---|
| Ingestion | `ndvi_puller.py`, `sensor_simulator.py`, `weather_poller.py` | ✅ Exists |
| Features | `features/` | ✅ Exists |
| Training | `training/` | ✅ Exists |
| Serving | `app.py`, `feast_client.py`, `model.py`, `schemas.py` | ✅ Exists |
| Monitoring | `monitoring/` | ✅ Exists |
| Orchestration | `orchestration/` | ✅ Exists |
| Infra / Helm | `infra/`, `helm/` | ✅ Exists |
| Public frontend | — | ❌ Missing |
| NASA POWER historical ingestion | partial in `weather_poller.py` | ⚠️ Needs wiring |
| Open-Meteo live feature fetch | — | ❌ Missing |
| CI/CD pipeline | — | ❌ Missing |
| Drift → retrain trigger | `monitoring/` + `orchestration/` | ⚠️ Needs end-to-end wire |
| Grafana dashboard JSON | — | ❌ Missing |

---

## Phase 1 — Real Data Wiring (Foundation)

The model is only as credible as the data it learns from. This phase replaces any synthetic placeholders with real UK agronomic signals.

### Step 1.1 — NASA POWER Historical Ingestion

**File:** `ingestion/nasa_power_historical.py`

```python
# Target: Pull 1981–present daily agronomic data for UK field coordinates
# Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point
# Parameters: PRECTOTCORR, ALLSKY_SFC_SW_DWN, T2M_MAX, T2M_MIN, RH2M, WS2M, EVPTRNS
# community=AG, format=JSON
# Output: Parquet partitioned by field_id/year to data/raw/nasa_power/
```

- Call the NASA POWER `AG` community endpoint (no API key required)
- Parameterise by `lat`, `lon`, `start` (19810101), `end` (yesterday)
- Write output to `data/raw/nasa_power/{field_id}/{year}.parquet`
- Register the output path in DVC: `dvc add data/raw/nasa_power/`
- Add a `Great Expectations` suite: null check on `PRECTOTCORR`, range check `T2M_MAX` between -20 and 50

### Step 1.2 — Open-Meteo Live Feature Fetch

**File:** `ingestion/openmeteo_live.py`

```python
# Target: Pull today + 7-day forecast for any field lat/lon
# Endpoint: https://api.open-meteo.com/v1/forecast
# Variables: precipitation, shortwave_radiation, temperature_2m_max/min,
#            relative_humidity_2m, windspeed_10m, et0_fao_evapotranspiration
# Output: dict keyed by field_id, consumed directly by serving/app.py at request time
```

- This replaces the `feast_client.py` online feature fetch for real-time predictions
- Cache response in Redis with TTL of 3600s (1 hour) to avoid rate limits
- Fallback: if Open-Meteo is unreachable, serve last cached features with a `stale_features: true` flag in the response

### Step 1.3 — UK Field Coordinates Seed File

**File:** `data/seed/uk_fields.csv`

```csv
field_id,name,lat,lon,crop_type,region
F001,Bicker Fen,52.9135,-0.1736,winter_wheat,Lincolnshire
F002,Holbeach St Marks,52.8389,-0.0325,oilseed_rape,Lincolnshire
F003,Spalding East,52.7862,-0.1532,sugar_beet,Lincolnshire
F004,Wisbech North,52.6793,0.1631,winter_barley,Cambridgeshire
F005,Boston West,52.9721,-0.0812,winter_wheat,Lincolnshire
F006,Kings Lynn South,52.7156,0.3928,oilseed_rape,Norfolk
F007,Downham Market,52.6009,0.3811,sugar_beet,Norfolk
F008,Ely Central,52.3983,0.2617,winter_wheat,Cambridgeshire
```

- Lincolnshire + Cambridgeshire + Norfolk = highest arable density in UK
- `crop_type` drives the yield target variable and harvest calendar offset
- Use these coordinates in both NASA POWER historical pulls and Open-Meteo live calls

---

## Phase 2 — Feature Store Materialisation

### Step 2.1 — Feast Feature Definitions

**File:** `features/field_weather_features.py`

Define a `FeatureView` for each of:

| Feature View | Entity | Features | TTL |
|---|---|---|---|
| `nasa_historical_fv` | `field_id` | 30-day rolling rainfall, GDD accumulation, drought index | 24h |
| `openmeteo_live_fv` | `field_id` | Today's T2M, rainfall, ET0, radiation | 1h |
| `ndvi_fv` | `field_id` | Latest NDVI, NDVI 14-day delta | 7d |
| `soil_fv` | `field_id` | Soil type (static), field area_ha | ∞ |

- Point-in-time correctness: always pass `timestamp` to `get_historical_features()`
- Use SQLite as offline store locally, PostgreSQL (Supabase) in production
- Materialise with: `feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)`

### Step 2.2 — Feature Validation Contract

**File:** `contracts/feature_schema.yaml`

```yaml
field_weather_features:
  rainfall_30d_mm:
    type: float
    min: 0
    max: 500
  gdd_accumulation:
    type: float
    min: 0
    max: 3000
  ndvi_latest:
    type: float
    min: -1.0
    max: 1.0
  t2m_max_today:
    type: float
    min: -20
    max: 50
```

- Run this validation in the Prefect materialisation flow before writing to the online store
- Fail fast: if >5% of records violate schema, raise a `DataQualityError` and alert

---

## Phase 3 — Model Training Pipeline

### Step 3.1 — Training Script

**File:** `training/train.py`

```python
# XGBoost with Optuna hyperparameter search
# Target variable: yield_kg_ha (float)
# Train/val split: temporal — never random shuffle on time-series data
# Use 2000–2020 for training, 2021–2023 for validation, 2024+ held out
```

Key requirements:
- Log all runs to MLflow: params, metrics (`RMSE`, `MAE`, `R²`), feature importances
- Log the Feast `feature_service` version used for that training run
- Save model artefact with `mlflow.xgboost.log_model()`
- Compute and log **prediction intervals** using XGBoost's `pred_contribs` or quantile regression (`objective="reg:quantileerror"` at q=0.1 and q=0.9)
- Register best model to MLflow Model Registry under stage `Staging`

### Step 3.2 — Model Card

**File:** `docs/model_card.md`

Document:
- Training data date range and geographic coverage
- Feature list with agronomic justification (link to NASA POWER parameter definitions)
- Performance by crop type (wheat vs oilseed rape vs sugar beet will differ)
- Known failure modes: model has not seen UK drought years after 2022 training cutoff
- Confidence interval calibration: what does the 80% CI actually cover?

---

## Phase 4 — Serving Layer

### Step 4.1 — FastAPI `/predict` Endpoint

**File:** `serving/app.py` — extend existing

Current `serving/app.py` exists. Wire in:

```python
# POST /predict
# Body: { field_id: str, prediction_date: date }
# Response: {
#   field_id: str,
#   predicted_yield_kg_ha: float,
#   lower_bound: float,
#   upper_bound: float,
#   confidence_level: 0.80,
#   drift_warning: bool,
#   psi_score: float,
#   stale_features: bool,
#   model_version: str,
#   last_updated: datetime
# }
```

- `drift_warning: true` if PSI score > 0.2 on any feature in the request
- `model_version` pulled from MLflow Model Registry (Production stage)
- Add Prometheus metrics: `prediction_latency_seconds`, `drift_warnings_total`, `requests_total`

### Step 4.2 — Prometheus Instrumentation

**File:** `serving/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge

PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "End-to-end /predict latency")
DRIFT_WARNINGS = Counter("drift_warnings_total", "PSI drift warnings fired", ["field_id"])
MODEL_VERSION = Gauge("model_version_deployed", "Active model version number")
REQUESTS = Counter("predictions_total", "Total prediction requests", ["crop_type"])
```

- Expose at `GET /metrics` (Prometheus scrape endpoint)
- Add to `serving/app.py`: `app.add_route("/metrics", make_asgi_app())`

---

## Phase 5 — Drift Detection and Retraining

### Step 5.1 — PSI Drift Detector

**File:** `monitoring/psi_detector.py`

Population Stability Index (PSI) measures feature distribution shift:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

- PSI < 0.1 → No significant shift (green)
- PSI 0.1–0.2 → Moderate shift, log warning (amber)
- PSI > 0.2 → Significant drift, trigger retraining (red)

Reference distribution: NASA POWER features from the most recent completed growing season.
Live distribution: Open-Meteo features from the last 14 days of requests.

Compute PSI for: `rainfall_30d_mm`, `gdd_accumulation`, `t2m_max_today`, `ndvi_latest`

### Step 5.2 — Prefect Retraining Flow

**File:** `orchestration/retrain_flow.py`

```python
@flow(name="agri-yield-retrain")
def retrain_on_drift(triggered_by: str = "scheduled"):
    # 1. Pull latest features from Feast offline store
    # 2. Run training/train.py via subprocess or direct import
    # 3. Compare new model RMSE vs Production model RMSE in MLflow
    # 4. If new model RMSE < current * 0.95: promote to Production
    # 5. Else: log as Staging, alert via Prefect notification
    # 6. Emit retraining_completed Prometheus counter
```

Schedule: run every Sunday at 02:00 UTC (weekly growing season cadence)
Trigger: also fire when `drift_warnings_total` crosses threshold (use Prefect automation)

---

## Phase 6 — Grafana Dashboard

### Step 6.1 — Dashboard JSON

**File:** `monitoring/grafana/agri_yield_dashboard.json`

Panels to include:

| Panel | Query | Purpose |
|---|---|---|
| Predictions per hour | `rate(predictions_total[1h])` | Traffic overview |
| Drift warning rate | `rate(drift_warnings_total[24h])` | Model health |
| P50/P95 latency | `histogram_quantile(0.95, prediction_latency_seconds_bucket)` | SLA tracking |
| PSI score by feature | Custom gauge from `psi_score` Gauge metric | Drift visibility |
| Active model version | `model_version_deployed` | Deployment awareness |
| Retrain events timeline | Annotation from Prefect webhook | MLOps audit trail |

Export the dashboard JSON and commit it. Import with:

```bash
curl -X POST http://localhost:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/agri_yield_dashboard.json
```

---

## Phase 7 — Public Frontend

This is the piece that transforms the project from backend engineering into something a non-technical person can immediately understand.

### Step 7.1 — Single Page Map UI

**File:** `frontend/index.html`

Stack: pure HTML + Leaflet.js + vanilla JS. No framework needed. Deploy to Vercel.

```html
<!-- Core structure -->
<!-- 1. Leaflet map centred on Lincolnshire (52.9°N, -0.1°E) -->
<!-- 2. GeoJSON layer of UK arable fields (8 seed fields minimum) -->
<!-- 3. Click a field → calls your /predict endpoint -->
<!-- 4. Popup renders: yield estimate, confidence band, drift warning badge -->
<!-- 5. Colour-coded markers: green (no drift), amber (PSI 0.1–0.2), red (PSI >0.2) -->
```

Field polygon GeoJSON source: [UKCEH Land Cover](https://www.ceh.ac.uk/data/ukceh-land-cover-maps) or draw manually for 8 seed fields.

### Step 7.2 — Vercel Deployment

**File:** `frontend/vercel.json`

```json
{
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://your-api-url.com/:path*" }
  ]
}
```

- Set `NEXT_PUBLIC_API_URL` as Vercel environment variable pointing to your FastAPI deployment
- Deploy: `vercel --prod` from `frontend/`
- CORS: add `allow_origins=["https://your-vercel-domain.vercel.app"]` to FastAPI

---

## Phase 8 — CI/CD Pipeline

**File:** `.github/workflows/ci.yml`

```yaml
name: CI

on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest tests/ -v --cov=serving --cov=ingestion --cov=monitoring

  docker-build:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t agri-yield-api .
      - run: docker run --rm agri-yield-api python -c "from serving.app import app; print('OK')"
```

**File:** `.github/workflows/deploy.yml`

```yaml
name: Deploy to Production

on:
  push:
    branches: [master]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy FastAPI to Railway / Render / GKE
        run: echo "Add your deployment target here"
      - name: Deploy Frontend to Vercel
        run: vercel --prod --token ${{ secrets.VERCEL_TOKEN }}
```

---

## Phase 9 — Infrastructure as Code

### Step 9.1 — docker-compose for local dev

**File:** `docker-compose.yml`

```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - REDIS_URL=redis://redis:6379
    depends_on: [redis, mlflow]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.11.0
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0

  prometheus:
    image: prom/prometheus:v2.51.0
    ports: ["9090:9090"]
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:10.4.0
    ports: ["3000:3000"]
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning/dashboards
```

Start everything: `docker compose up -d`

### Step 9.2 — Terraform (GKE production)

**File:** `infra/terraform/main.tf`

Provision:
- GKE Autopilot cluster (cheapest managed Kubernetes)
- Cloud SQL (PostgreSQL) for Feast offline store
- Redis Memorystore for Feast online store
- Artifact Registry for Docker images
- Secret Manager for any API credentials

---

## Phase 10 — Testing

**Directory:** `tests/`

| Test File | What It Tests |
|---|---|
| `tests/test_predict.py` | `/predict` returns correct schema, confidence bounds are valid (lower < upper) |
| `tests/test_drift.py` | PSI > 0.2 on known-shifted data returns `drift_warning: true` |
| `tests/test_ingestion.py` | NASA POWER puller returns expected columns and non-null values |
| `tests/test_features.py` | Feast materialisation produces point-in-time correct features |
| `tests/test_retrain.py` | Prefect flow completes without error on mock data |

Run: `uv run pytest tests/ -v`
Coverage target: **>80%** on `serving/` and `monitoring/`

---

## Execution Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10
  ↕            ↕         ↕         ↕
Real data  Features  MLflow    FastAPI
 wired     stored   tracked   live
```

The minimum viable demo path (fastest to something showable):

1. **Phase 1.1 + 1.3** — pull NASA POWER for 8 seed fields, save to parquet
2. **Phase 3.1** — train XGBoost, log to MLflow
3. **Phase 4.1** — wire `/predict` with confidence intervals
4. **Phase 7.1** — build single HTML page, call `/predict`, render on Leaflet map
5. **Phase 7.2** — deploy to Vercel

That sequence takes the project from where it is now to a live public URL in approximately 4–6 focused working sessions.

---

## Recruiter Pitch (After Completion)

> "I built a production ML system for agricultural yield prediction across UK arable fields. It ingests real historical weather data from NASA POWER, serves per-field predictions with 80% confidence intervals via FastAPI, monitors for data drift using PSI scoring with Prometheus + Grafana, and retrains automatically on a Prefect schedule. The model is versioned in MLflow. There is a live public map at [your-vercel-url] where you can click a field in Lincolnshire and get the current yield estimate."

That is a complete data engineering + MLOps portfolio statement with a demonstrable artefact.
