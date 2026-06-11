# 🌾 Agri Yield — UK Field Intelligence

![Live](https://img.shields.io/badge/status-live-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.13-blue) ![Docker](https://img.shields.io/badge/docker-ghcr.io-blue) [![CI](https://github.com/hulashc/agri-yield/actions/workflows/ci.yml/badge.svg)](https://github.com/hulashc/agri-yield/actions/workflows/ci.yml) [![Deploy](https://github.com/hulashc/agri-yield/actions/workflows/deploy.yml/badge.svg)](https://github.com/hulashc/agri-yield/actions/workflows/deploy.yml) [![Ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

A production-grade **ML-powered crop yield prediction platform** for UK agricultural fields.
Combines live weather data, soil attributes, and an XGBoost ensemble to predict yield (kg/ha)
per field — served via a real-time FastAPI backend and an interactive Leaflet map dashboard.

**🚀 Live demo → [agri-yield-latest.onrender.com](https://agri-yield-latest.onrender.com)**

> ⚠️ Hosted on Render free tier — expect a ~50s cold start on first load.

---

## ✨ What makes this production-grade

| Capability | Detail |
|---|---|
| **Real quantile CI** | Lower/upper bounds from q=0.10 and q=0.90 XGBoost quantile regressors — not a hardcoded fraction |
| **Live weather per field** | Open-Meteo API called at inference time with 4-level fallback chain |
| **PSI drift monitoring** | Population Stability Index computed per feature per field with green/amber/red levels |
| **Reference cache** | NASA POWER distributions loaded once at startup, not per-request |
| **Feature importance** | Served at `/model/feature-importance` — sorted by XGBoost gain |
| **Model metadata** | RMSE, dataset source, training date, CI method served at `/model/info` |
| **CI/CD pipeline** | Train → Quality Gate → Test → Docker → GHCR → Render deploy → Health verify |
| **Temporal split** | Train/test never shuffled — past fields train, future fields test |
| **Concurrency guard** | `asyncio.Semaphore(2)` throttles live weather fetches on Render’s 1-CPU free tier |

---

## 📊 Key API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive Leaflet map — all 113 fields with yield, CI bands, drift |
| `/health` | GET | Build version, model status, quantile CI presence, fields count |
| `/fields` | GET | All fields with live predictions, CI bounds, drift level |
| `/predict` | POST | Single field prediction with CI, drift, weather, model version |
| `/model/info` | GET | RMSE, dataset source, trained_at, CI method, feature list |
| `/model/feature-importance` | GET | Feature importance scores sorted by XGBoost gain |
| `/metrics` | GET | Prometheus metrics for Grafana |

---

## 🧠 ML Model

| Property | Value |
|---|---|
| Algorithm | XGBoost (mean + lower q0.10 + upper q0.90) |
| Target | `yield_kg_per_ha` |
| Features | lat, lon, area, crop type, soil type, region, temperature, precipitation, solar radiation, ET₀, soil moisture, NDVI, week of year |
| Train/test split | Temporal — past trains, future tests. Never shuffled |
| RMSE | ~1759 kg/ha (real CYCleSS) / ~1724 kg/ha (synthetic fallback) |
| Confidence interval | 80% — model-derived from quantile regressors (q0.10–q0.90) |
| CI ordering | Enforced: lower ≤ mean ≤ upper per row after prediction |
| Explainability | `/model/feature-importance` — XGBoost gain importance per feature |

The model is **retrained from scratch on every CI push to main** — no stale artefacts.
All three models (mean, lower, upper) are saved as `model_bundle.pkl` and baked into
the Docker image. The bundle also stores RMSE, training date, and dataset source,
all visible at runtime via `/model/info`.

---

## 🌤️ Live Weather

Weather data is fetched from the [Open-Meteo API](https://open-meteo.com/) (free, no API key required):

| Feature | Source |
|---|---|
| Temperature (°C) | `temperature_2m_max` today |
| Precipitation (mm) | `precipitation_sum` today |
| Solar radiation (MJ/m²) | `shortwave_radiation_sum` today |

Fallback chain: **Redis cache → Live API → In-memory cache → UK seasonal defaults**.
The `STALE DATA` badge appears when defaults are used.

---

## 🚨 Drift Monitoring

PSI (Population Stability Index) is computed per-feature per-field against NASA POWER
historical reference distributions loaded at startup:

| Level | PSI threshold | Action |
|---|---|---|
| 🟢 green | < 0.25 | No action |
| 🟡 amber | 0.25 – 0.50 | Log warning, Prometheus metric |
| 🔴 red | > 0.50 | Drift badge on field, retraining alert |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│               GitHub Actions CI                 │
│  1. train_and_export.py                          │
│     ├─ mean model (XGBRegressor)                │
│     ├─ lower model (quantile q=0.10)            │
│     └─ upper model (quantile q=0.90)            │
│  2. Quality gate: RMSE < 2000 kg/ha             │
│     + quantile models present                   │
│  3. pytest                                       │
│  4. docker build (bakes model_bundle.pkl)        │
│  5. push ghcr.io → Render deploy → /health verify│
└───────────────────────┤
                     │ docker pull
                     ▼
┌─────────────────────────────────────────────────┐
│             Render (Free Tier)                  │
│                                                  │
│  FastAPI (serving/app.py)                        │
│  Startup: load model_bundle.pkl                  │
│           warm reference cache (NASA POWER)      │
│  ├─ GET /         Leaflet map (113 UK fields)    │
│  ├─ GET /fields    live predictions (all fields) │
│  ├─ POST /predict  single field + CI + drift     │
│  ├─ GET /model/info           training metadata  │
│  ├─ GET /model/feature-importance  XGBoost gain │
│  ├─ GET /health    build + model + fields status │
│  └─ GET /metrics   Prometheus                    │
│                                                  │
│  ingestion/openmeteo_live.py                     │
│    └─ Redis → Open-Meteo API → in-memory → default│
│                                                  │
│  monitoring/psi_detector.py                      │
│    └─ PSI green/amber/red per feature per field   │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.13+
- Docker (optional)

### 1. Clone and install

```bash
git clone https://github.com/hulashc/agri-yield.git
cd agri-yield
pip install uv
uv sync
```

### 2. Train models

```bash
python training/train_and_export.py
# Produces model_bundle.pkl (mean + q0.10 + q0.90) and model.pkl (legacy)
```

### 3. Run the API

```bash
uvicorn serving.app:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — map loads immediately.

See model metadata: [http://localhost:8000/model/info](http://localhost:8000/model/info)

See feature importance: [http://localhost:8000/model/feature-importance](http://localhost:8000/model/feature-importance)

### 4. (Optional) Run with Docker

```bash
docker build -f Dockerfile.prod -t agri-yield .
docker run -p 8000:8000 agri-yield
```

---

## ⚡ Quickcheck the live API

```bash
# Health (shows quantile CI status, fields count, reference cache)
curl https://agri-yield-latest.onrender.com/health | python3 -m json.tool

# Model metadata (RMSE, dataset source, CI method, trained_at)
curl https://agri-yield-latest.onrender.com/model/info | python3 -m json.tool

# Feature importance (sorted by XGBoost gain)
curl https://agri-yield-latest.onrender.com/model/feature-importance | python3 -m json.tool

# Single field prediction
curl -X POST https://agri-yield-latest.onrender.com/predict \
  -H 'Content-Type: application/json' \
  -d '{"field_id": "field_001"}' | python3 -m json.tool
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `disabled` | Set to an MLflow server URL to use registry instead of bundle |
| `PICKLE_BUNDLE_PATH` | `/app/model_bundle.pkl` | Path to the quantile model bundle |
| `PICKLE_MODEL_PATH` | `/app/model.pkl` | Legacy mean-only model (fallback) |
| `FIELDS_CSV_PATH` | `/app/data/seed/uk_fields.csv` | Path to fields seed data |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL for weather caching |
| `RENDER_DEPLOY_HOOK` | — | Render deploy hook URL (GitHub secret) |
| `RENDER_SERVICE_URL` | — | Render app URL for post-deploy health verify (GitHub secret) |

---

## 🔄 CI/CD Pipeline

**`ci.yml`** (all branches — PR gate):
```
push / PR → Ruff lint → Train model → pytest → pass/fail
```

**`deploy.yml`** (main only):
```
push to main
    │
    ▼
[1] Train XGBoost — mean + lower (q0.10) + upper (q0.90)
    │
    ▼
[2] Quality gate: RMSE < 2000 kg/ha + quantile models present
    │
    ▼
[3] pytest
    │
    ▼
[4] docker build Dockerfile.prod (bakes model_bundle.pkl + model.pkl)
    │
    ▼
[5] push ghcr.io/hulashc/agri-yield:latest + :<sha>
    │
    ▼
[6] curl RENDER_DEPLOY_HOOK → Render redeploys
    │
    ▼
[7] Poll /health until 200 → print /model/info
```

---

## 📊 Monitoring

- **Prometheus metrics** at `/metrics`
- **PSI drift detection** — per feature per field, green/amber/red
- **Reference distributions** — NASA POWER historical data, pre-loaded at startup
- Recommended: connect Grafana to `/metrics` for live yield and latency dashboards

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost (mean + quantile), scikit-learn, pandas, numpy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Leaflet.js, vanilla JS/CSS |
| Weather | Open-Meteo API |
| Caching | Redis (in-memory fallback) |
| Monitoring | Prometheus, PSI drift detection |
| CI/CD | GitHub Actions |
| Registry | GitHub Container Registry (GHCR) |
| Hosting | Render (free tier) |
| Package mgmt | uv |

---

## 📄 License

MIT — see [LICENSE](LICENSE).
