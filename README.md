# 🌾 Agri Yield — UK Field Intelligence

![Live](https://img.shields.io/badge/status-live-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.13-blue) ![Docker](https://img.shields.io/badge/docker-ghcr.io-blue) [![CI](https://github.com/hulashc/agri-yield/actions/workflows/ci.yml/badge.svg)](https://github.com/hulashc/agri-yield/actions/workflows/ci.yml) [![Deploy](https://github.com/hulashc/agri-yield/actions/workflows/deploy.yml/badge.svg)](https://github.com/hulashc/agri-yield/actions/workflows/deploy.yml) [![Ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

A production-grade **ML-powered crop yield prediction platform** for UK agricultural fields. Combines live weather data, soil attributes, and an XGBoost regression model to predict yield (kg/ha) per field — served via a real-time FastAPI backend and an interactive Leaflet map dashboard.

**Live demo → [agri-yield-latest.onrender.com](https://agri-yield-latest.onrender.com)**

> ⚠️ Hosted on Render free tier — expect a ~50s cold start on first load.

---

## 📸 Screenshot

![Agri Yield Dashboard](https://agri-yield-latest.onrender.com)

---

## ✨ Features

- 🗺️ **Interactive UK field map** — 113 fields coloured by predicted yield (red → green)
- 🤖 **XGBoost yield model** — trained on real CYCleSS UK yield data (934 field-year records), with synthetic fallback, baked into the Docker image at CI time
- 🌤️ **Live weather integration** — Open-Meteo API fetches real-time temperature, precipitation, and solar radiation per field
- 📊 **Confidence intervals** — every prediction includes an 80% CI band
- 🚨 **Drift detection** — PSI-based feature drift monitoring with per-field warning badges
- 🔄 **CI/CD pipeline** — GitHub Actions trains the model, builds the Docker image, pushes to GHCR, and deploys to Render on every push to `main`
- 🧠 **Smart caching** — Redis (with in-memory fallback) caches weather API responses for 1 hour
- 📈 **Prometheus metrics** — `/metrics` endpoint for Grafana integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  GitHub Actions CI               │
│  scripts/archive/generate_data.py (fallback)      │
│  → model.pkl → Dockerfile.prod → ghcr.io image  │
└────────────────────┬────────────────────────────┘
                     │ docker pull
                     ▼
┌─────────────────────────────────────────────────┐
│               Render (Free Tier)                 │
│                                                  │
│  FastAPI (serving/app.py)                        │
│  ├── GET /fields  → predict all 113 fields       │
│  │    ├── serving/model.py  (XGBoost pkl)        │
│  │    └── ingestion/openmeteo_live.py            │
│  │         └── Open-Meteo API  (live weather)    │
│  ├── GET /metrics (Prometheus)                   │
│  └── Static HTML/JS dashboard (serving/static/index.html) │
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

### 2. Generate data and train the model

```bash
# Synthetic fallback only if no real CYCleSS data
python training/train_and_export.py
```

This produces `model.pkl` in the repo root.

### 3. Run the API

```bash
uvicorn serving.app:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — the dashboard loads immediately.

### 4. (Optional) Run with Docker

```bash
docker build -f Dockerfile.prod -t agri-yield .
docker run -p 8000:8000 agri-yield
```

---

## 🧠 ML Model

| Property | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Target | `yield_kg_per_ha` |
| Features | lat, lon, area, crop type, soil type, region, temperature, precipitation, solar radiation, ET₀, soil moisture, NDVI, week of year |
| Train/test split | 80/20 |
| RMSE | ~1759 kg/ha (real CYCleSS) / ~1724 kg/ha (synthetic fallback) |
| Confidence interval | ±15% of prediction (configurable via `CI_WIDTH` env var) |

The model is **retrained from scratch on every CI run** — no stale artefacts, no manual uploads. The trained `model.pkl` is baked directly into the production Docker image.

---

## 🌤️ Live Weather

Weather data is fetched from the [Open-Meteo API](https://open-meteo.com/) (free, no API key required) for each field's coordinates. Features used at inference time:

| Feature | Source |
|---|---|
| Temperature (°C) | `temperature_2m_max` today |
| Precipitation (mm) | `precipitation_sum` today |
| Solar radiation (MJ/m²) | `shortwave_radiation_sum` today |

Fallback chain: **Redis cache → Live API → In-memory cache → UK seasonal defaults**. The `STALE DATA` badge appears when defaults are used.

---

## 📁 Project Structure

```
agri-yield/
├── .github/workflows/
│   ├── ci.yml               # Lint, test, typecheck on every PR
│   └── deploy.yml           # Train → Build → Push → Deploy (main only)
├── data/
│   ├── seed/
│   │   └── uk_fields.csv    # 113 UK farm fields with metadata
│   └── features/            # Generated at CI time
├── ingestion/
│   └── openmeteo_live.py    # Live weather fetcher with caching
├── serving/
│   ├── app.py               # FastAPI app + /fields endpoint
│   ├── model.py             # Model loader + predict()
│   └── static/              # Dashboard HTML/CSS/JS
├── training/
│   ├── train_and_export.py  # XGBoost training script
│   └── utils/
│       └── features.py      # Canonical FEATURE_COLS (single source of truth)
├── monitoring/
│   └── drift.py             # PSI drift detector
├── scripts/
│   ├── archive/
│   │   └── generate_data.py # Synthetic data generator (fallback)
│   └── download_cycless.py  # CYCleSS UK yield data downloader
├── Dockerfile.prod          # Production image (bakes model.pkl)
└── pyproject.toml
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `disabled` | Set to an MLflow server URL to use registry instead of pkl |
| `PICKLE_MODEL_PATH` | `/app/model.pkl` | Path to the baked-in model |
| `FIELDS_CSV_PATH` | `/app/data/seed/uk_fields.csv` | Path to fields seed data |
| `CI_WIDTH` | `0.15` | Confidence interval half-width (fraction of prediction) |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL for weather caching |
| `RENDER_DEPLOY_HOOK` | — | Render deploy hook URL (set as GitHub secret) |

---

## 🔄 CI/CD Pipeline

Two workflows run on every push:

**`ci.yml`** (all branches — PR gate):
```
push / PR → Ruff lint → Train model → pytest → pass/fail
```

**`deploy.yml`** (main only — train, build, deploy):
```
push to main
    │
    ▼
[1] Train XGBoost (real CYCleSS or synthetic fallback) → model.pkl
    │
    ▼
[2] docker build -f Dockerfile.prod  (bakes model.pkl in)
    │
    ▼
[3] docker push ghcr.io/hulashc/agri-yield:latest
    │
    ▼
[4] curl RENDER_DEPLOY_HOOK  →  Render redeploys
```

---

## 📊 Monitoring

- **Prometheus metrics** available at `/metrics`
- **Drift detection** — PSI (Population Stability Index) computed per feature on each `/fields` call. Fields with PSI > 0.2 show a `⚠ DRIFT` badge
- Recommended: connect Grafana to `/metrics` for live yield and latency dashboards

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, scikit-learn, pandas, numpy |
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
