# Agri-Yield System Architecture

This document describes the end-to-end architecture of the agri-yield system:
from raw data ingestion through model training, prediction serving, monitoring,
and automated retraining.

---

## System Diagram

```mermaid
flowchart TD
    subgraph Data["Data Sources"]
        A["Open-Meteo Live API\n(daily weather)"] 
        B["NASA POWER Parquet\n(historical reference)"] 
        C["CYCleSS UK Yield CSV\n(ground truth labels)"]
        D["Synthetic Fallback\n(CI / no CYCleSS)"] 
    end

    subgraph Ingestion["Ingestion Layer"]
        E["openmeteo_live.py\nFetch + cache live features"]
        F["nasa_power_ingest.py\nBuild reference Parquet store"]
    end

    subgraph Training["Training Pipeline  (GitHub Actions CI)"]
        G["train_and_export.py"]
        H["XGBRegressor  mean model"]
        I["XGBRegressor  q=0.10 lower bound"]
        J["XGBRegressor  q=0.90 upper bound"]
        K["model_bundle.pkl\n{ mean, lower, upper, rmse,\n  feature_importance, metadata }"] 
        L{"Quality Gate\nRMSE < 2000 kg/ha\nquantile models present?"}
        M["Block deploy"]
        N["Docker build\ntagged :latest + :sha\npush to GHCR"]
    end

    subgraph Serving["Serving Layer  (FastAPI on Render)"]
        O["lifespan startup\n• load model_bundle.pkl\n• warm PSI reference cache\n• load uk_fields.csv"]
        P["GET /fields\nbulk predictions for map UI"]
        Q["POST /predict\nsingle field prediction"]
        R["POST /predict/explain\nfeature contribution breakdown"]
        S["GET /model/info\ntraining metadata"]
        T["GET /model/feature-importance\nsorted importance scores"]
        U["GET /health\nreadiness probe"]
    end

    subgraph Monitoring["Monitoring Layer"]
        V["psi_detector.py\nPSI per feature per field"]
        W[("Redis\nRolling drift buffer\n(500 values / feature / field)\nTTL: 7 days")]
        X["In-memory fallback\n(when REDIS_URL unset)"]
        Y["Prometheus metrics\n/metrics endpoint"]
        Z["Grafana dashboard"]
        AA["Drift level\ngreen / amber / red"]
    end

    subgraph Retraining["Retraining Loop"]
        AB["retrain-trigger.yml\nworkflow_dispatch"]
        AC["Manual trigger\nor automated alert"]
    end

    subgraph Frontend["Frontend"]
        AD["serving/static/index.html\nLeaflet.js map\n+ model info panel"]
    end

    %% Data flow
    A --> E
    B --> F
    C --> G
    D --> G
    F --> Training

    E --> Serving
    G --> H & I & J
    H & I & J --> K
    K --> L
    L -- fail --> M
    L -- pass --> N
    N --> O

    O --> P & Q & R & S & T & U
    E --> P & Q & R

    P & Q --> V
    V --> W
    V --> X
    W & X --> AA
    AA --> Y
    Y --> Z

    AA -- drift=red --> AC
    AC --> AB
    AB --> G

    AD --> P
```

---

## Key Design Decisions

### Three-Model Bundle

A single `model_bundle.pkl` contains three separately trained XGBoost models:

| Model | Objective | Purpose |
|---|---|---|
| `mean` | `reg:squarederror` | Point estimate |
| `lower` | `reg:quantileerror` q=0.10 | Lower CI bound |
| `upper` | `reg:quantileerror` q=0.90 | Upper CI bound |

CI ordering is enforced post-prediction: `lower = min(lower, mean)`, `upper = max(upper, mean)`.
This gives statistically derived 80% prediction intervals rather than a hardcoded ±15% heuristic.

### Startup Reference Cache

All NASA POWER Parquet files are read once at app startup into `_REFERENCE_CACHE` (a dict of
`feature → np.ndarray`). This eliminates per-request Parquet I/O which was a bottleneck on
the Render free tier during high-traffic `/fields` calls.

### Redis Drift Buffer

Per-field, per-feature rolling windows of live values are persisted in Redis (`REDIS_TTL=7d`).
When `REDIS_URL` is unset (local dev, CI), the module falls back transparently to an in-process
dict with no code changes required at the call site.

### Concurrency Control

`asyncio.Semaphore(2)` throttles concurrent Open-Meteo HTTP calls. On Render’s free
1-CPU instance this prevents thread pool exhaustion while still allowing bulk `/fields`
requests to run fields in parallel.

### Quantile CI Fallback Chain

If the bundle contains no quantile models (legacy `model.pkl` path), the system falls back
to `±15%` heuristic bounds and records `"ci_method": "heuristic"` in every response so
callers can distinguish the two cases.

---

## Endpoint Reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Readiness: model load, fields load, CI status |
| GET | `/fields` | Bulk predictions for all fields (map UI) |
| POST | `/predict` | Single field prediction with drift score |
| POST | `/predict/explain` | Feature contributions for a prediction |
| GET | `/model/info` | Training metadata, RMSE, CI method |
| GET | `/model/feature-importance` | Global feature importance (sorted) |
| GET | `/metrics` | Prometheus scrape endpoint |

---

## Drift Detection Thresholds

| PSI Range | Level | Action |
|---|---|---|
| < 0.25 | green | No action |
| 0.25 – 0.50 | amber | Log warning, Prometheus metric |
| > 0.50 | red | Warning + `drift_warning: true` in response |

Thresholds are wider than the classic 0.10/0.20 because training uses proxy/synthetic
feature values. PSI is only computed once a rolling buffer of `≥30` live requests exists
per field — before that, all fields return `drift_level: green`.
