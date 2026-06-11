# Agri-Yield System Architecture

End-to-end data flow from ingestion through training, serving, and monitoring.

```mermaid
flowchart TD
    subgraph Ingestion["Data Ingestion"]
        A["Open-Meteo Live API\n(free, no key)"] --> B["openmeteo_live.py\nget_live_features()"]
        C["NASA POWER Parquet\ndata/raw/nasa_power/"] --> D["PSI Reference\nDistributions"]
        E["CYCleSS UK Dataset\n(real crop yields)"] --> F["train_and_export.py"]
        E2["Synthetic fallback\n(CI / no real data)"] --> F
    end

    subgraph Training["Training Pipeline (GitHub Actions CI)"]
        F --> G["XGBRegressor\nmean model"]
        F --> H["XGBRegressor\nq=0.10 lower bound"]
        F --> I["XGBRegressor\nq=0.90 upper bound"]
        G & H & I --> J["model_bundle.pkl\n(mean + lower + upper + metadata)"]
        J --> K{"RMSE < 2000 kg/ha?\nhas_quantile_ci?"}
        K -- "FAIL" --> L["Block deploy"]
        K -- "PASS" --> M["Docker build\n+ push to GHCR"]
        M --> N["Deploy to Render\n+ /health verify"]
    end

    subgraph Serving["Serving Layer (FastAPI on Render)"]
        N --> O["lifespan startup"]
        O --> O1["Load model_bundle.pkl\nserving/model.py"]
        O --> O2["Load uk_fields.csv\n(field registry)"]
        O --> O3["Warm PSI reference cache\n(Parquet → memory, once)"]
        B --> P["/predict\nPOST {field_id}"]
        B --> Q["/fields\nGET bulk predictions"]
        O1 --> P & Q
        P --> R["/predict/explain\nPOST {field_id}\nfeature contributions"]
        Q --> S["/model/info\nGET training metadata"]
        Q --> T["/model/feature-importance\nGET importance scores"]
        O1 --> U["/health\nGET status + CI info"]
    end

    subgraph Monitoring["Monitoring"]
        P & Q --> V["psi_detector.py\nevaluate_drift()"]
        V --> W["_append_to_buffer()\nRolling 500-sample window"]
        W --> W2{"REDIS_URL set?"}
        W2 -- "Yes" --> W3["Redis\n(drift:buf:field:feature)\nSurvives restarts"]
        W2 -- "No" --> W4["In-memory dict\n(resets on restart)"]
        W --> X["compute_psi()\nPSI per feature"]
        X --> Y["Prometheus metrics\n/metrics endpoint"]
        Y --> Z["Grafana Dashboard\nmonitoring/grafana/"]
        X -- "drift=red" --> AA["Log warning +\nworkflow_dispatch\nretrain-trigger.yml"]
    end

    subgraph CI_Thresholds["PSI Drift Thresholds"]
        T1["PSI < 0.25 → green (no action)"]
        T2["PSI 0.25-0.50 → amber (log warning)"]
        T3["PSI > 0.50 → red (trigger retrain)"]
    end
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Temporal train/test split | Prevents future data leaking into training — correct for time-series crop data |
| Quantile regression CI (q=0.10/0.90) | Statistically honest 80% prediction intervals vs. heuristic ±15% |
| RMSE quality gate in CI | Blocks broken model deployments before they reach production |
| Reference cache warmed at startup | Eliminates per-request Parquet disk reads that caused latency spikes on `/fields` |
| Redis drift buffer with in-memory fallback | Drift history survives container restarts; falls back gracefully if Redis unavailable |
| Semaphore on Open-Meteo calls | Prevents CPU starvation on Render's free 1-CPU instance under bulk load |
| `model_bundle.pkl` (3 models + metadata) | Single artefact contains everything serving needs: mean, lower, upper, importance, RMSE |
| Per-field partial failure in `/fields` | One bad field doesn't crash the entire bulk response — error is isolated per field |

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + model status + CI info |
| GET | `/fields` | Bulk predictions for all UK fields (powers the map) |
| POST | `/predict` | Single field prediction with CI bounds + drift status |
| POST | `/predict/explain` | Feature contribution breakdown for a single prediction |
| GET | `/model/info` | Training metadata: RMSE, source, CI method, feature list |
| GET | `/model/feature-importance` | Global feature importance scores |
| GET | `/metrics` | Prometheus metrics scrape endpoint |
