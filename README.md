@'
# agri-yield — Agricultural Yield Prediction MLOps Pipeline

A production-grade MLOps pipeline that predicts weekly crop yield per field using multi-source sensor fusion, automated retraining, and drift-aware monitoring.

## What This Project Demonstrates

- **End-to-end MLOps** from raw IoT data to a served, monitored model
- **Temporal data engineering** — aligning 15-min sensors, hourly weather, and weekly satellite imagery
- **Feature store discipline** — point-in-time correct training with Feast
- **Drift detection** — seasonal vs genuine concept drift using dual reference windows
- **Full data lineage** — Kafka → DVC → Feast → MLflow → deployed model

## Tech Stack

| Layer | Tools |
|---|---|
| Ingestion | Kafka, Great Expectations, Faker (simulator) |
| Feature engineering | Apache Spark (Spark Operator on k8s), Feast, DVC |
| Training | XGBoost, Optuna, MLflow |
| Serving | FastAPI, Redis (Feast online store), Helm/k8s |
| Monitoring | Evidently AI, Prefect, Grafana, Prometheus |
| Infrastructure | Kubernetes (k3d local / GKE cloud), Terraform |

## Project Structure
