# features/

> **Status: Planned — Phase 7**

This directory is reserved for online/offline feature store integration using [Feast](https://feast.dev/).

## Planned scope

- `feature_store.yaml` — Feast repository config pointing at a Redis online store and S3/GCS offline store
- `feature_views.py` — Feast FeatureView definitions for field-level weather aggregates
- `retrieve_online.py` — Online feature retrieval for low-latency serving
- `retrieve_offline.py` — Historical feature joins for training dataset construction

## Current state

Features are currently engineered directly inside:
- `ingestion/` — raw NASA POWER / Open-Meteo fetch and aggregation
- `training/utils/features.py` — canonical `FEATURE_COLS` list and transforms
- `training/train_and_export.py` — full feature pipeline for training

When Phase 7 begins, the above will be refactored into Feast feature views to enable point-in-time correct joins and a consistent online/offline parity guarantee.
