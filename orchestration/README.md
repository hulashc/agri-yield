# orchestration/

> **Status: Planned — Phase 7**

This directory is reserved for workflow orchestration using [Prefect 3](https://docs.prefect.io/).

## Planned scope

- `flows/retrain_flow.py` — Scheduled retraining flow: ingest → validate → train → quality gate → register model
- `flows/ingest_flow.py` — Daily NASA POWER + Open-Meteo data fetch with retry logic
- `deployments/` — Prefect deployment definitions for each flow
- `schedules/` — Cron/interval schedule configs

## Current state

Training is currently triggered manually or via GitHub Actions `workflow_dispatch`:
- Training: `uv run python training/train_and_export.py`
- CI quality gate: `uv run python scripts/quality_gate.py`
- Retraining on drift: triggered via GitHub Actions workflow dispatch (Phase 4)

When Phase 7 begins, these will be orchestrated as Prefect flows with full observability, retry handling, and scheduled execution.
