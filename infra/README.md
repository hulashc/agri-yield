# infra/

> **Status: Planned — Phase 7**

This directory is reserved for infrastructure-as-code definitions.

## Planned scope

- `terraform/` — Terraform modules for AWS S3 (model artefacts), ElastiCache Redis (drift buffers), and ECR (Docker images)
- `render/` — Render service config (currently managed via `render.yaml` at root)
- `monitoring/` — Prometheus alerting rules and Grafana provisioning (currently in `monitoring/grafana/`)

## Current infrastructure

| Component | Current location | Technology |
|---|---|---|
| App hosting | Render.com free tier | Docker container from GHCR |
| Container registry | GitHub Container Registry (`ghcr.io`) | Docker |
| Model artefacts | Baked into Docker image at build time | `model.pkl` |
| Redis | Optional — configured via `REDIS_URL` env var | Redis Cloud free tier |
| Monitoring | `monitoring/grafana/` | Grafana + Prometheus |
| Kubernetes | `helm/` | Helm charts (see `helm/README.md`) |

## Production target

For a production deployment, model artefacts should be stored in S3/GCS and loaded at startup rather than baked into the image. This is tracked in Phase 7.
