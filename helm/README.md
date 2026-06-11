# helm/

> **Status: Draft — not production-tested**

Helm charts for deploying agri-yield to a Kubernetes cluster.

## Structure

```
helm/
  agri-yield/
    Chart.yaml
    values.yaml
    templates/
      deployment.yaml
      service.yaml
      ingress.yaml
      hpa.yaml
```

## Quick start

```bash
# Add your image tag
export IMAGE_TAG=ghcr.io/hulashc/agri-yield:latest

# Dry run
helm template agri-yield ./helm/agri-yield \
  --set image.repository=ghcr.io/hulashc/agri-yield \
  --set image.tag=latest

# Install to a cluster
helm upgrade --install agri-yield ./helm/agri-yield \
  --namespace agri-yield --create-namespace \
  --set image.tag=$IMAGE_TAG \
  --set redis.url=$REDIS_URL
```

## Key values

| Key | Default | Description |
|---|---|---|
| `image.repository` | `ghcr.io/hulashc/agri-yield` | Docker image repo |
| `image.tag` | `latest` | Image tag |
| `replicaCount` | `1` | Number of pods |
| `redis.url` | `""` | Redis connection string |
| `resources.limits.memory` | `512Mi` | Memory limit |
| `env.FIELDS_CSV_PATH` | `/app/data/uk_fields.csv` | Path to fields CSV |

## Notes

- Charts are validated against Kubernetes 1.30+
- HPA is configured for CPU-based autoscaling (min 1, max 3 replicas)
- Liveness and readiness probes point to `/health`
- Current Render deployment is the primary production route; Kubernetes is provided for teams with existing cluster infrastructure
