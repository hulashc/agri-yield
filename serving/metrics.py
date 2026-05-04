"""
serving/metrics.py

FastAPI route for Prometheus metrics scrape endpoint.
Add this to serving/app.py:
    from serving.metrics import metrics_router
    app.include_router(metrics_router)
"""

from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

metrics_router = APIRouter()


@metrics_router.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
