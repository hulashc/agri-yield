"""Ingest + Great Expectations validation subflow.

Wraps the existing ingestion_flow. Returns True on success,
False on validation failure so the parent DAG can abort cleanly.
"""
from __future__ import annotations

from prefect import flow, get_run_logger


@flow(name="ingest-and-validate", log_prints=True)
def ingest_and_validate(use_synthetic: bool = False) -> bool:
    logger = get_run_logger()
    logger.info(f"Starting ingest (synthetic={use_synthetic})")

    try:
        from orchestration.ingestion_flow import ingestion_flow
        ingestion_flow(topic="soil-sensors")
        logger.info("Ingest and GE validation passed")
        return True
    except Exception as exc:
        logger.error(f"Ingest/validation failed: {exc}")
        return False
