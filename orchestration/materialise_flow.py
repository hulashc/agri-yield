"""Feature materialisation subflow.

Wraps the existing materialize.py logic.
"""
from __future__ import annotations

from prefect import flow, get_run_logger


@flow(name="materialise-features", log_prints=True)
def materialise_features() -> None:
    logger = get_run_logger()
    logger.info("Starting Feast materialisation")

    try:
        from orchestration.materialize import materialize
        materialize()
        logger.info("Feast materialisation complete")
    except Exception as exc:
        logger.warning(f"Materialisation failed (continuing): {exc}")
