"""Monitoring subflow.

Runs Evidently drift report and logs PSI scores.
"""
from __future__ import annotations

from prefect import flow, get_run_logger


@flow(name="run-monitoring", log_prints=True)
def run_monitoring() -> None:
    logger = get_run_logger()
    logger.info("Starting monitoring flow")

    try:
        from monitoring.drift_report import generate_drift_report
        generate_drift_report()
        logger.info("Drift report generated")
    except Exception as exc:
        logger.warning(f"Monitoring not fully wired yet: {exc}")
        logger.info("Monitoring stub complete — skipping drift report")
