"""
orchestration/retrain_trigger_flow.py

Prefect flow: triggered when PSI drift exceeds the RED threshold (0.50).
Wraps the existing training pipeline and handles promotion logic.
Schedule: Every Sunday 02:00 UTC + on-demand drift trigger.
"""

import os
from datetime import datetime

import mlflow
from prefect import flow, get_run_logger, task

from monitoring.prometheus_metrics import (
    LAST_RETRAIN_TIMESTAMP,
    RETRAIN_EVENTS,
    RETRAIN_PROMOTED,
)


@task(name="check-current-production-rmse")
def get_production_rmse() -> float:
    """Pull RMSE of the current Production model from MLflow registry."""
    logger = get_run_logger()
    client = mlflow.tracking.MlflowClient(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    try:
        model_name = "agri-yield-xgboost"
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            logger.warning("No Production model found — will auto-promote new model")
            return float("inf")
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        rmse = float(run.data.metrics.get("rmse_val", float("inf")))
        logger.info(f"Current Production RMSE: {rmse:.2f}")
        return rmse
    except Exception as e:
        logger.error(f"Could not fetch Production RMSE: {e}")
        return float("inf")


@task(name="run-training-pipeline")
def run_training() -> dict[str, object]:
    """
    Execute the training pipeline via subprocess.
    Returns dict with run_id and rmse_val.
    """
    logger = get_run_logger()
    logger.info("Starting training run...")

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "training/train.py"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")

    logger.info(result.stdout)

    client = mlflow.tracking.MlflowClient(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment = client.get_experiment_by_name("agri-yield")
    # Guard against None experiment (first run before any experiment exists)
    if experiment is None:
        raise RuntimeError(
            "MLflow experiment 'agri-yield' not found. "
            "Run training/train.py at least once to create it."
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No MLflow runs found after training")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "rmse_val": float(run.data.metrics.get("rmse_val", 9999)),
    }


@task(name="promote-if-better")
def promote_if_better(new_run: dict[str, object], current_rmse: float) -> bool:
    """
    Promote new model to Production if RMSE improved by >5%.
    Returns True if promoted.
    """
    logger = get_run_logger()
    new_rmse = float(new_run["rmse_val"])  # type: ignore[arg-type]
    threshold = current_rmse * 0.95

    logger.info(
        f"New RMSE: {new_rmse:.2f} | Current: {current_rmse:.2f} | Threshold: {threshold:.2f}"
    )

    if new_rmse < threshold:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        model_name = "agri-yield-xgboost"
        run_id = str(new_run["run_id"])
        mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            f"Promoted version {mv.version} to Production (RMSE {new_rmse:.2f})"
        )
        RETRAIN_PROMOTED.inc()
        return True
    else:
        logger.info("New model not better enough — keeping current Production model")
        return False


@flow(
    name="agri-yield-retrain-trigger",
    description="Drift-triggered and scheduled retraining flow for agri-yield",
)
def retrain_trigger_flow(trigger_reason: str = "scheduled") -> dict[str, object]:
    """
    Main retraining flow.

    Args:
        trigger_reason: 'scheduled' | 'drift' | 'manual'
    """
    logger = get_run_logger()
    logger.info(f"Retraining triggered by: {trigger_reason}")

    RETRAIN_EVENTS.labels(trigger_reason=trigger_reason).inc()

    current_rmse = get_production_rmse()
    new_run = run_training()
    promoted = promote_if_better(new_run, current_rmse)

    LAST_RETRAIN_TIMESTAMP.set(datetime.utcnow().timestamp())

    return {
        "trigger_reason": trigger_reason,
        "new_run_id": new_run["run_id"],
        "new_rmse": new_run["rmse_val"],
        "promoted": promoted,
    }


if __name__ == "__main__":
    retrain_trigger_flow(trigger_reason="manual")
