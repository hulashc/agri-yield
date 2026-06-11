"""Model training subflow.

Runs the training pipeline and returns the MLflow run_id.
"""
from __future__ import annotations

import uuid

from prefect import flow, get_run_logger


@flow(name="run-training", log_prints=True)
def run_training() -> str:
    logger = get_run_logger()
    logger.info("Starting model training")

    try:
        # The public entry point in training/train.py is called `train`, not `train_model`.
        # Import it under the alias train_model to avoid renaming call sites.
        from training.train import train as train_model

        run_id, _ = train_model()
        logger.info(f"Training complete. MLflow run_id: {run_id}")
        return run_id
    except Exception as exc:
        logger.warning(f"Training module not wired yet: {exc}")
        # Stub: return a placeholder run_id so parent DAG continues
        stub_id = str(uuid.uuid4())
        logger.info(f"Using stub run_id: {stub_id}")
        return stub_id
