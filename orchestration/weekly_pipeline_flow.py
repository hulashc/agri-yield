from __future__ import annotations

from prefect import flow, get_run_logger

from orchestration.ingest_flow import ingest_and_validate
from orchestration.materialise_flow import materialise_features
from orchestration.training_flow import run_training
from orchestration.promotion_flow import promote_if_better
from orchestration.monitoring_flow import run_monitoring


def send_alert(message: str) -> None:
    # Replace with Slack webhook or email in production
    print(f"[ALERT] {message}")


@flow(name="weekly-pipeline", log_prints=True)
def weekly_pipeline(use_synthetic: bool = False) -> dict:
    logger = get_run_logger()

    # Stage 1 — ingest + Great Expectations validation
    logger.info("Stage 1: Ingest and validate")
    validated = ingest_and_validate(use_synthetic=use_synthetic)
    if validated is False:
        logger.error("GE validation failed — aborting pipeline")
        send_alert("GE validation failed. Pipeline aborted.")
        raise RuntimeError("Ingest validation failed")

    # Stage 2 — Feast materialisation
    logger.info("Stage 2: Materialise features")
    materialise_features()

    # Stage 3 — Training
    logger.info("Stage 3: Train model")
    run_id = run_training()

    # Stage 4 — Promotion (no-op if challenger loses)
    logger.info("Stage 4: Promote if better")
    promoted = promote_if_better(run_id=run_id)
    if not promoted:
        logger.warning("Challenger did not beat champion — champion retained")

    # Stage 5 — Monitoring + drift report
    logger.info("Stage 5: Run monitoring")
    run_monitoring()

    logger.info("Weekly pipeline complete")
    return {"run_id": run_id, "promoted": promoted}


if __name__ == "__main__":
    # Run once immediately
    weekly_pipeline()


# To deploy with a weekly schedule, run this instead:
#
#   uv run python -c "
#   from orchestration.weekly_pipeline_flow import weekly_pipeline
#   weekly_pipeline.serve(
#       name='weekly-pipeline-deployment',
#       cron='0 3 * * 1',
#   )
#   "
