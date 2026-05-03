"""Model promotion subflow.

Compares challenger vs champion RMSE. Promotes if challenger is better.
Returns True if promoted, False if champion retained.
"""
from __future__ import annotations

from prefect import flow, get_run_logger


@flow(name="promote-if-better", log_prints=True)
def promote_if_better(run_id: str) -> bool:
    logger = get_run_logger()
    logger.info(f"Checking promotion for run_id: {run_id}")

    try:
        import mlflow
        client = mlflow.MlflowClient()

        # Get challenger metrics
        run = client.get_run(run_id)
        challenger_rmse = run.data.metrics.get("rmse", None)

        if challenger_rmse is None:
            logger.warning("No RMSE metric found on challenger run — skipping promotion")
            return False

        # Get champion RMSE (latest Production model)
        champion_versions = client.get_latest_versions("yield-model", stages=["Production"])
        if not champion_versions:
            # No champion yet — promote automatically
            client.transition_model_version_stage(
                name="yield-model",
                version=client.get_latest_versions("yield-model")[0].version,
                stage="Production",
            )
            logger.info("No existing champion — challenger promoted automatically")
            return True

        champion_run_id = champion_versions[0].run_id
        champion_run = client.get_run(champion_run_id)
        champion_rmse = champion_run.data.metrics.get("rmse", float("inf"))

        if challenger_rmse < champion_rmse:
            client.transition_model_version_stage(
                name="yield-model",
                version=client.get_latest_versions("yield-model")[-1].version,
                stage="Production",
            )
            logger.info(f"Challenger promoted: {challenger_rmse:.2f} < {champion_rmse:.2f}")
            return True
        else:
            logger.info(f"Champion retained: {champion_rmse:.2f} <= {challenger_rmse:.2f}")
            return False

    except Exception as exc:
        logger.warning(f"Promotion check failed (stub mode): {exc}")
        return False
