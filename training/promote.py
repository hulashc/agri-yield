"""
Model promotion script.
Compares a candidate model against the current production model.
Gates the staging → production transition in MLflow Model Registry.
"""

from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "agri-yield-xgb"
RMSE_IMPROVEMENT_THRESHOLD = 0.02  # candidate must be 2% better
CROP_REGRESSION_THRESHOLD = 0.05  # candidate must not regress >5% on any crop


def get_production_run_metrics(client: MlflowClient) -> dict[str, float] | None:
    """Fetch metrics from the current production model's training run."""
    try:
        prod_versions = client.get_latest_versions(
            REGISTERED_MODEL_NAME, stages=["Production"]
        )
        if not prod_versions:
            print("No production model found. Promoting candidate directly.")
            return None
        prod_run_id = prod_versions[0].run_id
        run = client.get_run(prod_run_id)
        return run.data.metrics
    except Exception as e:
        print(f"Could not fetch production metrics: {e}")
        return None


def get_candidate_run_metrics(client: MlflowClient) -> tuple[str, dict[str, float]]:
    """Fetch metrics from the latest staging model."""
    # NEW — use aliases or search_model_versions instead

    client = MlflowClient()
    staging_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    staging_versions = [v for v in staging_versions if v.current_stage == "Staging"]
    if not staging_versions:
        raise ValueError("No staging model found. Run training first.")
    candidate_version = staging_versions[0]
    run = client.get_run(candidate_version.run_id)
    return candidate_version.version, run.data.metrics


def should_promote(
    prod_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> tuple[bool, str]:
    """
    Returns (should_promote, reason).
    Checks overall RMSE improvement and per-crop regression.
    """
    prod_rmse = prod_metrics.get("holdout_rmse")
    cand_rmse = candidate_metrics.get("holdout_rmse")

    if prod_rmse is None or cand_rmse is None:
        return False, "Missing holdout_rmse in one or both runs."

    # Check 1: overall improvement
    improvement = (prod_rmse - cand_rmse) / prod_rmse
    if improvement < RMSE_IMPROVEMENT_THRESHOLD:
        return False, (
            f"Candidate RMSE {cand_rmse:.4f} does not beat production {prod_rmse:.4f} "
            f"by required {RMSE_IMPROVEMENT_THRESHOLD * 100:.0f}% (actual: {improvement * 100:.2f}%)"
        )

    # Check 2: per-crop regression
    crop_keys = {
        k.split("_rmse")[0]
        for k in prod_metrics
        if k.endswith("_rmse") and "holdout" not in k and "cv" not in k
    }
    for crop in crop_keys:
        prod_crop_rmse = prod_metrics.get(f"{crop}_rmse")
        cand_crop_rmse = candidate_metrics.get(f"{crop}_rmse")
        if prod_crop_rmse and cand_crop_rmse:
            regression = (cand_crop_rmse - prod_crop_rmse) / prod_crop_rmse
            if regression > CROP_REGRESSION_THRESHOLD:
                return False, (
                    f"Candidate regresses on crop '{crop}': "
                    f"RMSE went from {prod_crop_rmse:.4f} to {cand_crop_rmse:.4f} "
                    f"({regression * 100:.2f}% worse, threshold {CROP_REGRESSION_THRESHOLD * 100:.0f}%)"
                )

    return (
        True,
        f"Candidate RMSE {cand_rmse:.4f} beats production {prod_rmse:.4f} by {improvement * 100:.2f}%",
    )


def run_promotion():
    client = MlflowClient()

    prod_metrics = get_production_run_metrics(client)
    candidate_version, candidate_metrics = get_candidate_run_metrics(client)

    if prod_metrics is None:
        # No production model yet — promote directly
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=candidate_version,
            stage="Production",
        )
        print(f"First production model: version {candidate_version} promoted.")
        return

    promote, reason = should_promote(prod_metrics, candidate_metrics)
    print(f"Promotion decision: {'PROMOTE' if promote else 'REJECT'}")
    print(f"Reason: {reason}")

    if promote:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=candidate_version,
            stage="Production",
        )
        print(f"Model version {candidate_version} promoted to Production.")
    else:
        print("Candidate rejected. Production model unchanged.")


if __name__ == "__main__":
    run_promotion()
