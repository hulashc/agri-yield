"""
Model promotion script.
Compares challenger alias against champion alias.
If challenger is better, it becomes the new champion.
Uses MLflow aliases — compatible with MLflow 3.x (stages are deprecated).
"""

from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "agri-yield-xgb"
RMSE_IMPROVEMENT_THRESHOLD = 0.02   # challenger must be 2% better
CROP_REGRESSION_THRESHOLD = 0.05    # challenger must not regress >5% on any crop


def get_version_by_alias(client: MlflowClient, alias: str):
    """Return model version object for an alias, or None if not set."""
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    except Exception:
        return None


def get_metrics(client: MlflowClient, version) -> dict[str, float]:
    run = client.get_run(version.run_id)
    return run.data.metrics


def should_promote(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
) -> tuple[bool, str]:
    champ_rmse = champion_metrics.get("holdout_rmse")
    chal_rmse = challenger_metrics.get("holdout_rmse")

    if champ_rmse is None or chal_rmse is None:
        return False, "Missing holdout_rmse in one or both runs."

    improvement = (champ_rmse - chal_rmse) / champ_rmse
    if improvement < RMSE_IMPROVEMENT_THRESHOLD:
        return False, (
            f"Challenger RMSE {chal_rmse:.4f} does not beat champion {champ_rmse:.4f} "
            f"by required {RMSE_IMPROVEMENT_THRESHOLD * 100:.0f}% (actual: {improvement * 100:.2f}%)"
        )

    crop_keys = {
        k.split("_rmse")[0]
        for k in champion_metrics
        if k.endswith("_rmse") and "holdout" not in k and "cv" not in k
    }
    for crop in crop_keys:
        champ_crop = champion_metrics.get(f"{crop}_rmse")
        chal_crop = challenger_metrics.get(f"{crop}_rmse")
        if champ_crop and chal_crop:
            regression = (chal_crop - champ_crop) / champ_crop
            if regression > CROP_REGRESSION_THRESHOLD:
                return False, (
                    f"Challenger regresses on '{crop}': "
                    f"{champ_crop:.4f} → {chal_crop:.4f} ({regression * 100:.2f}% worse)"
                )

    return True, f"Challenger RMSE {chal_rmse:.4f} beats champion {champ_rmse:.4f} by {improvement * 100:.2f}%"


def run_promotion():
    client = MlflowClient()

    challenger = get_version_by_alias(client, "challenger")
    if challenger is None:
        raise ValueError("No 'challenger' alias found. Run training/train.py first.")

    champion = get_version_by_alias(client, "champion")

    if champion is None:
        # First ever model — promote directly
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", challenger.version)
        print(f"No champion yet — version {challenger.version} is now champion.")
        return True

    challenger_metrics = get_metrics(client, challenger)
    champion_metrics = get_metrics(client, champion)

    promote, reason = should_promote(champion_metrics, challenger_metrics)
    print(f"Promotion decision: {'PROMOTE' if promote else 'REJECT'}")
    print(f"Reason: {reason}")

    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", challenger.version)
        print(f"Version {challenger.version} is now champion.")
    else:
        print("Challenger rejected. Champion unchanged.")

    return promote


if __name__ == "__main__":
    run_promotion()
