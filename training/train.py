"""
XGBoost training script — multi-model bundle with quantile confidence intervals.
Trains three models:
  - mean model  (XGBRegressor, standard MSE loss)
  - lower model (XGBRegressor, quantile alpha=0.10)
  - upper model (XGBRegressor, quantile alpha=0.90)

Saves model_bundle.pkl with full metadata so /model/info can surface:
  training timestamp, RMSE/MAE, CI coverage, interval width, top features,
  split policy, n_train, n_test.
"""

import argparse
import os
import pickle
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from training.utils.features import get_feature_cols
from training.utils.metrics import compute_metrics, compute_metrics_by_crop
from training.utils.splits import temporal_train_test_split

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

TARGET = "yield_kg_per_ha"
REGISTERED_MODEL_NAME = "agri-yield-xgb"

_REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = _REPO_ROOT / "model_bundle.pkl"
PICKLE_PATH = _REPO_ROOT / "model.pkl"


def _build_quantile_params(base_params: dict, alpha: float) -> dict:
    """Return XGBoost params configured for quantile regression at `alpha`."""
    p = {k: v for k, v in base_params.items() if k not in ("random_state",)}
    p["objective"] = "reg:quantileerror"
    p["quantile_alpha"] = alpha
    p["seed"] = base_params.get("random_state", 42)
    return p


def train(
    dataset_path: str = "data/features/weekly_field_features",
    params: dict | None = None,
    dvc_commit: str = "unknown",
    n_cv_folds: int = 5,
):
    if params is None:
        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "random_state": 42,
            "n_jobs": -1,
        }

    df = pd.read_parquet(dataset_path)
    df = df.dropna(subset=[TARGET])

    for col in ["ndvi_interpolated", "ndvi_proxied"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if df["crop_type"].dtype == object:
        df["crop_type"] = LabelEncoder().fit_transform(df["crop_type"])

    train_df, test_df = temporal_train_test_split(df)
    feature_cols = get_feature_cols(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    n_train = len(X_train)
    n_test = len(X_test)

    # ── Cross-validation on mean model ──────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    cv_rmse_scores, cv_mae_scores, cv_r2_scores = [], [], []
    X_train_arr = X_train.values
    y_train_arr = y_train.values

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_arr)):
        fold_model = xgb.XGBRegressor(**params)
        fold_model.fit(
            X_train_arr[train_idx],
            y_train_arr[train_idx],
            eval_set=[(X_train_arr[val_idx], y_train_arr[val_idx])],
            verbose=False,
        )
        fold_preds = fold_model.predict(X_train_arr[val_idx])
        fold_metrics = compute_metrics(y_train_arr[val_idx], fold_preds)
        cv_rmse_scores.append(fold_metrics["rmse"])
        cv_mae_scores.append(fold_metrics["mae"])
        cv_r2_scores.append(fold_metrics["r2"])

    # ── Train mean model ─────────────────────────────────────────────────────
    mean_model = xgb.XGBRegressor(**params)
    mean_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    test_preds = mean_model.predict(X_test)
    holdout_metrics = compute_metrics(y_test.values, test_preds)

    # ── Global feature importance (XGBoost gain, top 15) ────────────────────
    importances = mean_model.feature_importances_
    top_features = sorted(
        [
            {"rank": i + 1, "feature": name, "importance": round(float(imp), 6), "source": "xgboost_gain"}
            for i, (name, imp) in enumerate(
                sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:15]
            )
        ],
        key=lambda x: x["rank"],
    )

    # ── Train quantile models (10th and 90th percentile) ─────────────────────
    lower_params = _build_quantile_params(params, alpha=0.10)
    upper_params = _build_quantile_params(params, alpha=0.90)

    lower_model = xgb.XGBRegressor(**lower_params)
    lower_model.fit(X_train, y_train, verbose=False)
    lower_preds = lower_model.predict(X_test)

    upper_model = xgb.XGBRegressor(**upper_params)
    upper_model.fit(X_train, y_train, verbose=False)
    upper_preds = upper_model.predict(X_test)

    # Quantile crossing correction: ensure lower <= mean <= upper per sample
    lower_preds = np.minimum(lower_preds, test_preds)
    upper_preds = np.maximum(upper_preds, test_preds)

    quantile_rmse_lower = float(np.sqrt(np.mean((lower_preds - y_test.values) ** 2)))
    quantile_rmse_upper = float(np.sqrt(np.mean((upper_preds - y_test.values) ** 2)))
    coverage = float(np.mean((y_test.values >= lower_preds) & (y_test.values <= upper_preds)))
    avg_interval_width = float(np.mean(upper_preds - lower_preds))

    # ── Per-crop metrics ─────────────────────────────────────────────────────
    test_df_copy = test_df.copy()
    test_df_copy["_pred"] = test_preds
    per_crop_metrics = compute_metrics_by_crop(test_df_copy, TARGET, "_pred")

    # ── Save model bundle (enriched metadata for /model/info) ─────────────────
    trained_at = datetime.now(UTC).isoformat()
    bundle = {
        "mean_model": mean_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        # — Metadata surfaced by /model/info —
        "model_version": f"bundle-{trained_at[:10]}",
        "trained_at": trained_at,
        "split_policy": "temporal_train_test_split",
        "n_train": n_train,
        "n_test": n_test,
        "feature_cols": feature_cols,
        "holdout_rmse": round(holdout_metrics["rmse"], 4),
        "holdout_mae": round(holdout_metrics["mae"], 4),
        "holdout_r2": round(holdout_metrics["r2"], 4),
        "coverage_80pct": round(coverage, 4),
        "avg_interval_width_kg_ha": round(avg_interval_width, 1),
        "quantile_lower": 0.10,
        "quantile_upper": 0.90,
        "top_features": top_features,
        "dvc_commit": dvc_commit,
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    # Also save mean model as standalone pickle for backward compatibility
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(mean_model, f)

    print(f"\nModel bundle saved to {BUNDLE_PATH}")
    print(f"Trained at:            {trained_at}")
    print(f"n_train={n_train}  n_test={n_test}")
    print(f"Holdout RMSE:          {holdout_metrics['rmse']:.4f} kg/ha")
    print(f"Holdout MAE:           {holdout_metrics['mae']:.4f} kg/ha")
    print(f"CI coverage (80% band): {coverage:.1%}")
    print(f"Avg interval width:    {avg_interval_width:.1f} kg/ha")

    # ── MLflow logging ───────────────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost_quantile_run") as run:
        mlflow.log_params(params)
        mlflow.log_param("dvc_dataset_commit", dvc_commit)
        mlflow.log_param("n_cv_folds", n_cv_folds)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("quantile_lower", 0.10)
        mlflow.log_param("quantile_upper", 0.90)
        mlflow.log_param("n_train", n_train)
        mlflow.log_param("n_test", n_test)
        mlflow.log_param("split_policy", "temporal_train_test_split")
        mlflow.log_param("trained_at", trained_at)

        for i, (rmse, mae, r2) in enumerate(zip(cv_rmse_scores, cv_mae_scores, cv_r2_scores)):
            mlflow.log_metric(f"cv_fold_{i}_rmse", rmse)
            mlflow.log_metric(f"cv_fold_{i}_mae", mae)
            mlflow.log_metric(f"cv_fold_{i}_r2", r2)

        mlflow.log_metric("cv_rmse_mean", float(np.mean(cv_rmse_scores)))
        mlflow.log_metric("cv_rmse_std", float(np.std(cv_rmse_scores)))
        mlflow.log_metric("cv_mae_mean", float(np.mean(cv_mae_scores)))
        mlflow.log_metric("cv_r2_mean", float(np.mean(cv_r2_scores)))
        mlflow.log_metric("holdout_rmse", holdout_metrics["rmse"])
        mlflow.log_metric("holdout_mae", holdout_metrics["mae"])
        mlflow.log_metric("holdout_r2", holdout_metrics["r2"])
        mlflow.log_metric("quantile_rmse_lower", quantile_rmse_lower)
        mlflow.log_metric("quantile_rmse_upper", quantile_rmse_upper)
        mlflow.log_metric("ci_coverage_80pct", coverage)
        mlflow.log_metric("ci_avg_width_kg_ha", avg_interval_width)

        for crop, metrics in per_crop_metrics.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"{crop}_{k}", v)

        logged = mlflow.xgboost.log_model(mean_model, name="model")
        model_uri = logged.model_uri
        run_id = run.info.run_id

        result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

        client = MlflowClient()
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="challenger",
            version=result.version,
        )
        print(f"Model v{result.version} registered as '{REGISTERED_MODEL_NAME}' → alias=challenger")
        print(f"Run ID: {run_id}")
        return run_id, holdout_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/features/weekly_field_features")
    parser.add_argument("--dvc-commit", default="unknown")
    args = parser.parse_args()
    mlflow.set_experiment("agri-yield-training")
    train(dataset_path=args.dataset_path, dvc_commit=args.dvc_commit)
