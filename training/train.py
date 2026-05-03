"""
XGBoost training script. Accepts a DVC dataset path, trains, evaluates,
and registers the model to MLflow. Callable by Prefect.
"""

# training/train.py — top of file, before anything else
import argparse

import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from training.utils.metrics import compute_metrics, compute_metrics_by_crop
from training.utils.splits import temporal_train_test_split
from training.utils.features import get_feature_cols

TARGET = "yield_kg_per_ha"
NON_FEATURE_COLS = [TARGET, "week_start", "field_id"]


def train(
    dataset_path: str = "data/features/weekly_field_features",
    params: dict | None = None,
    dvc_commit: str = "unknown",
    n_cv_folds: int = 5,
):
    """
    Load features, train XGBoost with cross-validation, log everything to MLflow.
    """
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

    # Convert bool flags to int for XGBoost
    for col in ["ndvi_interpolated", "ndvi_proxied"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    train_df, test_df = temporal_train_test_split(df)
    feature_cols = get_feature_cols(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    model = xgb.XGBRegressor(**params)

    # Time-aware cross-validation on training set only
    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []

    # In your train() function, before building X_train_arr

    if df["crop_type"].dtype == object:
        df["crop_type"] = LabelEncoder().fit_transform(df["crop_type"])

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

    # Final model trained on full training set
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    test_preds = model.predict(X_test)
    holdout_metrics = compute_metrics(y_test.values, test_preds)

    test_df_copy = test_df.copy()
    test_df_copy["_pred"] = test_preds
    per_crop_metrics = compute_metrics_by_crop(test_df_copy, TARGET, "_pred")

    with mlflow.start_run(run_name="xgboost_run") as run:
        mlflow.log_params(params)
        mlflow.log_param("dvc_dataset_commit", dvc_commit)
        mlflow.log_param("n_cv_folds", n_cv_folds)
        mlflow.log_param("feature_count", len(feature_cols))

        # CV metrics per fold
        for i, (rmse, mae, r2) in enumerate(
            zip(cv_rmse_scores, cv_mae_scores, cv_r2_scores)
        ):
            mlflow.log_metric(f"cv_fold_{i}_rmse", rmse)
            mlflow.log_metric(f"cv_fold_{i}_mae", mae)
            mlflow.log_metric(f"cv_fold_{i}_r2", r2)

        # Aggregated CV metrics
        mlflow.log_metric("cv_rmse_mean", float(np.mean(cv_rmse_scores)))
        mlflow.log_metric("cv_rmse_std", float(np.std(cv_rmse_scores)))
        mlflow.log_metric("cv_mae_mean", float(np.mean(cv_mae_scores)))
        mlflow.log_metric("cv_r2_mean", float(np.mean(cv_r2_scores)))

        # Holdout metrics
        mlflow.log_metric("holdout_rmse", holdout_metrics["rmse"])
        mlflow.log_metric("holdout_mae", holdout_metrics["mae"])
        mlflow.log_metric("holdout_r2", holdout_metrics["r2"])

        # Per-crop metrics
        for crop, metrics in per_crop_metrics.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"{crop}_{k}", v)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="agri-yield-xgboost",
        )

        REGISTERED_MODEL_NAME = "agri-yield-xgb"  # must match promote.py exactly

        # Inside your mlflow.start_run() block, after mlflow.log_model(...)
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        # Register the model
        result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

        # Transition to Staging so promote.py can find it
        client = MlflowClient()
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=result.version,
            stage="Staging",
        )
        print(f"Model v{result.version} registered and moved to Staging")

        print(f"Run ID: {run.info.run_id}")
        print(f"Holdout RMSE: {holdout_metrics['rmse']:.4f}")
        return run.info.run_id, holdout_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/features/weekly_field_features")
    parser.add_argument("--dvc-commit", default="unknown")
    args = parser.parse_args()

    mlflow.set_experiment("agri-yield-training")
    train(dataset_path=args.dataset_path, dvc_commit=args.dvc_commit)
