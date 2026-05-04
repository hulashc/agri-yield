"""
Naive baseline: predict mean yield per crop type.
This is the floor. XGBoost must beat this to be worth deploying.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from training.utils.metrics import compute_metrics, compute_metrics_by_crop
from training.utils.splits import temporal_train_test_split


class MeanByCropBaseline(BaseEstimator, RegressorMixin):
    """Predicts mean yield observed for each crop type in training data."""

    def __init__(self):
        self.crop_means_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanByCropBaseline":
        df = X.copy()
        df["_target"] = y.values
        self.crop_means_ = df.groupby("crop_type")["_target"].mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                self.crop_means_.get(row["crop_type"], self.global_mean_)
                for _, row in X.iterrows()
            ]
        )


def run_baseline():
    df = pd.read_parquet("data/features/weekly_field_features")
    df = df.dropna(subset=["yield_kg_per_ha"])

    train, test = temporal_train_test_split(df)

    feature_cols = [
        c for c in train.columns if c not in ["yield_kg_per_ha", "week_start"]
    ]
    X_train = train[feature_cols]
    y_train = train["yield_kg_per_ha"]
    X_test = test[feature_cols]
    y_test = test["yield_kg_per_ha"]

    model = MeanByCropBaseline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = compute_metrics(y_test.values, preds)

    test_with_preds = test.copy()
    test_with_preds["_pred"] = preds
    per_crop = compute_metrics_by_crop(test_with_preds, "yield_kg_per_ha", "_pred")

    with mlflow.start_run(run_name="baseline_v0"):
        mlflow.set_tag("model_type", "baseline_mean_by_crop")
        mlflow.log_metrics(metrics)

        for crop, crop_metrics in per_crop.items():
            for metric_name, value in crop_metrics.items():
                mlflow.log_metric(f"{crop}_{metric_name}", value)

        mlflow.sklearn.log_model(
            model, name="model", registered_model_name="agri-yield-baseline"
        )
        print(
            f"Baseline logged: RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R²={metrics['r2']:.4f}"
        )


if __name__ == "__main__":
    mlflow.set_experiment("agri-yield-training")
    run_baseline()
