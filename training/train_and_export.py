"""
Standalone training script for CI.
Trains XGBoost on synthetic data and exports model.pkl to repo root.
No MLflow required — designed to run in GitHub Actions.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from training.utils.features import FEATURE_COLS

TARGET = "yield_kg_per_ha"
OUTPUT_PATH = "model.pkl"


def train_and_export(dataset_path: str = "data/features/weekly_field_features.parquet"):
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df = df.dropna(subset=[TARGET])

    # Encode crop_type if still string
    if df["crop_type"].dtype == object:
        df["crop_type"] = LabelEncoder().fit_transform(df["crop_type"])

    # Only keep feature cols that exist
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df[TARGET]

    # Fill any missing feature cols with 0
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    print("Training XGBoost...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test.values - preds) ** 2)))
    print(f"Holdout RMSE: {rmse:.2f}")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {OUTPUT_PATH}")
    return rmse


if __name__ == "__main__":
    train_and_export()
