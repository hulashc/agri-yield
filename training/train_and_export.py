"""
CI training script.
Trains XGBoost on synthetic data and saves model.pkl to repo root.
Uses the canonical FEATURE_COLS from training.utils.features.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from training.utils.features import FEATURE_COLS

TARGET = "yield_kg_per_ha"
OUTPUT_PATH = "model.pkl"


def train_and_export(
    dataset_path: str = "data/features/weekly_field_features.parquet",
):
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df = df.dropna(subset=[TARGET])
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  [warn] missing column '{col}' — filling with 0")
            df[col] = 0

    X = df[FEATURE_COLS]
    y = df[TARGET]

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

    print(f"Model saved to {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB)")
    return rmse


if __name__ == "__main__":
    train_and_export()
