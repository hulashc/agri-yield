"""
CI training script — fully self-contained, no local package imports.
Trains XGBoost on synthetic data and saves model.pkl to repo root.
"""

import os
import pickle
import sys

# Make sure repo root is on path so training.utils can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

TARGET = "yield_kg_per_ha"
OUTPUT_PATH = "model.pkl"

FEATURE_COLS = [
    "soil_temp_mean",
    "soil_temp_std",
    "moisture_mean",
    "moisture_std",
    "ph_mean",
    "nitrogen_mean",
    "phosphorus_mean",
    "potassium_mean",
    "air_temp_mean",
    "precip_total",
    "humidity_mean",
    "wind_speed_mean",
    "latest_ndvi",
    "cloud_cover_pct",
    "ndvi_interpolated",
    "ndvi_proxied",
]


def train_and_export(
    dataset_path: str = "data/features/weekly_field_features.parquet",
):
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df = df.dropna(subset=[TARGET])
    print(f"Loaded {len(df)} rows")

    # Fill any missing feature cols with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
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
