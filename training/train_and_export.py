"""
CI training script.
Trains XGBoost on real CYCleSS data when available, falls back to synthetic.
Saves model.pkl to repo root. Uses the canonical FEATURE_COLS.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb

from training.utils.features import FEATURE_COLS
from training.utils.splits import temporal_train_test_split

TARGET = "yield_kg_per_ha"
OUTPUT_PATH = "model.pkl"
DATASET_PATH = "data/features/weekly_field_features.parquet"


def ensure_dataset():
    """Prepare real CYCleSS data if available, otherwise generate synthetic data."""
    if not os.path.exists(DATASET_PATH):
        raw_dirs = [
            "cycless_data/CYCleSS_dataset/data/crop_yield_type_and_satellite_data",
            "data/raw/cycless/CYCleSS_dataset/data/crop_yield_type_and_satellite_data",
        ]
        if any(os.path.exists(d) for d in raw_dirs) or os.path.exists("data/raw/cycless.zip"):
            print("Real CYCleSS data found — processing into training features...")
            from training.prepare_real_data import prepare

            prepare()
        else:
            print("Real data not found — generating synthetic training data...")
            from scripts.archive.generate_data import generate

            generate()


def train_and_export():
    ensure_dataset()
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    df = df.dropna(subset=[TARGET])
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  [warn] missing column '{col}' — filling with 0")
            df[col] = 0

    # Derive week_start from year/week_of_year for temporal split
    df["week_start"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week_of_year"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    # Temporal split — never shuffles, never leaks future into past
    train_df, test_df = temporal_train_test_split(df)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET]

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
