"""
CI training script.
Trains XGBoost on real CYCleSS data when available, falls back to synthetic.

Saves model_bundle.pkl to repo root containing:
  - 'mean':  XGBRegressor (point prediction)
  - 'lower': XGBRegressor objective=reg:quantileerror alpha=0.10
  - 'upper': XGBRegressor objective=reg:quantileerror alpha=0.90
  - 'rmse':  holdout RMSE of the mean model
  - 'feature_importance': dict of feature -> importance score
  - 'trained_at': ISO timestamp
  - 'dataset_source': 'cycless' or 'synthetic'

Also keeps backward-compat model.pkl (mean model only) for legacy loaders.
"""

import os
import pickle
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb

from training.utils.features import FEATURE_COLS
from training.utils.splits import temporal_train_test_split

TARGET = "yield_kg_per_ha"
OUTPUT_BUNDLE = "model_bundle.pkl"
OUTPUT_LEGACY = "model.pkl"
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
            return "cycless"
        else:
            print("Real data not found — generating synthetic training data...")
            from scripts.archive.generate_data import generate
            generate()
            return "synthetic"
    return "cycless" if os.path.exists("data/raw/cycless") else "synthetic"


def train_and_export():
    dataset_source = ensure_dataset()
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    df = df.dropna(subset=[TARGET])
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  [warn] missing column '{col}' — filling with 0")
            df[col] = 0

    df["week_start"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week_of_year"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    train_df, test_df = temporal_train_test_split(df)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET]

    base_params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": 42,
        "n_jobs": -1,
    }

    # --- Mean model (point prediction) ---
    print("Training mean model...")
    mean_model = xgb.XGBRegressor(**base_params)
    mean_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = mean_model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test.values - preds) ** 2)))
    print(f"Mean model holdout RMSE: {rmse:.2f} kg/ha")

    # --- Lower quantile model (q=0.10) ---
    print("Training lower quantile model (q=0.10)...")
    lower_model = xgb.XGBRegressor(
        **{**base_params, "objective": "reg:quantileerror", "quantile_alpha": 0.10}
    )
    lower_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # --- Upper quantile model (q=0.90) ---
    print("Training upper quantile model (q=0.90)...")
    upper_model = xgb.XGBRegressor(
        **{**base_params, "objective": "reg:quantileerror", "quantile_alpha": 0.90}
    )
    upper_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Validate ordering: lower <= mean <= upper on test set
    lower_preds = lower_model.predict(X_test)
    upper_preds = upper_model.predict(X_test)
    ordering_violations = int(np.sum((lower_preds > preds) | (preds > upper_preds)))
    print(f"CI ordering violations on test set: {ordering_violations} / {len(preds)}")

    # Feature importance from mean model
    importance = {
        feat: float(score)
        for feat, score in zip(FEATURE_COLS, mean_model.feature_importances_, strict=False)
    }

    bundle = {
        "mean": mean_model,
        "lower": lower_model,
        "upper": upper_model,
        "rmse": rmse,
        "feature_importance": importance,
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_source": dataset_source,
        "feature_cols": FEATURE_COLS,
        "quantile_lower": 0.10,
        "quantile_upper": 0.90,
        "ci_coverage": "80%",
        "ordering_violations": ordering_violations,
    }

    with open(OUTPUT_BUNDLE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Bundle saved to {OUTPUT_BUNDLE} ({os.path.getsize(OUTPUT_BUNDLE) / 1024:.1f} KB)")

    # Keep backward-compat model.pkl (mean model only)
    with open(OUTPUT_LEGACY, "wb") as f:
        pickle.dump(mean_model, f)
    print(f"Legacy model.pkl saved ({os.path.getsize(OUTPUT_LEGACY) / 1024:.1f} KB)")

    # CI quality gate: RMSE must be below threshold
    rmse_threshold = float(os.getenv("RMSE_THRESHOLD", "2000"))
    if rmse > rmse_threshold:
        print(f"QUALITY GATE FAILED: RMSE {rmse:.2f} > {rmse_threshold}")
        sys.exit(1)

    return rmse


if __name__ == "__main__":
    train_and_export()
