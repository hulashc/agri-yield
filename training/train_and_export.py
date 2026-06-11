"""
CI training script.

Trains three XGBoost models on real CYCleSS data (or synthetic fallback):
  1. mean_model  - standard squared loss regressor
  2. lower_model - quantile regressor at alpha=0.10 (lower bound of 80% PI)
  3. upper_model - quantile regressor at alpha=0.90 (upper bound of 80% PI)

Outputs:
  model_bundle.pkl  - ModelBundle (3 models + feature importance + training meta)
  model.pkl         - legacy mean model only (kept for backward compat)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb

from training.model_bundle import LOWER_QUANTILE, UPPER_QUANTILE, ModelBundle
from training.utils.features import FEATURE_COLS
from training.utils.splits import temporal_train_test_split

TARGET = "yield_kg_per_ha"
BUNDLE_PATH = Path("model_bundle.pkl")
LEGACY_PKL_PATH = Path("model.pkl")  # kept for backward compat
META_JSON_PATH = Path("training_meta.json")
DATASET_PATH = "data/features/weekly_field_features.parquet"


def ensure_dataset() -> None:
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


def _base_params() -> dict:
    """Shared hyperparameters for all three models."""
    return {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": 42,
        "n_jobs": -1,
    }


def train_and_export() -> dict:
    """
    Train 3 XGBoost models and save ModelBundle + legacy pickle.
    Returns a dict of eval metrics.
    """
    ensure_dataset()

    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    df = df.dropna(subset=[TARGET])
    print(f"Loaded {len(df)} rows | {df['field_id'].nunique()} fields | "
          f"{df['year'].min()}–{df['year'].max()} years")

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
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET]

    print(f"Split: {len(train_df)} train / {len(test_df)} test rows")

    # ------------------------------------------------------------------
    # 1. Mean model (standard squared-error loss)
    # ------------------------------------------------------------------
    print("Training mean model (squared loss)...")
    mean_model = xgb.XGBRegressor(**_base_params())
    mean_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    mean_preds = mean_model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_test.values - mean_preds) ** 2)))
    mae = float(np.mean(np.abs(y_test.values - mean_preds)))
    print(f"  Mean model  RMSE: {rmse:.1f} kg/ha | MAE: {mae:.1f} kg/ha")

    # ------------------------------------------------------------------
    # 2. Lower quantile model (q=0.10)
    # ------------------------------------------------------------------
    print(f"Training lower quantile model (q={LOWER_QUANTILE})...")
    lower_model = xgb.XGBRegressor(
        **_base_params(),
        objective="reg:quantileerror",
        quantile_alpha=LOWER_QUANTILE,
    )
    lower_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ------------------------------------------------------------------
    # 3. Upper quantile model (q=0.90)
    # ------------------------------------------------------------------
    print(f"Training upper quantile model (q={UPPER_QUANTILE})...")
    upper_model = xgb.XGBRegressor(
        **_base_params(),
        objective="reg:quantileerror",
        quantile_alpha=UPPER_QUANTILE,
    )
    upper_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ------------------------------------------------------------------
    # Quantile ordering check on test set
    # ------------------------------------------------------------------
    lower_preds = lower_model.predict(X_test)
    upper_preds = upper_model.predict(X_test)
    crossing_pct = float(np.mean(lower_preds > upper_preds) * 100)
    coverage = float(np.mean((y_test.values >= lower_preds) & (y_test.values <= upper_preds)) * 100)
    print(f"  Quantile crossing rate : {crossing_pct:.1f}% (enforced to 0% at serve time)")
    print(f"  Empirical PI coverage  : {coverage:.1f}% (target ≥80% for 80% PI)")

    # ------------------------------------------------------------------
    # Feature importance from mean model (mean gain)
    # ------------------------------------------------------------------
    raw_importance = mean_model.get_booster().get_score(importance_type="gain")
    # Normalise keys to match FEATURE_COLS (XGBoost uses f0, f1... if no names set)
    feature_importance: dict[str, float] = {}
    for feat_name in FEATURE_COLS:
        feature_importance[feat_name] = float(raw_importance.get(feat_name, 0.0))

    # ------------------------------------------------------------------
    # Training metadata (model card)
    # ------------------------------------------------------------------
    training_meta = {
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_path": DATASET_PATH,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_cols": FEATURE_COLS,
        "target": TARGET,
        "split_policy": "temporal — no future leakage",
        "lower_quantile": LOWER_QUANTILE,
        "upper_quantile": UPPER_QUANTILE,
        "interval_label": "80% prediction interval",
        "confidence_method": "xgboost_quantile_regression",
        "rmse_kg_ha": round(rmse, 2),
        "mae_kg_ha": round(mae, 2),
        "pi_coverage_pct": round(coverage, 2),
        "quantile_crossing_pct_raw": round(crossing_pct, 2),
        "xgboost_version": xgb.__version__,
    }

    # ------------------------------------------------------------------
    # Save bundle
    # ------------------------------------------------------------------
    bundle = ModelBundle(
        mean_model=mean_model,
        lower_model=lower_model,
        upper_model=upper_model,
        feature_importance=feature_importance,
        training_meta=training_meta,
    )
    bundle.save(BUNDLE_PATH)
    print(f"ModelBundle saved to {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1024:.1f} KB)")

    # Keep legacy model.pkl so older serving code / tools still work
    with open(LEGACY_PKL_PATH, "wb") as f:
        pickle.dump(mean_model, f)
    print(f"Legacy model.pkl saved to {LEGACY_PKL_PATH} ({LEGACY_PKL_PATH.stat().st_size / 1024:.1f} KB)")

    # Write human-readable meta alongside the bundle
    META_JSON_PATH.write_text(json.dumps(training_meta, indent=2))
    print(f"Training metadata written to {META_JSON_PATH}")

    return training_meta


if __name__ == "__main__":
    train_and_export()
