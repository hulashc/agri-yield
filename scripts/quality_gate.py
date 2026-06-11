#!/usr/bin/env python
"""
Model quality gate — shared between ci.yml and deploy.yml.

Usage:
    uv run python scripts/quality_gate.py

Exits with code 1 if RMSE >= RMSE_THRESHOLD_KG_HA.
Exits with code 0 on pass.

The script writes a JSON artefact to scripts/quality_gate_result.json
so downstream CI steps can read it without re-running inference.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RMSE_THRESHOLD_KG_HA: float = 2000.0
FEATURES_PARQUET = Path("data/features/weekly_field_features.parquet")
MODEL_PKL = Path("model.pkl")
RESULT_PATH = Path("scripts/quality_gate_result.json")
TARGET = "yield_kg_per_ha"


def run_gate() -> dict:
    """Run quality gate and return result dict."""
    from training.utils.features import FEATURE_COLS
    from training.utils.splits import temporal_train_test_split

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"Feature parquet not found at {FEATURES_PARQUET}. "
            "Run training/train_and_export.py first."
        )

    df = pd.read_parquet(FEATURES_PARQUET)
    df = df.dropna(subset=[TARGET])

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df["week_start"] = pd.to_datetime(
        df["year"].astype(str)
        + "-W"
        + df["week_of_year"].astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    _, test_df = temporal_train_test_split(df)

    if test_df.empty:
        raise ValueError("Temporal split produced an empty test set — check data range.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Model pickle not found at {MODEL_PKL}.")

    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    preds = model.predict(test_df[FEATURE_COLS])
    actuals = test_df[TARGET].values
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mae = float(np.mean(np.abs(actuals - preds)))
    n_test = int(len(test_df))

    result = {
        "rmse_kg_ha": round(rmse, 2),
        "mae_kg_ha": round(mae, 2),
        "n_test_samples": n_test,
        "threshold_kg_ha": RMSE_THRESHOLD_KG_HA,
        "passed": rmse < RMSE_THRESHOLD_KG_HA,
    }

    # ------------------------------------------------------------------
    # Write artefact
    # ------------------------------------------------------------------
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))

    return result


def main() -> None:
    result = run_gate()

    print("\n==== Quality Gate Result ====")
    print(f"  RMSE   : {result['rmse_kg_ha']:.2f} kg/ha")
    print(f"  MAE    : {result['mae_kg_ha']:.2f} kg/ha")
    print(f"  N test : {result['n_test_samples']}")
    print(f"  Threshold : {result['threshold_kg_ha']:.0f} kg/ha")
    print("  Status :", "PASSED ✓" if result["passed"] else "FAILED ✗")
    print("==============================\n")

    if not result["passed"]:
        print(
            f"ERROR: RMSE {result['rmse_kg_ha']:.2f} kg/ha exceeds threshold "
            f"{result['threshold_kg_ha']:.0f} kg/ha — failing CI.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
