#!/usr/bin/env python
"""
Model quality gate — shared between ci.yml and deploy.yml.

Usage:
    uv run python scripts/quality_gate.py

Exits with code 1 if:
  - RMSE >= RMSE_THRESHOLD_KG_HA
  - Quantile ordering violated (lower > upper on any test sample)
  - PI empirical coverage < MIN_COVERAGE_PCT

Exits with code 0 on pass.

Writes scripts/quality_gate_result.json as a CI artefact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RMSE_THRESHOLD_KG_HA: float = 2000.0
MIN_COVERAGE_PCT: float = 60.0   # minimum empirical PI coverage on test set
FEATURES_PARQUET = Path("data/features/weekly_field_features.parquet")
BUNDLE_PATH = Path("model_bundle.pkl")
LEGACY_PKL_PATH = Path("model.pkl")
RESULT_PATH = Path("scripts/quality_gate_result.json")
TARGET = "yield_kg_per_ha"


def _load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load parquet, apply temporal split, return (X_test, y_test)."""
    from training.utils.features import FEATURE_COLS
    from training.utils.splits import temporal_train_test_split

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
        raise ValueError("Temporal split produced an empty test set.")

    from training.utils.features import FEATURE_COLS as FC
    return test_df[FC], test_df[TARGET]


def run_gate() -> dict:
    """Run quality gate against ModelBundle (preferred) or legacy pickle."""
    X_test, y_test = _load_test_data()
    actuals = y_test.values

    using_bundle = BUNDLE_PATH.exists()

    if using_bundle:
        from training.model_bundle import ModelBundle
        bundle = ModelBundle.load(BUNDLE_PATH)
        mean_preds, lower_preds, upper_preds = bundle.predict(X_test)
        confidence_method = "xgboost_quantile_regression"
    else:
        import pickle
        if not LEGACY_PKL_PATH.exists():
            raise FileNotFoundError("Neither model_bundle.pkl nor model.pkl found.")
        with open(LEGACY_PKL_PATH, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        mean_preds = model.predict(X_test)
        lower_preds = mean_preds * 0.90
        upper_preds = mean_preds * 1.10
        confidence_method = "legacy_symmetric_ci"

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    rmse = float(np.sqrt(np.mean((actuals - mean_preds) ** 2)))
    mae = float(np.mean(np.abs(actuals - mean_preds)))
    n_test = int(len(y_test))

    # Quantile ordering: after ModelBundle's crossing fix, lower <= upper always
    ordering_violations = int(np.sum(lower_preds > upper_preds))

    # Empirical PI coverage
    coverage_pct = float(
        np.mean((actuals >= lower_preds) & (actuals <= upper_preds)) * 100
    )

    result = {
        "rmse_kg_ha": round(rmse, 2),
        "mae_kg_ha": round(mae, 2),
        "n_test_samples": n_test,
        "threshold_kg_ha": RMSE_THRESHOLD_KG_HA,
        "pi_coverage_pct": round(coverage_pct, 2),
        "min_coverage_pct": MIN_COVERAGE_PCT,
        "ordering_violations": ordering_violations,
        "confidence_method": confidence_method,
        "using_bundle": using_bundle,
        "passed": (
            rmse < RMSE_THRESHOLD_KG_HA
            and ordering_violations == 0
            and coverage_pct >= MIN_COVERAGE_PCT
        ),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    result = run_gate()

    print("\n==== Quality Gate Result ====")
    print(f"  Method          : {result['confidence_method']}")
    print(f"  Using bundle    : {result['using_bundle']}")
    print(f"  RMSE            : {result['rmse_kg_ha']:.2f} kg/ha  (threshold: {result['threshold_kg_ha']:.0f})")
    print(f"  MAE             : {result['mae_kg_ha']:.2f} kg/ha")
    print(f"  N test samples  : {result['n_test_samples']}")
    print(f"  PI coverage     : {result['pi_coverage_pct']:.1f}%  (min: {result['min_coverage_pct']:.0f}%)")
    print(f"  Order violations: {result['ordering_violations']}  (must be 0)")
    print(f"  Status          : ", "PASSED ✓" if result["passed"] else "FAILED ✗")
    print("==============================\n")

    if not result["passed"]:
        failures = []
        if result["rmse_kg_ha"] >= result["threshold_kg_ha"]:
            failures.append(f"RMSE {result['rmse_kg_ha']:.2f} >= threshold {result['threshold_kg_ha']:.0f} kg/ha")
        if result["ordering_violations"] > 0:
            failures.append(f"{result['ordering_violations']} quantile ordering violations (lower > upper)")
        if result["pi_coverage_pct"] < result["min_coverage_pct"]:
            failures.append(f"PI coverage {result['pi_coverage_pct']:.1f}% < minimum {result['min_coverage_pct']:.0f}%")
        print("FAILURES:", " | ".join(failures), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
