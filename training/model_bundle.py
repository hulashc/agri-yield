"""
training/model_bundle.py

ModelBundle: a typed container for the three XGBoost models that power
statistically-derived prediction intervals.

The bundle stores:
  - mean_model   : standard XGBRegressor trained on squared loss
  - lower_model  : XGBRegressor trained with quantile loss at alpha=LOWER_QUANTILE
  - upper_model  : XGBRegressor trained with quantile loss at alpha=UPPER_QUANTILE
  - feature_importance : dict mapping feature name -> mean gain (from mean_model)
  - training_meta      : dict with dataset info, RMSE, MAE, split policy, timestamp

Save / load helpers use pickle for portability. In Phase 7 this will be
replaced by an MLflow model registry workflow.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOWER_QUANTILE: float = 0.10  # 80% prediction interval (10th — 90th percentile)
UPPER_QUANTILE: float = 0.90


@dataclass
class ModelBundle:
    """Three-model bundle for point prediction + calibrated prediction intervals."""

    mean_model: Any  # XGBRegressor (squared loss)
    lower_model: Any  # XGBRegressor (quantile loss, alpha=0.10)
    upper_model: Any  # XGBRegressor (quantile loss, alpha=0.90)
    feature_importance: dict[str, float] = field(default_factory=dict)
    training_meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, X: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mean_preds, lower_preds, upper_preds).

        Quantile ordering is enforced: lower <= mean <= upper.
        This handles the rare case of quantile crossing on extrapolation.
        """
        mean = self.mean_model.predict(X)
        lower = self.lower_model.predict(X)
        upper = self.upper_model.predict(X)

        # Enforce monotonic ordering — quantile crossing fix
        lower = np.minimum(lower, mean)
        upper = np.maximum(upper, mean)

        return mean, lower, upper

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)  # noqa: S301

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ModelBundle not found at {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ModelBundle, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def top_features(self, n: int = 10) -> list[dict[str, Any]]:
        """Return top-N features by mean gain, sorted descending."""
        sorted_items = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {"feature": name, "importance": round(score, 4)}
            for name, score in sorted_items[:n]
        ]

    def interval_width_stats(self, X: Any) -> dict[str, float]:
        """Compute mean, median, p95 of interval width on a given feature matrix."""
        _, lower, upper = self.predict(X)
        widths = upper - lower
        return {
            "mean_width_kg_ha": round(float(np.mean(widths)), 1),
            "median_width_kg_ha": round(float(np.median(widths)), 1),
            "p95_width_kg_ha": round(float(np.percentile(widths, 95)), 1),
        }
