"""
serving/explainability.py

SHAP-based explainability for the agri-yield model.

Provides:
  - get_global_importance()  → ranked list of features by mean |SHAP|
  - get_local_explanation()  → per-feature SHAP contributions for a single row

The TreeExplainer is built once (lazy, on first call) and reused.
If SHAP is not installed or the model is not loaded, both functions
return graceful fallbacks so the API never breaks.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from serving.model import _prepare_features, get_model, has_quantile_models, is_loaded
from training.utils.features import FEATURE_COLS

log = logging.getLogger(__name__)

_explainer = None
_background_data: np.ndarray | None = None


def _get_explainer():
    """Lazy-load the SHAP TreeExplainer against the mean model."""
    global _explainer
    if _explainer is not None:
        return _explainer
    if not is_loaded():
        return None
    try:
        import shap  # noqa: PLC0415
        model = get_model()
        _explainer = shap.TreeExplainer(model)
        log.info("SHAP TreeExplainer initialised against mean model.")
        return _explainer
    except ImportError:
        log.warning("shap not installed — explainability endpoints will return fallback.")
        return None
    except Exception as exc:
        log.warning("Could not build SHAP explainer: %s", exc)
        return None


def get_global_importance(top_n: int = 15) -> list[dict[str, Any]]:
    """
    Return global feature importance as ranked list.

    Falls back to XGBoost's built-in feature_importances_ if SHAP unavailable.

    Returns:
        List of {"feature": str, "importance": float, "rank": int}
        sorted descending by importance.
    """
    explainer = _get_explainer()

    if explainer is not None:
        # Use SHAP mean absolute values over a synthetic reference batch
        try:
            import shap  # noqa: PLC0415
            ref = pd.DataFrame(np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS)
            shap_vals = explainer.shap_values(ref)
            # For a single row, shap_vals shape is (1, n_features)
            importance = np.abs(shap_vals).mean(axis=0)
            source = "shap"
        except Exception as exc:
            log.warning("SHAP global failed (%s) — using XGBoost importance", exc)
            explainer = None

    if explainer is None:
        try:
            model = get_model()
            importance = model.feature_importances_
            source = "xgboost_gain"
        except Exception as exc:
            log.warning("Could not compute feature importance: %s", exc)
            return []

    ranked = sorted(
        zip(FEATURE_COLS, importance.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return [
        {"rank": i + 1, "feature": name, "importance": round(val, 6), "source": source}
        for i, (name, val) in enumerate(ranked)
    ]


def get_local_explanation(
    feature_df: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Return SHAP contributions for a single prediction row.

    Args:
        feature_df: Single-row DataFrame with raw feature values (same format as /predict input).
        top_n:      Number of top contributors to return.

    Returns:
        Dict with keys:
          - base_value: float  (expected model output)
          - prediction: float  (model output for this row)
          - top_contributors: list of {feature, shap_value, feature_value, direction}
          - source: str  ("shap" or "xgboost_gain_fallback")
    """
    df = _prepare_features(feature_df)
    model = get_model()
    prediction = float(model.predict(df)[0])

    explainer = _get_explainer()

    if explainer is None:
        # Graceful fallback: use XGBoost feature importances as proxy contributions
        try:
            importances = model.feature_importances_
            row_values = df.iloc[0].values
            # Scale importance by feature value magnitude for a rough contribution proxy
            contribs = importances * np.abs(row_values)
            ranked = sorted(
                zip(FEATURE_COLS, contribs.tolist(), row_values.tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]
            return {
                "base_value": None,
                "prediction": prediction,
                "top_contributors": [
                    {
                        "feature": name,
                        "shap_value": round(contrib, 4),
                        "feature_value": round(fval, 4),
                        "direction": "positive" if contrib >= 0 else "negative",
                    }
                    for name, contrib, fval in ranked
                ],
                "source": "xgboost_gain_fallback",
            }
        except Exception as exc:
            log.warning("Local explanation fallback also failed: %s", exc)
            return {"error": "explainability_unavailable", "prediction": prediction}

    try:
        import shap  # noqa: PLC0415
        shap_values = explainer.shap_values(df)  # shape: (1, n_features)
        base_value = float(explainer.expected_value)
        contribs = shap_values[0]  # shape: (n_features,)
        row_values = df.iloc[0].values

        ranked = sorted(
            zip(FEATURE_COLS, contribs.tolist(), row_values.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        return {
            "base_value": round(base_value, 2),
            "prediction": round(prediction, 2),
            "top_contributors": [
                {
                    "feature": name,
                    "shap_value": round(shap_val, 4),
                    "feature_value": round(fval, 4),
                    "direction": "positive" if shap_val >= 0 else "negative",
                }
                for name, shap_val, fval in ranked
            ],
            "source": "shap",
        }
    except Exception as exc:
        log.warning("SHAP local explanation failed: %s", exc)
        return {"error": str(exc), "prediction": prediction}
