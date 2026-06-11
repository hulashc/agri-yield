"""
training/utils/model_card.py

Generates a model_card.json artefact alongside model.pkl at training time.
Captures dataset source, split policy, metrics, feature list, and training date.
Loaded at startup by serving/app.py and exposed via GET /model/info.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_CARD_PATH = os.getenv("MODEL_CARD_PATH", "model_card.json")


def write_model_card(
    rmse: float,
    n_train: int,
    n_test: int,
    dataset_source: str,
    feature_cols: list[str],
    params: dict[str, Any],
    output_path: str = MODEL_CARD_PATH,
) -> dict[str, Any]:
    """
    Write model_card.json next to model.pkl.
    Called at the end of every training run.
    """
    card: dict[str, Any] = {
        "model_name": "agri-yield-xgb",
        "algorithm": "XGBoostRegressor",
        "target": "yield_kg_per_ha",
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_source": dataset_source,
        "split_policy": "temporal — latest 20% of weeks held out, no shuffle",
        "n_train": n_train,
        "n_test": n_test,
        "metrics": {
            "rmse_kg_per_ha": round(rmse, 2),
        },
        "hyperparameters": params,
        "feature_columns": feature_cols,
        "ci_method": "fixed_fraction",  # updated to 'quantile' in Phase 2
        "ci_note": "Confidence intervals are ±15% of prediction. Phase 2 will replace with quantile regression.",
        "known_limitations": [
            "Synthetic fallback dataset lacks real spatial diversity.",
            "Confidence intervals are not statistically derived (Phase 2 fix).",
            "Drift buffer is in-memory only and resets on restart (Phase 3 fix).",
        ],
    }
    Path(output_path).write_text(json.dumps(card, indent=2))
    print(f"Model card written to {output_path}")
    return card


def load_model_card(path: str = MODEL_CARD_PATH) -> dict[str, Any]:
    """Load model card from disk. Returns empty dict if not found."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}
