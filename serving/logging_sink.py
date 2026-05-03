from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", "data/prediction_logs/predictions.jsonl")
)


def log_prediction(
    field_id: str,
    predicted: float,
    lower: float,
    upper: float,
    model_version: str,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "field_id": field_id,
        "predicted_yield_kg_per_ha": predicted,
        "lower_bound": lower,
        "upper_bound": upper,
        "model_version": model_version,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
