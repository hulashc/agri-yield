from __future__ import annotations

from pathlib import Path
import pandas as pd

METRICS_PATH = Path("data/monitoring/weekly_metrics.parquet")


def ensure_parent() -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


def write_metrics(rows: list[dict]) -> None:
    ensure_parent()
    new_df = pd.DataFrame(rows)
    if METRICS_PATH.exists():
        old_df = pd.read_parquet(METRICS_PATH)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_parquet(METRICS_PATH, index=False)


def read_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(METRICS_PATH)
