"""Temporal train/test split utilities for time-series feature data."""

from __future__ import annotations

import pandas as pd


def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str = "week_start",
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into train/test by time, preserving temporal order.

    Args:
        df: Feature dataframe with a date column.
        date_col: Name of the datetime column to sort by.
        test_ratio: Fraction of most-recent rows to use as test set.

    Returns:
        (train_df, test_df) tuple.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df
