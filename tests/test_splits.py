"""
Smoke tests for temporal train/test split.
"""

import pandas as pd
import pytest

from training.utils.splits import temporal_train_test_split


def _make_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="W")
    return pd.DataFrame(
        {
            "week_start": dates,
            "field_id": "F001",
            "yield_kg_per_ha": range(n),
            "air_temp_mean": range(n),
        }
    )


def test_split_no_data_leakage():
    df = _make_df(100)
    train, test = temporal_train_test_split(df)
    assert train["week_start"].max() < test["week_start"].min()


def test_split_covers_all_rows():
    df = _make_df(100)
    train, test = temporal_train_test_split(df)
    assert len(train) + len(test) == len(df)


def test_split_train_larger_than_test():
    df = _make_df(100)
    train, test = temporal_train_test_split(df)
    assert len(train) > len(test)
