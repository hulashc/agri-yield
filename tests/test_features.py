"""
Smoke tests for feature utilities — verify the shared feature column
contract between training and serving is intact.
"""

import pandas as pd
import pytest

from training.utils.features import FEATURE_COLS, NON_FEATURE_COLS, get_feature_cols


def test_feature_cols_non_empty():
    assert len(FEATURE_COLS) > 0


def test_no_overlap_between_feature_and_non_feature():
    overlap = set(FEATURE_COLS) & set(NON_FEATURE_COLS)
    assert overlap == set(), f"Overlap found: {overlap}"


def test_get_feature_cols_filters_correctly():
    df = pd.DataFrame(columns=FEATURE_COLS + ["yield_kg_per_ha", "field_id", "week_start"])
    result = get_feature_cols(df)
    assert set(result) == set(FEATURE_COLS)
    assert "yield_kg_per_ha" not in result
    assert "field_id" not in result


def test_get_feature_cols_handles_missing_columns():
    df = pd.DataFrame(columns=["soil_temp_mean", "air_temp_mean", "yield_kg_per_ha"])
    result = get_feature_cols(df)
    assert result == ["soil_temp_mean", "air_temp_mean"]


def test_feature_cols_order_is_stable():
    """Serving depends on a fixed column order — this must never silently change."""
    assert FEATURE_COLS[0] == "soil_temp_mean"
    assert FEATURE_COLS[-1] == "ndvi_proxied"
