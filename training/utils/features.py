import pandas as pd

# These columns are never used as model features.
# This list is the single source of truth shared by training and serving.
NON_FEATURE_COLS = ["yield_kg_per_ha", "week_start", "field_id", "crop_type"]

# Canonical ordered feature list — serving/model.py imports this directly
# so training and inference always use identical columns in the same order.
FEATURE_COLS = [
    "soil_temp_mean",
    "soil_temp_std",
    "moisture_mean",
    "moisture_std",
    "ph_mean",
    "nitrogen_mean",
    "phosphorus_mean",
    "potassium_mean",
    "air_temp_mean",
    "precip_total",
    "humidity_mean",
    "wind_speed_mean",
    "latest_ndvi",
    "cloud_cover_pct",
    "ndvi_interpolated",
    "ndvi_proxied",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return FEATURE_COLS that are actually present in df."""
    return [col for col in FEATURE_COLS if col in df.columns]
