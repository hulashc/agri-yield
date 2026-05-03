from __future__ import annotations

import pandas as pd
from feast import FeatureStore

FEAST_REPO_PATH = "features/feast_repo/feature_repo"

FEATURES = [
    "soil_features:soil_temp_mean",
    "soil_features:soil_temp_std",
    "soil_features:moisture_mean",
    "soil_features:moisture_std",
    "soil_features:ph_mean",
    "soil_features:nitrogen_mean",
    "soil_features:phosphorus_mean",
    "soil_features:potassium_mean",
    "weather_features:air_temp_mean",
    "weather_features:precip_total",
    "weather_features:humidity_mean",
    "weather_features:wind_speed_mean",
    "vegetation_features:latest_ndvi",
    "vegetation_features:cloud_cover_pct",
    "vegetation_features:ndvi_interpolated",
    "vegetation_features:ndvi_proxied",
]

_store: FeatureStore | None = None


def get_store() -> FeatureStore:
    global _store
    if _store is None:
        _store = FeatureStore(repo_path=FEAST_REPO_PATH)
    return _store


def fetch_online_features(field_ids: list[str]) -> pd.DataFrame:
    """Fetch latest online features for a list of field_ids from Redis."""
    store = get_store()
    entity_rows = [{"field_id": fid} for fid in field_ids]
    feature_vector = store.get_online_features(
        features=FEATURES,
        entity_rows=entity_rows,
    ).to_df()
    return feature_vector
