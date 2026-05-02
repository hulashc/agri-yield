# training/data_loader.py
import pandas as pd


def load_features(
    parquet_path: str = "data/features/weekly_field_features",
) -> pd.DataFrame:
    """Option A — direct Parquet. Use during development."""
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["yield_kg_per_ha"])
    return df


def load_features_from_feast(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Option B — Feast offline store. Use for final MLflow-logged runs."""
    from feast import FeatureStore

    store = FeatureStore(repo_path="features/feast_repo/feature_repo")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
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
        ],
    ).to_df()
    return training_df
