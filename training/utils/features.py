import pandas as pd

NON_FEATURE_COLS = ["yield_kg_per_ha", "week_start", "field_id"]

# Static feature column list used by serving/model.py at prediction time.
# Must match the columns produced by the training pipeline.
FEATURE_COLS = [
    "lat",
    "lon",
    "area_ha",
    "temperature_2m_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture",
    "ndvi",
    "crop_type_encoded",
    "soil_type_encoded",
    "region_encoded",
    "week_of_year",
    "year",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLS]
