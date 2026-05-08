import pandas as pd

NON_FEATURE_COLS = ["yield_kg_per_ha", "week_start", "field_id"]

# Single source of truth for feature columns.
# Used by BOTH train_and_export.py (training) and serving/model.py (prediction).
# Must match the columns produced by generate_data.py.
FEATURE_COLS = [
    "lat",
    "lon",
    "area_ha",
    "crop_type_encoded",
    "soil_type_encoded",
    "region_encoded",
    "temperature_2m_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "soil_moisture",
    "ndvi",
    "week_of_year",
    "year",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLS]
