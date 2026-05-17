"""
Process the CYCleSS real UK crop yield dataset into training features.

Outputs data/features/weekly_field_features.parquet with columns
compatible with FEATURE_COLS (training/utils/features.py).
Since CYCleSS yields are annual (not weekly), we compute:
  - Climate: mean/sum across the growing season (Apr-Sep)
  - Satellite: mean/range across available dates
  - Soil: per-grid_ID static properties mapped to FEATURE_COLS space
  - week_of_year: set to 26 (mid-season) for annual records
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.utils.features import FEATURE_COLS

RAW_DIR = None
for candidate in [Path("data/raw/cycless/CYCleSS_dataset"), Path("cycless_data/CYCleSS_dataset")]:
    if candidate.exists():
        RAW_DIR = candidate
        break
if RAW_DIR is None:
    zip_candidate = Path("data/raw/cycless.zip")
    if zip_candidate.exists():
        print("Extracting CYCleSS zip...")
        extract_to = Path("data/raw/cycless")
        with zipfile.ZipFile(zip_candidate, "r") as zf:
            zf.extractall(extract_to)
        RAW_DIR = extract_to / "CYCleSS_dataset"
    else:
        raise FileNotFoundError(
            "CYCleSS data not found. Run `python scripts/download_cycless.py` first, "
            "or place cycless.zip at data/raw/cycless.zip"
        )

OUTPUT_PATH = "data/features/weekly_field_features.parquet"

CROP_MAP = {
    "Wheat": 0,
    "W-Barley": 2,
    "S-Barley": 3,
    "OSR": 4,
    "Beans": 5,
}

# SOIL_TEXT_TO_ENCODED = {}  # removed — 'text' is a numeric code, used directly


def osgb_to_latlon(east: float, north: float) -> tuple[float, float]:
    """Simplified OSTN15 approximation for England.
    Produces usable lat/lon for model features; exact precision isn't critical.
    """
    e = east - 400000
    n = north - 100000
    lat = 52.5 + n / 111320 / 1000
    lon = -1.5 + e / (111320 * np.cos(np.deg2rad(lat))) / 1000
    return round(float(lat), 5), round(float(lon), 5)


def load_yield(year: int) -> pd.DataFrame:
    path = RAW_DIR / "data" / "crop_yield_type_and_satellite_data" / f"Ratio_{year}_MeanYieldperField.csv"
    df = pd.read_csv(path)
    df["Year"] = int(year)
    return df


def extract_satellite_features(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Compute yearly satellite backscatter features from wide-format date columns."""
    sat_cols = [c for c in df.columns if c.startswith(f"X{year}") or (c.startswith(f"X{year}") and c.endswith(".1"))]
    ratio_cols = [c for c in sat_cols if not c.endswith(".1")]
    ratio2_cols = [c for c in sat_cols if c.endswith(".1")]

    result = pd.DataFrame({"ID": df["ID"].values})

    for col_list, name in [(ratio_cols, "ratio"), (ratio2_cols, "ratio2")]:
        if col_list:
            vals = df[col_list].apply(pd.to_numeric, errors="coerce")
            result[f"sar_{name}_mean"] = vals.mean(axis=1).values
            result[f"sar_{name}_range"] = (vals.max(axis=1) - vals.min(axis=1)).values
            result[f"sar_{name}_count"] = vals.notna().sum(axis=1).values

    return result


def load_climate(year: int) -> pd.DataFrame:
    """Load daily climate data for a given year and aggregate to growing season (Apr-Sep)."""
    climate_dir = RAW_DIR / "data" / "climate_data" / str(year)
    var_map = {"tas": "temperature_2m_mean", "precip": "precipitation_sum",
               "rsds": "shortwave_radiation_sum", "pet": "et0_fao_evapotranspiration",
               "sfcWind": "wind_speed", "huss": "specific_humidity"}

    merged = None
    for fname in os.listdir(climate_dir):
        if not fname.endswith(".csv"):
            continue
        var_key = fname.split("_")[0]
        feat_name = var_map.get(var_key)
        if feat_name is None:
            continue
        df = pd.read_csv(climate_dir / fname)
        # Drop unnamed index column if present
        unnamed = [c for c in df.columns if "Unnamed" in c]
        if unnamed:
            df = df.drop(columns=unnamed)
        # Find date columns (skip grid_ID column)
        date_cols = [c for c in df.columns if c.startswith("X")]
        # Filter to Apr-Sep (months 4-9) growing season
        season_cols = [c for c in date_cols if int(c.split(".")[1]) in range(4, 10)]
        df[feat_name] = df[season_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        # Average per grid_ID (climate file has multiple rows per cell)
        sub = df.groupby("grid_ID", as_index=False)[feat_name].mean()
        merged = sub if merged is None else merged.merge(sub, on="grid_ID")

    return merged


def load_soil() -> pd.DataFrame:
    soil = pd.read_csv(RAW_DIR / "data" / "soil_data" / "LandUseandSoil_2015_2016.csv")
    unnamed = [c for c in soil.columns if "Unnamed" in c]
    if unnamed:
        soil = soil.drop(columns=unnamed)
    # Average per grid_ID (some grid cells may have multiple soil records)
    return soil.groupby("grid_ID", as_index=False)[
        ["grid_ID", "clay", "sand", "silt", "awc", "bd", "fc", "ks", "text"]
    ].first()


def prepare():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_rows = []
    for year in [2015, 2016, 2017]:
        print(f"Processing {year}...")
        yield_df = load_yield(year)
        sat_feat = extract_satellite_features(yield_df, year)
        climate_feat = load_climate(year)
        soil_df = load_soil()

        for _, row in yield_df.iterrows():
            grid_id = row["grid_ID"]
            crop_raw = row["Crop"]
            crop_enc = CROP_MAP.get(crop_raw, 3)
            soil_row = soil_df[soil_df["grid_ID"] == grid_id]
            climate_row = climate_feat[climate_feat["grid_ID"] == grid_id] if climate_feat is not None else pd.DataFrame()

            lat, lon = osgb_to_latlon(float(row["east"]), float(row["north"]))

            sat_row = sat_feat[sat_feat["ID"] == row["ID"]]

            feat = {
                "lat": lat,
                "lon": lon,
                "area_ha": 50.0,
                "crop_type_encoded": crop_enc,
                "soil_type_encoded": int(float(soil_row["text"].values[0]))
                if not soil_row.empty and pd.notna(soil_row["text"].values[0]) else 0,
                "region_encoded": 0,
                "temperature_2m_mean": float(climate_row["temperature_2m_mean"].values[0])
                if not climate_row.empty else 15.0,
                "precipitation_sum": float(climate_row["precipitation_sum"].values[0])
                if not climate_row.empty else 2.0,
                "shortwave_radiation_sum": float(climate_row["shortwave_radiation_sum"].values[0])
                if not climate_row.empty else 15.0,
                "et0_fao_evapotranspiration": float(climate_row["et0_fao_evapotranspiration"].values[0])
                if not climate_row.empty else 3.0,
                "soil_moisture": float(
                    soil_row["awc"].values[0]
                    if not soil_row.empty and pd.notna(soil_row["awc"].values[0])
                    else 0.35
                ),
                "ndvi": sat_row["sar_ratio_mean"].values[0] / 1000
                if not sat_row.empty and pd.notna(sat_row["sar_ratio_mean"].values[0]) else 0.5,
                "week_of_year": 26,
                "year": int(year),
                "yield_kg_per_ha": float(row["Yield"]) * 1000,
                "field_id": str(row["ID"]),
            }
            all_rows.append(feat)

    df = pd.DataFrame(all_rows)
    df["year"] = df["year"].astype(int)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    df["week_start"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week_of_year"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"Yield range: {df['yield_kg_per_ha'].min():.0f} - {df['yield_kg_per_ha'].max():.0f} kg/ha")
    print(f"Years: {sorted(df['year'].unique())}")
    return df


if __name__ == "__main__":
    prepare()
