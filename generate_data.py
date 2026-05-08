"""
Generate synthetic UK agricultural training data.
Outputs data/features/weekly_field_features.parquet with columns
that exactly match FEATURE_COLS in training/utils/features.py.
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_FIELDS = 100
N_WEEKS = 52
OUTPUT_PATH = "data/features/weekly_field_features.parquet"

CROP_TYPES = ["winter_wheat", "spring_wheat", "winter_barley", "spring_barley", "oilseed_rape", "sugar_beet"]
SOIL_TYPES = ["clay", "sandy_loam", "silt", "loam", "peat"]
REGIONS = ["East Anglia", "Yorkshire", "Midlands", "South West", "North West", "Scotland", "Wales", "Northumberland"]

# Base yields per crop type (kg/ha)
CROP_BASE_YIELD = {
    "winter_wheat": 8000, "spring_wheat": 6500,
    "winter_barley": 7000, "spring_barley": 5800,
    "oilseed_rape": 3500, "sugar_beet": 70000,
}

def generate():
    rng = np.random.default_rng(RANDOM_SEED)

    crop_enc   = {c: i for i, c in enumerate(CROP_TYPES)}
    soil_enc   = {s: i for i, s in enumerate(SOIL_TYPES)}
    region_enc = {r: i for i, r in enumerate(REGIONS)}

    # Field-level static attributes
    field_ids   = [f"F{i:03d}" for i in range(N_FIELDS)]
    field_lats  = rng.uniform(50.5, 58.5, N_FIELDS)   # UK lat range
    field_lons  = rng.uniform(-4.5, 1.8, N_FIELDS)    # UK lon range
    field_areas = rng.uniform(10, 150, N_FIELDS)       # ha
    field_crops = rng.choice(CROP_TYPES, N_FIELDS)
    field_soils = rng.choice(SOIL_TYPES, N_FIELDS)
    field_regs  = rng.choice(REGIONS, N_FIELDS)

    rows = []
    for i, fid in enumerate(field_ids):
        crop   = field_crops[i]
        base_y = CROP_BASE_YIELD[crop]
        for week in range(N_WEEKS):
            week_of_year = week + 1
            year = 2024

            # Seasonal weather patterns
            season_factor = np.sin(np.pi * week / 52)  # peaks mid-year
            temp   = 8 + 12 * season_factor + rng.normal(0, 2)
            precip = max(0, 3 + 2 * (1 - season_factor) + rng.normal(0, 1.5))
            solar  = max(0, 8 + 14 * season_factor + rng.normal(0, 2))
            et0    = max(0, 1.5 + 3 * season_factor + rng.normal(0, 0.5))
            sm     = max(0, min(1, 0.35 - 0.1 * season_factor + rng.normal(0, 0.05)))
            ndvi   = max(0, min(1, 0.3 + 0.5 * season_factor + rng.normal(0, 0.05)))

            # Yield driven by weather + crop type
            yield_val = (
                base_y
                * (0.8 + 0.4 * ndvi)
                * (0.9 + 0.1 * sm)
                * (1 + 0.005 * temp)
                + rng.normal(0, base_y * 0.05)
            )

            rows.append({
                "field_id":               fid,
                "week_of_year":           week_of_year,
                "year":                   year,
                "lat":                    round(float(field_lats[i]), 4),
                "lon":                    round(float(field_lons[i]), 4),
                "area_ha":                round(float(field_areas[i]), 1),
                "crop_type_encoded":      crop_enc[crop],
                "soil_type_encoded":      soil_enc[field_soils[i]],
                "region_encoded":         region_enc[field_regs[i]],
                "temperature_2m_mean":    round(temp, 2),
                "precipitation_sum":      round(precip, 2),
                "shortwave_radiation_sum": round(solar, 2),
                "et0_fao_evapotranspiration": round(et0, 3),
                "soil_moisture":          round(sm, 4),
                "ndvi":                   round(ndvi, 4),
                "yield_kg_per_ha":        round(max(0, yield_val), 1),
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows → {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    generate()
