import os

import numpy as np
import pandas as pd

np.random.seed(42)
fields = pd.read_csv("data/seed/uk_fields.csv")
records = []

crop_yield_map = {
    "winter_wheat": (8000, 1200),
    "oilseed_rape": (3800, 600),
    "sugar_beet": (70000, 8000),
    "winter_barley": (6500, 900),
    "spring_barley": (5500, 800),
    "spring_wheat": (7000, 1000),
}

weeks = pd.date_range("2020-01-06", "2024-12-30", freq="W-MON")

for _, field in fields.iterrows():
    mean_y, std_y = crop_yield_map.get(field["crop_type"], (6000, 1000))
    for week in weeks:
        doy = week.day_of_year
        records.append(
            {
                "field_id": field["field_id"],
                "week_start": week,
                "crop_type": field["crop_type"],
                "region": field["region"],
                "area_ha": field["area_ha"],
                "soil_type": field["soil_type"],
                "lat": field["lat"],
                "lon": field["lon"],
                "soil_temp_mean": 12
                + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
                + np.random.normal(0, 1.5),
                "soil_temp_std": abs(np.random.normal(2, 0.5)),
                "moisture_mean": 55
                + 15 * np.sin(2 * np.pi * (doy + 90) / 365)
                + np.random.normal(0, 5),
                "moisture_std": abs(np.random.normal(8, 2)),
                "ph_mean": np.random.normal(6.5, 0.3),
                "nitrogen_mean": np.random.normal(85, 15),
                "phosphorus_mean": np.random.normal(28, 6),
                "potassium_mean": np.random.normal(125, 20),
                "air_temp_mean": 10
                + 8 * np.sin(2 * np.pi * (doy - 80) / 365)
                + np.random.normal(0, 2),
                "precip_total": abs(np.random.normal(15, 8)),
                "humidity_mean": np.random.normal(75, 10),
                "wind_speed_mean": abs(np.random.normal(5, 2)),
                "latest_ndvi": 0.3
                + 0.4 * np.sin(2 * np.pi * (doy - 60) / 365)
                + np.random.normal(0, 0.05),
                "cloud_cover_pct": abs(np.random.normal(60, 20)),
                "ndvi_interpolated": int(np.random.choice([0, 1], p=[0.8, 0.2])),
                "ndvi_proxied": int(np.random.choice([0, 1], p=[0.9, 0.1])),
                "yield_kg_per_ha": max(0, np.random.normal(mean_y, std_y)),
            }
        )

df = pd.DataFrame(records)
os.makedirs("data/features", exist_ok=True)
df.to_parquet("data/features/weekly_field_features", index=False)
print(f"Generated {len(df)} rows across {df['field_id'].nunique()} fields")
print(f"Date range: {df['week_start'].min()} to {df['week_start'].max()}")
