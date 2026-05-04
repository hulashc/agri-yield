"""
features/build_weekly_features.py

Pandas-based feature engineering script.
Reads NASA POWER daily parquet files from data/raw/nasa_power/
Aggregates to weekly features and writes:
    data/features/weekly_field_features.parquet

This is the training-ready dataset consumed by training/train.py.
Run after ingestion/nasa_power_historical.py has populated data/raw/nasa_power/.

Usage:
    python -m features.build_weekly_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/nasa_power")
OUT_PATH = Path("data/features/weekly_field_features.parquet")

# NASA POWER column → feature name mapping
COL_MAP = {
    "T2M_MAX": "air_temp_max",
    "T2M_MIN": "air_temp_min",
    "PRECTOTCORR": "precip_total",
    "RH2M": "humidity_mean",
    "WS2M": "wind_speed_mean",
}

# Field metadata (mirrors nasa_power_historical.py)
FIELD_META = {
    "F001": {"crop_type": "winter_wheat", "region": "Lincolnshire", "lat": 52.9135, "lon": -0.1736},
    "F002": {"crop_type": "oilseed_rape", "region": "Lincolnshire", "lat": 52.8389, "lon": -0.0325},
    "F003": {"crop_type": "sugar_beet", "region": "Lincolnshire", "lat": 52.7862, "lon": -0.1532},
    "F004": {"crop_type": "winter_barley", "region": "Cambridgeshire", "lat": 52.6793, "lon": 0.1631},
    "F005": {"crop_type": "winter_wheat", "region": "Lincolnshire", "lat": 52.9721, "lon": -0.0812},
    "F006": {"crop_type": "oilseed_rape", "region": "Norfolk", "lat": 52.7156, "lon": 0.3928},
    "F007": {"crop_type": "sugar_beet", "region": "Norfolk", "lat": 52.6009, "lon": 0.3811},
    "F008": {"crop_type": "winter_wheat", "region": "Cambridgeshire", "lat": 52.3983, "lon": 0.2617},
}

# Synthetic yield model: base yield per crop + weather sensitivity
# Real labels would come from DEFRA / farm records — this gives plausible targets for training
BASE_YIELD = {
    "winter_wheat": 8000,
    "oilseed_rape": 3500,
    "sugar_beet": 60000,
    "winter_barley": 6500,
}


def load_field_data(field_id: str) -> pd.DataFrame | None:
    field_dir = RAW_DIR / field_id
    if not field_dir.exists():
        log.warning("No data directory for %s — skipping", field_id)
        return None

    parts = []
    for parquet_file in sorted(field_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        parts.append(df)

    if not parts:
        log.warning("No parquet files found for %s — skipping", field_id)
        return None

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    log.info("Loaded %d daily rows for %s", len(df), field_id)
    return df


def aggregate_to_weekly(df: pd.DataFrame, field_id: str) -> pd.DataFrame:
    df = df.copy()
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")

    agg = df.groupby("week_start").agg(
        air_temp_mean=("T2M_MAX", "mean"),
        air_temp_std=("T2M_MAX", "std"),
        precip_total=("PRECTOTCORR", "sum"),
        humidity_mean=("RH2M", "mean"),
        wind_speed_mean=("WS2M", "mean"),
    ).reset_index()

    # Soil proxies — derived from weather (no sensor data yet)
    agg["soil_temp_mean"] = agg["air_temp_mean"] * 0.85
    agg["soil_temp_std"] = agg["air_temp_std"].fillna(0) * 0.85
    agg["moisture_mean"] = (agg["precip_total"] / 7).clip(0, 100)  # mm/day proxy
    agg["moisture_std"] = agg["moisture_mean"] * 0.1
    agg["ph_mean"] = 6.5  # typical UK arable soil
    agg["nitrogen_mean"] = 180.0  # kg/ha typical application
    agg["phosphorus_mean"] = 30.0
    agg["potassium_mean"] = 200.0

    # NDVI placeholders — will be replaced by real Kafka stream when ndvi_puller runs
    agg["latest_ndvi"] = 0.6
    agg["cloud_cover_pct"] = 40.0
    agg["ndvi_interpolated"] = 0
    agg["ndvi_proxied"] = 1

    agg["field_id"] = field_id

    # Attach field metadata
    meta = FIELD_META[field_id]
    agg["crop_type"] = meta["crop_type"]
    agg["region"] = meta["region"]
    agg["lat"] = meta["lat"]
    agg["lon"] = meta["lon"]

    # Synthetic yield label based on growing season weather
    # UK harvest: wheat/barley/rape = Aug, sugar beet = Oct-Nov
    base = BASE_YIELD[meta["crop_type"]]
    rng = np.random.default_rng(seed=abs(hash(field_id)) % (2**32))
    noise = rng.normal(0, base * 0.08, len(agg))
    temp_effect = (agg["air_temp_mean"] - 10).clip(-5, 15) * (base * 0.005)
    rain_effect = (agg["precip_total"] - 15).clip(-10, 20) * (base * 0.002)
    agg["yield_kg_per_ha"] = (base + temp_effect + rain_effect + noise).clip(base * 0.4, base * 1.6)

    return agg


def build_features() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for field_id in FIELD_META:
        daily = load_field_data(field_id)
        if daily is None:
            continue
        weekly = aggregate_to_weekly(daily, field_id)
        all_frames.append(weekly)
        log.info("Built %d weekly rows for %s", len(weekly), field_id)

    if not all_frames:
        raise RuntimeError(
            "No data found in data/raw/nasa_power/. "
            "Run ingestion/nasa_power_historical.py first."
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.dropna(subset=["yield_kg_per_ha", "air_temp_mean"])
    combined.to_parquet(OUT_PATH, index=False)
    log.info(
        "Written %d total weekly rows across %d fields → %s",
        len(combined),
        len(all_frames),
        OUT_PATH,
    )


if __name__ == "__main__":
    build_features()
