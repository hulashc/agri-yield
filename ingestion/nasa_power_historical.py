"""
ingestion/nasa_power_historical.py

Pull daily agronomic data from NASA POWER for UK field coordinates.
Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point
Community: AG (agronomy)
Output: data/raw/nasa_power/{field_id}/{year}.parquet
"""

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMETERS = ",".join(
    [
        "PRECTOTCORR",
        "ALLSKY_SFC_SW_DWN",
        "T2M_MAX",
        "T2M_MIN",
        "RH2M",
        "WS2M",
        "EVPTRNS",
    ]
)

UK_FIELDS: list[dict[str, str | float]] = [
    {
        "field_id": "F001",
        "lat": 52.9135,
        "lon": -0.1736,
        "crop_type": "winter_wheat",
        "region": "Lincolnshire",
    },
    {
        "field_id": "F002",
        "lat": 52.8389,
        "lon": -0.0325,
        "crop_type": "oilseed_rape",
        "region": "Lincolnshire",
    },
    {
        "field_id": "F003",
        "lat": 52.7862,
        "lon": -0.1532,
        "crop_type": "sugar_beet",
        "region": "Lincolnshire",
    },
    {
        "field_id": "F004",
        "lat": 52.6793,
        "lon": 0.1631,
        "crop_type": "winter_barley",
        "region": "Cambridgeshire",
    },
    {
        "field_id": "F005",
        "lat": 52.9721,
        "lon": -0.0812,
        "crop_type": "winter_wheat",
        "region": "Lincolnshire",
    },
    {
        "field_id": "F006",
        "lat": 52.7156,
        "lon": 0.3928,
        "crop_type": "oilseed_rape",
        "region": "Norfolk",
    },
    {
        "field_id": "F007",
        "lat": 52.6009,
        "lon": 0.3811,
        "crop_type": "sugar_beet",
        "region": "Norfolk",
    },
    {
        "field_id": "F008",
        "lat": 52.3983,
        "lon": 0.2617,
        "crop_type": "winter_wheat",
        "region": "Cambridgeshire",
    },
]


def fetch_nasa_power(
    lat: float,
    lon: float,
    start: str = "19810101",
    end: str | None = None,
) -> dict:  # type: ignore[type-arg]
    if end is None:
        end = (date.today() - timedelta(days=7)).strftime("%Y%m%d")

    params: dict[str, str | float] = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    resp = requests.get(NASA_POWER_URL, params=params, timeout=120)  # type: ignore[arg-type]
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def parse_to_dataframe(raw: dict, field_id: str) -> pd.DataFrame:  # type: ignore[type-arg]
    props = raw["properties"]["parameter"]
    df = pd.DataFrame(props)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df.reset_index(inplace=True)
    df["field_id"] = field_id
    df.replace(-999.0, pd.NA, inplace=True)
    return df


def validate_dataframe(df: pd.DataFrame, field_id: str) -> bool:  # type: ignore[type-arg]
    required_cols = ["PRECTOTCORR", "T2M_MAX", "T2M_MIN", "RH2M"]
    null_pct = df[required_cols].isna().mean()
    if (null_pct > 0.05).any():
        bad = null_pct[null_pct > 0.05].to_dict()
        logger.error(f"[{field_id}] Null rate exceeds 5%%: {bad}")
        return False
    t_range = df["T2M_MAX"].dropna()
    if ((t_range < -20) | (t_range > 50)).any():
        logger.error(f"[{field_id}] T2M_MAX out of range")
        return False
    rain = df["PRECTOTCORR"].dropna()
    if (rain < 0).any():
        logger.error(f"[{field_id}] Negative rainfall")
        return False
    return True


def save_by_year(
    df: pd.DataFrame,  # type: ignore[type-arg]
    field_id: str,
    output_root: str = "data/raw/nasa_power",
) -> None:
    for year, group in df.groupby(df["date"].dt.year):
        out_dir = Path(output_root) / field_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{year}.parquet"
        group.to_parquet(path, index=False)
        logger.info(f"Saved {len(group)} rows → {path}")


def run_historical_pull(start: str = "19810101") -> None:
    for field in UK_FIELDS:
        fid = str(field["field_id"])
        logger.info(f"Fetching NASA POWER for {fid}...")
        try:
            raw = fetch_nasa_power(
                float(field["lat"]), float(field["lon"]), start=start
            )
            df = parse_to_dataframe(raw, fid)
            if not validate_dataframe(df, fid):
                raise ValueError(f"Data quality check failed for {fid}")
            save_by_year(df, fid)
            logger.info(f"✅ {fid} complete — {len(df)} rows")
        except Exception as e:
            logger.error(f"❌ {fid} failed: {e}")
        time.sleep(2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_historical_pull()
