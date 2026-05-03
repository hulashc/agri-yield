"""
ingestion/openmeteo_live.py

Fetch live weather features from Open-Meteo for UK field coordinates.
Used at prediction time — called by serving/app.py per request.
Redis-cached with 1h TTL. Returns stale_features=True on cache fallback.
"""

import requests
import json
import logging
from datetime import date
from typing import Optional
import redis

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARIABLES = [
    "precipitation_sum",
    "shortwave_radiation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "windspeed_10m_max",
    "et0_fao_evapotranspiration",
]

REDIS_TTL = 3600  # 1 hour

_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        import os

        _redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True
        )
    return _redis_client


def fetch_live_weather(lat: float, lon: float) -> dict:
    """Call Open-Meteo for today + 7 day forecast."""
    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/London",
        "forecast_days": 7,
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def parse_today_features(raw: dict) -> dict:
    """Extract today's values (index 0) from the 7-day forecast."""
    daily = raw.get("daily", {})
    # Map Open-Meteo keys → internal feature names
    mapping = {
        "precipitation_sum": "rainfall_today_mm",
        "shortwave_radiation_sum": "solar_radiation_today",
        "temperature_2m_max": "t2m_max_today",
        "temperature_2m_min": "t2m_min_today",
        "relative_humidity_2m_max": "rh2m_today",
        "windspeed_10m_max": "windspeed_today",
        "et0_fao_evapotranspiration": "et0_today",
    }
    features = {}
    for om_key, internal_key in mapping.items():
        values = daily.get(om_key, [None])
        features[internal_key] = values[0] if values else None
    features["forecast_date"] = str(date.today())
    return features


def get_live_features(field_id: str, lat: float, lon: float) -> dict:
    """
    Return live weather features for a field.
    Tries Redis cache first, then Open-Meteo API, then raises with stale flag.
    """
    cache_key = f"live_features:{field_id}:{date.today()}"
    r = get_redis()

    # 1. Try cache
    try:
        cached = r.get(cache_key)
        if cached:
            data = json.loads(cached)
            data["stale_features"] = False
            return data
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")

    # 2. Try API
    try:
        raw = fetch_live_weather(lat, lon)
        features = parse_today_features(raw)
        features["stale_features"] = False

        # Cache result
        try:
            r.setex(cache_key, REDIS_TTL, json.dumps(features))
        except Exception:
            pass

        return features

    except Exception as e:
        logger.error(f"Open-Meteo API failed for {field_id}: {e}")

    # 3. Fallback — try yesterday's cache
    yesterday_key = (
        f"live_features:{field_id}:{date.fromordinal(date.today().toordinal() - 1)}"
    )
    try:
        stale = r.get(yesterday_key)
        if stale:
            data = json.loads(stale)
            data["stale_features"] = True
            logger.warning(f"Serving stale features for {field_id}")
            return data
    except Exception:
        pass

    raise RuntimeError(f"Cannot fetch live features for {field_id} — no cache, no API")
