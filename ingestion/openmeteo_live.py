"""
ingestion/openmeteo_live.py

Fetch live weather features from Open-Meteo for UK field coordinates.
Used at prediction time — called by serving/app.py per request.
Redis-cached with 1h TTL where available.
In-memory cache used as fallback when Redis is absent (e.g. Render free tier).
Returns stale_features=True on any cache fallback.
"""

import requests
import json
import logging
from datetime import date, datetime
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

# --------------------------------------------------------------------------- #
# UK seasonal defaults — used when both Redis and Open-Meteo are unavailable.
# Based on typical May values across the UK.
# --------------------------------------------------------------------------- #
_UK_MAY_DEFAULTS = {
    "rainfall_today_mm": 2.5,
    "solar_radiation_today": 14.0,
    "t2m_max_today": 15.0,
    "t2m_min_today": 7.0,
    "rh2m_today": 72.0,
    "windspeed_today": 18.0,
    "et0_today": 3.2,
    "forecast_date": str(date.today()),
    "stale_features": True,
}

# --------------------------------------------------------------------------- #
# In-memory weather cache — keyed by field_id.
# Stores (features_dict, timestamp) so we can serve fresh-ish data on timeout.
# Survives across requests within the same process lifetime.
# --------------------------------------------------------------------------- #
_mem_cache: dict[str, tuple[dict, datetime]] = {}
MEM_CACHE_TTL_SECONDS = 3600  # 1 hour


def _mem_cache_get(field_id: str) -> Optional[dict]:
    """Return cached features if present and not expired, else None."""
    entry = _mem_cache.get(field_id)
    if entry is None:
        return None
    features, ts = entry
    age = (datetime.utcnow() - ts).total_seconds()
    if age > MEM_CACHE_TTL_SECONDS:
        return None
    return features


def _mem_cache_set(field_id: str, features: dict) -> None:
    _mem_cache[field_id] = (features, datetime.utcnow())


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
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def parse_today_features(raw: dict) -> dict:
    """Extract today's values (index 0) from the 7-day forecast."""
    daily = raw.get("daily", {})
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

    Priority order:
      1. Redis cache (if Redis is available)
      2. Open-Meteo live API
      3. In-memory cache (from a previous successful API call this session)
      4. UK seasonal defaults (stale_features=True)

    Never raises — prediction always runs.
    """
    cache_key = f"live_features:{field_id}:{date.today()}"
    r = None

    # 1. Try Redis cache
    try:
        r = get_redis()
        cached = r.get(cache_key)
        if cached:
            data = json.loads(cached)
            data["stale_features"] = False
            _mem_cache_set(field_id, data)  # keep mem cache warm too
            return data
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        r = None

    # 2. Try Open-Meteo live API
    try:
        raw = fetch_live_weather(lat, lon)
        features = parse_today_features(raw)
        features["stale_features"] = False

        # Write to in-memory cache
        _mem_cache_set(field_id, features)

        # Best-effort Redis write
        try:
            if r is not None:
                r.setex(cache_key, REDIS_TTL, json.dumps(features))
        except Exception:
            pass

        return features

    except Exception as e:
        logger.warning(f"Open-Meteo API failed for {field_id}: {e}")

    # 3. In-memory fallback (previous successful call this session)
    mem = _mem_cache_get(field_id)
    if mem is not None:
        stale = {**mem, "stale_features": True}
        logger.info(f"Serving in-memory cached features for {field_id}")
        return stale

    # 4. Yesterday's Redis stale cache
    yesterday_key = (
        f"live_features:{field_id}:{date.fromordinal(date.today().toordinal() - 1)}"
    )
    try:
        if r is not None:
            stale = r.get(yesterday_key)
            if stale:
                data = json.loads(stale)
                data["stale_features"] = True
                logger.warning(f"Serving yesterday's stale Redis features for {field_id}")
                return data
    except Exception:
        pass

    # 5. Last resort — UK seasonal defaults so prediction always runs
    logger.warning(
        f"No live features available for {field_id} — using UK seasonal defaults."
    )
    return {**_UK_MAY_DEFAULTS, "forecast_date": str(date.today())}
