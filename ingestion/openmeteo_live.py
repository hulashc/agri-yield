"""
ingestion/openmeteo_live.py

Fetch live weather features from Open-Meteo for UK field coordinates.
Used at prediction time — called by serving/app.py per request.
Redis-cached with 1h TTL where available.
In-memory cache used as fallback when Redis is absent (e.g. Render free tier).
Returns stale_features=True on any cache fallback.
"""

import json
import logging
import os
from datetime import UTC, date, datetime

import requests

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

# ---------------------------------------------------------------------------
# UK monthly climate defaults — used only when both Redis AND Open-Meteo fail.
# Values are approximate median conditions for each month across UK farmland.
# Source: UK Met Office historical averages.
# ---------------------------------------------------------------------------
_UK_MONTHLY_DEFAULTS: dict[int, dict] = {
    1:  {"rainfall_today_mm": 3.8, "solar_radiation_today": 2.5,  "t2m_max_today": 7.0,  "t2m_min_today": 2.0,  "rh2m_today": 82.0, "windspeed_today": 22.0, "et0_today": 0.6},
    2:  {"rainfall_today_mm": 2.9, "solar_radiation_today": 4.5,  "t2m_max_today": 7.5,  "t2m_min_today": 2.0,  "rh2m_today": 79.0, "windspeed_today": 21.0, "et0_today": 0.9},
    3:  {"rainfall_today_mm": 2.8, "solar_radiation_today": 8.0,  "t2m_max_today": 10.0, "t2m_min_today": 3.5,  "rh2m_today": 75.0, "windspeed_today": 20.0, "et0_today": 1.6},
    4:  {"rainfall_today_mm": 2.2, "solar_radiation_today": 12.0, "t2m_max_today": 13.0, "t2m_min_today": 5.0,  "rh2m_today": 71.0, "windspeed_today": 18.0, "et0_today": 2.5},
    5:  {"rainfall_today_mm": 2.5, "solar_radiation_today": 15.0, "t2m_max_today": 16.0, "t2m_min_today": 7.5,  "rh2m_today": 70.0, "windspeed_today": 17.0, "et0_today": 3.4},
    6:  {"rainfall_today_mm": 2.4, "solar_radiation_today": 18.0, "t2m_max_today": 19.0, "t2m_min_today": 10.0, "rh2m_today": 68.0, "windspeed_today": 16.0, "et0_today": 4.1},
    7:  {"rainfall_today_mm": 2.7, "solar_radiation_today": 17.5, "t2m_max_today": 21.0, "t2m_min_today": 12.0, "rh2m_today": 70.0, "windspeed_today": 15.0, "et0_today": 4.0},
    8:  {"rainfall_today_mm": 3.0, "solar_radiation_today": 15.5, "t2m_max_today": 21.0, "t2m_min_today": 12.0, "rh2m_today": 72.0, "windspeed_today": 15.0, "et0_today": 3.5},
    9:  {"rainfall_today_mm": 3.2, "solar_radiation_today": 11.0, "t2m_max_today": 18.0, "t2m_min_today": 10.0, "rh2m_today": 75.0, "windspeed_today": 16.0, "et0_today": 2.5},
    10: {"rainfall_today_mm": 3.8, "solar_radiation_today": 7.0,  "t2m_max_today": 14.0, "t2m_min_today": 7.0,  "rh2m_today": 79.0, "windspeed_today": 18.0, "et0_today": 1.4},
    11: {"rainfall_today_mm": 3.9, "solar_radiation_today": 3.5,  "t2m_max_today": 10.0, "t2m_min_today": 4.0,  "rh2m_today": 82.0, "windspeed_today": 20.0, "et0_today": 0.8},
    12: {"rainfall_today_mm": 4.0, "solar_radiation_today": 2.0,  "t2m_max_today": 7.5,  "t2m_min_today": 2.5,  "rh2m_today": 84.0, "windspeed_today": 22.0, "et0_today": 0.5},
}


def _get_seasonal_defaults() -> dict:
    """Return the correct monthly default values for today's date."""
    month = date.today().month
    base = _UK_MONTHLY_DEFAULTS[month].copy()
    base["forecast_date"] = str(date.today())
    base["stale_features"] = True
    return base


# ---------------------------------------------------------------------------
# In-memory weather cache — keyed by field_id.
# Stores (features_dict, timestamp) — survives across requests in same process.
# ---------------------------------------------------------------------------
_mem_cache: dict[str, tuple[dict, datetime]] = {}
MEM_CACHE_TTL_SECONDS = 3600  # 1 hour


def _mem_cache_get(field_id: str) -> dict | None:
    """Return cached features if present and not expired, else None."""
    entry = _mem_cache.get(field_id)
    if entry is None:
        return None
    features, ts = entry
    age = (datetime.now(UTC) - ts).total_seconds()
    if age > MEM_CACHE_TTL_SECONDS:
        return None
    return features


def _mem_cache_set(field_id: str, features: dict) -> None:
    _mem_cache[field_id] = (features, datetime.now(UTC))


# ---------------------------------------------------------------------------
# Redis — lazily initialised once per process via a connection pool to avoid
# the global-mutable race condition on concurrent first requests.
# ---------------------------------------------------------------------------
_redis_pool = None


def get_redis():
    global _redis_pool
    if _redis_pool is None:
        import redis

        _redis_pool = redis.ConnectionPool.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
            max_connections=10,
        )
    return redis.Redis(connection_pool=_redis_pool)  # type: ignore[name-defined]


def fetch_live_weather(lat: float, lon: float) -> dict:
    """Call Open-Meteo for today + 7 day forecast."""
    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/London",
        "forecast_days": 7,
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=8)
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
      4. Yesterday's Redis stale cache
      5. UK monthly seasonal defaults (stale_features=True)

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
            _mem_cache_set(field_id, data)
            return data
    except Exception as exc:
        logger.warning("Redis unavailable: %s", exc)
        r = None

    # 2. Try Open-Meteo live API
    try:
        raw = fetch_live_weather(lat, lon)
        features = parse_today_features(raw)
        features["stale_features"] = False
        _mem_cache_set(field_id, features)
        try:
            if r is not None:
                r.setex(cache_key, REDIS_TTL, json.dumps(features))
        except Exception:
            pass
        return features
    except Exception as exc:
        logger.warning("Open-Meteo API failed for %s: %s", field_id, exc)

    # 3. In-memory fallback
    mem = _mem_cache_get(field_id)
    if mem is not None:
        logger.info("Serving in-memory cached features for %s", field_id)
        return {**mem, "stale_features": True}

    # 4. Yesterday's Redis stale cache
    yesterday = date.fromordinal(date.today().toordinal() - 1)
    yesterday_key = f"live_features:{field_id}:{yesterday}"
    try:
        if r is not None:
            stale = r.get(yesterday_key)
            if stale:
                data = json.loads(stale)
                data["stale_features"] = True
                logger.warning("Serving yesterday's stale Redis features for %s", field_id)
                return data
    except Exception:
        pass

    # 5. Last resort — correct monthly defaults so prediction always runs
    logger.warning(
        "No live features available for %s — using monthly seasonal defaults.", field_id
    )
    return _get_seasonal_defaults()
