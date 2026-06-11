from __future__ import annotations

import os
from datetime import UTC, datetime

import redis

from serving.model import get_model

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MAX_MATERIALIZATION_AGE_HOURS = int(os.getenv("MAX_MAT_AGE_HOURS", "25"))


def check_model() -> bool:
    try:
        get_model()
        return True
    except Exception:
        return False


def check_redis() -> bool:
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=1)
        return bool(r.ping())
    except Exception:
        return False


def check_materialization_age() -> bool:
    """Check a Redis key written by the nightly materialization job."""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=1)
        ts_bytes = r.get("feast:last_materialization_ts")
        if not ts_bytes:
            return False
        # r.get() returns bytes in the sync redis client.
        # Narrow the type explicitly before decode() so mypy is satisfied.
        if isinstance(ts_bytes, bytes):
            raw: str = ts_bytes.decode("utf-8")
        else:
            raw = str(ts_bytes)
        last_mat = datetime.fromisoformat(raw)
        age_hours = (datetime.now(UTC) - last_mat).total_seconds() / 3600
        return age_hours < MAX_MATERIALIZATION_AGE_HOURS
    except Exception:
        return False


def run_health_checks() -> dict[str, bool]:
    checks: dict[str, bool] = {
        "model_loaded": check_model(),
        "redis_connected": check_redis(),
        "materialization_fresh": check_materialization_age(),
    }
    checks["healthy"] = all(checks.values())
    return checks
