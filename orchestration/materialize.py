# orchestration/materialize.py
from datetime import UTC, datetime, timedelta

import redis
from feast import FeatureStore

REDIS_HOST = "localhost"
FEAST_REPO_PATH = "features/feast_repo/feature_repo"


def materialize() -> None:
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    store.materialize(start_date=start, end_date=end)

    # Stamp timestamp so /health can verify freshness
    r = redis.Redis(host=REDIS_HOST)
    r.set("feast:last_materialization_ts", end.isoformat())
    print(f"Materialization complete: {start} → {end}")


if __name__ == "__main__":
    materialize()
