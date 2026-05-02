# orchestration/ingestion_flow.py
"""
Prefect flow that validates each ingestion batch.
Aborts and alerts if Great Expectations fails.
"""

from prefect import flow, task
import pandas as pd
from contracts.sensor_expectations import validate_batch


@task
def read_kafka_batch(topic: str) -> pd.DataFrame:
    """Read the latest batch from a Kafka topic into a DataFrame."""
    # In production: use a Kafka consumer to read N messages
    # For now: placeholder that returns a sample DataFrame
    return pd.DataFrame(
        {
            "field_id": ["field_001", "field_002"],
            "device_id": ["sensor_001", "sensor_002"],
            "timestamp": [1700000000000, 1700000060000],
            "temperature": [22.5, 24.1],
            "moisture": [45.2, 51.3],
            "ph": [6.5, 6.8],
            "nitrogen": [85.0, 92.0],
            "phosphorus": [28.0, 31.0],
            "potassium": [125.0, 118.0],
            "fault_mode": ["NONE", "NONE"],
        }
    )


@task
def validate_and_route(df: pd.DataFrame, topic: str) -> bool:
    """Validate the batch. Route to quarantine on failure."""
    return validate_batch(df)


@task
def write_to_storage(df: pd.DataFrame, topic: str) -> None:
    import os

    path = f"data/landing/{topic}/{pd.Timestamp.now().isoformat().replace(':', '-')}.parquet"
    # ADD THIS LINE — creates data/landing/soil-sensors/ if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[WRITTEN] {len(df)} rows → {path}")


@flow(name="ingestion-validation")
def ingestion_flow(topic: str = "soil-sensors"):
    df = read_kafka_batch(topic)
    valid = validate_and_route(df, topic)

    if valid:
        write_to_storage(df, topic)
    else:
        print(f"[ABORT] Validation failed for {topic}. Data quarantined.")
        raise RuntimeError(f"Ingestion aborted: bad data in {topic}")


if __name__ == "__main__":
    ingestion_flow(topic="soil-sensors")
