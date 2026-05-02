# ingestion/weather_poller.py
"""
Polls NOAA GHCN for hourly weather data.
Uses exponential backoff for rate limits.
Sends permanent failures to dead-letter queue topic.
"""

import json
import logging
import time
from datetime import datetime, timezone

import requests
from confluent_kafka import Producer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "weather-events"
DLQ_TOPIC = "weather-events-dlq"  # dead-letter queue topic

# NOAA GHCN API — free, no key needed for basic access
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOAA_TOKEN = "YOUR_NOAA_TOKEN"  # register at ncdc.noaa.gov

# Target station IDs (GHCN station IDs near your fields)
STATION_IDS = ["GHCND:UK000003658", "GHCND:UK000003769"]

# Map stations to field IDs
STATION_TO_FIELD = {
    "GHCND:UK000003658": "field_001",
    "GHCND:UK000003769": "field_002",
}


# ── Fetching with retry ───────────────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(requests.exceptions.HTTPError),
)
def fetch_station_data(station_id: str, date: str) -> dict:
    """
    Fetch weather readings for a station and date from NOAA GHCN.
    Retries up to 5 times with exponential backoff on HTTP errors.
    """
    response = requests.get(
        f"{NOAA_BASE_URL}/data",
        params={
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": date,
            "enddate": date,
            "datatypeid": "TMAX,TMIN,PRCP,AWND",
            "units": "metric",
            "limit": "100",
        },
        headers={"token": NOAA_TOKEN},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


# ── Message builder ───────────────────────────────────────────────────────────


def parse_weather_message(station_id: str, raw: dict) -> dict:
    """Parse NOAA GHCN response into our weather event schema."""
    results = raw.get("results", [])
    data = {r["datatype"]: r["value"] for r in results}

    field_id = STATION_TO_FIELD.get(station_id, "unknown")
    temp = data.get("TMAX") or data.get("TMIN")
    awnd = data.get("AWND")

    return {
        "station_id": station_id,
        "field_id": field_id,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        "temperature": temp / 10
        if temp is not None
        else None,  # NOAA returns tenths of Celsius
        "precipitation": data.get("PRCP", 0) / 10,  # tenths of mm
        "humidity": None,
        "wind_speed": awnd / 10 if awnd is not None else None,
        "source": "noaa_ghcn",
    }


# ── Producer setup ────────────────────────────────────────────────────────────


def get_producer() -> Producer:
    return Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})


def send_to_dlq(producer: Producer, station_id: str, error: str):
    """Route permanently failed fetches to the dead-letter queue."""
    dlq_message = {
        "station_id": station_id,
        "error": error,
        "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
    }
    producer.produce(DLQ_TOPIC, key=station_id, value=json.dumps(dlq_message))
    producer.flush()
    logger.warning(f"[DLQ] Sent failed fetch for {station_id} to {DLQ_TOPIC}")


# ── Main loop ─────────────────────────────────────────────────────────────────


def run():
    """Poll all stations once per hour."""
    producer = get_producer()

    while True:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for station_id in STATION_IDS:
            try:
                raw = fetch_station_data(station_id, today)
                message = parse_weather_message(station_id, raw)

                producer.produce(TOPIC, key=station_id, value=json.dumps(message))
                producer.flush()
                logger.info(
                    f"Produced weather event for {station_id}: temp={message['temperature']}°C"
                )

            except Exception as e:
                # All retries exhausted — send to dead-letter queue
                send_to_dlq(producer, station_id, str(e))

        logger.info("Sleeping 1 hour until next poll...")
        time.sleep(3600)


if __name__ == "__main__":
    run()
