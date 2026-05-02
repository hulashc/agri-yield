# ingestion/ndvi_puller.py
"""
Fetches Sentinel-2 satellite scenes via STAC API.
Computes NDVI from B04 (Red) and B08 (NIR) bands.
Produces NDVI readings to the satellite-ndvi Kafka topic.
"""

import json
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import rasterio
from pystac_client import Client
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

STAC_URL = "https://earth-search.aws.element84.com/v1"  # Element84 Sentinel-2 catalog
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "satellite-ndvi"

# Define your field bounding boxes (lon_min, lat_min, lon_max, lat_max)
# These are example coordinates in the East Midlands, UK
FIELD_BBOXES = {
    "field_001": (-1.2, 52.5, -1.1, 52.6),
    "field_002": (-1.3, 52.6, -1.2, 52.7),
    # Add more fields here
}


# ── STAC scene query ──────────────────────────────────────────────────────────


def fetch_latest_scene(
    field_id: str, bbox: tuple, lookback_days: int = 14
) -> Optional[dict]:
    """
    Query STAC for the most recent Sentinel-2 scene covering this field.
    Returns scene metadata or None if no scenes found.
    """
    catalog = Client.open(STAC_URL)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={"eo:cloud_cover": {"lt": 80}},  # filter out heavily clouded scenes
        sortby="-datetime",
        max_items=1,
    )

    items = list(search.items())
    if not items:
        logger.warning(
            f"No scenes found for {field_id} in the last {lookback_days} days"
        )
        return None

    item = items[0]
    return {
        "scene_id": item.id,
        "datetime": item.datetime,
        "cloud_cover": item.properties.get("eo:cloud_cover", 100),
        "b04_url": item.assets["red"].href,  # Red band
        "b08_url": item.assets["nir"].href,  # Near Infrared band
    }


# ── NDVI computation ──────────────────────────────────────────────────────────


def compute_ndvi(b04_url: str, b08_url: str) -> Optional[float]:
    """
    Download B04 and B08 bands from a scene and compute mean NDVI over the scene.
    Returns float in [-1, 1] or None if download fails.
    """
    try:
        with rasterio.open(b04_url) as red_ds:
            red = red_ds.read(1).astype(float)

        with rasterio.open(b08_url) as nir_ds:
            nir = nir_ds.read(1).astype(float)

        # Avoid division by zero
        denominator = nir + red
        denominator[denominator == 0] = np.nan

        ndvi = (nir - red) / denominator

        # Mean NDVI across the scene, ignoring nodata
        mean_ndvi = float(np.nanmean(ndvi))
        return round(mean_ndvi, 4)

    except Exception as e:
        logger.error(f"Failed to compute NDVI: {e}")
        return None


# ── Producer ──────────────────────────────────────────────────────────────────


def produce_ndvi_message(
    producer: Producer, field_id: str, scene: dict, ndvi: Optional[float]
):
    """Produce an NDVI message to Kafka."""
    message = {
        "field_id": field_id,
        "scene_id": scene["scene_id"],
        "timestamp": int(scene["datetime"].timestamp() * 1000),
        "ndvi": ndvi,
        "cloud_cover_pct": scene["cloud_cover"],
        "ndvi_interpolated": ndvi is None,  # True if we couldn't compute it
        "ndvi_proxied": False,
        "source": "sentinel-2",
    }

    producer.produce(
        TOPIC,
        key=field_id,
        value=json.dumps(message, default=str),
    )
    producer.flush()
    logger.info(
        f"Produced NDVI for {field_id}: ndvi={ndvi}, cloud_cover={scene['cloud_cover']}%"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────


def run():
    """Poll for new scenes every 6 hours."""
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

    while True:
        for field_id, bbox in FIELD_BBOXES.items():
            logger.info(f"Checking for new scene: {field_id}")
            scene = fetch_latest_scene(field_id, bbox)

            if scene:
                ndvi = compute_ndvi(scene["b04_url"], scene["b08_url"])
                produce_ndvi_message(producer, field_id, scene, ndvi)
            else:
                logger.warning(f"Skipping {field_id} — no scene available")

        logger.info("Sleeping 6 hours before next scene check...")
        time.sleep(6 * 3600)


if __name__ == "__main__":
    run()
