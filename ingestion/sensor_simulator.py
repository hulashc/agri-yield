# ingestion/sensor_simulator.py
"""
IoT sensor simulator for agri-yield pipeline.
Produces realistic soil sensor readings to the soil-sensors Kafka topic.
Includes fault injection: dropout, calibration drift, network bursts.
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from confluent_kafka import Producer
from faker import Faker

fake = Faker()

# Ã¢â€â‚¬Ã¢â€â‚¬ Configuration Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

KAFKA_BOOTSTRAP = "localhost:9094"  # Change to k8s service when deployed
SCHEMA_REGISTRY_URL = "http://localhost:8081"
TOPIC = "soil-sensors"

# Field IDs to simulate (pretend these are real farm fields)
FIELD_IDS = [f"field_{i:03d}" for i in range(1, 11)]  # field_001 to field_010

# Crop types and their NPK priors (realistic agronomic values mg/kg)
CROP_NPK_PRIORS = {
    "wheat": {"N": (80, 15), "P": (25, 5), "K": (120, 20)},
    "maize": {"N": (110, 20), "P": (30, 8), "K": (150, 25)},
    "barley": {"N": (70, 12), "P": (20, 4), "K": (100, 15)},
    "oilseed": {"N": (90, 18), "P": (35, 7), "K": (130, 22)},
}

FIELD_CROPS = {fid: random.choice(list(CROP_NPK_PRIORS.keys())) for fid in FIELD_IDS}


# Ã¢â€â‚¬Ã¢â€â‚¬ Fault injection helpers Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


@dataclass
class DeviceState:
    """Tracks per-device fault state across time."""

    field_id: str
    drift_bias: float = 0.0  # accumulates slowly
    burst_buffer: list = field(default_factory=list)
    is_offline: bool = False
    offline_until: float = 0.0


def seasonal_temperature(day_of_year: int) -> float:
    """
    Returns a base soil temperature using a sinusoidal seasonal curve.
    Peaks in summer (~day 200), troughs in winter (~day 20).
    Range: 10Ã¢â‚¬â€œ35Ã‚Â°C
    """
    seasonal = 22.5 + 12.5 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
    noise = random.gauss(0, 1.5)
    return round(seasonal + noise, 2)


def daily_moisture(hour: int) -> float:
    """
    Moisture has a daily cycle Ã¢â‚¬â€ lowest in early afternoon, highest at dawn.
    Range: 20Ã¢â‚¬â€œ80%
    """
    daily = 50 + 20 * math.cos(2 * math.pi * (hour - 6) / 24)
    noise = random.gauss(0, 3)
    return round(max(20, min(80, daily + noise)), 2)


def npk_reading(crop: str) -> tuple:
    """Draw NPK values from crop-specific Gaussian distributions."""
    priors = CROP_NPK_PRIORS[crop]
    N = round(max(0, random.gauss(*priors["N"])), 2)
    P = round(max(0, random.gauss(*priors["P"])), 2)
    K = round(max(0, random.gauss(*priors["K"])), 2)
    return N, P, K


def inject_fault(
    value: Optional[float], device: DeviceState, fault_probability: float = 0.10
) -> tuple[Optional[float], str]:
    """
    Apply one of three fault modes with 10% combined probability:
    - DROPOUT: return null (simulates sensor failure)
    - DRIFT: add slow accumulating bias (simulates calibration drift)
    - BURST: buffer the reading and release many at once
    """
    if random.random() > fault_probability:
        return value, "NONE"

    fault = random.choice(["DROPOUT", "DRIFT", "BURST"])

    if fault == "DROPOUT":
        return None, "DROPOUT"

    elif fault == "DRIFT":
        device.drift_bias += random.uniform(0.1, 0.5)  # bias accumulates
        return round(value + device.drift_bias, 2) if value else None, "DRIFT"

    elif fault == "BURST":
        # Buffer this reading Ã¢â‚¬â€ it will be flushed later all at once
        device.burst_buffer.append(value)
        return None, "BURST"  # return null now, flush later

    return value, "NONE"


# Ã¢â€â‚¬Ã¢â€â‚¬ Message builder Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def build_message(field_id: str, device: DeviceState) -> dict:
    """Build one sensor reading message with potential faults injected."""
    now = datetime.now(timezone.utc)
    doy = now.timetuple().tm_yday
    hour = now.hour
    crop = FIELD_CROPS[field_id]

    temp = seasonal_temperature(doy)
    moisture = daily_moisture(hour)
    ph = round(random.gauss(6.5, 0.3), 2)
    N, P, K = npk_reading(crop)
    ph = max(5.5, min(7.5, ph))

    # Inject faults
    temp, fault = inject_fault(temp, device)  # type: ignore[assignment]
    temp = temp or 0.0
    moisture, _ = inject_fault(moisture, device, fault_probability=0.05)  # type: ignore[assignment]
    moisture = moisture or 0.0
    N, _ = inject_fault(N, device, fault_probability=0.05)  # type: ignore[assignment]
    N = N or 0.0

    return {
        "field_id": field_id,
        "device_id": f"sensor_{field_id}",
        "timestamp": int(now.timestamp() * 1000),  # milliseconds
        "temperature": temp,
        "moisture": moisture,
        "ph": ph,
        "nitrogen": N,
        "phosphorus": P,
        "potassium": K,
        "fault_mode": fault,
    }


# Ã¢â€â‚¬Ã¢â€â‚¬ Kafka producer Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def create_producer() -> Producer:
    return Producer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "client.id": "soil-sensor-simulator",
        }
    )


def delivery_report(err, msg):
    """Called when a message is delivered or fails."""
    if err:
        print(f"[DELIVERY FAILED] {err}")
    else:
        print(
            f"[DELIVERED] {msg.topic()} partition={msg.partition()} offset={msg.offset()}"
        )


# Ã¢â€â‚¬Ã¢â€â‚¬ Main simulation loop Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def run(interval_seconds: float = 15.0):
    """
    Produce one reading per field every `interval_seconds`.
    Real sensors run at 15-minute intervals. Set interval_seconds=1 for fast testing.
    """
    producer = create_producer()
    devices = {fid: DeviceState(field_id=fid) for fid in FIELD_IDS}

    print(f"Starting simulator. Producing to topic={TOPIC} every {interval_seconds}s")
    print(f"Simulating {len(FIELD_IDS)} fields: {FIELD_IDS}")

    try:
        while True:
            for field_id in FIELD_IDS:
                device = devices[field_id]
                message = build_message(field_id, device)

                # Flush burst buffer occasionally
                if device.burst_buffer and random.random() < 0.3:
                    for buffered in device.burst_buffer:
                        producer.produce(
                            TOPIC,
                            key=field_id,
                            value=json.dumps(
                                {"field_id": field_id, "buffered_value": buffered}
                            ),
                            callback=delivery_report,
                        )
                    device.burst_buffer.clear()

                producer.produce(
                    TOPIC,
                    key=field_id,
                    value=json.dumps(message),
                    callback=delivery_report,
                )

            producer.poll(0)  # trigger delivery callbacks
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")
    finally:
        producer.flush()


if __name__ == "__main__":
    run(interval_seconds=1.0)  # fast mode for testing
