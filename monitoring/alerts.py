from __future__ import annotations

import os
import requests

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
GENERIC_WEBHOOK_URL = os.getenv("GENERIC_WEBHOOK_URL", "")


def send_slack_alert(message: str) -> None:
    if not SLACK_WEBHOOK_URL:
        return
    requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)


def send_webhook_alert(payload: dict) -> None:
    if not GENERIC_WEBHOOK_URL:
        return
    requests.post(GENERIC_WEBHOOK_URL, json=payload, timeout=10)
