"""Bounded ntfy delivery with visible failures and HTTP acknowledgement."""
import logging
import os
import time
import requests

LOG = logging.getLogger(__name__)

def publish(title, message):
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        LOG.error("NTFY_TOPIC is missing; notification not sent")
        return False
    for attempt in range(3):
        try:
            response = requests.post(
                f"https://ntfy.sh/{topic}", data=message.encode("utf-8"),
                headers={"Title": title}, timeout=(3, 5),
            )
            response.raise_for_status()
            if not 200 <= response.status_code < 300:
                LOG.error("ntfy returned unexpected HTTP %s", response.status_code)
                return False
            LOG.info("ntfy accepted notification: %s (attempt %s)", title, attempt + 1)
            return True
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            LOG.warning("ntfy attempt %s failed: %s HTTP=%s", attempt + 1, type(exc).__name__, status)
            if status is not None and 400 <= status < 500 and status != 429:
                break
            if attempt < 2:
                time.sleep((1, 3)[attempt])
    LOG.error("ntfy delivery failed; alert was not acknowledged")
    return False
