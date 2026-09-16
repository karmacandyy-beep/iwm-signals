#!/usr/bin/env python3
"""Continuously check the IWM 3-minute strategy during ET market hours.

This process is intended for a long-running worker service. It wakes shortly
after each three-minute candle closes, calls run_once(DataFeed()), and sends
entry/exit alerts through the existing NTFY_TOPIC environment variable.
"""
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from accumulation_distribution_0dte import DataFeed, ET, MARKET_CLOSE_ET, MARKET_OPEN_ET, run_once

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("iwm-worker")
CHECK_SECONDS = 3 * 60
CANDLE_GRACE_SECONDS = 12


def market_window(now):
    if now.weekday() >= 5:
        return False
    oh, om = map(int, MARKET_OPEN_ET.split(":"))
    ch, cm = map(int, MARKET_CLOSE_ET.split(":"))
    return now.replace(hour=oh, minute=om, second=0, microsecond=0) <= now <= now.replace(hour=ch, minute=cm, second=0, microsecond=0)


def seconds_to_next_check(now):
    # Run 12 seconds after the next 3-minute wall-clock boundary.
    epoch = int(now.timestamp())
    next_boundary = ((epoch // CHECK_SECONDS) + 1) * CHECK_SECONDS
    return max(1, next_boundary - epoch + CANDLE_GRACE_SECONDS)


def main():
    feed = DataFeed()
    LOG.info("continuous worker started; 3-minute candles; yfinance feed")
    while True:
        now = datetime.now(ET)
        if market_window(now):
            try:
                run_once(feed)
            except Exception:
                LOG.exception("signal check failed; retrying on the next candle")
        else:
            LOG.info("outside ET market hours; waiting")
        time.sleep(seconds_to_next_check(datetime.now(ET)))


if __name__ == "__main__":
    main()
