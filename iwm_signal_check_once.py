"""
IWM Signal Check — SINGLE RUN version, designed to be triggered on a
schedule by GitHub Actions (free) instead of running continuously on a
computer you own.

Each run:
  1. Pulls the latest IWM 5-minute bars
  2. Computes the same absorption/accumulation/aggression + ORB signal
  3. If the most recently closed bar has a long/short signal, pushes an
     alert to your phone via ntfy.sh
  4. Exits. GitHub Actions runs this file again on the next schedule tick.

This intentionally does NOT track an open position or do entry/exit P&L
logic like the naked-option script — GitHub Actions runs are stateless
(no memory between runs) unless you add extra storage, which adds a lot
of complexity for a phone-only setup. This version is signal alerts only:
you decide what to do with each one.

Env vars expected (set as GitHub Actions "secrets", not hardcoded):
    NTFY_TOPIC   — your private ntfy.sh topic name
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from iwm_orderflow_strategy import volume_profile, generate_signals

TICKER = "IWM"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


def send_ntfy_alert(title: str, message: str, priority: str = "high"):
    if not NTFY_TOPIC:
        print("NTFY_TOPIC not set — skipping push, printing only.")
        print(title, message)
        return
    try:
        requests.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": "chart_with_upwards_trend"},
            timeout=10,
        )
    except Exception as e:
        print(f"[ntfy push failed: {e}]")


def main():
    df = yf.download(TICKER, period="5d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if len(df) < 30:
        print("Not enough bars yet — market may just be opening.")
        return

    vp = volume_profile(df)
    sig = generate_signals(df)

    last_ts = sig.index[-2] if len(sig) > 1 else sig.index[-1]
    last_price = float(df["Close"].loc[last_ts])
    long_fired = bool(sig.loc[last_ts, "long_signal"])
    short_fired = bool(sig.loc[last_ts, "short_signal"])

    print(f"[{datetime.utcnow().isoformat()}] Bar {last_ts} price {last_price:.2f} "
          f"long={long_fired} short={short_fired} POC={vp['poc']:.2f}")

    if long_fired:
        send_ntfy_alert(
            f"IWM LONG — {last_ts:%H:%M}",
            f"Price {last_price:.2f} | POC {vp['poc']:.2f} VAH {vp['vah']:.2f} VAL {vp['val']:.2f}\n"
            f"Signal only — check the chain and execute yourself if it fits your plan."
        )
    if short_fired:
        send_ntfy_alert(
            f"IWM SHORT — {last_ts:%H:%M}",
            f"Price {last_price:.2f} | POC {vp['poc']:.2f} VAH {vp['vah']:.2f} VAL {vp['val']:.2f}\n"
            f"Signal only — check the chain and execute yourself if it fits your plan."
        )
    if not (long_fired or short_fired):
        print("No signal this run.")


if __name__ == "__main__":
    main()
