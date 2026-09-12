#!/usr/bin/env python3
"""
accumulation_distribution_0dte.py
==================================

0DTE IWM options strategy built from two order-flow / volume-profile
concept videos (TikTok, @chartfanatics and @andrea.cimi / "Marketly").

WHERE THE RULES CAME FROM VS. WHAT I ASSUMED
---------------------------------------------
Neither source video had audio available for transcription in the
environment this was built in — only burned-in captions and on-screen
charts could be read. What follows is what was actually visible on
screen, and what had to be filled in to make it a runnable strategy.

From the videos (visual only):
  - Video 1 (@chartfanatics): a consolidation range at the lows with
    large buy-side order-flow clusters ("accumulation"), followed by a
    breakout, with similarly large buy-side clusters appearing again at
    the highs during the continuation ("the same" signature repeating).
  - Video 2 (@andrea.cimi / Marketly): a Volume Profile with a "fair
    value area" marked. Price pushes below the value-area low, forms a
    rejection candle ("sellers absorbed"), and the caption "wait for a
    [confirmation]" implies the entry trigger is a candle closing back
    inside the value area, not the rejection wick itself.

Neither video gave: exact lookback window for the profile, a numeric
rejection threshold, what precisely confirms the candle, stop/target
placement, or how far into a rally you should stop trusting the
"same" buy-side signature. ALL of the numeric rules below (marked
ASSUMPTION) are this script's own choices, not the videos' — treat
them as a starting point to tune or replace, not as verified rules.

STRATEGY LOGIC (translated to 0DTE IWM)
----------------------------------------
1. Build a rolling Volume Profile from IWM's intraday 5-min bars
   (developing value area for the current session, seeded with the
   prior session's bars so it isn't empty at the open). POC = highest
   volume price bin. Value area = tightest band of bins holding 70%
   of session volume (ASSUMPTION: 70% is the standard Market Profile
   convention, not something stated in either video).

2. LONG setup ("accumulation / absorption at value area low"):
   - A bar's low pushes below VAL by more than a buffer
     (ASSUMPTION: 0.15% of price) and closes back above VAL
     ("sellers absorbed").
   - The NEXT bar must close above the absorption bar's high
     ("confirmation candle") to trigger entry. No confirmation within
     3 bars (ASSUMPTION) invalidates the setup.

3. SHORT setup ("distribution at value area high") is the mirror image
   at VAH, using puts.

4. 0DTE options mechanics:
   - Strike chosen by target delta (ASSUMPTION: 0.45) via Black-Scholes,
     since no live option chain is wired in — see DataFeed/IVEstimator.
   - Hard risk rules (all ASSUMPTIONS, not from either video):
     stop = -35% of premium, target = +60% of premium,
     time-stop = 15:45 ET (flat well before the 16:00 close to avoid
     end-of-day gamma/liquidity chaos), one open position at a time.

5. Alerts only — this pushes a notification via ntfy.sh, same as the
   two strategies it replaces. It does NOT place trades through a
   broker.

THIS IS NOT VALIDATED. 0DTE options are extremely high variance.
Backtest and paper-trade before ever risking real capital, and treat
every numeric constant above as a knob to test, not a fact.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

ET = ZoneInfo("America/New_York")
SYMBOL = "IWM"
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "ad_positions.json")

# ---------------------------------------------------------------------------
# Config (every value here is an ASSUMPTION / tunable default, not a rule
# from the source videos)
# ---------------------------------------------------------------------------

VALUE_AREA_PCT = 0.70          # standard Market Profile convention
ABSORPTION_BUFFER_PCT = 0.0015 # how far past VAL/VAH the wick must reach
CONFIRMATION_MAX_BARS = 3      # bars allowed to wait for confirmation
TARGET_DELTA = 0.45
RISK_FREE_RATE = 0.05
STOP_LOSS_PCT = -0.35          # of option premium
PROFIT_TARGET_PCT = 0.60       # of option premium
TIME_STOP_ET = "15:45"
MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"
STRIKE_INCREMENT = 1.0         # IWM lists $1 strikes
CONTRACTS_PER_TRADE = 1        # ASSUMPTION: fixed size; wire in real sizing/account risk before live use


# ---------------------------------------------------------------------------
# Data feed
# ---------------------------------------------------------------------------

class DataFeed:
    """Wraps yfinance for intraday bars. Swap this out for a real broker /
    market-data feed (Tradier, Polygon, IBKR) before trusting live signals --
    yfinance intraday data is delayed/unreliable intraday for real execution.
    """

    def __init__(self, symbol=SYMBOL):
        self.symbol = symbol

    def intraday_bars(self, days=3, interval="5m"):
        import yfinance as yf

        df = yf.download(
            self.symbol,
            period=f"{days}d",
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise RuntimeError("No intraday data returned from yfinance.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def last_price(self, df):
        return float(df["Close"].iloc[-1])


class SyntheticDataFeed(DataFeed):
    """Deterministic fake data for --dry-run testing when markets are
    closed or you just want to exercise the pipeline without hitting
    yfinance."""

    def intraday_bars(self, days=3, interval="5m"):
        rng = np.random.default_rng(7)
        periods_per_day = 78  # 6.5h * 12 five-min bars
        n = periods_per_day * days
        now = datetime.now(ET).replace(second=0, microsecond=0)
        idx = pd.date_range(end=now, periods=n, freq="5min", tz=ET)

        base = 200.0
        # Build a fake accumulation-at-low -> breakout -> distribution-at-high path
        drift = np.concatenate(
            [
                np.linspace(0, -1.2, n // 4),
                np.full(n // 4, -1.2) + rng.normal(0, 0.05, n // 4),
                np.linspace(-1.2, 2.0, n // 4),
                np.full(n - 3 * (n // 4), 2.0) + rng.normal(0, 0.05, n - 3 * (n // 4)),
            ]
        )
        noise = rng.normal(0, 0.15, n)
        close = base + drift + noise
        high = close + np.abs(rng.normal(0.1, 0.08, n))
        low = close - np.abs(rng.normal(0.1, 0.08, n))
        open_ = close - rng.normal(0, 0.05, n)
        volume = rng.integers(50_000, 400_000, n).astype(float)
        # Inflate volume at the accumulation range and the distribution top
        volume[: n // 4] *= 1.8
        volume[-n // 8 :] *= 2.2

        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=idx,
        )
        return df


# ---------------------------------------------------------------------------
# Volume profile
# ---------------------------------------------------------------------------

@dataclass
class VolumeProfile:
    poc: float
    vah: float
    val: float


def build_volume_profile(df: pd.DataFrame, bin_size: float = 0.10) -> VolumeProfile:
    """Volume-weighted price profile over the given bars.

    Each bar's volume is assigned to the price bin containing its
    close (a simplification -- a true footprint would split volume
    across the bar's full range, but that needs tick/quote data this
    feed doesn't have).
    """
    if df.empty:
        raise ValueError("Cannot build a volume profile from empty data.")

    prices = df["Close"].to_numpy()
    volumes = df["Volume"].to_numpy()

    lo, hi = prices.min(), prices.max()
    if hi - lo < bin_size:
        hi = lo + bin_size

    bins = np.arange(lo, hi + bin_size, bin_size)
    bin_idx = np.clip(np.digitize(prices, bins) - 1, 0, len(bins) - 2)

    vol_by_bin = np.zeros(len(bins) - 1)
    for i, v in zip(bin_idx, volumes):
        vol_by_bin[i] += v

    total_vol = vol_by_bin.sum()
    if total_vol <= 0:
        raise ValueError("Zero total volume in profile window.")

    poc_i = int(np.argmax(vol_by_bin))
    poc_price = bins[poc_i] + bin_size / 2

    # Expand outward from POC, adding whichever neighboring bin has more
    # volume, until >= VALUE_AREA_PCT of total volume is captured.
    included = {poc_i}
    captured = vol_by_bin[poc_i]
    lo_i, hi_i = poc_i, poc_i
    while captured / total_vol < VALUE_AREA_PCT and (lo_i > 0 or hi_i < len(vol_by_bin) - 1):
        left_vol = vol_by_bin[lo_i - 1] if lo_i > 0 else -1
        right_vol = vol_by_bin[hi_i + 1] if hi_i < len(vol_by_bin) - 1 else -1
        if right_vol >= left_vol:
            hi_i += 1
            captured += vol_by_bin[hi_i]
            included.add(hi_i)
        else:
            lo_i -= 1
            captured += vol_by_bin[lo_i]
            included.add(lo_i)

    val_price = bins[lo_i]
    vah_price = bins[hi_i] + bin_size

    return VolumeProfile(poc=round(poc_price, 2), vah=round(vah_price, 2), val=round(val_price, 2))


# ---------------------------------------------------------------------------
# Signal engine: absorption + confirmation
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    side: str          # "long" or "short"
    reason: str
    absorption_bar_time: str
    confirmation_bar_time: str
    underlying_price: float


def find_signal(df: pd.DataFrame, vp: VolumeProfile) -> Signal | None:
    """Scan the most recent bars for an absorption bar followed by a
    confirmation bar, at either the value-area low (long) or value-area
    high (short)."""
    recent = df.tail(CONFIRMATION_MAX_BARS + 2)
    if len(recent) < 2:
        return None

    val_buf = vp.val * (1 - ABSORPTION_BUFFER_PCT)
    vah_buf = vp.vah * (1 + ABSORPTION_BUFFER_PCT)

    for i in range(len(recent) - CONFIRMATION_MAX_BARS - 1, len(recent) - 1):
        if i < 0:
            continue
        bar = recent.iloc[i]

        # --- long: absorption at value area low ---
        if bar["Low"] < val_buf and bar["Close"] > vp.val:
            window = recent.iloc[i + 1 : i + 1 + CONFIRMATION_MAX_BARS]
            confirm = window[window["Close"] > bar["High"]]
            if not confirm.empty:
                cbar = confirm.iloc[0]
                return Signal(
                    side="long",
                    reason="Sellers absorbed at value area low; confirmation candle closed above absorption bar high.",
                    absorption_bar_time=str(recent.index[i]),
                    confirmation_bar_time=str(cbar.name),
                    underlying_price=float(cbar["Close"]),
                )

        # --- short: distribution at value area high ---
        if bar["High"] > vah_buf and bar["Close"] < vp.vah:
            window = recent.iloc[i + 1 : i + 1 + CONFIRMATION_MAX_BARS]
            confirm = window[window["Close"] < bar["Low"]]
            if not confirm.empty:
                cbar = confirm.iloc[0]
                return Signal(
                    side="short",
                    reason="Buyers distributed at value area high; confirmation candle closed below absorption bar low.",
                    absorption_bar_time=str(recent.index[i]),
                    confirmation_bar_time=str(cbar.name),
                    underlying_price=float(cbar["Close"]),
                )

    return None


# ---------------------------------------------------------------------------
# Options: IV estimate + Black-Scholes strike/delta selection
# ---------------------------------------------------------------------------

def estimate_iv(df: pd.DataFrame) -> float:
    """ASSUMPTION / simplification: realized volatility of the last ~20
    sessions' worth of 5-min closes, annualized, used as an IV proxy.
    Replace with real implied vol from a live option chain before
    trusting this for actual strike selection -- realized vol
    systematically misprices event/earnings-driven IV skew."""
    closes = df["Close"].resample("1D").last().dropna()
    rets = np.log(closes / closes.shift(1)).dropna()
    if len(rets) < 5:
        return 0.20  # fallback default
    daily_vol = rets.std()
    annualized = daily_vol * math.sqrt(252)
    return float(np.clip(annualized, 0.10, 0.80))


def bs_delta(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return 1.0 if (option_type == "call" and S > K) else (-1.0 if option_type == "put" and S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def select_strike(S, option_type, T, sigma, target_delta=TARGET_DELTA):
    """Search strikes at $1 increments near spot for the one closest to
    the target delta."""
    best_strike, best_diff = None, None
    for offset in np.arange(-10, 10.5, STRIKE_INCREMENT):
        K = round(S + offset)
        if K <= 0:
            continue
        d = bs_delta(S, K, T, RISK_FREE_RATE, sigma, option_type)
        diff = abs(abs(d) - target_delta)
        if best_diff is None or diff < best_diff:
            best_diff, best_strike = diff, K
    return best_strike


# ---------------------------------------------------------------------------
# State + risk management
# ---------------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"open_position": None}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def is_past_time_stop(now_et: datetime) -> bool:
    h, m = map(int, TIME_STOP_ET.split(":"))
    return now_et.time() >= now_et.replace(hour=h, minute=m, second=0, microsecond=0).time()


def is_market_hours(now_et: datetime) -> bool:
    oh, om = map(int, MARKET_OPEN_ET.split(":"))
    ch, cm = map(int, MARKET_CLOSE_ET.split(":"))
    open_t = now_et.replace(hour=oh, minute=om, second=0, microsecond=0).time()
    close_t = now_et.replace(hour=ch, minute=cm, second=0, microsecond=0).time()
    return now_et.weekday() < 5 and open_t <= now_et.time() <= close_t


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def send_alert(title: str, message: str):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print(f"[no NTFY_TOPIC set] {title}: {message}")
        return
    try:
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"Failed to send ntfy alert: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def manage_open_position(state, current_price, now_et):
    pos = state["open_position"]
    if not pos:
        return
    entry_premium = pos["entry_premium_estimate"]
    T = max((pd.Timestamp(pos["expiry"]).tz_localize(ET) - now_et).total_seconds(), 0) / (365 * 24 * 3600)
    sigma = pos["iv_at_entry"]
    cur_premium = bs_price(current_price, pos["strike"], T, RISK_FREE_RATE, sigma, pos["option_type"])
    pnl_pct = (cur_premium - entry_premium) / entry_premium if entry_premium else 0.0

    exit_reason = None
    if pnl_pct <= STOP_LOSS_PCT:
        exit_reason = f"Stop loss hit ({pnl_pct:.0%})"
    elif pnl_pct >= PROFIT_TARGET_PCT:
        exit_reason = f"Profit target hit ({pnl_pct:.0%})"
    elif is_past_time_stop(now_et):
        exit_reason = f"Time-stop {TIME_STOP_ET} ET reached"

    if exit_reason:
        send_alert(
            f"IWM 0DTE EXIT {pos['option_type'].upper()}",
            f"{exit_reason}. Strike {pos['strike']} {pos['option_type']}, "
            f"est. premium {entry_premium:.2f} -> {cur_premium:.2f} ({pnl_pct:+.0%}). "
            f"Underlying {current_price:.2f}.",
        )
        state["open_position"] = None


def check_for_entry(state, df, vp, now_et):
    if state["open_position"] is not None:
        return  # one trade at a time (ASSUMPTION)
    if not is_market_hours(now_et) or is_past_time_stop(now_et):
        return

    signal = find_signal(df, vp)
    if signal is None:
        return

    S = signal.underlying_price
    sigma = estimate_iv(df)
    T = max((now_et.replace(hour=16, minute=0, second=0, microsecond=0) - now_et).total_seconds(), 60) / (365 * 24 * 3600)
    option_type = "call" if signal.side == "long" else "put"
    strike = select_strike(S, option_type, T, sigma)
    premium = bs_price(S, strike, T, RISK_FREE_RATE, sigma, option_type)
    delta = bs_delta(S, strike, T, RISK_FREE_RATE, sigma, option_type)

    state["open_position"] = {
        "side": signal.side,
        "option_type": option_type,
        "strike": strike,
        "expiry": now_et.strftime("%Y-%m-%d"),
        "entry_time": str(now_et),
        "entry_underlying": S,
        "entry_premium_estimate": round(premium, 2),
        "delta_at_entry": round(delta, 3),
        "iv_at_entry": sigma,
        "contracts": CONTRACTS_PER_TRADE,
        "reason": signal.reason,
    }

    send_alert(
        f"IWM 0DTE ENTER {option_type.upper()}",
        f"{signal.reason} Strike {strike} {option_type}, delta {delta:.2f}, "
        f"est. premium {premium:.2f}. Underlying {S:.2f}. "
        f"VAL {vp.val} / POC {vp.poc} / VAH {vp.vah}.",
    )


def run_once(feed: DataFeed):
    now_et = datetime.now(ET)
    df = feed.intraday_bars(days=3)
    vp = build_volume_profile(df)
    current_price = feed.last_price(df)

    state = load_state()
    manage_open_position(state, current_price, now_et)
    check_for_entry(state, df, vp, now_et)
    save_state(state)

    print(
        f"[{now_et.isoformat()}] IWM={current_price:.2f} "
        f"VAL={vp.val} POC={vp.poc} VAH={vp.vah} "
        f"open_position={state['open_position'] is not None}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use real yfinance data.")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data (default if --live not passed).")
    parser.add_argument("--iterations", type=int, default=1, help="Number of check cycles to run.")
    args = parser.parse_args()

    feed = DataFeed() if args.live else SyntheticDataFeed()

    for _ in range(args.iterations):
        run_once(feed)


if __name__ == "__main__":
    main()
