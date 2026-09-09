"""
IWM 0DTE Options Strategy — Signal Engine + Live Notifications
=================================================================

WHAT THIS IS
    A rules-based framework that watches IWM intraday, generates entry/exit
    signals for 0DTE long calls/puts based on opening-range + VWAP + relative
    volume, and sends you a notification (Telegram/console) the moment an
    entry or exit condition fires. It does NOT auto-execute trades — it tells
    you when to act. You pull the trigger.

WHAT THIS IS NOT
    - Not a guaranteed-edge system. Backtest it on YOUR data before risking
      capital. 0DTE long options are a negative-theta bet; win rate alone
      doesn't tell you if it's profitable — expectancy does.
    - Not "live" out of the box. Free feeds (yfinance) are delayed and
      unreliable for intraday 0DTE timing. For real use, plug in a real
      data source (see DataFeed section below).

REQUIRED FOR TRUE LIVE USE (you provide these — not included):
    1. A real-time market data feed: Tradier, Polygon.io, IBKR TWS API, or
       your broker's API. yfinance is a placeholder/dry-run stand-in only.
    2. A Telegram bot token + chat ID (free, 5 min setup) OR swap in
       email/Pushover/Discord webhook in the Notifier class.
    3. This script running continuously on a machine that's on during market
       hours (a cheap VPS or your own laptop left open works).

RUN MODES
    python3 orb.py --dry-run     # synthetic data, tests logic end-to-end
    python3 orb.py --live        # requires real DataFeed wired in
"""

import argparse
import logging
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("iwm_0dte")


# ----------------------------------------------------------------------------
# CONFIG — tune these. These are starting defaults, not "optimized" numbers.
# ----------------------------------------------------------------------------
@dataclass
class Config:
    symbol: str = "IWM"

    # --- Signal thresholds ---
    opening_range_minutes: int = 15        # ORB window length
    entry_window_start: dtime = dtime(9, 45)   # don't trade the first 15 min (noise)
    entry_window_end: dtime = dtime(13, 30)    # stop opening NEW positions after this
    hard_time_stop: dtime = dtime(14, 30)      # force-close ALL positions by here (theta cliff)
    rvol_threshold: float = 1.5            # relative volume vs 20-day avg to confirm breakout
    vwap_confirmation: bool = True         # require price on correct side of VWAP

    # --- Option selection ---
    target_delta: float = 0.35             # slightly OTM; leverage without needing a huge move
    assumed_iv: float = 0.22               # placeholder IV if chain data unavailable (IWM ~ typical range)

    # --- Risk management (per-trade, % of option premium) ---
    stop_loss_pct: float = 0.35            # exit if option loses 35% of entry value
    profit_target_pct: float = 0.60        # exit if option gains 60%
    max_concurrent_positions: int = 1
    risk_per_trade_pct: float = 0.01       # % of account risked per trade (position sizing)

    # --- Notifications ---
    # Reuses the same NTFY_TOPIC secret as iwm_signal_check_once.py by default, so you
    # don't need a second ntfy subscription — alerts are distinguished by title
    # ("ORB 0DTE ENTER/EXIT" vs "IWM LONG/SHORT"). Override via env var if you want
    # this strategy on a separate topic instead.
    ntfy_topic: Optional[str] = field(default_factory=lambda: __import__("os").environ.get("NTFY_TOPIC") or None)
    ntfy_server: str = "https://ntfy.sh"
    telegram_bot_token: Optional[str] = None   # alt option — fill in, or set via env var
    telegram_chat_id: Optional[str] = None

    poll_seconds: int = 30                 # how often to check for signals/exits during market hours


CFG = Config()


# ----------------------------------------------------------------------------
# NOTIFIER — swap the `send` internals for email/Discord/Pushover if preferred
# ----------------------------------------------------------------------------
class Notifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def send(self, message: str, title: str = "IWM 0DTE", priority: str = "default"):
        log.info(f"NOTIFY: {message}")

        if self.cfg.ntfy_topic:
            try:
                import requests
                url = f"{self.cfg.ntfy_server}/{self.cfg.ntfy_topic}"
                requests.post(
                    url,
                    data=message.encode("utf-8"),
                    headers={"Title": title, "Priority": priority, "Tags": "chart_with_upwards_trend"},
                    timeout=5,
                )
            except Exception as e:
                log.warning(f"ntfy send failed (falling back to console only): {e}")

        elif self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
            try:
                import requests
                url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
                requests.post(url, data={"chat_id": self.cfg.telegram_chat_id, "text": message}, timeout=5)
            except Exception as e:
                log.warning(f"Telegram send failed (falling back to console only): {e}")
        # else: console log above is the notification in dry-run / unconfigured mode


# ----------------------------------------------------------------------------
# DATA FEED — abstract interface. Swap DryRunFeed for a real broker/data feed.
# ----------------------------------------------------------------------------
class DataFeed:
    """Must return a DataFrame of 1-min bars: columns [open, high, low, close, volume],
    index = tz-aware datetime in ET, for the current session up to 'now'."""

    def get_intraday_bars(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_20day_avg_volume_by_minute(self, symbol: str) -> pd.Series:
        """Series indexed by minute-of-day (e.g. '09:30') -> average volume in that
        minute over the trailing 20 sessions. Needed for relative-volume calc."""
        raise NotImplementedError


class DryRunFeed(DataFeed):
    """Synthetic data generator so you can test the full signal/notification
    pipeline without any live data connection. NOT for real trading decisions."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_intraday_bars(self, symbol: str) -> pd.DataFrame:
        today = datetime.now(ET).date()
        start = datetime.combine(today, dtime(9, 30), tzinfo=ET)
        now = datetime.now(ET)
        minutes = max(1, int((min(now, datetime.combine(today, dtime(16, 0), tzinfo=ET)) - start).total_seconds() // 60))
        idx = pd.date_range(start, periods=minutes, freq="1min", tz=ET)
        price = 295.0 + np.cumsum(self.rng.normal(0, 0.05, size=minutes))
        vol = self.rng.integers(20000, 80000, size=minutes)
        df = pd.DataFrame({
            "open": price, "high": price + 0.03, "low": price - 0.03,
            "close": price, "volume": vol
        }, index=idx)
        return df

    def get_20day_avg_volume_by_minute(self, symbol: str) -> pd.Series:
        minutes = pd.date_range(dtime(9, 30).isoformat(), periods=390, freq="1min").strftime("%H:%M")
        return pd.Series(45000, index=minutes)  # flat placeholder baseline


# ----------------------------------------------------------------------------
# SIGNAL ENGINE
# ----------------------------------------------------------------------------
@dataclass
class Signal:
    direction: str          # "CALL", "PUT", or "NONE"
    reason: str
    price: float
    timestamp: datetime


class SignalEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute_vwap(self, bars: pd.DataFrame) -> pd.Series:
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3
        cum_vol = bars["volume"].cumsum()
        cum_pv = (typical * bars["volume"]).cumsum()
        return cum_pv / cum_vol

    def opening_range(self, bars: pd.DataFrame):
        session_start = bars.index[0]
        orb_end = session_start + timedelta(minutes=self.cfg.opening_range_minutes)
        orb = bars[bars.index < orb_end]
        if orb.empty:
            return None, None
        return orb["high"].max(), orb["low"].min()

    def relative_volume(self, bars: pd.DataFrame, avg_by_minute: pd.Series) -> float:
        last_min = bars.index[-1].strftime("%H:%M")
        recent_vol = bars["volume"].tail(5).mean()
        baseline = avg_by_minute.get(last_min, avg_by_minute.mean())
        return recent_vol / baseline if baseline else 1.0

    def evaluate(self, bars: pd.DataFrame, avg_vol: pd.Series) -> Signal:
        now = bars.index[-1]
        current_time = now.time()

        if not (self.cfg.entry_window_start <= current_time <= self.cfg.entry_window_end):
            return Signal("NONE", "outside entry window", bars["close"].iloc[-1], now)

        orb_high, orb_low = self.opening_range(bars)
        if orb_high is None:
            return Signal("NONE", "opening range not yet formed", bars["close"].iloc[-1], now)

        vwap = self.compute_vwap(bars).iloc[-1]
        price = bars["close"].iloc[-1]
        rvol = self.relative_volume(bars, avg_vol)

        if rvol < self.cfg.rvol_threshold:
            return Signal("NONE", f"rvol {rvol:.2f} below threshold", price, now)

        # Bullish breakout: price above opening range high AND above VWAP
        if price > orb_high and (not self.cfg.vwap_confirmation or price > vwap):
            return Signal("CALL", f"ORB breakout up, rvol={rvol:.2f}, price>{orb_high:.2f}>VWAP", price, now)

        # Bearish breakdown: price below opening range low AND below VWAP
        if price < orb_low and (not self.cfg.vwap_confirmation or price < vwap):
            return Signal("PUT", f"ORB breakdown, rvol={rvol:.2f}, price<{orb_low:.2f}<VWAP", price, now)

        return Signal("NONE", "no breakout condition met", price, now)


# ----------------------------------------------------------------------------
# OPTION PRICING (Black-Scholes) — used to estimate entry price / greeks when
# you don't have a live chain wired in. Replace with real chain data for
# actual fills; this is for signal-stage strike/price estimation only.
# ----------------------------------------------------------------------------
def bs_price_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
        return intrinsic, (1.0 if (option_type == "call" and S > K) else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from scipy.stats import norm
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    return price, delta


def select_strike(spot: float, option_type: str, target_delta: float, iv: float, minutes_to_close: int):
    T = max(minutes_to_close, 1) / (60 * 6.5) / 252  # fraction of a trading year remaining today
    strikes = np.arange(round(spot - 10), round(spot + 10), 1.0)
    best = None
    for K in strikes:
        _, delta = bs_price_delta(spot, K, T, 0.04, iv, option_type)
        d = abs(delta) if option_type == "call" else abs(delta)
        if best is None or abs(d - target_delta) < abs(best[1] - target_delta):
            best = (K, d)
    K = best[0]
    price, delta = bs_price_delta(spot, K, T, 0.04, iv, option_type)
    return K, price, delta


# ----------------------------------------------------------------------------
# POSITION MANAGER
# ----------------------------------------------------------------------------
@dataclass
class Position:
    option_type: str
    strike: float
    entry_price: float
    entry_time: datetime
    contracts: int


class PositionManager:
    def __init__(self, cfg: Config, notifier: Notifier):
        self.cfg = cfg
        self.notifier = notifier
        self.positions: List[Position] = []

    def can_enter(self) -> bool:
        return len(self.positions) < self.cfg.max_concurrent_positions

    def enter(self, signal: Signal, spot: float):
        opt_type = "call" if signal.direction == "CALL" else "put"
        now = datetime.now(ET)
        close_dt = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET)
        minutes_to_close = int((close_dt - now).total_seconds() // 60)

        strike, price, delta = select_strike(spot, opt_type, self.cfg.target_delta, self.cfg.assumed_iv, minutes_to_close)
        pos = Position(opt_type, strike, price, now, contracts=1)
        self.positions.append(pos)

        self.notifier.send(
            f"IWM ${spot:.2f} | Strike ${strike:.0f} | "
            f"Est. premium ${price:.2f} (Δ{delta:.2f}) | Reason: {signal.reason}\n"
            f"Plan: stop @ -{self.cfg.stop_loss_pct*100:.0f}%, target @ +{self.cfg.profit_target_pct*100:.0f}%, "
            f"hard exit by {self.cfg.hard_time_stop.strftime('%H:%M')} ET",
            title=f"🟢 ORB 0DTE ENTER {opt_type.upper()}",
            priority="high",
        )

    def check_exits(self, spot: float, iv: float = None):
        iv = iv or self.cfg.assumed_iv
        now = datetime.now(ET)
        close_dt = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET)
        minutes_to_close = max(int((close_dt - now).total_seconds() // 60), 0)
        T = minutes_to_close / (60 * 6.5) / 252

        still_open = []
        for pos in self.positions:
            current_price, _ = bs_price_delta(spot, pos.strike, T, 0.04, iv, pos.option_type)
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price if pos.entry_price else 0

            reason = None
            if now.time() >= self.cfg.hard_time_stop:
                reason = "hard time stop (theta cliff)"
            elif pnl_pct <= -self.cfg.stop_loss_pct:
                reason = f"stop loss hit ({pnl_pct:+.0%})"
            elif pnl_pct >= self.cfg.profit_target_pct:
                reason = f"profit target hit ({pnl_pct:+.0%})"

            if reason:
                self.notifier.send(
                    f"${pos.strike:.0f} {pos.option_type.upper()} — "
                    f"entry ${pos.entry_price:.2f} -> now ${current_price:.2f} ({pnl_pct:+.0%}) | {reason}",
                    title=f"🔴 ORB 0DTE EXIT {pos.option_type.upper()}",
                    priority="urgent",
                )
            else:
                still_open.append(pos)
        self.positions = still_open


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def market_is_open(now: datetime) -> bool:
    return dtime(9, 30) <= now.time() <= dtime(16, 0) and now.weekday() < 5


def run(feed: DataFeed, cfg: Config, max_iterations: Optional[int] = None):
    notifier = Notifier(cfg)
    engine = SignalEngine(cfg)
    pm = PositionManager(cfg, notifier)
    notifier.send(f"IWM 0DTE strategy started — polling every {cfg.poll_seconds}s")

    iteration = 0
    while True:
        now = datetime.now(ET)
        if not market_is_open(now):
            log.info("Market closed — sleeping.")
            if max_iterations is not None:
                break
            time.sleep(60)
            continue

        bars = feed.get_intraday_bars(cfg.symbol)
        avg_vol = feed.get_20day_avg_volume_by_minute(cfg.symbol)
        spot = bars["close"].iloc[-1]

        # 1. Check exits on any open position first
        pm.check_exits(spot)

        # 2. Look for new entry signal
        if pm.can_enter():
            signal = engine.evaluate(bars, avg_vol)
            if signal.direction != "NONE":
                pm.enter(signal, spot)
            else:
                log.info(f"No signal — {signal.reason} (spot ${spot:.2f})")

        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data, run a fixed number of iterations")
    parser.add_argument("--live", action="store_true", help="Use a real DataFeed (you must wire one in below)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations to run in --dry-run mode")
    args = parser.parse_args()

    if args.live:
        raise SystemExit(
            "No live data feed is wired in. Implement a DataFeed subclass using your broker/data "
            "provider's API (Tradier, Polygon, IBKR) and pass it to run() instead of DryRunFeed()."
        )
    else:
        log.info("Running in DRY-RUN mode with synthetic data (safe to test).")
        run(DryRunFeed(), CFG, max_iterations=args.iterations)
