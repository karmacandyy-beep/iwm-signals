"""
IWM VWAP Pullback Continuation — 0DTE Long Calls
=================================================================

RULES (as specified)
    1. TREND FILTER (5-minute chart):
       - 5m uptrend confirmed (EMA9 > EMA21)
       - Price above 5m VWAP
       - RVOL >= 1.5 (order-flow / volume confirmation)
    2. ENTRY TRIGGER (1-minute chart), only once trend filter is armed:
       - Wait for a pullback toward VWAP that does NOT break the 5m
         bullish structure (no 1m close meaningfully below VWAP)
       - Entry fires when a bullish 1m candle closes above the PRIOR
         1m candle's high, on higher volume than that prior candle
    3. EXIT:
       - Stop: 1m structure breaks (close below the pullback's low) OR
         option loses more than stop_loss_pct (hard backstop)
       - Targets: 1R and 2R (R = entry spot price - structural stop
         price, in underlying terms). At 1R: alert to take partial
         profit + trail stop to breakeven. At 2R: alert full exit.
         If structure breaks after 1R but before 2R: trail-stop exit.
    4. NO-TRADE FILTER:
       - Every cycle explicitly reports WHY no trade was taken when
         conditions aren't met (not just silence) — see NoTradeReason.

STATE / PERSISTENCE
    GitHub Actions runs this fresh every ~5 min (no memory between runs).
    Signal DETECTION (trend/pullback/trigger) is recomputed from scratch
    each run using the full day's bars, so it's stateless and safe.
    An OPEN POSITION, however, must persist across runs (you already
    got the entry alert — we can't lose track of it) — that's saved to
    vwap_pullback_state.json, which the GitHub Actions workflow commits
    back to the repo after each run.

DATA SOURCE: yfinance (same as this repo's other strategies). Delayed
    ~15-20 min. Real IWM option chain used for strike/premium where
    available; falls back to a labeled Black-Scholes estimate otherwise.

RUN MODES
    python3 vwap_pullback_0dte.py --dry-run   # synthetic data, tests logic
    python3 vwap_pullback_0dte.py --live      # real yfinance data
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

ET = ZoneInfo("America/New_York")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("vwap_pullback")

STATE_FILE = "vwap_pullback_state.json"


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
@dataclass
class Config:
    symbol: str = "IWM"

    # --- 5m trend filter ---
    ema_fast: int = 9
    ema_slow: int = 21
    rvol_threshold: float = 1.5

    # --- 1m pullback / trigger ---
    pullback_vwap_buffer_pct: float = 0.15   # "near VWAP" = within this % of VWAP
    structure_break_pct: float = 0.30        # pullback invalidated if 1m close this far below VWAP

    # --- Entry window ---
    entry_window_start: dtime = dtime(9, 45)
    entry_window_end: dtime = dtime(15, 0)    # leave room for a trade to play out before close
    hard_time_stop: dtime = dtime(15, 45)

    # --- Option selection ---
    target_delta: float = 0.35
    assumed_iv: float = 0.22

    # --- Risk management ---
    stop_loss_pct: float = 0.35     # hard option-level backstop, independent of structure stop
    r_multiple_targets: Tuple[float, float] = (1.0, 2.0)
    max_concurrent_positions: int = 1

    # --- Notifications ---
    ntfy_topic: Optional[str] = field(default_factory=lambda: os.environ.get("NTFY_TOPIC") or None)
    ntfy_server: str = "https://ntfy.sh"


CFG = Config()


# ----------------------------------------------------------------------------
# NOTIFIER
# ----------------------------------------------------------------------------
class Notifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def send(self, message: str, title: str = "VWAP Pullback", priority: str = "default"):
        log.info(f"NOTIFY [{title}]: {message}")
        if self.cfg.ntfy_topic:
            try:
                import requests
                url = f"{self.cfg.ntfy_server}/{self.cfg.ntfy_topic}"
                requests.post(
                    url, data=message.encode("utf-8"),
                    headers={"Title": title, "Priority": priority, "Tags": "chart_with_upwards_trend"},
                    timeout=10,
                )
            except Exception as e:
                log.warning(f"ntfy send failed: {e}")


# ----------------------------------------------------------------------------
# DATA FETCHING
# ----------------------------------------------------------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df.dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(ET)
    else:
        df.index = df.index.tz_convert(ET)
    return df


def fetch_bars(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    df = _normalize(df)
    today = datetime.now(ET).date()
    return df[df.index.date == today]


def fetch_20day_avg_volume_by_minute(symbol: str, interval: str = "5m") -> pd.Series:
    df = yf.download(symbol, period="20d", interval=interval, progress=False, auto_adjust=False)
    df = _normalize(df)
    minute_key = df.index.strftime("%H:%M")
    return df.groupby(minute_key)["volume"].mean()


class DryRunData:
    """Synthetic 5m + 1m bars that deliberately walk through: uptrend ->
    pullback to VWAP -> breakout trigger, so the full pipeline can be
    tested without live data."""

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def bars(self, interval_minutes: int, n: int, start_price: float = 295.0) -> pd.DataFrame:
        today = datetime.now(ET).date()
        start = datetime.combine(today, dtime(9, 30), tzinfo=ET)
        idx = pd.date_range(start, periods=n, freq=f"{interval_minutes}min", tz=ET)
        # ramp up, then pullback, then breakout
        third = n // 3
        up = np.linspace(0, 2.5, third)
        pull = np.linspace(2.5, 1.2, third)
        breakout = np.linspace(1.2, 3.0, n - 2 * third)
        moves = np.concatenate([up, pull, breakout])
        price = start_price + moves + self.rng.normal(0, 0.02, size=n)
        vol = self.rng.integers(20000, 40000, size=n).astype(float)
        vol[-3:] *= 2.2  # volume spike into the breakout
        return pd.DataFrame({
            "open": price - 0.02, "high": price + 0.04, "low": price - 0.05,
            "close": price, "volume": vol
        }, index=idx)


# ----------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_vwap(bars: pd.DataFrame) -> pd.Series:
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_vol = bars["volume"].cumsum()
    cum_pv = (typical * bars["volume"]).cumsum()
    return cum_pv / cum_vol


def relative_volume(bars: pd.DataFrame, avg_by_minute: pd.Series) -> float:
    last_min = bars.index[-1].strftime("%H:%M")
    recent_vol = bars["volume"].tail(3).mean()
    baseline = avg_by_minute.get(last_min, avg_by_minute.mean()) if avg_by_minute is not None else recent_vol
    return recent_vol / baseline if baseline else 1.0


# ----------------------------------------------------------------------------
# 5-MINUTE TREND FILTER
# ----------------------------------------------------------------------------
@dataclass
class TrendCheck:
    ok: bool
    reason: str
    vwap: float = 0.0
    rvol: float = 0.0


def evaluate_5m_trend(cfg: Config, bars_5m: pd.DataFrame, avg_vol_5m: Optional[pd.Series]) -> TrendCheck:
    if len(bars_5m) < cfg.ema_slow:
        return TrendCheck(False, f"not enough 5m bars yet ({len(bars_5m)}/{cfg.ema_slow})")

    ema_f = ema(bars_5m["close"], cfg.ema_fast).iloc[-1]
    ema_s = ema(bars_5m["close"], cfg.ema_slow).iloc[-1]
    vwap = compute_vwap(bars_5m).iloc[-1]
    price = bars_5m["close"].iloc[-1]
    rvol = relative_volume(bars_5m, avg_vol_5m)

    if ema_f <= ema_s:
        return TrendCheck(False, f"5m EMA{cfg.ema_fast} ({ema_f:.2f}) not above EMA{cfg.ema_slow} ({ema_s:.2f}) — no uptrend", vwap, rvol)
    if price <= vwap:
        return TrendCheck(False, f"price ${price:.2f} not above 5m VWAP ${vwap:.2f}", vwap, rvol)
    if rvol < cfg.rvol_threshold:
        return TrendCheck(False, f"rvol {rvol:.2f} below threshold {cfg.rvol_threshold}", vwap, rvol)

    return TrendCheck(True, f"5m uptrend confirmed (EMA{cfg.ema_fast}>{cfg.ema_slow}, price>VWAP, rvol={rvol:.2f})", vwap, rvol)


# ----------------------------------------------------------------------------
# 1-MINUTE PULLBACK + TRIGGER (stateless — recomputed from today's full 1m series)
# ----------------------------------------------------------------------------
@dataclass
class TriggerCheck:
    fired: bool
    reason: str
    structural_stop: Optional[float] = None
    entry_price: Optional[float] = None


def evaluate_1m_pullback_trigger(cfg: Config, bars_1m: pd.DataFrame, vwap_5m: float) -> TriggerCheck:
    """Scans today's 1m bars once the 5m trend is confirmed: looks for a
    pullback toward VWAP that doesn't break structure, then a breakout
    of the prior 1m candle's high on rising volume."""
    if len(bars_1m) < 3:
        return TriggerCheck(False, "not enough 1m bars yet to evaluate pullback")

    buffer_near = vwap_5m * (cfg.pullback_vwap_buffer_pct / 100)
    buffer_break = vwap_5m * (cfg.structure_break_pct / 100)

    pullback_low = None
    armed_pullback = False

    for i in range(1, len(bars_1m)):
        close = bars_1m["close"].iloc[i]
        low = bars_1m["low"].iloc[i]

        # Structure broken: a 1m close well below VWAP invalidates the setup for today's scan
        if close < vwap_5m - buffer_break:
            armed_pullback = False
            pullback_low = None
            continue

        # Pullback detected: price came near VWAP
        if abs(close - vwap_5m) <= buffer_near:
            armed_pullback = True
            pullback_low = low if pullback_low is None else min(pullback_low, low)
            continue

        # Once a pullback has formed, check for the breakout trigger candle
        if armed_pullback and i >= 1:
            prior_high = bars_1m["high"].iloc[i - 1]
            prior_vol = bars_1m["volume"].iloc[i - 1]
            cur_open = bars_1m["open"].iloc[i]
            cur_close = bars_1m["close"].iloc[i]
            cur_vol = bars_1m["volume"].iloc[i]
            is_bullish = cur_close > cur_open
            breaks_high = cur_close > prior_high
            higher_vol = cur_vol > prior_vol

            if is_bullish and breaks_high and higher_vol:
                stop = (pullback_low if pullback_low is not None else bars_1m["low"].iloc[i - 1]) - 0.02
                return TriggerCheck(
                    True,
                    f"1m bullish candle closed ${cur_close:.2f}, broke prior high ${prior_high:.2f} on rising volume",
                    structural_stop=stop,
                    entry_price=cur_close,
                )

    if armed_pullback:
        return TriggerCheck(False, f"pullback to VWAP formed (low ${pullback_low:.2f}), waiting for breakout trigger candle")
    return TriggerCheck(False, "trend confirmed but no pullback to VWAP has occurred yet today")


# ----------------------------------------------------------------------------
# OPTION PRICING (Black-Scholes fallback) + LIVE CHAIN
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


def select_strike_bs(spot, option_type, target_delta, iv, minutes_to_close):
    T = max(minutes_to_close, 1) / (60 * 6.5) / 252
    strikes = np.arange(round(spot - 10), round(spot + 10), 1.0)
    best = None
    for K in strikes:
        _, delta = bs_price_delta(spot, K, T, 0.04, iv, option_type)
        d = abs(delta)
        if best is None or abs(d - target_delta) < abs(best[1] - target_delta):
            best = (K, d)
    K = best[0]
    price, delta = bs_price_delta(spot, K, T, 0.04, iv, option_type)
    return K, price, delta


class OptionChainProvider:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol) if YF_AVAILABLE else None

    def get_expiry(self) -> Tuple[Optional[str], bool]:
        if not self.ticker:
            return None, False
        try:
            expiries = self.ticker.options
        except Exception as e:
            log.warning(f"Could not fetch expiries: {e}")
            return None, False
        if not expiries:
            return None, False
        today_str = datetime.now(ET).date().isoformat()
        return (today_str, True) if today_str in expiries else (expiries[0], False)

    def select_strike(self, spot, option_type, target_delta, minutes_to_close):
        expiry, is_true_0dte = self.get_expiry()
        if expiry is None:
            K, price, delta = select_strike_bs(spot, option_type, target_delta, CFG.assumed_iv, minutes_to_close)
            return K, price, delta, None, False, False
        try:
            chain = self.ticker.option_chain(expiry)
            df = (chain.calls if option_type == "call" else chain.puts).dropna(subset=["strike"])
        except Exception as e:
            log.warning(f"Chain fetch failed, using B-S estimate: {e}")
            K, price, delta = select_strike_bs(spot, option_type, target_delta, CFG.assumed_iv, minutes_to_close)
            return K, price, delta, expiry, is_true_0dte, False

        T = max(minutes_to_close, 1) / (60 * 6.5) / 252
        best = None
        for _, row in df.iterrows():
            iv = row.get("impliedVolatility", None)
            if not iv or iv <= 0:
                continue
            _, delta = bs_price_delta(spot, row["strike"], T, 0.04, iv, option_type)
            d = abs(delta)
            if best is None or abs(d - target_delta) < abs(best[1] - target_delta):
                best = (row, d)
        if best is None:
            K, price, delta = select_strike_bs(spot, option_type, target_delta, CFG.assumed_iv, minutes_to_close)
            return K, price, delta, expiry, is_true_0dte, False
        row, delta = best
        bid, ask, last = row.get("bid", 0), row.get("ask", 0), row.get("lastPrice", 0)
        premium = (bid + ask) / 2 if (bid and ask and bid > 0 and ask > 0) else (last or 0)
        return row["strike"], premium, delta, expiry, is_true_0dte, True

    def get_quote(self, strike, option_type, expiry) -> Optional[float]:
        try:
            chain = self.ticker.option_chain(expiry)
            df = chain.calls if option_type == "call" else chain.puts
            row = df.iloc[(df["strike"] - strike).abs().argsort()[:1]]
            if row.empty:
                return None
            bid, ask, last = row["bid"].iloc[0], row["ask"].iloc[0], row["lastPrice"].iloc[0]
            return (bid + ask) / 2 if (bid and ask and bid > 0 and ask > 0) else (last or None)
        except Exception as e:
            log.warning(f"Live quote fetch failed: {e}")
            return None


# ----------------------------------------------------------------------------
# STATE PERSISTENCE (open position must survive across GitHub Actions runs)
# ----------------------------------------------------------------------------
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"date": None, "position": None}
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        return {"date": None, "position": None}
    today_str = datetime.now(ET).date().isoformat()
    if state.get("date") != today_str:
        return {"date": today_str, "position": None}  # new trading day, reset
    return state


def save_state(state: dict):
    state["date"] = datetime.now(ET).date().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ----------------------------------------------------------------------------
# MAIN LOGIC — one evaluation pass (matches GitHub Actions single-shot model)
# ----------------------------------------------------------------------------
def market_is_open(now: datetime) -> bool:
    return dtime(9, 30) <= now.time() <= dtime(16, 0) and now.weekday() < 5


def run_once(cfg: Config, notifier: Notifier, chain: Optional[OptionChainProvider],
             bars_5m: pd.DataFrame, bars_1m: pd.DataFrame, avg_vol_5m: Optional[pd.Series]):
    state = load_state()
    now = datetime.now(ET)

    # ---- If a position is already open, only manage its exit ----
    if state.get("position"):
        pos = state["position"]
        spot = bars_1m["close"].iloc[-1] if not bars_1m.empty else bars_5m["close"].iloc[-1]
        close_dt = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET)
        minutes_to_close = max(int((close_dt - now).total_seconds() // 60), 0)
        T = minutes_to_close / (60 * 6.5) / 252

        current_price = None
        if chain and pos.get("expiry"):
            current_price = chain.get_quote(pos["strike"], "call", pos["expiry"])
        if current_price is None:
            current_price, _ = bs_price_delta(spot, pos["strike"], T, 0.04, cfg.assumed_iv, "call")

        pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] else 0
        r1_price = pos["entry_spot"] + 1.0 * pos["r_dollars"]
        r2_price = pos["entry_spot"] + 2.0 * pos["r_dollars"]

        # Priority: time stop > trailing/structural stop (post-1R) > hard %
        # backstop (pre-1R only — once trailed to breakeven, the structural
        # stop is what governs, not the original % loss limit).
        exit_reason = None
        if now.time() >= cfg.hard_time_stop:
            exit_reason = "hard time stop (theta cliff)"
        elif pos.get("hit_1r") and not bars_1m.empty and bars_1m["close"].iloc[-1] < pos["structural_stop"]:
            exit_reason = "trailing stop hit after 1R — momentum faded"
        elif not pos.get("hit_1r") and not bars_1m.empty and bars_1m["close"].iloc[-1] < pos["structural_stop"]:
            exit_reason = "1m structure broke before 1R — stop out"
        elif not pos.get("hit_1r") and pnl_pct <= -cfg.stop_loss_pct:
            exit_reason = f"option stop loss hit ({pnl_pct:+.0%})"
        elif not pos.get("hit_1r") and spot >= r1_price:
            pos["hit_1r"] = True
            pos["structural_stop"] = pos["entry_spot"]  # trail stop to breakeven
            notifier.send(
                f"IWM ${spot:.2f} reached 1R (${r1_price:.2f}). Option ~${current_price:.2f} ({pnl_pct:+.0%}).\n"
                f"Suggested: take partial profit here, trail stop to breakeven (${pos['entry_spot']:.2f}) on the rest, target 2R = ${r2_price:.2f}.",
                title="🟡 1R HIT — take partial, trail stop",
                priority="high",
            )
            save_state(state)
            return
        elif pos.get("hit_1r") and spot >= r2_price:
            exit_reason = f"2R target hit (${r2_price:.2f})"

        if exit_reason:
            notifier.send(
                f"CALL ${pos['strike']:.0f} — entry ${pos['entry_price']:.2f} -> now ${current_price:.2f} "
                f"({pnl_pct:+.0%}) | {exit_reason}",
                title="🔴 SELL CALL (close position)",
                priority="urgent",
            )
            state["position"] = None
            save_state(state)
        else:
            log.info(f"Position open: {pos['strike']} CALL, spot ${spot:.2f}, pnl {pnl_pct:+.0%}, no exit yet")
            save_state(state)
        return

    # ---- No open position: look for a new entry ----
    if not (cfg.entry_window_start <= now.time() <= cfg.entry_window_end):
        log.info("NO TRADE — outside entry window")
        return

    trend = evaluate_5m_trend(cfg, bars_5m, avg_vol_5m)
    if not trend.ok:
        log.info(f"NO TRADE — {trend.reason}")
        return

    trigger = evaluate_1m_pullback_trigger(cfg, bars_1m, trend.vwap)
    if not trigger.fired:
        log.info(f"NO TRADE — trend confirmed, but: {trigger.reason}")
        return

    spot = trigger.entry_price
    r_dollars = spot - trigger.structural_stop
    if r_dollars <= 0:
        log.info("NO TRADE — invalid stop distance (structural stop above entry), skipping")
        return

    close_dt = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET)
    minutes_to_close = int((close_dt - now).total_seconds() // 60)

    if chain:
        strike, price, delta, expiry, is_true_0dte, is_live = chain.select_strike(spot, "call", cfg.target_delta, minutes_to_close)
    else:
        strike, price, delta = select_strike_bs(spot, "call", cfg.target_delta, cfg.assumed_iv, minutes_to_close)
        expiry, is_true_0dte, is_live = None, False, False

    state["position"] = {
        "strike": float(strike), "entry_price": float(price), "entry_spot": float(spot),
        "structural_stop": float(trigger.structural_stop), "r_dollars": float(r_dollars),
        "expiry": expiry, "hit_1r": False, "entry_time": now.isoformat(),
    }
    save_state(state)

    expiry_note = f"\n⚠️ No same-day expiry — using {expiry} (NOT true 0DTE)" if (expiry and not is_true_0dte) else ""
    quote_note = "live chain quote" if is_live else "Black-Scholes estimate"
    r1_price, r2_price = spot + r_dollars, spot + 2 * r_dollars

    notifier.send(
        f"BUY CALL — IWM ${spot:.2f} | Strike ${strike:.0f} | Premium ${price:.2f} (Δ{delta:.2f}, {quote_note})\n"
        f"Reason: {trend.reason}; {trigger.reason}\n"
        f"Structural stop: ${trigger.structural_stop:.2f} | 1R: ${r1_price:.2f} | 2R: ${r2_price:.2f} | "
        f"hard exit by {cfg.hard_time_stop.strftime('%H:%M')} ET{expiry_note}",
        title="🟢 BUY CALL (VWAP pullback)",
        priority="high",
    )


def main_live(cfg: Config):
    if not YF_AVAILABLE:
        raise SystemExit("Install yfinance: pip install yfinance")
    now = datetime.now(ET)
    notifier = Notifier(cfg)
    if not market_is_open(now):
        log.info("Market closed — nothing to do this run.")
        return
    bars_5m = fetch_bars(cfg.symbol, "5m", "5d")
    bars_1m = fetch_bars(cfg.symbol, "1m", "5d")
    if bars_5m.empty or bars_1m.empty:
        log.warning("No bars returned — feed issue or market just opened.")
        return
    avg_vol_5m = fetch_20day_avg_volume_by_minute(cfg.symbol, "5m")
    chain = OptionChainProvider(cfg.symbol)
    run_once(cfg, notifier, chain, bars_5m, bars_1m, avg_vol_5m)


def main_dry_run(cfg: Config):
    notifier = Notifier(cfg)
    data = DryRunData()
    bars_5m = data.bars(5, 15)
    bars_1m = data.bars(1, 75)
    run_once(cfg, notifier, None, bars_5m, bars_1m, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.live:
        log.info("Running LIVE — real yfinance data + option chain.")
        main_live(CFG)
    else:
        log.info("Running DRY-RUN — synthetic uptrend/pullback/breakout data.")
        main_dry_run(CFG)
