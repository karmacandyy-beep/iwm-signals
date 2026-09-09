"""
IWM 0DTE Options Strategy — Signal Engine + Live Notifications
=================================================================

WHAT THIS IS
    A rules-based framework that watches IWM intraday, generates BUY CALL /
    BUY PUT entry+exit signals based on opening-range + VWAP + relative
    volume, and pushes a notification (ntfy/Telegram/console) the moment a
    signal fires. It does NOT auto-execute trades — it tells you when to
    act. You place the order yourself.

DATA SOURCE (as of this version)
    Uses yfinance (free, unofficial Yahoo Finance data) for both the
    underlying price bars AND the real IWM option chain (real strikes,
    real bid/ask, real implied vol per strike) — same data source your
    other strategy in this repo already uses successfully.
    - Delayed ~15-20 minutes, not true real-time.
    - IWM has Mon/Wed/Fri expirations, not necessarily every single day —
      if there's no same-day expiry available, the script automatically
      falls back to the nearest expiry and labels the alert clearly so
      you're never misled into thinking it's true 0DTE when it isn't.
    - If the live option chain can't be fetched for any reason, it falls
      back to a Black-Scholes estimate (clearly labeled as an estimate).

    For true real-time / official broker data, swap in a broker API
    (IBKR/Tradier/Polygon) — see the DataFeed class below.

RUN MODES
    python3 orb.py --dry-run     # synthetic data, tests logic end-to-end
    python3 orb.py --live        # real yfinance underlying + option chain
"""

import argparse
import logging
import time
import math
from dataclasses import dataclass, field
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
log = logging.getLogger("iwm_0dte")


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
@dataclass
class Config:
    symbol: str = "IWM"

    # --- Signal thresholds ---
    opening_range_minutes: int = 15
    entry_window_start: dtime = dtime(9, 45)
    entry_window_end: dtime = dtime(15, 30)
    hard_time_stop: dtime = dtime(15, 45)
    rvol_threshold: float = 1.5
    vwap_confirmation: bool = True

    # --- Option selection ---
    target_delta: float = 0.35
    assumed_iv: float = 0.22       # fallback only, used if live chain unavailable

    # --- Risk management ---
    stop_loss_pct: float = 0.35
    profit_target_pct: float = 0.60
    max_concurrent_positions: int = 1
    risk_per_trade_pct: float = 0.01

    # --- Notifications ---
    ntfy_topic: Optional[str] = field(default_factory=lambda: __import__("os").environ.get("NTFY_TOPIC") or None)
    ntfy_server: str = "https://ntfy.sh"
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    poll_seconds: int = 30

    # --- Persistence (so exit alerts survive GitHub Actions restarting fresh each run) ---
    state_file: str = "state/positions.json"


CFG = Config()


# ----------------------------------------------------------------------------
# NOTIFIER
# ----------------------------------------------------------------------------
class Notifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def send(self, message: str, title: str = "IWM 0DTE", priority: str = "default"):
        log.info(f"NOTIFY [{title}]: {message}")

        if self.cfg.ntfy_topic:
            try:
                import requests
                url = f"{self.cfg.ntfy_server}/{self.cfg.ntfy_topic}"
                requests.post(
                    url,
                    data=message.encode("utf-8"),
                    headers={"Title": title, "Priority": priority, "Tags": "chart_with_upwards_trend"},
                    timeout=10,
                )
            except Exception as e:
                log.warning(f"ntfy send failed (falling back to console only): {e}")
        elif self.cfg.telegram_bot_token and self.cfg.telegram_chat_id:
            try:
                import requests
                url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
                requests.post(url, data={"chat_id": self.cfg.telegram_chat_id, "text": message}, timeout=10)
            except Exception as e:
                log.warning(f"Telegram send failed (falling back to console only): {e}")


# ----------------------------------------------------------------------------
# DATA FEED
# ----------------------------------------------------------------------------
class DataFeed:
    def get_intraday_bars(self, symbol: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_20day_avg_volume_by_minute(self, symbol: str) -> pd.Series:
        raise NotImplementedError


class DryRunFeed(DataFeed):
    """Synthetic data — safe pipeline test only, not real signals."""

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
        return pd.DataFrame({
            "open": price, "high": price + 0.03, "low": price - 0.03,
            "close": price, "volume": vol
        }, index=idx)

    def get_20day_avg_volume_by_minute(self, symbol: str) -> pd.Series:
        minutes = pd.date_range(dtime(9, 30).isoformat(), periods=390, freq="1min").strftime("%H:%M")
        return pd.Series(45000, index=minutes)


class YFinanceFeed(DataFeed):
    """Real underlying price bars via yfinance. Delayed ~15-20 min."""

    def get_intraday_bars(self, symbol: str) -> pd.DataFrame:
        if not YF_AVAILABLE:
            raise ImportError("Install yfinance: pip install yfinance")
        df = yf.download(symbol, period="5d", interval="5m", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df.dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
        today = datetime.now(ET).date()
        df = df[df.index.date == today]
        return df

    def get_20day_avg_volume_by_minute(self, symbol: str) -> pd.Series:
        if not YF_AVAILABLE:
            raise ImportError("Install yfinance: pip install yfinance")
        df = yf.download(symbol, period="20d", interval="5m", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Volume": "volume"})
        df = df.dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
        minute_key = df.index.strftime("%H:%M")
        return df.groupby(minute_key)["volume"].mean()


# ----------------------------------------------------------------------------
# LIVE OPTION CHAIN — real strikes/bid-ask/IV from yfinance, with graceful
# fallback to a Black-Scholes estimate if the chain can't be fetched.
# ----------------------------------------------------------------------------
class OptionChainProvider:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol) if YF_AVAILABLE else None

    def get_expiry(self) -> Tuple[Optional[str], bool]:
        """Returns (expiry_date_str, is_true_0dte). Falls back to nearest
        available expiry if today isn't one (IWM isn't always same-day)."""
        if not self.ticker:
            return None, False
        try:
            expiries = self.ticker.options
        except Exception as e:
            log.warning(f"Could not fetch option expiries: {e}")
            return None, False
        if not expiries:
            return None, False
        today_str = datetime.now(ET).date().isoformat()
        if today_str in expiries:
            return today_str, True
        return expiries[0], False  # nearest future expiry, NOT same-day

    def select_strike(self, spot: float, option_type: str, target_delta: float, minutes_to_close: int):
        """Returns (strike, premium, delta, expiry, is_true_0dte, is_live_quote)."""
        expiry, is_true_0dte = self.get_expiry()
        if expiry is None:
            K, price, delta = select_strike_bs(spot, option_type, target_delta, CFG.assumed_iv, minutes_to_close)
            return K, price, delta, None, False, False

        try:
            chain = self.ticker.option_chain(expiry)
            df = chain.calls if option_type == "call" else chain.puts
            df = df.dropna(subset=["strike"])
        except Exception as e:
            log.warning(f"Could not fetch option chain, falling back to B-S estimate: {e}")
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

    def get_quote(self, strike: float, option_type: str, expiry: str) -> Optional[float]:
        """Fetch current premium for an already-open position, for exit P&L."""
        try:
            chain = self.ticker.option_chain(expiry)
            df = chain.calls if option_type == "call" else chain.puts
            row = df.iloc[(df["strike"] - strike).abs().argsort()[:1]]
            if row.empty:
                return None
            bid, ask, last = row["bid"].iloc[0], row["ask"].iloc[0], row["lastPrice"].iloc[0]
            return (bid + ask) / 2 if (bid and ask and bid > 0 and ask > 0) else (last or None)
        except Exception as e:
            log.warning(f"Could not fetch live quote for exit check: {e}")
            return None


# ----------------------------------------------------------------------------
# SIGNAL ENGINE
# ----------------------------------------------------------------------------
@dataclass
class Signal:
    direction: str
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

        if price > orb_high and (not self.cfg.vwap_confirmation or price > vwap):
            return Signal("CALL", f"ORB breakout up, rvol={rvol:.2f}, price>{orb_high:.2f}>VWAP", price, now)

        if price < orb_low and (not self.cfg.vwap_confirmation or price < vwap):
            return Signal("PUT", f"ORB breakdown, rvol={rvol:.2f}, price<{orb_low:.2f}<VWAP", price, now)

        return Signal("NONE", "no breakout condition met", price, now)


# ----------------------------------------------------------------------------
# OPTION PRICING (Black-Scholes) — fallback only, used when live chain
# data isn't available.
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


def select_strike_bs(spot: float, option_type: str, target_delta: float, iv: float, minutes_to_close: int):
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


# ----------------------------------------------------------------------------
# POSITION MANAGER
# ----------------------------------------------------------------------------
@dataclass
class Position:
    option_type: str
    strike: float
    entry_price: float
    entry_time: datetime
    expiry: Optional[str] = None
    is_true_0dte: bool = False
    is_live_quote: bool = False
    contracts: int = 1

    def to_dict(self) -> dict:
        return {
            "option_type": self.option_type,
            "strike": self.strike,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "expiry": self.expiry,
            "is_true_0dte": self.is_true_0dte,
            "is_live_quote": self.is_live_quote,
            "contracts": self.contracts,
        }

    @staticmethod
    def from_dict(d: dict) -> "Position":
        return Position(
            option_type=d["option_type"],
            strike=d["strike"],
            entry_price=d["entry_price"],
            entry_time=datetime.fromisoformat(d["entry_time"]),
            expiry=d.get("expiry"),
            is_true_0dte=d.get("is_true_0dte", False),
            is_live_quote=d.get("is_live_quote", False),
            contracts=d.get("contracts", 1),
        )


class PositionManager:
    def __init__(self, cfg: Config, notifier: Notifier, chain: Optional[OptionChainProvider] = None):
        self.cfg = cfg
        self.notifier = notifier
        self.chain = chain
        self.positions: List[Position] = []

    def load_state(self, path: str):
        """Load open positions from disk. GitHub Actions runs are stateless
        processes, so without this, exit alerts (stop-loss/take-profit) would
        never fire -- every run would forget any position from the last run."""
        import json, os
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.positions = [Position.from_dict(d) for d in data]
            if self.positions:
                log.info(f"Loaded {len(self.positions)} open position(s) from {path}")
        except Exception as e:
            log.warning(f"Could not load position state ({path}): {e}")

    def save_state(self, path: str):
        """Save open positions to disk so the next GitHub Actions run knows
        about them and can check for exits."""
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([p.to_dict() for p in self.positions], f, indent=2)

    def can_enter(self) -> bool:
        return len(self.positions) < self.cfg.max_concurrent_positions

    def enter(self, signal: Signal, spot: float):
        opt_type = "call" if signal.direction == "CALL" else "put"
        now = datetime.now(ET)
        close_dt = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET)
        minutes_to_close = int((close_dt - now).total_seconds() // 60)

        if self.chain:
            strike, price, delta, expiry, is_true_0dte, is_live = self.chain.select_strike(
                spot, opt_type, self.cfg.target_delta, minutes_to_close
            )
        else:
            strike, price, delta = select_strike_bs(spot, opt_type, self.cfg.target_delta, self.cfg.assumed_iv, minutes_to_close)
            expiry, is_true_0dte, is_live = None, False, False

        pos = Position(opt_type, strike, price, now, expiry, is_true_0dte, is_live, contracts=1)
        self.positions.append(pos)

        expiry_note = ""
        if expiry and not is_true_0dte:
            expiry_note = f"\n⚠️ No same-day expiry available — using nearest expiry {expiry} (NOT true 0DTE)"
        quote_note = "live chain quote" if is_live else "Black-Scholes estimate (no live chain data)"

        self.notifier.send(
            f"BUY {opt_type.upper()} - IWM ${strike:.0f} strike\n"
            f"Premium: ${price:.2f}/contract ({quote_note})\n"
            f"Why: {signal.reason}\n"
            f"Exit plan: stop -{self.cfg.stop_loss_pct*100:.0f}% / target +{self.cfg.profit_target_pct*100:.0f}% / "
            f"hard exit {self.cfg.hard_time_stop.strftime('%H:%M')} ET{expiry_note}",
            title=f"BUY {opt_type.upper()} - IWM ${spot:.2f}",
            priority="high",
        )

    def check_exits(self, spot: float):
        now = datetime.now(ET)
        close_dt = datetime.combine(now.date(), dtime(16, 0), tzinfo=ET)
        minutes_to_close = max(int((close_dt - now).total_seconds() // 60), 0)
        T = minutes_to_close / (60 * 6.5) / 252

        still_open = []
        for pos in self.positions:
            current_price = None
            if self.chain and pos.expiry:
                current_price = self.chain.get_quote(pos.strike, pos.option_type, pos.expiry)
            if current_price is None:
                current_price, _ = bs_price_delta(spot, pos.strike, T, 0.04, self.cfg.assumed_iv, pos.option_type)

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
                    f"CLOSE {pos.option_type.upper()} ${pos.strike:.0f}\n"
                    f"Entry: ${pos.entry_price:.2f}  ->  Now: ${current_price:.2f} ({pnl_pct:+.0%})\n"
                    f"Why: {reason}",
                    title=f"SELL {pos.option_type.upper()} - close now",
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


def run(feed: DataFeed, cfg: Config, chain: Optional[OptionChainProvider] = None, max_iterations: Optional[int] = None):
    notifier = Notifier(cfg)
    engine = SignalEngine(cfg)
    pm = PositionManager(cfg, notifier, chain)
    pm.load_state(cfg.state_file)
    mode = "LIVE (real yfinance data)" if isinstance(feed, YFinanceFeed) else "DRY-RUN (synthetic data)"
    log.info(f"IWM 0DTE strategy started — {mode}, polling every {cfg.poll_seconds}s")  # log only, no push

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
        if bars.empty:
            log.warning("No bars returned — market may be closed or feed issue. Retrying next cycle.")
            time.sleep(cfg.poll_seconds)
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            continue

        avg_vol = feed.get_20day_avg_volume_by_minute(cfg.symbol)
        spot = bars["close"].iloc[-1]

        pm.check_exits(spot)

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

    pm.save_state(cfg.state_file)
    log.info(f"Saved position state ({len(pm.positions)} open) to {cfg.state_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Synthetic data, tests pipeline only")
    parser.add_argument("--live", action="store_true", help="Real yfinance underlying + live option chain")
    parser.add_argument("--iterations", type=int, default=5, help="Max iterations (omit for --live to run continuously)")
    args = parser.parse_args()

    if args.live:
        if not YF_AVAILABLE:
            raise SystemExit("Install yfinance first: pip install yfinance")
        log.info("Running in LIVE mode — real yfinance underlying data + real option chain (delayed ~15-20min).")
        run(YFinanceFeed(), CFG, chain=OptionChainProvider(CFG.symbol), max_iterations=args.iterations if args.iterations != 5 else None)
    else:
        log.info("Running in DRY-RUN mode with synthetic data (safe to test).")
        run(DryRunFeed(), CFG, max_iterations=args.iterations)
