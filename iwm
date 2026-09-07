"""
IWM Order-Flow / Volume-Profile Scalping Tool
(Fabio Valentini-style: Absorption -> Accumulation -> Aggression, ORB, Volume Profile)

IMPORTANT — READ BEFORE USING
------------------------------
1. This is a DECISION-SUPPORT / BACKTESTING tool. It does NOT place trades.
   You review the signals and execute manually (or wire it into your own
   broker API later — see the note at the bottom of this file).
2. Free data (yfinance) has NO real bid/ask order flow. "Delta" here is an
   APPROXIMATION built from where price closes within each bar's range,
   weighted by volume. It is a reasonable proxy, not true tape/DOM data.
3. yfinance intraday bars (1m/5m) are only available for the trailing ~60
   days and may be ~15-20 min delayed. Fine for research/backtesting;
   not fine for real-time scalping execution.
4. This is not financial advice. Backtested performance does not guarantee
   future results. Paper-trade before risking real capital.

REQUIREMENTS (run on your own machine, needs internet):
    pip install yfinance pandas numpy matplotlib

USAGE:
    python iwm_orderflow_strategy.py
"""

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def fetch_data(ticker: str = "IWM", period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch OHLCV data. Requires internet + yfinance.
    NOTE: yfinance caps 5m bars at ~60 days and 1m bars at ~7 days — "period"
    beyond that will silently get clipped by Yahoo's API.
    """
    if not YF_AVAILABLE:
        raise ImportError("Install yfinance first: pip install yfinance")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index.name = "datetime"
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Alternative: load your own OHLCV CSV (columns: datetime,Open,High,Low,Close,Volume)."""
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    return df


# ---------------------------------------------------------------------------
# 2. ORDER-FLOW APPROXIMATION (delta)
# ---------------------------------------------------------------------------

def approximate_delta(df: pd.DataFrame) -> pd.Series:
    """
    Proxy for buy vs sell aggression within each bar.
    Close near the High of the bar -> mostly buy volume.
    Close near the Low of the bar -> mostly sell volume.
    """
    bar_range = (df["High"] - df["Low"]).replace(0, np.nan)
    buy_fraction = ((df["Close"] - df["Low"]) / bar_range).clip(0, 1).fillna(0.5)
    buy_vol = df["Volume"] * buy_fraction
    sell_vol = df["Volume"] * (1 - buy_fraction)
    delta = buy_vol - sell_vol
    return delta.rename("delta")


# ---------------------------------------------------------------------------
# 3. VOLUME PROFILE (POC / Value Area)
# ---------------------------------------------------------------------------

def volume_profile(df: pd.DataFrame, bins: int = 40, value_area_pct: float = 0.70):
    price_min, price_max = df["Low"].min(), df["High"].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_by_bin = np.zeros(bins)

    lows = df["Low"].to_numpy()
    highs = df["High"].to_numpy()
    vols = df["Volume"].to_numpy()

    for lo, hi, vol in zip(lows, highs, vols):
        idx_lo = max(0, np.searchsorted(bin_edges, lo, side="right") - 1)
        idx_hi = min(bins - 1, np.searchsorted(bin_edges, hi, side="right") - 1)
        idx_hi = max(idx_hi, idx_lo)
        span = idx_hi - idx_lo + 1
        vol_by_bin[idx_lo:idx_hi + 1] += vol / span

    poc_idx = int(np.argmax(vol_by_bin))
    poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    # Expand outward from POC until value_area_pct of total volume is captured
    total_vol = vol_by_bin.sum()
    target = total_vol * value_area_pct
    included = {poc_idx}
    captured = vol_by_bin[poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while captured < target and (lo_i > 0 or hi_i < bins - 1):
        vol_below = vol_by_bin[lo_i - 1] if lo_i > 0 else -1
        vol_above = vol_by_bin[hi_i + 1] if hi_i < bins - 1 else -1
        if vol_above >= vol_below:
            hi_i += 1
            captured += vol_by_bin[hi_i]
        else:
            lo_i -= 1
            captured += vol_by_bin[lo_i]
        included.add(lo_i)
        included.add(hi_i)

    val = bin_edges[lo_i]
    vah = bin_edges[hi_i + 1]

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "bin_edges": bin_edges,
        "vol_by_bin": vol_by_bin,
    }


# ---------------------------------------------------------------------------
# 4. PATTERN DETECTION: Absorption -> Accumulation -> Aggression
# ---------------------------------------------------------------------------

def detect_absorption(df: pd.DataFrame, lookback: int = 20,
                       vol_mult: float = 1.5, range_mult: float = 0.6) -> pd.Series:
    """High volume + small price range = someone absorbing aggressive orders."""
    avg_vol = df["Volume"].rolling(lookback).mean()
    avg_range = (df["High"] - df["Low"]).rolling(lookback).mean()
    bar_range = df["High"] - df["Low"]
    absorption = (df["Volume"] > avg_vol * vol_mult) & (bar_range < avg_range * range_mult)
    return absorption.fillna(False).rename("absorption")


def detect_accumulation(df: pd.DataFrame, lookback: int = 5, contraction_ratio: float = 0.75) -> pd.Series:
    """Range is contracting relative to the prior window -> coiling before a move."""
    bar_range = df["High"] - df["Low"]
    recent = bar_range.rolling(lookback).mean()
    prior = bar_range.rolling(lookback).mean().shift(lookback)
    accumulation = recent < (prior * contraction_ratio)
    return accumulation.fillna(False).rename("accumulation")


def detect_aggression_breakout(df: pd.DataFrame, delta: pd.Series, lookback: int = 10):
    """Breakout of recent range, confirmed by delta in the same direction."""
    recent_high = df["High"].rolling(lookback).max().shift(1)
    recent_low = df["Low"].rolling(lookback).min().shift(1)
    breakout_up = (df["Close"] > recent_high) & (delta > 0)
    breakout_down = (df["Close"] < recent_low) & (delta < 0)
    return breakout_up.fillna(False).rename("breakout_up"), breakout_down.fillna(False).rename("breakout_down")


def opening_range_breakout(df: pd.DataFrame, orb_minutes: int = 30):
    """Flags breakouts of the first orb_minutes of each session."""
    df = df.copy()
    df["date"] = df.index.date
    orb_high = pd.Series(index=df.index, dtype=float)
    orb_low = pd.Series(index=df.index, dtype=float)

    for _, day_df in df.groupby("date"):
        session_start = day_df.index[0]
        orb_end = session_start + pd.Timedelta(minutes=orb_minutes)
        orb_window = day_df[day_df.index <= orb_end]
        orb_high.loc[day_df.index] = orb_window["High"].max()
        orb_low.loc[day_df.index] = orb_window["Low"].min()

    orb_break_up = df["Close"] > orb_high
    orb_break_down = df["Close"] < orb_low
    return orb_break_up.rename("orb_break_up"), orb_break_down.rename("orb_break_down")


# ---------------------------------------------------------------------------
# 5. SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    delta = approximate_delta(df)
    absorption = detect_absorption(df)
    accumulation = detect_accumulation(df)
    breakout_up, breakout_down = detect_aggression_breakout(df, delta)
    orb_up, orb_down = opening_range_breakout(df)

    # Triple-A: absorption seen recently, then accumulation, then aggressive breakout
    absorption_recent = absorption.rolling(6).max().astype(bool).shift(1).fillna(False)
    accumulation_recent = accumulation.shift(1).fillna(False)

    triple_a_long = absorption_recent & accumulation_recent & breakout_up
    triple_a_short = absorption_recent & accumulation_recent & breakout_down

    signals = pd.DataFrame(index=df.index)
    signals["delta"] = delta
    signals["absorption"] = absorption
    signals["accumulation"] = accumulation
    signals["triple_a_long"] = triple_a_long
    signals["triple_a_short"] = triple_a_short
    signals["orb_long"] = orb_up
    signals["orb_short"] = orb_down
    signals["long_signal"] = triple_a_long | orb_up
    signals["short_signal"] = triple_a_short | orb_down
    return signals


# ---------------------------------------------------------------------------
# 6. SIMPLE BACKTEST (progressive exit, not fixed TP — closer to Fabio's style)
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def backtest(df: pd.DataFrame, signals: pd.DataFrame,
             stop_atr_mult: float = 1.0, trail_atr_mult: float = 1.5,
             max_hold_bars: int = 20) -> pd.DataFrame:
    """
    Very simple long/short backtest:
    - Enter on signal at next bar's open
    - Stop loss at stop_atr_mult * ATR
    - Trailing stop at trail_atr_mult * ATR once in profit
    - Time-based exit after max_hold_bars if neither hit
    """
    atr_series = atr(df)
    trades = []
    in_position = False
    direction = 0
    entry_price = entry_idx = stop_price = best_price = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx_list[i]

        if not in_position:
            if signals.loc[ts, "long_signal"]:
                direction = 1
            elif signals.loc[ts, "short_signal"]:
                direction = -1
            else:
                continue

            entry_price = df["Open"].iloc[i + 1]
            entry_idx = i + 1
            a = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else df["Close"].iloc[i] * 0.005
            stop_price = entry_price - direction * stop_atr_mult * a
            best_price = entry_price
            in_position = True
            continue

        bars_held = i - entry_idx
        price = df["Close"].iloc[i]
        a = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else price * 0.005

        if direction == 1:
            best_price = max(best_price, price)
            trail = best_price - trail_atr_mult * a
            stop_price = max(stop_price, trail) if price > entry_price else stop_price
            hit_stop = df["Low"].iloc[i] <= stop_price
        else:
            best_price = min(best_price, price)
            trail = best_price + trail_atr_mult * a
            stop_price = min(stop_price, trail) if price < entry_price else stop_price
            hit_stop = df["High"].iloc[i] >= stop_price

        if hit_stop or bars_held >= max_hold_bars:
            exit_price = stop_price if hit_stop else price
            pnl = (exit_price - entry_price) * direction
            trades.append({
                "entry_time": idx_list[entry_idx],
                "exit_time": ts,
                "direction": "long" if direction == 1 else "short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_points": pnl,
                "bars_held": bars_held,
            })
            in_position = False

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["cum_pnl"] = trades_df["pnl_points"].cumsum()
    return trades_df


def grid_search(df: pd.DataFrame, signals: pd.DataFrame,
                 stop_mults=(0.75, 1.0, 1.5), trail_mults=(1.0, 1.5, 2.0),
                 max_holds=(10, 20, 40)) -> pd.DataFrame:
    """
    Sweep stop/trail/hold-time parameters and rank by profit factor.
    Small grid on purpose — this is a starting point, not an optimizer
    you should curve-fit hard against a few weeks of data.
    """
    results = []
    for s in stop_mults:
        for t in trail_mults:
            for m in max_holds:
                trades = backtest(df, signals, stop_atr_mult=s, trail_atr_mult=t, max_hold_bars=m)
                if trades.empty:
                    continue
                wins = trades[trades["pnl_points"] > 0]
                losses = trades[trades["pnl_points"] <= 0]
                gross_win = wins["pnl_points"].sum()
                gross_loss = abs(losses["pnl_points"].sum())
                pf = gross_win / gross_loss if gross_loss > 0 else np.inf
                results.append({
                    "stop_atr_mult": s, "trail_atr_mult": t, "max_hold_bars": m,
                    "trades": len(trades),
                    "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
                    "total_pnl_pts": round(trades["pnl_points"].sum(), 3),
                    "profit_factor": round(pf, 2) if np.isfinite(pf) else pf,
                })
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("profit_factor", ascending=False).reset_index(drop=True)
    return results_df


def summarize(trades_df: pd.DataFrame):
    if trades_df.empty:
        print("No trades generated in this window.")
        return
    wins = trades_df[trades_df["pnl_points"] > 0]
    losses = trades_df[trades_df["pnl_points"] <= 0]
    print(f"Total trades:   {len(trades_df)}")
    print(f"Win rate:       {len(wins) / len(trades_df) * 100:.1f}%")
    print(f"Avg win:        {wins['pnl_points'].mean() if len(wins) else 0:.3f} pts")
    print(f"Avg loss:       {losses['pnl_points'].mean() if len(losses) else 0:.3f} pts")
    print(f"Total pnl:      {trades_df['pnl_points'].sum():.3f} pts")
    print(f"Profit factor:  {(wins['pnl_points'].sum() / abs(losses['pnl_points'].sum())) if len(losses) and losses['pnl_points'].sum() != 0 else float('inf'):.2f}")


# ---------------------------------------------------------------------------
# 7. CHARTING
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, signals: pd.DataFrame, vp: dict,
                  trades: pd.DataFrame = None, save_path: str = "iwm_signals_chart.png"):
    """
    Two-panel chart:
      Left:  price with POC/VAH/VAL lines, long/short signal markers, trade
             entry/exit markers if a backtest was run.
      Right: horizontal volume profile histogram.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 4, width_ratios=[3, 3, 3, 1], wspace=0.05)
    ax_price = fig.add_subplot(gs[0, :3])
    ax_vp = fig.add_subplot(gs[0, 3], sharey=ax_price)

    ax_price.plot(df.index, df["Close"], color="#1f77b4", linewidth=1, label="Close")

    ax_price.axhline(vp["poc"], color="orange", linestyle="--", linewidth=1, label="POC")
    ax_price.axhline(vp["vah"], color="gray", linestyle=":", linewidth=1, label="VAH")
    ax_price.axhline(vp["val"], color="gray", linestyle=":", linewidth=1, label="VAL")

    longs = df.index[signals["long_signal"]]
    shorts = df.index[signals["short_signal"]]
    ax_price.scatter(longs, df.loc[longs, "Close"], marker="^", color="green", s=60,
                      label="Long signal", zorder=5)
    ax_price.scatter(shorts, df.loc[shorts, "Close"], marker="v", color="red", s=60,
                      label="Short signal", zorder=5)

    if trades is not None and not trades.empty:
        win_exits = trades[trades["pnl_points"] > 0]
        loss_exits = trades[trades["pnl_points"] <= 0]
        ax_price.scatter(win_exits["exit_time"], win_exits["exit_price"], marker="o",
                          facecolors="none", edgecolors="green", s=50, label="Winning exit")
        ax_price.scatter(loss_exits["exit_time"], loss_exits["exit_price"], marker="x",
                          color="darkred", s=50, label="Losing exit")

    ax_price.set_title("IWM — price, signals, and volume profile levels")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.tick_params(axis="x", rotation=45)

    bin_edges = vp["bin_edges"]
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax_vp.barh(bin_mids, vp["vol_by_bin"], height=(bin_edges[1] - bin_edges[0]) * 0.9,
               color="steelblue", alpha=0.6)
    ax_vp.axhline(vp["poc"], color="orange", linestyle="--", linewidth=1)
    ax_vp.axhline(vp["vah"], color="gray", linestyle=":", linewidth=1)
    ax_vp.axhline(vp["val"], color="gray", linestyle=":", linewidth=1)
    ax_vp.set_title("Volume\nProfile", fontsize=9)
    ax_vp.tick_params(axis="y", labelleft=False)

    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path, dpi=150)
    print(f"\nChart saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKER = "IWM"

    if YF_AVAILABLE:
        print(f"Fetching {TICKER} 5m bars (up to 60 trading days)...")
        data = fetch_data(TICKER, period="60d", interval="5m")
    else:
        raise SystemExit("Install yfinance (`pip install yfinance`) and re-run, "
                          "or use load_csv() with your own data.")

    vp = volume_profile(data)
    print(f"\nVolume Profile — POC: {vp['poc']:.2f}  VAH: {vp['vah']:.2f}  VAL: {vp['val']:.2f}")

    sig = generate_signals(data)
    n_long = int(sig["long_signal"].sum())
    n_short = int(sig["short_signal"].sum())
    print(f"Signals found — Long: {n_long}  Short: {n_short}")

    print("\n--- Parameter sweep (top 5 by profit factor) ---")
    sweep = grid_search(data, sig)
    if not sweep.empty:
        print(sweep.head(5).to_string(index=False))
        best = sweep.iloc[0]
        stop_m, trail_m, hold_m = best["stop_atr_mult"], best["trail_atr_mult"], int(best["max_hold_bars"])
    else:
        print("No trades across the sweep grid — try a longer data window.")
        stop_m, trail_m, hold_m = 1.0, 1.5, 20

    trades = backtest(data, sig, stop_atr_mult=stop_m, trail_atr_mult=trail_m, max_hold_bars=hold_m)
    print(f"\n--- Backtest summary (stop={stop_m}, trail={trail_m}, max_hold={hold_m}) ---")
    summarize(trades)

    if not trades.empty:
        print("\nLast 5 trades:")
        print(trades.tail(5).to_string(index=False))

    try:
        plot_results(data, sig, vp, trades)
    except ImportError:
        print("\nInstall matplotlib to see the chart: pip install matplotlib")

# ---------------------------------------------------------------------------
# NOTE ON GOING LIVE
# ---------------------------------------------------------------------------
# To act on these signals automatically you'd need to:
#   1. Swap fetch_data() for a real-time feed (broker API: e.g. Alpaca,
#      Interactive Brokers, Tradier — IWM is a normal equity/ETF so any
#      US equities broker API works).
#   2. Run generate_signals() on each new bar close.
#   3. Call your broker's order-submission endpoint when long_signal /
#      short_signal fires, sized to your own risk rules.
# I did not wire this up because it requires your broker credentials and
# real capital at risk — that step should be done deliberately by you,
# tested on paper trading first.
