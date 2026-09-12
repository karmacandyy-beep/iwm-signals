# iwm-signals

IWM 0DTE options signal alerts via GitHub Actions + ntfy.sh.

This replaces the two prior strategies (order-flow/volume-profile alerts
and the standalone ORB 0DTE script) with a single combined strategy.

## Strategy: Accumulation/Distribution 0DTE

- `accumulation_distribution_0dte.py` — core strategy
- `.github/workflows/accumulation_distribution_0dte.yml` — schedule
  (every 5 min during market hours, real yfinance data, `--live`)
- `state/ad_positions.json` — open-position state, committed back by the
  workflow after each run

### Where this came from

Built from two TikTok clips (@chartfanatics "Fabio Valentini: Trading
Masterclass — Accumulation vs Distribution", and @andrea.cimi /
"Marketly"). **Neither video's audio was transcribable in the build
environment** — only burned-in captions and on-screen charts were
readable. The full mechanical ruleset below is this script's own
translation of that visual concept into code, not a verified transcript
of either creator's actual rules. See the module docstring in
`accumulation_distribution_0dte.py` for the exact breakdown of what came
from the videos vs. what was assumed.

### The logic, in short

1. Build a rolling Volume Profile from IWM's 5-min bars (POC, and a
   70%-of-volume value area — VAL/VAH).
2. **Long setup:** a bar wicks below VAL and closes back above it
   ("sellers absorbed"), then the next bar closes above that bar's high
   ("confirmation") → buy a ~0.45-delta 0DTE call.
3. **Short setup:** the mirror image at VAH ("buyers distributed") → buy
   a ~0.45-delta 0DTE put.
4. Strike selection uses Black-Scholes delta against an estimated IV
   (realized vol proxy — **not** a live option chain; see the
   `estimate_iv` docstring).
5. Risk management: -35% premium stop, +60% premium target, hard
   time-stop at 15:45 ET, one position open at a time.
6. Alerts only, via the same `NTFY_TOPIC` as before — this does **not**
   place trades through a broker.

Every numeric constant (value-area %, buffer size, confirmation window,
target delta, stop/target %, time-stop) is a tunable default at the top
of the script, not a verified rule — see the ASSUMPTION notes in the
docstring and inline comments before trusting this with real money.

## Setup

1. Repository secret `NTFY_TOPIC` (unchanged from before) — your private
   ntfy.sh topic.
2. The workflow runs automatically on market-hours schedule and pushes
   "IWM 0DTE ENTER CALL/PUT" and "IWM 0DTE EXIT ..." alerts.
3. To test locally without live data or market hours:
   `python3 accumulation_distribution_0dte.py --dry-run --iterations 3`

## Known limitations / before using this with real money

- **No transcript** — the entry logic is a best-effort reconstruction
  from silent video frames, not a faithful copy of either creator's
  actual rules. Treat it as a hypothesis to backtest, not a proven edge.
- **IV is estimated from realized volatility**, not pulled from a live
  option chain — this will misprice around events/earnings and doesn't
  reflect real bid/ask or liquidity.
- **yfinance intraday data is delayed and can gap** — not suitable for
  real execution; wire in a real data feed (Tradier/Polygon/IBKR) via
  the `DataFeed` class before trusting live signals.
- **0DTE options are extremely high variance** — fast theta decay, wide
  gamma swings, and this strategy has not been backtested. Alerts are
  not trade recommendations.
