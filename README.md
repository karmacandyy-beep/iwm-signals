# iwm-signals

IWM signal alerts via GitHub Actions + ntfy.sh. Two independent strategies, each with its own schedule and script — pick whichever alerts you want to act on.

## Strategies

### 1. Order-flow / Volume Profile (original)
- `iwm_orderflow_strategy.py` — core strategy (volume profile, absorption/accumulation/aggression, ORB)
- `iwm_signal_check_once.py` — single-run checker triggered by GitHub Actions
- `.github/workflows/iwm_signals.yml` — schedule (every 5 min during market hours)

### 2. ORB 0DTE Options (new)
- `orb.py` — opening-range breakout + VWAP + relative-volume signal engine for 0DTE long calls/puts, with position/risk management (stop loss, profit target, hard time-stop) and Black-Scholes strike selection
- `.github/workflows/orb_0dte.yml` — separate schedule (every 5 min during market hours)
- Currently runs in `--dry-run` mode (synthetic data) until a real intraday data feed (Tradier/Polygon/IBKR) is wired into the `DataFeed` class — see the script's docstring for details

Both strategies push to the same `NTFY_TOPIC` by default; alerts are distinguished by title (e.g. "IWM LONG" vs "ORB 0DTE ENTER CALL") so you can tell which strategy fired.

## Setup
1. Add a repository secret named `NTFY_TOPIC` with your private ntfy.sh topic.
2. Both workflows will push alerts when their respective signals fire.
3. `orb.py` needs a real `DataFeed` implementation before its signals reflect real market conditions — right now it's proving the pipeline runs, not generating live signals.
