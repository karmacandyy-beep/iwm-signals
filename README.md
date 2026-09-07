# iwm-signals

IWM Order-Flow / Volume-Profile signal alerts via GitHub Actions + ntfy.sh

## Files
- `iwm_orderflow_strategy.py` — core strategy (volume profile, absorption/accumulation/aggression, ORB)
- `iwm_signal_check_once.py` — single-run checker triggered by GitHub Actions
- `.github/workflows/iwm_signals.yml` — schedule (every 5 min during market hours)

## Setup
1. Add a repository secret named `NTFY_TOPIC` with your private ntfy.sh topic.
2. The workflow will push alerts when long/short signals fire.
