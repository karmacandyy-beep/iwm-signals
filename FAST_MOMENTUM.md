# Separate Fast Momentum strategy

This addition leaves `accumulation_distribution_0dte.py`, its workflow, and its
state unchanged. Both workflows can run; new alerts start with **Fast Momentum**.
It never places orders. Source concepts came from the uploaded momentum clip's
charts and visible captions; the numerical rules below are proposed research
parameters, not the speaker's verified rules or evidence of profitability.

## Signal rules

Default: **1-minute triggers** for the fast setup described in chat. An optional
`--bar-minutes 3` lets you use your usual 3-minute candles; it is a different
configuration, not claimed to have the same results. The 5-minute maximum hold
stays fixed in either mode, so it can supersede the two-bar timeout in 3m mode.

- CALL: close above the prior three completed highs, above same-session VWAP,
  and within the top quarter of the confirmation candle.
- PUT: reverse those conditions and close within the bottom quarter.
- Confirmation volume >= 1.5 times the mean of the preceding ten same-session
  bars. No prior-day warmup. Earliest 1m signal is 09:41 ET; 3m is 10:03 ET.
- Only complete candles, continuous minute history, regular NYSE calendar sessions,
  including holidays and early closes. No volume-profile or opening-range setup.
- Skip if current price has reversed behind the confirmation close or advanced
  over 25% of the confirmation range. Skip confirmations older than 45 seconds.
- Chart stop: opposite extreme of confirmation candle. Momentum timeout after
  two full post-entry candles without exceeding its high (calls) or low (puts).
  Maximum hold five minutes. No new setups in the last 15 minutes of the session.
- One paper setup at a time. Two nonprofitable/unknown underlying outcomes stop
  that day's paper entries. This is a proxy; actual option losses are unknown.

Ordinary OHLCV does not establish absorption. No order-flow data is available.

## What is active and what is not

`fast_momentum.py paper` downloads Yahoo's 1m bars every 15 seconds while running.
It produces **paper direction/setup and chart/time-exit notifications only**.
A poll is not a live tick stream. Provider lag and throttling can suppress entries;
repeated errors stop the runner and attempt a health notification. No made-up
option chain, delta, strike, premium, or profit target is sent.

The independent workflow runs three bounded sessions per trading day. GitHub may
start late or skip scheduled jobs; startup/state-save handoffs can interrupt
monitoring. This is not a guaranteed real-time execution system. The repository's
existing `NTFY_TOPIC` secret is used only inside Actions, never printed or committed.
A successful publish verifies server acceptance, not notification receipt on iOS.

State: `state/fast_momentum.json`, isolated from the existing strategy. A persisted
outbox retries failed deliveries; expired entry alerts are dropped. Ambiguous
network timeouts can still duplicate delivery: stable event IDs identify retries.

## Contract rules and actual option replay

The option replay engine implements same-day expiry, absolute delta 0.55–0.70,
spread <=5% of midpoint, quotes <=5 seconds old, ask-side entry, bid-side exit,
-20% premium stop / +30% target, commissions and adverse slippage. The total
premium plus entry fees is capped at 1% of supplied account equity (USD), with
two losing/censored trades stopping that day. This is **not active option
monitoring** without a real quote feed and account/fill information.

```sh
pip install -r requirements-fast-momentum.txt
python -m unittest discover -s tests -p 'test_fast_momentum.py' -v
python fast_momentum.py backtest
python fast_momentum.py paper --duration-minutes 180 --poll-seconds 15
python fast_momentum.py test-notification
```

For exact historical quotes from a licensed source:

```sh
python fast_momentum.py options-backtest --bars-csv bars.csv \
  --quotes-csv quotes.csv --equity 10000 --output research/options_replay
```

The equity value above is an example, not your actual balance.
Bars CSV: `timestamp,Open,High,Low,Close,Volume` (minute-start timestamps).
Quote CSV: `timestamp,contract,side,expiry,bid,ask,delta,ask_size`.
Timestamps must include timezone offsets; side CALL/PUT; expiry YYYY-MM-DD.
Quote snapshots require no more than 15-second gaps during open positions;
missing coverage censors the result. Entry assumes a one-second decision latency.
Chart stops in quote replay become observable at completed minute close; tick
history would be needed to reconstruct exact intraminute fills. No backtest can
guarantee execution at bid/ask or displayed size.

## Interpreting research results

Default replay is **underlying-only**, with $0.01/share adverse slippage each side,
next-bar-open entries, conservative stop-gap fills, and time exits. It does not
translate +30%/-20% option rules into underlying targets. Any reported win rate or
P&L from this mode describes underlying movement, not 0DTE options. Missing data
is flagged, never silently replaced with synthetic performance. Tests use
fabricated fixtures only to verify code behavior; they are not performance tests.

Real option profitability remains untested until actual historical option data
is provided. The available short sample is exploratory, not an out-of-sample
validation or the 50 paper trades needed for an initial evaluation.

## Sources and operational constraints

- [GitHub schedule limitations](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
- [Alpaca option chains and indicative vs OPRA data](https://docs.alpaca.markets/us/reference/optionchain)
- [ntfy publishing](https://docs.ntfy.sh/publish/)
- [0DTE mechanics and risks](https://www.schwab.com/learn/story/zeroing-on-0dte-options-learn-basics)

A licensed live stock/options feed and a continuously hosted runner are necessary
before representing this as execution-grade real-time option alerts. No paid
subscriptions were purchased, broker trades placed, or actual positions managed.


## Recovery and verification
The separate GitHub Actions workflow schedules recovery attempts at :17 and :47
through a UTC window covering US market hours. A concurrency lock keeps only one
monitor active; a queued attempt can resume after a three-hour run or failure.
The exchange calendar gates all scheduled runs, including holidays and early closes.
GitHub scheduling can be delayed or dropped; this is not guaranteed uninterrupted
real-time hosting. The `check` dispatch mode verifies current completed-bar freshness
and can send a labeled test. Entry messages contain only `BUY CALL IWM @ price`
or `BUY PUT IWM @ price`; the price is the underlying IWM price, not an option premium.
