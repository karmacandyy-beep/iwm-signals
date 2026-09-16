#!/usr/bin/env python3
"""Independent fast-momentum research/paper-alert engine. Never submits orders.
Numeric rules are proposed adaptations, not a transcript or validated edge.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo('America/New_York')
COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

@dataclass(frozen=True)
class Config:
    symbol: str = 'IWM'
    bar_minutes: int = 1
    pause_bars: int = 3
    volume_bars: int = 10
    volume_multiple: float = 1.5
    close_fraction: float = .75
    chase_fraction: float = .25
    stop_premium: float = .20
    target_premium: float = .30
    max_hold_minutes: int = 5
    max_losses: int = 2
    max_signal_age_seconds: int = 45
    delta_min: float = .55
    delta_max: float = .70
    max_spread: float = .05
    max_quote_age_seconds: int = 5


def utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError('Timestamps must include a timezone')
    return t.tz_convert('UTC')


def session_bounds(day):
    import exchange_calendars as xc
    cal = xc.get_calendar('XNYS')
    day = pd.Timestamp(day).normalize().tz_localize(None)
    if not cal.is_session(day):
        return None
    return cal.session_open(day), cal.session_close(day)


def normalize(raw):
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'timestamp' in df:
        df = df.set_index('timestamp')
    # CSV inputs must explicitly specify timezone; no silent local/UTC guesses.
    idx = pd.DatetimeIndex([utc(t) for t in df.index])
    df.index = idx
    df = df[COLS].apply(pd.to_numeric, errors='raise').sort_index()
    if df.index.has_duplicates:
        raise ValueError('Duplicate bar timestamps')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('Missing or nonfinite OHLCV')
    if ((df.High < df[['Open', 'Close', 'Low']].max(axis=1)) |
        (df.Low > df[['Open', 'Close', 'High']].min(axis=1)) |
        (df.Volume < 0) | (df.Low <= 0)).any():
        raise ValueError('Invalid OHLCV bar')
    return df


def complete_bars(raw, cfg=Config(), now=None):
    """Input timestamps are minute STARTS. Excludes unfinished/missing buckets."""
    df = normalize(raw)
    cutoff = utc(now) if now is not None else None
    chunks = []
    for day, g in df.groupby(df.index.tz_convert(ET).date):
        bounds = session_bounds(day)
        if bounds is None:
            continue
        opening, closing = bounds
        g = g[(g.index >= opening) & (g.index < closing)]
        if cutoff is not None:
            g = g[g.index + pd.Timedelta(minutes=1) <= cutoff - pd.Timedelta(seconds=2)]
        if g.empty:
            continue
        rule = f'{cfg.bar_minutes}min'
        b = g.resample(rule, origin=opening, label='left', closed='left').agg(
            {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
        n = g.Close.resample(rule, origin=opening).count()
        b = b[n == cfg.bar_minutes].dropna()
        b = b[b.index + pd.Timedelta(minutes=cfg.bar_minutes) <= closing]
        if b.empty:
            continue
        typical = (b.High + b.Low + b.Close) / 3
        b['VWAP'] = (typical * b.Volume).cumsum() / b.Volume.cumsum().replace(0, np.nan)
        chunks.append(b)
    return pd.concat(chunks) if chunks else pd.DataFrame(columns=COLS + ['VWAP'])


def signal_at(bars, i, cfg=Config()):
    if i < cfg.volume_bars:
        return None
    w = bars.iloc[i-cfg.volume_bars:i+1]
    if len(set(w.index.tz_convert(ET).date)) != 1:
        return None
    if not (w.index.to_series().diff().dropna() == pd.Timedelta(minutes=cfg.bar_minutes)).all():
        return None
    row = bars.iloc[i]
    previous = bars.iloc[i-cfg.pause_bars:i]
    avg = float(w.Volume.iloc[:-1].mean())
    span = float(row.High - row.Low)
    if avg <= 0 or span <= 0 or row.Volume < cfg.volume_multiple * avg:
        return None
    side = None
    if row.Close > previous.High.max() and row.Close > row.VWAP and (row.Close-row.Low)/span >= cfg.close_fraction:
        side = 'CALL'
    elif row.Close < previous.Low.min() and row.Close < row.VWAP and (row.High-row.Close)/span >= cfg.close_fraction:
        side = 'PUT'
    if not side:
        return None
    ts = bars.index[i] + pd.Timedelta(minutes=cfg.bar_minutes)
    return {'id': f'{cfg.symbol}-{cfg.bar_minutes}m-{ts.isoformat()}-{side}',
            'side':side, 'time':ts.isoformat(), 'close':float(row.Close),
            'high':float(row.High), 'low':float(row.Low),
            'stop':float(row.Low if side == 'CALL' else row.High),
            'volume_ratio':float(row.Volume/avg), 'vwap':float(row.VWAP)}


def entry_allowed(sig, price, now, cfg=Config()):
    age = (utc(now)-utc(sig['time'])).total_seconds()
    direction = 1 if sig['side']=='CALL' else -1
    drift = direction * (price-sig['close'])
    # Still beyond the confirmation close; no reversal entries or chasing.
    return (0 <= age <= cfg.max_signal_age_seconds and
            0 <= drift <= cfg.chase_fraction*(sig['high']-sig['low']) and
            direction*(price-sig['stop']) > 0)


def chart_exit(pos, bars, now, price, cfg=Config()):
    """Stops use observations after entry; no pre-entry candle highs/lows."""
    direction = 1 if pos['side']=='CALL' else -1
    after = bars[bars.index >= utc(pos['entry_time'])]
    if direction*(price-pos['stop']) <= 0:
        return 'CHART_STOP'
    if len(after) and ((after.Low <= pos['stop']).any() if direction==1 else (after.High >= pos['stop']).any()):
        return 'CHART_STOP'
    if len(after) >= 2:
        first_two = after.iloc[:2]
        progressed = first_two.High.max() > pos['high'] if direction==1 else first_two.Low.min() < pos['low']
        if not progressed:
            return 'MOMENTUM_TIMEOUT'
    if (utc(now)-utc(pos['entry_time'])).total_seconds() >= cfg.max_hold_minutes*60:
        return 'TIME_EXIT'
    bounds = session_bounds(utc(now).tz_convert(ET).date())
    if bounds and utc(now) >= bounds[1] - pd.Timedelta(minutes=10):
        return 'SESSION_EXIT'
    return None


def valid_quote(q, now, cfg=Config(), entry=True):
    try:
        bid, ask = float(q['bid']), float(q['ask'])
        age = (utc(now)-utc(q['timestamp'])).total_seconds()
        basic = (math.isfinite(bid) and math.isfinite(ask) and 0 <= bid <= ask and ask>0
                 and 0 <= age <= cfg.max_quote_age_seconds)
        if not basic:
            return False
        if not entry:
            return True
        delta = abs(float(q['delta']))
        return (bid > 0 and cfg.delta_min <= delta <= cfg.delta_max and
                (ask-bid)/((ask+bid)/2) <= cfg.max_spread and
                str(q['expiry']) == str(utc(now).tz_convert(ET).date()) and
                float(q.get('ask_size',0)) >= 1)
    except (ValueError, KeyError, TypeError):
        return False


def contract_count(ask, equity, fee=.65):
    if ask <= 0 or equity <= 0 or not math.isfinite(ask+equity+fee):
        return 0
    return max(0, math.floor(equity*.01/(100*ask+fee)))


def premium_exit(entry_ask, bid, cfg=Config()):
    if bid <= entry_ask*(1-cfg.stop_premium):
        return 'PREMIUM_STOP'
    if bid >= entry_ask*(1+cfg.target_premium):
        return 'PREMIUM_TARGET'
    return None


def download_bars(days=7):
    import yfinance as yf
    df = yf.download('IWM',period=f'{days}d',interval='1m',auto_adjust=False,
                     progress=False,threads=False,timeout=20)
    if df.empty:
        raise RuntimeError('No IWM minute data returned')
    return normalize(df)


def replay_underlying(raw, cfg=Config(), slippage=.01):
    """Directional feasibility study, NOT an options P&L backtest.
    Entries at NEXT bar open; stop gaps filled adversely; stop first; no option
    target translated to shares. One position at a time, two proxy losses/day.
    """
    bars = complete_bars(raw,cfg)
    trades=[]; candidates=0; skipped=0; i=0; losses={}
    while i < len(bars)-1:
        sig=signal_at(bars,i,cfg)
        if sig is None:
            i+=1; continue
        candidates+=1
        day=str(bars.index[i].tz_convert(ET).date())
        if losses.get(day,0)>=cfg.max_losses:
            skipped+=1; i+=1; continue
        j=i+1; t=bars.index[j]
        bounds=session_bounds(day)
        if t != utc(sig['time']) or t >= bounds[1]-pd.Timedelta(minutes=15):
            skipped+=1; i+=1; continue
        direction=1 if sig['side']=='CALL' else -1
        entry=float(bars.iloc[j].Open)+direction*slippage
        if not entry_allowed(sig,entry,t,cfg):
            skipped+=1; i+=1; continue
        stop=sig['stop']; risk=direction*(entry-stop)
        end=t+pd.Timedelta(minutes=cfg.max_hold_minutes)
        exit_price=None; reason=None; k=j
        for k in range(j,len(bars)):
            row=bars.iloc[k]; bt=bars.index[k]
            expected=t+pd.Timedelta(minutes=cfg.bar_minutes*(k-j))
            if bt!=expected or str(bt.tz_convert(ET).date())!=day:
                reason='DATA_GAP'; break
            if (row.Low<=stop if direction==1 else row.High>=stop):
                fill=min(float(row.Open),stop) if direction==1 else max(float(row.Open),stop)
                exit_price=fill-direction*slippage;reason='CHART_STOP';break
            if k-j+1>=2:
                first=bars.iloc[j:j+2]
                progressed=first.High.max()>sig['high'] if direction==1 else first.Low.min()<sig['low']
                if not progressed:
                    exit_price=float(row.Close)-direction*slippage;reason='MOMENTUM_TIMEOUT';break
            if bt+pd.Timedelta(minutes=cfg.bar_minutes)>=end:
                exit_price=float(row.Close)-direction*slippage;reason='TIME_EXIT';break
        if exit_price is None:
            # Censored trades stay visible, never silently counted as winners.
            trades.append({**sig,'entry_time':t.isoformat(),'entry':entry,'exit':None,
                           'exit_reason':reason or 'END_OF_DATA','pnl_per_share':None,'r_multiple':None})
        else:
            pnl=direction*(exit_price-entry)
            losses[day]=losses.get(day,0)+int(pnl<0)
            trades.append({**sig,'entry_time':t.isoformat(),'entry':entry,'exit':exit_price,
                           'exit_time':(bars.index[k]+pd.Timedelta(minutes=cfg.bar_minutes)).isoformat(),
                           'exit_reason':reason,'pnl_per_share':pnl,'r_multiple':pnl/risk})
        i=k+1
    finished=[x for x in trades if x['pnl_per_share'] is not None]
    pnl=[x['pnl_per_share'] for x in finished]
    curve=np.r_[0,np.cumsum(pnl)]
    report={'kind':'UNDERLYING_ONLY_DIAGNOSTIC_NOT_OPTIONS_BACKTEST',
            'config':asdict(cfg),'source':'Yahoo Finance 1-minute OHLCV',
            'first_bar':str(bars.index[0]) if len(bars) else None,
            'last_bar':str(bars.index[-1]) if len(bars) else None,
            'bars':len(bars),'sessions':len(set(bars.index.tz_convert(ET).date)) if len(bars) else 0,
            'raw_candidates':candidates,'skipped_candidates':skipped,
            'completed_trades':len(finished),'censored_trades':len(trades)-len(finished),
            'win_rate':sum(x>0 for x in pnl)/len(pnl) if pnl else None,
            'net_dollars_per_share':sum(pnl),
            'max_drawdown_dollars_per_share':float(np.max(np.maximum.accumulate(curve)-curve)),
            'slippage_per_side_dollars_per_share':slippage,
            'options_backtest_status':'BLOCKED: no historical option bid/ask/greeks supplied',
            'limitations':['No option returns, premium stops/targets, theta, IV, or option fees modeled.',
                          'Daily loss cap uses underlying proxy losses, not actual option losses.',
                          'Stop ordering assumed adverse when minute bars are ambiguous.',
                          'Observed sample is short and in-sample; no profitability claim.']}
    return report,trades


def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def publish(title,body,event_id):
    """No secrets or endpoint paths in logs. Server acceptance != phone receipt."""
    import requests
    topic=os.environ.get('NTFY_TOPIC','').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',topic):
        raise RuntimeError('NTFY_TOPIC missing or invalid')
    headers={'Title':title,'Tags':'chart_with_upwards_trend'}
    try:
        r=requests.post(f'https://ntfy.sh/{topic}',data=body.encode(),
                        headers=headers,timeout=15)
    except requests.RequestException:
        raise RuntimeError('ntfy network error; destination withheld') from None
    if not r.ok:
        raise RuntimeError(f'ntfy HTTP {r.status_code}; destination withheld')
    result=r.json()
    return {'event_id':event_id,'server_message_id':result.get('id'),
            'server_accepted':True,'phone_receipt_verified':False}


def new_state(day):
    return {'version':1,'day':day,'position':None,'last_signal':None,'losses':0,
            'outbox':[],'receipts':[],'health_alerted':False}


def queue(state,title,body,event_id):
    if any(x['event_id']==event_id for x in state['outbox'] + state['receipts']):
        return
    state['outbox'].append({'title':title,'body':body,'event_id':event_id})


def flush(state,path):
    save_json(path,state)
    for item in list(state['outbox']):
        # Do not deliver a delayed ENTRY after an outage. EXIT/health still useful.
        expiry=item.get('expires')
        if expiry and utc(datetime.now(ET)) > utc(expiry):
            state['outbox'].remove(item);save_json(path,state);continue
        receipt=publish(item['title'],item['body'],item['event_id'])
        state['receipts']=(state['receipts']+[receipt])[-20:]
        state['outbox'].remove(item);save_json(path,state)
        print(json.dumps(receipt),flush=True)


def paper_tick(raw,state,now,cfg=Config()):
    """Underlying paper alerts only: no invented contract prices or fills."""
    now=utc(now);day=str(now.tz_convert(ET).date());bounds=session_bounds(day)
    if state['day']!=day:
        old=state.get('position')
        state.update(new_state(day))
        if old:
            queue(state,'Fast Momentum: monitoring gap',
                  'A prior paper setup was not observed to exit. Check any actual broker position manually.',
                  day+'-overnight-gap')
    if not bounds or now < bounds[0] or now >= bounds[1]:
        if state['position']:
            queue(state,'Fast Momentum: session ended','Paper monitoring ended. Actual positions are not managed.',day+'-session-end')
            state['position']=None
        return 'MARKET_CLOSED'
    df=normalize(raw)
    today=df[(df.index>=bounds[0]) & (df.index<bounds[1]) & (df.index<=now)]
    bars=complete_bars(today,cfg,now)
    if bars.empty:
        return 'WARMUP'
    age=(now-(bars.index[-1]+pd.Timedelta(minutes=cfg.bar_minutes))).total_seconds()
    price=float(today.Close.iloc[-1])
    pos=state['position']
    if pos:
        # A timeout is meaningful even if prices are stale; never invent P&L.
        reason=chart_exit(pos,bars,now,price,cfg) if age<=cfg.max_signal_age_seconds else None
        if (now-utc(pos['entry_time'])).total_seconds()>=cfg.max_hold_minutes*60:
            reason=reason or ('TIME_EXIT' if age<=cfg.max_signal_age_seconds else 'TIME_EXIT_DATA_STALE')
        if reason:
            stale=age>cfg.max_signal_age_seconds
            direction=1 if pos['side']=='CALL' else -1
            # Unknown outcomes consume a loss slot conservatively.
            state['losses']+=int(stale or direction*(price-pos['entry_price'])<=0)
            queue(state,f"Fast Momentum PAPER EXIT {pos['side']}",
                  f"{reason}. Underlying observation: {'stale/unusable' if stale else format(price,'.2f')}. "
                  'No option quote or actual fill is tracked. Verify any real position in your broker.',pos['id']+'-exit')
            state['position']=None
            return reason
    if age>cfg.max_signal_age_seconds:
        if not state['health_alerted']:
            queue(state,'Fast Momentum: data delayed',
                  'New paper entries paused: recent completed bars are unavailable. This feed is not execution-grade.',day+'-stale')
            state['health_alerted']=True
        return 'STALE_DATA'
    state['health_alerted']=False
    if pos or state['losses']>=cfg.max_losses or now>=bounds[1]-pd.Timedelta(minutes=15):
        return 'NO_ENTRY'
    sig=signal_at(bars,len(bars)-1,cfg)
    if not sig or sig['id']==state['last_signal']:
        return 'NO_SIGNAL'
    state['last_signal']=sig['id']
    if not entry_allowed(sig,price,now,cfg):
        return 'CHASE_OR_REVERSAL'
    state['position']={**sig,'entry_time':now.isoformat(),'entry_price':price}
    queue(state, f"BUY {sig['side']} IWM", f"BUY {sig['side']} IWM @ {price:.2f}", sig['id'])
    state['outbox'][-1]['expires']=(utc(sig['time'])+pd.Timedelta(seconds=cfg.max_signal_age_seconds)).isoformat()
    return 'PAPER_SETUP'


def run_paper(args,cfg):
    path=Path(args.state)
    day=str(datetime.now(ET).date())
    state=json.loads(path.read_text()) if path.exists() else new_state(day)
    # Refuse incompatible state instead of overwriting the existing strategy.
    if state.get('version')!=1:
        raise ValueError('Unsupported fast-momentum state version')
    deadline=time.monotonic()+args.duration_minutes*60
    errors=0
    while True:
        now=pd.Timestamp.now(tz='UTC');bounds=session_bounds(now.tz_convert(ET).date())
        if not bounds or now>=bounds[1] or now<bounds[0]-pd.Timedelta(minutes=15):
            print('Outside scheduled market session',flush=True);break
        if now<bounds[0]:
            time.sleep(min(args.poll_seconds,30));continue
        try:
            raw=download_bars(days=1)
            now=pd.Timestamp.now(tz='UTC')
            status=paper_tick(raw,state,now,cfg)
            flush(state,path)
            latest = raw.index[-1] + pd.Timedelta(minutes=1)
            print(f'{now.isoformat()} {status} latest_bar_end={latest.isoformat()}',flush=True)
            errors=0
        except Exception as exc:
            # Never print HTTP URLs/credentials embedded in exception messages.
            errors+=1
            print(f'Cycle failed: {type(exc).__name__}; no new actionable signal',flush=True)
            if errors>=3:
                queue(state,'Fast Momentum: feed unavailable',
                      'Paper monitoring cannot obtain data or deliver alerts. Check any actual broker position manually.',
                      day+'-unavailable')
                try: flush(state,path)
                except Exception: save_json(path,state)
                raise RuntimeError('Repeated data/delivery failures; runner stopped') from None
        if args.once or time.monotonic()>=deadline:
            break
        time.sleep(max(1,min(args.poll_seconds,30)))
    save_json(path,state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['backtest','paper','test-notification','options-backtest'])
    p.add_argument('--bar-minutes',type=int,choices=[1,3],default=1)
    p.add_argument('--bars-csv')
    p.add_argument('--quotes-csv')
    p.add_argument('--output',default='research/fast_momentum')
    p.add_argument('--state',default='state/fast_momentum.json')
    p.add_argument('--duration-minutes',type=int,default=180)
    p.add_argument('--poll-seconds',type=int,default=15)
    p.add_argument('--once',action='store_true')
    p.add_argument('--equity',type=float)
    args=p.parse_args();cfg=Config(bar_minutes=args.bar_minutes)
    if args.mode=='test-notification':
        receipt=publish('TEST - NOT A TRADE', 'Notification connection test only. No buy signal.',
                        'fast-momentum-test-'+pd.Timestamp.now(tz='UTC').isoformat())
        print(json.dumps(receipt));return
    if args.mode=='paper':
        run_paper(args,cfg);return
    raw=normalize(pd.read_csv(args.bars_csv)) if args.bars_csv else download_bars()
    if args.mode=='options-backtest':
        if not args.quotes_csv or not args.equity:
            p.error('Options replay requires historical timestamped bid/ask/greeks CSV and --equity; no synthetic substitution.')
        from fast_momentum_options import replay_options
        report,trades=replay_options(raw,pd.read_csv(args.quotes_csv),args.equity,cfg)
    else:
        report,trades=replay_underlying(raw,cfg)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    save_json(out/'summary.json',report)
    pd.DataFrame(trades).to_csv(out/'trades.csv',index=False)
    (out/'report.md').write_text('# Fast Momentum research run\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
