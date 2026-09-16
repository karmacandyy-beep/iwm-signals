"""Historical 0DTE bid/ask replay. Actual quote input required; never prices options.
Input is an NBBO-style snapshot stream with timestamp,contract,side,expiry,bid,
ask,delta,ask_size. Prices USD; equity USD. Quotes must be timezone-aware.
Missing quote coverage censors results instead of inventing fills.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from fast_momentum import (Config, utc, normalize, complete_bars, signal_at,
                           valid_quote, contract_count, premium_exit, entry_allowed,
                           session_bounds)


def replay_options(raw,quotes,equity,cfg=Config(),fee=.65,slippage=.01):
    if equity<=0:
        raise ValueError('Positive USD equity required')
    required=['timestamp','contract','side','expiry','bid','ask','delta','ask_size']
    if not set(required)<=set(quotes):
        raise ValueError('Missing required historical option quote fields')
    q=quotes.copy()
    q['timestamp']=[utc(t) for t in q.timestamp]
    q['side']=q.side.str.upper()
    if q.duplicated(['timestamp','contract']).any():
        raise ValueError('Duplicate contract/quote timestamps')
    for col in ['bid','ask','delta','ask_size']:
        q[col]=pd.to_numeric(q[col],errors='raise')
    if not np.isfinite(q[['bid','ask','delta','ask_size']].to_numpy()).all():
        raise ValueError('Nonfinite historical quote fields')
    q=q.sort_values('timestamp')
    bars=complete_bars(raw,cfg);minute=normalize(raw)
    trades=[];losses={};busy_until=None;skipped=0;balance=equity
    for i in range(len(bars)):
        sig=signal_at(bars,i,cfg)
        if not sig: continue
        # One-second assumed decision/entry latency after a closed signal bar.
        t=utc(sig['time'])+pd.Timedelta(seconds=1);day=str(t.tz_convert('America/New_York').date())
        if busy_until is not None and t<=busy_until: continue
        if losses.get(day,0)>=cfg.max_losses: continue
        bounds=session_bounds(day)
        if t>=bounds[1]-pd.Timedelta(minutes=15):continue
        # Only a completed minute close available as of t is used; no future open.
        known=minute[minute.index+pd.Timedelta(minutes=1)<=t]
        if known.empty or not entry_allowed(sig,float(known.Close.iloc[-1]),t,cfg):continue
        candidates=q[(q.timestamp<=t)&(q.timestamp>=t-pd.Timedelta(seconds=cfg.max_quote_age_seconds))&
                     (q.side==sig['side'])&(q.expiry.astype(str)==day)]
        candidates=candidates.groupby('contract',sort=False).tail(1)
        candidates=[r for r in candidates.to_dict('records') if valid_quote(r,t,cfg)]
        candidates.sort(key=lambda r:(abs(abs(r['delta'])-.625),r['ask']-r['bid'],r['contract']))
        if not candidates:
            skipped+=1;continue
        chosen=candidates[0];entry=float(chosen['ask'])+slippage
        n=min(contract_count(entry,balance,fee),int(chosen['ask_size']))
        if n<1:skipped+=1;continue
        end=t+pd.Timedelta(minutes=cfg.max_hold_minutes)
        stream=q[(q.contract==chosen['contract'])&(q.timestamp>t)&(q.timestamp<=end+pd.Timedelta(seconds=5))]
        last=t;reason=None;fill=None;exit_time=None
        for r in stream.to_dict('records'):
            qt=r['timestamp']
            if (qt-last).total_seconds()>15:
                reason='QUOTE_GAP_CENSORED';break
            if not valid_quote(r,qt,cfg,entry=False):
                reason='INVALID_QUOTE_CENSORED';break
            last=qt
            seen=minute[(minute.index>=t.ceil('min'))&
                        (minute.index+pd.Timedelta(minutes=1)<=qt)]
            chart_stop=(seen.Low<=sig['stop']).any() if sig['side']=='CALL' else (seen.High>=sig['stop']).any()
            # Chart stops are detectable only on completed bars in this dataset.
            reason='CHART_STOP_AT_MINUTE_CLOSE' if chart_stop else premium_exit(entry,float(r['bid']),cfg)
            completed=bars[(bars.index>=t.ceil(f'{cfg.bar_minutes}min'))&
                           (bars.index+pd.Timedelta(minutes=cfg.bar_minutes)<=qt)]
            if reason is None and len(completed)>=2:
                first=completed.iloc[:2]
                progressed=first.High.max()>sig['high'] if sig['side']=='CALL' else first.Low.min()<sig['low']
                if not progressed:reason='MOMENTUM_TIMEOUT'
            if reason is None and qt>=end:reason='TIME_EXIT'
            if reason:
                fill=max(0,float(r['bid'])-slippage);exit_time=qt;break
        pnl=None if fill is None else (fill-entry)*100*n-2*fee*n
        censored=fill is None
        if pnl is not None:
            balance+=pnl
        else:
            # Conservative risk capacity after an unresolved exit: reserve full debit.
            balance-=entry*100*n+fee*n
        if pnl is None or pnl<=0:losses[day]=losses.get(day,0)+1
        busy_until=exit_time if exit_time is not None else end
        trades.append({**sig,'contract':chosen['contract'],'entry_time':t.isoformat(),
                       'entry_ask_with_slippage':entry,'contracts':n,'exit_bid_with_slippage':fill,
                       'exit_time':exit_time.isoformat() if exit_time is not None else None,
                       'reason':reason or 'END_OF_QUOTES_CENSORED','net_pnl_usd':pnl,
                       'censored':censored})
    done=[r for r in trades if not r['censored']]
    pnls=[r['net_pnl_usd'] for r in done];curve=np.r_[0,np.cumsum(pnls)]
    censored=sum(r['censored'] for r in trades)
    report={'kind':'HISTORICAL_OPTION_QUOTE_REPLAY','config':cfg.__dict__,
            'starting_equity_usd':equity,'completed_trades':len(done),'censored_trades':censored,
            'missing_contract_or_budget':skipped,'net_pnl_completed_only_usd':sum(pnls),
            'win_rate_completed_only':sum(p>0 for p in pnls)/len(pnls) if pnls else None,
            'max_drawdown_completed_only_usd':float(np.max(np.maximum.accumulate(curve)-curve)),
            'result_usable':censored==0 and len(done)>0,
            'fee_per_contract_per_side_usd':fee,'slippage_per_option_share_per_side':slippage,
            'limitations':['Historical data provenance must be verified by its supplier.',
                          'Uses bid/ask, not confirmed fills; displayed size is not guaranteed.',
                          'Chart stops seen at minute close; intraminute stop timing unavailable.',
                          'Censored trades excluded from reported P&L; reserve full debit for sizing.',
                          'No out-of-sample validation or proven profitability.']}
    return report,trades
