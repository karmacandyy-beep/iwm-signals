import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from fast_momentum import *
from fast_momentum_options import replay_options


def fixture():
    idx=pd.date_range('2026-09-14 09:30',periods=25,freq='min',tz=ET)
    df=pd.DataFrame({'Open':100.,'High':100.1,'Low':99.9,'Close':100.,'Volume':1000.},index=idx)
    df.iloc[10]=[100,100.6,99.99,100.55,2000]
    for i in range(11,len(df)):
        df.iloc[i]=[100.55,100.9,100.5,100.7,1000]
    return df


class MomentumTests(unittest.TestCase):
    def test_call_signal_and_no_peeking(self):
        df=fixture();b=complete_bars(df)
        s=signal_at(b,10);self.assertEqual(s['side'],'CALL')
        df.iloc[11:]=df.iloc[11:]*1.2
        self.assertEqual(signal_at(complete_bars(df),10),s)

    def test_put(self):
        d=fixture();d[['Open','High','Low','Close']]=200-d[['Open','Low','High','Close']].to_numpy()
        self.assertEqual(signal_at(complete_bars(d),10)['side'],'PUT')

    def test_volume_threshold(self):
        d=fixture();d.iloc[10,d.columns.get_loc('Volume')]=1400
        self.assertIsNone(signal_at(complete_bars(d),10))

    def test_unfinished_bar_is_excluded(self):
        b=complete_bars(fixture(),now=pd.Timestamp('2026-09-14 09:40:59',tz=ET))
        self.assertEqual(len(b),10)

    def test_missing_minute_suppresses_signal(self):
        b=complete_bars(fixture().drop(fixture().index[8]))
        self.assertIsNone(signal_at(b,9))

    def test_three_minute_requires_all_minutes(self):
        d=fixture();b=complete_bars(d,Config(bar_minutes=3))
        self.assertEqual(len(b),8)
        d=d.drop(d.index[4]);self.assertEqual(len(complete_bars(d,Config(bar_minutes=3))),7)

    def test_session_reset_and_holiday(self):
        d=fixture();other=d.copy();other.index=other.index-pd.Timedelta(days=3)
        b=complete_bars(pd.concat([other,d]));self.assertIsNone(signal_at(b,25))
        holiday=d.copy();holiday.index=pd.date_range('2026-09-07 09:30',periods=len(d),freq='min',tz=ET)
        self.assertEqual(len(complete_bars(holiday)),0)

    def test_early_close(self):
        d=fixture();d.index=pd.date_range('2026-11-27 12:50',periods=len(d),freq='min',tz=ET)
        self.assertEqual(len(complete_bars(d)),10)

    def test_chase_and_stale(self):
        s=signal_at(complete_bars(fixture()),10);t=utc(s['time'])
        self.assertTrue(entry_allowed(s,s['close'],t))
        self.assertFalse(entry_allowed(s,101,t))
        self.assertFalse(entry_allowed(s,100.4,t))
        self.assertFalse(entry_allowed(s,s['close'],t+pd.Timedelta(seconds=46)))

    def test_dedup_and_isolated_state(self):
        d=fixture().iloc[:12].copy();d.iloc[11]=[100.55,100.56,100.54,100.55,100]
        st=new_state('2026-09-14');now=pd.Timestamp('2026-09-14 09:41:05',tz=ET)
        self.assertEqual(paper_tick(d,st,now),'PAPER_SETUP')
        self.assertEqual(len(st['outbox']),1)
        paper_tick(d,st,now+pd.Timedelta(seconds=10));self.assertEqual(len(st['outbox']),1)

    def test_stale_data_never_enters(self):
        st=new_state('2026-09-14');paper_tick(fixture().iloc[:11],st,pd.Timestamp('2026-09-14 10:00',tz=ET))
        self.assertIsNone(st['position']);self.assertEqual(st['outbox'][0]['title'],'Fast Momentum: data delayed')

    def test_failure_to_deliver_keeps_outbox(self):
        st=new_state('2026-09-14');queue(st,'test','body','x')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.json'
            with patch('fast_momentum.publish',side_effect=RuntimeError('network')):
                with self.assertRaises(RuntimeError):flush(st,p)
            self.assertEqual(len(json.loads(p.read_text())['outbox']),1)

    def test_quotes_sizing_and_premium(self):
        now=pd.Timestamp('2026-09-14 10:00',tz=ET)
        q={'timestamp':now,'bid':1.,'ask':1.02,'delta':.6,'expiry':'2026-09-14','ask_size':3}
        self.assertTrue(valid_quote(q,now))
        self.assertFalse(valid_quote({**q,'ask':1.5},now))
        self.assertFalse(valid_quote({**q,'expiry':'2026-09-15'},now))
        self.assertFalse(valid_quote(q,now+pd.Timedelta(seconds=6)))
        self.assertEqual(contract_count(1.02,1000),0)
        self.assertEqual(contract_count(1.02,20000),1)
        self.assertEqual(premium_exit(1.,.79),'PREMIUM_STOP')
        self.assertEqual(premium_exit(1.,1.31),'PREMIUM_TARGET')

    def test_replay_stop_gap_is_not_filled_at_better_stop(self):
        d=fixture();d.iloc[12]=[99,99.1,98.9,99,1000]
        report,trades=replay_underlying(d)
        self.assertEqual(trades[0]['exit_reason'],'CHART_STOP')
        self.assertLess(trades[0]['exit'],99)
        self.assertIn('NOT_OPTIONS',report['kind'])

    def test_option_replay_real_quote_target_and_fee(self):
        t=pd.Timestamp('2026-09-14 09:41:00',tz=ET)
        q=pd.DataFrame([{'timestamp':t+pd.Timedelta(seconds=k),'contract':'IWM260914C00100000',
                         'side':'CALL','expiry':'2026-09-14','bid':1. if k<4 else 1.4,
                         'ask':1.02 if k<4 else 1.42,'delta':.6,'ask_size':10} for k in range(7)])
        report,trades=replay_options(fixture(),q,20000)
        self.assertEqual(trades[0]['reason'],'PREMIUM_TARGET')
        self.assertAlmostEqual(trades[0]['net_pnl_usd'],(1.39-1.03)*100-1.3)
        self.assertTrue(report['result_usable'])

    def test_options_missing_quotes_are_censored(self):
        t=pd.Timestamp('2026-09-14 09:41:00',tz=ET)
        q=pd.DataFrame([{'timestamp':t,'contract':'IWM260914C00100000','side':'CALL',
                         'expiry':'2026-09-14','bid':1.,'ask':1.02,'delta':.6,'ask_size':10}])
        report,trades=replay_options(fixture(),q,20000)
        self.assertEqual(report['censored_trades'],1);self.assertFalse(report['result_usable'])

if __name__=='__main__':unittest.main()
