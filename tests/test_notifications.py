import ast
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from datetime import datetime
from types import SimpleNamespace
import requests
from notifications import publish

class DeliveryTests(unittest.TestCase):
    def response(self, status=200):
        r = requests.Response()
        r.status_code = status
        return r

    @patch.dict(os.environ, {"NTFY_TOPIC": "test-topic"})
    @patch("notifications.time.sleep")
    @patch("notifications.requests.post")
    def test_network_failure_recovers(self, post, sleep):
        post.side_effect = [requests.ConnectionError(), self.response()]
        self.assertTrue(publish("TEST", "test"))
        self.assertEqual(post.call_count, 2)

    @patch.dict(os.environ, {"NTFY_TOPIC": "test-topic"})
    @patch("notifications.time.sleep")
    @patch("notifications.requests.post")
    def test_exhaustion_is_not_success(self, post, sleep):
        post.side_effect = requests.ConnectionError()
        self.assertFalse(publish("TEST", "test"))
        self.assertEqual(post.call_count, 3)

    @patch.dict(os.environ, {"NTFY_TOPIC": "test-topic"})
    @patch("notifications.time.sleep")
    @patch("notifications.requests.post")
    def test_http_errors(self, post, sleep):
        post.side_effect = [self.response(503), self.response()]
        self.assertTrue(publish("TEST", "test"))
        post.reset_mock(side_effect=True)
        post.return_value = self.response(403)
        self.assertFalse(publish("TEST", "test"))
        self.assertEqual(post.call_count, 1)

    def test_failed_delivery_does_not_open_position(self):
        source = (Path(__file__).resolve().parents[1] / "accumulation_distribution_0dte.py").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "check_for_entry")
        send = Mock(return_value=False)
        ctx = dict(is_market_hours=lambda _: True, is_past_time_stop=lambda _: False,
                   find_signal=lambda *_: SimpleNamespace(underlying_price=286, side="long", reason="test"),
                   estimate_iv=lambda _: .2, select_strike=lambda *_: 286,
                   bs_price=lambda *_: 1, bs_delta=lambda *_: .5,
                   RISK_FREE_RATE=.04, CONTRACTS_PER_TRADE=1, send_alert=send)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "entry", "exec"), ctx)
        state = {"open_position": None}
        now = datetime(2026, 9, 16, 10)
        ctx["check_for_entry"](state, None, None, now)
        self.assertIsNone(state["open_position"])
        send.return_value = True
        ctx["check_for_entry"](state, None, None, now)
        self.assertEqual(state["open_position"]["option_type"], "call")

if __name__ == "__main__":
    unittest.main()
