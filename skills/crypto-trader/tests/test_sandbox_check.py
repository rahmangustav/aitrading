"""Tests for the pure/mockable logic in sandbox_check.py.

sandbox_check.py's `main()` drives a REAL end-to-end flow against Binance
testnet (market buy, place native stop, cancel, sell back) -- that part can't
be unit tested without live testnet credentials and network access, and is
out of scope here. What CAN and SHOULD be covered without touching a real
exchange is the decision logic that guards that flow:

- `preflight()`: the safety gate that refuses to run outside demo/sandbox
  mode -- if this regresses, the script could silently run against a live
  account.
- `_test_amount()`: the sizing math that keeps the test order just above the
  exchange's minimum notional.
- `_stop_orders()`: the filter used to confirm a protective stop actually
  rests on the exchange (and later, that cancellation cleared it).
- `record()` / `summary()`: pass/fail bookkeeping that decides the script's
  exit code ("safe to review for live" vs "do NOT go live").

Checker.__init__ unconditionally constructs a real ExchangeManager/RiskManager,
so tests build instances via `Checker.__new__` and inject MagicMocks instead
of calling __init__ -- the same dependency-injection-free pattern __init__
uses, applied at the test boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sandbox_check import Checker  # noqa: E402


def _make_checker(exchange="binance", symbol="BTC/USDT"):
    checker = Checker.__new__(Checker)
    checker.exchange = exchange
    checker.symbol = symbol
    checker.results = []
    checker.exchange_mgr = MagicMock()
    checker.risk_mgr = MagicMock()
    return checker


class TestPreflight:
    def test_refuses_when_not_demo_mode(self):
        checker = _make_checker()
        checker.exchange_mgr.demo = False
        checker.exchange_mgr.available_exchanges = ["binance"]
        assert checker.preflight() is False

    def test_refuses_when_exchange_not_initialized(self):
        checker = _make_checker(exchange="kraken")
        checker.exchange_mgr.demo = True
        checker.exchange_mgr.available_exchanges = ["binance"]
        assert checker.preflight() is False

    def test_passes_when_demo_and_exchange_available(self):
        checker = _make_checker()
        checker.exchange_mgr.demo = True
        checker.exchange_mgr.available_exchanges = ["binance", "kraken"]
        assert checker.preflight() is True

    def test_not_demo_takes_priority_over_missing_exchange(self):
        # Even if the exchange also isn't initialized, the demo-mode gate
        # must be the one that fires -- it's the check that stands between
        # this script and a live account.
        checker = _make_checker(exchange="kraken")
        checker.exchange_mgr.demo = False
        checker.exchange_mgr.available_exchanges = []
        assert checker.preflight() is False


class TestRecord:
    def test_appends_result_tuple(self):
        checker = _make_checker()
        checker.record("market buy filled", True, "id=123")
        checker.record("cancel_stop_orders clears the stop", False, "remaining_stops=1")
        assert checker.results == [
            ("market buy filled", True, "id=123"),
            ("cancel_stop_orders clears the stop", False, "remaining_stops=1"),
        ]

    def test_detail_defaults_to_empty_string(self):
        checker = _make_checker()
        checker.record("precision guard rejects dust order", True)
        assert checker.results == [("precision guard rejects dust order", True, "")]


class TestStopOrders:
    def test_filters_to_only_stop_orders(self):
        checker = _make_checker()
        orders = [
            {"id": "1", "type": "market"},
            {"id": "2", "type": "stop_loss_limit"},
            {"id": "3", "type": "limit", "stopPrice": "50000"},
        ]
        checker.exchange_mgr.get_open_orders.return_value = orders
        checker.exchange_mgr._is_stop_order.side_effect = lambda o: (
            "stop" in str(o.get("type", "")).lower() or bool(o.get("stopPrice"))
        )
        result = checker._stop_orders()
        assert [o["id"] for o in result] == ["2", "3"]
        checker.exchange_mgr.get_open_orders.assert_called_once_with("binance", "BTC/USDT")

    def test_no_open_orders_returns_empty(self):
        checker = _make_checker()
        checker.exchange_mgr.get_open_orders.return_value = []
        assert checker._stop_orders() == []


class TestTestAmount:
    def test_sizes_thirty_percent_above_min_cost(self):
        checker = _make_checker()
        checker.exchange_mgr.get_markets.return_value = {
            "BTC/USDT": {"limits": {"cost": {"min": 10.0}}},
        }
        checker.exchange_mgr.get_min_order_amount.return_value = 0.0001
        price = 50000.0
        amount = checker._test_amount(price)
        # max((10*1.3)/50000, 0.0001*1.3) -> the cost-based leg dominates here.
        assert amount == pytest.approx((10.0 * 1.3) / price)

    def test_min_amount_leg_dominates_when_larger(self):
        checker = _make_checker()
        checker.exchange_mgr.get_markets.return_value = {
            "BTC/USDT": {"limits": {"cost": {"min": 1.0}}},
        }
        checker.exchange_mgr.get_min_order_amount.return_value = 1.0
        price = 100.0
        amount = checker._test_amount(price)
        # cost leg: (1*1.3)/100 = 0.013 ; amount leg: 1.0*1.3 = 1.3 -> amount leg wins.
        assert amount == pytest.approx(1.3)

    def test_falls_back_to_default_min_cost_when_market_missing_limits(self):
        checker = _make_checker()
        checker.exchange_mgr.get_markets.return_value = {"BTC/USDT": {}}
        checker.exchange_mgr.get_min_order_amount.return_value = None
        price = 25000.0
        amount = checker._test_amount(price)
        # min_cost falls back to 10.0, min_amt falls back to 0.0.
        assert amount == pytest.approx((10.0 * 1.3) / price)

    def test_falls_back_when_symbol_not_in_markets(self):
        checker = _make_checker()
        checker.exchange_mgr.get_markets.return_value = {}
        checker.exchange_mgr.get_min_order_amount.return_value = None
        price = 10000.0
        amount = checker._test_amount(price)
        assert amount == pytest.approx((10.0 * 1.3) / price)


class TestSummary:
    def test_all_passed_returns_zero(self, capsys):
        checker = _make_checker()
        checker.record("check a", True)
        checker.record("check b", True)
        code = checker.summary()
        assert code == 0
        assert "2/2 checks passed" in capsys.readouterr().out

    def test_any_failure_returns_one(self, capsys):
        checker = _make_checker()
        checker.record("check a", True)
        checker.record("check b", False, "boom")
        code = checker.summary()
        assert code == 1
        out = capsys.readouterr().out
        assert "1/2 checks passed" in out
        assert "do NOT go live" in out

    def test_no_checks_run_counts_as_all_passed(self):
        # Edge case: zero-of-zero -- summary() treats an empty run as a pass
        # (passed == total == 0). Documented here so a future guard against
        # "no checks executed" is a deliberate change, not an accidental one.
        checker = _make_checker()
        assert checker.summary() == 0
