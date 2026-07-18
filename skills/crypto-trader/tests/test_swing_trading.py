"""Tests for SwingTradingStrategy.

Focus: the sell-amount bug where exits used to recompute the order size
from the *current* market price instead of the amount actually bought
(the same class of bug already fixed in trend_following.py), and the
missing `_persist_attrs` that dropped open-position state on restart.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.swing_trading import SwingTradingStrategy  # noqa: E402


def _candles(prices, spread=0.5, vol=1000.0):
    out = []
    prev = prices[0]
    for i, c in enumerate(prices):
        o = prev
        h = max(o, c) + spread
        l = min(o, c) - spread
        out.append([i * 3600000, o, h, l, c, vol])
        prev = c
    return out


def _make_strategy(**params):
    exchange_mgr = MagicMock()
    exchange_mgr.available_exchanges = ["binance"]
    risk_mgr = MagicMock()
    risk_mgr.check_stop_loss.return_value = False
    risk_mgr.check_trailing_stop.return_value = False
    risk_mgr.check_take_profit.return_value = False
    strat = SwingTradingStrategy(
        strategy_id="swing_test",
        params={"exchange": "binance", **params},
        exchange_manager=exchange_mgr,
        risk_manager=risk_mgr,
    )
    strat.on_start()
    return strat


class TestPersistAttrs:
    def test_position_state_is_persisted(self):
        strat = _make_strategy()
        assert set(strat._persist_attrs) == {
            "position", "position_amount", "entry_price",
            "entry_time", "highest_since_entry",
        }

    def test_get_state_round_trips_through_restore(self):
        strat = _make_strategy()
        strat.position = "long"
        strat.position_amount = 0.0123
        strat.entry_price = 100.0
        strat.entry_time = 111.0
        strat.highest_since_entry = 105.0

        state = strat.get_state()

        fresh = _make_strategy()
        fresh.restore_state(state)

        assert fresh.position == "long"
        assert fresh.position_amount == pytest.approx(0.0123)
        assert fresh.entry_price == pytest.approx(100.0)
        assert fresh.entry_time == pytest.approx(111.0)
        assert fresh.highest_since_entry == pytest.approx(105.0)


class TestOnOrderFilled:
    def test_buy_fill_sets_position_amount_from_filled(self):
        strat = _make_strategy()
        strat.on_order_filled({"side": "buy", "filled": 0.5, "average": 100.0})

        assert strat.position == "long"
        assert strat.position_amount == pytest.approx(0.5)
        assert strat.entry_price == pytest.approx(100.0)
        assert strat.highest_since_entry == pytest.approx(100.0)

    def test_buy_fill_accumulates_across_partial_fills(self):
        strat = _make_strategy()
        strat.on_order_filled({"side": "buy", "filled": 0.3, "average": 100.0})
        strat.on_order_filled({"side": "buy", "filled": 0.2, "average": 100.0})

        assert strat.position_amount == pytest.approx(0.5)

    def test_buy_fill_falls_back_to_price_field(self):
        strat = _make_strategy()
        strat.on_order_filled({"side": "buy", "amount": 0.4, "price": 200.0})

        assert strat.position_amount == pytest.approx(0.4)
        assert strat.entry_price == pytest.approx(200.0)

    def test_buy_fill_derives_price_from_cost_and_filled(self):
        strat = _make_strategy()
        strat.on_order_filled({"side": "buy", "filled": 2.0, "cost": 400.0})

        assert strat.entry_price == pytest.approx(200.0)

    def test_sell_fill_resets_position_amount(self):
        strat = _make_strategy()
        strat.on_order_filled({"side": "buy", "filled": 0.5, "average": 100.0})
        strat.on_order_filled({"side": "sell", "filled": 0.5, "average": 110.0})

        assert strat.position is None
        assert strat.position_amount == 0.0
        assert strat.entry_price == 0.0

    def test_sell_fill_records_pnl(self):
        strat = _make_strategy()
        strat.on_order_filled({"side": "buy", "filled": 1.0, "average": 100.0})
        strat.on_order_filled({"side": "sell", "filled": 1.0, "average": 110.0})

        assert strat.stats["total_pnl"] == pytest.approx(10.0)


class TestSellAmountUsesActualPosition:
    """Regression tests for the current-price sell-amount bug."""

    def test_sell_signal_uses_position_amount_not_current_price(self):
        strat = _make_strategy(order_amount_usdt=1000.0, bb_period=5, macd_slow=5, macd_fast=2, macd_signal=2)
        # Bought 10 units at 100 (position_amount tracked from the fill,
        # independent of order_amount_usdt / price).
        strat.on_order_filled({"side": "buy", "filled": 10.0, "average": 100.0})
        strat.risk_manager.check_stop_loss.return_value = True

        current_price = 250.0  # price has drifted a lot since entry
        signal = strat._sell_signal(strat.position_amount, current_price, "Stop-loss triggered")

        # Old buggy behaviour would have been order_amount_usdt / current_price
        # = 1000 / 250 = 4.0, i.e. selling less than half of what was bought.
        assert signal["amount"] == pytest.approx(10.0)
        wrong_amount = round(1000.0 / current_price, 8)
        assert signal["amount"] != wrong_amount

    def test_evaluate_sell_signal_amount_matches_tracked_position(self):
        strat = _make_strategy(order_amount_usdt=1000.0, bb_period=5, macd_slow=5, macd_fast=2, macd_signal=2, max_hold_days=9999)
        strat.on_order_filled({"side": "buy", "filled": 3.0, "average": 100.0})
        strat.risk_manager.check_stop_loss.return_value = True

        # Rising price series so current close (400) is far from entry (100),
        # which is exactly the scenario the old code got wrong.
        prices = [100 + i * 30 for i in range(30)]
        strat.exchange_manager.get_ohlcv.return_value = _candles(prices)

        signals = strat.evaluate()

        sell_signals = [s for s in signals if s["side"] == "sell"]
        assert len(sell_signals) == 1
        assert sell_signals[0]["amount"] == pytest.approx(3.0)

    def test_evaluate_falls_back_to_fresh_amount_when_no_tracked_position(self):
        # If a strategy instance somehow has position == "long" but no
        # position_amount (e.g. restored from old state before this fix),
        # it should fall back to the order_amount_usdt / price calculation
        # rather than sell nothing.
        strat = _make_strategy(order_amount_usdt=500.0, bb_period=5, macd_slow=5, macd_fast=2, macd_signal=2, max_hold_days=9999)
        strat.position = "long"
        strat.entry_price = 100.0
        strat.highest_since_entry = 100.0
        strat.risk_manager.check_stop_loss.return_value = True

        prices = [100 + i * 5 for i in range(30)]
        strat.exchange_manager.get_ohlcv.return_value = _candles(prices)

        signals = strat.evaluate()
        sell_signals = [s for s in signals if s["side"] == "sell"]
        assert len(sell_signals) == 1
        expected = round(500.0 / prices[-1], 8)
        assert sell_signals[0]["amount"] == pytest.approx(expected)
