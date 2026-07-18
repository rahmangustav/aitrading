"""Tests for ScalpingStrategy.

Focus: the sell-amount bug where exits recomputed the order size from
order_amount_usdt / entry_price instead of the amount actually bought
(the same class of bug already fixed in trend_following.py and
swing_trading.py) — and the missing position_amount / _persist_attrs
that dropped that state on restart.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.scalping import ScalpingStrategy  # noqa: E402


def _make_strategy(**params):
    exchange_mgr = MagicMock()
    exchange_mgr.available_exchanges = ["binance"]
    risk_mgr = MagicMock()
    strat = ScalpingStrategy(
        strategy_id="scalp_test",
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
            "position", "position_amount", "entry_price", "entry_time",
        }

    def test_get_state_round_trips_through_restore(self):
        strat = _make_strategy()
        strat.position = "long"
        strat.position_amount = 0.0123
        strat.entry_price = 100.0
        strat.entry_time = 111.0

        state = strat.get_state()

        fresh = _make_strategy()
        fresh.restore_state(state)

        assert fresh.position == "long"
        assert fresh.position_amount == pytest.approx(0.0123)
        assert fresh.entry_price == pytest.approx(100.0)
        assert fresh.entry_time == pytest.approx(111.0)


class TestOnOrderFilled:
    def test_buy_fill_sets_position_amount_from_filled(self):
        strat = _make_strategy()
        strat.on_order_filled({"side": "buy", "filled": 0.5, "average": 100.0})

        assert strat.position == "long"
        assert strat.position_amount == pytest.approx(0.5)
        assert strat.entry_price == pytest.approx(100.0)

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
    """Regression tests for the current-entry-price sell-amount bug."""

    def _evaluate_with_price(self, strat, current_price, spread_pct=1.0):
        strat.exchange_manager.get_orderbook.return_value = {
            "spread_pct": spread_pct, "bids": [[current_price, 1.0]],
        }
        strat.exchange_manager.get_ticker.return_value = {"last": current_price}
        return strat.evaluate()

    def test_sell_signal_uses_position_amount_not_recomputed_from_entry_price(self):
        strat = _make_strategy(order_amount_usdt=1000.0, profit_target_pct=0.1)
        # Order was sized for 1000 USDT at 100, but the actual fill was a
        # partial fill of only 3 units (e.g. partial limit fill).
        strat.on_order_filled({"side": "buy", "filled": 3.0, "average": 100.0})

        signals = self._evaluate_with_price(strat, current_price=105.0)

        sell_signals = [s for s in signals if s["side"] == "sell"]
        assert len(sell_signals) == 1
        # Old buggy behaviour would have sold order_amount_usdt / entry_price
        # = 1000 / 100 = 10.0, i.e. selling more than what was ever bought.
        assert sell_signals[0]["amount"] == pytest.approx(3.0)
        wrong_amount = round(1000.0 / 100.0, 8)
        assert sell_signals[0]["amount"] != wrong_amount

    def test_stop_loss_exit_uses_tracked_position_amount(self):
        strat = _make_strategy(order_amount_usdt=1000.0, profit_target_pct=0.1)
        strat.on_order_filled({"side": "buy", "filled": 2.5, "average": 100.0})

        signals = self._evaluate_with_price(strat, current_price=90.0)

        sell_signals = [s for s in signals if s["side"] == "sell"]
        assert len(sell_signals) == 1
        assert sell_signals[0]["amount"] == pytest.approx(2.5)

    def test_timeout_exit_uses_tracked_position_amount(self):
        strat = _make_strategy(order_amount_usdt=1000.0, profit_target_pct=50.0, max_hold_seconds=1)
        strat.on_order_filled({"side": "buy", "filled": 1.7, "average": 100.0})
        strat.entry_time = time.time() - 10

        signals = self._evaluate_with_price(strat, current_price=100.5)

        sell_signals = [s for s in signals if s["side"] == "sell"]
        assert len(sell_signals) == 1
        assert sell_signals[0]["amount"] == pytest.approx(1.7)

    def test_falls_back_to_order_amount_when_no_tracked_position(self):
        # If a strategy instance somehow has position == "long" but no
        # position_amount (e.g. restored from old state before this fix),
        # it should fall back to order_amount_usdt / entry_price rather
        # than sell nothing.
        strat = _make_strategy(order_amount_usdt=500.0, profit_target_pct=0.1)
        strat.position = "long"
        strat.entry_price = 100.0
        strat.entry_time = time.time()

        signals = self._evaluate_with_price(strat, current_price=105.0)

        sell_signals = [s for s in signals if s["side"] == "sell"]
        assert len(sell_signals) == 1
        expected = round(500.0 / 100.0, 8)
        assert sell_signals[0]["amount"] == pytest.approx(expected)


class TestEntry:
    def test_entry_signal_when_spread_within_threshold(self):
        strat = _make_strategy(order_amount_usdt=100.0, spread_threshold_pct=0.5)
        strat.exchange_manager.get_orderbook.return_value = {
            "spread_pct": 0.2, "bids": [[50.0, 2.0]],
        }
        strat.exchange_manager.get_ticker.return_value = {"last": 50.0}

        signals = strat.evaluate()

        assert len(signals) == 1
        assert signals[0]["side"] == "buy"
        assert signals[0]["amount"] == pytest.approx(2.0)

    def test_no_entry_when_spread_too_wide(self):
        strat = _make_strategy(order_amount_usdt=100.0, spread_threshold_pct=0.1)
        strat.exchange_manager.get_orderbook.return_value = {
            "spread_pct": 0.5, "bids": [[50.0, 2.0]],
        }
        strat.exchange_manager.get_ticker.return_value = {"last": 50.0}

        signals = strat.evaluate()

        assert signals == []

    def test_inactive_strategy_returns_no_signals(self):
        strat = _make_strategy()
        strat.active = False

        assert strat.evaluate() == []
