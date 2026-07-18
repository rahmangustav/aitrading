"""Tests for the Grid Trading strategy."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.grid_trading import GridTradingStrategy  # noqa: E402


def _make_strategy(**params):
    defaults = {
        "symbol": "BTC/USDT",
        "price_range": [90000, 110000],
        "num_grids": 10,
        "order_amount_usdt": 10.0,
    }
    defaults.update(params)
    exchange_manager = MagicMock()
    exchange_manager.available_exchanges = ["binance"]
    risk_manager = MagicMock()
    return GridTradingStrategy("sid-1", defaults, exchange_manager, risk_manager)


class TestCalculateGridLevels:
    def test_evenly_spaced_levels(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        assert len(s.grid_levels) == 11
        assert s.grid_levels[0] == 90000
        assert s.grid_levels[-1] == 110000
        assert s.grid_spacing == pytest.approx(2000.0)

    def test_invalid_range_deactivates(self):
        s = _make_strategy(price_range=[110000, 90000], num_grids=10)
        s._calculate_grid_levels()
        assert s.active is False
        assert s.grid_levels == []


class TestOnStart:
    def test_picks_first_available_exchange_when_unset(self):
        s = _make_strategy(exchange="")
        s.on_start()
        assert s.exchange == "binance"
        assert s._initialized is True
        assert s.active is True

    def test_no_exchanges_available_deactivates(self):
        s = _make_strategy(exchange="")
        s.exchange_manager.available_exchanges = []
        s.on_start()
        assert s.active is False
        assert s._initialized is False


class TestGenerateGridSignals:
    def test_buy_below_and_sell_above_current_price(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10, order_amount_usdt=100.0)
        s._calculate_grid_levels()
        current_price = 100000.0
        signals = s._generate_grid_signals(current_price)

        below_levels = [lv for lv in s.grid_levels if lv < current_price]
        above_levels = [lv for lv in s.grid_levels if lv > current_price]

        buys = [sig for sig in signals if sig["side"] == "buy"]
        sells = [sig for sig in signals if sig["side"] == "sell"]
        assert len(buys) == len(below_levels)
        assert len(sells) == len(above_levels)
        assert all(sig["price"] < current_price for sig in buys)
        assert all(sig["price"] > current_price for sig in sells)

    def test_level_equal_to_current_price_gets_no_signal(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10, order_amount_usdt=100.0)
        s._calculate_grid_levels()
        current_price = 100000.0
        assert current_price in s.grid_levels
        signals = s._generate_grid_signals(current_price)
        assert all(sig["price"] != current_price for sig in signals)

    def test_skips_levels_with_active_orders(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10, order_amount_usdt=100.0)
        s._calculate_grid_levels()
        current_price = 100000.0
        occupied_level = next(lv for lv in s.grid_levels if lv < current_price)
        s.active_orders[f"{occupied_level:.2f}"] = {"order_id": "x", "side": "buy"}

        signals = s._generate_grid_signals(current_price)
        assert all(sig["price"] != occupied_level for sig in signals)

    def test_non_positive_price_returns_no_signals(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        assert s._generate_grid_signals(0) == []
        assert s._generate_grid_signals(-5) == []


class TestEvaluate:
    def test_returns_empty_when_not_initialized(self):
        s = _make_strategy()
        assert s.evaluate() == []

    def test_returns_empty_when_inactive(self):
        s = _make_strategy()
        s.on_start()
        s.active = False
        assert s.evaluate() == []

    def test_returns_empty_on_ticker_exception(self):
        s = _make_strategy()
        s.on_start()
        s.exchange_manager.get_ticker.side_effect = RuntimeError("network down")
        assert s.evaluate() == []

    def test_returns_empty_on_zero_price(self):
        s = _make_strategy()
        s.on_start()
        s.exchange_manager.get_ticker.return_value = {"last": 0}
        assert s.evaluate() == []

    def test_in_range_price_generates_grid_signals(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s.on_start()
        s.exchange_manager.get_ticker.return_value = {"last": 100000.0}
        signals = s.evaluate()
        assert len(signals) > 0

    def test_breakout_never_returns_signals_but_shifts_range(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10, rebalance_on_breakout=True)
        s.on_start()
        s.exchange_manager.get_ticker.return_value = {"last": 115000.0}
        signals = s.evaluate()
        assert signals == []
        assert s.price_range == [105000.0, 125000.0]

    def test_breakout_without_rebalance_leaves_range_untouched(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10, rebalance_on_breakout=False)
        s.on_start()
        original_range = list(s.price_range)
        s.exchange_manager.get_ticker.return_value = {"last": 115000.0}
        signals = s.evaluate()
        assert signals == []
        assert s.price_range == original_range


class TestHandleBreakout:
    def test_shifts_range_up_and_recenters_on_breakout_above(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        s.active_orders["100000.00"] = {"order_id": "x", "side": "buy"}
        s._handle_breakout(120000.0)
        assert s.price_range == [110000.0, 130000.0]
        assert s.active_orders == {}
        assert s.grid_levels[0] == 110000.0
        assert s.grid_levels[-1] == 130000.0

    def test_shifts_range_down_and_recenters_on_breakout_below(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        s.active_orders["100000.00"] = {"order_id": "x", "side": "sell"}
        s._handle_breakout(80000.0)
        assert s.price_range == [70000.0, 90000.0]
        assert s.active_orders == {}


class TestOnOrderPlacedAndFilled:
    def test_on_order_placed_tracks_by_grid_level(self):
        s = _make_strategy()
        signal = {"grid_level": 98000.0, "side": "buy"}
        order = {"id": "abc123"}
        s.on_order_placed(signal, order)
        assert s.active_orders["98000.00"] == {"order_id": "abc123", "side": "buy"}

    def test_on_order_placed_falls_back_to_price(self):
        s = _make_strategy()
        signal = {"side": "sell", "price": 102000.0}
        order = {"id": "def456"}
        s.on_order_placed(signal, order)
        assert s.active_orders["102000.00"] == {"order_id": "def456", "side": "sell"}

    def test_on_order_placed_ignores_missing_level(self):
        s = _make_strategy()
        s.on_order_placed({"side": "buy"}, {"id": "x"})
        assert s.active_orders == {}

    def test_buy_fill_creates_counter_sell_one_level_up(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        s.active_orders["98000.00"] = {"order_id": "x", "side": "buy"}
        follow_up = s.on_order_filled({"side": "buy", "price": 98000.0, "amount": 0.001})
        assert follow_up["side"] == "sell"
        assert follow_up["price"] == pytest.approx(98000.0 + s.grid_spacing)
        assert "98000.00" not in s.active_orders
        assert s.stats["trades_executed"] == 1

    def test_sell_fill_creates_counter_buy_one_level_down(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        s.active_orders["102000.00"] = {"order_id": "x", "side": "sell"}
        follow_up = s.on_order_filled({"side": "sell", "price": 102000.0, "amount": 0.001})
        assert follow_up["side"] == "buy"
        assert follow_up["price"] == pytest.approx(102000.0 - s.grid_spacing)
        assert "102000.00" not in s.active_orders

    def test_prefers_average_price_over_price(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        follow_up = s.on_order_filled({"side": "buy", "price": 98000.0, "average": 98050.0})
        assert follow_up["price"] == pytest.approx(98050.0 + s.grid_spacing)

    def test_missing_price_returns_none_but_still_counts_trade(self):
        s = _make_strategy()
        follow_up = s.on_order_filled({"side": "buy", "price": 0})
        assert follow_up is None
        assert s.stats["trades_executed"] == 1

    def test_unknown_side_returns_none(self):
        s = _make_strategy(price_range=[90000, 110000], num_grids=10)
        s._calculate_grid_levels()
        follow_up = s.on_order_filled({"side": "hold", "price": 98000.0})
        assert follow_up is None
