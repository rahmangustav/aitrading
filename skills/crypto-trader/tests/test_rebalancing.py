"""Tests for RebalancingStrategy.

This strategy had zero test coverage before this file — unlike
grid_trading/swing_trading/trend_following/scalping/dca, which already
went through this same audit pattern in prior PRs. Focus areas: weight
calculation from live balances, drift-threshold gating, and the
buy/sell signal math derived from target vs. current allocation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.rebalancing import RebalancingStrategy  # noqa: E402


def _make_strategy(**params):
    defaults = {
        "target_allocation": {"BTC/USDT": 60.0, "ETH/USDT": 40.0},
        "rebalance_threshold_pct": 5.0,
        "interval": "daily",
    }
    defaults.update(params)
    exchange_mgr = MagicMock()
    exchange_mgr.available_exchanges = ["binance"]
    risk_mgr = MagicMock()
    strat = RebalancingStrategy(
        strategy_id="rebal_test",
        params=defaults,
        exchange_manager=exchange_mgr,
        risk_manager=risk_mgr,
    )
    return strat


def _balances(**assets):
    """Build a get_balance()-shaped dict from {ASSET: total_qty}."""
    return {asset: {"total": qty} for asset, qty in assets.items()}


def _ticker_side_effect(prices):
    """Build a get_ticker(exchange, symbol) side_effect from {symbol: price}."""
    def _get(exchange, symbol):
        price = prices[symbol]
        return {"last": price, "bid": price, "ask": price}
    return _get


class TestOnStart:
    def test_picks_first_available_exchange_when_unset(self):
        strat = _make_strategy(exchange="")
        strat.on_start()
        assert strat.exchange == "binance"
        assert strat.active is True

    def test_deactivates_when_no_exchange_available(self):
        strat = _make_strategy(exchange="")
        strat.exchange_manager.available_exchanges = []
        strat.on_start()
        assert strat.active is False

    def test_keeps_explicit_exchange(self):
        strat = _make_strategy(exchange="kraken")
        strat.on_start()
        assert strat.exchange == "kraken"

    def test_does_not_crash_when_allocation_does_not_sum_to_100(self):
        strat = _make_strategy(target_allocation={"BTC/USDT": 60.0, "ETH/USDT": 60.0})
        strat.on_start()
        assert strat.active is True


class TestEvaluateThrottle:
    def test_skips_when_interval_has_not_elapsed(self):
        strat = _make_strategy(interval="daily")
        strat.on_start()
        strat.last_rebalance_time = time.time()
        assert strat.evaluate() == []
        strat.exchange_manager.get_balance.assert_not_called()

    def test_returns_empty_and_logs_on_weight_calc_failure(self):
        strat = _make_strategy()
        strat.on_start()
        strat.last_rebalance_time = 0.0
        strat.exchange_manager.get_balance.side_effect = RuntimeError("boom")
        assert strat.evaluate() == []

    def test_updates_last_rebalance_time_only_when_signals_produced(self):
        strat = _make_strategy(rebalance_threshold_pct=5.0)
        strat.on_start()
        strat.last_rebalance_time = 0.0
        # BTC value 600 (60%), ETH value 400 (40%) -> exactly on target, no drift.
        strat.exchange_manager.get_balance.return_value = _balances(BTC=0.6, ETH=4.0)
        strat.exchange_manager.get_ticker.side_effect = _ticker_side_effect(
            {"BTC/USDT": 1000.0, "ETH/USDT": 100.0}
        )
        before = strat.last_rebalance_time
        strat.evaluate()
        assert strat.last_rebalance_time == before


class TestGetCurrentWeights:
    def test_computes_weights_proportional_to_value(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 50.0, "ETH/USDT": 50.0, "_cash": 0.0}
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = {
            "BTC": {"total": 1.0},
            "ETH": {"total": 10.0},
            "USDT": {"total": 0.0},
        }
        strat.exchange_manager.get_ticker.side_effect = _ticker_side_effect(
            {"BTC/USDT": 30000.0, "ETH/USDT": 3000.0}
        )
        weights = strat._get_current_weights()
        # BTC value 30000, ETH value 30000 -> 50/50.
        assert weights["BTC/USDT"] == pytest.approx(50.0)
        assert weights["ETH/USDT"] == pytest.approx(50.0)

    def test_includes_cash_bucket(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 50.0, "_cash": 50.0}
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = {
            "BTC": {"total": 1.0},
            "USDT": {"total": 30000.0},
        }
        strat.exchange_manager.get_ticker.side_effect = _ticker_side_effect(
            {"BTC/USDT": 30000.0}
        )
        weights = strat._get_current_weights()
        assert weights["_cash"] == pytest.approx(50.0)
        assert weights["BTC/USDT"] == pytest.approx(50.0)

    def test_ticker_failure_treats_asset_value_as_zero(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 50.0, "ETH/USDT": 50.0}
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = {
            "BTC": {"total": 1.0},
            "ETH": {"total": 10.0},
        }

        def _get(exchange, symbol):
            if symbol == "ETH/USDT":
                raise RuntimeError("ticker down")
            return {"last": 30000.0}

        strat.exchange_manager.get_ticker.side_effect = _get
        weights = strat._get_current_weights()
        # ETH contributes 0 value, so BTC (the only priced asset) is 100%.
        assert weights["BTC/USDT"] == pytest.approx(100.0)
        assert weights["ETH/USDT"] == pytest.approx(0.0)

    def test_zero_total_value_returns_empty_weights_without_crashing(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 50.0, "ETH/USDT": 50.0}
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = {
            "BTC": {"total": 0.0},
            "ETH": {"total": 0.0},
        }
        weights = strat._get_current_weights()
        assert weights == {}


class TestCalculateRebalanceOrders:
    def test_drift_below_threshold_produces_no_signal(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 60.0, "ETH/USDT": 40.0},
            rebalance_threshold_pct=5.0,
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = _balances(BTC=1.0, ETH=1.0)
        signals = strat._calculate_rebalance_orders({"BTC/USDT": 62.0, "ETH/USDT": 38.0})
        assert signals == []

    def test_underweight_asset_produces_buy_signal(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 60.0, "ETH/USDT": 40.0},
            rebalance_threshold_pct=5.0,
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = _balances(BTC=0.3, ETH=1.0)
        strat.exchange_manager.get_ticker.side_effect = _ticker_side_effect(
            {"BTC/USDT": 10000.0, "ETH/USDT": 3000.0}
        )
        # BTC value 3000, ETH value 3000 -> total 6000, BTC actually at 50% not 30%,
        # but current_weights passed in says BTC is way underweight at 30%.
        signals = strat._calculate_rebalance_orders({"BTC/USDT": 30.0, "ETH/USDT": 70.0})
        buy_signals = [s for s in signals if s["symbol"] == "BTC/USDT"]
        assert len(buy_signals) == 1
        assert buy_signals[0]["side"] == "buy"
        assert buy_signals[0]["exchange"] == strat.exchange
        assert buy_signals[0]["order_type"] == "market"

    def test_overweight_asset_produces_sell_signal(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 40.0, "ETH/USDT": 60.0},
            rebalance_threshold_pct=5.0,
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = _balances(BTC=1.0, ETH=0.3)
        strat.exchange_manager.get_ticker.side_effect = _ticker_side_effect(
            {"BTC/USDT": 10000.0, "ETH/USDT": 3000.0}
        )
        signals = strat._calculate_rebalance_orders({"BTC/USDT": 70.0, "ETH/USDT": 30.0})
        sell_signals = [s for s in signals if s["symbol"] == "BTC/USDT"]
        assert len(sell_signals) == 1
        assert sell_signals[0]["side"] == "sell"

    def test_cash_bucket_never_gets_a_direct_order(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 50.0, "_cash": 50.0},
            rebalance_threshold_pct=5.0,
        )
        strat.on_start()
        strat.exchange_manager.get_balance.return_value = _balances(BTC=1.0)
        strat.exchange_manager.get_ticker.side_effect = _ticker_side_effect(
            {"BTC/USDT": 10000.0}
        )
        signals = strat._calculate_rebalance_orders({"BTC/USDT": 10.0, "_cash": 90.0})
        assert all(s["symbol"] != "_cash" for s in signals)

    def test_nonpositive_price_skips_symbol(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 60.0, "ETH/USDT": 40.0},
            rebalance_threshold_pct=5.0,
        )
        strat.on_start()
        strat.exchange_manager.get_ticker.return_value = {"last": 0}
        signals = strat._calculate_rebalance_orders({"BTC/USDT": 30.0, "ETH/USDT": 70.0})
        assert signals == []

    def test_ticker_exception_skips_symbol(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 60.0, "ETH/USDT": 40.0},
            rebalance_threshold_pct=5.0,
        )
        strat.on_start()
        strat.exchange_manager.get_ticker.side_effect = RuntimeError("down")
        signals = strat._calculate_rebalance_orders({"BTC/USDT": 30.0, "ETH/USDT": 70.0})
        assert signals == []


class TestGetAssetValue:
    def test_cash_bucket_returns_usdt_total(self):
        strat = _make_strategy()
        strat.on_start()
        value = strat._get_asset_value("_cash", {"USDT": {"total": 500.0}})
        assert value == pytest.approx(500.0)

    def test_zero_quantity_returns_zero_without_calling_ticker(self):
        strat = _make_strategy()
        strat.on_start()
        value = strat._get_asset_value("BTC/USDT", {"BTC": {"total": 0}})
        assert value == 0
        strat.exchange_manager.get_ticker.assert_not_called()

    def test_ticker_exception_returns_zero(self):
        strat = _make_strategy()
        strat.on_start()
        strat.exchange_manager.get_ticker.side_effect = RuntimeError("down")
        value = strat._get_asset_value("BTC/USDT", {"BTC": {"total": 1.0}})
        assert value == 0


class TestToDict:
    def test_includes_rebalancing_fields(self):
        strat = _make_strategy(
            target_allocation={"BTC/USDT": 60.0, "ETH/USDT": 40.0},
            rebalance_threshold_pct=7.5,
            interval="weekly",
        )
        strat.on_start()
        d = strat.to_dict()
        assert d["target_allocation"] == {"BTC/USDT": 60.0, "ETH/USDT": 40.0}
        assert d["threshold_pct"] == pytest.approx(7.5)
        assert d["interval"] == "weekly"
