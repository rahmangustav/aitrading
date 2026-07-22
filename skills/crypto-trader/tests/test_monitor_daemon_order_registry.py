"""Regression test: BUY fills detected via polling must get the same
protective stop-loss + learning-bridge entry as BUY fills that close
immediately when placed.

Before this fix, `_place_protective_stop` and `learning_bridge.record_entry`
were only called from `_execute_signal()`'s immediate-fill branch. A limit
BUY (used by grid_trading and arbitrage) that rests and fills later is
detected by `_check_open_orders()` instead, which called
`strategy.on_order_filled(order)` but never placed a protective stop or
recorded the entry -- leaving the position unprotected at the exchange.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import monitor_daemon as md_mod  # noqa: E402
from monitor_daemon import MonitorDaemon  # noqa: E402


class FakeStrategy:
    name = "grid_trading"

    def __init__(self):
        self.filled_orders = []

    def on_order_filled(self, order):
        self.filled_orders.append(order)
        return None


class FakeStrategyEngine:
    def __init__(self, strategy):
        self._strategy = strategy
        self.saved = False

    def get_strategy_instance(self, strategy_id):
        return self._strategy

    def save_state(self):
        self.saved = True


class FakeExchangeManager:
    def __init__(self, order):
        self._order = order
        self.stop_orders_placed = []

    def get_order(self, exchange_name, order_id, symbol=None):
        return self._order

    def place_stop_loss_order(self, exchange, symbol, amount, stop_price):
        self.stop_orders_placed.append(
            {"exchange": exchange, "symbol": symbol, "amount": amount, "stop_price": stop_price}
        )
        return {"id": "stop-1"}


class FakeRiskManager:
    def stop_loss_price(self, entry_price, side="buy"):
        return entry_price * 0.95


class FakeLearningBridge:
    def __init__(self):
        self.recorded = []

    def record_entry(self, symbol, price, confidence=None, ta=None):
        self.recorded.append({"symbol": symbol, "price": price, "confidence": confidence, "ta": ta})


def _daemon_with_registry(order_id, side, meta_extra=None):
    daemon = MonitorDaemon.__new__(MonitorDaemon)
    daemon._order_registry = {
        order_id: {"strategy_id": "s1", "exchange": "binance", "symbol": "BTC/USDT", "side": side, **(meta_extra or {})},
    }
    return daemon


def test_buy_fill_detected_by_polling_gets_protective_stop(monkeypatch):
    fake_bridge = FakeLearningBridge()
    monkeypatch.setattr(md_mod, "learning_bridge", fake_bridge)

    filled_order = {
        "id": "order-1", "status": "closed", "price": 100.0,
        "cost": 100.0, "filled": 1.0, "amount": 1.0,
    }
    strategy = FakeStrategy()
    engine = FakeStrategyEngine(strategy)
    exchange_manager = FakeExchangeManager(filled_order)
    risk_manager = FakeRiskManager()

    daemon = _daemon_with_registry("order-1", side="buy")
    daemon._check_open_orders(exchange_manager, engine, risk_manager, notifier=None)

    assert exchange_manager.stop_orders_placed == [
        {"exchange": "binance", "symbol": "BTC/USDT", "amount": 1.0, "stop_price": 95.0}
    ]
    assert fake_bridge.recorded == [
        {"symbol": "BTC/USDT", "price": 100.0, "confidence": None, "ta": {"strategy": "grid_trading", "reason": ""}}
    ]
    assert strategy.filled_orders == [filled_order]
    assert "order-1" not in daemon._order_registry


def test_sell_fill_detected_by_polling_does_not_place_stop(monkeypatch):
    fake_bridge = FakeLearningBridge()
    monkeypatch.setattr(md_mod, "learning_bridge", fake_bridge)

    filled_order = {"id": "order-2", "status": "filled", "price": 100.0, "filled": 1.0}
    strategy = FakeStrategy()
    engine = FakeStrategyEngine(strategy)
    exchange_manager = FakeExchangeManager(filled_order)
    risk_manager = FakeRiskManager()

    daemon = _daemon_with_registry("order-2", side="sell")
    daemon._check_open_orders(exchange_manager, engine, risk_manager, notifier=None)

    assert exchange_manager.stop_orders_placed == []
    assert fake_bridge.recorded == []


def test_buy_fill_learning_bridge_disabled_still_places_stop(monkeypatch):
    monkeypatch.setattr(md_mod, "learning_bridge", None)

    filled_order = {"id": "order-3", "status": "closed", "price": 50.0, "filled": 2.0}
    strategy = FakeStrategy()
    engine = FakeStrategyEngine(strategy)
    exchange_manager = FakeExchangeManager(filled_order)
    risk_manager = FakeRiskManager()

    daemon = _daemon_with_registry("order-3", side="buy")
    daemon._check_open_orders(exchange_manager, engine, risk_manager, notifier=None)

    assert exchange_manager.stop_orders_placed == [
        {"exchange": "binance", "symbol": "BTC/USDT", "amount": 2.0, "stop_price": 47.5}
    ]
