"""Tests for monitor_daemon.py -- the daemon that owns real order placement,
protective stops, and PID/state bookkeeping for the live monitoring loop.

This module had zero test coverage before this file: every other script
under scripts/ has a companion test, but the orchestrator that actually
calls exchange_manager.place_order() in production did not. Bugs here are
the ones that would first show up as unprotected positions or lost track of
open orders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import monitor_daemon  # noqa: E402
from monitor_daemon import MonitorDaemon  # noqa: E402
from exchange_manager import ExchangeError  # noqa: E402
from risk_manager import RiskLimitExceeded  # noqa: E402


def _daemon(**attrs):
    """Build a MonitorDaemon without running __init__ (which touches disk
    paths and env vars), setting only the attributes a test needs."""
    daemon = MonitorDaemon.__new__(MonitorDaemon)
    daemon._order_registry = {}
    daemon._entry_quality_on = False
    daemon._state = {
        "running": False, "started_at": None, "last_check": None,
        "checks_performed": 0, "errors": [],
    }
    for key, value in attrs.items():
        setattr(daemon, key, value)
    return daemon


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    def send_alert(self, alert_type, data):
        self.alerts.append((alert_type, data))
        return {"status": "sent"}


class FakeExchangeManager:
    def __init__(self):
        self.available_exchanges = ["binance"]
        self.balances = {}
        self.tickers = {}
        self.orders = {}
        self.placed_orders = []
        self.stop_orders_cancelled = []
        self.open_orders = {}
        self.raise_on_get_ticker = False

    def get_balance(self, ex_name):
        return self.balances[ex_name]

    def get_ticker(self, ex_name, symbol):
        if self.raise_on_get_ticker:
            raise ExchangeError("ticker unavailable")
        return self.tickers[symbol]

    def get_order(self, ex_name, order_id, symbol=None):
        return self.orders[order_id]

    def get_open_orders(self, ex_name, symbol):
        return self.open_orders.get(symbol, [])

    def place_order(self, exchange, symbol, side, amount, price=None, order_type="market"):
        order = {
            "id": f"order-{len(self.placed_orders)}",
            "status": "closed",
            "price": price or self.tickers.get(symbol, {}).get("last"),
            "filled": amount,
            "amount": amount,
            "cost": amount * (price or self.tickers.get(symbol, {}).get("last", 0)),
        }
        self.placed_orders.append((exchange, symbol, side, amount, price, order_type))
        return order

    def place_stop_loss_order(self, exchange, symbol, amount, stop_price):
        return {"id": "stop-1", "stop_price": stop_price}

    def cancel_stop_orders(self, exchange, symbol):
        self.stop_orders_cancelled.append((exchange, symbol))


class FakeStrategy:
    def __init__(self, name="grid", follow_up=None):
        self.name = name
        self.follow_up = follow_up
        self.placed_calls = []
        self.filled_calls = []

    def on_order_placed(self, signal_data, order):
        self.placed_calls.append((signal_data, order))

    def on_order_filled(self, order):
        self.filled_calls.append(order)
        follow_up, self.follow_up = self.follow_up, None
        return follow_up


class FakeStrategyEngine:
    def __init__(self, strategy=None):
        self.strategy = strategy
        self.saved = 0

    def get_strategy_instance(self, strategy_id):
        return self.strategy

    def save_state(self):
        self.saved += 1


class FakeRiskManager:
    def __init__(self, stop_price=9000.0, raise_exc=None):
        self.stop_price = stop_price
        self.raise_exc = raise_exc
        self.validate_calls = []

    def validate_order(self, **kwargs):
        self.validate_calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc

    def stop_loss_price(self, entry_price, side="buy"):
        return self.stop_price


# ---------------------------------------------------------------------------
# PID / state file management
# ---------------------------------------------------------------------------

def test_load_state_defaults_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor_daemon, "_STATE_PATH", tmp_path / "missing.json")
    state = MonitorDaemon._load_state()
    assert state == {
        "running": False, "started_at": None, "last_check": None,
        "checks_performed": 0, "errors": [],
    }


def test_save_then_load_state_roundtrip(tmp_path, monkeypatch):
    state_path = tmp_path / "nested" / "state.json"
    monkeypatch.setattr(monitor_daemon, "_STATE_PATH", state_path)
    daemon = _daemon(_state={"running": True, "checks_performed": 7,
                              "started_at": "t0", "last_check": "t1", "errors": []})
    daemon._save_state()
    assert MonitorDaemon._load_state()["checks_performed"] == 7


def test_load_state_survives_corrupt_json(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text("{not valid json")
    monkeypatch.setattr(monitor_daemon, "_STATE_PATH", state_path)
    state = MonitorDaemon._load_state()
    assert state["checks_performed"] == 0


def test_pid_write_read_remove_roundtrip(tmp_path, monkeypatch):
    pid_path = tmp_path / "nested" / "daemon.pid"
    monkeypatch.setattr(monitor_daemon, "_PID_PATH", pid_path)
    MonitorDaemon._write_pid()
    assert MonitorDaemon._read_pid() == __import__("os").getpid()
    MonitorDaemon._remove_pid()
    assert MonitorDaemon._read_pid() is None


def test_read_pid_returns_none_for_garbage_content(tmp_path, monkeypatch):
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("not-a-pid")
    monkeypatch.setattr(monitor_daemon, "_PID_PATH", pid_path)
    assert MonitorDaemon._read_pid() is None


def test_is_process_running_true_for_self():
    import os
    assert MonitorDaemon._is_process_running(os.getpid()) is True


def test_is_process_running_false_for_bogus_pid():
    # PID 2**31-1 is not a real process on any sane system.
    assert MonitorDaemon._is_process_running(2**31 - 1) is False


def test_get_status_reports_not_running_without_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor_daemon, "_PID_PATH", tmp_path / "none.pid")
    daemon = _daemon()
    status = daemon.get_status()
    assert status["running"] is False
    assert status["pid"] is None


def test_get_status_reports_recent_errors_tail_only():
    daemon = _daemon(_state={
        "running": True, "started_at": "t0", "last_check": "t1",
        "checks_performed": 3, "errors": [f"e{i}" for i in range(10)],
    })
    status = daemon.get_status()
    assert status["recent_errors"] == [f"e{i}" for i in range(5, 10)]


# ---------------------------------------------------------------------------
# _exchange_portfolio_value
# ---------------------------------------------------------------------------

def test_portfolio_value_sums_stablecoins_directly():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.balances["binance"] = {"USDT": {"total": 100}, "USDC": {"total": 50}}
    value = daemon._exchange_portfolio_value(exm, "binance")
    assert value == 150


def test_portfolio_value_converts_non_stable_via_ticker():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.balances["binance"] = {"BTC": {"total": 2}}
    exm.tickers["BTC/USDT"] = {"last": 30000}
    value = daemon._exchange_portfolio_value(exm, "binance")
    assert value == 60000


def test_portfolio_value_skips_asset_when_ticker_fetch_fails():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.balances["binance"] = {"USDT": {"total": 10}, "BTC": {"total": 1}}
    exm.raise_on_get_ticker = True
    value = daemon._exchange_portfolio_value(exm, "binance")
    # USDT still counts; BTC silently contributes 0 since its ticker errored.
    assert value == 10


def test_portfolio_value_ignores_non_dict_balance_entries():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.balances["binance"] = {"info": "not a balance dict", "USDT": {"total": 5}}
    value = daemon._exchange_portfolio_value(exm, "binance")
    assert value == 5


def test_portfolio_value_returns_zero_when_get_balance_raises():
    daemon = _daemon()

    class RaisingExchangeManager(FakeExchangeManager):
        def get_balance(self, ex_name):
            raise ExchangeError("down")

    value = daemon._exchange_portfolio_value(RaisingExchangeManager(), "binance")
    assert value == 0.0


# ---------------------------------------------------------------------------
# _check_open_orders
# ---------------------------------------------------------------------------

def test_check_open_orders_dispatches_fill_and_clears_registry():
    strategy = FakeStrategy()
    daemon = _daemon(_order_registry={
        "o1": {"strategy_id": "s1", "exchange": "binance", "symbol": "BTC/USDT"},
    })
    exm = FakeExchangeManager()
    exm.orders["o1"] = {"id": "o1", "status": "closed"}
    engine = FakeStrategyEngine(strategy=strategy)

    daemon._check_open_orders(exm, engine, FakeRiskManager(), FakeNotifier())

    assert "o1" not in daemon._order_registry
    assert strategy.filled_calls == [exm.orders["o1"]]
    assert engine.saved == 1


def test_check_open_orders_follows_up_chained_signal():
    """A filled order whose strategy returns a follow-up signal (e.g. a grid
    counter-order) must have that follow-up executed immediately."""
    follow_up_signal = {
        "exchange": "binance", "symbol": "BTC/USDT", "side": "sell",
        "amount": 1, "price": 100, "order_type": "limit",
    }
    strategy = FakeStrategy(follow_up=follow_up_signal)
    daemon = _daemon(_order_registry={
        "o1": {"strategy_id": "s1", "exchange": "binance", "symbol": "BTC/USDT"},
    })
    exm = FakeExchangeManager()
    exm.orders["o1"] = {"id": "o1", "status": "filled"}
    engine = FakeStrategyEngine(strategy=strategy)

    daemon._check_open_orders(exm, engine, FakeRiskManager(), FakeNotifier())

    # The follow-up sell must actually have been placed on the exchange.
    assert len(exm.placed_orders) == 1
    assert exm.placed_orders[0][2] == "sell"


def test_check_open_orders_removes_cancelled_without_dispatch():
    strategy = FakeStrategy()
    daemon = _daemon(_order_registry={
        "o1": {"strategy_id": "s1", "exchange": "binance", "symbol": "BTC/USDT"},
    })
    exm = FakeExchangeManager()
    exm.orders["o1"] = {"id": "o1", "status": "canceled"}
    engine = FakeStrategyEngine(strategy=strategy)

    daemon._check_open_orders(exm, engine, FakeRiskManager(), FakeNotifier())

    assert "o1" not in daemon._order_registry
    assert strategy.filled_calls == []


def test_check_open_orders_keeps_tracking_when_get_order_raises():
    daemon = _daemon(_order_registry={
        "o1": {"strategy_id": "s1", "exchange": "binance", "symbol": "BTC/USDT"},
    })

    class RaisingExchangeManager(FakeExchangeManager):
        def get_order(self, ex_name, order_id, symbol=None):
            raise ExchangeError("network blip")

    daemon._check_open_orders(RaisingExchangeManager(), FakeStrategyEngine(), FakeRiskManager(), FakeNotifier())

    assert "o1" in daemon._order_registry


# ---------------------------------------------------------------------------
# _execute_signal
# ---------------------------------------------------------------------------

def _buy_signal(**overrides):
    signal = {
        "strategy_id": "s1", "strategy_name": "grid", "exchange": "binance",
        "symbol": "BTC/USDT", "side": "buy", "amount": 0.01, "price": None,
        "order_type": "market",
    }
    signal.update(overrides)
    return signal


def test_execute_signal_places_protective_stop_on_buy_fill():
    daemon = _daemon(_entry_quality_on=False)
    exm = FakeExchangeManager()
    exm.tickers["BTC/USDT"] = {"last": 30000}
    strategy = FakeStrategy()
    engine = FakeStrategyEngine(strategy=strategy)
    risk = FakeRiskManager(stop_price=28500.0)
    notifier = FakeNotifier()

    daemon._execute_signal(_buy_signal(), engine, exm, risk, notifier)

    assert len(exm.placed_orders) == 1
    assert exm.placed_orders[0][2] == "buy"
    assert risk.validate_calls, "risk_manager.validate_order must be consulted before placing"
    alert_types = [a for a, _ in notifier.alerts]
    assert "trade_executed" in alert_types


def test_execute_signal_blocked_by_entry_quality_gate():
    daemon = _daemon(_entry_quality_on=True)
    daemon._passes_entry_quality = lambda exm, ex, sym: (False, "dead-cat bounce")
    exm = FakeExchangeManager()
    exm.tickers["BTC/USDT"] = {"last": 30000}
    risk = FakeRiskManager()
    notifier = FakeNotifier()

    daemon._execute_signal(_buy_signal(), FakeStrategyEngine(), exm, risk, notifier)

    assert exm.placed_orders == []
    assert risk.validate_calls == []


def test_execute_signal_no_order_when_risk_manager_rejects():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.tickers["BTC/USDT"] = {"last": 30000}
    risk = FakeRiskManager(raise_exc=RiskLimitExceeded("max_position_size_pct", "too big"))
    notifier = FakeNotifier()

    daemon._execute_signal(_buy_signal(), FakeStrategyEngine(), exm, risk, notifier)

    assert exm.placed_orders == []
    alert_types = [a for a, _ in notifier.alerts]
    assert alert_types == ["strategy_error"]


def test_execute_signal_cancels_resting_stop_before_selling():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.tickers["BTC/USDT"] = {"last": 30000}
    strategy = FakeStrategy()
    engine = FakeStrategyEngine(strategy=strategy)
    risk = FakeRiskManager()
    notifier = FakeNotifier()

    sell_signal = _buy_signal(side="sell", amount=0.01)
    daemon._execute_signal(sell_signal, engine, exm, risk, notifier)

    assert exm.stop_orders_cancelled == [("binance", "BTC/USDT")]
    assert exm.placed_orders[0][2] == "sell"


def test_execute_signal_tracks_unfilled_order_in_registry():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.tickers["BTC/USDT"] = {"last": 30000}

    def open_order(exchange, symbol, side, amount, price=None, order_type="market"):
        return {"id": "pending-1", "status": "open", "amount": amount}

    exm.place_order = open_order
    risk = FakeRiskManager()
    notifier = FakeNotifier()

    daemon._execute_signal(_buy_signal(), FakeStrategyEngine(strategy=FakeStrategy()), exm, risk, notifier)

    assert daemon._order_registry["pending-1"]["strategy_id"] == "s1"


def test_execute_signal_swallows_place_order_error_and_alerts():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.tickers["BTC/USDT"] = {"last": 30000}

    def failing_place_order(*args, **kwargs):
        raise ExchangeError("insufficient balance")

    exm.place_order = failing_place_order
    risk = FakeRiskManager()
    notifier = FakeNotifier()

    result = daemon._execute_signal(_buy_signal(), FakeStrategyEngine(), exm, risk, notifier)

    assert result is None
    alert_types = [a for a, _ in notifier.alerts]
    assert alert_types == ["strategy_error"]


def test_execute_signal_protective_stop_failure_alerts_but_does_not_raise():
    daemon = _daemon()
    exm = FakeExchangeManager()
    exm.tickers["BTC/USDT"] = {"last": 30000}

    def failing_stop(*args, **kwargs):
        raise ExchangeError("exchange rejected stop order")

    exm.place_stop_loss_order = failing_stop
    strategy = FakeStrategy()
    engine = FakeStrategyEngine(strategy=strategy)
    risk = FakeRiskManager(stop_price=28500.0)
    notifier = FakeNotifier()

    # Must not raise even though the protective stop placement fails --
    # an unprotected position is bad, but crashing the daemon is worse.
    daemon._execute_signal(_buy_signal(), engine, exm, risk, notifier)

    alert_types = [a for a, _ in notifier.alerts]
    assert "trade_executed" in alert_types
    assert "strategy_error" in alert_types
