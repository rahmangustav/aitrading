"""Tests for order-safety behaviour in the exchange manager and risk manager.

Covers the three fixes:
  1. Exchange-native protective stop-loss orders.
  2. Idempotent order placement (create calls are never auto-retried on
     ambiguous network errors, so a timeout cannot double-fill).
  3. Amount/price precision rounding + minimum amount/notional enforcement.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import ccxt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import exchange_manager as em_mod  # noqa: E402
from exchange_manager import ExchangeManager, ExchangeError  # noqa: E402
from cache import TTLCache  # noqa: E402
from risk_manager import RiskManager  # noqa: E402


_MARKET = {
    "BTC/USDT": {
        "symbol": "BTC/USDT",
        "precision": {"amount": 4, "price": 2},
        "limits": {"amount": {"min": 0.01}, "cost": {"min": 10.0}},
    }
}


class FakeExchange:
    """Minimal ccxt-like exchange that records calls and can fail on demand."""

    def __init__(self):
        self.created = []
        self.create_calls = 0
        self.fail_create_with = None  # exception instance to raise on create_order
        self.open_orders = []
        self.cancelled = []

    def load_markets(self):
        return _MARKET

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.4f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.create_calls += 1
        if self.fail_create_with is not None:
            raise self.fail_create_with
        order = {
            "id": f"o{self.create_calls}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "closed",
            "clientOrderId": (params or {}).get("clientOrderId"),
            "params": params or {},
        }
        self.created.append(order)
        return order

    def fetch_open_orders(self, symbol=None):
        return self.open_orders

    def cancel_order(self, order_id, symbol=None):
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "canceled"}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip retry backoff sleeps to keep tests fast."""
    monkeypatch.setattr(em_mod.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def manager():
    fake = FakeExchange()
    mgr = ExchangeManager.__new__(ExchangeManager)
    mgr._exchanges = {"binance": fake}
    mgr._cache = TTLCache()
    mgr._config = {}
    mgr._rate_limits = {}
    mgr._demo = True
    return mgr, fake


# --- Fix 3: precision & minimums ------------------------------------------

def test_amount_rounded_to_precision(manager):
    mgr, fake = manager
    mgr.place_order("binance", "BTC/USDT", "buy", 0.123456, order_type="market")
    assert fake.created[0]["amount"] == 0.1235


def test_below_min_amount_rejected(manager):
    mgr, fake = manager
    with pytest.raises(ExchangeError):
        mgr.place_order("binance", "BTC/USDT", "buy", 0.005, order_type="market")
    assert fake.create_calls == 0  # never reached the exchange


def test_below_min_notional_rejected(manager):
    mgr, fake = manager
    with pytest.raises(ExchangeError):
        mgr.place_order("binance", "BTC/USDT", "buy", 0.02, price=100.0, order_type="limit")
    assert fake.create_calls == 0


# --- Fix 2: idempotency ----------------------------------------------------

def test_client_order_id_attached(manager):
    mgr, fake = manager
    mgr.place_order("binance", "BTC/USDT", "buy", 0.5, order_type="market")
    assert fake.created[0]["params"]["clientOrderId"].startswith("ct-")


def test_order_creation_not_retried_on_network_error(manager):
    mgr, fake = manager
    fake.fail_create_with = ccxt.NetworkError("timeout")
    with pytest.raises(ExchangeError):
        mgr.place_order("binance", "BTC/USDT", "buy", 0.5, order_type="market")
    # Critical: exactly one attempt -- no retry that could double-fill.
    assert fake.create_calls == 1


def test_read_operations_still_retry(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def flaky_read():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ccxt.NetworkError("blip")
        return "ok"

    result = mgr._execute_with_retry("binance", "fetch", flaky_read)
    assert result == "ok"
    assert calls["n"] == 2  # retried once


# --- Fix 1: native stop-loss ----------------------------------------------

def test_stop_loss_order_params(manager):
    mgr, fake = manager
    result = mgr.place_stop_loss_order("binance", "BTC/USDT", 0.5, stop_price=90.0)
    order = fake.created[0]
    assert order["type"] == "STOP_LOSS_LIMIT"
    assert order["side"] == "sell"
    assert order["params"]["stopPrice"] == 90.0
    # Limit price sits just below the stop so it still fills.
    assert result["limit_price"] < 90.0


def test_cancel_stop_orders_only_cancels_stops(manager):
    mgr, fake = manager
    fake.open_orders = [
        {"id": "limit1", "symbol": "BTC/USDT", "type": "limit"},
        {"id": "stop1", "symbol": "BTC/USDT", "type": "stop_loss_limit"},
        {"id": "stop2", "symbol": "BTC/USDT", "type": "limit", "stopPrice": 90.0},
    ]
    cancelled = mgr.cancel_stop_orders("binance", "BTC/USDT")
    ids = {c["id"] for c in cancelled}
    assert ids == {"stop1", "stop2"}
    assert "limit1" not in fake.cancelled


# --- Risk manager stop-loss price -----------------------------------------

def test_stop_loss_price_long_and_short():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CRYPTO_RISK_STATE_PATH"] = str(Path(tmp) / "state.json")
        rm = RiskManager()
        rm._stop_loss_cfg = {"enabled": True, "default_pct": 5.0}
        assert rm.stop_loss_price(100.0, "buy") == pytest.approx(95.0)
        assert rm.stop_loss_price(100.0, "sell") == pytest.approx(105.0)
        rm._stop_loss_cfg = {"enabled": False, "default_pct": 5.0}
        assert rm.stop_loss_price(100.0, "buy") is None
