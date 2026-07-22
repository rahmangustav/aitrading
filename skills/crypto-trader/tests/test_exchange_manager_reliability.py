"""Tests for ExchangeManager reliability paths that test_exchange_safety.py
does not cover: the retry/backoff exception matrix in _execute_with_retry,
and the account/order data-shaping methods (get_balance, get_open_orders,
cancel_all_orders, get_order_history, get_orderbook, precision lookups).

These sit on the path every strategy and monitor_daemon.py goes through for
every single exchange call, so a bug here (e.g. retrying an auth failure, or
silently swallowing a cancel failure) affects every strategy at once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import ccxt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import exchange_manager as em_mod  # noqa: E402
from exchange_manager import ExchangeManager, ExchangeError  # noqa: E402
from cache import TTLCache  # noqa: E402


_MARKET = {
    "BTC/USDT": {
        "symbol": "BTC/USDT",
        "precision": {"amount": 4, "price": 2},
        "limits": {"amount": {"min": 0.01}, "cost": {"min": 10.0}},
    }
}


class FakeExchange:
    """Minimal ccxt-like exchange with per-method scripted responses/failures."""

    def __init__(self):
        self.open_orders = []
        self.cancelled = []
        self.cancel_fail_ids = set()
        self.balance = {"total": {}, "free": {}, "used": {}}
        self.balance_calls = 0
        self.closed_orders = []
        self.all_orders = []
        self.fetch_closed_orders_raises = None
        self.fetch_orders_raises = None
        self.ticker = {}
        self.orderbook = {"bids": [], "asks": [], "timestamp": None}

    def load_markets(self):
        return _MARKET

    def fetch_balance(self):
        self.balance_calls += 1
        return self.balance

    def fetch_open_orders(self, symbol=None):
        return self.open_orders

    def cancel_order(self, order_id, symbol=None):
        if order_id in self.cancel_fail_ids:
            raise ccxt.ExchangeError("cancel rejected")
        self.cancelled.append(order_id)
        return {"id": order_id, "status": "canceled"}

    def fetch_closed_orders(self, symbol=None, since=None, limit=None):
        if self.fetch_closed_orders_raises is not None:
            raise self.fetch_closed_orders_raises
        return self.closed_orders

    def fetch_orders(self, symbol=None, since=None, limit=None):
        if self.fetch_orders_raises is not None:
            raise self.fetch_orders_raises
        return self.all_orders

    def fetch_ticker(self, symbol):
        return self.ticker

    def fetch_order_book(self, symbol, limit):
        return self.orderbook


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


# --- _execute_with_retry exception matrix -----------------------------------

def test_rate_limit_exceeded_is_retried_then_succeeds(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ccxt.RateLimitExceeded("slow down")
        return "ok"

    assert mgr._execute_with_retry("binance", "op", flaky) == "ok"
    assert calls["n"] == 2


def test_rate_limit_exceeded_exhausts_after_max_retries(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def always_limited():
        calls["n"] += 1
        raise ccxt.RateLimitExceeded("still slow")

    with pytest.raises(ExchangeError):
        mgr._execute_with_retry("binance", "op", always_limited)
    assert calls["n"] == em_mod._MAX_RETRIES


def test_authentication_error_raises_immediately(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def bad_auth():
        calls["n"] += 1
        raise ccxt.AuthenticationError("bad key")

    with pytest.raises(ExchangeError) as exc_info:
        mgr._execute_with_retry("binance", "op", bad_auth)
    assert calls["n"] == 1
    assert exc_info.value.status_code == 401


def test_insufficient_funds_raises_immediately(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def no_funds():
        calls["n"] += 1
        raise ccxt.InsufficientFunds("not enough")

    with pytest.raises(ExchangeError) as exc_info:
        mgr._execute_with_retry("binance", "op", no_funds)
    assert calls["n"] == 1
    assert exc_info.value.status_code == 400


def test_invalid_order_raises_immediately(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def bad_order():
        calls["n"] += 1
        raise ccxt.InvalidOrder("bad params")

    with pytest.raises(ExchangeError) as exc_info:
        mgr._execute_with_retry("binance", "op", bad_order)
    assert calls["n"] == 1
    assert exc_info.value.status_code == 400


def test_generic_exchange_error_raises_immediately(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def generic_failure():
        calls["n"] += 1
        raise ccxt.ExchangeError("something else")

    with pytest.raises(ExchangeError) as exc_info:
        mgr._execute_with_retry("binance", "op", generic_failure)
    assert calls["n"] == 1
    assert exc_info.value.status_code is None


def test_non_idempotent_network_error_never_retried(manager):
    mgr, _fake = manager
    calls = {"n": 0}

    def timeout_on_write():
        calls["n"] += 1
        raise ccxt.NetworkError("timeout")

    with pytest.raises(ExchangeError):
        mgr._execute_with_retry("binance", "op", timeout_on_write, idempotent=False)
    assert calls["n"] == 1


# --- get_balance -------------------------------------------------------------

def test_get_balance_filters_zero_and_none_entries(manager):
    mgr, fake = manager
    fake.balance = {
        "total": {"BTC": 1.5, "USDT": 0, "ETH": None, "DOGE": 100.0},
        "free": {"BTC": 1.0, "DOGE": 90.0},
        "used": {"BTC": 0.5, "DOGE": 10.0},
    }
    result = mgr.get_balance("binance")
    assert set(result.keys()) == {"BTC", "DOGE"}
    assert result["BTC"] == {"total": 1.5, "free": 1.0, "used": 0.5}


def test_get_balance_caches_across_calls(manager):
    mgr, fake = manager
    fake.balance = {"total": {"BTC": 1.0}, "free": {"BTC": 1.0}, "used": {}}
    mgr.get_balance("binance")
    mgr.get_balance("binance")
    assert fake.balance_calls == 1


def test_get_balance_single_asset_found(manager):
    mgr, fake = manager
    fake.balance = {"total": {"BTC": 2.0}, "free": {"BTC": 1.5}, "used": {"BTC": 0.5}}
    result = mgr.get_balance("binance", asset="btc")
    assert result == {"asset": "BTC", "total": 2.0, "free": 1.5, "used": 0.5}


def test_get_balance_single_asset_missing_returns_zeros(manager):
    mgr, fake = manager
    fake.balance = {"total": {"BTC": 2.0}, "free": {}, "used": {}}
    result = mgr.get_balance("binance", asset="ETH")
    assert result == {"asset": "ETH", "total": 0, "free": 0, "used": 0}


# --- get_open_orders -----------------------------------------------------

def test_get_open_orders_preserves_stop_trigger_fields(manager):
    mgr, fake = manager
    fake.open_orders = [
        {"id": "o1", "symbol": "BTC/USDT", "type": "limit", "stopPrice": 90.0,
         "triggerPrice": None, "amount": 1, "price": 100, "filled": 0,
         "remaining": 1, "status": "open", "timestamp": 1},
    ]
    orders = mgr.get_open_orders("binance")
    assert orders[0]["stopPrice"] == 90.0
    assert orders[0]["id"] == "o1"


def test_get_open_orders_caches(manager):
    mgr, fake = manager
    fake.open_orders = [{"id": "o1", "symbol": "BTC/USDT"}]
    mgr.get_open_orders("binance")
    fake.open_orders = [{"id": "o2", "symbol": "BTC/USDT"}]
    cached = mgr.get_open_orders("binance")
    assert cached[0]["id"] == "o1"


# --- cancel_all_orders -----------------------------------------------------

def test_cancel_all_orders_continues_past_individual_failure(manager):
    mgr, fake = manager
    fake.open_orders = [
        {"id": "ok1", "symbol": "BTC/USDT"},
        {"id": "bad1", "symbol": "BTC/USDT"},
        {"id": "ok2", "symbol": "BTC/USDT"},
    ]
    fake.cancel_fail_ids = {"bad1"}
    results = mgr.cancel_all_orders("binance")
    statuses = {r["id"]: r["status"] for r in results}
    assert statuses["ok1"] == "canceled"
    assert statuses["bad1"] == "cancel_failed"
    assert statuses["ok2"] == "canceled"
    assert set(fake.cancelled) == {"ok1", "ok2"}


# --- get_order_history -----------------------------------------------------

def test_get_order_history_uses_fetch_closed_orders_directly(manager):
    mgr, fake = manager
    fake.closed_orders = [{"id": "c1", "status": "closed"}]
    result = mgr.get_order_history("binance")
    assert [o["id"] for o in result] == ["c1"]


def test_get_order_history_falls_back_when_not_supported(manager):
    mgr, fake = manager
    fake.fetch_closed_orders_raises = ccxt.NotSupported("nope")
    fake.all_orders = [
        {"id": "a1", "status": "closed"},
        {"id": "a2", "status": "open"},
        {"id": "a3", "status": "canceled"},
    ]
    result = mgr.get_order_history("binance")
    assert {o["id"] for o in result} == {"a1", "a3"}


def test_get_order_history_returns_empty_when_both_unsupported(manager):
    mgr, fake = manager
    fake.fetch_closed_orders_raises = ccxt.NotSupported("nope")
    fake.fetch_orders_raises = ccxt.NotSupported("also nope")
    assert mgr.get_order_history("binance") == []


# --- precision / min-amount utilities --------------------------------------

def test_precision_lookups_none_when_market_missing(manager):
    mgr, _fake = manager
    assert mgr.get_min_order_amount("binance", "ETH/USDT") is None
    assert mgr.get_price_precision("binance", "ETH/USDT") is None
    assert mgr.get_amount_precision("binance", "ETH/USDT") is None


def test_precision_lookups_return_configured_values(manager):
    mgr, _fake = manager
    assert mgr.get_min_order_amount("binance", "BTC/USDT") == 0.01
    assert mgr.get_price_precision("binance", "BTC/USDT") == 2
    assert mgr.get_amount_precision("binance", "BTC/USDT") == 4


# --- get_orderbook -----------------------------------------------------------

def test_get_orderbook_computes_spread(manager):
    mgr, fake = manager
    fake.orderbook = {
        "bids": [[100.0, 1.0], [99.5, 2.0]],
        "asks": [[100.5, 1.0], [101.0, 2.0]],
        "timestamp": 123,
    }
    result = mgr.get_orderbook("binance", "BTC/USDT", limit=2)
    assert result["spread"] == pytest.approx(0.5)
    assert result["spread_pct"] == pytest.approx(0.4975, rel=1e-3)


def test_get_orderbook_no_spread_when_one_side_empty(manager):
    mgr, fake = manager
    fake.orderbook = {"bids": [], "asks": [[100.5, 1.0]], "timestamp": 1}
    result = mgr.get_orderbook("binance", "BTC/USDT")
    assert result["spread"] is None
