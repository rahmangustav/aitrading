"""Tests for CopyTradingStrategy.

This strategy has never had test coverage. It is a framework/stub
implementation (leaderboard and wallet monitoring both always return no
signals today), but the parts that already run in this repo -- exchange
fallback selection on start, the evaluate() rate limiter, and copy-amount
sizing in _create_copy_signal -- are plain logic worth locking down before
the leaderboard/wallet integrations are filled in.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.copy_trading import CopyTradingStrategy  # noqa: E402


def _make_strategy(**params):
    exchange_mgr = MagicMock()
    exchange_mgr.available_exchanges = ["binance"]
    risk_mgr = MagicMock()
    return CopyTradingStrategy(
        strategy_id="copy_test",
        params=params,
        exchange_manager=exchange_mgr,
        risk_manager=risk_mgr,
    )


class TestInitDefaults:
    def test_defaults(self):
        s = _make_strategy()
        assert s.source == "binance_leaderboard"
        assert s.wallet_addresses == []
        assert s.max_copy_amount == pytest.approx(25.0)
        assert s.delay_seconds == 5
        assert s.exchange == ""
        assert s.tracked_traders == []
        assert s.last_check_time == 0.0
        assert s.known_trades == {}

    def test_params_override_defaults(self):
        s = _make_strategy(
            source="wallet_monitor",
            wallet_addresses=["0xabc"],
            max_copy_amount_usdt=100.0,
            delay_seconds=10,
            exchange="kraken",
            tracked_traders=["trader1"],
        )
        assert s.source == "wallet_monitor"
        assert s.wallet_addresses == ["0xabc"]
        assert s.max_copy_amount == pytest.approx(100.0)
        assert s.delay_seconds == 10
        assert s.exchange == "kraken"
        assert s.tracked_traders == ["trader1"]


class TestOnStart:
    def test_keeps_explicit_exchange(self):
        s = _make_strategy(exchange="kraken")
        s.on_start()
        assert s.exchange == "kraken"
        assert s.active is True

    def test_falls_back_to_first_available_exchange(self):
        s = _make_strategy()
        s.exchange_manager.available_exchanges = ["binance", "kraken"]
        s.on_start()
        assert s.exchange == "binance"
        assert s.active is True

    def test_deactivates_when_no_exchange_available(self):
        s = _make_strategy()
        s.exchange_manager.available_exchanges = []
        s.on_start()
        assert s.exchange == ""
        assert s.active is False


class TestEvaluateRateLimit:
    def test_inactive_strategy_returns_no_signals(self):
        s = _make_strategy(exchange="binance")
        s.active = False
        assert s.evaluate() == []

    def test_first_call_after_start_runs_check(self, monkeypatch):
        s = _make_strategy(exchange="binance", source="binance_leaderboard")
        s.on_start()
        monkeypatch.setattr("strategies.copy_trading.time.time", lambda: 1_000_000.0)
        s.evaluate()
        assert s.last_check_time == pytest.approx(1_000_000.0)

    def test_second_call_within_interval_is_skipped(self, monkeypatch):
        s = _make_strategy(exchange="binance", delay_seconds=5)
        s.on_start()
        clock = {"t": 1_000_000.0}
        monkeypatch.setattr("strategies.copy_trading.time.time", lambda: clock["t"])
        s.evaluate()
        first_check = s.last_check_time
        clock["t"] += 10  # interval floors at 30s, so 10s later is still too soon
        result = s.evaluate()
        assert result == []
        assert s.last_check_time == first_check

    def test_call_after_interval_elapses_runs_again(self, monkeypatch):
        s = _make_strategy(exchange="binance", delay_seconds=5)
        s.on_start()
        clock = {"t": 1_000_000.0}
        monkeypatch.setattr("strategies.copy_trading.time.time", lambda: clock["t"])
        s.evaluate()
        clock["t"] += 31  # past the 30s floor
        s.evaluate()
        assert s.last_check_time == pytest.approx(1_000_031.0)

    def test_delay_seconds_above_30_is_honored(self, monkeypatch):
        s = _make_strategy(exchange="binance", delay_seconds=60)
        s.on_start()
        clock = {"t": 1_000_000.0}
        monkeypatch.setattr("strategies.copy_trading.time.time", lambda: clock["t"])
        s.evaluate()
        clock["t"] += 45  # past the 30s floor but before the configured 60s
        result = s.evaluate()
        assert result == []
        assert s.last_check_time == pytest.approx(1_000_000.0)


class TestSourceDispatch:
    def test_binance_leaderboard_source_is_checked(self, monkeypatch):
        s = _make_strategy(exchange="binance", source="binance_leaderboard")
        s.on_start()
        monkeypatch.setattr("strategies.copy_trading.time.time", lambda: 2_000_000.0)
        called = {"leaderboard": False, "wallets": False}
        s._check_leaderboard = lambda: called.__setitem__("leaderboard", True) or []
        s._check_wallets = lambda: called.__setitem__("wallets", True) or []
        s.evaluate()
        assert called == {"leaderboard": True, "wallets": False}

    def test_wallet_monitor_source_is_checked(self, monkeypatch):
        s = _make_strategy(exchange="binance", source="wallet_monitor")
        s.on_start()
        monkeypatch.setattr("strategies.copy_trading.time.time", lambda: 2_000_000.0)
        called = {"leaderboard": False, "wallets": False}
        s._check_leaderboard = lambda: called.__setitem__("leaderboard", True) or []
        s._check_wallets = lambda: called.__setitem__("wallets", True) or []
        s.evaluate()
        assert called == {"leaderboard": False, "wallets": True}

    def test_unknown_source_checks_neither(self, monkeypatch):
        s = _make_strategy(exchange="binance", source="something_else")
        s.on_start()
        monkeypatch.setattr("strategies.copy_trading.time.time", lambda: 2_000_000.0)
        called = {"leaderboard": False, "wallets": False}
        s._check_leaderboard = lambda: called.__setitem__("leaderboard", True) or []
        s._check_wallets = lambda: called.__setitem__("wallets", True) or []
        assert s.evaluate() == []
        assert called == {"leaderboard": False, "wallets": False}


class TestFrameworkStubs:
    def test_check_leaderboard_always_empty(self):
        s = _make_strategy(tracked_traders=["a", "b"])
        assert s._check_leaderboard() == []

    def test_check_wallets_always_empty(self):
        s = _make_strategy(wallet_addresses=["0xabc"])
        assert s._check_wallets() == []


class TestCreateCopySignal:
    def test_caps_amount_at_max_copy_amount(self):
        s = _make_strategy(exchange="binance", max_copy_amount_usdt=25.0)
        s.exchange_manager.get_ticker.return_value = {"last": 100.0}
        sig = s._create_copy_signal("whale1", "BTC/USDT", "buy", source_amount_usdt=1000.0)
        assert sig["amount"] == pytest.approx(25.0 / 100.0)
        assert "25.00 USDT" in sig["reason"]
        assert "1000.00 USDT" in sig["reason"]

    def test_uses_source_amount_when_below_cap(self):
        s = _make_strategy(exchange="binance", max_copy_amount_usdt=25.0)
        s.exchange_manager.get_ticker.return_value = {"last": 50.0}
        sig = s._create_copy_signal("whale1", "ETH/USDT", "sell", source_amount_usdt=10.0)
        assert sig["amount"] == pytest.approx(10.0 / 50.0)

    def test_signal_shape(self):
        s = _make_strategy(exchange="binance")
        s.exchange_manager.get_ticker.return_value = {"last": 100.0}
        sig = s._create_copy_signal("whale1", "BTC/USDT", "buy", source_amount_usdt=10.0)
        assert sig["symbol"] == "BTC/USDT"
        assert sig["side"] == "buy"
        assert sig["price"] is None
        assert sig["order_type"] == "market"
        assert sig["exchange"] == "binance"
        assert sig["copy_source"] == "whale1"
        assert sig["amount"] == round(10.0 / 100.0, 8)

    def test_zero_price_returns_empty_dict(self):
        s = _make_strategy(exchange="binance")
        s.exchange_manager.get_ticker.return_value = {"last": 0}
        assert s._create_copy_signal("whale1", "BTC/USDT", "buy", source_amount_usdt=10.0) == {}

    def test_missing_last_price_returns_empty_dict(self):
        s = _make_strategy(exchange="binance")
        s.exchange_manager.get_ticker.return_value = {}
        assert s._create_copy_signal("whale1", "BTC/USDT", "buy", source_amount_usdt=10.0) == {}

    def test_ticker_exception_returns_empty_dict(self):
        s = _make_strategy(exchange="binance")
        s.exchange_manager.get_ticker.side_effect = RuntimeError("network down")
        assert s._create_copy_signal("whale1", "BTC/USDT", "buy", source_amount_usdt=10.0) == {}
