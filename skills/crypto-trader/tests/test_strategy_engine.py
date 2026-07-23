"""Tests for the Strategy Engine module."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategy_engine import BaseStrategy, StrategyEngine


class MockStrategy(BaseStrategy):
    name = "mock"
    display_name = "Mock Strategy"

    def evaluate(self):
        return [{"symbol": "BTC/USDT", "side": "buy", "amount": 0.001, "reason": "test"}]


class FailingStrategy(BaseStrategy):
    name = "failing"
    display_name = "Failing Strategy"

    def evaluate(self):
        raise RuntimeError("boom")


class TestStrategyEngine:
    @pytest.fixture
    def state_path(self, tmp_path):
        return str(tmp_path / "strategies.json")

    @pytest.fixture
    def engine(self, state_path):
        with patch.dict(os.environ, {"CRYPTO_STRATEGY_STATE_PATH": state_path}):
            exchange_mgr = MagicMock()
            exchange_mgr.available_exchanges = ["binance"]
            risk_mgr = MagicMock()
            engine = StrategyEngine(exchange_mgr, risk_mgr)
            engine.register_strategy(MockStrategy)
            return engine

    def _make_engine(self, state_path, strategy_classes=(MockStrategy,)):
        """Build a fresh StrategyEngine instance sharing the same state file.

        Used to simulate two processes (e.g. CLI + monitor daemon) reading and
        writing the same persisted strategy state.
        """
        with patch.dict(os.environ, {"CRYPTO_STRATEGY_STATE_PATH": state_path}):
            exchange_mgr = MagicMock()
            exchange_mgr.available_exchanges = ["binance"]
            risk_mgr = MagicMock()
            eng = StrategyEngine(exchange_mgr, risk_mgr)
            for cls in strategy_classes:
                eng.register_strategy(cls)
            return eng

    def test_register_strategy(self, engine):
        assert "mock" in engine.get_available_strategies()

    def test_start_strategy(self, engine):
        result = engine.start_strategy("mock", {"symbol": "ETH/USDT"})
        assert result["status"] == "ok"
        assert result["strategy_name"] == "mock"
        assert "strategy_id" in result

    def test_start_unknown_strategy(self, engine):
        result = engine.start_strategy("nonexistent")
        assert result["status"] == "error"

    def test_stop_strategy(self, engine):
        start = engine.start_strategy("mock")
        sid = start["strategy_id"]
        stop = engine.stop_strategy(sid)
        assert stop["status"] == "ok"

    def test_stop_nonexistent_strategy(self, engine):
        result = engine.stop_strategy("fake_id")
        assert result["status"] == "error"

    def test_list_strategies(self, engine):
        engine.start_strategy("mock")
        strategies = engine.list_strategies()
        assert len(strategies) == 1
        assert strategies[0]["name"] == "mock"

    def test_evaluate_all(self, engine):
        engine.start_strategy("mock")
        signals = engine.evaluate_all()
        assert len(signals) >= 1
        assert signals[0]["symbol"] == "BTC/USDT"

    def test_stop_all(self, engine):
        engine.start_strategy("mock")
        engine.start_strategy("mock")
        results = engine.stop_all()
        assert len(results) == 2
        assert engine.list_strategies() == []

    def test_start_strategy_disabled_in_config(self, engine):
        engine._config["mock"] = {"enabled": False}
        result = engine.start_strategy("mock")
        assert result["status"] == "error"
        assert "disabled" in result["message"]
        assert engine.list_strategies() == []

    def test_evaluate_all_skips_inactive_strategy(self, engine):
        start = engine.start_strategy("mock")
        sid = start["strategy_id"]
        engine.get_strategy_instance(sid).active = False
        signals = engine.evaluate_all()
        assert signals == []

    def test_evaluate_all_handles_strategy_exception(self, engine):
        engine.register_strategy(FailingStrategy)
        engine.start_strategy("mock")
        engine.start_strategy("failing")
        # The failing strategy must not blow up evaluate_all(); the healthy
        # strategy's signal should still come through.
        signals = engine.evaluate_all()
        assert len(signals) == 1
        assert signals[0]["strategy_name"] == "mock"

    def test_get_strategy_returns_none_for_unknown(self, engine):
        assert engine.get_strategy("nonexistent") is None

    def test_get_strategy_instance_returns_none_for_unknown(self, engine):
        assert engine.get_strategy_instance("nonexistent") is None

    def test_load_config_missing_file_returns_empty_dict(self, state_path):
        with patch.dict(os.environ, {"CRYPTO_STRATEGY_STATE_PATH": state_path}), \
                patch("strategy_engine._CONFIG_DIR", Path("/nonexistent/config/dir")):
            exchange_mgr = MagicMock()
            exchange_mgr.available_exchanges = ["binance"]
            eng = StrategyEngine(exchange_mgr, MagicMock())
            assert eng._config == {}

    def test_read_state_file_missing_returns_none(self, engine):
        assert engine._read_state_file() is None

    def test_read_state_file_corrupt_returns_none(self, engine, state_path):
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(state_path).write_text("{ not valid json", encoding="utf-8")
        assert engine._read_state_file() is None

    def test_load_state_restores_strategy_from_disk(self, state_path):
        # Process A starts a strategy and persists it.
        engine_a = self._make_engine(state_path)
        start = engine_a.start_strategy("mock", {"symbol": "ETH/USDT"})
        sid = start["strategy_id"]

        # Process B starts fresh (e.g. after a restart) and restores state.
        engine_b = self._make_engine(state_path)
        restored = engine_b.load_state()

        assert restored == 1
        got = engine_b.get_strategy(sid)
        assert got is not None
        assert got["params"]["symbol"] == "ETH/USDT"
        assert got["active"] is True

    def test_load_state_no_file_returns_zero(self, engine):
        assert engine.load_state() == 0

    def test_load_state_skips_strategy_already_in_memory(self, state_path):
        """A running in-memory strategy must survive load_state() untouched.

        Regression guard for the 'if sid in self._strategies: continue' check
        in load_state() -- without it, restoring from a stale state file
        would silently replace a live strategy instance (and its runtime
        state) with a fresh one built from the last-saved snapshot.
        """
        engine_a = self._make_engine(state_path)
        start = engine_a.start_strategy("mock")
        sid = start["strategy_id"]

        live_instance = engine_a.get_strategy_instance(sid)
        live_instance.stats["total_pnl"] = 999.0  # unsaved in-memory mutation

        engine_a.load_state()

        assert engine_a.get_strategy_instance(sid) is live_instance
        assert engine_a.get_strategy_instance(sid).stats["total_pnl"] == 999.0

    def test_load_state_skips_unregistered_strategy_class(self, state_path):
        engine_a = self._make_engine(state_path)
        engine_a.start_strategy("mock")

        # Process B never registered the "mock" strategy class.
        engine_b = self._make_engine(state_path, strategy_classes=())
        restored = engine_b.load_state()

        assert restored == 0
        assert engine_b.list_strategies() == []

    def test_sync_from_disk_removes_strategy_stopped_elsewhere(self, state_path):
        engine_a = self._make_engine(state_path)
        start = engine_a.start_strategy("mock")
        sid = start["strategy_id"]

        engine_b = self._make_engine(state_path)
        engine_b.load_state()
        assert engine_b.get_strategy(sid) is not None

        # Process A stops the strategy and persists the change.
        engine_a.stop_strategy(sid)

        # Process B reconciles and should drop the now-stopped strategy.
        engine_b.sync_from_disk()
        assert engine_b.get_strategy(sid) is None

    def test_sync_from_disk_adds_strategy_started_elsewhere(self, state_path):
        engine_a = self._make_engine(state_path)
        engine_b = self._make_engine(state_path)
        assert engine_b.list_strategies() == []

        start = engine_a.start_strategy("mock")
        sid = start["strategy_id"]

        engine_b.sync_from_disk()
        assert engine_b.get_strategy(sid) is not None
