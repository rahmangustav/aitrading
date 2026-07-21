"""Tests for main.py -- the crypto-trader CLI entrypoint.

main.py is the single entrypoint used for every operation (status, balance,
start/stop strategy, backtest, monitor, emergency_stop, ...). It had zero
test coverage before this file: a mistake in its JSON error-handling or mode
dispatch would silently break every command built on top of it (including
the emergency_stop kill switch), and nothing would catch it.

These tests exercise the `_run_*` handlers directly with fake exchange/risk/
engine objects (the same style used by test_monitor_daemon_order_registry.py
and test_strategy_engine.py) so they don't need real exchange connectivity,
config files, or network access.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import main as main_mod  # noqa: E402
from exchange_manager import ExchangeError  # noqa: E402


# ----------------------------------------------------------------------
# _output / _error
# ----------------------------------------------------------------------

class TestOutputAndError:
    def test_output_writes_json_to_stdout(self, capsys):
        main_mod._output({"status": "ok", "value": 1})
        printed = json.loads(capsys.readouterr().out)
        assert printed == {"status": "ok", "value": 1}

    def test_error_writes_structured_json_and_exits_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main_mod._error("something broke", code="boom")

        assert exc_info.value.code == 1
        printed = json.loads(capsys.readouterr().out)
        assert printed == {
            "status": "error",
            "code": "boom",
            "message": "something broke",
        }

    def test_error_defaults_code_to_error(self, capsys):
        with pytest.raises(SystemExit):
            main_mod._error("oops")
        printed = json.loads(capsys.readouterr().out)
        assert printed["code"] == "error"


# ----------------------------------------------------------------------
# Mode: status
# ----------------------------------------------------------------------

class TestRunStatus:
    def _exchange_mgr(self, demo: bool):
        mgr = MagicMock()
        mgr.demo = demo
        mgr.available_exchanges = ["binance"]
        mgr.get_balance.return_value = {"USDT": {"total": 100.0}}
        return mgr

    def test_reports_paper_environment_when_demo(self, capsys):
        exchange_mgr = self._exchange_mgr(demo=True)
        risk_mgr = MagicMock()
        risk_mgr.get_status.return_value = {"kill_switch": False}
        engine = MagicMock()
        engine.list_strategies.return_value = []

        main_mod._run_status(exchange_mgr, risk_mgr, engine)

        result = json.loads(capsys.readouterr().out)
        assert result["environment"] == "paper"
        assert result["exchanges"]["binance"] == {
            "connected": True,
            "balances": {"USDT": {"total": 100.0}},
        }

    def test_reports_live_environment_when_not_demo(self, capsys):
        exchange_mgr = self._exchange_mgr(demo=False)
        risk_mgr = MagicMock()
        risk_mgr.get_status.return_value = {}
        engine = MagicMock()
        engine.list_strategies.return_value = []

        main_mod._run_status(exchange_mgr, risk_mgr, engine)

        result = json.loads(capsys.readouterr().out)
        assert result["environment"] == "LIVE"

    def test_exchange_error_is_reported_not_raised(self, capsys):
        exchange_mgr = self._exchange_mgr(demo=True)
        exchange_mgr.get_balance.side_effect = ExchangeError("binance", "rate limited")
        risk_mgr = MagicMock()
        risk_mgr.get_status.return_value = {}
        engine = MagicMock()
        engine.list_strategies.return_value = []

        main_mod._run_status(exchange_mgr, risk_mgr, engine)

        result = json.loads(capsys.readouterr().out)
        assert result["exchanges"]["binance"] == {
            "connected": False,
            "error": "[binance] rate limited",
        }


# ----------------------------------------------------------------------
# Mode: balance
# ----------------------------------------------------------------------

class TestRunBalance:
    def test_specific_exchange_only_queries_that_exchange(self, capsys):
        exchange_mgr = MagicMock()
        exchange_mgr.demo = True
        exchange_mgr.available_exchanges = ["binance", "kraken"]
        exchange_mgr.get_balance.return_value = {"USDT": {"total": 50.0}}

        main_mod._run_balance(exchange_mgr, exchange="binance")

        exchange_mgr.get_balance.assert_called_once_with("binance")
        result = json.loads(capsys.readouterr().out)
        assert list(result["balances"].keys()) == ["binance"]

    def test_no_exchange_arg_queries_all_available(self, capsys):
        exchange_mgr = MagicMock()
        exchange_mgr.demo = True
        exchange_mgr.available_exchanges = ["binance", "kraken"]
        exchange_mgr.get_balance.return_value = {}

        main_mod._run_balance(exchange_mgr, exchange=None)

        result = json.loads(capsys.readouterr().out)
        assert set(result["balances"].keys()) == {"binance", "kraken"}

    def test_exchange_error_reported_per_exchange(self, capsys):
        exchange_mgr = MagicMock()
        exchange_mgr.demo = True
        exchange_mgr.available_exchanges = ["binance"]
        exchange_mgr.get_balance.side_effect = ExchangeError("binance", "down")

        main_mod._run_balance(exchange_mgr, exchange=None)

        result = json.loads(capsys.readouterr().out)
        assert result["balances"]["binance"] == {"error": "[binance] down"}


# ----------------------------------------------------------------------
# Mode: start_strategy / stop_strategy / list_strategies
# ----------------------------------------------------------------------

class TestRunStartStopListStrategy:
    def test_invalid_json_params_reports_error_and_never_starts_strategy(self, capsys):
        engine = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            main_mod._run_start_strategy(engine, "grid_trading", params_json="{not json")

        assert exc_info.value.code == 1
        engine.start_strategy.assert_not_called()
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == "error"

    def test_valid_json_params_passed_through_to_engine(self, capsys):
        engine = MagicMock()
        engine.start_strategy.return_value = {"status": "ok", "strategy_id": "s1"}

        main_mod._run_start_strategy(engine, "grid_trading", params_json='{"symbol": "BTC/USDT"}')

        engine.start_strategy.assert_called_once_with(
            "grid_trading", {"symbol": "BTC/USDT"},
        )
        result = json.loads(capsys.readouterr().out)
        assert result["strategy_id"] == "s1"

    def test_no_params_starts_with_empty_dict(self, capsys):
        engine = MagicMock()
        engine.start_strategy.return_value = {"status": "ok"}

        main_mod._run_start_strategy(engine, "grid_trading", params_json=None)

        engine.start_strategy.assert_called_once_with("grid_trading", {})
        capsys.readouterr()

    def test_stop_strategy_delegates_to_engine(self, capsys):
        engine = MagicMock()
        engine.stop_strategy.return_value = {"status": "ok"}

        main_mod._run_stop_strategy(engine, "s1")

        engine.stop_strategy.assert_called_once_with("s1")
        capsys.readouterr()

    def test_list_strategies_reports_running_count(self, capsys):
        engine = MagicMock()
        engine.list_strategies.return_value = [{"id": "s1"}, {"id": "s2"}]
        engine.get_available_strategies.return_value = ["grid_trading", "dca"]

        main_mod._run_list_strategies(engine)

        result = json.loads(capsys.readouterr().out)
        assert result["running_count"] == 2
        assert result["available_strategies"] == ["grid_trading", "dca"]


# ----------------------------------------------------------------------
# Mode: history
# ----------------------------------------------------------------------

class TestRunHistory:
    def test_since_is_approximately_n_days_ago_in_ms(self, capsys):
        import time

        exchange_mgr = MagicMock()
        exchange_mgr.available_exchanges = ["binance"]
        exchange_mgr.get_order_history.return_value = []

        before_ms = int(time.time() * 1000)
        main_mod._run_history(exchange_mgr, days=7)
        after_ms = int(time.time() * 1000)

        _, kwargs = exchange_mgr.get_order_history.call_args
        since = kwargs["since"]
        seven_days_ms = 7 * 24 * 60 * 60 * 1000
        # since should land within the [before, after] window minus 7 days,
        # generous 5s slack for slow CI runners.
        assert (before_ms - seven_days_ms - 5000) <= since <= (after_ms - seven_days_ms + 5000)

    def test_exchange_error_reported_per_exchange(self, capsys):
        exchange_mgr = MagicMock()
        exchange_mgr.available_exchanges = ["binance"]
        exchange_mgr.get_order_history.side_effect = ExchangeError("binance", "nope")

        main_mod._run_history(exchange_mgr, days=1)

        result = json.loads(capsys.readouterr().out)
        assert result["orders"]["binance"] == {"error": "[binance] nope"}


# ----------------------------------------------------------------------
# Mode: emergency_stop
# ----------------------------------------------------------------------

class TestRunEmergencyStop:
    def test_kill_switch_activated_even_when_cancel_fails(self, capsys):
        """The kill switch is the last line of defense -- it must still
        activate even if cancelling orders on an exchange errors out."""
        exchange_mgr = MagicMock()
        exchange_mgr.available_exchanges = ["binance"]
        exchange_mgr.cancel_all_orders.side_effect = ExchangeError("binance", "network down")
        risk_mgr = MagicMock()
        engine = MagicMock()
        engine.stop_all.return_value = ["s1", "s2"]

        main_mod._run_emergency_stop(exchange_mgr, risk_mgr, engine)

        risk_mgr.activate_kill_switch.assert_called_once_with(reason="emergency_stop")
        result = json.loads(capsys.readouterr().out)
        assert result["kill_switch"] == "activated"
        assert result["orders_cancelled"]["binance"] == {"error": "[binance] network down"}
        assert result["strategies_stopped"] == ["s1", "s2"]

    def test_all_strategies_stopped_and_orders_cancelled_on_success(self, capsys):
        exchange_mgr = MagicMock()
        exchange_mgr.available_exchanges = ["binance"]
        exchange_mgr.cancel_all_orders.return_value = ["o1", "o2"]
        risk_mgr = MagicMock()
        engine = MagicMock()
        engine.stop_all.return_value = ["s1"]

        main_mod._run_emergency_stop(exchange_mgr, risk_mgr, engine)

        result = json.loads(capsys.readouterr().out)
        assert result["orders_cancelled"]["binance"] == ["o1", "o2"]
        assert result["status"] == "ok"
