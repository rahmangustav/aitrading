"""Tests for the Risk Manager module."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from risk_manager import RiskManager, RiskLimitExceeded


@pytest.fixture
def risk_config(tmp_path):
    """Create a temporary risk config file."""
    config = {
        "global_limits": {
            "max_position_size_pct": 25.0,
            "max_daily_loss_eur": 50.0,
            "max_drawdown_pct": 15.0,
            "max_order_size_eur": 100.0,
            "max_open_orders": 50,
            "min_cash_reserve_pct": 10.0,
            "emergency_stop_loss_pct": 20.0,
        },
        "strategy_overrides": {
            "grid_trading": {
                "max_order_size_eur": 50.0,
                "max_open_orders": 20,
            },
        },
        "stop_loss": {
            "enabled": True,
            "default_pct": 5.0,
            "trailing_enabled": True,
            "trailing_pct": 3.0,
        },
        "take_profit": {
            "enabled": True,
            "default_pct": 10.0,
            "partial_exit_enabled": True,
            "partial_exit_pct": 50.0,
            "partial_exit_trigger_pct": 5.0,
        },
    }
    config_path = tmp_path / "risk_limits.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def risk_manager(risk_config, tmp_path):
    """Create a RiskManager with temporary state."""
    state_path = str(tmp_path / "risk-state.json")
    with patch.dict(os.environ, {"CRYPTO_RISK_STATE_PATH": state_path}):
        return RiskManager(config_path=risk_config)


def _make_risk_manager(tmp_path, config, state=None, state_name="risk-state.json"):
    """Build a RiskManager from an arbitrary config dict, optionally with a
    pre-seeded state file (written as raw text so callers can also inject
    corrupt/malformed content)."""
    import yaml

    config_path = tmp_path / f"{state_name}.config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    state_path = tmp_path / state_name
    if state is not None:
        if isinstance(state, str):
            state_path.write_text(state)
        else:
            state_path.write_text(json.dumps(state))

    with patch.dict(os.environ, {"CRYPTO_RISK_STATE_PATH": str(state_path)}):
        return RiskManager(config_path=str(config_path))


class TestValidateOrder:
    def test_valid_order_passes(self, risk_manager):
        risk_manager.validate_order(
            strategy="grid_trading",
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            amount=0.001,
            price=40000,
            portfolio_value_eur=1000,
            open_order_count=0,
        )

    def test_max_order_size_global(self, risk_manager):
        with pytest.raises(RiskLimitExceeded, match="max_order_size"):
            risk_manager.validate_order(
                strategy="dca",
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=1.0,
                price=200,
                portfolio_value_eur=10000,
                open_order_count=0,
            )

    def test_max_order_size_strategy_override(self, risk_manager):
        with pytest.raises(RiskLimitExceeded, match="max_order_size"):
            risk_manager.validate_order(
                strategy="grid_trading",
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=1.0,
                price=60,
                portfolio_value_eur=10000,
                open_order_count=0,
            )

    def test_max_open_orders(self, risk_manager):
        with pytest.raises(RiskLimitExceeded, match="max_open_orders"):
            risk_manager.validate_order(
                strategy="grid_trading",
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=0.001,
                price=40,
                portfolio_value_eur=1000,
                open_order_count=20,
            )

    def test_max_position_size(self, risk_manager):
        with pytest.raises(RiskLimitExceeded, match="max_position_size"):
            risk_manager.validate_order(
                strategy=None,
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=1.0,
                price=30,
                portfolio_value_eur=100,
                open_order_count=0,
            )

    def test_kill_switch_blocks_orders(self, risk_manager):
        risk_manager.activate_kill_switch("test")
        with pytest.raises(RiskLimitExceeded, match="kill_switch"):
            risk_manager.validate_order(
                strategy=None,
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=0.001,
                price=40,
                portfolio_value_eur=1000,
                open_order_count=0,
            )

    def test_missing_price_blocks_order(self, risk_manager):
        with pytest.raises(RiskLimitExceeded, match="unknown_order_cost"):
            risk_manager.validate_order(
                strategy=None,
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=0.001,
                price=None,
                portfolio_value_eur=1000,
                open_order_count=0,
            )

    def test_zero_price_blocks_order(self, risk_manager):
        with pytest.raises(RiskLimitExceeded, match="unknown_order_cost"):
            risk_manager.validate_order(
                strategy=None,
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=0.001,
                price=0,
                portfolio_value_eur=1000,
                open_order_count=0,
            )

    def test_max_daily_loss_blocks_orders(self, risk_manager):
        # Fixture config caps max_daily_loss_eur at 50.0.
        risk_manager.record_trade({
            "symbol": "BTC/USDT", "side": "sell", "amount": 0.01,
            "price": 40000, "realized_pnl_eur": -60.0,
        })
        with pytest.raises(RiskLimitExceeded, match="max_daily_loss"):
            risk_manager.validate_order(
                strategy=None,
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=0.0001,
                price=40,
                portfolio_value_eur=1000,
                open_order_count=0,
            )

    def test_max_daily_loss_not_yet_reached_passes(self, risk_manager):
        risk_manager.record_trade({
            "symbol": "BTC/USDT", "side": "sell", "amount": 0.01,
            "price": 40000, "realized_pnl_eur": -10.0,
        })
        risk_manager.validate_order(
            strategy=None,
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            amount=0.0001,
            price=40,
            portfolio_value_eur=1000,
            open_order_count=0,
        )

    def test_zero_ath_skips_drawdown_check(self, risk_manager):
        # No portfolio value has ever been recorded (ATH stays 0) -- the
        # drawdown gate must not divide by zero or block the order.
        risk_manager.validate_order(
            strategy=None,
            exchange="binance",
            symbol="BTC/USDT",
            side="sell",
            amount=0.0001,
            price=40,
            portfolio_value_eur=0.0,
            open_order_count=0,
        )

    def test_drawdown_beyond_limit_blocks_orders(self, risk_manager):
        # Fixture config caps max_drawdown_pct at 15.0. Establish an ATH,
        # then validate an order against a portfolio that has fallen well
        # past that drawdown from the peak.
        risk_manager.update_portfolio_value(1000.0)
        with pytest.raises(RiskLimitExceeded, match="max_drawdown"):
            risk_manager.validate_order(
                strategy=None,
                exchange="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=0.0001,
                price=40,
                portfolio_value_eur=800.0,
                open_order_count=0,
            )


class TestStopLoss:
    def test_stop_loss_triggers(self, risk_manager):
        assert risk_manager.check_stop_loss(entry_price=100, current_price=94) is True

    def test_stop_loss_does_not_trigger(self, risk_manager):
        assert risk_manager.check_stop_loss(entry_price=100, current_price=97) is False

    def test_stop_loss_zero_entry_price_returns_false_not_zerodivisionerror(self, risk_manager):
        # A strategy can end up with entry_price == 0 if the exchange's order-fill
        # response was missing average/price/cost (see swing_trading.on_order_filled).
        # This must not crash the strategy's evaluate() cycle every time it runs.
        assert risk_manager.check_stop_loss(entry_price=0, current_price=94) is False

    def test_stop_loss_negative_entry_price_returns_false(self, risk_manager):
        assert risk_manager.check_stop_loss(entry_price=-5, current_price=94) is False

    def test_trailing_stop_triggers(self, risk_manager):
        assert risk_manager.check_trailing_stop(highest_price=110, current_price=106) is True

    def test_trailing_stop_does_not_trigger(self, risk_manager):
        assert risk_manager.check_trailing_stop(highest_price=110, current_price=108) is False

    def test_trailing_stop_zero_highest_price_returns_false(self, risk_manager):
        assert risk_manager.check_trailing_stop(highest_price=0, current_price=106) is False

    def test_check_stop_loss_disabled_never_triggers(self, tmp_path):
        rm = _make_risk_manager(tmp_path, {"stop_loss": {"enabled": False}})
        # A price crash that would trivially trigger a 5%-default stop.
        assert rm.check_stop_loss(entry_price=100, current_price=50) is False

    def test_stop_loss_price_disabled_returns_none(self, tmp_path):
        rm = _make_risk_manager(tmp_path, {"stop_loss": {"enabled": False}})
        assert rm.stop_loss_price(100.0, side="buy") is None

    def test_stop_loss_price_zero_pct_returns_none(self, tmp_path):
        rm = _make_risk_manager(tmp_path, {"stop_loss": {"enabled": True, "default_pct": 5.0}})
        assert rm.stop_loss_price(100.0, side="buy", custom_pct=0) is None

    def test_stop_loss_price_zero_entry_returns_none(self, tmp_path):
        rm = _make_risk_manager(tmp_path, {"stop_loss": {"enabled": True, "default_pct": 5.0}})
        assert rm.stop_loss_price(0.0, side="buy") is None

    def test_check_trailing_stop_disabled_never_triggers(self, tmp_path):
        rm = _make_risk_manager(tmp_path, {"stop_loss": {"trailing_enabled": False}})
        # A sharp drop from the high that would trivially trigger a 3%-default trail.
        assert rm.check_trailing_stop(highest_price=100, current_price=50) is False


class TestTieredTrailingStop:
    def test_non_positive_entry_price_returns_prev_stop_unchanged(self, risk_manager):
        assert risk_manager.tiered_trailing_stop(0.0, 105.0, prev_stop=99.0) == 99.0
        assert risk_manager.tiered_trailing_stop(-10.0, 105.0, prev_stop=None) is None


class TestTakeProfit:
    def test_take_profit_triggers(self, risk_manager):
        assert risk_manager.check_take_profit(entry_price=100, current_price=111) is True

    def test_take_profit_does_not_trigger(self, risk_manager):
        assert risk_manager.check_take_profit(entry_price=100, current_price=108) is False

    def test_take_profit_zero_entry_price_returns_false_not_zerodivisionerror(self, risk_manager):
        assert risk_manager.check_take_profit(entry_price=0, current_price=111) is False

    def test_partial_take_profit(self, risk_manager):
        fraction = risk_manager.check_partial_take_profit(entry_price=100, current_price=106)
        assert fraction == 0.5

    def test_partial_take_profit_zero_entry_price_returns_none(self, risk_manager):
        assert risk_manager.check_partial_take_profit(entry_price=0, current_price=106) is None

    def test_partial_take_profit_below_trigger_returns_none(self, risk_manager):
        assert risk_manager.check_partial_take_profit(entry_price=100, current_price=102) is None

    def test_check_take_profit_disabled_never_triggers(self, tmp_path):
        rm = _make_risk_manager(tmp_path, {"take_profit": {"enabled": False}})
        # A huge gain that would trivially trigger a 10%-default take-profit.
        assert rm.check_take_profit(entry_price=100, current_price=200) is False

    def test_partial_take_profit_disabled_returns_none(self, tmp_path):
        rm = _make_risk_manager(tmp_path, {"take_profit": {"partial_exit_enabled": False}})
        assert rm.check_partial_take_profit(entry_price=100, current_price=200) is None


class TestKillSwitch:
    def test_activate_and_deactivate(self, risk_manager):
        assert risk_manager.is_killed is False
        risk_manager.activate_kill_switch("test")
        assert risk_manager.is_killed is True
        risk_manager.deactivate_kill_switch()
        assert risk_manager.is_killed is False


class TestStatus:
    def test_get_status_returns_dict(self, risk_manager):
        status = risk_manager.get_status()
        assert "daily_pnl_eur" in status
        assert "drawdown_pct" in status
        assert "kill_switch_active" in status
        assert "limits" in status

    def test_get_status_computes_drawdown_from_ath(self, risk_manager):
        risk_manager.update_portfolio_value(1000.0)
        risk_manager.update_portfolio_value(900.0)
        status = risk_manager.get_status()
        assert status["portfolio_ath_eur"] == 1000.0
        assert status["current_portfolio_eur"] == 900.0
        assert status["drawdown_pct"] == pytest.approx(10.0)

    def test_get_status_zero_drawdown_before_any_portfolio_value(self, risk_manager):
        status = risk_manager.get_status()
        assert status["drawdown_pct"] == 0.0


class TestUpdatePortfolioValue:
    def test_tracks_current_value_and_raises_ath(self, risk_manager):
        risk_manager.update_portfolio_value(500.0)
        status = risk_manager.get_status()
        assert status["current_portfolio_eur"] == 500.0
        assert status["portfolio_ath_eur"] == 500.0

    def test_does_not_lower_ath_on_a_dip(self, risk_manager):
        risk_manager.update_portfolio_value(1000.0)
        risk_manager.update_portfolio_value(600.0)
        status = risk_manager.get_status()
        assert status["current_portfolio_eur"] == 600.0
        assert status["portfolio_ath_eur"] == 1000.0


class TestLoadConfig:
    def test_missing_config_file_falls_back_to_permissive_defaults(self, tmp_path):
        state_path = tmp_path / "risk-state.json"
        missing_config = tmp_path / "does-not-exist.yaml"
        with patch.dict(os.environ, {"CRYPTO_RISK_STATE_PATH": str(state_path)}):
            rm = RiskManager(config_path=str(missing_config))
        # No limits configured at all -- an order that would blow past any
        # of the fixture's limits must still pass since nothing is set.
        rm.validate_order(
            strategy=None,
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            amount=1000.0,
            price=1000.0,
            portfolio_value_eur=1.0,
            open_order_count=999,
        )


class TestLoadState:
    def test_corrupt_state_file_resets_to_default(self, tmp_path):
        rm = _make_risk_manager(
            tmp_path,
            {"global_limits": {}, "stop_loss": {}, "take_profit": {}},
            state="{not valid json!!",
        )
        status = rm.get_status()
        assert status["daily_pnl_eur"] == 0.0
        assert status["trades_today_count"] == 0
        assert status["kill_switch_active"] is False


class TestResetDailyIfNeeded:
    def test_stale_date_resets_daily_counters_but_keeps_kill_switch(self, tmp_path):
        stale_state = {
            "date": "2000-01-01",
            "daily_pnl_eur": -999.0,
            "portfolio_ath_eur": 1000.0,
            "current_portfolio_eur": 1000.0,
            "open_order_count": 0,
            "trades_today": [{"symbol": "BTC/USDT"}],
            "killed": True,
        }
        rm = _make_risk_manager(
            tmp_path,
            {"global_limits": {}, "stop_loss": {}, "take_profit": {}},
            state=stale_state,
        )
        status = rm.get_status()
        assert status["date"] != "2000-01-01"
        assert status["daily_pnl_eur"] == 0.0
        assert status["trades_today_count"] == 0
        # The kill switch is a deliberate manual gate -- a new day must not
        # silently clear it.
        assert status["kill_switch_active"] is True
        # ATH/current portfolio tracking is not a "daily" counter and must survive.
        assert status["portfolio_ath_eur"] == 1000.0


class TestRecordTrade:
    def test_record_trade_updates_pnl(self, risk_manager):
        risk_manager.record_trade({
            "symbol": "BTC/USDT",
            "side": "sell",
            "amount": 0.01,
            "price": 50000,
            "realized_pnl_eur": 15.0,
        })
        status = risk_manager.get_status()
        assert status["daily_pnl_eur"] == 15.0
        assert status["trades_today_count"] == 1
