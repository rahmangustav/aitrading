"""Tests for the Trend Following strategy.

This is the strategy that actually generates every signal currently sitting
in ct_signal_db.json (see PR #11 in this repo), yet it had zero direct unit
tests before this file -- only the standalone backtest replica (tf_backtest.py)
was covered. These tests exercise the live evaluate()/on_order_filled() logic
that real trading depends on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.trend_following import (  # noqa: E402
    TrendFollowingStrategy,
    _calculate_ema,
    _calculate_rsi,
)

MODULE = "strategies.trend_following"


def _make_ohlcv(n: int, last_close: float = 100.0):
    """Build n synthetic candles; only the last close price is meaningful
    since evaluate()'s EMA/RSI computation is patched out in most tests."""
    candles = []
    for i in range(n):
        close = last_close if i == n - 1 else 50.0 + i * 0.01
        candles.append([i * 14_400_000, close, close + 1, close - 1, close, 1000.0])
    return candles


def _tail_series(n: int, prev: float, current: float) -> pd.Series:
    """A length-n series where only the last two values (what evaluate()
    actually reads) are set; everything else is filler."""
    values = [0.0] * n
    values[-2] = prev
    values[-1] = current
    return pd.Series(values)


def _flat_series(n: int, value: float) -> pd.Series:
    return pd.Series([value] * n)


class TestCalculateEMA:
    def test_ema_converges_toward_recent_values(self):
        series = pd.Series([10.0] * 20)
        ema = _calculate_ema(series, 5)
        assert ema.iloc[-1] == pytest.approx(10.0)

    def test_ema_reacts_to_a_jump(self):
        series = pd.Series([10.0] * 10 + [20.0])
        ema = _calculate_ema(series, 5)
        assert 10.0 < ema.iloc[-1] < 20.0


class TestCalculateRSI:
    def test_pure_gains_give_rsi_100_not_0(self):
        series = pd.Series([float(x) for x in range(1, 20)])
        rsi = _calculate_rsi(series, 14)
        assert rsi.iloc[-1] == pytest.approx(100.0)

    def test_pure_losses_give_rsi_0(self):
        series = pd.Series([float(x) for x in range(20, 1, -1)])
        rsi = _calculate_rsi(series, 14)
        assert rsi.iloc[-1] == pytest.approx(0.0)

    def test_flat_series_gives_neutral_50(self):
        series = pd.Series([5.0] * 20)
        rsi = _calculate_rsi(series, 14)
        assert rsi.iloc[-1] == pytest.approx(50.0)


class TestOnStart:
    def _strategy(self, available_exchanges):
        exchange_mgr = MagicMock()
        exchange_mgr.available_exchanges = available_exchanges
        risk_mgr = MagicMock()
        return TrendFollowingStrategy(
            "tf_test", {"symbol": "BTC/USDT"}, exchange_mgr, risk_mgr,
        )

    def test_auto_selects_first_available_exchange(self):
        strat = self._strategy(["binance", "kraken"])
        strat.on_start()
        assert strat.exchange == "binance"
        assert strat.active is True

    def test_deactivates_when_no_exchange_available(self):
        strat = self._strategy([])
        strat.on_start()
        assert strat.active is False

    def test_keeps_explicitly_configured_exchange(self):
        exchange_mgr = MagicMock()
        exchange_mgr.available_exchanges = ["binance"]
        risk_mgr = MagicMock()
        strat = TrendFollowingStrategy(
            "tf_test", {"symbol": "BTC/USDT", "exchange": "kraken"}, exchange_mgr, risk_mgr,
        )
        strat.on_start()
        assert strat.exchange == "kraken"


@pytest.fixture
def strategy():
    exchange_mgr = MagicMock()
    exchange_mgr.available_exchanges = ["binance"]
    risk_mgr = MagicMock()
    risk_mgr.check_stop_loss.return_value = False
    risk_mgr.check_trailing_stop.return_value = False
    risk_mgr.check_take_profit.return_value = False

    params = {
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "ema_short": 9,
        "ema_long": 21,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "order_amount_usdt": 100.0,
        "exchange": "binance",
    }
    strat = TrendFollowingStrategy("tf_test", params, exchange_mgr, risk_mgr)
    strat.on_start()
    return strat


class TestEvaluateGuards:
    def test_inactive_strategy_returns_no_signals(self, strategy):
        strategy.active = False
        assert strategy.evaluate() == []

    def test_not_enough_candles_returns_no_signals(self, strategy):
        strategy.exchange_manager.get_ohlcv.return_value = _make_ohlcv(10)
        assert strategy.evaluate() == []

    def test_exchange_error_returns_no_signals(self, strategy):
        strategy.exchange_manager.get_ohlcv.side_effect = Exception("network down")
        assert strategy.evaluate() == []


class TestEvaluateCrossSignals:
    N = 40

    def _run(self, strategy, ema_short, ema_long, rsi, last_close=100.0):
        strategy.exchange_manager.get_ohlcv.return_value = _make_ohlcv(self.N, last_close)
        with patch(f"{MODULE}._calculate_ema") as mock_ema, \
             patch(f"{MODULE}._calculate_rsi") as mock_rsi:
            mock_ema.side_effect = [ema_short, ema_long]
            mock_rsi.return_value = rsi
            return strategy.evaluate()

    def test_bullish_cross_opens_long(self, strategy):
        ema_short = _tail_series(self.N, prev=9.0, current=11.0)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 50.0)

        signals = self._run(strategy, ema_short, ema_long, rsi, last_close=100.0)

        assert len(signals) == 1
        assert signals[0]["side"] == "buy"
        assert signals[0]["amount"] == pytest.approx(1.0)  # 100 USDT / 100 price
        assert strategy.last_signal == "buy"

    def test_bullish_cross_ignored_when_rsi_overbought(self, strategy):
        ema_short = _tail_series(self.N, prev=9.0, current=11.0)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 75.0)  # >= rsi_overbought (70)

        signals = self._run(strategy, ema_short, ema_long, rsi)

        assert signals == []

    def test_bullish_cross_does_not_duplicate_existing_long(self, strategy):
        strategy.position = "long"
        ema_short = _tail_series(self.N, prev=9.0, current=11.0)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 50.0)

        signals = self._run(strategy, ema_short, ema_long, rsi)

        assert signals == []

    def test_bearish_cross_sells_actual_position_not_recomputed_amount(self, strategy):
        # Regression: a long was opened at a much lower price, then price
        # rallied hard before the bearish cross. The exit must sell exactly
        # what was bought (7.3), not order_amount_usdt / current_price.
        strategy.position = "long"
        strategy.position_amount = 7.3
        strategy.entry_price = 10.0

        ema_short = _tail_series(self.N, prev=11.0, current=9.0)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 50.0)

        signals = self._run(strategy, ema_short, ema_long, rsi, last_close=250.0)

        assert len(signals) == 1
        assert signals[0]["side"] == "sell"
        assert signals[0]["amount"] == pytest.approx(7.3)
        assert strategy.last_signal == "sell"

    def test_bearish_cross_without_open_position_is_a_no_op(self, strategy):
        ema_short = _tail_series(self.N, prev=11.0, current=9.0)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 50.0)

        signals = self._run(strategy, ema_short, ema_long, rsi)

        assert signals == []

    def test_rsi_overbought_alone_closes_long_without_bearish_cross(self, strategy):
        strategy.position = "long"
        strategy.position_amount = 2.0

        # No cross: short stays above long both before and after.
        ema_short = _tail_series(self.N, prev=12.0, current=12.5)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 80.0)

        signals = self._run(strategy, ema_short, ema_long, rsi)

        assert len(signals) == 1
        assert signals[0]["side"] == "sell"
        assert signals[0]["amount"] == pytest.approx(2.0)

    def test_no_cross_and_rsi_neutral_produces_no_signal(self, strategy):
        strategy.position = "long"
        ema_short = _tail_series(self.N, prev=12.0, current=12.5)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 50.0)

        signals = self._run(strategy, ema_short, ema_long, rsi)

        assert signals == []


class TestEvaluateRiskExits:
    N = 40

    def _run(self, strategy, last_close=100.0):
        strategy.exchange_manager.get_ohlcv.return_value = _make_ohlcv(self.N, last_close)
        # Flat, non-crossing EMAs and neutral RSI so only the risk-manager
        # checks below can produce a signal.
        ema_short = _tail_series(self.N, prev=12.0, current=12.0)
        ema_long = _tail_series(self.N, prev=10.0, current=10.0)
        rsi = _flat_series(self.N, 50.0)
        with patch(f"{MODULE}._calculate_ema") as mock_ema, \
             patch(f"{MODULE}._calculate_rsi") as mock_rsi:
            mock_ema.side_effect = [ema_short, ema_long]
            mock_rsi.return_value = rsi
            return strategy.evaluate()

    def test_stop_loss_triggers_sell_of_actual_position(self, strategy):
        strategy.position = "long"
        strategy.position_amount = 3.0
        strategy.entry_price = 100.0
        strategy.risk_manager.check_stop_loss.return_value = True

        signals = self._run(strategy, last_close=94.0)

        assert len(signals) == 1
        assert signals[0]["side"] == "sell"
        assert signals[0]["amount"] == pytest.approx(3.0)
        assert "Stop-loss" in signals[0]["reason"]

    def test_trailing_stop_triggers_sell(self, strategy):
        strategy.position = "long"
        strategy.position_amount = 3.0
        strategy.entry_price = 100.0
        strategy.highest_since_entry = 120.0
        strategy.risk_manager.check_trailing_stop.return_value = True

        signals = self._run(strategy, last_close=115.0)

        assert len(signals) == 1
        assert signals[0]["side"] == "sell"
        assert "Trailing stop" in signals[0]["reason"]

    def test_take_profit_triggers_sell(self, strategy):
        strategy.position = "long"
        strategy.position_amount = 3.0
        strategy.entry_price = 100.0
        strategy.risk_manager.check_take_profit.return_value = True

        signals = self._run(strategy, last_close=112.0)

        assert len(signals) == 1
        assert signals[0]["side"] == "sell"
        assert "Take-profit" in signals[0]["reason"]

    def test_no_risk_exit_without_open_position(self, strategy):
        strategy.risk_manager.check_stop_loss.return_value = True
        strategy.risk_manager.check_trailing_stop.return_value = True
        strategy.risk_manager.check_take_profit.return_value = True

        signals = self._run(strategy)

        assert signals == []

    def test_highest_since_entry_ratchets_up_with_price(self, strategy):
        strategy.position = "long"
        strategy.entry_price = 100.0
        strategy.highest_since_entry = 105.0

        self._run(strategy, last_close=130.0)

        assert strategy.highest_since_entry == pytest.approx(130.0)

    def test_highest_since_entry_does_not_ratchet_down(self, strategy):
        strategy.position = "long"
        strategy.entry_price = 100.0
        strategy.highest_since_entry = 150.0

        self._run(strategy, last_close=130.0)

        assert strategy.highest_since_entry == pytest.approx(150.0)


class TestOnOrderFilled:
    def test_buy_fill_opens_long_and_sets_entry(self, strategy):
        strategy.on_order_filled({"side": "buy", "filled": 2.0, "price": 100.0})

        assert strategy.position == "long"
        assert strategy.position_amount == pytest.approx(2.0)
        assert strategy.entry_price == pytest.approx(100.0)
        assert strategy.highest_since_entry == pytest.approx(100.0)
        assert strategy.stats["trades_executed"] == 1

    def test_buy_fills_accumulate_position_amount(self, strategy):
        strategy.on_order_filled({"side": "buy", "filled": 2.0, "price": 100.0})
        strategy.on_order_filled({"side": "buy", "filled": 1.5, "price": 105.0})

        assert strategy.position_amount == pytest.approx(3.5)
        assert strategy.entry_price == pytest.approx(105.0)  # last fill wins
        assert strategy.stats["trades_executed"] == 2

    def test_buy_fill_prefers_average_over_price(self, strategy):
        strategy.on_order_filled({"side": "buy", "filled": 1.0, "average": 99.0, "price": 101.0})

        assert strategy.entry_price == pytest.approx(99.0)

    def test_buy_fill_derives_price_from_cost_and_filled(self, strategy):
        strategy.on_order_filled({"side": "buy", "filled": 2.0, "cost": 220.0})

        assert strategy.entry_price == pytest.approx(110.0)

    def test_buy_fill_with_no_price_info_still_tracks_amount(self, strategy):
        strategy.on_order_filled({"side": "buy", "filled": 2.0})

        assert strategy.position == "long"
        assert strategy.position_amount == pytest.approx(2.0)
        assert strategy.entry_price == 0.0

    def test_sell_fill_records_pnl_and_resets_position(self, strategy):
        strategy.position = "long"
        strategy.position_amount = 2.0
        strategy.entry_price = 100.0
        strategy.highest_since_entry = 110.0

        strategy.on_order_filled({"side": "sell", "filled": 2.0, "price": 110.0})

        assert strategy.stats["total_pnl"] == pytest.approx(10.0)
        assert strategy.position is None
        assert strategy.position_amount == 0.0
        assert strategy.entry_price == 0.0
        assert strategy.highest_since_entry == 0.0
        assert strategy.stats["trades_executed"] == 1

    def test_sell_fill_without_prior_entry_price_skips_pnl(self, strategy):
        strategy.position = "long"
        strategy.position_amount = 2.0
        strategy.entry_price = 0.0

        strategy.on_order_filled({"side": "sell", "filled": 2.0, "price": 110.0})

        assert strategy.stats["total_pnl"] == pytest.approx(0.0)
        assert strategy.position is None

    def test_unknown_side_is_ignored(self, strategy):
        strategy.on_order_filled({"side": "hold", "filled": 2.0, "price": 100.0})

        assert strategy.position is None
        assert strategy.stats["trades_executed"] == 0


class TestToDict:
    def test_to_dict_includes_strategy_state(self, strategy):
        strategy.position = "long"
        strategy.entry_price = 100.0
        strategy.highest_since_entry = 105.0
        strategy.last_signal = "buy"

        data = strategy.to_dict()

        assert data["position"] == "long"
        assert data["entry_price"] == pytest.approx(100.0)
        assert data["highest_since_entry"] == pytest.approx(105.0)
        assert data["last_signal"] == "buy"
        assert data["timeframe"] == "4h"
        assert data["name"] == "trend_following"
