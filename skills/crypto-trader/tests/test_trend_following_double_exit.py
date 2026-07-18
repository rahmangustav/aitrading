"""Regression test: TrendFollowingStrategy.evaluate() must not emit two
sell signals for the same position when a bearish EMA cross / RSI
overbought exit and a risk-manager exit (stop-loss / trailing stop /
take-profit) both trigger on the same candle.

Before the fix, the cross/RSI block and the risk-manager block were two
independent ``if`` statements, so a sharp drop that both flipped the EMA
cross *and* breached the stop-loss produced two identical "sell" signals
for the same position in a single evaluate() call.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.trend_following import TrendFollowingStrategy  # noqa: E402


def _flat_then_drop_candles(flat_price: float = 100.0, drop_price: float = 50.0, flat_rows: int = 35):
    """OHLCV rows: long flat run (EMAs converge) then one sharp drop.

    The drop is large enough that EMA(9) crosses below EMA(21) on the
    final row while both were ~equal on the previous row — a genuine
    bearish cross, not a contrived indicator value.
    """
    rows = []
    for i in range(flat_rows):
        rows.append([i * 1000, flat_price, flat_price, flat_price, flat_price, 1.0])
    rows.append([flat_rows * 1000, flat_price, flat_price, drop_price, drop_price, 1.0])
    return rows


def _make_strategy(risk_overrides=None):
    exchange_manager = MagicMock()
    risk_manager = MagicMock()
    risk_manager.check_stop_loss.return_value = False
    risk_manager.check_trailing_stop.return_value = False
    risk_manager.check_take_profit.return_value = False
    if risk_overrides:
        for name, value in risk_overrides.items():
            getattr(risk_manager, name).return_value = value

    strategy = TrendFollowingStrategy(
        strategy_id="tf-test",
        params={"symbol": "BTC/USDT", "exchange": "binance"},
        exchange_manager=exchange_manager,
        risk_manager=risk_manager,
    )
    strategy.active = True
    return strategy, exchange_manager, risk_manager


def test_bearish_cross_and_stop_loss_together_yield_one_sell_signal():
    strategy, exchange_manager, risk_manager = _make_strategy(
        risk_overrides={"check_stop_loss": True, "check_trailing_stop": True, "check_take_profit": True}
    )
    exchange_manager.get_ohlcv.return_value = _flat_then_drop_candles()

    strategy.position = "long"
    strategy.position_amount = 5.0
    strategy.entry_price = 100.0
    strategy.highest_since_entry = 100.0

    signals = strategy.evaluate()

    sell_signals = [s for s in signals if s["side"] == "sell"]
    assert len(sell_signals) == 1, f"expected exactly one sell signal, got {len(sell_signals)}: {sell_signals}"
    assert sell_signals[0]["amount"] == pytest.approx(5.0)
    # The cross/RSI branch ran first, so the risk manager should never
    # have been consulted for this candle.
    risk_manager.check_stop_loss.assert_not_called()


def test_risk_exit_alone_still_fires_when_no_cross():
    """Guard must not suppress a legitimate solo risk-manager exit."""
    strategy, exchange_manager, risk_manager = _make_strategy(
        risk_overrides={"check_stop_loss": True}
    )
    # Flat candles throughout: no EMA cross, RSI stays neutral.
    exchange_manager.get_ohlcv.return_value = _flat_then_drop_candles(drop_price=100.0)

    strategy.position = "long"
    strategy.position_amount = 5.0
    strategy.entry_price = 100.0
    strategy.highest_since_entry = 100.0

    signals = strategy.evaluate()

    sell_signals = [s for s in signals if s["side"] == "sell"]
    assert len(sell_signals) == 1
    assert "Stop-loss" in sell_signals[0]["reason"]
    risk_manager.check_stop_loss.assert_called_once()
