"""Tests for the Cross-Exchange Arbitrage strategy."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.arbitrage import ArbitrageStrategy  # noqa: E402


def _make_strategy(**params):
    defaults = {
        "symbol": "BTC/USDT",
        "exchanges": ["binance", "kraken"],
        "min_spread_pct": 0.3,
        "order_amount_usdt": 50.0,
        "fee_pct": 0.1,
    }
    defaults.update(params)
    exchange_manager = MagicMock()
    exchange_manager.available_exchanges = ["binance", "kraken"]
    risk_manager = MagicMock()
    return ArbitrageStrategy("sid-1", defaults, exchange_manager, risk_manager)


def _ticker(bid, ask):
    return {"bid": bid, "ask": ask}


class TestOnStart:
    def test_uses_available_exchanges_when_unset(self):
        s = _make_strategy(exchanges=[])
        s.exchange_manager.available_exchanges = ["binance", "kraken", "okx"]
        s.on_start()
        assert s.target_exchanges == ["binance", "kraken", "okx"]
        assert s.active is True

    def test_keeps_explicit_exchanges_when_set(self):
        s = _make_strategy(exchanges=["binance", "kraken"])
        s.exchange_manager.available_exchanges = ["binance", "kraken", "okx"]
        s.on_start()
        assert s.target_exchanges == ["binance", "kraken"]

    def test_fewer_than_two_exchanges_deactivates(self):
        s = _make_strategy(exchanges=["binance"])
        s.on_start()
        assert s.active is False

    def test_no_exchanges_available_deactivates(self):
        s = _make_strategy(exchanges=[])
        s.exchange_manager.available_exchanges = []
        s.on_start()
        assert s.active is False


class TestEvaluateGating:
    def test_inactive_returns_no_signals(self):
        s = _make_strategy()
        s.active = False
        assert s.evaluate() == []

    def test_fewer_than_two_target_exchanges_returns_no_signals(self):
        s = _make_strategy(exchanges=["binance"])
        s.active = True
        assert s.evaluate() == []

    def test_ticker_failure_on_one_exchange_is_skipped(self):
        s = _make_strategy()
        s.active = True
        s.exchange_manager.get_ticker.side_effect = [
            _ticker(100, 101),
            Exception("network error"),
        ]
        assert s.evaluate() == []

    def test_zero_bid_or_ask_excludes_exchange(self):
        s = _make_strategy()
        s.active = True
        s.exchange_manager.get_ticker.side_effect = [
            _ticker(100, 101),
            _ticker(0, 0),
        ]
        assert s.evaluate() == []

    def test_spread_below_threshold_returns_no_signals(self):
        s = _make_strategy(min_spread_pct=0.3, fee_pct=0.1)
        s.active = True
        # spread ab = (100.2 - 100) / 100 = 0.2% < fee(0.2%) + min_spread(0.3%)
        s.exchange_manager.get_ticker.side_effect = [
            _ticker(99.9, 100.0),
            _ticker(100.2, 100.3),
        ]
        assert s.evaluate() == []


class TestEvaluateSignals:
    def test_profitable_spread_generates_paired_buy_sell(self):
        s = _make_strategy(min_spread_pct=0.3, fee_pct=0.1, order_amount_usdt=50.0)
        s.active = True
        # ask on binance = 100, bid on kraken = 101 -> spread_ab = 1% > fee(0.2%) + min(0.3%)
        s.exchange_manager.get_ticker.side_effect = [
            _ticker(99.5, 100.0),   # binance
            _ticker(101.0, 101.5),  # kraken
        ]
        signals = s.evaluate()

        assert len(signals) == 2
        buy, sell = signals
        assert buy["side"] == "buy"
        assert buy["exchange"] == "binance"
        assert buy["price"] == 100.0
        assert buy["amount"] == round(50.0 / 100.0, 8)
        assert buy["arb_pair"] == {"buy_exchange": "binance", "sell_exchange": "kraken"}

        assert sell["side"] == "sell"
        assert sell["exchange"] == "kraken"
        assert sell["price"] == 101.0
        assert sell["amount"] == buy["amount"]
        assert sell["arb_pair"] == buy["arb_pair"]

    def test_reverse_direction_spread_generates_paired_signals(self):
        s = _make_strategy(min_spread_pct=0.3, fee_pct=0.1, order_amount_usdt=50.0)
        s.active = True
        # ask on kraken = 100, bid on binance = 101 -> spread_ba profitable, spread_ab not
        s.exchange_manager.get_ticker.side_effect = [
            _ticker(101.0, 101.5),  # binance
            _ticker(99.5, 100.0),   # kraken
        ]
        signals = s.evaluate()

        assert len(signals) == 2
        buy, sell = signals
        assert buy["side"] == "buy"
        assert buy["exchange"] == "kraken"
        assert buy["price"] == 100.0
        assert sell["side"] == "sell"
        assert sell["exchange"] == "binance"
        assert sell["price"] == 101.0
        assert buy["arb_pair"] == {"buy_exchange": "kraken", "sell_exchange": "binance"}

    def test_three_exchanges_evaluates_every_pair(self):
        s = _make_strategy(
            exchanges=["binance", "kraken", "okx"],
            min_spread_pct=0.3,
            fee_pct=0.1,
            order_amount_usdt=50.0,
        )
        s.active = True
        # binance and kraken flat (no opportunity), okx bid high enough to be
        # profitable against both binance and kraken asks.
        s.exchange_manager.get_ticker.side_effect = [
            _ticker(99.9, 100.0),   # binance
            _ticker(99.9, 100.0),   # kraken
            _ticker(101.5, 101.6),  # okx
        ]
        signals = s.evaluate()

        pairs = {(sig["arb_pair"]["buy_exchange"], sig["arb_pair"]["sell_exchange"]) for sig in signals}
        assert ("binance", "okx") in pairs
        assert ("kraken", "okx") in pairs
        assert ("binance", "kraken") not in pairs

    def test_amount_rounded_to_eight_decimals(self):
        s = _make_strategy(min_spread_pct=0.3, fee_pct=0.1, order_amount_usdt=33.333333333)
        s.active = True
        s.exchange_manager.get_ticker.side_effect = [
            _ticker(99.5, 100.0),
            _ticker(101.0, 101.5),
        ]
        signals = s.evaluate()
        buy = signals[0]
        assert buy["amount"] == round(33.333333333 / 100.0, 8)
