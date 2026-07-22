"""Tests for the pure analysis functions in whale_tracker.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from whale_tracker import analyze_large_trades, analyze_orderbook_imbalance


def _trade(price, qty, is_buyer_maker, time_ms):
    return {
        "price": str(price),
        "qty": str(qty),
        "isBuyerMaker": is_buyer_maker,
        "time": time_ms,
    }


class TestAnalyzeLargeTrades:
    def test_empty_trades_returns_empty_list_and_zero_threshold(self):
        large_trades, threshold = analyze_large_trades([])
        assert large_trades == []
        assert threshold == 0

    def test_flags_trades_at_or_above_threshold(self):
        trades = [_trade(100, i + 1, False, 1000 + i) for i in range(10)]
        large_trades, threshold = analyze_large_trades(trades, threshold_percentile=90)
        assert threshold == max(float(t["qty"]) * float(t["price"]) for t in trades)
        assert all(t["value"] >= threshold for t in large_trades)
        assert len(large_trades) >= 1

    def test_side_derived_from_is_buyer_maker(self):
        buy_trade = _trade(100, 1, False, 1)
        sell_trade = _trade(100, 1, True, 2)
        large_trades, _ = analyze_large_trades([buy_trade, sell_trade], threshold_percentile=0)
        sides = {t["time"]: t["side"] for t in large_trades}
        assert sides[1] == "BUY"
        assert sides[2] == "SELL"


class TestAnalyzeOrderbookImbalance:
    def _orderbook(self, bids, asks):
        return {
            "bids": [[str(p), str(q)] for p, q in bids],
            "asks": [[str(p), str(q)] for p, q in asks],
        }

    def test_bullish_when_bid_volume_dominates(self):
        orderbook = self._orderbook(
            bids=[(99, 10), (98, 10)],
            asks=[(101, 2), (102, 2)],
        )
        result = analyze_orderbook_imbalance(orderbook, depth=20)
        assert result["imbalance"] == "BULLISH"
        assert result["volume_ratio"] > 1.5

    def test_bearish_when_ask_volume_dominates(self):
        orderbook = self._orderbook(
            bids=[(99, 2), (98, 2)],
            asks=[(101, 10), (102, 10)],
        )
        result = analyze_orderbook_imbalance(orderbook, depth=20)
        assert result["imbalance"] == "BEARISH"
        assert result["volume_ratio"] < 0.67

    def test_neutral_when_balanced(self):
        orderbook = self._orderbook(
            bids=[(99, 10), (98, 10)],
            asks=[(101, 10), (102, 10)],
        )
        result = analyze_orderbook_imbalance(orderbook, depth=20)
        assert result["imbalance"] == "NEUTRAL"
        assert result["volume_ratio"] == 1.0

    def test_zero_ask_volume_gives_infinite_ratio(self):
        orderbook = self._orderbook(bids=[(99, 5)], asks=[])
        result = analyze_orderbook_imbalance(orderbook, depth=20)
        assert result["volume_ratio"] == float("inf")
        assert result["imbalance"] == "BULLISH"

    def test_wall_detected_when_order_is_3x_average(self):
        orderbook = self._orderbook(
            bids=[(99, 30), (98, 1), (97, 1), (96, 1), (95, 1)],
            asks=[(101, 2), (102, 2), (103, 2)],
        )
        result = analyze_orderbook_imbalance(orderbook, depth=20)
        assert any(w["price"] == 99 for w in result["bid_walls"])
        assert result["ask_walls"] == []
