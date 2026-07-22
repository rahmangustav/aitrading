"""Tests for the pure filter/ranking functions in market_scanner.py.

Only the functions that transform ticker data are covered here -- fetch_json,
get_all_tickers, get_exchange_info, and the print-based format_*/scan_market/main
functions do network I/O or print to stdout and aren't unit-tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from market_scanner import (  # noqa: E402
    filter_active_pairs,
    filter_usdt_pairs,
    find_breakout_candidates,
    find_top_gainers,
    find_top_losers,
    find_volatile_pairs,
    find_volume_spikes,
)


def _ticker(symbol, price_change_pct=0.0, last=100.0, high=100.0, low=100.0, volume=1_000_000.0):
    return {
        "symbol": symbol,
        "priceChangePercent": str(price_change_pct),
        "lastPrice": str(last),
        "highPrice": str(high),
        "lowPrice": str(low),
        "quoteVolume": str(volume),
    }


class TestFilterUsdtPairs:
    def test_keeps_only_usdt_quoted_pairs(self):
        tickers = [_ticker("BTCUSDT"), _ticker("ETHBTC"), _ticker("BNBBUSD")]
        result = filter_usdt_pairs(tickers)
        assert [t["symbol"] for t in result] == ["BTCUSDT"]

    def test_excludes_bare_usdt_symbol(self):
        tickers = [_ticker("USDT"), _ticker("BTCUSDT")]
        result = filter_usdt_pairs(tickers)
        assert [t["symbol"] for t in result] == ["BTCUSDT"]


class TestFilterActivePairs:
    def test_default_threshold_excludes_low_volume(self):
        tickers = [_ticker("A", volume=50_000), _ticker("B", volume=150_000)]
        result = filter_active_pairs(tickers)
        assert [t["symbol"] for t in result] == ["B"]

    def test_threshold_is_inclusive(self):
        tickers = [_ticker("A", volume=100_000)]
        result = filter_active_pairs(tickers, min_volume=100_000)
        assert len(result) == 1


class TestFindTopGainersLosers:
    def test_gainers_sorted_descending(self):
        tickers = [
            _ticker("A", price_change_pct=1),
            _ticker("B", price_change_pct=5),
            _ticker("C", price_change_pct=-2),
        ]
        result = find_top_gainers(tickers)
        assert [t["symbol"] for t in result] == ["B", "A", "C"]

    def test_losers_sorted_ascending(self):
        tickers = [
            _ticker("A", price_change_pct=1),
            _ticker("B", price_change_pct=5),
            _ticker("C", price_change_pct=-2),
        ]
        result = find_top_losers(tickers)
        assert [t["symbol"] for t in result] == ["C", "A", "B"]

    def test_limit_truncates_results(self):
        tickers = [_ticker(str(i), price_change_pct=i) for i in range(20)]
        assert len(find_top_gainers(tickers, limit=3)) == 3


class TestFindVolumeSpikes:
    def test_sorted_by_volume_descending(self):
        tickers = [_ticker("A", volume=10), _ticker("B", volume=1000), _ticker("C", volume=100)]
        result = find_volume_spikes(tickers)
        assert [t["symbol"] for t in result] == ["B", "C", "A"]


class TestFindVolatilePairs:
    def test_computes_high_low_spread_percentage(self):
        tickers = [_ticker("A", high=110, low=100)]
        result = find_volatile_pairs(tickers)
        assert result[0]["volatility"] == pytest.approx(10.0)

    def test_sorted_descending_by_volatility(self):
        tickers = [_ticker("A", high=105, low=100), _ticker("B", high=150, low=100)]
        result = find_volatile_pairs(tickers)
        assert [t["symbol"] for t in result] == ["B", "A"]

    def test_skips_zero_low_price_to_avoid_division_by_zero(self):
        tickers = [_ticker("A", high=10, low=0)]
        assert find_volatile_pairs(tickers) == []


class TestFindBreakoutCandidates:
    def test_includes_pairs_near_high_with_volume(self):
        tickers = [_ticker("A", last=99, high=100, volume=600_000)]
        result = find_breakout_candidates(tickers)
        assert [t["symbol"] for t in result] == ["A"]
        assert result[0]["distance_from_high"] == pytest.approx(1.0)

    def test_excludes_pairs_far_from_high(self):
        tickers = [_ticker("A", last=90, high=100, volume=600_000)]
        assert find_breakout_candidates(tickers) == []

    def test_excludes_pairs_below_volume_threshold(self):
        tickers = [_ticker("A", last=99, high=100, volume=100_000)]
        assert find_breakout_candidates(tickers) == []

    def test_sorted_ascending_by_distance_from_high(self):
        tickers = [
            _ticker("A", last=98.5, high=100, volume=600_000),
            _ticker("B", last=99.5, high=100, volume=600_000),
        ]
        result = find_breakout_candidates(tickers)
        assert [t["symbol"] for t in result] == ["B", "A"]

    def test_skips_zero_high_price_to_avoid_division_by_zero(self):
        tickers = [_ticker("A", last=0, high=0, volume=600_000)]
        assert find_breakout_candidates(tickers) == []
