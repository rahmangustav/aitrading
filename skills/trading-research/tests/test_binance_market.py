"""Tests for skills/trading-research/scripts/binance_market.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import binance_market as bm


ORDERBOOK = {
    "bids": [["100.50", "1.2000"], ["100.40", "0.5000"]],
    "asks": [["100.60", "0.8000"], ["100.70", "2.1000"], ["100.80", "0.3000"]],
}


class TestFormatOrderbookOutput:
    def test_prints_both_bid_and_ask_price(self, capsys):
        bm.format_orderbook_output(ORDERBOOK, depth=10)
        out = capsys.readouterr().out
        assert "$100.50" in out
        assert "$100.60" in out

    def test_every_ask_row_shows_its_price(self, capsys):
        bm.format_orderbook_output(ORDERBOOK, depth=10)
        out = capsys.readouterr().out
        for ask_price, _ in ORDERBOOK["asks"]:
            assert f"${float(ask_price):,.2f}" in out

    def test_handles_unequal_bid_ask_depth(self, capsys):
        bm.format_orderbook_output(ORDERBOOK, depth=10)
        out = capsys.readouterr().out
        rows = [line for line in out.splitlines() if line.strip().startswith("$")]
        assert len(rows) == max(len(ORDERBOOK["bids"]), len(ORDERBOOK["asks"]))
        last_row = rows[-1]
        assert "$100.40" not in last_row
        assert "$100.80" in last_row

    def test_respects_depth_limit(self, capsys):
        bm.format_orderbook_output(ORDERBOOK, depth=1)
        out = capsys.readouterr().out
        assert "$100.70" not in out
        assert "$100.80" not in out


class TestGetKlines:
    def test_maps_raw_kline_array_to_named_fields(self, monkeypatch):
        raw = [
            [1700000000000, "10.0", "12.0", "9.0", "11.0", "1000.0",
             1700000060000, "11000.0", 42, "600.0", "6600.0", "0"],
        ]
        monkeypatch.setattr(bm, "fetch_json", lambda url: raw)

        result = bm.get_klines("BTCUSDT", "1h", limit=1)

        assert result == [{
            "open_time": 1700000000000,
            "open": 10.0,
            "high": 12.0,
            "low": 9.0,
            "close": 11.0,
            "volume": 1000.0,
            "close_time": 1700000060000,
            "quote_volume": 11000.0,
            "trades": 42,
            "taker_buy_base": 600.0,
            "taker_buy_quote": 6600.0,
        }]


class TestGetFundingRate:
    def test_returns_friendly_error_when_symbol_has_no_futures_market(self, monkeypatch):
        def boom(url):
            raise SystemExit(1)

        monkeypatch.setattr(bm, "fetch_json", boom)

        result = bm.get_funding_rate("NOTASYMBOL")

        assert result == {"error": "Funding rate only available for futures symbols"}
