"""Tests for the pair/parameter selection logic feeding the winrate-gate harness.

validate_mr.py / validate_tf.py / validate_csm.py decide WHICH pairs and
parameter sets get backtested to produce the numbers in VERDICT.md (the
>=60% winrate gate that blocks real-money trading). None of top_liquid_pairs()
or the three build_param_sets() had any test coverage before this file,
despite being the selection logic upstream of every number in that report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_mr import top_liquid_pairs  # noqa: E402
from validate_tf import build_param_sets  # noqa: E402
from validate_csm import build_param_sets as build_param_sets_csm  # noqa: E402


class _FakeExchange:
    def __init__(self, tickers):
        self._tickers = tickers

    def fetch_tickers(self):
        return self._tickers


def _ticker(quote_volume):
    return {"quoteVolume": quote_volume}


class TestTopLiquidPairs:
    def test_sorts_by_quote_volume_descending(self):
        ex = _FakeExchange({
            "BTC/USDT": _ticker(1000),
            "ETH/USDT": _ticker(5000),
            "SOL/USDT": _ticker(2000),
        })
        assert top_liquid_pairs(ex, 3) == ["ETH/USDT", "SOL/USDT", "BTC/USDT"]

    def test_respects_n_limit(self):
        ex = _FakeExchange({
            "BTC/USDT": _ticker(1000),
            "ETH/USDT": _ticker(5000),
            "SOL/USDT": _ticker(2000),
        })
        assert top_liquid_pairs(ex, 2) == ["ETH/USDT", "SOL/USDT"]

    def test_excludes_non_usdt_quote(self):
        ex = _FakeExchange({
            "BTC/USDT": _ticker(1000),
            "BTC/BUSD": _ticker(9999),
            "ETH/BTC": _ticker(9999),
        })
        assert top_liquid_pairs(ex, 5) == ["BTC/USDT"]

    def test_excludes_futures_and_leveraged_symbols(self):
        ex = _FakeExchange({
            "BTC/USDT": _ticker(1000),
            "BTC/USDT:USDT": _ticker(9999),  # perpetual futures
            "BTCUP/USDT": _ticker(9999),
            "BTCDOWN/USDT": _ticker(9999),
            "BTCBULL/USDT": _ticker(9999),
            "BTCBEAR/USDT": _ticker(9999),
        })
        assert top_liquid_pairs(ex, 5) == ["BTC/USDT"]

    def test_excludes_stablecoin_and_fiat_bases(self):
        ex = _FakeExchange({
            "SOL/USDT": _ticker(1000),
            "USDC/USDT": _ticker(9999),
            "FDUSD/USDT": _ticker(9999),
            "EUR/USDT": _ticker(9999),
            "PAXG/USDT": _ticker(9999),
        })
        assert top_liquid_pairs(ex, 5) == ["SOL/USDT"]

    def test_zero_or_missing_quote_volume_excluded(self):
        ex = _FakeExchange({
            "BTC/USDT": _ticker(1000),
            "ETH/USDT": _ticker(0),
            "SOL/USDT": {},
        })
        assert top_liquid_pairs(ex, 5) == ["BTC/USDT"]

    def test_empty_tickers_returns_empty_list(self):
        assert top_liquid_pairs(_FakeExchange({}), 10) == []


class TestBuildParamSets:
    def test_no_grid_returns_single_live_param_set(self):
        sets = build_param_sets(grid=False)
        assert len(sets) == 1
        assert sets[0]["sl_pct"] == 5.0
        assert sets[0]["tp_pct"] == 10.0

    def test_grid_returns_seven_variants_with_unique_labels(self):
        sets = build_param_sets(grid=True)
        assert len(sets) == 7
        labels = [s["label"] for s in sets]
        assert len(labels) == len(set(labels))

    def test_grid_first_variant_matches_live_params(self):
        sets = build_param_sets(grid=True)
        live = sets[0]
        assert live["sl_pct"] == 5.0
        assert live["tp_pct"] == 10.0
        assert "intrabar_stops" not in live

    def test_grid_includes_close_only_variant_disabling_intrabar_stops(self):
        sets = build_param_sets(grid=True)
        close_only = next(s for s in sets if s.get("intrabar_stops") is False)
        assert close_only["sl_pct"] == 5.0

    def test_grid_includes_adx_gate_variant(self):
        sets = build_param_sets(grid=True)
        adx = next(s for s in sets if "adx_trend_threshold" in s)
        assert adx["adx_trend_threshold"] == 25


class TestBuildParamSetsCsm:
    def test_no_grid_returns_single_live_param_set(self):
        sets = build_param_sets_csm(grid=False)
        assert len(sets) == 1
        assert sets[0]["lookback_bars"] == 30
        assert sets[0]["hold_bars"] == 30
        assert sets[0]["top_k"] == 3

    def test_grid_returns_six_variants_with_unique_labels(self):
        sets = build_param_sets_csm(grid=True)
        assert len(sets) == 6
        labels = [s["label"] for s in sets]
        assert len(labels) == len(set(labels))

    def test_grid_first_variant_matches_live_params(self):
        sets = build_param_sets_csm(grid=True)
        live = sets[0]
        assert live["lookback_bars"] == 30
        assert live["hold_bars"] == 30
        assert live["top_k"] == 3

    def test_grid_varies_lookback_hold_and_top_k_independently(self):
        sets = build_param_sets_csm(grid=True)
        lookbacks = {s["lookback_bars"] for s in sets}
        holds = {s["hold_bars"] for s in sets}
        top_ks = {s["top_k"] for s in sets}
        assert lookbacks == {14, 30, 60}
        assert holds == {7, 14, 30}
        assert top_ks == {1, 3, 5}
