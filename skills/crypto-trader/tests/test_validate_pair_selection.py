"""Tests for the pair/parameter selection logic feeding the winrate-gate harness.

validate_mr.py / validate_tf.py decide WHICH pairs and parameter sets get
backtested to produce the numbers in VERDICT.md (the >=60% winrate gate that
blocks real-money trading). Neither top_liquid_pairs() nor build_param_sets()
had any test coverage before this file, despite being the selection logic
upstream of every number in that report.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_mr import aggregate_by_param, top_liquid_pairs  # noqa: E402
from validate_tf import build_param_sets  # noqa: E402


def _row(symbol, label, wins, losses, total_return_pct, avg_win_pct=0.0, avg_loss_pct=0.0):
    return {
        "symbol": symbol,
        "params": label,
        "wins": wins,
        "losses": losses,
        "total_return_pct": total_return_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
    }


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


class TestAggregateByParam:
    """aggregate_by_param() produces the "AGREGAT per parameter" rows --
    the last step before a param set is declared LOLOS-GATE/win-rate-OK in
    validate_mr.py and validate_tf.py's output, so a bug here misreports the
    exact number VERDICT.md quotes for the real-money gate.
    """

    def test_single_symbol_single_label(self):
        rows = [_row("BTC/USDT", "base", wins=7, losses=3, total_return_pct=12.5)]
        agg = aggregate_by_param(rows)
        assert len(agg) == 1
        assert agg[0]["label"] == "base"
        assert agg[0]["trades"] == 10
        assert agg[0]["wins"] == 7
        assert agg[0]["losses"] == 3
        assert agg[0]["win_rate_pct"] == 70.0
        assert agg[0]["total_return_pct"] == 12.5

    def test_sums_across_symbols_for_same_label(self):
        rows = [
            _row("BTC/USDT", "base", wins=6, losses=4, total_return_pct=10.0),
            _row("ETH/USDT", "base", wins=3, losses=7, total_return_pct=-5.0),
        ]
        agg = aggregate_by_param(rows)
        assert len(agg) == 1
        assert agg[0]["trades"] == 20
        assert agg[0]["wins"] == 9
        assert agg[0]["losses"] == 11
        assert agg[0]["win_rate_pct"] == 45.0
        assert agg[0]["total_return_pct"] == 5.0

    def test_keeps_labels_separate_and_sorted(self):
        rows = [
            _row("BTC/USDT", "zeta", wins=1, losses=0, total_return_pct=1.0),
            _row("BTC/USDT", "alpha", wins=1, losses=0, total_return_pct=1.0),
        ]
        agg = aggregate_by_param(rows)
        assert [a["label"] for a in agg] == ["alpha", "zeta"]

    def test_zero_trades_win_rate_is_zero_not_a_crash(self):
        rows = [_row("BTC/USDT", "base", wins=0, losses=0, total_return_pct=0.0)]
        agg = aggregate_by_param(rows)
        assert agg[0]["trades"] == 0
        assert agg[0]["win_rate_pct"] == 0
        assert agg[0]["profit_factor"] == 0

    def test_profit_factor_weighted_by_win_loss_counts(self):
        # gross win = 3 wins * 2.0 avg = 6.0 ; gross loss = 2 losses * -1.0 avg = 2.0
        # -> PF = 6.0 / 2.0 = 3.0
        rows = [_row("BTC/USDT", "base", wins=3, losses=2, total_return_pct=4.0,
                     avg_win_pct=2.0, avg_loss_pct=-1.0)]
        agg = aggregate_by_param(rows)
        assert agg[0]["profit_factor"] == pytest.approx(3.0)

    def test_profit_factor_zero_when_no_losses(self):
        rows = [_row("BTC/USDT", "base", wins=5, losses=0, total_return_pct=8.0,
                     avg_win_pct=1.5, avg_loss_pct=0.0)]
        agg = aggregate_by_param(rows)
        assert agg[0]["profit_factor"] == 0

    def test_profit_factor_aggregates_across_symbols(self):
        rows = [
            _row("BTC/USDT", "base", wins=2, losses=1, total_return_pct=1.0,
                 avg_win_pct=1.0, avg_loss_pct=-1.0),
            _row("ETH/USDT", "base", wins=2, losses=1, total_return_pct=1.0,
                 avg_win_pct=1.0, avg_loss_pct=-1.0),
        ]
        agg = aggregate_by_param(rows)
        # gross win = (2*1.0) + (2*1.0) = 4.0 ; gross loss = |(1*-1.0)+(1*-1.0)| = 2.0
        assert agg[0]["profit_factor"] == pytest.approx(2.0)

    def test_empty_results_returns_empty_list(self):
        assert aggregate_by_param([]) == []
