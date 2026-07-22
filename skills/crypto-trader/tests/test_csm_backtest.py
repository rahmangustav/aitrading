"""Tests for the cross-sectional momentum backtest core."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from csm_backtest import backtest_cross_sectional_momentum


def _candles(prices, spread=0.5, vol=1000.0):
    out = []
    prev = prices[0]
    for i, c in enumerate(prices):
        o = prev
        h = max(o, c) + spread
        l = min(o, c) - spread
        out.append([i * 3600000, o, h, l, c, vol])
        prev = c
    return out


def _trending(n, start=100.0, step=0.0):
    return _candles([start + i * step for i in range(n)])


class TestGuards:
    def test_no_trades_with_fewer_than_two_symbols(self):
        data = {"A/USDT": _trending(200, step=1.0)}
        res = backtest_cross_sectional_momentum(data)
        assert res["total_trades"] == 0
        assert res["signals"] == 0

    def test_no_trades_on_thin_data(self):
        data = {"A/USDT": _trending(20), "B/USDT": _trending(20)}
        res = backtest_cross_sectional_momentum(data, {"lookback_bars": 30, "hold_bars": 30})
        assert res["total_trades"] == 0
        assert res["signals"] == 0

    def test_no_trades_when_top_k_is_zero(self):
        data = {"A/USDT": _trending(200, step=1.0), "B/USDT": _trending(200, step=-1.0)}
        res = backtest_cross_sectional_momentum(data, {"top_k": 0})
        assert res["total_trades"] == 0


class TestRanking:
    def test_picks_the_highest_momentum_symbol(self):
        data = {
            "UP/USDT": _trending(200, start=100.0, step=1.0),
            "FLAT/USDT": _trending(200, start=100.0, step=0.0),
            "DOWN/USDT": _trending(200, start=100.0, step=-0.5),
        }
        res = backtest_cross_sectional_momentum(
            data, {"lookback_bars": 30, "hold_bars": 30, "top_k": 1},
        )
        assert res["total_trades"] > 0
        assert all(t["symbol"] == "UP/USDT" for t in res["trades"])

    def test_top_k_two_holds_the_two_best_each_period(self):
        data = {
            "BEST/USDT": _trending(200, step=2.0),
            "MID/USDT": _trending(200, step=1.0),
            "WORST/USDT": _trending(200, step=-1.0),
        }
        res = backtest_cross_sectional_momentum(
            data, {"lookback_bars": 30, "hold_bars": 30, "top_k": 2},
        )
        symbols_traded = {t["symbol"] for t in res["trades"]}
        assert symbols_traded == {"BEST/USDT", "MID/USDT"}

    def test_multiple_rebalance_periods_accumulate_signals(self):
        data = {"A/USDT": _trending(400, step=0.5), "B/USDT": _trending(400, step=-0.5)}
        res = backtest_cross_sectional_momentum(
            data, {"lookback_bars": 30, "hold_bars": 30, "top_k": 1},
        )
        assert res["signals"] > 3
        assert res["periods"] == res["signals"]


class TestPnl:
    def test_trade_pnl_reflects_forward_return_minus_costs(self):
        # A single trending winner, alone against a flat symbol so ranking
        # is unambiguous every period.
        data = {
            "UP/USDT": _trending(200, start=100.0, step=1.0),
            "FLAT/USDT": _trending(200, start=100.0, step=0.0),
        }
        res = backtest_cross_sectional_momentum(
            data, {"lookback_bars": 30, "hold_bars": 30, "top_k": 1},
            fee_pct=0.0, slippage_pct=0.0,
        )
        assert res["total_trades"] > 0
        # UP/USDT rises 1.0/bar; over a 30-bar hold that's a solidly positive
        # forward return with zero cost drag.
        assert res["avg_win_pct"] > 5.0

    def test_costs_are_charged_on_every_trade(self):
        data = {
            "UP/USDT": _trending(200, start=100.0, step=1.0),
            "FLAT/USDT": _trending(200, start=100.0, step=0.0),
        }
        params = {"lookback_bars": 30, "hold_bars": 30, "top_k": 1}
        free = backtest_cross_sectional_momentum(data, params, fee_pct=0.0, slippage_pct=0.0)
        paid = backtest_cross_sectional_momentum(data, params)
        assert paid["total_return_pct"] < free["total_return_pct"]

    def test_symbols_with_shorter_history_truncate_the_whole_run(self):
        data = {
            "LONG/USDT": _trending(400, step=1.0),
            "SHORT/USDT": _trending(100, step=1.0),
        }
        res = backtest_cross_sectional_momentum(
            data, {"lookback_bars": 30, "hold_bars": 30, "top_k": 1},
        )
        # n_bars is min(400, 100) = 100, so no rebalance can look past bar 100.
        assert all(t["exit_ts"] <= 99 * 3600000 for t in res["trades"])
