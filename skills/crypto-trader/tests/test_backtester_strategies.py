"""Tests for Backtester._backtest_grid / _backtest_dca / _backtest_trend --
the three strategy simulators actually driven by `main.py --mode backtest`.

Before this file, `Backtester` only had coverage for `_compute_metrics`
(PR #28, equity-curve fee accounting). The simulation loops that generate
the trades those metrics are computed from -- the part a user is trusting
when they run a backtest -- had none.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backtester import Backtester  # noqa: E402


def _bt(**kwargs):
    defaults = {"slippage_pct": 0.0, "fee_pct": 0.0, "initial_balance": 1000.0}
    defaults.update(kwargs)
    return Backtester(exchange_manager=None, **defaults)


def _candle(ts, o, h, l, c, v=1000.0):
    return [ts, o, h, l, c, v]


class TestBacktestDca:
    def test_buys_every_interval_when_balance_allows(self):
        bt = _bt(initial_balance=100.0)
        ohlcv = [_candle(i * 3600000, 100, 100, 100, 100) for i in range(5)]
        metrics = bt._backtest_dca(ohlcv, {"amount_per_buy_usdt": 10.0, "interval": "hourly"})
        assert metrics["total_trades"] == 5
        assert metrics["total_invested_usdt"] == 50.0
        assert metrics["total_crypto_bought"] == 0.5
        assert metrics["avg_buy_price"] == 100.0
        assert metrics["final_value_usdt"] == 100.0  # flat price, no fees: breakeven

    def test_stops_buying_once_balance_is_insufficient(self):
        bt = _bt(initial_balance=25.0)
        ohlcv = [_candle(i * 3600000, 100, 100, 100, 100) for i in range(5)]
        metrics = bt._backtest_dca(ohlcv, {"amount_per_buy_usdt": 10.0, "interval": "hourly"})
        # Only 2 buys fit a $25 balance at $10/buy; later candles are
        # skipped outright rather than buying a partial amount.
        assert metrics["total_trades"] == 2
        assert metrics["total_invested_usdt"] == 20.0

    def test_respects_interval_spacing(self):
        bt = _bt(initial_balance=1000.0)
        # 48 hourly candles, weekly interval (skip=168) -> only the first
        # candle is ever visited.
        ohlcv = [_candle(i * 3600000, 100, 100, 100, 100) for i in range(48)]
        metrics = bt._backtest_dca(ohlcv, {"amount_per_buy_usdt": 10.0, "interval": "weekly"})
        assert metrics["total_trades"] == 1


class TestBacktestGrid:
    def test_buy_then_sell_at_same_level_round_trips_cleanly(self):
        bt = _bt(initial_balance=1000.0)
        params = {"price_range": [100, 120], "num_grids": 1, "order_amount_usdt": 100.0}
        ohlcv = [
            # dips to touch level 100, closes back above it -> buy
            _candle(0, 105, 110, 90, 105),
            # dips again, closes back below level 100 -> sell what we bought
            _candle(3600000, 100, 105, 95, 95),
        ]
        metrics = bt._backtest_grid(ohlcv, params)
        assert metrics["final_value_usdt"] == 1000.0  # flat round trip, no fees
        assert metrics["total_trades"] == 1
        assert [o["side"] for o in metrics["orders"]] == ["buy", "sell"]

    def test_fees_reduce_final_value_on_a_flat_round_trip(self):
        bt = _bt(initial_balance=1000.0, fee_pct=1.0)
        params = {"price_range": [100, 120], "num_grids": 1, "order_amount_usdt": 100.0}
        ohlcv = [
            _candle(0, 105, 110, 90, 105),
            _candle(3600000, 100, 105, 95, 95),
        ]
        metrics = bt._backtest_grid(ohlcv, params)
        # 1% fee on the $100 buy and on the ~$100 sell proceeds: ~$2 total.
        assert metrics["total_fees_usdt"] == 2.0
        assert metrics["final_value_usdt"] == 998.0

    def test_repeated_touches_at_an_already_open_level_keep_buying(self):
        """Documents a real gap vs. the live GridTradingStrategy: the live
        strategy skips a level while `level_key in self.active_orders`
        (strategies/grid_trading.py:131), so it never opens a second
        position at a level before the first is closed. `_backtest_grid`
        has no equivalent guard on its buy side -- every candle that
        touches an already-bought level buys again, silently pyramiding
        exposure the live bot would never take.

        Not fixed here: a correct fix means reworking the simulator's grid
        state machine to mirror `active_orders`/`on_order_filled`'s
        shifting counter-order behaviour, not a one-line change. Recording
        it as a test so the divergence is visible and isn't fixed by
        accident (i.e. so a future change to this method has to
        consciously update this assertion, not silently break past it).
        """
        bt = _bt(initial_balance=1000.0)
        params = {"price_range": [100, 120], "num_grids": 1, "order_amount_usdt": 100.0}
        # Same level (100) touched and closed-above on three separate
        # candles, with no intervening sell.
        ohlcv = [_candle(i * 3600000, 105, 110, 90, 105) for i in range(3)]
        metrics = bt._backtest_grid(ohlcv, params)
        buys = [o for o in metrics["orders"] if o["side"] == "buy"]
        assert len(buys) == 3  # live strategy would only ever place 1


class TestBacktestTrend:
    @staticmethod
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

    def _noisy_flat(self, n, level=100.0, seed=3):
        rng = random.Random(seed)
        return [level + rng.gauss(0, 0.6) for _ in range(n)]

    def _dip_then_rally(self, n_flat=80, n_up=60):
        prices = self._noisy_flat(n_flat) + [100.0 + i * 1.5 for i in range(n_up)]
        return self._candles(prices)

    def test_no_trades_on_flat_data(self):
        bt = _bt()
        ohlcv = self._candles([100.0] * 30)
        metrics = bt._backtest_trend(ohlcv, {})
        assert metrics["total_trades"] == 0
        assert metrics["orders"] == []
        assert metrics["final_value_usdt"] == 1000.0

    def test_takes_a_trade_on_a_bullish_cross(self):
        bt = _bt()
        ohlcv = self._dip_then_rally()
        metrics = bt._backtest_trend(ohlcv, {})
        assert len(metrics["orders"]) >= 1
        assert metrics["orders"][0]["side"] == "buy"

    def test_fees_and_slippage_are_reflected_in_total_fees_and_final_value(self):
        bt_no_cost = _bt(slippage_pct=0.0, fee_pct=0.0)
        bt_with_cost = _bt(slippage_pct=0.5, fee_pct=0.5)
        ohlcv = self._dip_then_rally()
        m1 = bt_no_cost._backtest_trend(ohlcv, {})
        m2 = bt_with_cost._backtest_trend(ohlcv, {})
        assert m1["orders"], "fixture must actually produce a trade"
        assert m2["total_fees_usdt"] > 0
        assert m2["final_value_usdt"] <= m1["final_value_usdt"]
