"""Tests for the trend-following backtest core."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tf_backtest import _ema_series, _rsi_series, backtest_trend_following


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


def _noisy_flat(n, level=100.0, seed=3):
    """Seeded noise around a level. A perfectly flat series pins RSI at 100 on
    the first up-tick (avg_loss == 0), which no real market does."""
    import random
    rng = random.Random(seed)
    return [level + rng.gauss(0, 0.6) for _ in range(n)]


def _dip_then_rally(n_flat=80, n_up=60):
    """Noisy base (EMAs converge), then a rally that forces a bullish cross."""
    prices = _noisy_flat(n_flat) + [100.0 + i * 1.5 for i in range(n_up)]
    return _candles(prices)


class TestIndicators:
    def test_ema_matches_pandas_semantics(self):
        import pandas as pd
        closes = [100 + i * 0.7 for i in range(50)]
        want = pd.Series(closes).ewm(span=9, adjust=False).mean().tolist()
        got = _ema_series(closes, 9)
        assert all(abs(a - b) < 1e-9 for a, b in zip(want, got))

    def test_rsi_is_high_in_a_pure_uptrend(self):
        rsi = _rsi_series([100 + i for i in range(60)], 14)
        assert rsi[-1] > 90

    def test_rsi_is_low_in_a_pure_downtrend(self):
        rsi = _rsi_series([200 - i for i in range(60)], 14)
        assert rsi[-1] < 10

    def test_rsi_neutral_on_thin_data(self):
        assert _rsi_series([100, 101, 102], 14) == [50.0, 50.0, 50.0]


class TestBacktest:
    def test_no_trades_on_thin_data(self):
        res = backtest_trend_following(_candles([100] * 20))
        assert res["total_trades"] == 0
        assert res["signals"] == 0

    def test_takes_a_trade_on_a_bullish_cross(self):
        res = backtest_trend_following(_dip_then_rally())
        assert res["signals"] >= 1
        assert res["total_trades"] >= 1

    def test_rsi_exit_fires_before_take_profit_in_a_sharp_rally(self):
        # Faithful to the live strategy: the RSI-overbought sell is emitted
        # before price has time to reach a +10% target.
        res = backtest_trend_following(_dip_then_rally(), {"tp_pct": 10.0, "sl_pct": 5.0})
        assert res["exit_reasons"]["rsi"] >= 1
        assert res["exit_reasons"]["tp"] == 0

    def test_take_profit_fills_intrabar(self):
        # rsi_overbought=100 disables the RSI exit so TP mechanics are isolated.
        res = backtest_trend_following(
            _dip_then_rally(), {"tp_pct": 10.0, "sl_pct": 5.0, "rsi_overbought": 100},
        )
        assert res["exit_reasons"]["tp"] >= 1
        # TP exit nets roughly tp_pct minus round-trip costs.
        assert 8.0 < res["avg_win_pct"] < 10.5

    def test_stop_loss_caps_the_loss(self):
        # Cross up, then a crash steep enough that the stop fills before the
        # slower EMA cross can signal an exit.
        prices = _noisy_flat(80) + [100 + i * 1.5 for i in range(6)] + [109 - i * 30 for i in range(3)] + [20.0] * 20
        res = backtest_trend_following(
            _candles(prices), {"sl_pct": 5.0, "tp_pct": 100.0, "rsi_overbought": 100},
        )
        assert res["losses"] >= 1
        assert res["avg_loss_pct"] > -7.0  # 5% stop + costs, not a free fall

    def test_sma_master_switch_blocks_entries_below_the_average(self):
        # A long decline: crosses that do occur sit below the SMA200.
        prices = [300 - i * 0.5 for i in range(400)]
        res = backtest_trend_following(_candles(prices), {"trend_sma_period": 200})
        assert res["total_trades"] == 0
        assert res["blocked_by_bull"] == res["signals"]

    def test_close_only_mode_differs_from_intrabar(self):
        candles = _dip_then_rally()
        intrabar = backtest_trend_following(candles, {"intrabar_stops": True})
        close_only = backtest_trend_following(candles, {"intrabar_stops": False})
        assert intrabar["total_trades"] >= close_only["total_trades"] - 1
        assert close_only["total_trades"] >= 1

    def test_costs_are_charged_on_every_trade(self):
        candles = _dip_then_rally()
        free = backtest_trend_following(candles, {}, fee_pct=0.0, slippage_pct=0.0)
        paid = backtest_trend_following(candles, {})
        assert paid["total_return_pct"] < free["total_return_pct"]
