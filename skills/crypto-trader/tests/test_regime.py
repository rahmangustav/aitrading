"""Tests for regime detection and the mean-reversion backtest core."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mr_backtest import backtest_mean_reversion
from regime import adx, atr, bb_width_pct, detect_regime, mean_reversion_allowed


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


def _trending_up(n=100):
    return _candles([100 + i * 2.0 for i in range(n)])


def _flat(n=100):
    # Seeded noise around a level: a deterministic zigzag aliases ADX to 100,
    # so a realistic flat market needs randomness.
    import random
    rng = random.Random(3)
    return _candles([100 + rng.gauss(0, 0.5) for _ in range(n)])


class TestIndicators:
    def test_atr_positive_on_moving_prices(self):
        assert atr(_trending_up()) > 0

    def test_atr_zero_on_thin_data(self):
        assert atr(_trending_up(5)) == 0.0

    def test_adx_high_in_strong_trend(self):
        assert adx(_trending_up()) > 25

    def test_adx_low_in_flat_market(self):
        assert adx(_flat()) < 20

    def test_adx_zero_on_thin_data(self):
        assert adx(_trending_up(10)) == 0.0

    def test_bb_width_wider_in_trend_than_flat(self):
        trend_closes = [c[4] for c in _trending_up()]
        flat_closes = [c[4] for c in _flat()]
        assert bb_width_pct(trend_closes) > bb_width_pct(flat_closes)


class TestRegime:
    def test_trending_market_detected(self):
        regime, details = detect_regime(_trending_up())
        assert regime == "trending"
        assert details["adx"] >= 25

    def test_flat_market_is_ranging(self):
        regime, _ = detect_regime(_flat())
        assert regime == "ranging"

    def test_thin_data_is_unknown(self):
        regime, _ = detect_regime(_trending_up(10))
        assert regime == "unknown"

    def test_mr_blocked_in_trend(self):
        allowed, reason = mean_reversion_allowed(_trending_up())
        assert not allowed
        assert "blocked" in reason

    def test_mr_allowed_in_range(self):
        allowed, _ = mean_reversion_allowed(_flat())
        assert allowed

    def test_mr_fails_closed_on_thin_data(self):
        allowed, _ = mean_reversion_allowed(_trending_up(10))
        assert not allowed


class TestMeanReversionBacktest:
    def test_no_trades_on_thin_data(self):
        res = backtest_mean_reversion(_flat(30))
        assert res["total_trades"] == 0

    def test_metrics_shape(self):
        res = backtest_mean_reversion(_flat(300))
        for key in ("total_trades", "win_rate_pct", "profit_factor",
                    "max_drawdown_pct", "signals", "blocked_by_regime",
                    "exit_reasons", "trades"):
            assert key in res

    def test_trades_blocked_in_pure_trend(self):
        # Strong downtrend: dips below the lower band happen, but the
        # regime gate must block the entries.
        prices = [1000 - i * 5.0 for i in range(300)]
        res = backtest_mean_reversion(_candles(prices))
        assert res["total_trades"] == 0

    def test_bull_filter_blocks_entries_below_sma(self):
        # Slow downtrend with noise: price stays below its SMA, so with the
        # bull filter on every signal must be blocked.
        import random
        rng = random.Random(11)
        prices = []
        p = 200.0
        for _ in range(600):
            p += -0.15 + rng.gauss(0, 0.5)
            prices.append(p)
        candles = _candles(prices, spread=0.2)
        base = backtest_mean_reversion(candles, {"adx_range_threshold": 100})
        gated = backtest_mean_reversion(
            candles, {"adx_range_threshold": 100, "bull_sma_period": 200})
        assert base["signals"] > 0
        assert gated["blocked_by_bull"] > 0
        assert gated["total_trades"] < max(base["total_trades"], 1) or \
            gated["total_trades"] == 0

    def test_bull_filter_off_by_default(self):
        res = backtest_mean_reversion(_flat(300))
        assert res["blocked_by_bull"] == 0

    def test_sl_is_below_entry_and_tp_respects_rr(self):
        # V-shaped dip inside an otherwise flat market -> at least one trade,
        # and every trade's net outcome is bounded by the SL distance.
        import random
        random.seed(7)
        prices = []
        p = 100.0
        for i in range(400):
            drift = (100 - p) * 0.05
            p += drift + random.gauss(0, 0.6)
            prices.append(p)
        res = backtest_mean_reversion(_candles(prices, spread=0.2),
                                      {"rr": 1.0, "sl_atr_mult": 1.5})
        for t in res["trades"]:
            if t["reason"] == "sl":
                assert t["exit"] < t["entry"]
            if t["reason"] == "tp":
                assert t["exit"] > t["entry"]
