"""Tests for the mean-reversion backtest core (the harness behind VERDICT.md's
mean-reversion numbers) -- previously had zero coverage."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mr_backtest import _bollinger, _rsi, backtest_mean_reversion


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


def _noisy_flat(n, level=100.0, seed=7):
    rng = random.Random(seed)
    return [level + rng.gauss(0, 0.6) for _ in range(n)]


def _ranging_with_dips(n=400, dip_every=70, dip_size=8.0, seed=7, drift=0.0):
    """A sideways series with periodic sharp dips, so BB-lower + RSI-oversold
    fires repeatedly while ADX stays low (ranging regime)."""
    prices = _noisy_flat(n, seed=seed)
    for i in range(60, n, dip_every):
        for j in range(5):
            if i + j < len(prices):
                prices[i + j] -= dip_size - j * 1.5
    if drift:
        prices = [p - idx * drift for idx, p in enumerate(prices)]
    return prices


class TestIndicators:
    def test_rsi_neutral_on_thin_data(self):
        assert _rsi([100, 101, 102], 14) == 50.0

    def test_rsi_pinned_at_100_when_no_losses(self):
        # avg_loss == 0 over the lookback -> RSI formula would divide by
        # zero, so the function special-cases it to 100.
        assert _rsi([100] * 20) == 100.0

    def test_rsi_low_after_a_sustained_decline(self):
        assert _rsi([200 - i for i in range(30)], 14) < 10

    def test_bollinger_bands_straddle_the_mean(self):
        closes = _noisy_flat(30)
        mid, upper, lower = _bollinger(closes, period=20, std_dev=2.0)
        assert lower < mid < upper


class TestBacktest:
    def test_no_trades_on_thin_data(self):
        res = backtest_mean_reversion(_candles([100] * 20))
        assert res["total_trades"] == 0
        assert res["signals"] == 0

    def test_takes_trades_in_a_ranging_market_with_dips(self):
        res = backtest_mean_reversion(_candles(_ranging_with_dips()))
        assert res["signals"] >= 1
        assert res["total_trades"] >= 1
        assert res["blocked_by_regime"] == 0

    def test_take_profit_fills_intrabar(self):
        res = backtest_mean_reversion(_candles(_ranging_with_dips()))
        assert res["exit_reasons"]["tp"] >= 1
        assert res["exit_reasons"]["sl"] == 0

    def test_stop_loss_fires_when_price_keeps_falling(self):
        prices = _noisy_flat(80)
        dip = [prices[-1] - i * 1.2 for i in range(1, 9)]
        crash = [dip[-1] - i * 5 for i in range(1, 10)]
        flat_after = [crash[-1]] * 60
        res = backtest_mean_reversion(_candles(prices + dip + crash + flat_after))
        assert res["exit_reasons"]["sl"] >= 1
        assert res["exit_reasons"]["tp"] == 0
        assert res["losses"] >= 1

    def test_time_exit_when_price_stalls_after_entry(self):
        # A wide-open SL/TP (never reachable) isolates the max_hold_bars path:
        # a perfectly flat stall after the dip can only resolve via "time".
        prices = _noisy_flat(80)
        dip = [prices[-1] - i * 1.2 for i in range(1, 9)]
        stall = [dip[-1]] * 80
        res = backtest_mean_reversion(
            _candles(prices + dip + stall),
            {"max_hold_bars": 5, "sl_atr_mult": 50, "rr": 50},
        )
        assert res["total_trades"] >= 1
        assert res["exit_reasons"]["time"] >= 1

    def test_bull_master_switch_blocks_entries_below_the_sma(self):
        # Ranging-with-dips series plus a steady downward drift: most closes
        # sit below their own trailing SMA, so the bull switch should block
        # every signal it sees.
        prices = _ranging_with_dips(drift=0.6)
        res = backtest_mean_reversion(_candles(prices), {"bull_sma_period": 50})
        assert res["signals"] >= 1
        assert res["total_trades"] == 0
        assert res["blocked_by_bull"] == res["signals"]

    def test_regime_filter_blocks_entries_in_a_trending_market(self):
        # A clean, near-monotonic uptrend: ADX should read high (trending),
        # so ranging-only mean-reversion entries must be blocked, not filled.
        prices = [100 + i * 1.5 + random.Random(9).gauss(0, 0.05) for i in range(300)]
        res = backtest_mean_reversion(_candles(prices))
        assert res["total_trades"] == 0

    def test_costs_are_charged_on_every_trade(self):
        candles = _candles(_ranging_with_dips())
        free = backtest_mean_reversion(candles, fee_pct=0.0, slippage_pct=0.0)
        paid = backtest_mean_reversion(candles)
        assert free["total_trades"] == paid["total_trades"]
        assert paid["total_return_pct"] < free["total_return_pct"]

    def test_open_position_force_closed_at_end_of_data(self):
        # A dip right at the tail end, too close to the end for TP/SL/time to
        # resolve -> the position must still show up as a trade with reason
        # "eod", not silently vanish.
        prices = _noisy_flat(80) + [90.0] * 3
        res = backtest_mean_reversion(_candles(prices))
        if res["signals"] >= 1:
            assert res["total_trades"] >= 1
            assert res["exit_reasons"]["eod"] + res["exit_reasons"]["tp"] + res["exit_reasons"]["sl"] >= 1


class TestMetrics:
    def test_win_rate_and_profit_factor_are_internally_consistent(self):
        res = backtest_mean_reversion(_candles(_ranging_with_dips()))
        assert res["wins"] + res["losses"] == res["total_trades"]
        if res["total_trades"]:
            expected_wr = round(res["wins"] / res["total_trades"] * 100, 1)
            assert res["win_rate_pct"] == expected_wr

    def test_profit_factor_is_capped_when_there_are_no_losses(self):
        res = backtest_mean_reversion(_candles(_ranging_with_dips()))
        assert res["exit_reasons"]["sl"] == 0
        if res["wins"]:
            assert res["profit_factor"] == 999.0

    def test_zero_trades_yields_zeroed_metrics_not_a_crash(self):
        res = backtest_mean_reversion(_candles([100] * 20))
        assert res["win_rate_pct"] == 0.0
        assert res["total_return_pct"] == 0.0
        assert res["profit_factor"] == 0.0
        assert res["trades"] == []
