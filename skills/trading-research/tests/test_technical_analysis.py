"""Tests for the pure calculation functions in technical_analysis.py.

skills/trading-research/ had zero test coverage before this file. Focused on
the indicator math (SMA/EMA/RSI/MACD/Bollinger/support-resistance/volume/trend)
since those are pure functions of already-fetched kline data — no network
mocking required. fetch_klines/main (I/O + CLI) are intentionally untested.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from technical_analysis import (
    analyze,
    analyze_volume,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_ema_series,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    find_support_resistance,
    get_trend_signal,
)


def _klines(closes, highs=None, lows=None, volumes=None):
    highs = highs or closes
    lows = lows or closes
    volumes = volumes or [1000.0] * len(closes)
    return [
        {"close": c, "high": h, "low": l, "volume": v}
        for c, h, l, v in zip(closes, highs, lows, volumes)
    ]


def _ref_ema_series(values, period):
    """Independent EMA implementation (v*k + e*(1-k) instead of (v-e)*k+e) for cross-check."""
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    out = [e]
    for v in values[period:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def _ref_macd(prices, fast=12, slow=26, signal=9):
    fast_series = _ref_ema_series(prices, fast)
    slow_series = _ref_ema_series(prices, slow)
    offset = slow - fast
    macd_series = [f - s for f, s in zip(fast_series[offset:], slow_series)]
    signal_series = _ref_ema_series(macd_series, signal)
    if not signal_series:
        return None, None, None
    return macd_series[-1], signal_series[-1], macd_series[-1] - signal_series[-1]


class TestSma:
    def test_insufficient_data_returns_none(self):
        assert calculate_sma([1, 2], 3) is None

    def test_averages_last_n_values(self):
        assert calculate_sma([1, 2, 3, 4, 5], 3) == 4

    def test_exact_length(self):
        assert calculate_sma([1, 2, 3], 3) == 2


class TestEma:
    def test_insufficient_data_returns_none(self):
        assert calculate_ema([1, 2], 3) is None
        assert calculate_ema_series([1, 2], 3) == []

    def test_seeded_with_sma_then_recurses(self):
        # period=2 seed = mean([10,20])=15; next = (30-15)*(2/3)+15 = 25
        assert calculate_ema_series([10, 20, 30], 2) == [15, 25.0]
        assert calculate_ema([10, 20, 30], 2) == 25.0

    def test_ema_is_last_value_of_series(self):
        prices = [100 + i for i in range(30)]
        assert calculate_ema(prices, 12) == calculate_ema_series(prices, 12)[-1]


class TestRsi:
    def test_insufficient_data_returns_none(self):
        assert calculate_rsi([1, 2, 3], period=5) is None

    def test_all_gains_is_100(self):
        assert calculate_rsi([1, 2, 3, 4, 5], period=4) == 100

    def test_balanced_gains_and_losses_is_50(self):
        # changes: +1,-1,+1,-1 -> avg_gain == avg_loss -> RSI 50
        assert calculate_rsi([1, 2, 1, 2, 1], period=4) == 50.0

    def test_all_losses_approaches_zero(self):
        assert calculate_rsi([5, 4, 3, 2, 1], period=4) == 0


class TestMacd:
    def test_insufficient_data_returns_none_tuple(self):
        # needs slow + signal - 1 points
        prices = [100 + i for i in range(26 + 9 - 2)]  # one short
        assert calculate_macd(prices) == (None, None, None)

    def test_matches_independent_reference_implementation(self):
        prices = [100 + 10 * math.sin(i / 5) + i * 0.3 for i in range(80)]
        got = calculate_macd(prices, fast=12, slow=26, signal=9)
        want = _ref_macd(prices, fast=12, slow=26, signal=9)
        assert got == pytest.approx(want)

    def test_signal_line_is_ema_of_macd_not_a_copy_of_it(self):
        # Regression test: calculate_macd used to set signal_line = macd_line
        # verbatim ("In production, would track MACD history"), which made
        # the histogram always 0 and any crossover signal meaningless.
        prices = [100 + 10 * math.sin(i / 5) + i * 0.3 for i in range(80)]
        macd_line, signal_line, histogram = calculate_macd(prices)
        assert signal_line != macd_line
        assert histogram == macd_line - signal_line

    def test_custom_periods_align_series_correctly(self):
        # Hand-traceable case: fast=2, slow=3, signal=2.
        prices = [1, 2, 4, 7, 11, 16, 22, 29]
        got = calculate_macd(prices, fast=2, slow=3, signal=2)
        want = _ref_macd(prices, fast=2, slow=3, signal=2)
        assert got == pytest.approx(want)
        assert got[0] is not None


class TestBollingerBands:
    def test_insufficient_data_returns_none_tuple(self):
        assert calculate_bollinger_bands([1, 2, 3], period=5) == (None, None, None)

    def test_bands_straddle_the_moving_average(self):
        prices = [2, 4, 4, 4, 5, 5, 7, 9]
        upper, middle, lower = calculate_bollinger_bands(prices, period=8, std_dev=2)
        assert middle == 5
        assert upper > middle > lower
        assert upper - middle == middle - lower  # symmetric band


class TestSupportResistance:
    def test_insufficient_data_returns_empty_lists(self):
        assert find_support_resistance(_klines([1, 2, 3]), lookback=20) == ([], [])

    def test_finds_overall_high_and_low(self):
        closes = [10, 12, 9, 15, 8, 11, 13, 7, 14, 10, 12, 9, 15, 8, 11, 13, 7, 14, 10, 16]
        resistance, support = find_support_resistance(_klines(closes), lookback=20)
        assert max(closes) in resistance
        assert min(closes) in support


class TestVolumeAnalysis:
    def test_insufficient_data_returns_none(self):
        assert analyze_volume(_klines([1, 2]), period=20) is None

    def test_classifies_high_normal_low(self):
        base = [1000.0] * 19
        high = analyze_volume(_klines([1] * 20, volumes=base + [2500.0]), period=20)
        assert high["status"] == "High"

        normal = analyze_volume(_klines([1] * 20, volumes=base + [1000.0]), period=20)
        assert normal["status"] == "Normal"

        low = analyze_volume(_klines([1] * 20, volumes=base + [100.0]), period=20)
        assert low["status"] == "Low"


class TestTrendSignal:
    def test_insufficient_data(self):
        assert get_trend_signal([1, 2, 3]) == "Insufficient data"

    def test_strong_uptrend_when_price_above_both_smas(self):
        prices = [100 + i for i in range(60)]
        assert get_trend_signal(prices) == "Strong Uptrend"

    def test_strong_downtrend_when_price_below_both_smas(self):
        prices = [200 - i for i in range(60)]
        assert get_trend_signal(prices) == "Strong Downtrend"

    def test_uptrend_without_enough_data_for_sma50(self):
        prices = [100 + i for i in range(25)]
        assert get_trend_signal(prices) == "Uptrend"


class TestAnalyze:
    """analyze() aggregates every indicator above into the signal dict that
    technical_analysis.py's CLI (and callers via --json) actually returns.
    It had zero coverage even though the pure indicator functions did.
    """

    def test_flat_price_data_does_not_crash_on_zero_width_bollinger_band(self, capsys):
        # Regression: when every close in the BB period is identical, stdev
        # is 0 so bb_upper == bb_lower. The old code computed
        # (price - bb_lower) / (bb_upper - bb_lower) unconditionally, which
        # raised ZeroDivisionError for any flat/no-movement pair (e.g. a
        # stablecoin pair, or a low-liquidity symbol with a stale kline feed)
        # instead of returning a result.
        klines = [
            {"close": 100.0, "high": 100.0, "low": 100.0, "volume": 1000.0}
            for _ in range(60)
        ]
        result = analyze(klines)
        assert result["bollinger_bands"] == {"upper": 100.0, "middle": 100.0, "lower": 100.0}
        capsys.readouterr()

    def test_returns_expected_shape_and_values_for_trending_data(self, capsys):
        prices = [100 + i * 0.5 + 10 * math.sin(i / 5) for i in range(80)]
        klines = [
            {"close": c, "high": c + 1, "low": c - 1, "volume": 1000.0 + i}
            for i, c in enumerate(prices)
        ]
        result = analyze(klines)
        capsys.readouterr()

        assert result["price"] == prices[-1]
        assert result["sma_20"] == calculate_sma(prices, 20)
        assert result["rsi"] == calculate_rsi(prices, 14)
        assert result["macd"]["line"] == calculate_macd(prices)[0]
        assert result["bollinger_bands"]["upper"] == calculate_bollinger_bands(prices, 20)[0]
        assert result["support"] and result["resistance"]
        assert result["volume"]["current"] == klines[-1]["volume"]
        assert isinstance(result["signals"], list)

    def test_insufficient_history_omits_indicators_instead_of_crashing(self, capsys):
        # Fewer than 20 candles: SMA(20)/BB/volume/support-resistance/trend
        # all fall back to None/"Insufficient data" rather than raising.
        klines = [
            {"close": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i, "volume": 500.0}
            for i in range(10)
        ]
        result = analyze(klines)
        capsys.readouterr()

        assert result["sma_20"] is None
        assert result["bollinger_bands"] == {"upper": None, "middle": None, "lower": None}
        assert result["volume"] is None
        assert result["trend"] == "Insufficient data"

    def test_overbought_rsi_produces_a_warning_signal(self, capsys):
        # Monotonically rising prices for the whole window -> every change is
        # a gain -> RSI 100 -> the overbought warning must be in `signals`.
        prices = [100 + i for i in range(30)]
        klines = [{"close": c, "high": c, "low": c, "volume": 1000.0} for c in prices]
        result = analyze(klines)
        capsys.readouterr()

        assert result["rsi"] == 100
        assert any("overbought" in s.lower() for s in result["signals"])
