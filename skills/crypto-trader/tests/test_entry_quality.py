"""Tests for the ported entry-quality filters and tiered trailing stop."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from entry_quality import entry_quality, trend_filter  # noqa: E402
from risk_manager import RiskManager  # noqa: E402


def candle(o, h, l, c, v):
    return [0, o, h, l, c, v]


def _healthy_uptrend(n=21, vol=100.0):
    """Steady uptrend with strong-bodied green candles and stable volume."""
    rows = []
    price = 100.0
    for _ in range(n):
        o = price
        c = price + 1.0
        rows.append(candle(o, c + 0.1, o - 0.1, c, vol))
        price = c
    return rows


# --- entry_quality --------------------------------------------------------

def test_insufficient_data_fails_open():
    ok, reason = entry_quality([candle(1, 1, 1, 1, 1)] * 3)
    assert ok is True
    assert "insufficient" in reason


def test_clean_uptrend_passes():
    ok, reason = entry_quality(_healthy_uptrend())
    assert ok is True
    assert reason == "All checks passed"


def test_volume_crash_blocks():
    rows = _healthy_uptrend()
    for i in (-4, -3, -2):  # completed candles feeding recent_vol_3
        rows[i][5] = 3.0
    ok, reason = entry_quality(rows)
    assert ok is False
    assert "Volume crash" in reason


def test_dead_cat_bounce_blocks():
    rows = _healthy_uptrend(n=21)
    # Drop a recent low then a weak red bounce candle (tiny body, long wick).
    rows[-3] = candle(101, 101, 100, 100.0, 100)   # recent low ~100
    rows[-1] = candle(103.1, 106.0, 102.9, 103.0, 100)  # +3% bounce, red, tiny body
    ok, reason = entry_quality(rows)
    assert ok is False
    assert "Dead cat bounce" in reason


def test_doji_is_warning_not_block():
    rows = []
    for _ in range(21):
        rows.append(candle(100.0, 100.4, 99.6, 100.05, 100))  # flat-ish, green
    rows[-1] = candle(100.0, 101.0, 99.0, 100.05, 100)  # tiny body, wide range -> doji
    ok, reason = entry_quality(rows)
    assert ok is True
    assert "Doji" in reason


# --- trend_filter ---------------------------------------------------------

def test_trend_thin_data_fails_open():
    ok, reason = trend_filter([candle(1, 1, 1, 1, 1)] * 10)
    assert ok is True


def test_trend_oversold_allows():
    rows = [candle(c, c + 0.5, c - 0.5, c, 100) for c in [120 - i for i in range(25)]]
    ok, reason = trend_filter(rows)
    assert ok is True  # monotonic drop -> RSI deeply oversold -> allow reversal


def test_trend_downtrend_blocks():
    # Steep early decline (established downtrend) + choppy tail so RSI stays
    # above the deep-oversold threshold (>25) and the price-below-SMA20 block
    # fires rather than the oversold-reversal allowance.
    closes = [120 - i * 0.8 for i in range(14)]
    closes += [108, 109, 107.5, 108.5, 107, 108.5, 107, 108.5, 107.5, 108, 107]
    rows = [candle(c, c + 0.3, c - 0.3, c, 100) for c in closes]
    ok, reason = trend_filter(rows)
    assert ok is False
    assert "bearish" in reason


# --- tiered trailing stop -------------------------------------------------

@pytest.fixture
def risk():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["CRYPTO_RISK_STATE_PATH"] = str(Path(tmp) / "state.json")
        yield RiskManager()


def test_trailing_none_when_flat(risk):
    assert risk.tiered_trailing_stop(100.0, 100.0, None) is None


def test_trailing_breakeven_above_2pct(risk):
    assert risk.tiered_trailing_stop(100.0, 102.5, None) == pytest.approx(100.0)


def test_trailing_locks_profit_above_3pct(risk):
    assert risk.tiered_trailing_stop(100.0, 104.0, None) == pytest.approx(101.5)


def test_trailing_only_ratchets_up(risk):
    # Price fell back to +2.5%; stop must not drop below the locked 101.5.
    assert risk.tiered_trailing_stop(100.0, 102.5, prev_stop=101.5) == pytest.approx(101.5)
