"""
Regime detection -- classify market as trending vs ranging.

Pure functions over ccxt OHLCV candles (``[ts, open, high, low, close, vol]``).
Mean reversion gets destroyed in trending markets (the 2026-07 432-coin
backtest showed decent win rates but net losses because losing trades rode
against trends). The regime filter blocks mean-reversion entries when the
market is trending, and can gate trend-following the opposite way.

All checks fail CLOSED for mean reversion: with too little data we report
"unknown" and the caller should skip the trade.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

_O, _H, _L, _C, _V = 1, 2, 3, 4, 5


def atr(ohlcv: List[Sequence[float]], period: int = 14) -> float:
    """Average True Range (Wilder smoothing) of the last `period` candles."""
    if len(ohlcv) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(ohlcv)):
        h, l = ohlcv[i][_H], ohlcv[i][_L]
        prev_c = ohlcv[i - 1][_C]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    # Wilder smoothing
    val = sum(trs[:period]) / period
    for tr in trs[period:]:
        val = (val * (period - 1) + tr) / period
    return val


def adx(ohlcv: List[Sequence[float]], period: int = 14) -> float:
    """Average Directional Index (Wilder). Returns 0.0 on thin data."""
    if len(ohlcv) < 2 * period + 1:
        return 0.0

    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, len(ohlcv)):
        h, l = ohlcv[i][_H], ohlcv[i][_L]
        ph, pl, pc = ohlcv[i - 1][_H], ohlcv[i - 1][_L], ohlcv[i - 1][_C]
        up, down = h - ph, pl - l
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    def _wilder(values: List[float]) -> List[float]:
        smoothed = [sum(values[:period])]
        for v in values[period:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / period + v)
        return smoothed

    tr_s, pdm_s, mdm_s = _wilder(trs), _wilder(plus_dm), _wilder(minus_dm)
    dxs = []
    for t, p, m in zip(tr_s, pdm_s, mdm_s):
        if t == 0:
            continue
        pdi, mdi = 100 * p / t, 100 * m / t
        denom = pdi + mdi
        if denom > 0:
            dxs.append(100 * abs(pdi - mdi) / denom)
    if len(dxs) < period:
        return 0.0
    val = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        val = (val * (period - 1) + dx) / period
    return val


def bb_width_pct(closes: Sequence[float], period: int = 20, std_dev: float = 2.0) -> float:
    """Bollinger Band width as % of the middle band. 0.0 on thin data."""
    if len(closes) < period:
        return 0.0
    window = list(closes[-period:])
    mid = sum(window) / period
    if mid == 0:
        return 0.0
    var = sum((c - mid) ** 2 for c in window) / period
    std = var ** 0.5
    return (2 * std_dev * std / mid) * 100


def detect_regime(
    ohlcv: List[Sequence[float]],
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
) -> Tuple[str, Dict[str, float]]:
    """Classify the market regime from OHLCV candles.

    Returns (regime, details) where regime is one of:
      "trending"  -- ADX >= adx_trend_threshold: directional move in progress
      "ranging"   -- ADX <= adx_range_threshold: sideways, mean reversion OK
      "choppy"    -- in between: no clear read, treat as unsafe for MR entries
      "unknown"   -- not enough data (fail closed for mean reversion)
    """
    if len(ohlcv) < 2 * adx_period + 1:
        return "unknown", {"adx": 0.0}

    adx_val = adx(ohlcv, adx_period)
    details = {"adx": round(adx_val, 2)}

    if adx_val >= adx_trend_threshold:
        return "trending", details
    if adx_val <= adx_range_threshold:
        return "ranging", details
    return "choppy", details


def mean_reversion_allowed(ohlcv: List[Sequence[float]], **kwargs) -> Tuple[bool, str]:
    """Gate for mean-reversion entries: only allowed in a ranging regime."""
    regime, details = detect_regime(ohlcv, **kwargs)
    if regime == "ranging":
        return True, f"ranging (ADX={details['adx']})"
    return False, f"{regime} (ADX={details['adx']}) -- mean reversion blocked"
