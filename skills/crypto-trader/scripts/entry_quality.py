"""
Entry-quality filters -- ported from the binance-spot-trader monolith.

Pure functions over OHLCV candles that reduce false BUY entries. They take the
ccxt OHLCV shape (list of ``[timestamp, open, high, low, close, volume]``) so
they plug straight into ExchangeManager.get_ohlcv().

Ported layers (from the monolith's entry_quality_filter / pair_trend_ok):
  1. Volume confirmation -- a bounce needs volume support.
  2. Bounce quality -- reject dead-cat bounces / weak follow-through.
  3. Candlestick quality -- doji vs engulfing / morning-star.
  4. 4h trend filter -- block BUYs in an established downtrend.

The monolith's file-backed "false BUY memory" (layer 4 there) is intentionally
left out; it is infra state, not pure logic. Everything here is deterministic
and unit-tested. All checks fail OPEN: with too little data they allow the
trade rather than block it.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

# ccxt OHLCV column indices.
_O, _H, _L, _C, _V = 1, 2, 3, 4, 5

_HARD_BLOCKS = ("Volume crash", "Dead cat bounce")
_SOFT_BLOCKS = ("Low volume", "First green after 3 reds")
_WARNINGS = ("Doji", "Weak bounce", "Weak volume")


def _rsi(prices: Sequence[float], period: int = 14) -> float:
    if len(prices) <= period:
        return 50.0
    deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def entry_quality(ohlcv: List[Sequence[float]]) -> Tuple[bool, str]:
    """Return (passed, reason) for a candidate BUY entry.

    Fails open (returns True) when there are too few candles to judge.
    """
    if not ohlcv or len(ohlcv) < 5:
        return True, "insufficient data -- allow"

    opens = [c[_O] for c in ohlcv]
    highs = [c[_H] for c in ohlcv]
    lows = [c[_L] for c in ohlcv]
    closes = [c[_C] for c in ohlcv]
    volumes = [c[_V] for c in ohlcv]
    current = closes[-1]
    reasons: List[str] = []

    # Layer 1: volume confirmation (use completed candles, skip latest).
    completed_vol = volumes[-4:-1] if len(volumes) >= 4 else volumes[:-1]
    if len(volumes) >= 21:
        avg_vol_20 = sum(volumes[-21:-1]) / 20
    else:
        avg_vol_20 = sum(volumes[:-1]) / max(len(volumes) - 1, 1)
    if completed_vol:
        recent_vol_3 = sum(completed_vol[-3:]) / min(len(completed_vol), 3)
    else:
        recent_vol_3 = 0
    vol_ratio = recent_vol_3 / avg_vol_20 if avg_vol_20 > 0 else 0

    if vol_ratio < 0.2:
        reasons.append(f"Volume crash: {vol_ratio:.1f}x avg -- manipulation risk")
    elif vol_ratio < 0.5:
        reasons.append(f"Low volume: {vol_ratio:.1f}x avg -- weak conviction")

    # Layer 2: bounce quality (follow-through verification).
    lookback_6 = closes[-7:-1] if len(closes) >= 7 else closes[:-1]
    recent_low = min(lookback_6) if lookback_6 else current
    bounce_pct = ((current - recent_low) / recent_low) * 100 if recent_low > 0 else 0

    if bounce_pct > 2.0:
        last_body = abs(current - opens[-1])
        last_range = highs[-1] - lows[-1]
        body_ratio = last_body / last_range if last_range > 0 else 0

        if body_ratio < 0.3 and current < opens[-1]:
            reasons.append(f"Dead cat bounce: body={body_ratio:.1%} + red candle")
        elif body_ratio < 0.25:
            reasons.append(f"Weak bounce: body={body_ratio:.1%} -- no conviction")

        if len(closes) >= 4:
            prev_3_green = sum(1 for i in range(-4, -1) if closes[i] >= opens[i])
            if prev_3_green == 0 and closes[-1] > opens[-1]:
                if body_ratio < 0.4 or vol_ratio < 0.9:
                    reasons.append(
                        f"First green after 3 reds -- weak confirmation "
                        f"(body={body_ratio:.1%}, vol={vol_ratio:.1f}x)"
                    )

    # Layer 3: candlestick pattern quality.
    body = abs(closes[-1] - opens[-1])
    range_total = highs[-1] - lows[-1]
    body_pct = body / range_total if range_total > 0 else 0

    if body_pct < 0.15:
        reasons.append(f"Doji candle: body={body_pct:.1%} -- indecision")

    if len(ohlcv) >= 2:
        prev_red = closes[-2] < opens[-2]
        engulfs = current > opens[-1] and current > opens[-2] and opens[-1] < closes[-2]
        if prev_red and engulfs:
            reasons = [r for r in reasons if "Doji" not in r]
            if vol_ratio < 0.8:
                reasons.append(f"Engulfing without volume: {vol_ratio:.1f}x")

    if len(closes) >= 3:
        c1_red = closes[-3] < opens[-3]
        rng2 = highs[-2] - lows[-2]
        c2_small = (abs(closes[-2] - opens[-2]) / rng2) if rng2 > 0 else 0
        c3_green = closes[-1] > opens[-1]
        if c1_red and c2_small < 0.3 and c3_green:
            reasons = [r for r in reasons if "Doji" not in r]

    return _decide(reasons)


def _decide(reasons: List[str]) -> Tuple[bool, str]:
    if not reasons:
        return True, "All checks passed"
    reason_str = " | ".join(reasons)
    has_hard = any(any(b in r for b in _HARD_BLOCKS) for r in reasons)
    if has_hard:
        return False, reason_str
    soft_count = sum(1 for r in reasons if any(b in r for b in _SOFT_BLOCKS))
    warn_count = sum(1 for r in reasons if any(w in r for w in _WARNINGS))
    if soft_count >= 2 or (soft_count >= 1 and warn_count >= 2):
        return False, reason_str
    if soft_count >= 1:
        return True, f"CAUTION: {reason_str}"
    return True, f"OK: {reason_str}"


def trend_filter(ohlcv_4h: List[Sequence[float]]) -> Tuple[bool, str]:
    """Block BUYs in an established 4h downtrend (unless deeply oversold).

    Ported from the monolith's pair_trend_ok. Fails open on thin data.
    """
    if not ohlcv_4h or len(ohlcv_4h) < 20:
        return True, "insufficient 4h data -- allow"

    closes = [c[_C] for c in ohlcv_4h]
    highs = [c[_H] for c in ohlcv_4h]
    current = closes[-1]
    sma7 = sum(closes[-7:]) / 7
    sma20 = sum(closes[-20:]) / 20
    r = _rsi(closes)

    if sma7 < sma20:
        if r < 25:
            return True, f"4h bearish but RSI={r:.0f} deep oversold -- allow reversal"
        if current < sma20:
            return False, f"4h bearish: SMA7<SMA20 + price<SMA20 + RSI={r:.0f}"

    # Lower-high pattern: 3 declining ~daily swing highs.
    chunk = 6  # 6 x 4h = 24h
    swing_highs = [max(highs[i:i + chunk]) for i in range(0, len(highs) - chunk, chunk)]
    if len(swing_highs) >= 3:
        a, b, c = swing_highs[-3:]
        if a > b > c and r >= 25:
            return False, f"Lower-high pattern ({a:.4f}>{b:.4f}>{c:.4f}) + RSI={r:.0f}"

    return True, "4h trend OK"
