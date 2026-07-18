"""
Trend-following backtest core -- pure simulation over OHLCV candles.

Mirrors the strategy that actually runs live (strategies/trend_following.py):
entry on a bullish EMA(9/21) cross while RSI < overbought; exit on a bearish
cross, on RSI > overbought, or on the risk_manager's fixed SL/TP percentages.

Why this exists: the 432-coin and MR v2 studies validated *mean reversion*,
but every live signal in ct_signal_db.json came from trend following, which
had never been tested on a large sample. Same rigor, same fee/slippage model
as mr_backtest so the numbers are comparable.

Long-only (spot). Conservative intrabar rule: if both SL and TP sit inside
one candle's range, the SL is assumed to fill first.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from mr_backtest import _metrics
from regime import detect_regime

_O, _H, _L, _C, _V = 1, 2, 3, 4, 5


def _ema_series(closes: Sequence[float], period: int) -> List[float]:
    """EMA with adjust=False, matching pandas ewm in the live strategy."""
    alpha = 2.0 / (period + 1)
    out: List[float] = []
    prev = closes[0]
    for i, c in enumerate(closes):
        prev = c if i == 0 else alpha * c + (1 - alpha) * prev
        out.append(prev)
    return out


def _rsi_series(closes: Sequence[float], period: int = 14) -> List[float]:
    """Wilder RSI (ewm com=period-1), matching the live strategy."""
    n = len(closes)
    out = [50.0] * n
    if n <= period:
        return out
    alpha = 1.0 / period
    avg_gain = avg_loss = None
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        if avg_gain is None:
            avg_gain, avg_loss = gain, loss
        else:
            avg_gain = alpha * gain + (1 - alpha) * avg_gain
            avg_loss = alpha * loss + (1 - alpha) * avg_loss
        if i < period:
            continue
        if avg_loss == 0:
            out[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - (100 / (1 + rs))
    return out


def backtest_trend_following(
    ohlcv: List[Sequence[float]],
    params: Dict[str, Any] | None = None,
    fee_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> Dict[str, Any]:
    """Simulate the live EMA-cross trend follower on OHLCV data.

    params (all optional):
      ema_short (9), ema_long (21), rsi_period (14), rsi_overbought (70),
      sl_pct (5.0), tp_pct (10.0), trailing_pct (0 = off),
      max_hold_bars (0 = off), warmup (60),
      trend_sma_period (0 = off) -- master switch: only enter above this SMA,
      adx_trend_threshold (0 = off) -- only enter when ADX says "trending",
      intrabar_stops (True) -- SL/TP fill inside the candle (OCO semantics);
      set False to reproduce the live close-only evaluation.
    """
    p = params or {}
    ema_short_p = p.get("ema_short", 9)
    ema_long_p = p.get("ema_long", 21)
    rsi_period = p.get("rsi_period", 14)
    rsi_overbought = p.get("rsi_overbought", 70)
    sl_pct = p.get("sl_pct", 5.0)
    tp_pct = p.get("tp_pct", 10.0)
    trailing_pct = p.get("trailing_pct", 0.0)
    max_hold_bars = p.get("max_hold_bars", 0)
    warmup = max(p.get("warmup", 60), ema_long_p + 10, rsi_period + 1)
    trend_sma_period = p.get("trend_sma_period", 0)
    adx_trend_threshold = p.get("adx_trend_threshold", 0)
    adx_period = p.get("adx_period", 14)
    regime_lookback = p.get("regime_lookback", 100)
    intrabar = p.get("intrabar_stops", True)

    round_trip_cost_pct = 2 * fee_pct + 2 * slippage_pct

    closes = [c[_C] for c in ohlcv]
    if len(closes) <= warmup + 2:
        return _metrics([], 0, 0, 0)

    ema_s = _ema_series(closes, ema_short_p)
    ema_l = _ema_series(closes, ema_long_p)
    rsi = _rsi_series(closes, rsi_period)

    # Rolling SMA for the trend master switch (fails closed on thin data).
    sma_ok: List[bool] = [True] * len(ohlcv)
    if trend_sma_period > 0:
        sma_ok = [False] * len(ohlcv)
        running = 0.0
        for i, c in enumerate(closes):
            running += c
            if i >= trend_sma_period:
                running -= closes[i - trend_sma_period]
            if i >= trend_sma_period - 1:
                sma_ok[i] = c > running / trend_sma_period

    trades: List[Dict[str, Any]] = []
    position = None
    signals = 0
    blocked_by_regime = 0
    blocked_by_sma = 0

    for i in range(warmup, len(ohlcv)):
        candle = ohlcv[i]
        ts, high, low, close = candle[0], candle[_H], candle[_L], candle[_C]

        if position is not None:
            if close > position["peak"]:
                position["peak"] = close
            exit_price = reason = None

            if intrabar and low <= position["sl"]:
                exit_price, reason = position["sl"], "sl"
            elif intrabar and high >= position["tp"]:
                exit_price, reason = position["tp"], "tp"
            elif not intrabar and close <= position["sl"]:
                exit_price, reason = close, "sl"
            elif not intrabar and close >= position["tp"]:
                exit_price, reason = close, "tp"
            elif trailing_pct > 0 and close <= position["peak"] * (1 - trailing_pct / 100):
                exit_price, reason = close, "trail"
            elif ema_s[i] < ema_l[i] and ema_s[i - 1] >= ema_l[i - 1]:
                exit_price, reason = close, "cross"
            elif rsi[i] > rsi_overbought:
                exit_price, reason = close, "rsi"
            elif max_hold_bars and i - position["entry_idx"] >= max_hold_bars:
                exit_price, reason = close, "time"

            if exit_price is not None:
                exit_fill = exit_price * (1 - slippage_pct / 100) if reason != "sl" else exit_price
                gross_pct = (exit_fill - position["entry"]) / position["entry"] * 100
                trades.append({
                    "entry_ts": position["entry_ts"],
                    "exit_ts": ts,
                    "entry": position["entry"],
                    "exit": exit_fill,
                    "net_pct": round(gross_pct - round_trip_cost_pct, 4),
                    "reason": reason,
                    "hold_bars": i - position["entry_idx"],
                })
                position = None
            continue

        bullish_cross = ema_s[i - 1] <= ema_l[i - 1] and ema_s[i] > ema_l[i]
        if not (bullish_cross and rsi[i] < rsi_overbought):
            continue

        signals += 1
        if not sma_ok[i]:
            blocked_by_sma += 1
            continue
        if adx_trend_threshold:
            window = ohlcv[max(0, i - regime_lookback): i + 1]
            regime, details = detect_regime(window, adx_period=adx_period)
            if details.get("adx", 0) < adx_trend_threshold:
                blocked_by_regime += 1
                continue

        entry = close * (1 + slippage_pct / 100)
        position = {
            "entry": entry,
            "sl": entry * (1 - sl_pct / 100),
            "tp": entry * (1 + tp_pct / 100),
            "peak": entry,
            "entry_idx": i,
            "entry_ts": ts,
        }

    # Force-close any open position at the last close.
    if position is not None:
        last = ohlcv[-1]
        gross_pct = (last[_C] - position["entry"]) / position["entry"] * 100
        trades.append({
            "entry_ts": position["entry_ts"],
            "exit_ts": last[0],
            "entry": position["entry"],
            "exit": last[_C],
            "net_pct": round(gross_pct - round_trip_cost_pct, 4),
            "reason": "eod",
            "hold_bars": len(ohlcv) - 1 - position["entry_idx"],
        })

    res = _metrics(trades, signals, blocked_by_regime, blocked_by_sma)
    res["exit_reasons"] = {
        r: sum(1 for t in trades if t["reason"] == r)
        for r in ("tp", "sl", "trail", "cross", "rsi", "time", "eod")
    }
    return res
