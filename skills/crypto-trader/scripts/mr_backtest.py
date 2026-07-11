"""
Mean-reversion backtest core -- pure simulation over OHLCV candles.

Fixes the two failure modes found in the 2026-07 backtests (win rate 43-56%
but net losses):
  1. No regime gate -> entries during trends got run over. Entries here are
     blocked unless ADX says the market is ranging (see regime.py).
  2. No hard stop -> avg loss ended up larger than avg win. Every position
     gets an ATR-based SL fixed BEFORE entry and a TP at a fixed R multiple,
     per the fund rules (SL never loosened; OCO semantics).

Long-only (spot). Conservative intrabar rule: if both SL and TP are inside
one candle's range, the SL is assumed to fill first.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from regime import atr, detect_regime

_O, _H, _L, _C, _V = 1, 2, 3, 4, 5


def _rsi(closes: Sequence[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [d if d > 0 else 0.0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0.0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bollinger(closes: Sequence[float], period: int = 20, std_dev: float = 2.0):
    window = list(closes[-period:])
    mid = sum(window) / period
    std = (sum((c - mid) ** 2 for c in window) / period) ** 0.5
    return mid, mid + std_dev * std, mid - std_dev * std


def backtest_mean_reversion(
    ohlcv: List[Sequence[float]],
    params: Dict[str, Any] | None = None,
    fee_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> Dict[str, Any]:
    """Simulate regime-gated mean reversion with ATR stops on OHLCV data.

    params (all optional):
      bb_period (20), bb_std (2.0), rsi_period (14), rsi_oversold (30),
      atr_period (14), sl_atr_mult (1.5), rr (1.0), max_hold_bars (48),
      adx_period (14), adx_range_threshold (20), warmup (60)
    """
    p = params or {}
    bb_period = p.get("bb_period", 20)
    bb_std = p.get("bb_std", 2.0)
    rsi_period = p.get("rsi_period", 14)
    rsi_oversold = p.get("rsi_oversold", 30)
    atr_period = p.get("atr_period", 14)
    sl_atr_mult = p.get("sl_atr_mult", 1.5)
    rr = p.get("rr", 1.0)
    max_hold_bars = p.get("max_hold_bars", 48)
    adx_period = p.get("adx_period", 14)
    adx_range_threshold = p.get("adx_range_threshold", 20.0)
    warmup = max(p.get("warmup", 60), bb_period, rsi_period + 1, 2 * adx_period + 1)
    regime_lookback = p.get("regime_lookback", 100)

    round_trip_cost_pct = 2 * fee_pct + 2 * slippage_pct

    trades: List[Dict[str, Any]] = []
    position = None  # dict(entry, sl, tp, entry_idx)
    blocked_by_regime = 0
    signals = 0

    for i in range(warmup, len(ohlcv)):
        candle = ohlcv[i]
        ts, high, low, close = candle[0], candle[_H], candle[_L], candle[_C]

        if position is not None:
            exit_price = None
            reason = None
            if low <= position["sl"]:
                exit_price, reason = position["sl"], "sl"
            elif high >= position["tp"]:
                exit_price, reason = position["tp"], "tp"
            elif i - position["entry_idx"] >= max_hold_bars:
                exit_price, reason = close, "time"

            if exit_price is not None:
                gross_pct = (exit_price - position["entry"]) / position["entry"] * 100
                net_pct = gross_pct - round_trip_cost_pct
                trades.append({
                    "entry_ts": position["entry_ts"],
                    "exit_ts": ts,
                    "entry": position["entry"],
                    "exit": exit_price,
                    "net_pct": round(net_pct, 4),
                    "reason": reason,
                    "hold_bars": i - position["entry_idx"],
                })
                position = None
            continue

        closes = [c[_C] for c in ohlcv[max(0, i - warmup): i + 1]]
        mid, _upper, lower = _bollinger(closes, bb_period, bb_std)
        rsi_val = _rsi(closes, rsi_period)

        if close < lower and rsi_val < rsi_oversold:
            signals += 1
            regime_window = ohlcv[max(0, i - regime_lookback): i + 1]
            regime, _details = detect_regime(
                regime_window, adx_period=adx_period,
                adx_range_threshold=adx_range_threshold,
            )
            if regime != "ranging":
                blocked_by_regime += 1
                continue

            atr_val = atr(regime_window, atr_period)
            if atr_val <= 0:
                continue
            entry = close * (1 + slippage_pct / 100)
            sl = entry - sl_atr_mult * atr_val
            if sl <= 0 or sl >= entry:
                continue
            tp = entry + rr * (entry - sl)
            position = {
                "entry": entry, "sl": sl, "tp": tp,
                "entry_idx": i, "entry_ts": ts,
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

    return _metrics(trades, signals, blocked_by_regime)


def _metrics(trades: List[Dict[str, Any]], signals: int, blocked: int) -> Dict[str, Any]:
    wins = [t for t in trades if t["net_pct"] > 0]
    losses = [t for t in trades if t["net_pct"] <= 0]
    total = len(trades)
    gross_win = sum(t["net_pct"] for t in wins)
    gross_loss = abs(sum(t["net_pct"] for t in losses))

    # Compound the per-trade net returns into a total return.
    equity = 1.0
    peak, max_dd = 1.0, 0.0
    for t in trades:
        equity *= 1 + t["net_pct"] / 100
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / total * 100, 1) if total else 0.0,
        "total_return_pct": round((equity - 1) * 100, 2),
        "avg_win_pct": round(gross_win / len(wins), 3) if wins else 0.0,
        "avg_loss_pct": round(-gross_loss / len(losses), 3) if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
        "max_drawdown_pct": round(max_dd, 2),
        "signals": signals,
        "blocked_by_regime": blocked,
        "exit_reasons": {
            r: sum(1 for t in trades if t["reason"] == r)
            for r in ("tp", "sl", "time", "eod")
        },
        "trades": trades,
    }
