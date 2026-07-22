"""
Cross-sectional momentum backtest core -- pure simulation over aligned
multi-symbol OHLCV candles.

Why this exists: VERDICT.md (18 Juli 2026) proved both strategy families
tested so far -- mean reversion (mr_backtest.py) and trend following
(tf_backtest.py) -- have no edge. Both bet on a SINGLE coin's own price
history. VERDICT.md names cross-sectional momentum (ranking coins against
EACH OTHER, not against their own past) as an untested avenue with a
different theoretical basis. This module makes that avenue testable with
the same rigor (fee/slippage model, _metrics() report shape) as the two
vetoed strategies, so any result is directly comparable to their tables.

Long-only, top-K, equal-weight, non-overlapping rebalance windows: every
`hold_bars`, rank all symbols by trailing return over `lookback_bars` and
hold the top K until the next rebalance.

This file contains no network calls and no verdict -- see validate_csm.py
to run it against real Binance data. No results have been fabricated or
claimed anywhere in this module; it has not yet been run against real data
in this environment (no exchange access here -- see validate_csm.py docstring).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from mr_backtest import _metrics

_TS, _O, _H, _L, _C, _V = 0, 1, 2, 3, 4, 5


def backtest_cross_sectional_momentum(
    data: Dict[str, List[Sequence[float]]],
    params: Dict[str, Any] | None = None,
    fee_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> Dict[str, Any]:
    """Simulate a long-only top-K momentum rotation across symbols.

    `data` maps symbol -> OHLCV candles, all on the same timeframe and
    aligned index-for-index (candle[i] must be the same timestamp across
    every symbol). Callers are responsible for alignment -- this function
    trusts it and does not re-check timestamps per bar. Symbols with fewer
    candles than the rest are effectively truncated by `n_bars = min(len)`.

    params (all optional):
      lookback_bars (30) -- momentum ranking window (trailing return).
      hold_bars (30) -- holding period between rebalances (non-overlapping).
      top_k (3) -- number of top-momentum symbols to hold each period.
      warmup (0) -- extra bars to skip before the first ranking, on top of
        lookback_bars.

    Needs at least 2 symbols -- ranking one symbol against itself isn't
    cross-sectional. Returns the same report shape as mr_backtest/
    tf_backtest (`_metrics`), so results sit in the same comparison table
    as the vetoed strategies. `exit_reasons` in that report is not
    meaningful here: every exit is a scheduled rebalance, never an
    sl/tp/time/eod event -- that breakdown is a leftover from the shared
    helper, not a claim about this strategy's exit mix.
    """
    p = params or {}
    lookback = p.get("lookback_bars", 30)
    hold = p.get("hold_bars", 30)
    top_k = p.get("top_k", 3)
    warmup = p.get("warmup", 0)

    symbols = list(data.keys())
    if len(symbols) < 2 or top_k < 1:
        return _metrics([], 0, 0, 0)

    n_bars = min(len(c) for c in data.values())
    start = lookback + warmup
    if n_bars <= start + hold:
        return _metrics([], 0, 0, 0)

    round_trip_cost_pct = 2 * fee_pct + 2 * slippage_pct
    trades: List[Dict[str, Any]] = []
    signals = 0

    t = start
    while t + hold < n_bars:
        rankings = []
        for sym in symbols:
            candles = data[sym]
            c_now = candles[t][_C]
            c_then = candles[t - lookback][_C]
            if c_then <= 0:
                continue
            rankings.append(((c_now - c_then) / c_then, sym))
        rankings.sort(reverse=True)
        signals += 1

        exit_idx = t + hold
        for _ret, sym in rankings[:top_k]:
            candles = data[sym]
            entry = candles[t][_C] * (1 + slippage_pct / 100)
            exit_price = candles[exit_idx][_C] * (1 - slippage_pct / 100)
            gross_pct = (exit_price - entry) / entry * 100
            trades.append({
                "entry_ts": candles[t][_TS],
                "exit_ts": candles[exit_idx][_TS],
                "symbol": sym,
                "entry": entry,
                "exit": exit_price,
                "net_pct": round(gross_pct - round_trip_cost_pct, 4),
                "reason": "rebalance",
                "hold_bars": hold,
            })

        t += hold

    res = _metrics(trades, signals, 0, 0)
    res["periods"] = signals
    res["top_k"] = top_k
    return res
