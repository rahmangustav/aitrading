"""
Validate the live trend-following strategy (EMA 9/21 cross) on real data.

Same harness as validate_mr.py -- public ccxt OHLCV, no API key -- so the
trend-following numbers are directly comparable to the mean-reversion study.

Usage:
  python3 validate_tf.py --top 40 --timeframe 4h --months 24 --grid --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ccxt  # noqa: E402

from tf_backtest import backtest_trend_following  # noqa: E402
from validate_mr import (  # noqa: E402
    DEFAULT_PAIRS, aggregate_by_param, fetch_ohlcv, top_liquid_pairs,
)


def build_param_sets(grid: bool) -> list:
    base = {"label": "live      sl=5% tp=10%", "sl_pct": 5.0, "tp_pct": 10.0}
    if not grid:
        return [base]
    return [
        base,
        {"label": "close-only sl=5% tp=10%", "sl_pct": 5.0, "tp_pct": 10.0,
         "intrabar_stops": False},
        {"label": "rr1:2     sl=5% tp=10% +SMA200", "sl_pct": 5.0, "tp_pct": 10.0,
         "trend_sma_period": 200},
        {"label": "tight     sl=3% tp=9%", "sl_pct": 3.0, "tp_pct": 9.0},
        {"label": "wide      sl=8% tp=16%", "sl_pct": 8.0, "tp_pct": 16.0},
        {"label": "trail8    sl=5% trail=8%", "sl_pct": 5.0, "tp_pct": 10.0,
         "trailing_pct": 8.0},
        {"label": "adx>25    sl=5% tp=10%", "sl_pct": 5.0, "tp_pct": 10.0,
         "adx_trend_threshold": 25},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    ex = ccxt.binance({"enableRateLimit": True})
    if args.top > 0:
        args.pairs = top_liquid_pairs(ex, args.top)
        print(f"Top-{args.top} pair likuid: {', '.join(args.pairs)}")

    param_sets = build_param_sets(args.grid)
    all_results = []

    for symbol in args.pairs:
        print(f"\n=== {symbol} ({args.timeframe}, {args.months} bln) ===", flush=True)
        try:
            ohlcv = fetch_ohlcv(ex, symbol, args.timeframe, args.months)
        except Exception as exc:
            print(f"  gagal fetch: {exc}")
            continue
        print(f"  {len(ohlcv)} candle")

        for ps in param_sets:
            params = {k: v for k, v in ps.items() if k != "label"}
            res = backtest_trend_following(ohlcv, params)
            res_slim = {k: v for k, v in res.items() if k != "trades"}
            all_results.append({"symbol": symbol, "params": ps["label"], **res_slim})
            gate = "LOLOS-GATE" if res["win_rate_pct"] >= 60 and res["total_trades"] >= 10 else "-"
            print(
                f"  {ps['label']:32s} trades={res['total_trades']:3d} "
                f"WR={res['win_rate_pct']:5.1f}% PF={res['profit_factor']:5.2f} "
                f"ret={res['total_return_pct']:+8.2f}% dd={res['max_drawdown_pct']:5.2f}% {gate}"
            )

    print("\n=== AGREGAT per parameter (semua pair) ===")
    for agg in aggregate_by_param(all_results):
        print(f"  {agg['label']:32s} trades={agg['trades']:4d} WR={agg['win_rate_pct']:5.1f}% "
              f"PF={agg['profit_factor']:4.2f} sum_ret={agg['total_return_pct']:+9.2f}%  "
              f"{'>=60% GATE OK' if agg['win_rate_pct'] >= 60 else 'belum lolos gate'}")

    if args.json:
        out = Path(__file__).resolve().parent.parent / "data" / "backtests"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"validate_tf_{int(time.time())}.json"
        path.write_text(json.dumps(all_results, indent=2))
        print(f"\nJSON: {path}")


if __name__ == "__main__":
    main()
