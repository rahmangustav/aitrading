"""
Validate cross-sectional momentum (csm_backtest.py) on real Binance data.

Same harness as validate_mr.py / validate_tf.py -- public ccxt OHLCV, no API
key -- so results land in the same comparison table as the two strategies
VERDICT.md already vetoed (18 Juli 2026, both <40% winrate, PF<1). This
script has NOT been run in the sandbox that wrote it: that environment has
no route to api.binance.com (`curl api.binance.com` times out there), so
running this and reading the printed numbers must happen on a machine with
real exchange access -- per CLAUDE.md, no backtest numbers get claimed
without actually running them.

Usage:
  python3 validate_csm.py --top 40 --timeframe 4h --months 24 --grid --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ccxt  # noqa: E402

from csm_backtest import backtest_cross_sectional_momentum  # noqa: E402
from validate_mr import fetch_ohlcv, top_liquid_pairs  # noqa: E402


def build_param_sets(grid: bool) -> list:
    base = {"label": "lb30 hold30 k3", "lookback_bars": 30, "hold_bars": 30, "top_k": 3}
    if not grid:
        return [base]
    return [
        base,
        {"label": "lb14 hold14 k3", "lookback_bars": 14, "hold_bars": 14, "top_k": 3},
        {"label": "lb60 hold30 k3", "lookback_bars": 60, "hold_bars": 30, "top_k": 3},
        {"label": "lb30 hold30 k1", "lookback_bars": 30, "hold_bars": 30, "top_k": 1},
        {"label": "lb30 hold30 k5", "lookback_bars": 30, "hold_bars": 30, "top_k": 5},
        {"label": "lb30 hold7  k3", "lookback_bars": 30, "hold_bars": 7, "top_k": 3},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    ex = ccxt.binance({"enableRateLimit": True})
    pairs = args.pairs or top_liquid_pairs(ex, args.top)
    print(f"Pair: {', '.join(pairs)}")

    data = {}
    for symbol in pairs:
        try:
            ohlcv = fetch_ohlcv(ex, symbol, args.timeframe, args.months)
        except Exception as exc:
            print(f"  {symbol}: gagal fetch: {exc}")
            continue
        if len(ohlcv) < 100:
            print(f"  {symbol}: cuma {len(ohlcv)} candle, dibuang (kurang dari 100)")
            continue
        data[symbol] = ohlcv
        print(f"  {symbol}: {len(ohlcv)} candle")

    if len(data) < 2:
        print("\nKurang dari 2 pair punya data cukup -- tidak bisa jalan cross-sectional.")
        return

    n_bars = min(len(c) for c in data.values())
    print(f"\nDialign ke {n_bars} candle terpendek dari {len(data)} pair.")

    all_results = []
    for ps in build_param_sets(args.grid):
        params = {k: v for k, v in ps.items() if k != "label"}
        res = backtest_cross_sectional_momentum(data, params)
        res_slim = {k: v for k, v in res.items() if k != "trades"}
        all_results.append({"params": ps["label"], **res_slim})
        gate = "LOLOS-GATE" if res["win_rate_pct"] >= 60 and res["total_trades"] >= 10 else "-"
        print(
            f"  {ps['label']:16s} periods={res['periods']:4d} trades={res['total_trades']:4d} "
            f"WR={res['win_rate_pct']:5.1f}% PF={res['profit_factor']:5.2f} "
            f"ret={res['total_return_pct']:+8.2f}% dd={res['max_drawdown_pct']:5.2f}% {gate}"
        )

    if args.json:
        out = Path(__file__).resolve().parent.parent / "data" / "backtests"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"validate_csm_{int(time.time())}.json"
        path.write_text(json.dumps(all_results, indent=2))
        print(f"\nJSON: {path}")


if __name__ == "__main__":
    main()
