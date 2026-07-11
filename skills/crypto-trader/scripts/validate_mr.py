"""
Validate the regime-gated mean-reversion strategy on real Binance data.

Fetches public OHLCV via ccxt (no API key needed) and runs mr_backtest
across pairs/timeframes/param grids. Prints a summary table and flags
whether the >=60% win-rate gate (fund rule) is met.

Usage:
  python3 validate_mr.py                     # default pairs, 1h, ~6 months
  python3 validate_mr.py --pairs SOL/USDT   # specific pair
  python3 validate_mr.py --grid              # small param grid search
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ccxt  # noqa: E402

from mr_backtest import backtest_mean_reversion  # noqa: E402

DEFAULT_PAIRS = ["SOL/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"]

# Quote assets / token patterns that are not real trading candidates.
_EXCLUDE_BASES = {
    "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "EUR", "TRY", "BRL", "ARS",
    "USDP", "AEUR", "XUSD", "USD1", "PAXG",
}


def top_liquid_pairs(exchange, n: int) -> list:
    """Top-n spot USDT pairs by 24h quote volume (stables/leveraged excluded)."""
    tickers = exchange.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") or ":" in sym:
            continue
        base = sym.split("/")[0]
        if base in _EXCLUDE_BASES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        qv = t.get("quoteVolume") or 0
        if qv:
            rows.append((qv, sym))
    rows.sort(reverse=True)
    return [sym for _qv, sym in rows[:n]]


def fetch_ohlcv(exchange, symbol: str, timeframe: str, months: int) -> list:
    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - months * 30 * 24 * 3600 * 1000
    out = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        out.extend(batch)
        since = batch[-1][0] + ms_per_candle
        if len(batch) < 1000 or since > exchange.milliseconds():
            break
        time.sleep(exchange.rateLimit / 1000)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--grid", action="store_true", help="run a small parameter grid")
    ap.add_argument("--json", action="store_true", help="dump full JSON results")
    ap.add_argument("--top", type=int, default=0,
                    help="ignore --pairs and use the top-N most liquid USDT pairs")
    ap.add_argument("--bull-sma", type=int, default=0,
                    help="add a bull master switch: only enter above this SMA")
    args = ap.parse_args()

    ex = ccxt.binance({"enableRateLimit": True})

    if args.top > 0:
        args.pairs = top_liquid_pairs(ex, args.top)
        print(f"Top-{args.top} pair likuid: {', '.join(args.pairs)}")

    if args.grid:
        param_sets = [
            {"label": "base       rr=1.0 sl=1.5atr adx<20", "rr": 1.0, "sl_atr_mult": 1.5, "adx_range_threshold": 20},
            {"label": "wider-stop rr=1.0 sl=2.0atr adx<20", "rr": 1.0, "sl_atr_mult": 2.0, "adx_range_threshold": 20},
            {"label": "rr1.5      rr=1.5 sl=1.5atr adx<20", "rr": 1.5, "sl_atr_mult": 1.5, "adx_range_threshold": 20},
            {"label": "strict-adx rr=1.0 sl=1.5atr adx<15", "rr": 1.0, "sl_atr_mult": 1.5, "adx_range_threshold": 15},
            {"label": "no-regime  rr=1.0 sl=1.5atr no-gate", "rr": 1.0, "sl_atr_mult": 1.5, "adx_range_threshold": 100},
        ]
    else:
        param_sets = [{"label": "base rr=1.0 sl=1.5atr adx<20", "rr": 1.0, "sl_atr_mult": 1.5, "adx_range_threshold": 20}]

    if args.bull_sma > 0:
        for ps in param_sets:
            ps["bull_sma_period"] = args.bull_sma
            ps["label"] = f"{ps['label']} +bullSMA{args.bull_sma}"

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
            res = backtest_mean_reversion(ohlcv, params)
            res_slim = {k: v for k, v in res.items() if k != "trades"}
            all_results.append({"symbol": symbol, "params": ps["label"], **res_slim})
            gate = "LOLOS-GATE" if res["win_rate_pct"] >= 60 and res["total_trades"] >= 10 else "-"
            print(
                f"  {ps['label']:38s} trades={res['total_trades']:3d} "
                f"WR={res['win_rate_pct']:5.1f}% PF={res['profit_factor']:5.2f} "
                f"ret={res['total_return_pct']:+7.2f}% dd={res['max_drawdown_pct']:5.2f}% "
                f"blocked={res['blocked_by_regime']:3d} {gate}"
            )

    # Aggregate per param set
    print("\n=== AGREGAT per parameter (semua pair) ===")
    labels = {r["params"] for r in all_results}
    for label in sorted(labels):
        rows = [r for r in all_results if r["params"] == label]
        tw = sum(r["wins"] for r in rows)
        tl = sum(r["losses"] for r in rows)
        tot = tw + tl
        wr = tw / tot * 100 if tot else 0
        ret = sum(r["total_return_pct"] for r in rows)
        print(f"  {label:38s} trades={tot:4d} WR={wr:5.1f}% sum_ret={ret:+8.2f}%  "
              f"{'>=60% GATE OK' if wr >= 60 else 'belum lolos gate'}")

    if args.json:
        out = Path(__file__).resolve().parent.parent / "data" / "backtests"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"validate_mr_{int(time.time())}.json"
        path.write_text(json.dumps(all_results, indent=2))
        print(f"\nJSON: {path}")


if __name__ == "__main__":
    main()
