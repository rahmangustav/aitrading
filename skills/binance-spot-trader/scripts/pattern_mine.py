#!/usr/bin/env python3
"""Moovon Pattern Miner — analyze trades.jsonl for profitable patterns."""
import json, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

TRADES_LOG = Path(os.path.dirname(__file__)) / "trades.jsonl"

def analyze():
    if not TRADES_LOG.exists():
        print("No trades yet. Run the bot first.")
        return None

    trades = []
    with open(TRADES_LOG) as f:
        for line in f:
            try: trades.append(json.loads(line))
            except: pass

    if not trades:
        print("No trades logged.")
        return None

    # ── Aggregate ──
    wins = [t for t in trades if t.get("side") == "SELL" and t.get("price", 0) > 0]
    buys = [t for t in trades if t.get("side") == "BUY"]

    patterns = {
        "total_buys": len(buys),
        "total_sells": len(wins),
        "by_symbol": defaultdict(lambda: {"buys": 0, "sells": 0, "total_pnl": 0}),
        "by_hour": defaultdict(lambda: {"buys": 0, "sells": 0, "total_pnl": 0}),
        "by_day": defaultdict(lambda: {"buys": 0, "sells": 0, "total_pnl": 0}),
    }

    # Simplified pattern detection — expand later with more data
    for t in buys:
        sym = t.get("symbol", "?")
        patterns["by_symbol"][sym]["buys"] += 1
        try:
            ts = datetime.fromisoformat(t["ts"])
            patterns["by_hour"][f"{ts.hour:02d}h"]["buys"] += 1
            patterns["by_day"][ts.strftime("%a")]["buys"] += 1
        except: pass

    print(f"=== Pattern Mining ===")
    print(f"Total BUYs: {patterns['total_buys']} | Total SELLs: {patterns['total_sells']}")
    print()

    if patterns["by_symbol"]:
        print("By Symbol:")
        for sym, d in sorted(patterns["by_symbol"].items()):
            print(f"  {sym}: {d['buys']}B / {d['sells']}S")

    if patterns["by_day"]:
        print("\nBy Day:")
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            d = patterns["by_day"].get(day, {"buys": 0, "sells": 0})
            if d["buys"] + d["sells"] > 0:
                print(f"  {day}: {d['buys']}B / {d['sells']}S")

    if patterns["by_hour"]:
        print("\nBy Hour (UTC):")
        for h, d in sorted(patterns["by_hour"].items()):
            if d["buys"] + d["sells"] > 0:
                print(f"  {h}: {d['buys']}B / {d['sells']}S")

    return patterns

if __name__ == "__main__":
    analyze()
