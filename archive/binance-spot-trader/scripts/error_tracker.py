#!/usr/bin/env python3
"""Error Tracker — classify & persist all failed/UNKNOWN trades."""
import json, os, sys, argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
TRADES_FILE = SKILL_DIR / "trades.jsonl"
ERROR_FILE = SKILL_DIR / "error_log.jsonl"
SUMMARY_FILE = SKILL_DIR / "error_summary.json"

def classify_error(trade):
    qty = float(trade.get("qty", 0))
    symbol = trade.get("symbol", "???")
    price = float(trade.get("price", 0))
    result = trade.get("result", "UNKNOWN")
    
    if symbol == "BTCUSDT" and qty <= 0.0001:
        return "DUST_BTC", "BTC qty below minimum notional"
    if qty < 0.0001:
        return "DUST_QTY", "Qty " + str(qty) + " too small"
    if price == 0 and result == "UNKNOWN":
        return "LOW_NOTIONAL", "Notional below $5 minimum"
    if result == "UNKNOWN":
        return "API_UNKNOWN", "Order submitted but result unknown"
    if result in ("FAILED", "EXPIRED", "CANCELED"):
        return "ORDER_" + result, "Order status: " + result
    return "OTHER", "Result: " + str(result)

def run(verbose=False):
    if not TRADES_FILE.exists():
        print("ERROR: trades.jsonl not found")
        sys.exit(1)
    
    existing_hashes = set()
    if ERROR_FILE.exists():
        with open(ERROR_FILE) as f:
            for line in f:
                t = json.loads(line.strip())
                existing_hashes.add(t.get("trade_hash", ""))
    
    new_errors = []
    all_errors = []
    stats = Counter()
    
    with open(TRADES_FILE) as f:
        for line in f:
            t = json.loads(line.strip())
            if t.get("result") == "FILLED":
                continue
            
            category, reason = classify_error(t)
            trade_hash = t["ts"] + ":" + t["symbol"] + ":" + t["side"] + ":" + str(t["qty"])
            
            entry = {
                "trade_hash": trade_hash,
                "ts": t["ts"],
                "symbol": t["symbol"],
                "side": t["side"],
                "qty": float(t.get("qty", 0)),
                "price": float(t.get("price", 0)),
                "result": t.get("result"),
                "error_category": category,
                "error_reason": reason,
                "logged_at": datetime.now(timezone.utc).isoformat()
            }
            
            all_errors.append(entry)
            stats[category] += 1
            
            if trade_hash not in existing_hashes:
                new_errors.append(entry)
    
    if new_errors:
        with open(ERROR_FILE, "a") as f:
            for e in new_errors:
                f.write(json.dumps(e) + "\n")
        print("OK: " + str(len(new_errors)) + " new errors logged")
    else:
        print("OK: No new errors")
    
    # Summary
    first_ts = min((e["ts"] for e in all_errors), default="N/A")
    last_ts = max((e["ts"] for e in all_errors), default="N/A")
    date_stats = Counter()
    for e in all_errors:
        date_stats[e["ts"][:10]] += 1
    
    summary = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total_errors": len(all_errors),
        "new_errors": len(new_errors),
        "first_error": first_ts,
        "last_error": last_ts,
        "by_category": dict(stats),
        "by_date": dict(sorted(date_stats.items())),
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Report
    sep = "=" * 55
    print("\n" + sep)
    print("  Error Report — " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print(sep)
    print("  Total errors : " + str(len(all_errors)))
    print("  Date range   : " + first_ts[:10] + " to " + last_ts[:10])
    print()
    
    if stats:
        print("  By category:")
        max_w = max(len(k) for k in stats.keys())
        for cat, count in stats.most_common():
            bar = "#" * min(count, 50)
            print("  " + cat.ljust(max_w) + " " + str(count).rjust(5) + "  " + bar)
    
    print("\n  By date:")
    for date, count in sorted(date_stats.items()):
        status = "!!" if count > 10 else ("--" if count > 0 else "OK")
        print("  " + status + " " + date + ": " + str(count) + " errors")
    
    print("\n  Files: " + ERROR_FILE.name + " (" + str(len(all_errors)) + " entries)")
    print("         " + SUMMARY_FILE.name)
    
    if verbose and all_errors:
        print("\n" + "-" * 55)
        print("  Recent errors:")
        for e in all_errors[-10:]:
            ts = e["ts"][:19].replace("T", " ")
            line = "  " + ts + " | " + e["symbol"].ljust(10) + " " + e["side"].ljust(4)
            line += " | " + e["error_category"].ljust(20) + " | " + e["error_reason"]
            print(line)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Moovon Error Tracker")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    run(verbose=args.verbose)
