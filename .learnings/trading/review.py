#!/usr/bin/env python3
"""Moovon Fund Trade Review & Learning Engine.
Analyzes trades, tracks patterns, adjusts strategy parameters."""
import os, sys, json
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from collections import defaultdict

SKILL_DIR = Path(os.path.expanduser("~/.openclaw/workspace/skills/binance-spot-trader"))
TRADES_FILE = SKILL_DIR / "trades.jsonl"
STATE_FILE = SKILL_DIR / "state.json"
LEARNINGS_DIR = Path(os.path.expanduser("~/.openclaw/workspace/.learnings/trading"))
PATTERNS_FILE = LEARNINGS_DIR / "PATTERNS.md"
REVIEW_FILE = LEARNINGS_DIR / f"review-{date.today().isoformat()}.md"

# ── Load data ──
def load_trades():
    trades = []
    if TRADES_FILE.exists():
        for line in open(TRADES_FILE):
            try:
                t = json.loads(line.strip())
                if t.get("result") == "FILLED":
                    trades.append(t)
            except:
                pass
    return trades

def load_state():
    if STATE_FILE.exists():
        return json.load(open(STATE_FILE))
    return {}

# ── Analysis ──
def analyze_trades(trades):
    if not trades:
        return {"total": 0, "winrate": 0, "pairs": {}, "signals": []}

    # Pair buys to sells to calculate PnL
    by_pair = defaultdict(lambda: {"buys": [], "sells": [], "closed": []})
    
    for t in sorted(trades, key=lambda x: x["ts"]):
        pair = t["symbol"]
        if t["side"] == "BUY":
            by_pair[pair]["buys"].append(t)
        elif t["side"] == "SELL":
            matched = False
            # FIFO match
            remaining = t["qty"]
            while remaining > 0 and by_pair[pair]["buys"]:
                buy = by_pair[pair]["buys"][0]
                match_qty = min(buy["qty"], remaining)
                pnl_pct = ((t["price"] - buy["price"]) / buy["price"]) * 100
                pnl_usd = (t["price"] - buy["price"]) * match_qty
                by_pair[pair]["closed"].append({
                    "buy_ts": buy["ts"],
                    "sell_ts": t["ts"],
                    "buy_price": buy["price"],
                    "sell_price": t["price"],
                    "qty": match_qty,
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd,
                    "win": pnl_pct > 0
                })
                buy["qty"] -= match_qty
                remaining -= match_qty
                if buy["qty"] <= 0:
                    by_pair[pair]["buys"].pop(0)
                matched = True
            if not matched:
                by_pair[pair]["sells"].append(t)

    # Aggregate
    all_closed = []
    pair_stats = {}
    for pair, data in by_pair.items():
        closed = data["closed"]
        wins = [c for c in closed if c["win"]]
        losses = [c for c in closed if not c["win"]]
        
        total_pnl = sum(c["pnl_usd"] for c in closed)
        avg_pnl = sum(c["pnl_pct"] for c in closed) / len(closed) if closed else 0
        
        pair_stats[pair] = {
            "total": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": (len(wins) / len(closed) * 100) if closed else 0,
            "total_pnl_usd": total_pnl,
            "avg_pnl_pct": avg_pnl,
            "best": max(closed, key=lambda c: c["pnl_pct"]) if closed else None,
            "worst": min(closed, key=lambda c: c["pnl_pct"]) if closed else None,
        }
        all_closed.extend(closed)

    wins = [c for c in all_closed if c["win"]]
    total = len(all_closed)
    
    return {
        "total": total,
        "wins": len(wins),
        "losses": total - len(wins),
        "winrate": (len(wins) / total * 100) if total > 0 else 0,
        "total_pnl_usd": sum(c["pnl_usd"] for c in all_closed),
        "avg_pnl_pct": sum(c["pnl_pct"] for c in all_closed) / total if total > 0 else 0,
        "pairs": pair_stats,
        "open": {pair: sum(b["qty"] for b in data["buys"]) for pair, data in by_pair.items()}
    }

# ── State tracking ──
def check_tp_sl(state):
    """Check if any open positions are near TP/SL and flag learnings."""
    entries = state.get("entries", {})
    flags = []
    for symbol, entry in entries.items():
        entry_price = entry.get("entry_price", 0)
        if entry_price <= 0:
            continue
        # We'd need current price here — approximate from last known
        # For now, just flag the entry with its age
        ts = entry.get("entry_ts", "")
        try:
            entry_dt = datetime.fromisoformat(ts)
            hours_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
            flags.append({"symbol": symbol, "entry": entry_price, "hours_held": round(hours_held, 1)})
        except:
            pass
    return flags

# ── Learning patterns ──
def update_patterns(results, state):
    """Update PATTERNS.md with new insights."""
    today = date.today().isoformat()
    
    patterns = {
        "last_review": today,
        "total_trades": results["total"],
        "total_closed": results["total"],
        "overall_winrate": results["winrate"],
        "open_positions": results.get("open", {}),
        "per_pair": results["pairs"]
    }
    
    # Load existing patterns
    existing = {}
    if PATTERNS_FILE.exists():
        content = PATTERNS_FILE.read_text()
        # Extract existing stats from markdown
        for line in content.split("\n"):
            if ":" in line and not line.startswith("#"):
                pass  # Simplified for now
    
    # Write updated patterns
    report = f"""# Trading Patterns — Moovon Fund

Last review: {today}
Total closed trades: {results['total']}
Overall winrate: {results['winrate']:.1f}%
Total PnL: ${results['total_pnl_usd']:+.2f}

## Per Pair Performance

| Pair | Trades | Wins | Losses | Winrate | Avg PnL | Total PnL |
|------|--------|------|--------|---------|---------|-----------|
"""
    for pair, stats in sorted(results["pairs"].items()):
        report += f"| {pair} | {stats['total']} | {stats['wins']} | {stats['losses']} | {stats['winrate']:.0f}% | {stats['avg_pnl_pct']:+.2f}% | ${stats['total_pnl_usd']:+.2f} |\n"
    
    report += f"""
## Open Positions

"""
    if results.get("open"):
        for pair, qty in results["open"].items():
            report += f"- {pair}: {qty:.4f} open\n"
    else:
        report += "- No open positions\n"
    
    report += """
## Strategy Insights

"""
    # Generate insights
    if results["total"] > 0:
        if results["winrate"] >= 55:
            report += "- ✅ Winrate >55% — strategy working, stay the course\n"
        elif results["winrate"] >= 45:
            report += "- 🟡 Winrate borderline — review entry criteria\n"
        else:
            report += "- 🔴 Winrate below 45% — consider tightening LLM veto or entry conditions\n"
    
    if results["total_pnl_usd"] > 0:
        report += f"- ✅ Net positive PnL: ${results['total_pnl_usd']:+.2f}\n"
    else:
        report += f"- 🔴 Net negative PnL: ${results['total_pnl_usd']:+.2f} — review risk management\n"
    
    report += f"""
---

*Auto-generated by Moovon Fund Learning Engine — {today}*
"""
    PATTERNS_FILE.write_text(report)
    return report

# ── Daily review ──
def daily_review():
    trades = load_trades()
    state = load_state()
    results = analyze_trades(trades)
    positions = check_tp_sl(state)
    
    today = date.today().isoformat()
    review = f"""# Trade Review — {today}

## Summary
- Total trades: {results['total']}
- Wins: {results['wins']} | Losses: {results['losses']}
- Winrate: {results['winrate']:.1f}%
- Total PnL: ${results['total_pnl_usd']:+.2f}
- Avg PnL/trade: {results['avg_pnl_pct']:+.2f}%

## Open Positions
"""
    for pos in positions:
        review += f"- {pos['symbol']}: entry ${pos['entry']}, held {pos['hours_held']}h\n"
    
    review += f"""
## Per Pair
"""
    for pair, stats in sorted(results["pairs"].items()):
        review += f"- {pair}: {stats['winrate']:.0f}% WR ({stats['wins']}/{stats['total']}), PnL ${stats['total_pnl_usd']:+.2f}\n"
    
    review += f"""
## Learnings
"""
    if results["total"] == 0:
        review += "- No closed trades yet — learning phase, collecting data\n"
        review += "- Focus: let positions mature to TP/SL before evaluating\n"
    elif results["winrate"] >= 50:
        review += "- ✅ Positive edge detected — continue current strategy\n"
        best_pair = max(results["pairs"].items(), key=lambda x: x[1]["winrate"])
        review += f"- Best performer: {best_pair[0]} ({best_pair[1]['winrate']:.0f}%)\n"
    else:
        review += "- 🔴 Below break-even — needs parameter review\n"
    
    review += f"""
---
*Moovon Fund Learning Engine — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*
"""
    
    REVIEW_FILE.write_text(review)
    update_patterns(results, state)
    
    return review

if __name__ == "__main__":
    review = daily_review()
    print(review)
