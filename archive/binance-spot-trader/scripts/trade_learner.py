#!/usr/bin/env python3
"""Moovon Trade-to-Learning — watches trades.jsonl, auto-creates learning entries."""
import os, json, time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TRADES_LOG = SCRIPT_DIR / "trades.jsonl"
STATE_FILE = SCRIPT_DIR / "learning_state.json"
LEARNING_DIR = Path("/root/.openclaw/workspace/.learnings/trading")
LEARNING_FILE = LEARNING_DIR / "LEARNINGS_AUTO.md"

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_processed_ts": None, "processed_count": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def analyze_trade(trade):
    """Extract pattern from a trade."""
    entry = trade.get("entry_ts")
    symbol = trade.get("symbol", "?")
    side = trade.get("side", "")
    pnl = trade.get("pnl_pct", 0)
    reason = trade.get("reason", "SIGNAL")
    
    win = pnl > 0
    emoji = "🟢" if win else "🔴"
    tag = "profit" if win else "loss"
    
    return {
        "ts": datetime.now().isoformat(),
        "symbol": symbol,
        "side": side,
        "pnl": pnl,
        "reason": reason,
        "win": win,
        "emoji": emoji,
        "tag": tag
    }

def generate_learning_entry(analyses):
    """Generate a learning markdown entry from trade analyses."""
    if not analyses:
        return ""
    
    wins = [a for a in analyses if a["win"]]
    losses = [a for a in analyses if not a["win"]]
    
    entry = f"\n## Trade Session — {datetime.now().strftime('%d %b %Y %H:%M WIB')}\n\n"
    
    for a in analyses:
        entry += f"- {a['emoji']} **{a['side']}** {a['symbol']} | PnL: {a['pnl']:+.2f}% ({a['reason']})\n"
    
    if analyses:
        total_pnl = sum(a['pnl'] for a in analyses)
        entry += f"\n**Session**: {len(wins)}W / {len(losses)}L | Total PnL: {total_pnl:+.2f}%\n"
        entry += f"**Tags**: {', '.join(set(a['tag'] for a in analyses))}\n"
        entry += "\n---\n"
    
    return entry

# ── MAIN ──
state = load_state()
if not TRADES_LOG.exists():
    print("No trades yet")
    exit(0)

processed = []
last_ts = state.get("last_processed_ts")

with open(TRADES_LOG) as f:
    for line in f:
        try:
            trade = json.loads(line)
            trade_ts = trade.get("ts", "")
            if last_ts and trade_ts <= last_ts:
                continue
            processed.append(trade)
        except: pass

if not processed:
    print("No new trades")
    exit(0)

analyses = [analyze_trade(t) for t in processed]
entry_text = generate_learning_entry(analyses)

# Append to auto learning file
LEARNING_FILE.parent.mkdir(parents=True, exist_ok=True)
if not LEARNING_FILE.exists():
    LEARNING_FILE.write_text("# Trade Learnings (Auto)\n\nAuto-generated from trading activity.\n")

with open(LEARNING_FILE, "a") as f:
    f.write(entry_text)

# Update state
state["last_processed_ts"] = processed[-1].get("ts", "")
state["processed_count"] += len(processed)
save_state(state)

print(f"Processed {len(processed)} trades → {LEARNING_FILE}")
for a in analyses:
    print(f"  {a['emoji']} {a['side']} {a['symbol']} PnL={a['pnl']:+.2f}% ({a['reason']})")
