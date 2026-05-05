#!/usr/bin/env python3
"""Moovon Auto Research — runs every Monday morning. Backtest + Pattern Mine + Save."""
import os, sys, json, subprocess
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RISET_DIR = Path("/root/.openclaw/workspace/.learnings/trading/riset")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "6146842621")

def tg(msg):
    if not TG_TOKEN: return
    import httpx
    httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={
        "chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"
    }, timeout=10)

def backtest():
    """Run backtest on all 10 pairs."""
    result = subprocess.run(
        ["python3", str(SCRIPT_DIR / "backtest.py"), "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,TRUMPUSDT,BNBUSDT,NEARUSDT,SUIUSDT,ADAUSDT", "3"],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout

def pattern_mine():
    """Analyze trade patterns."""
    result = subprocess.run(
        ["python3", str(SCRIPT_DIR / "pattern_mine.py")],
        capture_output=True, text=True, timeout=30
    )
    return result.stdout

def save_report(bt_output, pm_output):
    """Save weekly report to riset folder."""
    week_num = datetime.now().strftime("%Y-W%W")
    filename = RISET_DIR / f"weekly-{week_num}.md"
    
    with open(filename, "w") as f:
        f.write(f"# Weekly Auto-Report — {datetime.now().strftime('%d %b %Y')}\n\n")
        f.write("## Backtest Results\n```\n")
        f.write(bt_output[-3000:])  # Last 3000 chars
        f.write("\n```\n\n")
        f.write("## Pattern Mining\n```\n")
        f.write(pm_output[-2000:])
        f.write("\n```\n")
    
    return filename

# ── MAIN ──
print(f"=== Auto Research {datetime.now()} ===")
bt = backtest()
print("Backtest done")
pm = pattern_mine()
print("Pattern mine done")
fname = save_report(bt, pm)
print(f"Saved to {fname}")

# Send summary to Telegram
pairs_btc = [l for l in bt.split("\n") if "+" in l and "PnL" in l]
summary = f"📊 <b>Moovon Weekly Research</b>\n{datetime.now().strftime('%d %b %Y')}\n\n"
summary += "⚡ <b>Backtest:</b>\n"
for line in pairs_btc[:12]:
    summary += f"<code>{line.strip()}</code>\n"
summary += f"\n📁 Full report: riset/{fname.name}"
tg(summary)
print("Done")
