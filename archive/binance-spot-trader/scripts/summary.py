#!/usr/bin/env python3
"""Moovon Fund — Daily Summary Reporter. Runs at 21:00 WITA."""
import os, sys, json
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

API_KEY = os.environ["BINANCE_API_KEY"]
SECRET_KEY = os.environ["BINANCE_SECRET_KEY"]
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "6146842621")
TRADES_LOG = Path(os.path.dirname(__file__)) / "trades.jsonl"
STATE_FILE = Path(os.path.dirname(__file__)) / "state.json"
BASE = "https://api.binance.com"

def sign(params: dict) -> dict:
    import hmac, hashlib, time
    from urllib.parse import urlencode
    params["timestamp"] = int(time.time() * 1000)
    q = urlencode(params)
    params["signature"] = hmac.new(SECRET_KEY.encode(), q.encode(), hashlib.sha256).hexdigest()
    return params

def api_get(path, params=None):
    with httpx.Client(timeout=15) as c:
        if params and "signature" in params:
            r = c.get(f"{BASE}{path}", params=params, headers={"X-MBX-APIKEY": API_KEY})
        else:
            r = c.get(f"{BASE}{path}", params=params or {})
        return r.json()

def tg(msg):
    if not TG_TOKEN: return
    httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={
        "chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"
    }, timeout=10)

def get_balance(asset):
    info = api_get("/api/v3/account", sign({}))
    for b in info.get("balances", []):
        if b["asset"] == asset:
            return float(b["free"])
    return 0

def get_total_value():
    info = api_get("/api/v3/account", sign({}))
    total = 0.0
    positions = []
    for b in info.get("balances", []):
        free = float(b["free"])
        locked = float(b["locked"])
        amt = free + locked
        if amt <= 0: continue
        if b["asset"] == "USDT":
            total += amt
        else:
            try:
                px = api_get("/api/v3/ticker/price", {"symbol": f"{b['asset']}USDT"})
                val = amt * float(px["price"])
                total += val
                positions.append((b["asset"], amt, float(px["price"]), val))
            except:
                pass
    return total, positions

def get_today_trades():
    today = str(date.today())
    trades = []
    if TRADES_LOG.exists():
        with open(TRADES_LOG) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if t.get("ts", "").startswith(today):
                        trades.append(t)
                except:
                    pass
    return trades

# ── MAIN ──
trades = get_today_trades()
total_value, positions = get_total_value()

# Load state for baseline
start_val = 0
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
        start_val = state.get("start_balance_snapshot", 0)

if start_val > 0:
    day_pnl = ((total_value - start_val) / start_val) * 100
else:
    day_pnl = 0

# Build message
usdt = get_balance("USDT")
msg = f"📊 <b>Moovon Daily — {date.today().strftime('%d %b %Y')}</b>\n"
msg += f"💰 Portfolio: <b>\${total_value:.2f}</b> (${usdt:.2f} USDT)\n"

if day_pnl:
    emoji = "🟢" if day_pnl >= 0 else "🔴"
    msg += f"{emoji} Day PnL: <b>{day_pnl:+.2f}%</b>\n"

# Open positions
if positions:
    msg += f"\n📂 <b>Open Positions ({len(positions)}):</b>\n"
    for asset, amt, px, val in positions:
        msg += f"  • {asset}: {amt:.6f} @ ${px:.4f} = ${val:.2f}\n"
else:
    msg += "\n📂 No open positions\n"

# Today's trades
if trades:
    buys = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]
    msg += f"\n🔄 <b>Trades Today:</b> {len(trades)} ({len(buys)}B / {len(sells)}S)\n"
    for t in trades[:10]:
        em = "🟢" if t["side"] == "BUY" else "🔴"
        px_str = f"@ ${t.get('price', 0):.4f}" if t.get("price") else ""
        msg += f"  {em} {t['side']} {t['symbol']} {t['qty']:.6f} {px_str}\n"
else:
    msg += "\n🔄 No trades today\n"

# Weekly check (Sunday only)
if date.today().weekday() == 6:
    msg += "\n📅 Weekly report coming soon — gue masih ngumpulin data.\n"

msg += "\n🐾 <i>Moovon Bot v3.0 — auto</i>"

tg(msg)
print(f"Summary sent. Portfolio: ${total_value:.2f} | PnL: {day_pnl:+.2f}% | Trades: {len(trades)}")
