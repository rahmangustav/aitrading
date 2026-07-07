#!/usr/bin/env python3
"""Moovon Pair Scanner — auto-scan Binance for new/listings with high volume.
Runs every Monday. Scores pairs. Adds to watchlist if score >= 50."""
import os, sys, json
from datetime import datetime
from pathlib import Path
import httpx

BASE = "https://api.binance.com"
RISET_DIR = Path("/root/.openclaw/workspace/.learnings/trading/riset")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "6146842621")

SCAN_PAIRS = [
    # L1s
    "ATOMUSDT","DOTUSDT","SEIUSDT","FTMUSDT","ICPUSDT","KASUSDT",
    # AI
    "RENDERUSDT","WLDUSDT","TAOUSDT","AKTUSDT",
    # DeFi
    "UNIUSDT","AAVEUSDT","MKRUSDT","CRVUSDT","GMXUSDT",
    # Gaming
    "IMXUSDT","BEAMUSDT","PRIMEUSDT",
    # Meme
    "PEPEUSDT","SHIBUSDT","BONKUSDT",
    # Infra
    "ARBUSDT","OPUSDT","STRKUSDT",
    # Others
    "ENAUSDT","MANTAUSDT","ALTUSDT",
]

ALREADY_RESEARCHED = {
    "BTC","ETH","SOL","XRP","DOGE","TRUMP","BNB","NEAR","SUI","ADA",
    "LUMIA","ORCA","AVAX","LINK","APTOS","FET","INJ","JUP","TIA"
}

def tg(msg):
    if not TG_TOKEN: return
    httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={
        "chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"
    }, timeout=10)

def get_next_riset_num():
    existing = list(RISET_DIR.glob("*.md"))
    nums = []
    for f in existing:
        try: nums.append(int(f.name[:3]))
        except: pass
    return max(nums) + 1 if nums else 23

def scan_pair(pair):
    try:
        with httpx.Client(timeout=10) as c:
            ticker = c.get(f'{BASE}/api/v3/ticker/24hr', params={'symbol': pair}).json()
            if 'lastPrice' not in ticker: return None
            
            price = float(ticker['lastPrice'])
            ch24 = float(ticker.get('priceChangePercent', 0))
            qvol = float(ticker.get('quoteVolume', 0))
            
            # Fetch 30d candles
            resp = c.get(f'{BASE}/api/v3/klines', params={'symbol': pair, 'interval': '1d', 'limit': 30})
            data = resp.json()
            if not data or len(data) < 20: return None
            
            prices = [float(k[4]) for k in data]
            highs = [float(k[2]) for k in data]
            lows = [float(k[3]) for k in data]
            
            d30_ch = ((prices[-1]-prices[0])/prices[0])*100 if prices[0]>0 else 0
            rw = ((max(highs[-20:])-min(lows[-20:]))/min(lows[-20:]))*100 if min(lows[-20:])>0 else 0
            
            # Score
            vol_score = min(30, int((qvol/5e6)*30))
            rq_score = min(25, int(rw/3)*8) if rw>=3 else 0
            fund_score = 5  # Unknown project
            manip_score = 8 if qvol > 10e6 else 4
            cons_score = 10 if abs(d30_ch) < 20 else (5 if abs(d30_ch) < 50 else 2)
            total = vol_score + rq_score + fund_score + manip_score + cons_score
            
            return {
                'pair': pair, 'price': price, 'ch24': ch24, 'qvol': qvol,
                'rw': rw, 'd30_ch': d30_ch,
                'vol_score': vol_score, 'rq_score': rq_score,
                'fund_score': fund_score, 'manip_score': manip_score,
                'cons_score': cons_score, 'total': total
            }
    except: return None

# ── MAIN ──
print(f"=== Pair Scanner {datetime.now()} ===")
results = []

for pair in SCAN_PAIRS:
    sym = pair.replace("USDT", "")
    if sym in ALREADY_RESEARCHED:
        continue
    data = scan_pair(pair)
    if data and data['total'] >= 50:
        results.append(data)
        print(f"  {pair}: score={data['total']} vol=${data['qvol']/1e6:.1f}M range={data['rw']:.1f}%")

# Sort by score
results.sort(key=lambda r: r['total'], reverse=True)

# Save to watchlist
if results:
    num = get_next_riset_num()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Summary telegram
    msg = f"🔍 <b>Weekly Pair Scan</b>\n{today}\n\n"
    top5 = results[:5]
    for r in top5:
        emoji = "✅" if r['total'] >= 70 else "👀"
        msg += f"{emoji} {r['pair']}: {r['total']}/100 (vol=${r['qvol']/1e6:.1f}M)\n"
    if len(results) > 5:
        msg += f"\n+{len(results)-5} more in report"
    
    # Save watchlist file
    wl_file = RISET_DIR / f"watchlist-{today}.md"
    with open(wl_file, "w") as f:
        f.write(f"# Watchlist — {today}\n\n")
        f.write("| # | Pair | Score | Vol | Range | 30d Ch |\n")
        f.write("|---|------|-------|-----|-------|--------|\n")
        for i, r in enumerate(results, 1):
            f.write(f"| {i} | {r['pair']} | {r['total']} | ${r['qvol']/1e6:.1f}M | {r['rw']:.1f}% | {r['d30_ch']:+.1f}% |\n")
    
    tg(msg)
    print(f"Saved {len(results)} pairs to {wl_file}")
else:
    print("No new pairs with score >= 50")
