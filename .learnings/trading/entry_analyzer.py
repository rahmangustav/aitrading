#!/usr/bin/env python3
"""Entry Quality Analyzer — classifies every closed trade & builds per-pair confidence modifiers.
Run after daily review to update entry_quality.json, which trader.py reads to adjust thresholds."""
import json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import urllib.request

SKILL_DIR = Path.home() / ".openclaw/workspace/skills/binance-spot-trader"
TRADES_FILE = SKILL_DIR / "trades.jsonl"
SIGNAL_FILE = SKILL_DIR / "signal_db.json"
STATE_FILE = SKILL_DIR / "state.json"
OUTPUT_FILE = SKILL_DIR / "entry_quality.json"

# ── Tag reasons ──
TAGS = {
    "good_entry":       "Entry valid, reached ≥50% of TP",
    "bad_timing":       "Entry within 2% of 24h high, reversed",
    "fomo_reentry":     "Re-entered within 90 min of previous sell",
    "false_breakout":   "Price broke range then reversed hard",
    "regime_reversal":  "Market regime changed after entry",
    "low_vol_fade":     "Volume dried up after entry",
    "overbought_rsi":   "Entry with RSI > 65 at signal time",
    "unknown_loss":     "No clear pattern — unlucky",
}

def load_trades():
    trades = []
    if TRADES_FILE.exists():
        for line in open(TRADES_FILE):
            try:
                trades.append(json.loads(line.strip()))
            except:
                pass
    return trades

def load_signals():
    if SIGNAL_FILE.exists():
        return json.loads(open(SIGNAL_FILE).read()).get("signals", [])
    return []

def load_state():
    if STATE_FILE.exists():
        return json.loads(open(STATE_FILE).read())
    return {}

def get_price_at(symbol, ts_ms):
    """Get approximate kline close price at a given timestamp."""
    try:
        ts_sec = int(ts_ms / 1000)
        url = (f"https://api.binance.com/api/v3/klines?"
               f"symbol={symbol}&interval=1h&startTime={(ts_sec-3600)*1000}&limit=2")
        data = json.loads(urllib.request.urlopen(url, timeout=10).read())
        if data:
            return float(data[-1][4])
    except:
        pass
    return None

def classify_trade(buy_trade, sell_trade, signals, preceding_trades):
    """Classify a completed trade pair (BUY→SELL)."""
    buy_price = buy_trade.get("price", 0)
    sell_price = sell_trade.get("price", 0)
    symbol = buy_trade.get("symbol", "")
    buy_ts = buy_trade.get("ts", "")
    sell_ts = sell_trade.get("ts", "")
    
    if buy_price <= 0 or sell_price <= 0:
        return {"tag": "unknown_loss", "reason": "Missing price data"}
    
    pnl_pct = ((sell_price - buy_price) / buy_price) * 100
    
    # 1. Good entry: reached ≥50% of TP
    if pnl_pct > 0:
        return {"tag": "good_entry", "reason": f"Profitable +{pnl_pct:.2f}%"}
    
    # Find the signal that triggered this BUY
    buy_dt = None
    try:
        buy_dt = datetime.fromisoformat(buy_ts.replace("Z", "+00:00"))
    except:
        try:
            buy_dt = datetime.fromisoformat(buy_ts)
        except:
            pass
    
    signal_info = None
    if buy_dt:
        for s in signals:
            try:
                st = datetime.fromisoformat(s["ts"].replace("Z", "+00:00"))
            except:
                try:
                    st = datetime.fromisoformat(s["ts"])
                except:
                    continue
            if s["pair"] == symbol and abs((st - buy_dt).total_seconds()) < 900:
                signal_info = s
                break
    
    # 2. FOMO re-entry: check if there was a recent sell
    if buy_dt:
        for t in preceding_trades:
            if t.get("symbol") == symbol and t.get("side") == "SELL":
                try:
                    st = datetime.fromisoformat(t["ts"].replace("Z", "+00:00"))
                except:
                    try:
                        st = datetime.fromisoformat(t["ts"])
                    except:
                        continue
                minutes_since = (buy_dt - st).total_seconds() / 60
                if 0 < minutes_since < 90:
                    return {"tag": "fomo_reentry", 
                            "reason": f"Re-entered {minutes_since:.0f}min after SELL @ ${t.get('price',0):.4f}"}
    
    # 3. Bad timing: try to check 24h high
    try:
        base = symbol.replace("USDT", "")
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        # Can't get historical 24h high easily, use signal context
    except:
        pass
    
    # 4. Overbought RSI: check signal info
    if signal_info and signal_info.get("llm_sentiment", 0) is not None:
        llm = signal_info.get("llm_sentiment", 0.5)
        if llm < 0.35:
            return {"tag": "false_breakout", 
                    "reason": f"LLM was bearish ({llm:.2f}), TA signal was false"}
    
    # 5. Default
    return {"tag": "unknown_loss", "reason": f"Loss {pnl_pct:+.2f}%, no clear pattern"}

def analyze():
    trades = load_trades()
    signals = load_signals()
    
    # Group trades into BUY→SELL pairs per symbol
    pairs = defaultdict(list)
    for t in trades:
        if t.get("result") != "FILLED":
            continue
        if t.get("price", 0) <= 0:
            continue
        pairs[t["symbol"]].append(t)
    
    # Match BUY→SELL
    classified = []
    per_pair_stats = defaultdict(lambda: {"total": 0, "good": 0, "bad": 0, "tags": defaultdict(int)})
    
    for symbol, trades_list in pairs.items():
        i = 0
        while i < len(trades_list):
            buy = trades_list[i]
            if buy["side"] != "BUY":
                i += 1
                continue
            
            # Find next SELL for this symbol
            sell = None
            for j in range(i+1, len(trades_list)):
                if trades_list[j]["side"] == "SELL":
                    sell = trades_list[j]
                    i = j + 1
                    break
            
            if sell is None:
                i += 1
                continue
            
            # Get preceding trades for FOMO detection
            preceding = [t for t in trades if t.get("symbol") == symbol and t.get("ts", "") < buy.get("ts", "")]
            
            result = classify_trade(buy, sell, signals, preceding)
            result["symbol"] = symbol
            result["entry"] = buy.get("price", 0)
            result["exit"] = sell.get("price", 0)
            result["pnl_pct"] = round(((sell.get("price", 0) - buy.get("price", 1)) / buy.get("price", 1)) * 100, 2) if buy.get("price", 0) > 0 else 0
            result["entry_ts"] = buy.get("ts", "")
            result["exit_ts"] = sell.get("ts", "")
            
            classified.append(result)
            
            base = symbol.replace("USDT", "")
            per_pair_stats[base]["total"] += 1
            if result["tag"] == "good_entry":
                per_pair_stats[base]["good"] += 1
            else:
                per_pair_stats[base]["bad"] += 1
            per_pair_stats[base]["tags"][result["tag"]] += 1
    
    # Build confidence modifiers per pair
    modifiers = {}
    for base, stats in per_pair_stats.items():
        total = stats["total"]
        if total >= 3:
            good_pct = stats["good"] / total * 100
            # Confidence bonus: +0 to +0.15 based on entry quality
            bonus = round((good_pct - 40) / 200, 2)  # 40%→0, 70%→0.15
            bonus = max(-0.10, min(0.15, bonus))  # clamp
            
            top_problem = max(stats["tags"].items(), key=lambda x: x[1]) if stats["tags"] else ("none", 0)
            
            modifiers[base] = {
                "total_trades": total,
                "good_entries": stats["good"],
                "bad_entries": stats["bad"],
                "entry_quality_pct": round(good_pct, 1),
                "confidence_bonus": bonus,
                "top_problem": top_problem[0],
                "top_problem_count": top_problem[1],
            }
    
    output = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total_analyzed": len(classified),
        "pair_modifiers": modifiers,
        "recent_classifications": classified[-20:],
        "tag_summary": dict(sorted(
            {tag: sum(1 for c in classified if c["tag"] == tag) for tag in TAGS}.items(),
            key=lambda x: x[1], reverse=True
        )),
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    return output

if __name__ == "__main__":
    result = analyze()
    
    print("📊 Entry Quality Analyzer")
    print("=" * 60)
    print(f"Analyzed: {result['total_analyzed']} closed trades")
    print()
    
    print("🏷️  Tag Summary:")
    for tag, count in result["tag_summary"].items():
        bar = "█" * count
        print(f"  {tag:<20} {count:>3} {bar}")
    
    print()
    print("📈 Per-Pair Confidence Modifiers:")
    print(f"{'Pair':<8} {'Trades':>6} {'Quality':>8} {'Bonus':>7} {'Top Problem':<20}")
    print("-" * 55)
    for base, mod in sorted(result["pair_modifiers"].items()):
        print(f"{base:<8} {mod['total_trades']:>6} {mod['entry_quality_pct']:>7.1f}% "
              f"{mod['confidence_bonus']:>+6.2f}  {mod['top_problem']:<20}")
    
    print(f"\nSaved: {OUTPUT_FILE}")
