#!/usr/bin/env python3
"""Moovon Fund Live Learning Engine — learns from every signal, open or closed.
Tracks: sentiment accuracy, signal quality, time patterns, pair reliability."""
import os, sys, json, time, math
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import httpx

load_dotenv(dotenv_path=Path.home() / ".openclaw/workspace/moovon/.env")
SKILL_DIR = Path.home() / ".openclaw/workspace/skills/binance-spot-trader"
TRADES_FILE = SKILL_DIR / "trades.jsonl"
STATE_FILE = SKILL_DIR / "state.json"
LEARN_DIR = Path.home() / ".openclaw/workspace/.learnings/trading"
SIGNAL_DB = SKILL_DIR / "signal_db.json"
ACCURACY_FILE = LEARN_DIR / "ACCURACY.md"

API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
PAIRS = os.getenv("PAIRS", "BTCUSDT").split(",")
LLM_VETO = float(os.getenv("LLM_VETO_THRESHOLD", "0.30"))

# ── Binance API ──
def api_get(path, params=None):
    with httpx.Client(timeout=10) as c:
        r = c.get(f"https://api.binance.com{path}", params=params)
        return r.json()

# ── Signal DB ──
def load_signal_db():
    if SIGNAL_DB.exists():
        return json.loads(SIGNAL_DB.read_text())
    return {"signals": [], "accuracy": {}}

def save_signal_db(db):
    SIGNAL_DB.write_text(json.dumps(db, indent=2, default=str))

# ── Record signals ──
def record_signal(pair, signal_type, price, confidence, llm_sentiment, ta_data):
    """Record a signal with metadata for later accuracy tracking."""
    db = load_signal_db()
    db["signals"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "pair": pair,
        "signal": signal_type,
        "price": price,
        "confidence": confidence,  # confluence score B70/S70 etc
        "llm_sentiment": llm_sentiment,
        "ta": ta_data,  # regex, rsi, bollinger position
        "status": "open" if signal_type == "BUY" else "pending",
        "verified": False,
        "result_4h": None,
        "result_8h": None,
        "result_24h": None,
    })
    save_signal_db(db)

# ── Verify past signals ──
def bulk_get_prices():
    """Fetch ALL Binance prices in ONE API call."""
    try:
        with httpx.Client(timeout=15) as c:
            r = c.get("https://api.binance.com/api/v3/ticker/price")
            return {p["symbol"]: float(p["price"]) for p in r.json()}
    except:
        return {}

def get_price_at(symbol, target_dt):
    """Price of `symbol` AT `target_dt` (UTC), from the 1-minute kline opening then.

    Returns 0 when the price is unavailable (pair delisted, target in the future,
    network error) so the caller can leave the horizon unfilled and retry later.
    """
    start_ms = int(target_dt.timestamp() * 1000)
    try:
        klines = api_get("/api/v3/klines", {
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ms,
            "limit": 1,
        })
        if not isinstance(klines, list) or not klines:
            return 0
        return float(klines[0][1])  # open of the minute the horizon lands on
    except Exception:
        return 0

def verify_signals():
    """Verify signals against the price at each horizon — 30 signals per run.

    Each horizon is measured at its own moment (ts+4h, ts+8h, ts+24h) via historical
    klines. Using the price at verification time instead would collapse all three
    horizons onto one number whenever a run lands late (daemon down, backfilled
    signals), which silently fabricates the 4h and 8h results.
    """
    db = load_signal_db()
    now = datetime.now(timezone.utc)
    updated = 0
    checked = 0
    BATCH = 30

    # Cheap liveness + pair-existence check before spending klines calls
    all_prices = bulk_get_prices()
    if not all_prices:
        return 0

    for sig in db["signals"]:
        if checked >= BATCH:
            break
        if sig.get("verified"):
            continue

        sig_ts = datetime.fromisoformat(sig["ts"])
        hours_ago = (now - sig_ts).total_seconds() / 3600

        if hours_ago < 4:
            continue

        checked += 1
        if all_prices.get(sig["pair"], 0) <= 0:
            continue

        filled = 0
        for mark, key in [(4, "result_4h"), (8, "result_8h"), (24, "result_24h")]:
            if sig.get(key) is not None:
                filled += 1
                continue
            if hours_ago < mark:
                continue
            price_at = get_price_at(sig["pair"], sig_ts + timedelta(hours=mark))
            if price_at <= 0:
                continue
            sig[key] = round(((price_at - sig["price"]) / sig["price"]) * 100, 2)
            filled += 1
            updated += 1

        # Only close the book once every horizon actually landed, otherwise a
        # transient fetch failure would freeze a horizon at None forever.
        if hours_ago >= 24 and filled == 3:
            sig["verified"] = True

    if updated > 0:
        save_signal_db(db)
    return updated

def get_current_price(symbol):
    try:
        ticker = api_get("/api/v3/ticker/price", {"symbol": symbol})
        return float(ticker["price"])
    except:
        return 0

# ── Accuracy Analysis ──
def analyze_accuracy():
    """Calculate signal accuracy by pair, by hour, by regime."""
    db = load_signal_db()
    signals = db["signals"]
    
    if not signals:
        return None
    
    # By pair
    pair_stats = defaultdict(lambda: {"total": 0, "correct_4h": 0, "correct_8h": 0, "avg_move": 0})
    
    # By hour
    hour_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    # By LLM sentiment range
    sentiment_buckets = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for sig in signals:
        pair = sig["pair"]
        s_type = sig["signal"]
        
        # Parse hour
        try:
            sig_hour = datetime.fromisoformat(sig["ts"]).hour
        except:
            sig_hour = -1
        
        # Bucket sentiment
        sent = sig.get("llm_sentiment") or 0
        if sent < 0.3: bucket = "0.00-0.29"
        elif sent < 0.5: bucket = "0.30-0.49"
        elif sent < 0.7: bucket = "0.50-0.69"
        else: bucket = "0.70-1.00"
        
        # Check 4h result
        r4 = sig.get("result_4h")
        if r4 is not None:
            pair_stats[pair]["total"] += 1
            pair_stats[pair]["avg_move"] += abs(r4)
            
            # For BUY: correct if price went up
            correct = (s_type == "BUY" and r4 > 0) or (s_type == "SELL" and r4 < 0)
            if correct:
                pair_stats[pair]["correct_4h"] += 1
            
            if sig_hour >= 0:
                hour_stats[sig_hour]["total"] += 1
                if correct:
                    hour_stats[sig_hour]["correct"] += 1
            
            sentiment_buckets[bucket]["total"] += 1
            if correct:
                sentiment_buckets[bucket]["correct"] += 1
    
    # Compute rates
    for p, s in pair_stats.items():
        if s["total"] > 0:
            s["accuracy"] = round(s["correct_4h"] / s["total"] * 100, 1)
            s["avg_move"] = round(s["avg_move"] / s["total"], 2)
    
    for h, s in hour_stats.items():
        if s["total"] >= 3:
            s["accuracy"] = round(s["correct"] / s["total"] * 100, 1)
    
    for b, s in sentiment_buckets.items():
        if s["total"] > 0:
            s["accuracy"] = round(s["correct"] / s["total"] * 100, 1)
    
    return {
        "pair_stats": dict(pair_stats),
        "hour_stats": dict(hour_stats),
        "sentiment_buckets": dict(sentiment_buckets),
        "total_signals": len(signals),
        "verified": len([s for s in signals if s.get("verified")]),
        "pending": len([s for s in signals if not s.get("verified")]),
    }

def generate_accuracy_report(result):
    if not result or result["total_signals"] == 0:
        return "No signals to analyze yet."
    
    rpt = f"""# Signal Accuracy Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Overview
- Total signals: {result['total_signals']}
- Verified: {result['verified']} | Pending: {result['pending']}

## Per Pair Accuracy (4h)
| Pair | Signals | Accuracy | Avg Move |
|------|---------|----------|----------|
"""
    pstats = result["pair_stats"]
    for pair in sorted(pstats.keys()):
        s = pstats[pair]
        if s["total"] >= 3:
            emoji = "🟢" if s.get("accuracy", 0) >= 55 else "🟡" if s.get("accuracy", 0) >= 40 else "🔴"
            rpt += f"| {pair} | {s['total']} | {emoji} {s.get('accuracy',0)}% | {s.get('avg_move',0):.1f}% |\n"
    
    rpt += f"""
## Best Trading Hours
| Hour (UTC) | Signals | Accuracy |
|------------|---------|----------|
"""
    hstats = result["hour_stats"]
    for hour in sorted(hstats.keys()):
        s = hstats[hour]
        if s["total"] >= 3:
            rpt += f"| {hour:02d}:00 | {s['total']} | {s.get('accuracy',0)}% |\n"
    
    rpt += f"""
## LLM Sentiment Reliability
| Range | Signals | Signal Accuracy |
|-------|---------|-----------------|
"""
    for bucket in ["0.00-0.29", "0.30-0.49", "0.50-0.69", "0.70-1.00"]:
        s = result["sentiment_buckets"].get(bucket, {"total": 0, "accuracy": 0})
        if s["total"] > 0:
            rpt += f"| {bucket} | {s['total']} | {s.get('accuracy',0)}% |\n"
    
    # Generate recommendations
    rpt += "\n## 🧠 Recommendations\n"
    
    # Best pair
    best = max(pstats.items(), key=lambda x: x[1].get("accuracy", 0), default=(None, {}))
    worst = min(pstats.items(), key=lambda x: x[1].get("accuracy", 0), default=(None, {}))
    
    if best[0] and best[1].get("accuracy", 0) >= 55:
        rpt += f"- ✅ **{best[0]}** most accurate ({best[1].get('accuracy',0)}%) — consider larger allocation\n"
    if worst[0] and worst[1].get("accuracy", 0) < 40 and worst[1]["total"] >= 3:
        rpt += f"- 🔴 **{worst[0]}** least accurate ({worst[1].get('accuracy',0)}%) — consider removing from pairs\n"
    
    # Best sentiment range
    best_bucket = max(result["sentiment_buckets"].items(), key=lambda x: x[1].get("accuracy", 0), default=(None, {}))
    if best_bucket[0] and best_bucket[1].get("accuracy", 0) >= 55:
        rpt += f"- ✅ LLM sentiment {best_bucket[0]} is most reliable ({best_bucket[1].get('accuracy',0)}% accurate)\n"
    
    # Veto threshold check
    zero_range = result["sentiment_buckets"].get("0.00-0.29", {})
    mid_range = result["sentiment_buckets"].get("0.30-0.49", {})
    if zero_range.get("accuracy", 0) < 30 and zero_range.get("total", 0) >= 3:
        rpt += "- ✅ LLM veto at 0.30 is working — signals below threshold are unreliable\n"
    if mid_range.get("accuracy", 0) >= 50 and mid_range.get("total", 0) >= 3:
        rpt += "- 🟢 Signals in 0.30-0.49 range are profitable — current threshold is good\n"
    
    rpt += f"\n---\n*Auto-generated by Live Learning Engine — {datetime.now(timezone.utc).isoformat()}*"
    return rpt

def capture_live_metrics():
    """Capture current state of open positions for drawdown tracking."""
    db = load_signal_db()
    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    
    entries = state.get("entries", {})
    metrics = []
    for symbol, entry in entries.items():
        try:
            price = get_current_price(f"{symbol}USDT")
            if price > 0 and entry.get("entry_price", 0) > 0:
                pnl = ((price - entry["entry_price"]) / entry["entry_price"]) * 100
                hours = 0
                if entry.get("entry_ts"):
                    et = datetime.fromisoformat(entry["entry_ts"])
                    hours = (datetime.now(timezone.utc) - et).total_seconds() / 3600
                metrics.append({
                    "symbol": symbol,
                    "entry": entry["entry_price"],
                    "current": price,
                    "pnl_pct": round(pnl, 2),
                    "hours_held": round(hours, 1),
                    "max_dd": min(pnl, db.get("max_dd", {}).get(symbol, 0)),
                })
        except:
            pass
    
    # Update max drawdown tracking
    if "max_dd" not in db:
        db["max_dd"] = {}
    for m in metrics:
        db["max_dd"][m["symbol"]] = min(m["pnl_pct"], db["max_dd"].get(m["symbol"], 0))
    save_signal_db(db)
    
    return metrics

# ── Main ──
def run():
    verified = verify_signals()
    result = analyze_accuracy()
    metrics = capture_live_metrics()
    
    report = generate_accuracy_report(result)
    ACCURACY_FILE.write_text(report)
    
    print(report)
    print()
    print(f"--- Verified {verified} signals ---")
    print()
    if metrics:
        print("## 📊 Live Positions")
        for m in metrics:
            emoji = "🟢" if m["pnl_pct"] > 1 else "🟡" if m["pnl_pct"] > -1 else "🔴"
            print(f"{emoji} {m['symbol']}: {m['pnl_pct']:+.2f}% | ${m['current']} | entry ${m['entry']} | {m['hours_held']}h | max DD: {m['max_dd']}%")
    
    return report

if __name__ == "__main__":
    run()
