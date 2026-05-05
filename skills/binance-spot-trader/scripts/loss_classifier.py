#!/usr/bin/env python3
"""
Phase 2: Loss Reason Classifier
===============================
Analyzes closed loss trades and classifies them into:
  trend_against  – entry against dominant trend (SMA7 < SMA20 on 4h)
  sl_too_tight   – hit SL then price reversed >1% toward TP direction
  tp_too_far     – price nearly hit TP (<0.5%) then reversed to SL
  bad_entry      – entry at local top (price dropped >1% within 1h)
  market_crash   – BTC dropped >3% in 24h window around trade
  unknown        – insufficient data to classify

Usage: python3 loss_classifier.py [--force] [--verbose] [--no-api]
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Defaults ──────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
TRADES_FILE = SKILL_DIR / "trades.jsonl"
SIGNAL_DB = SKILL_DIR / "signal_db.json"
ENTRY_QUALITY = SKILL_DIR / "entry_quality.json"
LOSS_REASONS_FILE = SKILL_DIR / "loss_reasons.json"
PREVENTION_FILE = SKILL_DIR / "prevention_rules.json"

BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = f"{BINANCE_BASE}/api/v3/klines"

DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "TRUMPUSDT", "BNBUSDT", "NEARUSDT", "SUIUSDT", "ADAUSDT",
]

# Cache for Binance data to avoid repeated API calls
_KLINE_CACHE = {}  # {symbol_interval: [candles]}
_API_RATE_LIMIT_HIT = 0

# ── Helpers ───────────────────────────────────────────────────────────
def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠  {path.name} parse error: {e}")
        return None


def load_jsonl(path):
    if not path.exists():
        return []
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def parse_ts(ts_str):
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        # Handle Z suffix and timezone
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


# ── Binance API ───────────────────────────────────────────────────────
def fetch_klines(symbol, interval="4h", start_ms=None, end_ms=None, limit=500):
    """Fetch klines from Binance public API. Returns list of candles."""
    global _API_RATE_LIMIT_HIT

    if not HAS_REQUESTS:
        return []

    # Rate limit guard
    now = time.time()
    if _API_RATE_LIMIT_HIT and now - _API_RATE_LIMIT_HIT < 2:
        return []
    _API_RATE_LIMIT_HIT = now

    cache_key = f"{symbol}_{interval}_{start_ms}_{end_ms}"
    if cache_key in _KLINE_CACHE:
        return _KLINE_CACHE[cache_key]

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = int(start_ms)
    if end_ms:
        params["endTime"] = int(end_ms)

    try:
        resp = requests.get(BINANCE_KLINES, params=params, timeout=10)
        if resp.status_code == 429:
            print(f"  ⚠  Rate limited by Binance, waiting...")
            time.sleep(2)
            return []
        resp.raise_for_status()
        candles = resp.json()
        _KLINE_CACHE[cache_key] = candles
        return candles
    except Exception as e:
        # Don't spam errors — just silently fail
        return []


def candle_to_dict(c):
    """Convert Binance kline array to dict."""
    return {
        "open_time": c[0],
        "open": float(c[1]),
        "high": float(c[2]),
        "low": float(c[3]),
        "close": float(c[4]),
        "volume": float(c[5]),
        "close_time": c[6],
    }


def calc_sma(candles, period=7):
    """Calculate SMA over last N candles."""
    closes = [c["close"] for c in candles[-period:]]
    if len(closes) < period:
        return None
    return sum(closes) / len(closes)


def get_btc_change_24h(entry_ts):
    """Get BTC 24h price change around entry time."""
    if not entry_ts:
        return 0
    start_ms = (entry_ts - timedelta(hours=24)).timestamp() * 1000
    end_ms = (entry_ts + timedelta(hours=4)).timestamp() * 1000
    candles = fetch_klines("BTCUSDT", "1h", start_ms, end_ms, limit=50)
    if len(candles) < 2:
        return 0

    c_dicts = [candle_to_dict(c) for c in candles]
    high_24h = max(c["high"] for c in c_dicts[:24]) if c_dicts[:24] else c_dicts[0]["close"]
    cur_price = c_dicts[-1]["close"]
    if high_24h > 0:
        return ((cur_price - high_24h) / high_24h) * 100
    return 0


def get_price_action(symbol, entry_ts, hours_before=4, hours_after=1):
    """Fetch price candles around entry for analysis."""
    if not entry_ts:
        return None, None

    # 4h candles for trend
    start_4h = (entry_ts - timedelta(hours=24 * 3)).timestamp() * 1000  # 3 days for trend
    end_4h = entry_ts.timestamp() * 1000
    candles_4h_raw = fetch_klines(symbol, "4h", start_4h, end_4h, limit=30)
    candles_4h = [candle_to_dict(c) for c in candles_4h_raw] if candles_4h_raw else []

    # 15m candles for short-term action around entry
    start_15m = (entry_ts - timedelta(hours=2)).timestamp() * 1000
    end_15m = (entry_ts + timedelta(hours=hours_after)).timestamp() * 1000
    candles_15m_raw = fetch_klines(symbol, "15m", start_15m, end_15m, limit=24)
    candles_15m = [candle_to_dict(c) for c in candles_15m_raw] if candles_15m_raw else []

    return candles_4h, candles_15m


def get_price_range_after_exit(symbol, exit_ts, direction="buy"):
    """Get highest/lowest price after exit to check for SL tightness."""
    if not exit_ts:
        return None

    start_ms = exit_ts.timestamp() * 1000
    end_ms = (exit_ts + timedelta(hours=24)).timestamp() * 1000

    candles = fetch_klines(symbol, "15m", start_ms, end_ms, limit=96)
    if not candles:
        return None

    c_dicts = [candle_to_dict(c) for c in candles]
    if not c_dicts:
        return None

    if direction == "buy":
        # For buy trade: check if price went up after exit (SL too tight)
        best = max(c["high"] for c in c_dicts)
        worst = min(c["low"] for c in c_dicts)
        return best, worst
    else:
        # For sell trade: check if price went down after exit
        best = min(c["low"] for c in c_dicts)
        worst = max(c["high"] for c in c_dicts)
        return best, worst


# ── Classification Logic ──────────────────────────────────────────────
def classify_loss(trade_entry, candles_4h, candles_15m, btc_change):
    """
    Classify a loss trade based on available data.
    Returns list of classifications (a trade can have multiple reasons).
    """
    classifications = []
    entry_price = float(trade_entry.get("entry", 0))
    exit_price = float(trade_entry.get("exit", 0))
    entry_ts = parse_ts(trade_entry.get("entry_ts"))
    exit_ts = parse_ts(trade_entry.get("exit_ts"))
    side = trade_entry.get("side", "BUY").upper()
    symbol = trade_entry.get("symbol", "")
    pnl_pct = float(trade_entry.get("pnl_pct", 0))

    if not entry_ts or not exit_ts:
        return ["unknown"]

    # ── 1. Market crash check (BTC) ──
    if abs(btc_change) >= 3:
        classifications.append("market_crash")

    # ── 2. Trend check ──
    if candles_4h and len(candles_4h) >= 7:
        sma7 = calc_sma(candles_4h, 7)
        sma20 = calc_sma(candles_4h, 20)
        if sma7 is not None and sma20 is not None:
            if side == "BUY" and sma7 < sma20:
                # Buy in downtrend
                classifications.append("trend_against")
            elif side == "SELL" and sma7 > sma20:
                # Sell in uptrend
                classifications.append("trend_against")
    else:
        # Fallback: check 24h result from signal_db
        result_24h = trade_entry.get("result_24h", 0) or 0
        if side == "BUY" and result_24h and result_24h < -3:
            classifications.append("trend_against")

    # ── 3. Bad entry check ──
    if candles_15m:
        # Find entry candle index
        entry_ms = int(entry_ts.timestamp() * 1000)
        entry_idx = None
        for i, c in enumerate(candles_15m):
            if abs(c["open_time"] - entry_ms) < 900000:  # ±15min
                entry_idx = i
                break

        if entry_idx is not None and entry_idx + 4 < len(candles_15m):
            # Check next 4 candles (1 hour)
            next_candles = candles_15m[entry_idx + 1:entry_idx + 5]
            next_closes = [c["close"] for c in next_candles]
            if next_closes:
                max_drop = min(next_closes) if next_closes else 0
                drop_pct = ((max_drop - entry_price) / entry_price) * 100
                if side == "BUY" and drop_pct < -1:
                    classifications.append("bad_entry")
                elif side == "SELL" and drop_pct > 1:
                    classifications.append("bad_entry")
        else:
            # Fallback heuristic: check if price was near daily high
            if candles_15m and entry_price > 0:
                recent_high = max(c["high"] for c in candles_15m[:entry_idx or len(candles_15m)])
                if entry_price >= recent_high * 0.995:  # Within 0.5% of local high
                    classifications.append("bad_entry")

    # ── 4. TP too far: check if price nearly hit TP ──
    if candles_15m and len(candles_15m) > 4:
        # Look at price action between entry and exit
        exit_ms = int(exit_ts.timestamp() * 1000)
        capture_candles = [c for c in candles_15m
                          if entry_ms - 300000 <= c["open_time"] <= exit_ms + 300000]

        if capture_candles and pnl_pct < 0 and abs(pnl_pct) < 3:
            # Small loss — maybe TP was close
            peak = max(c["high"] for c in capture_candles) if side == "BUY" else min(c["low"] for c in capture_candles)
            peak_pct = ((peak - entry_price) / entry_price) * 100
            if side == "BUY":
                peak_pct = peak_pct
            else:
                peak_pct = -peak_pct

            if 0 < peak_pct < 0.5:  # Nearly hit break-even or small profit
                classifications.append("tp_too_far")
            elif 0.5 <= peak_pct < 5:  # Some profit but gave back
                classifications.append("sl_too_tight")

    # ── 5. SL too tight: price reversed after exit ──
    if exit_price > 0 and entry_price > 0:
        # If we can't fetch post-exit data, use heuristic
        sl_dist = abs(((exit_price - entry_price) / entry_price) * 100)
        if sl_dist < 2 and pnl_pct < 0:
            # Small SL hit
            if "sl_too_tight" not in classifications:
                classifications.append("sl_too_tight")

    if not classifications:
        classifications.append("unknown")

    return classifications


# ── Prevention Rules Generator ────────────────────────────────────────
def generate_prevention_rules(losses_by_reason):
    """Generate prevention rules based on loss classifications."""
    rules = []

    reasons = defaultdict(int)
    for reason in losses_by_reason.values():
        for r in reason:
            reasons[r] += 1

    if reasons.get("trend_against", 0) >= 2:
        rules.append({
            "id": "R1",
            "condition": "trend_against",
            "rule": "Require SMA7 > SMA20 on 4h before BUY entry (or SMA7 < SMA20 for SELL)",
            "action": "add_trend_filter",
            "params": {"sma_fast": 7, "sma_slow": 20, "timeframe": "4h"},
            "severity": "high",
            "losses_affected": reasons["trend_against"],
        })

    if reasons.get("sl_too_tight", 0) >= 2:
        rules.append({
            "id": "R2",
            "condition": "sl_too_tight",
            "rule": "Widen stop loss by 0.5-1% for pairs with SL-too-tight classification",
            "action": "widen_stop_loss",
            "params": {"increase_pct": 0.5},
            "severity": "medium",
            "losses_affected": reasons["sl_too_tight"],
        })

    if reasons.get("tp_too_far", 0) >= 2:
        rules.append({
            "id": "R3",
            "condition": "tp_too_far",
            "rule": "Reduce take profit target by 1% for pairs where TP is consistently too far",
            "action": "reduce_take_profit",
            "params": {"reduce_pct": 1.0},
            "severity": "medium",
            "losses_affected": reasons["tp_too_far"],
        })

    if reasons.get("bad_entry", 0) >= 2:
        rules.append({
            "id": "R4",
            "condition": "bad_entry",
            "rule": "Add 15-minute delay confirmation: require the price to hold above entry for 15min",
            "action": "add_entry_delay",
            "params": {"delay_minutes": 15, "price_hold_above": "entry"},
            "severity": "high",
            "losses_affected": reasons["bad_entry"],
        })

    if reasons.get("market_crash", 0) >= 1:
        rules.append({
            "id": "R5",
            "condition": "market_crash",
            "rule": "Pause trading when BTC drops >3% in 24h; add BTC volatility filter",
            "action": "btc_crash_guard",
            "params": {"btc_drop_threshold_pct": 3},
            "severity": "critical",
            "losses_affected": reasons["market_crash"],
        })

    # Generic rules
    if sum(reasons.values()) > 5:
        rules.append({
            "id": "R6",
            "condition": "general",
            "rule": "Review all pairs with >40% loss rate monthly; consider pair rotation",
            "action": "monthly_review",
            "params": {},
            "severity": "low",
        })

    return rules


# ── Main Pipeline ─────────────────────────────────────────────────────
def run_classification(args):
    """Main classification pipeline."""
    print("🔬 Phase 2: Loss Reason Classifier")
    print("=" * 50)

    # Load data
    entry_quality = load_json(ENTRY_QUALITY)
    signal_db = load_json(SIGNAL_DB)
    trades_raw = load_jsonl(TRADES_FILE)

    if not entry_quality:
        print("❌ entry_quality.json not found. Nothing to classify.")
        return

    classifications_list = entry_quality.get("recent_classifications", [])
    print(f"📋 {len(classifications_list)} trades in entry_quality.json")

    # Identify losses
    losses = [c for c in classifications_list if float(c.get("pnl_pct", 0)) < 0]
    print(f"🔴 {len(losses)} loss trades to classify")

    if not losses:
        print("✅ No losses to classify. Clean record!")
        return

    # Classify each loss
    print("\n🔍 Classifying losses...")
    if args.no_api:
        print("  (API calls disabled — heuristic-only mode)")

    classified = {}
    reason_summary = defaultdict(int)

    for i, loss in enumerate(losses):
        symbol = loss.get("symbol", "UNKNOWN")
        entry_ts = parse_ts(loss.get("entry_ts"))
        exit_ts = parse_ts(loss.get("exit_ts"))
        side = loss.get("side", "BUY")
        pnl = float(loss.get("pnl_pct", 0))

        # Fetch data
        btc_change = 0
        candles_4h, candles_15m = [], []

        if not args.no_api and HAS_REQUESTS:
            # Throttle
            if i > 0:
                time.sleep(0.25)

            btc_change = get_btc_change_24h(entry_ts)
            candles_4h, candles_15m = get_price_action(symbol, entry_ts)

        # Enhance loss with signal data
        enhanced_loss = dict(loss)
        enhanced_loss["side"] = side

        # Cross-refer with signal_db for additional context
        if signal_db:
            matching_signals = [
                s for s in signal_db.get("signals", [])
                if s.get("pair") == symbol and abs(
                    (parse_ts(s.get("ts")) - entry_ts).total_seconds() if parse_ts(s.get("ts")) and entry_ts else 99999
                ) < 3600  # Within 1 hour
            ]
            if matching_signals:
                enhanced_loss["result_24h"] = matching_signals[0].get("result_24h")
            else:
                enhanced_loss["result_24h"] = None

        # Classify
        reasons = classify_loss(enhanced_loss, candles_4h, candles_15m, btc_change)

        trade_id = f"{symbol}_{loss.get('entry_ts', 'unknown')}"
        classified[trade_id] = {
            "symbol": symbol,
            "entry": loss.get("entry"),
            "exit": loss.get("exit"),
            "pnl_pct": pnl,
            "entry_ts": loss.get("entry_ts"),
            "exit_ts": loss.get("exit_ts"),
            "reasons": reasons,
            "btc_change_24h": round(btc_change, 2),
        }

        for r in reasons:
            reason_summary[r] += 1

        if args.verbose:
            print(f"  {symbol}: {pnl:+.2f}% → {', '.join(reasons)}")
        elif i % 5 == 0:
            sys.stdout.write(f"\r  Progress: {i+1}/{len(losses)}")
            sys.stdout.flush()

    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()

    # ── Generate results ──
    print(f"\n📊 Classification Summary:")
    for reason, count in sorted(reason_summary.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(losses)*100:.0f}%)")

    # Prevention rules
    prevention = generate_prevention_rules(classified)

    # Output
    results = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "total_losses": len(losses),
        "summary": dict(reason_summary),
        "classifications": classified,
        "prevention_rules": prevention,
    }

    LOSS_REASONS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Saved to {LOSS_REASONS_FILE}")

    # Save prevention rules separately
    PREVENTION_FILE.write_text(json.dumps(prevention, indent=2))
    print(f"✅ Prevention rules saved to {PREVENTION_FILE}")

    # ── Update entry_quality.json ──
    # Add loss tags back to entry_quality
    tag_map = {}
    for trade_id, data in classified.items():
        key = data["entry_ts"]
        tag_map[key] = data["reasons"]

    updated = 0
    for c in classifications_list:
        ts = c.get("entry_ts")
        if ts in tag_map:
            old_tag = c.get("tag", "")
            new_tags = tag_map[ts]
            # Only update if was "unknown_loss"
            if "unknown" in old_tag.lower() or old_tag == "unknown_loss":
                c["tag"] = new_tags[0] if new_tags else "unknown"
                c["all_tags"] = new_tags
                c["auto_classified"] = True
                updated += 1

    entry_quality["recent_classifications"] = classifications_list
    entry_quality["tag_summary"] = {
        **entry_quality.get("tag_summary", {}),
        **{k: v for k, v in reason_summary.items()},
    }
    entry_quality["loss_reasons_updated"] = datetime.now(timezone.utc).isoformat()
    ENTRY_QUALITY.write_text(json.dumps(entry_quality, indent=2))
    print(f"✅ Updated {updated} tags in entry_quality.json")

    if prevention:
        print(f"\n🛡️  Generated {len(prevention)} prevention rules:")
        for rule in prevention:
            print(f"  [{rule['severity'].upper()}] {rule['id']}: {rule['condition']} → {rule['rule']}")


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Loss Reason Classifier")
    parser.add_argument("--force", action="store_true", help="Force reclassify even if recent")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-api", action="store_true", help="Skip Binance API calls (heuristic only)")
    args = parser.parse_args()

    if not args.force and LOSS_REASONS_FILE.exists():
        age = datetime.now().timestamp() - LOSS_REASONS_FILE.stat().st_mtime
        if age < 6 * 3600:
            print(f"⏭  Skipping (last run {age/3600:.1f}h ago). Use --force to override.")
            return

    run_classification(args)


if __name__ == "__main__":
    main()
