#!/usr/bin/env python3
"""
Phase 2: Per-Pair Confidence Scoring Engine
============================================
Reads trades.jsonl + signal_db.json + entry_quality.json and produces
confidence_scores.json with per-pair win rate, avg PnL, profit factor,
and confidence ratings (B0-100 / S0-100, grade A/B/C/D).

Usage: python3 confidence_scorer.py [--force] [--verbose]
"""

import json
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
TRADES_FILE = SKILL_DIR / "trades.jsonl"
SIGNAL_DB = SKILL_DIR / "signal_db.json"
ENTRY_QUALITY = SKILL_DIR / "entry_quality.json"
OUTPUT_FILE = SKILL_DIR / "confidence_scores.json"
DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "TRUMPUSDT", "BNBUSDT", "NEARUSDT", "SUIUSDT", "ADAUSDT",
]


# ── Helpers ───────────────────────────────────────────────────────────
def load_json(path):
    """Load a JSON file, return {} or [] on missing/corrupt."""
    if not path.exists():
        print(f"  ⚠  {path.name} not found, skipping.")
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠  {path.name} parse error: {e}")
        return None


def load_jsonl(path):
    """Load a JSONL file, return list of dicts."""
    if not path.exists():
        print(f"  ⚠  {path.name} not found, skipping.")
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


# ── Data Extraction ───────────────────────────────────────────────────
def extract_from_classifications(entry_quality_data):
    """Extract per-pair win/loss from entry_quality.json classifications."""
    pair_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0,
                                        "pnls": [], "entries": []})

    classifications = (entry_quality_data or {}).get("recent_classifications", [])
    for c in classifications:
        symbol = c.get("symbol", "UNKNOWN")
        pnl = float(c.get("pnl_pct", 0))
        pair_stats[symbol]["pnls"].append(pnl)
        pair_stats[symbol]["total_pnl"] += pnl
        pair_stats[symbol]["entries"].append(c)
        if pnl > 0:
            pair_stats[symbol]["wins"] += 1
        else:
            pair_stats[symbol]["losses"] += 1

    return pair_stats


def extract_from_signals(signal_db_data):
    """Extract per-pair signal accuracy from signal_db.json."""
    pair_stats = defaultdict(lambda: {"buy_signals": 0, "sell_signals": 0,
                                        "buy_correct": 0, "sell_correct": 0,
                                        "total_signals": 0})

    signals = (signal_db_data or {}).get("signals", [])
    for s in signals:
        pair = s.get("pair", "UNKNOWN")
        side = s.get("signal", "BUY").upper()
        pair_stats[pair]["total_signals"] += 1

        if side == "BUY":
            pair_stats[pair]["buy_signals"] += 1
            if s.get("result_24h", 0) and s["result_24h"] > 0:
                pair_stats[pair]["buy_correct"] += 1
        elif side == "SELL":
            pair_stats[pair]["sell_signals"] += 1
            if s.get("result_24h", 0) and s["result_24h"] < 0:
                pair_stats[pair]["sell_correct"] += 1

    return pair_stats


def extract_from_trades(trades_data):
    """Extract per-pair trade counts by side from trades.jsonl."""
    pair_stats = defaultdict(lambda: {"buy_trades": 0, "sell_trades": 0,
                                        "filled": 0, "total": 0})

    for t in trades_data:
        symbol = t.get("symbol", "UNKNOWN")
        side = t.get("side", "BUY").upper()
        pair_stats[symbol]["total"] += 1
        if side == "BUY":
            pair_stats[symbol]["buy_trades"] += 1
        elif side == "SELL":
            pair_stats[symbol]["sell_trades"] += 1
        if t.get("result") == "FILLED":
            pair_stats[symbol]["filled"] += 1

    return pair_stats


# ── Scoring Logic ─────────────────────────────────────────────────────
def calculate_profit_factor(gross_profit, gross_loss):
    """Profit factor = gross_profit / |gross_loss|. ∞ if no losses."""
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def calculate_confidence_score(wr, profit_factor, avg_pnl, n_trades):
    """Calculate confidence score 0-100 based on composite metrics."""
    if n_trades == 0:
        return 0

    # Base from win rate (0-60 pts)
    wr_score = min(wr * 100, 60)

    # Profit factor bonus (0-25 pts) — PF>=1.5 gives full 25
    pf_score = min((profit_factor / 1.5) * 25, 25) if profit_factor != float("inf") else 25

    # Avg PnL bonus (0-10 pts) — avg_pnl>=2% gives full 10
    pnl_score = min((avg_pnl / 2.0) * 10, 10) if avg_pnl > 0 else 0

    # Experience bonus (0-5 pts) — >20 trades gives full 5
    exp_score = min((n_trades / 20) * 5, 5)

    return round(wr_score + pf_score + pnl_score + exp_score)


def assign_rating(confidence):
    """A/B/C/D rating based on confidence score."""
    if confidence >= 70:
        return "A"
    elif confidence >= 55:
        return "B"
    elif confidence >= 40:
        return "C"
    else:
        return "D"


# ── Main Pipeline ─────────────────────────────────────────────────────
def compute_confidence(args):
    """Main confidence computation pipeline."""
    print("📊 Phase 2: Confidence Scoring Engine")
    print("=" * 50)

    # 1. Load all data sources
    print("\n📁 Loading data...")
    trades_data = load_jsonl(TRADES_FILE)
    signal_db_data = load_json(SIGNAL_DB)
    entry_quality_data = load_json(ENTRY_QUALITY)

    if not trades_data and not signal_db_data and not entry_quality_data:
        print("❌ No data sources found. Aborting.")
        return

    print(f"   trades.jsonl: {len(trades_data)} entries")
    print(f"   signal_db.json: {len((signal_db_data or {}).get('signals', []))} signals")
    print(f"   entry_quality.json: {len((entry_quality_data or {}).get('recent_classifications', []))} classified")

    # 2. Extract per-pair stats from all sources
    print("\n🔍 Extracting per-pair statistics...")
    cls_stats = extract_from_classifications(entry_quality_data)
    sig_stats = extract_from_signals(signal_db_data)
    trd_stats = extract_from_trades(trades_data)

    # 3. Get list of all known pairs
    all_pairs = set(DEFAULT_PAIRS)
    all_pairs.update(cls_stats.keys())
    all_pairs.update(sig_stats.keys())
    all_pairs.update(trd_stats.keys())

    # 4. Compute confidence per pair
    print("\n📈 Computing confidence scores...")
    results = {"updated": datetime.now(timezone.utc).isoformat(), "pairs": {}}

    for pair in sorted(all_pairs):
        cls = cls_stats.get(pair, {})
        sig = sig_stats.get(pair, {})
        trd = trd_stats.get(pair, {})

        n_trades = cls.get("wins", 0) + cls.get("losses", 0)
        n_signals = sig.get("total_signals", 0)
        n_filled = trd.get("filled", 0)

        # Win rate from classified trades (primary)
        if n_trades > 0:
            wr = cls["wins"] / n_trades
            avg_pnl = cls["total_pnl"] / n_trades
        else:
            wr = 0.0
            avg_pnl = 0.0

        # Profit factor from classified trades
        wins_pnl = sum(p for p in cls.get("pnls", []) if p > 0)
        loss_pnl = sum(p for p in cls.get("pnls", []) if p < 0)
        profit_factor = calculate_profit_factor(wins_pnl, loss_pnl)

        # Buy confidence from signal accuracy + trade WR
        buy_correct = sig.get("buy_correct", 0)
        buy_signals = sig.get("buy_signals", 0)
        buy_signal_wr = (buy_correct / buy_signals) if buy_signals > 0 else 0

        # Sell confidence
        sell_correct = sig.get("sell_correct", 0)
        sell_signals = sig.get("sell_signals", 0)
        sell_signal_wr = (sell_correct / sell_signals) if sell_signals > 0 else 0

        # Combined WR: blend trade WR and signal WR if both available
        if n_trades > 0 and buy_signals > 0:
            combined_wr_buy = (wr * 0.6 + buy_signal_wr * 0.4)
        elif n_trades > 0:
            combined_wr_buy = wr
        elif buy_signals > 0:
            combined_wr_buy = buy_signal_wr
        else:
            combined_wr_buy = 0.5  # default neutral

        combined_wr_sell = sell_signal_wr if sell_signals > 0 else 0.5

        # Confidence scores
        buy_confidence = calculate_confidence_score(
            combined_wr_buy, profit_factor, avg_pnl, n_trades + buy_signals
        )
        sell_confidence = calculate_confidence_score(
            combined_wr_sell, profit_factor, avg_pnl, n_signals
        )

        # Overall confidence (max of buy/sell)
        overall_confidence = max(buy_confidence, sell_confidence)
        rating = assign_rating(overall_confidence)

        results["pairs"][pair] = {
            "wr": round(wr, 3),
            "signal_wr": round(buy_signal_wr if buy_signals > 0 else sell_signal_wr, 3),
            "avg_pnl_pct": round(avg_pnl, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999,
            "confidence_buy": buy_confidence,
            "confidence_sell": sell_confidence,
            "overall_confidence": overall_confidence,
            "rating": rating,
            "n_trades": n_trades,
            "n_signals": n_signals,
            "n_filled": n_filled,
            "wins": cls.get("wins", 0),
            "losses": cls.get("losses", 0),
        }

        if args.verbose:
            print(f"  {pair}: WR={wr:.1%} PnL={avg_pnl:+.2f}% PF={profit_factor:.2f} "
                  f"B{buy_confidence}/S{sell_confidence} [{rating}]")

    # 5. Summary
    print(f"\n📋 Rating Distribution:")
    rating_counts = defaultdict(int)
    for p, data in results["pairs"].items():
        rating_counts[data["rating"]] += 1
    for grade in ["A", "B", "C", "D"]:
        cnt = rating_counts[grade]
        bar = "█" * cnt
        print(f"  {grade}: {bar} ({cnt})")

    # 6. Write output
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Saved to {OUTPUT_FILE}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Per-Pair Confidence Scorer")
    parser.add_argument("--force", action="store_true", help="Force recompute even if recent")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Check if recent run (< 6 hours)
    if not args.force and OUTPUT_FILE.exists():
        age = datetime.now().timestamp() - OUTPUT_FILE.stat().st_mtime
        if age < 6 * 3600:
            print(f"⏭  Skipping (last run {age/3600:.1f}h ago). Use --force to override.")
            return

    compute_confidence(args)


if __name__ == "__main__":
    main()
