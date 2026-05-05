#!/usr/bin/env python3
"""
Phase 2: Auto Threshold Tuner
=============================
Reads trades.jsonl + confidence_scores.json + .env and adjusts trading
parameters based on live performance rules.

Rules:
  WR < 45% & 3 consecutive losses → widen SL by 0.5%, reduce TP 1%
  WR > 55% & PnL positive → keep current (no change)
  5+ consecutive losses on 1 pair → halve trade size temporarily
  Profit factor < 0.8 overall → switch strategy (momentum ↔ mean_reversion)

Usage: python3 auto_tuner.py [--dry-run] [--verbose]
"""

import json
import os
import sys
import re
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy

# ── Defaults ──────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
TRADES_FILE = SKILL_DIR / "trades.jsonl"
SIGNAL_DB = SKILL_DIR / "signal_db.json"
ENTRY_QUALITY = SKILL_DIR / "entry_quality.json"
CONFIDENCE_FILE = SKILL_DIR / "confidence_scores.json"
ENV_FILE = SKILL_DIR / ".env"
TUNING_LOG = SKILL_DIR / "tuning_log.jsonl"
ENV_BACKUP_DIR = SKILL_DIR

DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "TRUMPUSDT", "BNBUSDT", "NEARUSDT", "SUIUSDT", "ADAUSDT",
]

# Per-pair override suffix patterns
PAIR_SUFFIXES = ["_TP_PCT", "_SL_PCT", "_TRADE_SIZE_PCT", "_STRATEGY"]


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


def load_env(path):
    """Load .env into dict, preserving comments as _comments list."""
    if not path.exists():
        return {}, []
    env = {}
    comments = []
    with path.open() as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("#") or stripped.strip() == "":
                comments.append(stripped)
                continue
            if "=" in stripped:
                key, _, val = stripped.partition("=")
                env[key.strip()] = val.strip()
    return env, comments


def write_env(path, env_dict, original_lines):
    """Write env back maintaining original structure + comments."""
    lines = []
    env_keys_written = set()

    for orig_line in original_lines:
        stripped = orig_line.rstrip("\n")
        if stripped.startswith("#") or stripped.strip() == "":
            lines.append(stripped)
            continue
        if "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in env_dict:
                lines.append(f"{key}={env_dict[key]}")
                env_keys_written.add(key)
            else:
                lines.append(stripped)

    # Append any new keys not in original
    for key, val in env_dict.items():
        if key not in env_keys_written:
            lines.append(f"{key}={val}")

    path.write_text("\n".join(lines) + "\n")
    return True


def backup_env():
    """Create timestamped backup of .env."""
    if not ENV_FILE.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = ENV_BACKUP_DIR / f".env.bak-autotune-{ts}"
    backup_path.write_text(ENV_FILE.read_text())
    return backup_path


def log_change(entry):
    """Log a tuning change to tuning_log.jsonl."""
    entry["ts"] = datetime.now(timezone.utc).isoformat()
    with TUNING_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Analysis ──────────────────────────────────────────────────────────
def get_per_pair_wr(entry_quality_data):
    """Compute per-pair win rate from classified trades."""
    pair_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnls": []})
    classifications = (entry_quality_data or {}).get("recent_classifications", [])
    for c in classifications:
        symbol = c.get("symbol", "UNKNOWN")
        pnl = float(c.get("pnl_pct", 0))
        if pnl > 0:
            pair_stats[symbol]["wins"] += 1
        else:
            pair_stats[symbol]["losses"] += 1
        pair_stats[symbol]["pnls"].append(pnl)
    return pair_stats


def get_overall_stats(entry_quality_data):
    """Compute overall WR and profit factor."""
    classifications = (entry_quality_data or {}).get("recent_classifications", [])
    if not classifications:
        return {"wr": 0, "profit_factor": 0, "avg_pnl": 0, "n": 0}

    wins = sum(1 for c in classifications if float(c.get("pnl_pct", 0)) > 0)
    losses = len(classifications) - wins
    wr = wins / len(classifications) if classifications else 0
    avg_pnl = sum(float(c.get("pnl_pct", 0)) for c in classifications) / len(classifications)

    gross_profit = sum(float(c.get("pnl_pct", 0)) for c in classifications if float(c.get("pnl_pct", 0)) > 0)
    gross_loss = abs(sum(float(c.get("pnl_pct", 0)) for c in classifications if float(c.get("pnl_pct", 0)) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {"wr": wr, "profit_factor": profit_factor, "avg_pnl": avg_pnl,
            "n": len(classifications), "wins": wins, "losses": losses}


def count_consecutive_losses(trades_data, pair=None, max_lookback=10):
    """Count consecutive losses for a pair from entry_quality data.
    Uses a simplified approach: check last N classified trades."""
    # Since trades.jsonl only has entries (not PNL), we need entry_quality
    # Actually, let's use the entry_quality classifications
    # But for consecutive analysis, we need exit order
    # For now: load entry_quality directly
    eq = load_json(ENTRY_QUALITY)
    if not eq:
        return 0

    classifications = eq.get("recent_classifications", [])
    if pair:
        classifications = [c for c in classifications if c.get("symbol") == pair]

    # Sort by exit_ts (when trade closed)
    closed = [c for c in classifications if c.get("exit_ts")]
    closed.sort(key=lambda c: c["exit_ts"])

    consecutive = 0
    for c in reversed(closed[-max_lookback:]):
        if float(c.get("pnl_pct", 0)) < 0:
            consecutive += 1
        else:
            break

    return consecutive


# ── Tuning Rules ──────────────────────────────────────────────────────
def apply_tuning_rules(env_dict, entry_quality_data, pairs, dry_run=False, verbose=False):
    """Apply auto-tuning rules and return list of changes."""
    changes = []
    stats = get_overall_stats(entry_quality_data)
    pair_stats = get_per_pair_wr(entry_quality_data)
    confidence = load_json(CONFIDENCE_FILE) or {}

    if verbose:
        print(f"📊 Overall: WR={stats['wr']:.1%} PF={stats['profit_factor']:.2f} "
              f"AvgPnL={stats['avg_pnl']:+.2f}% ({stats['n']} trades)")

    # ── Rule 1: Overall strategy switch based on profit factor ──
    current_strategy = env_dict.get("STRATEGY", "mean_reversion")
    if stats["profit_factor"] < 0.8 and stats["n"] >= 5:
        new_strategy = "momentum" if current_strategy == "mean_reversion" else "mean_reversion"
        env_dict["STRATEGY"] = new_strategy
        change = {
            "type": "strategy_switch",
            "reason": f"Profit factor {stats['profit_factor']:.2f} < 0.8",
            "from": current_strategy,
            "to": new_strategy,
            "stats": stats,
        }
        changes.append(change)
        print(f"  🔄 Strategy: {current_strategy} → {new_strategy} (PF={stats['profit_factor']:.2f})")

    # ── Rule 2: Per-pair adjustments ──
    for pair in pairs:
        ps = pair_stats.get(pair, {"wins": 0, "losses": 0, "pnls": []})
        total = ps["wins"] + ps["losses"]
        if total < 3:  # Need at least 3 trades to judge
            continue

        wr = ps["wins"] / total
        avg_pnl = sum(ps["pnls"]) / total if ps["pnls"] else 0
        cons_losses = count_consecutive_losses(None, pair)

        # Get current per-pair params
        tp_key = f"{pair}_TP_PCT"
        sl_key = f"{pair}_SL_PCT"
        size_key = f"{pair}_TRADE_SIZE_PCT"

        tp_current = float(env_dict.get(tp_key) or env_dict.get("TAKE_PROFIT_PCT", "5"))
        sl_current = float(env_dict.get(sl_key) or env_dict.get("STOP_LOSS_PCT", "3"))
        size_current = float(env_dict.get(size_key) or env_dict.get("TRADE_SIZE_PCT", "15"))

        # Rule 2a: Low WR with consecutive losses → widen SL, reduce TP
        if wr < 0.45 and cons_losses >= 3:
            new_sl = min(sl_current + 0.5, 8.0)  # Cap SL at 8%
            new_tp = max(tp_current - 1.0, 2.0)   # Floor TP at 2%
            env_dict[sl_key] = str(new_sl)
            env_dict[tp_key] = str(new_tp)
            change = {
                "type": "per_pair_adjust",
                "pair": pair,
                "reason": f"WR={wr:.1%} with {cons_losses} consecutive losses",
                "sl": {"from": sl_current, "to": new_sl},
                "tp": {"from": tp_current, "to": new_tp},
            }
            changes.append(change)
            print(f"  ⚠  {pair}: WR={wr:.1%}, {cons_losses} cons. losses → SL {sl_current}→{new_sl}%, TP {tp_current}→{new_tp}%")

        # Rule 2b: 5+ consecutive losses → halve trade size
        if cons_losses >= 5:
            new_size = max(size_current / 2, 5.0)  # Floor at 5%
            env_dict[size_key] = str(new_size)
            change = {
                "type": "size_reduce",
                "pair": pair,
                "reason": f"{cons_losses} consecutive losses",
                "size": {"from": size_current, "to": new_size},
            }
            changes.append(change)
            print(f"  🛑 {pair}: {cons_losses} consecutive losses → Size {size_current}→{new_size}%")

        # Rule 2c: Good WR + positive PnL → no change (log success)
        if wr > 0.55 and avg_pnl > 0:
            if verbose:
                print(f"  ✅ {pair}: WR={wr:.1%} PnL={avg_pnl:+.2f}% → keeping params")

    # ── Rule 3: Global SL/TP adjustment if overall WR low ──
    if stats["wr"] < 0.40 and stats["n"] >= 6:
        global_sl = float(env_dict.get("STOP_LOSS_PCT", "3"))
        global_tp = float(env_dict.get("TAKE_PROFIT_PCT", "5"))
        new_sl = min(global_sl + 1.0, 6.0)
        new_tp = max(global_tp - 1.0, 3.0)
        if new_sl != global_sl or new_tp != global_tp:
            env_dict["STOP_LOSS_PCT"] = str(new_sl)
            env_dict["TAKE_PROFIT_PCT"] = str(new_tp)
            change = {
                "type": "global_adjust",
                "reason": f"Overall WR {stats['wr']:.1%} too low",
                "sl": {"from": global_sl, "to": new_sl},
                "tp": {"from": global_tp, "to": new_tp},
            }
            changes.append(change)
            print(f"  🌐 Global: SL {global_sl}→{new_sl}%, TP {global_tp}→{new_tp}%")

    # ── Rule 4: Recover size for recovering pairs ──
    for pair in pairs:
        ps = pair_stats.get(pair, {"wins": 0, "losses": 0, "pnls": []})
        total = ps["wins"] + ps["losses"]
        if total < 5:
            continue
        wr = ps["wins"] / total

        size_key = f"{pair}_TRADE_SIZE_PCT"
        default_size = float(env_dict.get("TRADE_SIZE_PCT", "15"))
        current_size = float(env_dict.get(size_key, str(default_size)))

        if wr > 0.50 and current_size < default_size:
            # Gradually restore
            new_size = min(current_size * 1.5, default_size)
            if new_size > current_size:
                env_dict[size_key] = str(new_size)
                change = {
                    "type": "size_restore",
                    "pair": pair,
                    "reason": f"Recovered WR={wr:.1%}",
                    "size": {"from": current_size, "to": new_size},
                }
                changes.append(change)
                print(f"  📈 {pair}: Recovering → Size {current_size}→{new_size}%")

    return changes


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Auto Threshold Tuner")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("🎯 Phase 2: Auto Threshold Tuner")
    print("=" * 50)

    # Load data
    entry_quality_data = load_json(ENTRY_QUALITY)
    trades_data = load_jsonl(TRADES_FILE)
    env_dict, env_lines = load_env(ENV_FILE)

    if not env_dict:
        print("❌ .env not found. Aborting.")
        return

    # Get pairs list
    pairs_str = env_dict.get("PAIRS", "")
    pairs = [p.strip() for p in pairs_str.split(",") if p.strip()] if pairs_str else DEFAULT_PAIRS
    print(f"📋 Pairs: {len(pairs)}")

    # Apply rules
    print("\n🔍 Analyzing performance...")
    changes = apply_tuning_rules(env_dict, entry_quality_data, pairs,
                                  dry_run=args.dry_run, verbose=args.verbose)

    if not changes:
        print("\n✅ No adjustments needed. All parameters within targets.")
        return

    print(f"\n📝 {len(changes)} change(s) proposed:")

    # Apply or dry-run
    if args.dry_run:
        print("\n🔬 DRY RUN — no changes applied. Would write:")
        for i, c in enumerate(changes, 1):
            print(f"  {i}. [{c['type']}] {c.get('pair', 'global')}: {c['reason']}")
    else:
        # Backup
        backup_path = backup_env()
        print(f"\n💾 Backed up .env → {backup_path.name}")

        # Write new env
        write_env(ENV_FILE, env_dict, env_lines)
        print("📝 Updated .env")

        # Log all changes
        for c in changes:
            log_change(c)
        print(f"📋 Logged to {TUNING_LOG.name} ({len(changes)} entries)")

        print("\n✅ Auto-tuning complete!")


if __name__ == "__main__":
    main()
