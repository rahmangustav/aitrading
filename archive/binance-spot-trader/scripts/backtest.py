#!/usr/bin/env python3
"""Moovon Backtesting Engine — test strategies on Binance historical data."""
import os, sys, json, time
from datetime import datetime, timedelta
from pathlib import Path
import httpx

BASE = "https://api.binance.com"
DIR = Path(__file__).parent
RESULTS_FILE = DIR / "backtest_results.json"

# ── Data Fetch ────────────────────────────────────────────────
def fetch_klines(symbol, interval="1h", months=6):
    """Fetch historical klines. Binance limits 1000 candles per request."""
    all_data = []
    limit = 1000
    end_ms = int(time.time() * 1000)
    start_ms = int((datetime.now() - timedelta(days=months * 30)).timestamp() * 1000)

    with httpx.Client(timeout=30) as c:
        while start_ms < end_ms:
            resp = c.get(f"{BASE}/api/v3/klines", params={
                "symbol": symbol, "interval": interval, "limit": limit,
                "startTime": start_ms, "endTime": end_ms
            })
            data = resp.json()
            if not data or not isinstance(data, list): break
            all_data.extend(data)
            start_ms = data[-1][0] + 1
            if len(data) < limit: break

    return [{"t": k[0], "o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
             "c": float(k[4]), "v": float(k[5])} for k in all_data]

# ── Indicators ─────────────────────────────────────────────────
def ema(prices, period):
    k = 2 / (period + 1)
    e = prices[0]
    for p in prices[1:]: e = p * k + e * (1 - k)
    return e

def rsi(prices, period=14):
    deltas = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    if sum(losses) == 0: return 100
    return 100 - (100 / (1 + sum(gains) / sum(losses)))

def bollinger(prices, period=20, std_mult=2):
    sma = sum(prices[-period:]) / period
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std = variance ** 0.5
    return sma - std_mult * std, sma, sma + std_mult * std

def pair_trend_bearish(prices, rsi_val):
    """Check medium-term trend from 1h data (approx 4h trend).
    Returns True if bearish enough to skip BUY."""
    if len(prices) < 40:
        return False
    # Approximate 4h SMA7 vs SMA20 using 1h data:
    # 4h SMA7 ≈ 1h SMA28, 4h SMA20 ≈ 1h SMA80
    # But we only have 50 candles, so use 10h vs 40h as proxy
    sma_fast = sum(prices[-10:]) / 10
    sma_slow = sum(prices[-40:]) / 40
    if sma_fast < sma_slow:
        if rsi_val < 25:  # deep oversold → reversal opportunity
            return False
        return True
    return False

# ── Strategy Simulators ─────────────────────────────────────────
def backtest_strategy(klines, strategy, params):
    """Simulate a strategy. Returns trades list."""
    trades = []
    position = None  # None or {"entry": price, "qty": qty}
    balance = params.get("initial_balance", 100)
    position_size = params.get("position_size_pct", 15) / 100
    tp_pct = params.get("tp_pct", 5) / 100
    sl_pct = params.get("sl_pct", 3) / 100
    trade_fee = 0.001  # 0.1%

    for i in range(49, len(klines)):  # Need 50 candles for indicators
        window = klines[i-49:i+1]
        prices = [k["c"] for k in window]
        current = prices[-1]

        # Check TP/SL if in position
        if position:
            pnl = (current - position["entry"]) / position["entry"]
            if pnl >= tp_pct or pnl <= -sl_pct:
                exit_value = position["qty"] * current * (1 - trade_fee)
                trades.append({
                    "entry_ts": position["ts"], "exit_ts": klines[i]["t"],
                    "symbol": params.get("symbol", "?"),
                    "entry": position["entry"], "exit": current,
                    "pnl_pct": pnl * 100, "pnl_usdt": exit_value - position["cost"],
                    "reason": "TP" if pnl > 0 else "SL"
                })
                balance += exit_value - position["cost"]
                position = None
                continue

        # Generate signal
        signal = "HOLD"
        if strategy == "momentum":
            ema20 = ema(prices, 20)
            avg_vol = sum(k["v"] for k in window[-20:]) / 20
            vol_spike = window[-1]["v"] > avg_vol * 1.5
            if current > ema20 and vol_spike: signal = "BUY"
            elif current < ema20: signal = "SELL"
        elif strategy == "mean_reversion":
            r = rsi(prices)
            lower, mid, upper = bollinger(prices)
            if r < 30 and current <= lower * 1.02: signal = "BUY"
            elif r > 70 or current >= upper * 0.98: signal = "SELL"

        if signal == "BUY" and not position and balance >= 10:
            # Pair trend filter: skip BUY in downtrend (align with live trader.py)
            if pair_trend_bearish(prices, rsi(prices)):
                continue
            trade_cost = balance * position_size
            qty = trade_cost / current
            position = {"entry": current, "qty": qty, "ts": klines[i]["t"], "cost": trade_cost}
            balance -= trade_cost * trade_fee
        elif signal == "SELL" and position:
            exit_value = position["qty"] * current * (1 - trade_fee)
            pnl = (current - position["entry"]) / position["entry"]
            trades.append({
                "entry_ts": position["ts"], "exit_ts": klines[i]["t"],
                "symbol": params.get("symbol", "?"),
                "entry": position["entry"], "exit": current,
                "pnl_pct": pnl * 100, "pnl_usdt": exit_value - position["cost"],
                "reason": "SIGNAL"
            })
            balance += exit_value - position["cost"]
            position = None

    metrics = {}
    if trades:
        wins = [t for t in trades if t["pnl_usdt"] > 0]
        metrics = {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
            "win_rate": round(len(wins) / len(trades) * 100, 1),
            "total_pnl_pct": round(sum(t["pnl_pct"] for t in trades), 2),
            "avg_win": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t["pnl_pct"] for t in trades if t["pnl_usdt"] <= 0) / max(len(trades)-len(wins), 1), 2),
            "max_drawdown_pct": 0,  # Simplified
            "profit_factor": round(sum(t["pnl_usdt"] for t in wins) / abs(sum(t["pnl_usdt"] for t in trades if t["pnl_usdt"] <= 0)), 2) if len(wins) < len(trades) else 999,
        }
    return trades, metrics

# ── Main ────────────────────────────────────────────────────────
def run(symbols=None, months=6):
    if symbols is None:
        symbols = os.getenv("PAIRS", "BTCUSDT,ETHUSDT").split(",")
    else:
        symbols = [s.strip() for s in symbols.split(",")]

    strategies = ["momentum", "mean_reversion"]
    results = {}

    for sym in symbols:
        print(f"Backtesting {sym}...")
        klines = fetch_klines(sym, "1h", months)
        if len(klines) < 50:
            print(f"  ⚠️  Insufficient data")
            continue

        results[sym] = {}
        for strat in strategies:
            trades, metrics = backtest_strategy(klines, strat, {"symbol": sym})
            results[sym][strat] = metrics
            wr = metrics.get("win_rate", 0)
            total = metrics.get("total_pnl_pct", 0)
            print(f"  {strat:15s}: {metrics.get('total_trades',0):3d} trades | WR={wr:5.1f}% | PnL={total:+6.2f}%")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")
    return results

if __name__ == "__main__":
    pairs = sys.argv[1] if len(sys.argv) > 1 else os.getenv("PAIRS", "BTCUSDT,ETHUSDT")
    months = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run(pairs, months)
