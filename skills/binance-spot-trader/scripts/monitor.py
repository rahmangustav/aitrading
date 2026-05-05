#!/usr/bin/env python3
"""Moovon Server Monitor — checks CPU, RAM, disk, trader health. Alerts via Telegram."""
import os, subprocess, json
import httpx

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8735147566:AAFHpmhO2jEIiDK3atm7Se1om2vSG0Y-ViU")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "6146842621")
THRESHOLDS = {"disk": 80, "mem": 85, "cpu": 80}

def tg(msg):
    try:
        httpx.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={
            "chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"
        }, timeout=10)
    except:
        pass

def check_disk():
    df = subprocess.run(["df", "-h", "/"], capture_output=True, text=True).stdout.splitlines()
    pct = int(df[-1].split()[4].replace("%", ""))
    return pct

def check_mem():
    free = subprocess.run(["free"], capture_output=True, text=True).stdout.splitlines()
    mem = free[1].split()
    used = int(mem[2])
    total = int(mem[1])
    return round((used / total) * 100, 1)

def check_cpu():
    load = os.getloadavg()
    cores = os.cpu_count()
    return round((load[0] / cores) * 100, 1)

def check_trader_running():
    result = subprocess.run(["pgrep", "-f", "trader.py"], capture_output=True)
    return result.returncode == 0

# ── Check ──
alerts = []
disk = check_disk()
mem = check_mem()
cpu_load = check_cpu()
trader = check_trader_running()

if disk > THRESHOLDS["disk"]:
    alerts.append(f"💾 Disk: <b>{disk}%</b> (limit {THRESHOLDS['disk']}%)")
if mem > THRESHOLDS["mem"]:
    alerts.append(f"🧠 RAM: <b>{mem}%</b> (limit {THRESHOLDS['mem']}%)")
if cpu_load > THRESHOLDS["cpu"]:
    alerts.append(f"⚡ CPU: <b>{cpu_load}%</b> (limit {THRESHOLDS['cpu']}%)")
if not trader:
    alerts.append(f"🤖 <b>Trader CRASHED!</b> Process not running.")

if alerts:
    msg = "⚠️ <b>Moovon Alert!</b>\n" + "\n".join(alerts)
    tg(msg)
    print(f"ALERT: {alerts}")
else:
    print(f"OK | Disk: {disk}% | RAM: {mem}% | CPU: {cpu_load}% | Trader: {'UP' if trader else 'DOWN'}")
