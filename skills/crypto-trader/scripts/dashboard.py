#!/usr/bin/env python3
"""Dashboard pemantau bot crypto-trader (Moovon Fund).

Server web LOKAL read-only: menampilkan kondisi bot secara live —
modal, posisi, order & stop-loss, status risiko, strategi aktif, daemon.
TIDAK pernah menaruh/membatalkan order. Aman dijalankan kapan saja.

Jalankan:  python dashboard.py            (buka http://localhost:8787)
           python dashboard.py --port 9000
"""
from __future__ import annotations
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
_env = D.parent.parent.parent / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv; load_dotenv(_env)
    except ImportError:
        pass

from exchange_manager import ExchangeManager
from risk_manager import RiskManager
from strategy_engine import StrategyEngine
from monitor_daemon import MonitorDaemon
from main import _register_strategies

EX = "binance"
STABLE = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP"}
MAJORS = {"BTC", "ETH", "BNB"}

_em = None
def em():
    global _em
    if _em is None:
        _em = ExchangeManager()
    return _em


def snapshot() -> dict:
    """Kumpulkan kondisi bot saat ini. Tiap bagian tahan-error sendiri."""
    out = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "demo": True, "exchange": EX, "errors": []}
    try:
        e = em()
        out["demo"] = bool(e.demo)
        out["connected"] = EX in e.available_exchanges
    except Exception as ex:
        out["connected"] = False
        out["errors"].append(f"init: {ex}")
        return out

    # --- Daemon ---
    try:
        out["daemon"] = MonitorDaemon().get_status()
    except Exception as ex:
        out["errors"].append(f"daemon: {ex}")

    # --- Risk ---
    try:
        rm = RiskManager()
        out["risk"] = rm.get_status()
    except Exception as ex:
        out["errors"].append(f"risk: {ex}")

    # --- Strategi aktif ---
    try:
        rm = RiskManager()
        eng = StrategyEngine(em(), rm)
        _register_strategies(eng)
        eng.sync_from_disk()
        out["strategies"] = eng.list_strategies()
    except Exception as ex:
        out["strategies"] = []
        out["errors"].append(f"strategies: {ex}")

    # --- Order terbuka (semua simbol, 1 panggilan) ---
    try:
        e = em()
        try:  # Binance minta acknowledge warning fetch-tanpa-simbol
            e._get_exchange(EX).options["warnOnFetchOpenOrdersWithoutSymbol"] = False
        except Exception:
            pass
        orders = e.get_open_orders(EX)
        norm = []
        for o in orders:
            is_stop = e._is_stop_order(o)
            norm.append({
                "symbol": o.get("symbol"), "side": o.get("side"),
                "type": o.get("type"), "amount": o.get("amount"),
                "price": o.get("price"),
                "stop": o.get("stopPrice") or o.get("triggerPrice"),
                "is_stop": is_stop,
            })
        out["orders"] = norm
        out["n_stops"] = sum(1 for x in norm if x["is_stop"])
    except Exception as ex:
        out["orders"] = []
        out["errors"].append(f"orders: {ex}")

    # --- Saldo & posisi ---
    try:
        e = em()
        bal = e.get_balance(EX)
        usdt = bal.get("USDT", {}) or {}
        out["usdt"] = {"free": usdt.get("free", 0) or 0,
                       "used": usdt.get("used", 0) or 0,
                       "total": usdt.get("total", 0) or 0}
        # Koin non-stable dgn saldo > 0; nilai hanya majors + yg ada di order
        in_orders = {o["symbol"].split("/")[0] for o in out.get("orders", []) if o.get("symbol")}
        holdings = []
        for coin, d in bal.items():
            if not isinstance(d, dict):
                continue
            tot = d.get("total", 0) or 0
            if coin in STABLE or tot <= 0:
                continue
            row = {"coin": coin, "amount": tot,
                   "free": d.get("free", 0) or 0, "used": d.get("used", 0) or 0,
                   "value_usdt": None}
            if coin in MAJORS or coin in in_orders:
                try:
                    tk = e.get_ticker(EX, f"{coin}/USDT")
                    px = tk.get("last") or 0
                    row["value_usdt"] = round(tot * px, 2)
                except Exception:
                    pass
            holdings.append(row)
        holdings.sort(key=lambda r: (r["value_usdt"] or 0, r["amount"]), reverse=True)
        out["holdings"] = holdings[:25]
        out["n_holdings"] = len(holdings)
    except Exception as ex:
        out["holdings"] = []
        out["errors"].append(f"balance: {ex}")

    # --- PnL per koin (posisi yang dilacak strategi apa pun: DCA / trend / dll) ---
    try:
        e = em()
        rm = RiskManager()
        eng = StrategyEngine(em(), rm)
        _register_strategies(eng)
        eng.sync_from_disk()
        # Gabung per simbol; baca posisi langsung dari instance strategi.
        # DCA: total_bought/avg_price/total_invested. Trend & lainnya:
        # position_amount/entry_price.
        agg = {}
        for inst in (eng.get_strategy_instance(x["strategy_id"])
                     for x in eng.list_strategies()):
            if inst is None:
                continue
            sym = getattr(inst, "symbol", None) or (inst.params or {}).get("symbol")
            bought = getattr(inst, "total_bought", 0) or getattr(inst, "position_amount", 0) or 0
            entry = getattr(inst, "avg_price", 0) or getattr(inst, "entry_price", 0) or 0
            invested = getattr(inst, "total_invested", 0) or (bought * entry)
            if not sym or bought <= 0 or entry <= 0:
                continue
            a = agg.setdefault(sym, {"bought": 0.0, "invested": 0.0})
            a["bought"] += bought
            a["invested"] += invested
        positions = []
        total_pnl = 0.0
        for sym, a in agg.items():
            entry = a["invested"] / a["bought"] if a["bought"] else 0
            try:
                px = e.get_ticker(EX, sym).get("last") or 0
            except Exception:
                px = 0
            value_now = a["bought"] * px
            pnl = value_now - a["invested"]
            pnl_pct = ((px / entry) - 1) * 100 if entry else 0
            total_pnl += pnl
            positions.append({
                "symbol": sym, "amount": round(a["bought"], 8),
                "entry": round(entry, 4), "price": round(px, 4),
                "cost": round(a["invested"], 2), "value": round(value_now, 2),
                "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            })
        positions.sort(key=lambda r: r["pnl"], reverse=True)
        out["positions"] = positions
        out["total_pnl"] = round(total_pnl, 2)
        out["total_cost"] = round(sum(p["cost"] for p in positions), 2)
    except Exception as ex:
        out["positions"] = []
        out["total_pnl"] = 0.0
        out["errors"].append(f"pnl: {ex}")

    return out


PAGE = """<!doctype html><html lang=id><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Moovon Fund — Monitor Bot</title>
<style>
:root{--bg:#0f1311;--panel:#171d1a;--panel2:#1e2622;--line:#2a332e;--tx:#eef2ea;--dim:#8a978f;--brand:#c6f24e;--up:#38d39f;--down:#ff5d5d;--amber:#f5c451}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 'Segoe UI',system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:20px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.brand{font-weight:800;letter-spacing:.5px;font-size:20px}
.brand b{color:var(--brand)}
.pill{font:600 12px/1 ui-monospace,monospace;padding:6px 10px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
.pill.demo{color:#0f1311;background:var(--amber);border-color:var(--amber)}
.pill.on{color:#0f1311;background:var(--up);border-color:var(--up)}
.pill.off{color:var(--down);border-color:var(--down)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.card h3{margin:0 0 6px;font:600 11px/1 ui-monospace,monospace;letter-spacing:1px;color:var(--dim);text-transform:uppercase}
.big{font:800 26px/1.1 ui-monospace,monospace}
.sub{color:var(--dim);font-size:12px;margin-top:4px}
.up{color:var(--up)}.down{color:var(--down)}.amber{color:var(--amber)}.brand{color:var(--brand)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:6px 0;margin-bottom:16px;overflow-x:auto}
.panel h2{font:600 12px/1 ui-monospace,monospace;letter-spacing:1px;color:var(--dim);text-transform:uppercase;padding:12px 16px 8px;margin:0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 16px;border-top:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font:600 11px/1 ui-monospace,monospace;letter-spacing:.5px;border:0}
td.num,th.num{text-align:right;font-family:ui-monospace,monospace}
.tag{font:600 10px/1 ui-monospace,monospace;padding:3px 7px;border-radius:6px}
.tag.buy{background:rgba(56,211,159,.15);color:var(--up)}
.tag.sell{background:rgba(255,93,93,.15);color:var(--down)}
.tag.stop{background:rgba(245,196,81,.18);color:var(--amber)}
.empty{color:var(--dim);padding:14px 16px;font-size:13px}
.bar{height:7px;border-radius:99px;background:var(--panel2);overflow:hidden;margin-top:8px}
.bar>i{display:block;height:100%;border-radius:99px}
footer{color:var(--dim);font-size:12px;text-align:center;margin:10px 0 30px}
a{color:var(--brand)}
</style></head><body><div class=wrap>
<header>
 <div class=brand>▚ MOOVON <b>FUND</b> · Monitor Bot</div>
 <div id=pills></div>
</header>
<div class=grid id=cards></div>
<div class=panel><h2>Posisi Bot & PnL (belum terealisasi)</h2><div id=pos></div></div>
<div class=panel><h2>Order Terbuka & Stop-Loss</h2><div id=orders></div></div>
<div class=panel><h2>Saldo Dompet</h2><div id=holds></div></div>
<div class=panel><h2>Strategi Aktif</h2><div id=strats></div></div>
<footer>Auto-refresh 10 dtk · <span id=upd>—</span> · read-only, tak pernah trading ·
 <span id=err></span></footer>
</div>
<script>
const f=(n,d=2)=>n==null?'—':Number(n).toLocaleString('id-ID',{maximumFractionDigits:d});
function pct(v){return (v>0?'+':'')+f(v,2)+'%'}
async function tick(){
 let s; try{s=await (await fetch('/data')).json()}catch(e){document.getElementById('err').textContent='koneksi dashboard gagal';return}
 // pills
 const P=[];
 P.push(`<span class="pill ${s.demo?'demo':'off'}">${s.demo?'TESTNET (DEMO)':'⚠ LIVE — UANG NYATA'}</span>`);
 P.push(`<span class="pill ${s.connected?'on':'off'}">${s.connected?'TERHUBUNG':'TAK TERHUBUNG'}</span>`);
 const dae=s.daemon||{}; P.push(`<span class="pill ${dae.running?'on':'off'}">DAEMON ${dae.running?'HIDUP':'MATI'}</span>`);
 const r=s.risk||{};
 if(r.kill_switch_active)P.push('<span class="pill off">KILL-SWITCH AKTIF</span>');
 document.getElementById('pills').innerHTML=P.join(' ');
 // cards
 const u=s.usdt||{}; const lim=r.limits||{};
 const pnl=r.daily_pnl_eur||0, maxLoss=lim.max_daily_loss_eur||0, dd=r.drawdown_pct||0, maxDd=lim.max_drawdown_pct||0;
 const lossPct=maxLoss?Math.min(100,Math.abs(Math.min(0,pnl))/maxLoss*100):0;
 const ddPct=maxDd?Math.min(100,dd/maxDd*100):0;
 const C=[];
 C.push(`<div class=card><h3>Modal USDT (bebas)</h3><div class=big>${f(u.free)}</div>
   <div class=sub>terkunci di order: ${f(u.used)}</div></div>`);
 C.push(`<div class=card><h3>P&L Hari Ini</h3><div class="big ${pnl>0?'up':pnl<0?'down':''}">${f(pnl)}</div>
   <div class=sub>batas rugi harian: ${f(maxLoss)}</div>
   <div class=bar><i style="width:${lossPct}%;background:${lossPct>80?'var(--down)':'var(--amber)'}"></i></div></div>`);
 C.push(`<div class=card><h3>Drawdown</h3><div class="big ${dd>0?'down':''}">${f(dd)}%</div>
   <div class=sub>batas: ${f(maxDd)}%</div>
   <div class=bar><i style="width:${ddPct}%;background:${ddPct>80?'var(--down)':'var(--amber)'}"></i></div></div>`);
 const tp=s.total_pnl||0, tc=s.total_cost||0, tpp=tc?tp/tc*100:0;
 C.push(`<div class=card><h3>PnL Posisi (belum real.)</h3><div class="big ${tp>0?'up':tp<0?'down':''}">${tp>0?'+':''}${f(tp)}</div>
   <div class=sub>modal posisi: ${f(tc)} USDT · ${pct(tpp)}</div></div>`);
 C.push(`<div class=card><h3>Trade Hari Ini</h3><div class=big>${r.trades_today_count??'—'}</div>
   <div class=sub>order terbuka: ${(s.orders||[]).length} · stop: ${s.n_stops||0}</div></div>`);
 document.getElementById('cards').innerHTML=C.join('');
 // positions + PnL per koin
 const Pz=s.positions||[];
 document.getElementById('pos').innerHTML = Pz.length? `<table><tr>
   <th>Koin</th><th class=num>Jumlah</th><th class=num>Harga Masuk</th><th class=num>Harga Kini</th>
   <th class=num>Modal</th><th class=num>Nilai</th><th class=num>PnL</th><th class=num>PnL %</th></tr>`+
   Pz.map(p=>`<tr><td>${p.symbol}</td><td class=num>${f(p.amount,6)}</td>
     <td class=num>${f(p.entry,4)}</td><td class=num>${f(p.price,4)}</td>
     <td class=num>${f(p.cost)}</td><td class=num>${f(p.value)}</td>
     <td class="num ${p.pnl>0?'up':p.pnl<0?'down':''}">${p.pnl>0?'+':''}${f(p.pnl)}</td>
     <td class="num ${p.pnl_pct>0?'up':p.pnl_pct<0?'down':''}">${pct(p.pnl_pct)}</td></tr>`).join('')
   +`</table>` : '<div class=empty>Belum ada posisi yang dilacak strategi.</div>';
 // orders
 const O=s.orders||[];
 document.getElementById('orders').innerHTML = O.length? `<table><tr>
   <th>Simbol</th><th>Sisi</th><th>Tipe</th><th class=num>Jumlah</th><th class=num>Harga</th><th class=num>Stop</th></tr>`+
   O.map(o=>`<tr><td>${o.symbol||'—'}</td>
     <td><span class="tag ${o.is_stop?'stop':o.side}">${o.is_stop?'STOP':(o.side||'').toUpperCase()}</span></td>
     <td>${o.type||'—'}</td><td class=num>${f(o.amount,6)}</td>
     <td class=num>${o.price?f(o.price):'mkt'}</td><td class=num>${o.stop?f(o.stop):'—'}</td></tr>`).join('')
   +`</table>` : '<div class=empty>Tidak ada order terbuka.</div>';
 // holdings
 const H=s.holdings||[];
 document.getElementById('holds').innerHTML = H.length? `<table><tr>
   <th>Koin</th><th class=num>Jumlah</th><th class=num>Bebas</th><th class=num>Terkunci</th><th class=num>≈ USDT</th></tr>`+
   H.map(h=>`<tr><td>${h.coin}</td><td class=num>${f(h.amount,6)}</td>
     <td class=num>${f(h.free,6)}</td><td class=num>${f(h.used,6)}</td>
     <td class=num>${h.value_usdt==null?'—':f(h.value_usdt)}</td></tr>`).join('')
   +`</table>`+ (s.n_holdings>H.length?`<div class=empty>…dan ${s.n_holdings-H.length} aset lain (kebanyakan token uji).</div>`:'')
   : '<div class=empty>Tidak ada posisi non-stablecoin.</div>';
 // strategies
 const St=s.strategies||[];
 document.getElementById('strats').innerHTML = St.length? `<table><tr>
   <th>ID</th><th>Nama</th><th>Simbol</th><th>Status</th></tr>`+
   St.map(x=>`<tr><td>${x.id||x.strategy_id||'—'}</td><td>${x.name||'—'}</td>
     <td>${(x.params&&x.params.symbol)||x.symbol||'—'}</td>
     <td><span class="tag ${x.active?'buy':'sell'}">${x.active?'AKTIF':'BERHENTI'}</span></td></tr>`).join('')
   +`</table>` : '<div class=empty>Tidak ada strategi aktif. (Bot idle — normal saat belum dijalankan.)</div>';
 document.getElementById('upd').textContent='diperbarui '+s.time;
 document.getElementById('err').textContent=(s.errors&&s.errors.length)?('catatan: '+s.errors.join(' | ')):'';
}
tick(); setInterval(tick,10000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path.startswith("/data"):
            body = json.dumps(snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(f"Dashboard Moovon Fund jalan di  http://localhost:{args.port}")
    print("Tekan Ctrl+C untuk berhenti.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard berhenti.")


if __name__ == "__main__":
    main()
