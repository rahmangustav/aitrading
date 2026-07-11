#!/usr/bin/env python3
"""Jembatan antara crypto-trader (daemon) dan Live Learning Engine (`learn_live.py`).

Memakai ULANG fungsi asli learn_live (record_signal / verify_signals /
analyze_accuracy / generate_accuracy_report) tapi mengarahkan file DB & laporan ke
lokasi khusus crypto-trader supaya TIDAK menimpa data bot lama. Verifikasi tetap
pakai harga Binance MAINNET nyata (learn_live.bulk_get_prices), jadi walau order
dieksekusi di testnet, akurasi sinyal dinilai terhadap pasar sebenarnya.

Semua fungsi FAIL-OPEN: kalau ada error, kembalikan None / log — belajar tak boleh
mengganggu trading.
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger("crypto-trader.learning")

_ready = False
_ll = None
_REPORT = None


def _setup():
    """Import learn_live sekali & arahkan path-nya ke ruang crypto-trader."""
    global _ready, _ll, _REPORT
    if _ready:
        return _ll is not None
    _ready = True
    try:
        import sys
        repo = Path(__file__).resolve().parents[3]          # .../aitrading
        learn_dir = repo / ".learnings" / "trading"
        learn_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(learn_dir))
        import learn_live as ll                              # noqa: E402
        # Arahkan ke file crypto-trader (jangan sentuh data bot lama)
        ll.SKILL_DIR = learn_dir
        ll.SIGNAL_DB = learn_dir / "ct_signal_db.json"
        ll.LEARN_DIR = learn_dir
        ll.ACCURACY_FILE = learn_dir / "ACCURACY_crypto_trader.md"
        _ll = ll
        _REPORT = ll.ACCURACY_FILE
        logger.info("Learning bridge siap. DB=%s", ll.SIGNAL_DB)
        return True
    except Exception as exc:
        logger.warning("Learning bridge tak aktif (%s).", exc)
        _ll = None
        return False


def record_entry(symbol: str, price: float, confidence=None, ta: dict | None = None):
    """Catat sinyal BUY (entry) ke learning engine. Fail-open."""
    if not _setup():
        return
    try:
        pair = symbol.replace("/", "")                       # BTC/USDT -> BTCUSDT
        _ll.record_signal(pair, "BUY", float(price), confidence, None, ta or {})
        logger.info("Sinyal dicatat utk belajar: %s @ %s", pair, price)
    except Exception as exc:
        logger.warning("Gagal catat sinyal %s: %s", symbol, exc)


def refresh() -> dict | None:
    """Verifikasi sinyal lama ke harga mainnet + tulis ulang laporan akurasi.
    Return ringkasan ringkas. Fail-open."""
    if not _setup():
        return None
    try:
        verified = _ll.verify_signals()
        result = _ll.analyze_accuracy()
        if result:
            report = _ll.generate_accuracy_report(result)
            _ll.ACCURACY_FILE.write_text(report)
        return {
            "verified_now": verified,
            "total_signals": (result or {}).get("total_signals", 0),
            "verified": (result or {}).get("verified", 0),
            "pending": (result or {}).get("pending", 0),
            "report": str(_ll.ACCURACY_FILE),
        }
    except Exception as exc:
        logger.warning("Refresh learning gagal: %s", exc)
        return None
