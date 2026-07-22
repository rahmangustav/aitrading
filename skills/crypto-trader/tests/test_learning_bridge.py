"""Tests for learning_bridge — jembatan crypto-trader ke Live Learning Engine."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import learning_bridge


@pytest.fixture(autouse=True)
def reset_bridge_state():
    """learning_bridge cache setup-nya lewat variabel modul global (_ready/_ll) —
    reset tiap test supaya tidak bocor antar test, dan cabut learn_live palsu
    dari sys.modules supaya tidak menempel ke test lain / modul lain."""
    learning_bridge._ready = False
    learning_bridge._ll = None
    learning_bridge._REPORT = None
    sys.modules.pop("learn_live", None)
    yield
    learning_bridge._ready = False
    learning_bridge._ll = None
    learning_bridge._REPORT = None
    sys.modules.pop("learn_live", None)


def _install_fake_learn_live():
    fake = types.ModuleType("learn_live")
    fake.record_signal = MagicMock()
    fake.verify_signals = MagicMock(return_value=3)
    fake.analyze_accuracy = MagicMock(return_value={
        "total_signals": 10, "verified": 7, "pending": 3,
    })
    fake.generate_accuracy_report = MagicMock(return_value="# laporan")
    fake.SKILL_DIR = None
    fake.SIGNAL_DB = None
    fake.LEARN_DIR = None
    fake.ACCURACY_FILE = MagicMock()
    sys.modules["learn_live"] = fake
    return fake


class TestSetup:
    def test_setup_success_redirects_paths_ke_ruang_crypto_trader(self):
        fake = _install_fake_learn_live()
        assert learning_bridge._setup() is True
        assert fake.SIGNAL_DB.name == "ct_signal_db.json"
        assert fake.ACCURACY_FILE.name == "ACCURACY_crypto_trader.md"
        assert fake.LEARN_DIR == fake.SKILL_DIR
        assert learning_bridge._REPORT is fake.ACCURACY_FILE

    def test_setup_caches_setelah_sukses_pertama(self):
        fake = _install_fake_learn_live()
        assert learning_bridge._setup() is True
        # Cabut modul dari sys.modules — kalau _setup() diam-diam mengimpor
        # ulang, panggilan kedua akan gagal (dependensi asli learn_live.py
        # seperti dotenv/httpx belum tentu terpasang di sandbox test ini).
        sys.modules["learn_live"] = None
        assert learning_bridge._setup() is True
        assert learning_bridge._ll is fake

    def test_setup_gagal_import_fail_open(self):
        sys.modules["learn_live"] = None  # trik standar Python: paksa ImportError
        assert learning_bridge._setup() is False
        assert learning_bridge._ll is None

    def test_setup_gagal_juga_di_cache_tak_dicoba_ulang(self):
        sys.modules["learn_live"] = None
        assert learning_bridge._setup() is False
        # Modul jadi tersedia belakangan — tapi _setup() sengaja tak mencoba
        # lagi (fail-open berarti berhenti mencoba, bukan retry tiap panggilan).
        fake = _install_fake_learn_live()
        assert learning_bridge._setup() is False
        fake.record_signal.assert_not_called()


class TestRecordEntry:
    def test_konversi_simbol_dan_catat_sinyal_buy(self):
        fake = _install_fake_learn_live()
        learning_bridge.record_entry("BTC/USDT", 65000.5, confidence=0.8, ta={"rsi": 30})
        fake.record_signal.assert_called_once_with(
            "BTCUSDT", "BUY", 65000.5, 0.8, None, {"rsi": 30}
        )

    def test_ta_default_dict_kosong(self):
        fake = _install_fake_learn_live()
        learning_bridge.record_entry("ETH/USDT", 3000.0)
        fake.record_signal.assert_called_once_with(
            "ETHUSDT", "BUY", 3000.0, None, None, {}
        )

    def test_noop_saat_setup_gagal(self):
        sys.modules["learn_live"] = None
        learning_bridge.record_entry("BTC/USDT", 65000.0)  # tak boleh melempar

    def test_fail_open_saat_record_signal_error(self):
        fake = _install_fake_learn_live()
        fake.record_signal.side_effect = RuntimeError("boom")
        # Trading tak boleh terganggu oleh kegagalan learning
        learning_bridge.record_entry("BTC/USDT", 65000.0)


class TestRefresh:
    # _setup() SELALU mengarahkan ACCURACY_FILE ke Path asli di
    # .learnings/trading/ (disengaja, supaya laporan tertulis ke lokasi
    # nyata) — jadi write_text harus di-monkeypatch di sini, bukan lewat
    # atribut fake module, kalau tidak test ini diam-diam menimpa file
    # ACCURACY_crypto_trader.md sungguhan di repo.
    def test_kembalikan_ringkasan_dan_tulis_laporan(self, monkeypatch):
        fake = _install_fake_learn_live()
        write_text = MagicMock()
        monkeypatch.setattr(Path, "write_text", write_text)
        result = learning_bridge.refresh()
        assert result == {
            "verified_now": 3,
            "total_signals": 10,
            "verified": 7,
            "pending": 3,
            "report": str(learning_bridge._REPORT),
        }
        write_text.assert_called_once_with("# laporan")

    def test_none_saat_setup_gagal(self):
        sys.modules["learn_live"] = None
        assert learning_bridge.refresh() is None

    def test_lewati_tulis_laporan_saat_analyze_accuracy_falsy(self, monkeypatch):
        fake = _install_fake_learn_live()
        fake.analyze_accuracy.return_value = None
        write_text = MagicMock()
        monkeypatch.setattr(Path, "write_text", write_text)
        result = learning_bridge.refresh()
        write_text.assert_not_called()
        assert result == {
            "verified_now": 3,
            "total_signals": 0,
            "verified": 0,
            "pending": 0,
            "report": str(learning_bridge._REPORT),
        }

    def test_fail_open_saat_verify_signals_error(self):
        fake = _install_fake_learn_live()
        fake.verify_signals.side_effect = RuntimeError("network down")
        assert learning_bridge.refresh() is None
