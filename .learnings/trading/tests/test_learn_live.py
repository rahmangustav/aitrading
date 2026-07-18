"""Tests for the Live Learning Engine's signal verification & accuracy math.

verify_signals() feeds the winrate track record that gates real-money trading
(MEMORY.md: "Wajib backtest, win rate >=60%"). It was previously buggy — it
priced all three horizons (4h/8h/24h) off the price AT VERIFICATION TIME
instead of the price at each horizon's own timestamp, which silently fabricated
results whenever a run landed late (see commit "Perbaiki bug horizon di
verify_signals"). These tests pin the fixed behaviour so that regression can't
sneak back in unnoticed.
"""
from __future__ import annotations

import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import learn_live


def iso(dt):
    return dt.isoformat()


def make_signal(pair="BTCUSDT", signal="BUY", price=100.0, hours_ago=0.0, **overrides):
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    sig = {
        "ts": iso(ts),
        "pair": pair,
        "signal": signal,
        "price": price,
        "confidence": 70,
        "llm_sentiment": 0.5,
        "ta": {},
        "status": "open",
        "verified": False,
        "result_4h": None,
        "result_8h": None,
        "result_24h": None,
    }
    sig.update(overrides)
    return sig


@pytest.fixture
def signal_db_path(tmp_path, monkeypatch):
    db_file = tmp_path / "signal_db.json"
    monkeypatch.setattr(learn_live, "SIGNAL_DB", db_file)
    return db_file


def write_db(path, signals, **extra):
    payload = {"signals": signals, "accuracy": {}}
    payload.update(extra)
    path.write_text(json.dumps(payload, default=str))


# ── get_price_at ──

class TestGetPriceAt:
    def test_returns_open_of_kline(self):
        with patch.object(learn_live, "api_get", return_value=[["ignored", "123.45", "x"]]):
            price = learn_live.get_price_at("BTCUSDT", datetime.now(timezone.utc))
        assert price == 123.45

    def test_empty_klines_returns_zero(self):
        with patch.object(learn_live, "api_get", return_value=[]):
            assert learn_live.get_price_at("BTCUSDT", datetime.now(timezone.utc)) == 0

    def test_non_list_response_returns_zero(self):
        with patch.object(learn_live, "api_get", return_value={"code": -1121}):
            assert learn_live.get_price_at("BTCUSDT", datetime.now(timezone.utc)) == 0

    def test_exception_returns_zero(self):
        with patch.object(learn_live, "api_get", side_effect=RuntimeError("network down")):
            assert learn_live.get_price_at("BTCUSDT", datetime.now(timezone.utc)) == 0

    def test_uses_target_datetime_as_start_time(self):
        target = datetime(2026, 1, 1, tzinfo=timezone.utc)
        captured = {}

        def fake_api_get(path, params=None):
            captured["params"] = params
            return [["_", "1.0", "_"]]

        with patch.object(learn_live, "api_get", side_effect=fake_api_get):
            learn_live.get_price_at("BTCUSDT", target)

        assert captured["params"]["startTime"] == int(target.timestamp() * 1000)


# ── bulk_get_prices ──

class TestBulkGetPrices:
    def test_maps_symbol_to_float_price(self):
        raw = [{"symbol": "BTCUSDT", "price": "50000.5"}, {"symbol": "ETHUSDT", "price": "3000"}]
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value.json.return_value = raw
            prices = learn_live.bulk_get_prices()
        assert prices == {"BTCUSDT": 50000.5, "ETHUSDT": 3000.0}

    def test_network_error_returns_empty_dict(self):
        with patch("httpx.Client", side_effect=RuntimeError("boom")):
            assert learn_live.bulk_get_prices() == {}


# ── verify_signals ──

class TestVerifySignals:
    def test_signal_younger_than_4h_is_untouched(self, signal_db_path):
        write_db(signal_db_path, [make_signal(hours_ago=1)])
        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 100}):
            updated = learn_live.verify_signals()
        assert updated == 0
        sig = json.loads(signal_db_path.read_text())["signals"][0]
        assert sig["result_4h"] is None
        assert sig["verified"] is False

    def test_already_verified_signal_is_skipped(self, signal_db_path):
        write_db(signal_db_path, [make_signal(hours_ago=30, verified=True)])
        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 999}) as bulk, \
             patch.object(learn_live, "get_price_at") as gpa:
            updated = learn_live.verify_signals()
        gpa.assert_not_called()
        assert updated == 0

    def test_no_prices_available_short_circuits(self, signal_db_path):
        write_db(signal_db_path, [make_signal(hours_ago=30)])
        with patch.object(learn_live, "bulk_get_prices", return_value={}):
            updated = learn_live.verify_signals()
        assert updated == 0

    def test_pair_missing_from_bulk_prices_is_skipped(self, signal_db_path):
        write_db(signal_db_path, [make_signal(pair="DOGEUSDT", hours_ago=30)])
        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 100}), \
             patch.object(learn_live, "get_price_at") as gpa:
            updated = learn_live.verify_signals()
        gpa.assert_not_called()
        assert updated == 0

    def test_fills_only_horizons_that_have_elapsed(self, signal_db_path):
        # 5h old: only the 4h horizon has landed, 8h/24h have not.
        write_db(signal_db_path, [make_signal(price=100.0, hours_ago=5)])
        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 100}), \
             patch.object(learn_live, "get_price_at", return_value=110.0):
            updated = learn_live.verify_signals()
        sig = json.loads(signal_db_path.read_text())["signals"][0]
        assert updated == 1
        assert sig["result_4h"] == 10.0
        assert sig["result_8h"] is None
        assert sig["result_24h"] is None
        assert sig["verified"] is False

    def test_each_horizon_priced_at_its_own_timestamp_not_now(self, signal_db_path):
        """Regression test for the fixed bug: 4h/8h/24h must NOT collapse onto one price."""
        sig_time = datetime.now(timezone.utc) - timedelta(hours=30)
        write_db(signal_db_path, [make_signal(price=100.0, hours_ago=30)])

        prices_by_horizon = {4: 105.0, 8: 110.0, 24: 120.0}

        def fake_get_price_at(pair, target_dt):
            sig_ts = datetime.fromisoformat(json.loads(signal_db_path.read_text())["signals"][0]["ts"])
            elapsed = round((target_dt - sig_ts).total_seconds() / 3600)
            return prices_by_horizon[elapsed]

        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 999}), \
             patch.object(learn_live, "get_price_at", side_effect=fake_get_price_at):
            updated = learn_live.verify_signals()

        sig = json.loads(signal_db_path.read_text())["signals"][0]
        assert updated == 3
        assert sig["result_4h"] == 5.0
        assert sig["result_8h"] == 10.0
        assert sig["result_24h"] == 20.0
        # All three landed and differ — proves horizons aren't collapsed onto
        # a single "current price" the way the old buggy code did.
        assert len({sig["result_4h"], sig["result_8h"], sig["result_24h"]}) == 3
        assert sig["verified"] is True

    def test_stays_unverified_if_any_horizon_never_lands(self, signal_db_path):
        # 24h+ old, but one horizon keeps failing to fetch (e.g. delisted pair briefly).
        write_db(signal_db_path, [make_signal(price=100.0, hours_ago=30)])

        def flaky_get_price_at(pair, target_dt):
            # 8h horizon fetch fails (returns 0), the others succeed.
            sig_ts = datetime.fromisoformat(json.loads(signal_db_path.read_text())["signals"][0]["ts"])
            elapsed = round((target_dt - sig_ts).total_seconds() / 3600)
            return 0 if elapsed == 8 else 110.0

        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 999}), \
             patch.object(learn_live, "get_price_at", side_effect=flaky_get_price_at):
            learn_live.verify_signals()

        sig = json.loads(signal_db_path.read_text())["signals"][0]
        assert sig["result_4h"] is not None
        assert sig["result_8h"] is None
        assert sig["result_24h"] is not None
        assert sig["verified"] is False

    def test_batch_limit_of_30_signals_per_run(self, signal_db_path):
        signals = [make_signal(pair="BTCUSDT", hours_ago=30) for _ in range(35)]
        write_db(signal_db_path, signals)
        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 100}), \
             patch.object(learn_live, "get_price_at", return_value=110.0):
            learn_live.verify_signals()
        db = json.loads(signal_db_path.read_text())
        touched = [s for s in db["signals"] if s["result_4h"] is not None]
        assert len(touched) == 30

    def test_no_save_when_nothing_updated(self, signal_db_path):
        write_db(signal_db_path, [make_signal(hours_ago=1)])
        before = signal_db_path.read_text()
        with patch.object(learn_live, "bulk_get_prices", return_value={"BTCUSDT": 100}):
            learn_live.verify_signals()
        assert signal_db_path.read_text() == before


# ── analyze_accuracy ──

class TestAnalyzeAccuracy:
    def test_no_signals_returns_none(self, signal_db_path):
        write_db(signal_db_path, [])
        assert learn_live.analyze_accuracy() is None

    def test_buy_correct_when_price_went_up(self, signal_db_path):
        sig = make_signal(pair="BTCUSDT", signal="BUY", result_4h=5.0, ts=iso(datetime(2026, 1, 1, 10, tzinfo=timezone.utc)))
        write_db(signal_db_path, [sig])
        result = learn_live.analyze_accuracy()
        assert result["pair_stats"]["BTCUSDT"]["correct_4h"] == 1
        assert result["pair_stats"]["BTCUSDT"]["accuracy"] == 100.0

    def test_sell_correct_when_price_went_down(self, signal_db_path):
        sig = make_signal(pair="BTCUSDT", signal="SELL", result_4h=-5.0, ts=iso(datetime(2026, 1, 1, 10, tzinfo=timezone.utc)))
        write_db(signal_db_path, [sig])
        result = learn_live.analyze_accuracy()
        assert result["pair_stats"]["BTCUSDT"]["correct_4h"] == 1

    def test_buy_incorrect_when_price_went_down(self, signal_db_path):
        sig = make_signal(pair="BTCUSDT", signal="BUY", result_4h=-5.0, ts=iso(datetime(2026, 1, 1, 10, tzinfo=timezone.utc)))
        write_db(signal_db_path, [sig])
        result = learn_live.analyze_accuracy()
        assert result["pair_stats"]["BTCUSDT"]["correct_4h"] == 0
        assert result["pair_stats"]["BTCUSDT"]["accuracy"] == 0.0

    def test_signals_without_result_4h_are_not_counted(self, signal_db_path):
        sig = make_signal(pair="BTCUSDT", signal="BUY", result_4h=None)
        write_db(signal_db_path, [sig])
        result = learn_live.analyze_accuracy()
        assert result["pair_stats"] == {}

    def test_verified_and_pending_counts(self, signal_db_path):
        write_db(signal_db_path, [
            make_signal(verified=True),
            make_signal(verified=False),
            make_signal(verified=False),
        ])
        result = learn_live.analyze_accuracy()
        assert result["total_signals"] == 3
        assert result["verified"] == 1
        assert result["pending"] == 2

    def test_sentiment_bucketing(self, signal_db_path):
        sig = make_signal(pair="BTCUSDT", signal="BUY", result_4h=1.0, llm_sentiment=0.8,
                           ts=iso(datetime(2026, 1, 1, 10, tzinfo=timezone.utc)))
        write_db(signal_db_path, [sig])
        result = learn_live.analyze_accuracy()
        assert result["sentiment_buckets"]["0.70-1.00"]["total"] == 1
        assert result["sentiment_buckets"]["0.70-1.00"]["correct"] == 1


# ── generate_accuracy_report ──

class TestGenerateAccuracyReport:
    def test_no_signals_message(self):
        assert learn_live.generate_accuracy_report(None) == "No signals to analyze yet."
        assert learn_live.generate_accuracy_report({"total_signals": 0}) == "No signals to analyze yet."

    def test_report_includes_pair_row_above_min_sample(self, signal_db_path):
        signals = [
            make_signal(pair="BTCUSDT", signal="BUY", result_4h=v, ts=iso(datetime(2026, 1, 1, 10, tzinfo=timezone.utc)))
            for v in (5.0, 3.0, 2.0)
        ]
        write_db(signal_db_path, signals)
        result = learn_live.analyze_accuracy()
        report = learn_live.generate_accuracy_report(result)
        assert "BTCUSDT" in report
        assert "Total signals: 3" in report
