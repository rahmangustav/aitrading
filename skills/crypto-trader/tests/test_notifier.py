"""Tests for notifier.py -- the alert system monitor_daemon.py relies on for
protective-stop and risk notifications (no test coverage before this file)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from notifier import Notifier


def _notifier(tmp_path, config=None):
    if config is not None:
        config_path = tmp_path / "notifications.yaml"
        import yaml
        config_path.write_text(yaml.dump(config))
        return Notifier(config_path=str(config_path))
    return Notifier(config_path=str(tmp_path / "does_not_exist.yaml"))


class TestLoadConfig:
    def test_missing_config_file_yields_empty_defaults(self, tmp_path):
        n = _notifier(tmp_path)
        assert n._channels == {}
        assert n._alert_rules == []
        assert n._max_per_minute == 10
        assert n._cooldown == 60

    def test_config_values_are_read(self, tmp_path):
        n = _notifier(tmp_path, {
            "notifications": {"telegram": {"enabled": True}},
            "alerts": [{"type": "large_loss", "channels": ["telegram"], "priority": "critical"}],
            "rate_limit": {"max_alerts_per_minute": 3, "cooldown_seconds": 30},
        })
        assert n._channels == {"telegram": {"enabled": True}}
        assert n._max_per_minute == 3
        assert n._cooldown == 30


class TestFindRule:
    def test_finds_matching_rule(self, tmp_path):
        n = _notifier(tmp_path, {
            "alerts": [
                {"type": "large_loss", "channels": ["email"], "priority": "critical"},
                {"type": "daily_summary", "channels": ["telegram"]},
            ],
        })
        rule = n._find_rule("large_loss")
        assert rule["priority"] == "critical"

    def test_returns_none_when_no_rule_matches(self, tmp_path):
        n = _notifier(tmp_path)
        assert n._find_rule("unknown_type") is None


class TestRateLimit:
    def test_allows_sends_under_the_limit(self, tmp_path):
        n = _notifier(tmp_path, {"rate_limit": {"max_alerts_per_minute": 2, "cooldown_seconds": 60}})
        assert n._check_rate_limit() is True
        n._record_send()
        assert n._check_rate_limit() is True
        n._record_send()
        assert n._check_rate_limit() is False

    def test_expired_timestamps_free_up_the_window(self, tmp_path):
        n = _notifier(tmp_path, {"rate_limit": {"max_alerts_per_minute": 1, "cooldown_seconds": 60}})
        n._sent_timestamps = [time.time() - 61]
        assert n._check_rate_limit() is True

    def test_send_alert_reports_rate_limited_without_sending(self, tmp_path):
        n = _notifier(tmp_path, {"rate_limit": {"max_alerts_per_minute": 1, "cooldown_seconds": 60}})
        n._record_send()
        result = n.send_alert("daily_summary", {})
        assert result == {"status": "rate_limited", "alert_type": "daily_summary"}


class TestFormatMessage:
    def test_unknown_alert_type_falls_back_to_key_value_dump(self, tmp_path):
        n = _notifier(tmp_path)
        msg = n._format_message("custom_event", {"foo": "bar", "baz": 1}, "normal")
        assert "foo: bar" in msg
        assert "baz: 1" in msg

    def test_trade_executed_includes_side_and_symbol(self, tmp_path):
        n = _notifier(tmp_path)
        msg = n._format_message(
            "trade_executed",
            {"strategy": "grid_trading", "side": "buy", "symbol": "BTC/USDT",
             "amount": 0.5, "price": 100.0, "exchange": "binance", "reason": "signal"},
            "normal",
        )
        assert "Action: BUY BTC/USDT" in msg
        assert "Strategy: grid_trading" in msg

    def test_priority_icon_selected_from_table(self, tmp_path):
        n = _notifier(tmp_path)
        critical_msg = n._format_message("emergency_stop", {}, "critical")
        low_msg = n._format_message("emergency_stop", {}, "low")
        assert critical_msg.startswith("[!!!]")
        assert low_msg.startswith("[.]")

    def test_unknown_priority_defaults_to_normal_icon(self, tmp_path):
        n = _notifier(tmp_path)
        msg = n._format_message("daily_summary", {}, "not_a_real_priority")
        assert msg.startswith("[i]")

    def test_large_loss_formats_percent_and_amount(self, tmp_path):
        n = _notifier(tmp_path)
        msg = n._format_message("large_loss", {"loss_pct": 12.345, "loss_eur": 99.5}, "high")
        assert "Loss: 12.35%" in msg
        assert "Amount: 99.50 EUR" in msg


class TestSendAlertChannelDispatch:
    def test_disabled_channel_is_reported_without_raising(self, tmp_path):
        n = _notifier(tmp_path, {
            "notifications": {"telegram": {"enabled": False}},
            "alerts": [{"type": "large_loss", "channels": ["telegram"]}],
        })
        result = n.send_alert("large_loss", {"loss_pct": 5, "loss_eur": 10})
        assert result["channels"]["telegram"] == "disabled"

    def test_missing_channel_config_is_treated_as_disabled(self, tmp_path):
        n = _notifier(tmp_path, {"alerts": [{"type": "large_loss", "channels": ["discord"]}]})
        result = n.send_alert("large_loss", {"loss_pct": 5, "loss_eur": 10})
        assert result["channels"]["discord"] == "disabled"

    def test_no_matching_rule_defaults_to_telegram_normal(self, tmp_path):
        n = _notifier(tmp_path)
        result = n.send_alert("unmapped_alert", {})
        assert result["priority"] == "normal"
        assert "telegram" in result["channels"]

    def test_channel_exception_is_caught_and_reported(self, tmp_path):
        n = _notifier(tmp_path, {
            "notifications": {"telegram": {"enabled": True}},
            "alerts": [{"type": "large_loss", "channels": ["telegram"]}],
        })
        with patch.object(n, "_send_telegram", side_effect=RuntimeError("boom")):
            result = n.send_alert("large_loss", {"loss_pct": 5, "loss_eur": 10})
        assert result["channels"]["telegram"] == "error: boom"


class TestSendChannelsWithoutCredentials:
    def test_telegram_without_credentials_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        n = _notifier(tmp_path)
        assert n._send_telegram("hello") is False

    def test_discord_without_webhook_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        n = _notifier(tmp_path)
        assert n._send_discord("hello") is False

    def test_email_without_credentials_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EMAIL_FROM", raising=False)
        monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
        monkeypatch.delenv("EMAIL_TO", raising=False)
        n = _notifier(tmp_path)
        assert n._send_email("subject", "body") is False
