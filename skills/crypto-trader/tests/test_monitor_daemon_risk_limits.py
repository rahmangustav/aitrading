"""Regression test: _check_risk_limits() must alert on BOTH daily-loss and
drawdown breaches.

Before this fix, the daily-loss branch called `notifier.send_alert(...)`
when approaching the limit, but the drawdown branch only logged a warning
-- the daemon accepts a `notifier` argument specifically to page someone
when risk limits are approached, and a portfolio drawing down toward its
max-drawdown limit is exactly the kind of thing that should not depend on
someone tailing a local log file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from monitor_daemon import MonitorDaemon  # noqa: E402
from notifier import Notifier  # noqa: E402


class FakeRiskManager:
    def __init__(self, daily_pnl_eur=0, drawdown_pct=0,
                 max_daily_loss_eur=1000, max_drawdown_pct=20):
        self._status = {
            "daily_pnl_eur": daily_pnl_eur,
            "drawdown_pct": drawdown_pct,
            "limits": {
                "max_daily_loss_eur": max_daily_loss_eur,
                "max_drawdown_pct": max_drawdown_pct,
            },
        }

    def get_status(self):
        return self._status


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    def send_alert(self, alert_type, data):
        self.alerts.append((alert_type, data))
        return {"status": "sent"}


def _daemon():
    return MonitorDaemon.__new__(MonitorDaemon)


def test_no_alert_when_limits_not_configured():
    daemon = _daemon()
    risk_manager = FakeRiskManager(daily_pnl_eur=-999999, drawdown_pct=999,
                                    max_daily_loss_eur=0, max_drawdown_pct=0)
    notifier = FakeNotifier()
    daemon._check_risk_limits(risk_manager, None, None, notifier)
    assert notifier.alerts == []


def test_no_alert_below_daily_loss_threshold():
    daemon = _daemon()
    risk_manager = FakeRiskManager(daily_pnl_eur=-500, max_daily_loss_eur=1000)
    notifier = FakeNotifier()
    daemon._check_risk_limits(risk_manager, None, None, notifier)
    assert notifier.alerts == []


def test_alert_sent_at_daily_loss_threshold():
    daemon = _daemon()
    # -800 / 1000 = 80% of the limit, the documented trigger point.
    risk_manager = FakeRiskManager(daily_pnl_eur=-800, max_daily_loss_eur=1000)
    notifier = FakeNotifier()
    daemon._check_risk_limits(risk_manager, None, None, notifier)
    assert len(notifier.alerts) == 1
    alert_type, data = notifier.alerts[0]
    assert alert_type == "risk_limit_hit"
    assert data["type"] == "daily_loss_warning"
    assert data["current_loss"] == 800
    assert data["limit"] == 1000


def test_no_alert_below_drawdown_threshold():
    daemon = _daemon()
    risk_manager = FakeRiskManager(drawdown_pct=10, max_drawdown_pct=20)
    notifier = FakeNotifier()
    daemon._check_risk_limits(risk_manager, None, None, notifier)
    assert notifier.alerts == []


def test_alert_sent_at_drawdown_threshold():
    """The bug: this branch never notified anyone before this fix."""
    daemon = _daemon()
    # 16 / 20 = 80% of the limit, the documented trigger point.
    risk_manager = FakeRiskManager(drawdown_pct=16, max_drawdown_pct=20)
    notifier = FakeNotifier()
    daemon._check_risk_limits(risk_manager, None, None, notifier)
    assert len(notifier.alerts) == 1
    alert_type, data = notifier.alerts[0]
    assert alert_type == "risk_limit_hit"
    assert data["type"] == "drawdown_warning"
    assert data["current_drawdown_pct"] == 16
    assert data["limit_pct"] == 20


def test_both_limits_breached_send_two_alerts():
    daemon = _daemon()
    risk_manager = FakeRiskManager(daily_pnl_eur=-900, max_daily_loss_eur=1000,
                                    drawdown_pct=18, max_drawdown_pct=20)
    notifier = FakeNotifier()
    daemon._check_risk_limits(risk_manager, None, None, notifier)
    assert len(notifier.alerts) == 2
    types = {alert_type for alert_type, _ in notifier.alerts}
    assert types == {"risk_limit_hit"}
    sub_types = {data["type"] for _, data in notifier.alerts}
    assert sub_types == {"daily_loss_warning", "drawdown_warning"}


def test_no_notifier_does_not_crash():
    daemon = _daemon()
    risk_manager = FakeRiskManager(daily_pnl_eur=-900, max_daily_loss_eur=1000,
                                    drawdown_pct=18, max_drawdown_pct=20)
    daemon._check_risk_limits(risk_manager, None, None, None)  # must not raise


def test_drawdown_alert_formats_as_percentage_not_eur():
    """Bridges to notifier.py: the drawdown payload must render with '%',
    not fall through to the EUR-shaped branch built for daily-loss alerts.
    """
    notifier = Notifier()
    message = notifier._format_message(
        "risk_limit_hit",
        {"type": "drawdown_warning", "current_drawdown_pct": 16.4, "limit_pct": 20},
        "normal",
    )
    assert "16.4%" in message
    assert "20.0%" in message
    assert "EUR" not in message
