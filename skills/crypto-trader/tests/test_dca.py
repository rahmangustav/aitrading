"""Tests for DCAStrategy.

Focus: on_order_filled's price fallback used to derive price from
cost / requested "amount" instead of cost / actually-"filled" amount --
the same class of bug already found and fixed in trend_following.py,
swing_trading.py and scalping.py (on_order_filled reading the wrong
order field). For DCA this corrupts avg_price and total_invested, which
directly feeds the max_total_investment_usdt spending cap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from strategies.dca import DCAStrategy  # noqa: E402


def _make_strategy(**params):
    exchange_mgr = MagicMock()
    exchange_mgr.available_exchanges = ["binance"]
    risk_mgr = MagicMock()
    strat = DCAStrategy(
        strategy_id="dca_test",
        params={"exchange": "binance", **params},
        exchange_manager=exchange_mgr,
        risk_manager=risk_mgr,
    )
    strat.on_start()
    return strat


class TestOnStart:
    def test_uses_first_available_exchange_when_unset(self):
        strat = _make_strategy(exchange="")
        assert strat.exchange == "binance"
        assert strat.active is True

    def test_deactivates_when_no_exchanges_available(self):
        exchange_mgr = MagicMock()
        exchange_mgr.available_exchanges = []
        strat = DCAStrategy(
            strategy_id="dca_test",
            params={},
            exchange_manager=exchange_mgr,
            risk_manager=MagicMock(),
        )
        strat.on_start()
        assert strat.active is False


class TestEvaluate:
    def test_first_call_buys_immediately(self):
        strat = _make_strategy(amount_per_buy_usdt=10.0)
        strat.exchange_manager.get_ticker.return_value = {"last": 100.0}

        signals = strat.evaluate()

        assert len(signals) == 1
        assert signals[0]["side"] == "buy"
        assert signals[0]["amount"] == pytest.approx(0.1)

    def test_skips_when_interval_has_not_elapsed(self):
        strat = _make_strategy(amount_per_buy_usdt=10.0, interval="daily")
        strat.exchange_manager.get_ticker.return_value = {"last": 100.0}
        strat.on_order_filled({"filled": 0.1, "average": 100.0, "cost": 10.0})

        signals = strat.evaluate()

        assert signals == []

    def test_stops_once_max_total_investment_reached(self):
        strat = _make_strategy(amount_per_buy_usdt=10.0, max_total_investment_usdt=50.0)
        strat.total_invested = 50.0

        signals = strat.evaluate()

        assert signals == []

    def test_returns_no_signal_when_ticker_unavailable(self):
        strat = _make_strategy(amount_per_buy_usdt=10.0)
        strat.exchange_manager.get_ticker.return_value = {"last": 0}

        signals = strat.evaluate()

        assert signals == []

    def test_returns_no_signal_when_ticker_raises(self):
        strat = _make_strategy(amount_per_buy_usdt=10.0)
        strat.exchange_manager.get_ticker.side_effect = RuntimeError("network down")

        signals = strat.evaluate()

        assert signals == []


class TestOnOrderFilled:
    def test_accumulates_across_buys_and_tracks_avg_price(self):
        strat = _make_strategy()
        strat.on_order_filled({"filled": 1.0, "average": 100.0, "cost": 100.0})
        strat.on_order_filled({"filled": 1.0, "average": 200.0, "cost": 200.0})

        assert strat.total_bought == pytest.approx(2.0)
        assert strat.total_invested == pytest.approx(300.0)
        assert strat.avg_price == pytest.approx(150.0)
        assert strat.buy_count == 2
        assert strat.stats["trades_executed"] == 2

    def test_price_prefers_average_field(self):
        strat = _make_strategy()
        strat.on_order_filled({"filled": 1.0, "average": 100.0, "price": 999.0, "cost": 100.0})

        assert strat.avg_price == pytest.approx(100.0)

    def test_price_falls_back_to_price_field(self):
        strat = _make_strategy()
        strat.on_order_filled({"filled": 1.0, "price": 150.0, "cost": 150.0})

        assert strat.avg_price == pytest.approx(150.0)

    def test_partial_fill_does_not_corrupt_accounting(self):
        """total_invested/total_bought/avg_price come straight from the
        order's own "cost" and "filled" fields, so a requested "amount"
        that differs from what actually filled (partial fill) must not
        skew them.
        """
        strat = _make_strategy()
        strat.on_order_filled({"amount": 0.1, "filled": 0.05, "cost": 5.0})

        assert strat.total_bought == pytest.approx(0.05)
        assert strat.total_invested == pytest.approx(5.0)
        assert strat.avg_price == pytest.approx(100.0)

    def test_logged_price_derived_from_filled_not_requested_amount(self, caplog):
        """Regression test for the cost/amount bug in the log line.

        total_invested/avg_price are accounted for correctly regardless
        (see test above), but the *logged* per-unit price used to derive
        cost when "average"/"price" are both absent divided cost by the
        requested "amount" instead of the actually-"filled" amount --
        wrong by 2x here (50 instead of 100) and misleading to anyone
        reading the logs to sanity-check a DCA buy.
        """
        strat = _make_strategy()
        with caplog.at_level("INFO", logger="crypto-trader.strategy.dca"):
            strat.on_order_filled({"amount": 0.1, "filled": 0.05, "cost": 5.0})

        [record] = [r for r in caplog.records if "DCA buy filled" in r.message]
        assert "at 100.00" in record.message
        assert "at 50.00" not in record.message

    def test_amount_prefers_filled_over_requested_amount(self):
        strat = _make_strategy()
        strat.on_order_filled({"amount": 0.1, "filled": 0.05, "average": 100.0})

        assert strat.total_bought == pytest.approx(0.05)


class TestPersistAttrs:
    def test_persist_attrs_cover_dca_progress_fields(self):
        strat = _make_strategy()
        assert set(strat._persist_attrs) == {
            "total_invested", "total_bought", "buy_count",
            "avg_price", "last_buy_time",
        }

    def test_get_state_round_trips_through_restore(self):
        strat = _make_strategy()
        strat.on_order_filled({"filled": 2.0, "average": 100.0, "cost": 200.0})

        state = strat.get_state()

        fresh = _make_strategy()
        fresh.restore_state(state)

        assert fresh.total_invested == pytest.approx(200.0)
        assert fresh.total_bought == pytest.approx(2.0)
        assert fresh.avg_price == pytest.approx(100.0)
        assert fresh.buy_count == 1


class TestToDict:
    def test_includes_dca_progress_fields(self):
        strat = _make_strategy(interval="daily")
        strat.on_order_filled({"filled": 1.0, "average": 100.0, "cost": 100.0})

        data = strat.to_dict()

        assert data["total_invested_usdt"] == pytest.approx(100.0)
        assert data["total_bought"] == pytest.approx(1.0)
        assert data["buy_count"] == 1
        assert data["avg_price"] == pytest.approx(100.0)
        assert data["interval"] == "daily"
        assert data["next_buy_in_seconds"] >= 0
