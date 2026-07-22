"""Tests for the read-only monitor dashboard's data aggregation (`snapshot()`).

dashboard.py never places or cancels orders — it only reads exchange/risk/
strategy state and reshapes it for the web UI (modal, posisi, order &
stop-loss, PnL per koin). It had zero test coverage. `snapshot()` is a
sizeable pure-ish function (holdings filtering/valuation, order
normalization, per-symbol PnL aggregation across strategy instances) with
several places a refactor could silently break the numbers shown to the
user, so it is worth pinning down with real inputs/outputs.

All exchange/strategy/risk collaborators are mocked — no network, no ccxt
exchange, no real orders are touched.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import dashboard  # noqa: E402


def make_exchange_manager(
    demo=True,
    connected=True,
    orders=None,
    balance=None,
    tickers=None,
):
    """Build a MagicMock standing in for ExchangeManager."""
    e = MagicMock()
    e.demo = demo
    e.available_exchanges = ["binance"] if connected else []
    e._get_exchange.return_value = SimpleNamespace(options={})
    e.get_open_orders.return_value = orders or []
    e.get_balance.return_value = balance or {}
    tickers = tickers or {}

    def _get_ticker(exchange_name, symbol):
        if symbol not in tickers:
            raise KeyError(symbol)
        return {"last": tickers[symbol]}

    e.get_ticker.side_effect = _get_ticker
    # Use the real static stop-order detector so tests double as a
    # (light) integration check of that logic too.
    from exchange_manager import ExchangeManager
    e._is_stop_order = ExchangeManager._is_stop_order
    return e


def make_strategy_instance(symbol, bought=0.0, entry=0.0, invested=None,
                            position_amount=0.0, entry_price=0.0):
    """A fake strategy instance exposing either DCA-style or generic
    position attributes, matching what dashboard.py reads via getattr."""
    return SimpleNamespace(
        symbol=symbol,
        total_bought=bought,
        avg_price=entry,
        total_invested=invested if invested is not None else (bought * entry),
        position_amount=position_amount,
        entry_price=entry_price,
        params={},
    )


@pytest.fixture(autouse=True)
def patch_collaborators():
    """Patch every dashboard.py collaborator so tests never touch the
    real exchange_manager/risk_manager/strategy_engine/monitor_daemon
    modules (which need ccxt, a live env, disk state, etc.)."""
    with patch("dashboard.RiskManager") as rm_cls, \
         patch("dashboard.StrategyEngine") as se_cls, \
         patch("dashboard.MonitorDaemon") as md_cls, \
         patch("dashboard._register_strategies"):
        dashboard._em = None  # reset the module-level ExchangeManager cache
        yield SimpleNamespace(rm_cls=rm_cls, se_cls=se_cls, md_cls=md_cls)


def _wire_engine(se_cls, strategies_list, instances_by_id):
    """Make StrategyEngine(...) return a mock engine exposing
    list_strategies()/get_strategy_instance() as dashboard.py uses them."""
    eng = MagicMock()
    eng.list_strategies.return_value = strategies_list
    eng.get_strategy_instance.side_effect = lambda sid: instances_by_id.get(sid)
    se_cls.return_value = eng
    return eng


class TestSnapshotInitFailure:
    def test_exchange_manager_init_error_short_circuits(self, patch_collaborators):
        """If ExchangeManager() itself blows up, snapshot() should report
        connected=False and stop — it must not try to touch the other
        (now-uninitialized) collaborators."""
        with patch("dashboard.ExchangeManager", side_effect=RuntimeError("no api key")):
            out = dashboard.snapshot()
        assert out["connected"] is False
        assert any("init:" in e for e in out["errors"])
        # Sections that depend on `e` must never have been attempted.
        assert "orders" not in out
        assert "holdings" not in out


class TestOrdersAndHoldings:
    def test_orders_normalized_and_stop_flagged(self, patch_collaborators):
        orders = [
            {"symbol": "BTC/USDT", "side": "sell", "type": "stop_loss_limit",
             "amount": 0.01, "price": None, "stopPrice": 25000},
            {"symbol": "ETH/USDT", "side": "buy", "type": "limit",
             "amount": 1.5, "price": 1800},
        ]
        em = make_exchange_manager(orders=orders, balance={})
        _wire_engine(patch_collaborators.se_cls, [], {})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        assert out["n_stops"] == 1
        norm = {o["symbol"]: o for o in out["orders"]}
        assert norm["BTC/USDT"]["is_stop"] is True
        assert norm["BTC/USDT"]["stop"] == 25000
        assert norm["ETH/USDT"]["is_stop"] is False
        assert norm["ETH/USDT"]["price"] == 1800

    def test_holdings_exclude_stablecoins_and_zero_balances(self, patch_collaborators):
        balance = {
            "USDT": {"free": 100.0, "used": 0.0, "total": 100.0},
            "BTC": {"free": 0.01, "used": 0.0, "total": 0.01},
            "DOGE": {"free": 0.0, "used": 0.0, "total": 0.0},  # zero -> excluded
        }
        em = make_exchange_manager(balance=balance, tickers={"BTC/USDT": 50000.0})
        _wire_engine(patch_collaborators.se_cls, [], {})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        coins = {h["coin"] for h in out["holdings"]}
        assert coins == {"BTC"}
        assert out["holdings"][0]["value_usdt"] == pytest.approx(500.0)
        assert out["usdt"]["free"] == 100.0

    def test_holdings_only_price_majors_or_in_open_orders(self, patch_collaborators):
        """A random small-cap coin with no open order should be listed
        (so the operator can see it) but NOT priced, to avoid an extra
        ticker call per random dust asset."""
        orders = [{"symbol": "ADA/USDT", "side": "buy", "type": "limit",
                   "amount": 10, "price": 0.5}]
        balance = {
            "ADA": {"free": 100.0, "used": 0.0, "total": 100.0},  # in open orders
            "XRP": {"free": 50.0, "used": 0.0, "total": 50.0},    # not major, no order
        }
        em = make_exchange_manager(orders=orders, balance=balance,
                                    tickers={"ADA/USDT": 0.6})
        _wire_engine(patch_collaborators.se_cls, [], {})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        by_coin = {h["coin"]: h for h in out["holdings"]}
        assert by_coin["ADA"]["value_usdt"] == pytest.approx(60.0)
        assert by_coin["XRP"]["value_usdt"] is None

    def test_holdings_sorted_by_value_descending(self, patch_collaborators):
        balance = {
            "BTC": {"free": 0.001, "used": 0.0, "total": 0.001},
            "ETH": {"free": 1.0, "used": 0.0, "total": 1.0},
        }
        em = make_exchange_manager(
            balance=balance,
            tickers={"BTC/USDT": 50000.0, "ETH/USDT": 2000.0},
        )
        _wire_engine(patch_collaborators.se_cls, [], {})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        values = [h["value_usdt"] for h in out["holdings"]]
        assert values == sorted(values, reverse=True)
        # BTC 0.001 * 50000 = 50 USDT; ETH 1.0 * 2000 = 2000 USDT -> ETH first.
        assert out["holdings"][0]["coin"] == "ETH"


class TestPositionsPnl:
    def test_single_symbol_pnl_and_pct(self, patch_collaborators):
        """DCA-style instance: total_bought/avg_price/total_invested."""
        inst = make_strategy_instance("BTC/USDT", bought=0.02, entry=40000.0,
                                       invested=800.0)
        strategies_list = [{"strategy_id": "dca_1"}]
        em = make_exchange_manager(tickers={"BTC/USDT": 50000.0})
        _wire_engine(patch_collaborators.se_cls, strategies_list, {"dca_1": inst})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        assert len(out["positions"]) == 1
        pos = out["positions"][0]
        assert pos["symbol"] == "BTC/USDT"
        assert pos["entry"] == pytest.approx(40000.0)
        assert pos["price"] == pytest.approx(50000.0)
        assert pos["cost"] == pytest.approx(800.0)
        assert pos["value"] == pytest.approx(0.02 * 50000.0)
        assert pos["pnl"] == pytest.approx(200.0)
        assert pos["pnl_pct"] == pytest.approx(25.0)
        assert out["total_pnl"] == pytest.approx(200.0)
        assert out["total_cost"] == pytest.approx(800.0)

    def test_two_instances_same_symbol_are_aggregated(self, patch_collaborators):
        """Two strategies both holding BTC/USDT should be merged into one
        weighted-average position, not listed twice or averaged naively."""
        inst_a = make_strategy_instance("BTC/USDT", bought=0.01, entry=40000.0,
                                         invested=400.0)
        inst_b = make_strategy_instance("BTC/USDT", bought=0.03, entry=44000.0,
                                         invested=1320.0)
        strategies_list = [{"strategy_id": "a"}, {"strategy_id": "b"}]
        em = make_exchange_manager(tickers={"BTC/USDT": 50000.0})
        _wire_engine(patch_collaborators.se_cls, strategies_list,
                     {"a": inst_a, "b": inst_b})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        assert len(out["positions"]) == 1
        pos = out["positions"][0]
        total_bought = 0.04
        total_invested = 1720.0
        weighted_entry = total_invested / total_bought
        assert pos["amount"] == pytest.approx(total_bought)
        assert pos["cost"] == pytest.approx(total_invested)
        assert pos["entry"] == pytest.approx(weighted_entry, rel=1e-4)

    def test_zero_position_or_zero_entry_instances_are_skipped(self, patch_collaborators):
        flat = make_strategy_instance("ETH/USDT", bought=0.0, entry=0.0)
        strategies_list = [{"strategy_id": "flat"}]
        em = make_exchange_manager()
        _wire_engine(patch_collaborators.se_cls, strategies_list, {"flat": flat})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        assert out["positions"] == []
        assert out["total_pnl"] == 0.0

    def test_ticker_failure_prices_position_at_zero_not_crash(self, patch_collaborators):
        """If the ticker lookup fails for a held symbol, the position
        should still show up (px falls back to 0) instead of the whole
        section erroring out."""
        inst = make_strategy_instance("BTC/USDT", bought=0.01, entry=40000.0,
                                       invested=400.0)
        strategies_list = [{"strategy_id": "a"}]
        em = make_exchange_manager(tickers={})  # no BTC/USDT ticker -> KeyError
        _wire_engine(patch_collaborators.se_cls, strategies_list, {"a": inst})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        assert "pnl" not in " ".join(out["errors"])  # section itself didn't error
        pos = out["positions"][0]
        assert pos["price"] == 0
        assert pos["value"] == 0
        assert pos["pnl"] == pytest.approx(-400.0)

    def test_positions_sorted_by_pnl_descending(self, patch_collaborators):
        winner = make_strategy_instance("BTC/USDT", bought=0.01, entry=40000.0,
                                         invested=400.0)
        loser = make_strategy_instance("ETH/USDT", bought=1.0, entry=3000.0,
                                        invested=3000.0)
        strategies_list = [{"strategy_id": "w"}, {"strategy_id": "l"}]
        em = make_exchange_manager(tickers={"BTC/USDT": 50000.0, "ETH/USDT": 2000.0})
        _wire_engine(patch_collaborators.se_cls, strategies_list,
                     {"w": winner, "l": loser})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        assert [p["symbol"] for p in out["positions"]] == ["BTC/USDT", "ETH/USDT"]
        assert out["positions"][0]["pnl"] > out["positions"][1]["pnl"]


class TestSectionErrorIsolation:
    def test_daemon_error_does_not_break_rest_of_snapshot(self, patch_collaborators):
        patch_collaborators.md_cls.side_effect = RuntimeError("daemon file missing")
        em = make_exchange_manager()
        _wire_engine(patch_collaborators.se_cls, [], {})
        with patch("dashboard.ExchangeManager", return_value=em):
            out = dashboard.snapshot()

        assert "daemon" not in out
        assert any("daemon:" in e for e in out["errors"])
        # unrelated sections still populated
        assert out["holdings"] == []
        assert out["positions"] == []
