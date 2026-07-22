"""Tests for backtester.py -- the generic historical backtest harness used to
validate grid_trading, dca, and trend_following before any live use.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backtester import Backtester, SimulatedOrder, _calculate_rsi


def _candles(prices, spread=0.5, vol=1000.0):
    out = []
    prev = prices[0]
    for i, c in enumerate(prices):
        o = prev
        h = max(o, c) + spread
        l = min(o, c) - spread
        out.append([i * 3600000, o, h, l, c, vol])
        prev = c
    return out


class FakeExchangeManager:
    """Minimal stand-in for ExchangeManager.get_ohlcv/available_exchanges.

    _candles() timestamps start at epoch 0, far before any real start/end
    date the tests pass in -- so `since` is ignored (real exchanges return
    each candle once; here we just hand back the whole fixture on the first
    call). _fetch_historical_data's own `last_ts <= current_ts` check stops
    the loop after that single batch, exactly like it would if an exchange
    had no more data left to give.
    """

    def __init__(self, candles, available=("binance",)):
        self._candles = candles
        self.available_exchanges = list(available)

    def get_ohlcv(self, exchange, symbol, timeframe, limit=1000, since=0):
        return self._candles[:limit]


class TestCalculateRsi:
    """The regression this PR fixes: avg_loss == 0 (a pure uptrend, no red
    candles at all) must read RSI 100, not 0. backtester.py used to compute
    `avg_gain / avg_loss.replace(0, inf)`, which divides by infinity and
    silently produces 0 -- the opposite of "extremely overbought". The
    correct behavior already exists in strategies/trend_following.py's
    _calculate_rsi; this module now shares that exact formula.
    """

    def test_pure_uptrend_is_100_not_0(self):
        closes = pd.Series([100.0 + i for i in range(30)])
        rsi = _calculate_rsi(closes, period=14)
        assert rsi.iloc[-1] == 100.0
        assert rsi.iloc[-5:].tolist() == [100.0] * 5

    def test_pure_downtrend_is_0(self):
        closes = pd.Series([100.0 - i for i in range(30)])
        rsi = _calculate_rsi(closes, period=14)
        assert rsi.iloc[-1] == 0.0

    def test_flat_series_is_neutral_50(self):
        closes = pd.Series([100.0] * 30)
        rsi = _calculate_rsi(closes, period=14)
        assert rsi.iloc[-1] == 50.0

    def test_mixed_series_stays_within_bounds(self):
        closes = pd.Series([100.0, 101.0, 99.5, 102.0, 101.0, 103.0, 102.5, 104.0] * 4)
        rsi = _calculate_rsi(closes, period=14)
        valid = rsi.iloc[14:]
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()


class TestSimulatedOrder:
    def test_to_dict_computes_cost_and_fee(self):
        order = SimulatedOrder("BTC/USDT", "buy", amount=2.0, price=100.0, timestamp=123, fee_pct=0.1)
        d = order.to_dict()
        assert d["cost"] == 200.0
        assert d["fee"] == pytest.approx(0.2)
        assert d["symbol"] == "BTC/USDT"
        assert d["side"] == "buy"


class TestComputeMetrics:
    def test_win_rate_and_return(self):
        bt = Backtester(exchange_manager=None, initial_balance=1000.0, fee_pct=0.0, slippage_pct=0.0)
        ohlcv = _candles([100.0, 110.0])
        orders = [
            {"side": "buy", "cost": 100.0},
            {"side": "sell", "cost": 120.0},
        ]
        metrics = bt._compute_metrics(final_value=1020.0, orders=orders, total_fees=0.0, wins=1, losses=0, ohlcv=ohlcv)
        assert metrics["total_return_pct"] == 2.0
        assert metrics["win_rate_pct"] == 100.0
        assert metrics["total_trades"] == 1

    def test_zero_trades_does_not_crash(self):
        bt = Backtester(exchange_manager=None, initial_balance=1000.0)
        ohlcv = _candles([100.0, 100.0])
        metrics = bt._compute_metrics(final_value=1000.0, orders=[], total_fees=0.0, wins=0, losses=0, ohlcv=ohlcv)
        assert metrics["win_rate_pct"] == 0
        assert metrics["total_trades"] == 0
        assert metrics["sharpe_ratio"] == 0.0

    def test_drawdown_reflects_a_losing_trade(self):
        bt = Backtester(exchange_manager=None, initial_balance=1000.0)
        ohlcv = _candles([100.0, 90.0])
        orders = [
            {"side": "buy", "cost": 500.0},
            {"side": "sell", "cost": 400.0},
        ]
        metrics = bt._compute_metrics(final_value=900.0, orders=orders, total_fees=0.0, wins=0, losses=1, ohlcv=ohlcv)
        assert metrics["max_drawdown_pct"] > 0


class TestBacktestTrendEndToEnd:
    """Smoke coverage for _backtest_trend: a full run (entry on a bullish
    EMA cross, exit on overbought RSI / bearish cross) must produce a
    complete, well-formed buy+sell round trip. The RSI edge-case regression
    itself is proven precisely by TestCalculateRsi above; this only checks
    the simulation wiring around it doesn't break.
    """

    def _params(self):
        return {
            "symbol": "BTC/USDT",
            "ema_short": 3,
            "ema_long": 6,
            "rsi_period": 6,
            "rsi_overbought": 70,
            "order_amount_usdt": 25.0,
        }

    def test_rally_then_pullback_produces_a_buy_and_a_sell(self):
        bt = Backtester(exchange_manager=None, initial_balance=1000.0, slippage_pct=0.0, fee_pct=0.0)
        base = [100.0, 99.0, 100.2, 99.3, 100.1, 99.4, 100.0, 99.5, 100.0, 99.6]
        rally = [100.0 + i * 0.8 for i in range(1, 20)]
        pullback = [rally[-1] - i * 1.5 for i in range(1, 15)]
        ohlcv = _candles(base + rally + pullback)

        result = bt._backtest_trend(ohlcv, self._params())
        sides = [o["side"] for o in result["orders"]]
        assert sides.count("buy") >= 1
        assert sides.count("sell") >= 1
        assert sides[0] == "buy"


class TestRun:
    def test_unknown_strategy_returns_error(self):
        bt = Backtester(exchange_manager=FakeExchangeManager(_candles([100.0] * 60)))
        result = bt.run("not_a_real_strategy", {"symbol": "BTC/USDT"}, "2026-01-01", "2026-01-02")
        assert result["status"] == "error"
        assert "not implemented" in result["message"]

    def test_no_exchange_available_returns_error(self):
        bt = Backtester(exchange_manager=FakeExchangeManager(_candles([100.0] * 60), available=()))
        result = bt.run("dca", {"symbol": "BTC/USDT"}, "2026-01-01", "2026-01-02")
        assert result["status"] == "error"
        assert "No exchanges" in result["message"]

    def test_insufficient_candles_returns_error(self, tmp_path, monkeypatch):
        import backtester as backtester_mod
        monkeypatch.setattr(backtester_mod, "_BACKTESTS_DIR", tmp_path)
        bt = Backtester(exchange_manager=FakeExchangeManager(_candles([100.0] * 10)))
        result = bt.run("dca", {"symbol": "BTC/USDT"}, "2026-01-01", "2026-01-02")
        assert result["status"] == "error"
        assert "Not enough historical data" in result["message"]

    def test_dca_run_ok_saves_results(self, tmp_path, monkeypatch):
        import backtester as backtester_mod
        monkeypatch.setattr(backtester_mod, "_BACKTESTS_DIR", tmp_path)
        prices = [100.0 + (i % 5) for i in range(120)]
        bt = Backtester(exchange_manager=FakeExchangeManager(_candles(prices)))
        result = bt.run(
            "dca",
            {"symbol": "BTC/USDT", "amount_per_buy_usdt": 10.0, "interval": "hourly"},
            "2026-01-01", "2026-01-02",
        )
        assert result["status"] == "ok"
        assert result["strategy"] == "dca"
        assert "total_invested_usdt" in result
        saved = list(tmp_path.glob("dca_*.json"))
        assert len(saved) == 1
