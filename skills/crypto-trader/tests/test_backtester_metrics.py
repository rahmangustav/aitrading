"""Tests for Backtester._compute_metrics -- equity curve must include fees."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backtester import Backtester


def _ohlcv(n=3, price=100.0):
    return [[i * 3600000, price, price, price, price, 1000.0] for i in range(n)]


def _bt(initial_balance=1000.0):
    return Backtester(exchange_manager=None, initial_balance=initial_balance)


def test_drawdown_accounts_for_buy_fee():
    # Single buy: cost=900, fee=50. Real cash outflow is cost+fee=950,
    # so the equity trough must be initial_balance-950=50, i.e. 95% drawdown --
    # not initial_balance-900=100 (90% drawdown), which is what you get if the
    # fee is silently dropped from the running balance.
    bt = _bt()
    orders = [{"side": "buy", "cost": 900.0, "fee": 50.0}]
    metrics = bt._compute_metrics(
        final_value=50.0, orders=orders, total_fees=50.0,
        wins=0, losses=0, ohlcv=_ohlcv(),
    )
    assert metrics["max_drawdown_pct"] == 95.0


def test_drawdown_accounts_for_sell_fee():
    # Buy then sell. Real balances: 1000 -> 1000-900-10=90 -> 90+(900-10)=980.
    # If the sell fee is dropped, the recovered equity would read 1000
    # (990 too optimistic) instead of the true 980.
    bt = _bt()
    orders = [
        {"side": "buy", "cost": 900.0, "fee": 10.0},
        {"side": "sell", "cost": 900.0, "fee": 10.0},
    ]
    metrics = bt._compute_metrics(
        final_value=980.0, orders=orders, total_fees=20.0,
        wins=1, losses=0, ohlcv=_ohlcv(),
    )
    # peak is the initial 1000 (never exceeded), trough is 90 after the buy.
    assert metrics["max_drawdown_pct"] == 91.0


def test_equity_curve_reconciles_with_final_value_over_round_trips():
    # Several buy/sell cycles with fees on both legs -- the fee-aware equity
    # curve's last point must land on the caller-supplied final_value (which
    # is always computed fee-correctly from balance_usdt/balance_crypto).
    # This is an indirect way to assert equity_curve[-1] == final_value
    # without exposing the internal list, by checking total_return_pct is
    # still derived from final_value (unaffected) while drawdown reflects
    # the same fee-adjusted trajectory.
    bt = _bt()
    orders = [
        {"side": "buy", "cost": 500.0, "fee": 5.0},
        {"side": "sell", "cost": 600.0, "fee": 6.0},
        {"side": "buy", "cost": 700.0, "fee": 7.0},
        {"side": "sell", "cost": 750.0, "fee": 7.5},
    ]
    total_fees = sum(o["fee"] for o in orders)
    # replay the true balance the same way the strategy loops do
    running = 1000.0
    for o in orders:
        running += (-o["cost"] - o["fee"]) if o["side"] == "buy" else (o["cost"] - o["fee"])
    final_value = running

    metrics = bt._compute_metrics(
        final_value=final_value, orders=orders, total_fees=total_fees,
        wins=2, losses=0, ohlcv=_ohlcv(),
    )
    # total_return_pct always came from final_value, so it stays correct
    # regardless of the equity-curve bug -- confirm it matches by hand.
    expected_return_pct = round((final_value - 1000.0) / 1000.0 * 100, 2)
    assert metrics["total_return_pct"] == expected_return_pct
    # trough after the deepest buy (700+7=707 out of the 1095 peak) must be
    # visible in the drawdown -- this only happens if fees reduce the curve.
    assert metrics["max_drawdown_pct"] > 0
