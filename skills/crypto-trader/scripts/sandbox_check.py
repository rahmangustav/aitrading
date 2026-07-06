#!/usr/bin/env python3
"""
Sandbox verification for the order-safety changes (PR: native stop-loss,
idempotent orders, precision guards).

Runs a REAL end-to-end flow against Binance **testnet** so you can confirm the
exchange accepts our STOP_LOSS_LIMIT format and that the cancel-before-sell path
works -- BEFORE risking real money.

Safety
------
- Refuses to run unless the exchange manager is in demo/sandbox mode
  (CRYPTO_DEMO=true). It will never touch a live account.
- Uses a tiny notional (just above the symbol minimum).
- Cleans up after itself: cancels stops and sells the test position back.

Prerequisites
-------------
- Testnet API keys from https://testnet.binance.vision in your .env:
      CRYPTO_DEMO=true
      BINANCE_API_KEY=<testnet key>
      BINANCE_API_SECRET=<testnet secret>
- Some testnet USDT balance (the testnet faucet funds it).

Usage
-----
    python sandbox_check.py                # BTC/USDT, default
    python sandbox_check.py --symbol ETH/USDT --exchange binance
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

_env_path = _SCRIPTS_DIR.parent.parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

from exchange_manager import ExchangeManager, ExchangeError  # noqa: E402
from risk_manager import RiskManager  # noqa: E402


_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


class Checker:
    def __init__(self, exchange: str, symbol: str) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.results: list[tuple[str, bool, str]] = []
        self.exchange_mgr = ExchangeManager()
        self.risk_mgr = RiskManager()

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok, detail))
        marker = _PASS if ok else _FAIL
        print(f"  [{marker}] {name}" + (f" -- {detail}" if detail else ""))

    # --- safety gate ------------------------------------------------------

    def preflight(self) -> bool:
        if not self.exchange_mgr.demo:
            print("\033[91mABORT: exchange manager is NOT in demo/sandbox mode.\033[0m")
            print("Set CRYPTO_DEMO=true before running. This script never trades live.")
            return False
        if self.exchange not in self.exchange_mgr.available_exchanges:
            print(f"ABORT: exchange '{self.exchange}' not initialized. "
                  f"Available: {self.exchange_mgr.available_exchanges}")
            return False
        print(f"Environment: SANDBOX (demo=True) | exchange={self.exchange} | symbol={self.symbol}")
        return True

    # --- helpers ----------------------------------------------------------

    def _stop_orders(self) -> list:
        orders = self.exchange_mgr.get_open_orders(self.exchange, self.symbol)
        return [o for o in orders if self.exchange_mgr._is_stop_order(o)]

    def _test_amount(self, price: float) -> float:
        markets = self.exchange_mgr.get_markets(self.exchange)
        market = markets.get(self.symbol, {})
        min_cost = (market.get("limits", {}).get("cost", {}).get("min")) or 10.0
        min_amt = self.exchange_mgr.get_min_order_amount(self.exchange, self.symbol) or 0.0
        # Aim ~30% above min notional so rounding never drops us under it.
        amount = max((min_cost * 1.3) / price, min_amt * 1.3)
        return amount

    # --- steps ------------------------------------------------------------

    def run(self) -> int:
        # Step 0: balance + price
        ticker = self.exchange_mgr.get_ticker(self.exchange, self.symbol)
        price = ticker.get("last") or ticker.get("bid")
        print(f"Reference price: {price}")

        # Step A: precision/min guard should REJECT a dust order.
        try:
            self.exchange_mgr.place_order(
                self.exchange, self.symbol, "buy", 1e-9, order_type="market",
            )
            self.record("precision guard rejects dust order", False, "order was NOT rejected")
        except ExchangeError:
            self.record("precision guard rejects dust order", True)

        # Step B: market buy a small test position.
        amount = self._test_amount(price)
        try:
            order = self.exchange_mgr.place_order(
                self.exchange, self.symbol, "buy", amount, order_type="market",
            )
            filled = order.get("filled") or order.get("amount") or amount
            self.record("market buy filled", order.get("status") in ("closed", "filled"),
                        f"id={order.get('id')} filled={filled} coid={order.get('clientOrderId')}")
        except ExchangeError as exc:
            self.record("market buy filled", False, str(exc))
            return self.summary()

        time.sleep(1.0)

        # Step C: place native protective stop (what monitor_daemon now does).
        entry = order.get("price") or price
        stop_price = self.risk_mgr.stop_loss_price(entry, side="buy")
        stop_ok = False
        try:
            stop = self.exchange_mgr.place_stop_loss_order(
                self.exchange, self.symbol, filled, stop_price,
            )
            time.sleep(1.0)
            resting = self._stop_orders()
            stop_ok = len(resting) >= 1
            self.record("native stop-loss rests on exchange", stop_ok,
                        f"stop_id={stop.get('id')} stop_price={stop.get('stop_price')} "
                        f"resting={len(resting)}")
        except ExchangeError as exc:
            self.record("native stop-loss rests on exchange", False, str(exc))

        # Step D: cancel-before-sell removes the stop (double-sell guard).
        try:
            self.exchange_mgr.cancel_stop_orders(self.exchange, self.symbol)
            time.sleep(1.0)
            remaining = self._stop_orders()
            self.record("cancel_stop_orders clears the stop", len(remaining) == 0,
                        f"remaining_stops={len(remaining)}")
        except ExchangeError as exc:
            self.record("cancel_stop_orders clears the stop", False, str(exc))

        # Step E: flatten -- sell the test position back.
        try:
            sell = self.exchange_mgr.place_order(
                self.exchange, self.symbol, "sell", filled, order_type="market",
            )
            self.record("test position flattened (sell back)",
                        sell.get("status") in ("closed", "filled"), f"id={sell.get('id')}")
        except ExchangeError as exc:
            self.record("test position flattened (sell back)", False,
                        f"{exc} -- FLATTEN MANUALLY on testnet!")

        # Final safety net: cancel anything left over for this symbol.
        try:
            self.exchange_mgr.cancel_all_orders(self.exchange, self.symbol)
        except ExchangeError:
            pass

        return self.summary()

    def summary(self) -> int:
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        print(f"\n=== {passed}/{total} checks passed ===")
        if passed == total:
            print("\033[92mAll sandbox checks passed. Safe to review for live "
                  "(start with tiny size).\033[0m")
            return 0
        print("\033[91mSome checks FAILED -- do NOT go live until resolved.\033[0m")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Sandbox order-safety verification")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    args = parser.parse_args()

    checker = Checker(args.exchange, args.symbol)
    if not checker.preflight():
        sys.exit(2)
    sys.exit(checker.run())


if __name__ == "__main__":
    main()
