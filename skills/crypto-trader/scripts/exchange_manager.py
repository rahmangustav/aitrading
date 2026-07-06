"""
Exchange Manager -- unified CCXT wrapper for multi-exchange crypto trading.

Handles authentication, rate-limit awareness, caching, retry logic, sandbox
mode, and all required REST endpoints for balances, tickers, order books,
orders, and OHLCV data.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt
import yaml

from cache import TTLCache

logger = logging.getLogger("crypto-trader.exchange")

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.5

# Protective stop-loss orders are placed as STOP_LOSS_LIMIT with the limit
# price sitting this percentage below the stop trigger, so they still fill in a
# fast-moving market instead of resting unfilled.
_STOP_ORDER_TYPE = "STOP_LOSS_LIMIT"
_STOP_LIMIT_OFFSET_PCT = 0.5

_ERROR_MESSAGES: Dict[int, str] = {
    400: "Bad request. Check your parameters.",
    401: "Invalid API credentials. Check your API key and secret.",
    403: "Access denied. Your API key lacks the required permissions.",
    404: "Resource not found. Check the symbol or order ID.",
    429: "Rate limit reached. Retrying after delay.",
    418: "IP has been auto-banned for exceeding rate limits.",
}

_ENV_KEY_MAP: Dict[str, Dict[str, str]] = {
    "binance": {"key": "BINANCE_API_KEY", "secret": "BINANCE_API_SECRET"},
    "bybit": {"key": "BYBIT_API_KEY", "secret": "BYBIT_API_SECRET"},
    "kraken": {"key": "KRAKEN_API_KEY", "secret": "KRAKEN_API_SECRET"},
    "coinbase": {"key": "COINBASE_API_KEY", "secret": "COINBASE_API_SECRET"},
}


class ExchangeError(Exception):
    """Raised when an exchange API call fails after retries."""

    def __init__(self, exchange: str, message: str, status_code: Optional[int] = None) -> None:
        self.exchange = exchange
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{exchange}] {message}")


class ExchangeManager:
    """Unified wrapper around CCXT exchanges with retry, caching, and sandbox."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config = self._load_config(config_path)
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        self._cache = TTLCache(default_ttl=30.0)
        self._rate_limits: Dict[str, float] = {}
        self._demo = os.environ.get("CRYPTO_DEMO", "true").lower() == "true"

        self._init_exchanges()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        path = Path(config_path) if config_path else _CONFIG_DIR / "exchanges.yaml"
        if not path.exists():
            return {"exchanges": {}}
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {"exchanges": {}}

    def _init_exchanges(self) -> None:
        """Initialize all enabled exchanges from config."""
        exchanges_cfg = self._config.get("exchanges", {})

        for name, cfg in exchanges_cfg.items():
            if not cfg.get("enabled", False):
                continue

            env_keys = _ENV_KEY_MAP.get(name, {})
            api_key = os.environ.get(env_keys.get("key", ""), "")
            api_secret = os.environ.get(env_keys.get("secret", ""), "")

            if not api_key or not api_secret:
                logger.warning("Skipping %s: API key or secret not set in environment.", name)
                continue

            exchange_class = getattr(ccxt, name, None)
            if exchange_class is None:
                logger.warning("Skipping %s: not supported by CCXT.", name)
                continue

            exchange_opts: Dict[str, Any] = {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": cfg.get("default_type", "spot"),
                },
            }

            exchange = exchange_class(exchange_opts)

            # Demo mode always forces sandbox; the per-exchange config can
            # only opt out of sandbox when CRYPTO_DEMO is explicitly false.
            use_sandbox = self._demo or cfg.get("sandbox", True)
            if use_sandbox:
                sandbox_urls = cfg.get("sandbox_urls")
                if sandbox_urls:
                    if "api" in sandbox_urls:
                        if isinstance(exchange.urls.get("api"), dict):
                            for key in exchange.urls["api"]:
                                exchange.urls["api"][key] = sandbox_urls["api"]
                        else:
                            exchange.urls["api"] = sandbox_urls["api"]
                    if "ws" in sandbox_urls:
                        exchange.urls["ws"] = sandbox_urls["ws"]
                    logger.info("Exchange %s: using custom sandbox URLs.", name)
                else:
                    try:
                        exchange.set_sandbox_mode(True)
                        logger.info("Exchange %s: sandbox mode enabled via CCXT.", name)
                    except Exception:
                        logger.error(
                            "Exchange %s: sandbox mode not available. Skipping "
                            "exchange instead of falling back to live URLs.",
                            name,
                        )
                        continue

            self._exchanges[name] = exchange
            logger.info("Exchange %s initialized (sandbox=%s).", name, use_sandbox)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def demo(self) -> bool:
        return self._demo

    @property
    def available_exchanges(self) -> List[str]:
        return list(self._exchanges.keys())

    def _get_exchange(self, name: str) -> ccxt.Exchange:
        exchange = self._exchanges.get(name)
        if exchange is None:
            available = ", ".join(self._exchanges.keys()) or "none"
            raise ExchangeError(name, f"Exchange not initialized. Available: {available}")
        return exchange

    def _wait_for_rate_limit(self, exchange_name: str, min_interval_ms: float) -> None:
        key = exchange_name
        now = time.time()
        earliest = self._rate_limits.get(key, 0.0)
        if now < earliest:
            sleep_time = earliest - now
            logger.debug("Rate limit: sleeping %.2fs for %s", sleep_time, exchange_name)
            time.sleep(sleep_time)
        self._rate_limits[key] = time.time() + (min_interval_ms / 1000.0)

    def _execute_with_retry(
        self,
        exchange_name: str,
        operation: str,
        func: Any,
        *args: Any,
        idempotent: bool = True,
        **kwargs: Any,
    ) -> Any:
        exchange_cfg = self._config.get("exchanges", {}).get(exchange_name, {})
        rate_limit_ms = exchange_cfg.get("rate_limit_ms", 100)

        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                self._wait_for_rate_limit(exchange_name, rate_limit_ms)
                result = func(*args, **kwargs)
                return result
            except ccxt.RateLimitExceeded as exc:
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "%s %s: rate limit (attempt %d/%d), waiting %.1fs",
                    exchange_name, operation, attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_error = exc
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                # For non-idempotent calls (order creation) the request may have
                # reached the exchange and executed even though we received no
                # response. Retrying could place a DUPLICATE order, so never
                # auto-retry -- surface it and let the caller reconcile.
                if not idempotent:
                    raise ExchangeError(
                        exchange_name,
                        f"{operation} failed with a network/availability error "
                        f"and was NOT retried to avoid placing a duplicate "
                        f"order: {exc}. Verify the order status on the exchange "
                        f"before retrying.",
                    ) from exc
                factor = 2 if isinstance(exc, ccxt.ExchangeNotAvailable) else 1
                wait = _RETRY_BACKOFF_BASE ** attempt * factor
                logger.warning(
                    "%s %s: network/availability error (attempt %d/%d): %s",
                    exchange_name, operation, attempt, _MAX_RETRIES, str(exc),
                )
                time.sleep(wait)
                last_error = exc
            except ccxt.AuthenticationError as exc:
                raise ExchangeError(
                    exchange_name,
                    f"Authentication failed: {exc}",
                    status_code=401,
                ) from exc
            except ccxt.InsufficientFunds as exc:
                raise ExchangeError(
                    exchange_name,
                    f"Insufficient funds: {exc}",
                    status_code=400,
                ) from exc
            except ccxt.InvalidOrder as exc:
                raise ExchangeError(
                    exchange_name,
                    f"Invalid order: {exc}",
                    status_code=400,
                ) from exc
            except ccxt.ExchangeError as exc:
                raise ExchangeError(
                    exchange_name,
                    f"Exchange error: {exc}",
                ) from exc

        raise ExchangeError(
            exchange_name,
            f"Failed after {_MAX_RETRIES} retries for {operation}: {last_error}",
        )

    # ------------------------------------------------------------------
    # Public API: Market Data
    # ------------------------------------------------------------------

    def get_ticker(self, exchange_name: str, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker for a symbol."""
        cache_key = f"ticker:{exchange_name}:{symbol}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        exchange = self._get_exchange(exchange_name)
        ticker = self._execute_with_retry(
            exchange_name, f"fetch_ticker({symbol})",
            exchange.fetch_ticker, symbol,
        )
        result = {
            "symbol": ticker.get("symbol", symbol),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "last": ticker.get("last"),
            "high": ticker.get("high"),
            "low": ticker.get("low"),
            "volume": ticker.get("baseVolume"),
            "quote_volume": ticker.get("quoteVolume"),
            "timestamp": ticker.get("timestamp"),
            "change_pct": ticker.get("percentage"),
        }
        self._cache.set(cache_key, result, ttl=10.0)
        return result

    def get_orderbook(
        self, exchange_name: str, symbol: str, limit: int = 10
    ) -> Dict[str, Any]:
        """Fetch order book for a symbol."""
        cache_key = f"orderbook:{exchange_name}:{symbol}:{limit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        exchange = self._get_exchange(exchange_name)
        book = self._execute_with_retry(
            exchange_name, f"fetch_order_book({symbol})",
            exchange.fetch_order_book, symbol, limit,
        )
        result = {
            "symbol": symbol,
            "bids": book.get("bids", [])[:limit],
            "asks": book.get("asks", [])[:limit],
            "timestamp": book.get("timestamp"),
            "spread": None,
        }
        if result["bids"] and result["asks"]:
            best_bid = result["bids"][0][0]
            best_ask = result["asks"][0][0]
            result["spread"] = round(best_ask - best_bid, 8)
            result["spread_pct"] = round((result["spread"] / best_ask) * 100, 4) if best_ask > 0 else 0
        self._cache.set(cache_key, result, ttl=5.0)
        return result

    def get_ohlcv(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[int] = None,
    ) -> List[List[Any]]:
        """Fetch OHLCV candlestick data.

        Returns list of [timestamp, open, high, low, close, volume].
        """
        cache_key = f"ohlcv:{exchange_name}:{symbol}:{timeframe}:{limit}:{since}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        exchange = self._get_exchange(exchange_name)
        ohlcv = self._execute_with_retry(
            exchange_name, f"fetch_ohlcv({symbol}, {timeframe})",
            exchange.fetch_ohlcv, symbol, timeframe, since, limit,
        )
        self._cache.set(cache_key, ohlcv, ttl=30.0)
        return ohlcv

    def get_markets(self, exchange_name: str) -> Dict[str, Any]:
        """Fetch available markets / trading pairs."""
        cache_key = f"markets:{exchange_name}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        exchange = self._get_exchange(exchange_name)
        markets = self._execute_with_retry(
            exchange_name, "load_markets",
            exchange.load_markets,
        )
        self._cache.set(cache_key, markets, ttl=3600.0)
        return markets

    # ------------------------------------------------------------------
    # Public API: Account
    # ------------------------------------------------------------------

    def get_balance(self, exchange_name: str, asset: Optional[str] = None) -> Dict[str, Any]:
        """Fetch account balances. If asset is given, return only that asset."""
        cache_key = f"balance:{exchange_name}"
        cached = self._cache.get(cache_key)
        if cached is None:
            exchange = self._get_exchange(exchange_name)
            raw_balance = self._execute_with_retry(
                exchange_name, "fetch_balance",
                exchange.fetch_balance,
            )
            balances: Dict[str, Any] = {}
            total = raw_balance.get("total", {})
            free = raw_balance.get("free", {})
            used = raw_balance.get("used", {})

            for currency in total:
                t = total.get(currency, 0) or 0
                if t > 0:
                    balances[currency] = {
                        "total": t,
                        "free": free.get(currency, 0) or 0,
                        "used": used.get(currency, 0) or 0,
                    }
            cached = balances
            self._cache.set(cache_key, cached, ttl=15.0)

        if asset:
            entry = cached.get(asset.upper())
            if entry is None:
                return {"asset": asset.upper(), "total": 0, "free": 0, "used": 0}
            return {"asset": asset.upper(), **entry}
        return cached

    # ------------------------------------------------------------------
    # Public API: Orders
    # ------------------------------------------------------------------

    @staticmethod
    def _new_client_order_id(prefix: str = "ct") -> str:
        """Generate a unique clientOrderId used as an idempotency key."""
        return f"{prefix}-{uuid.uuid4().hex[:24]}"

    def _normalize_amount(self, exchange_name: str, symbol: str, amount: float) -> float:
        """Round an amount to the exchange lot precision and enforce the minimum."""
        exchange = self._get_exchange(exchange_name)
        rounded = amount
        try:
            rounded = float(exchange.amount_to_precision(symbol, amount))
        except Exception as exc:  # pragma: no cover - depends on live market data
            logger.debug("amount_to_precision failed for %s: %s", symbol, exc)
        if rounded <= 0:
            raise ExchangeError(
                exchange_name,
                f"Order amount {amount} for {symbol} rounds to zero at the "
                f"exchange lot size.", status_code=400,
            )
        min_amt = self.get_min_order_amount(exchange_name, symbol)
        if min_amt is not None and rounded < min_amt:
            raise ExchangeError(
                exchange_name,
                f"Order amount {rounded} is below the minimum {min_amt} for {symbol}.",
                status_code=400,
            )
        return rounded

    def _normalize_price(self, exchange_name: str, symbol: str, price: float) -> float:
        """Round a price to the exchange tick precision."""
        exchange = self._get_exchange(exchange_name)
        try:
            return float(exchange.price_to_precision(symbol, price))
        except Exception as exc:  # pragma: no cover - depends on live market data
            logger.debug("price_to_precision failed for %s: %s", symbol, exc)
            return price

    def _check_min_notional(
        self, exchange_name: str, symbol: str, amount: float, price: Optional[float]
    ) -> None:
        """Reject orders whose notional value is below the exchange minimum."""
        if not price or price <= 0:
            return
        markets = self.get_markets(exchange_name)
        market = markets.get(symbol) or {}
        min_cost = market.get("limits", {}).get("cost", {}).get("min")
        notional = amount * price
        if min_cost is not None and notional < min_cost:
            raise ExchangeError(
                exchange_name,
                f"Order notional {notional:.4f} is below the exchange minimum "
                f"{min_cost} for {symbol}.", status_code=400,
            )

    def place_order(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place a new order. Returns order info dict.

        The amount (and price for limit orders) is rounded to the exchange
        precision and validated against the symbol's minimum amount/notional
        before submission. A clientOrderId is attached and the create call is
        never auto-retried on ambiguous network errors, so a timed-out request
        cannot silently place a duplicate order.
        """
        exchange = self._get_exchange(exchange_name)

        if order_type == "limit" and price is None:
            raise ExchangeError(exchange_name, "Limit orders require a price parameter.")

        amount = self._normalize_amount(exchange_name, symbol, amount)
        if price is not None:
            price = self._normalize_price(exchange_name, symbol, price)
            self._check_min_notional(exchange_name, symbol, amount, price)

        coid = client_order_id or self._new_client_order_id()
        params = {"clientOrderId": coid}

        logger.info(
            "Placing %s %s order: %s %s @ %s on %s (coid=%s)",
            order_type, side, amount, symbol,
            price if price else "market", exchange_name, coid,
        )

        order = self._execute_with_retry(
            exchange_name, f"create_{side}_order({symbol})",
            exchange.create_order, symbol, order_type, side, amount, price, params,
            idempotent=False,
        )

        self._cache.invalidate(f"balance:{exchange_name}")
        self._cache.invalidate(f"open_orders:{exchange_name}:{symbol}")
        self._cache.invalidate(f"open_orders:{exchange_name}:None")

        return {
            "id": order.get("id"),
            "clientOrderId": order.get("clientOrderId", coid),
            "symbol": order.get("symbol", symbol),
            "side": order.get("side", side),
            "type": order.get("type", order_type),
            "amount": order.get("amount", amount),
            "price": order.get("price", price),
            "cost": order.get("cost"),
            "filled": order.get("filled"),
            "remaining": order.get("remaining"),
            "status": order.get("status"),
            "timestamp": order.get("timestamp"),
            "exchange": exchange_name,
        }

    def place_stop_loss_order(
        self,
        exchange_name: str,
        symbol: str,
        amount: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place an exchange-native protective stop-loss (sell) order.

        This rests on the exchange, so the position stays protected even if the
        bot process is not running. Uses a STOP_LOSS_LIMIT order whose limit
        price defaults to just below the stop trigger so it still fills quickly.
        """
        amount = self._normalize_amount(exchange_name, symbol, amount)
        stop_price = self._normalize_price(exchange_name, symbol, stop_price)
        if limit_price is None:
            limit_price = stop_price * (1 - _STOP_LIMIT_OFFSET_PCT / 100.0)
        limit_price = self._normalize_price(exchange_name, symbol, limit_price)

        exchange = self._get_exchange(exchange_name)
        coid = client_order_id or self._new_client_order_id(prefix="sl")
        params = {"stopPrice": stop_price, "clientOrderId": coid}

        logger.info(
            "Placing protective stop-loss: sell %s %s stop=%s limit=%s on %s (coid=%s)",
            amount, symbol, stop_price, limit_price, exchange_name, coid,
        )

        order = self._execute_with_retry(
            exchange_name, f"stop_loss({symbol})",
            exchange.create_order, symbol, _STOP_ORDER_TYPE, "sell", amount,
            limit_price, params,
            idempotent=False,
        )

        self._cache.invalidate(f"open_orders:{exchange_name}:{symbol}")
        self._cache.invalidate(f"open_orders:{exchange_name}:None")

        return {
            "id": order.get("id"),
            "clientOrderId": order.get("clientOrderId", coid),
            "symbol": order.get("symbol", symbol),
            "side": order.get("side", "sell"),
            "type": order.get("type", _STOP_ORDER_TYPE),
            "amount": order.get("amount", amount),
            "stop_price": stop_price,
            "limit_price": limit_price,
            "status": order.get("status"),
            "exchange": exchange_name,
        }

    def cancel_stop_orders(
        self, exchange_name: str, symbol: str
    ) -> List[Dict[str, Any]]:
        """Cancel any resting stop orders for a symbol.

        Called before submitting a manual/strategy sell so a protective stop
        cannot fire on top of it and sell the same position twice.
        """
        exchange = self._get_exchange(exchange_name)
        try:
            raw_orders = self._execute_with_retry(
                exchange_name, "fetch_open_orders",
                exchange.fetch_open_orders, symbol,
            )
        except ExchangeError as exc:
            logger.error("Could not fetch open orders to cancel stops: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        for order in raw_orders:
            if not self._is_stop_order(order):
                continue
            try:
                self.cancel_order(exchange_name, order.get("id"), order.get("symbol", symbol))
                results.append({"id": order.get("id"), "status": "canceled"})
            except ExchangeError as exc:
                logger.error("Failed to cancel stop order %s: %s", order.get("id"), exc)
                results.append({
                    "id": order.get("id"), "status": "cancel_failed", "error": str(exc),
                })
        return results

    @staticmethod
    def _is_stop_order(order: Dict[str, Any]) -> bool:
        """Best-effort detection of a stop / stop-limit order across exchanges."""
        otype = str(order.get("type", "")).lower()
        if "stop" in otype:
            return True
        info = order.get("info", {})
        if isinstance(info, dict) and "stop" in str(info.get("type", "")).lower():
            return True
        return bool(order.get("stopPrice") or order.get("triggerPrice"))

    def cancel_order(
        self, exchange_name: str, order_id: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel an open order."""
        exchange = self._get_exchange(exchange_name)
        result = self._execute_with_retry(
            exchange_name, f"cancel_order({order_id})",
            exchange.cancel_order, order_id, symbol,
        )
        self._cache.invalidate(f"open_orders:{exchange_name}:{symbol}")
        self._cache.invalidate(f"open_orders:{exchange_name}:None")
        return {
            "id": result.get("id", order_id),
            "status": result.get("status", "canceled"),
            "exchange": exchange_name,
        }

    def cancel_all_orders(
        self, exchange_name: str, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Cancel all open orders for a symbol (or all symbols)."""
        open_orders = self.get_open_orders(exchange_name, symbol)
        results = []
        for order in open_orders:
            try:
                result = self.cancel_order(
                    exchange_name, order["id"], order.get("symbol")
                )
                results.append(result)
            except ExchangeError as exc:
                logger.error("Failed to cancel order %s: %s", order["id"], exc)
                results.append({"id": order["id"], "status": "cancel_failed", "error": str(exc)})
        return results

    def get_open_orders(
        self, exchange_name: str, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all open orders."""
        cache_key = f"open_orders:{exchange_name}:{symbol}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        exchange = self._get_exchange(exchange_name)
        raw_orders = self._execute_with_retry(
            exchange_name, "fetch_open_orders",
            exchange.fetch_open_orders, symbol,
        )
        orders = [
            {
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("type"),
                "amount": o.get("amount"),
                "price": o.get("price"),
                "filled": o.get("filled"),
                "remaining": o.get("remaining"),
                "status": o.get("status"),
                "timestamp": o.get("timestamp"),
            }
            for o in raw_orders
        ]
        self._cache.set(cache_key, orders, ttl=10.0)
        return orders

    def get_order_history(
        self,
        exchange_name: str,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch closed/completed order history."""
        exchange = self._get_exchange(exchange_name)

        try:
            raw_orders = self._execute_with_retry(
                exchange_name, "fetch_closed_orders",
                exchange.fetch_closed_orders, symbol, since, limit,
            )
        except (ExchangeError, ccxt.NotSupported):
            try:
                raw_orders = self._execute_with_retry(
                    exchange_name, "fetch_orders",
                    exchange.fetch_orders, symbol, since, limit,
                )
                raw_orders = [o for o in raw_orders if o.get("status") in ("closed", "canceled")]
            except (ExchangeError, ccxt.NotSupported):
                return []

        return [
            {
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("type"),
                "amount": o.get("amount"),
                "price": o.get("price"),
                "cost": o.get("cost"),
                "filled": o.get("filled"),
                "status": o.get("status"),
                "timestamp": o.get("timestamp"),
                "fee": o.get("fee"),
            }
            for o in raw_orders
        ]

    def get_order(self, exchange_name: str, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch a single order by ID."""
        exchange = self._get_exchange(exchange_name)
        order = self._execute_with_retry(
            exchange_name, f"fetch_order({order_id})",
            exchange.fetch_order, order_id, symbol,
        )
        return {
            "id": order.get("id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "type": order.get("type"),
            "amount": order.get("amount"),
            "price": order.get("price"),
            "cost": order.get("cost"),
            "filled": order.get("filled"),
            "remaining": order.get("remaining"),
            "status": order.get("status"),
            "timestamp": order.get("timestamp"),
            "fee": order.get("fee"),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_min_order_amount(self, exchange_name: str, symbol: str) -> Optional[float]:
        """Return the minimum order amount for a symbol, or None if unknown."""
        markets = self.get_markets(exchange_name)
        market = markets.get(symbol)
        if market is None:
            return None
        limits = market.get("limits", {}).get("amount", {})
        return limits.get("min")

    def get_price_precision(self, exchange_name: str, symbol: str) -> Optional[int]:
        """Return the price precision (decimal places) for a symbol."""
        markets = self.get_markets(exchange_name)
        market = markets.get(symbol)
        if market is None:
            return None
        return market.get("precision", {}).get("price")

    def get_amount_precision(self, exchange_name: str, symbol: str) -> Optional[int]:
        """Return the amount precision (decimal places) for a symbol."""
        markets = self.get_markets(exchange_name)
        market = markets.get(symbol)
        if market is None:
            return None
        return market.get("precision", {}).get("amount")
