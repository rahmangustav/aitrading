"""
Strategy Engine -- manages the lifecycle of trading strategies.

Handles starting, stopping, listing, and status reporting for all
registered strategies. Each strategy runs as a managed object with
its own configuration and state.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import yaml

logger = logging.getLogger("crypto-trader.engine")

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = _PROJECT_ROOT / "data"
_DEFAULT_STATE_PATH = Path.home() / ".openclaw" / ".crypto-trader-strategies.json"


class BaseStrategy:
    """Base class for all trading strategies.

    Subclasses must implement:
    - evaluate(): check conditions and optionally return trade signals
    - on_start(): called when strategy is activated
    - on_stop(): called when strategy is deactivated
    """

    name: str = "base"
    display_name: str = "Base Strategy"

    # Attribute names persisted across process restarts (per subclass).
    _persist_attrs: tuple = ()

    def __init__(
        self,
        strategy_id: str,
        params: Dict[str, Any],
        exchange_manager: Any,
        risk_manager: Any,
    ) -> None:
        self.strategy_id = strategy_id
        self.params = params
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        self.active = False
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_run: Optional[str] = None
        self.stats: Dict[str, Any] = {
            "trades_executed": 0,
            "total_pnl": 0.0,
            "signals_generated": 0,
        }

    def on_start(self) -> None:
        """Called when the strategy is activated."""
        self.active = True
        logger.info("Strategy %s (%s) started.", self.display_name, self.strategy_id)

    def on_stop(self) -> None:
        """Called when the strategy is deactivated."""
        self.active = False
        logger.info("Strategy %s (%s) stopped.", self.display_name, self.strategy_id)

    def evaluate(self) -> List[Dict[str, Any]]:
        """Evaluate market conditions and return trade signals.

        Returns a list of signal dicts:
        [{"symbol": ..., "side": "buy"/"sell", "amount": ..., "price": ..., "reason": ...}]
        """
        raise NotImplementedError

    def on_order_placed(self, signal: Dict[str, Any], order: Dict[str, Any]) -> None:
        """Called after an order for one of this strategy's signals is placed."""

    def on_order_filled(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Called when an order fills. May return a follow-up signal dict."""
        return None

    def get_state(self) -> Dict[str, Any]:
        """Return strategy-specific mutable state to persist across restarts."""
        state: Dict[str, Any] = {}
        if hasattr(self, "exchange"):
            state["exchange"] = getattr(self, "exchange")
        for key in self._persist_attrs:
            if hasattr(self, key):
                state[key] = getattr(self, key)
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state produced by get_state()."""
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize strategy state for persistence."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "display_name": self.display_name,
            "params": self.params,
            "active": self.active,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "stats": self.stats,
        }


class StrategyEngine:
    """Manages the lifecycle of trading strategies."""

    def __init__(self, exchange_manager: Any, risk_manager: Any) -> None:
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        self._strategies: Dict[str, BaseStrategy] = {}
        self._registry: Dict[str, Type[BaseStrategy]] = {}
        self._config = self._load_config()
        self._lock = threading.Lock()
        # Resolved per instance so the env var set after import is honored.
        self._state_path = Path(os.environ.get(
            "CRYPTO_STRATEGY_STATE_PATH", str(_DEFAULT_STATE_PATH),
        ))

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        path = _CONFIG_DIR / "strategies.yaml"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register_strategy(self, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class for use."""
        self._registry[strategy_class.name] = strategy_class
        logger.info("Registered strategy: %s", strategy_class.name)

    def get_available_strategies(self) -> List[str]:
        """Return names of all registered strategies."""
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_strategy(
        self,
        strategy_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create and start a new strategy instance."""
        with self._lock:
            strategy_class = self._registry.get(strategy_name)
            if strategy_class is None:
                available = ", ".join(self._registry.keys()) or "none"
                return {
                    "status": "error",
                    "message": f"Unknown strategy '{strategy_name}'. Available: {available}",
                }

            strat_config = self._config.get(strategy_name, {})
            if not strat_config.get("enabled", True):
                return {
                    "status": "error",
                    "message": f"Strategy '{strategy_name}' is disabled in config.",
                }

            merged_params = {**strat_config.get("default_params", {})}
            if params:
                merged_params.update(params)

            strategy_id = f"{strategy_name}_{uuid.uuid4().hex[:8]}"

            strategy = strategy_class(
                strategy_id=strategy_id,
                params=merged_params,
                exchange_manager=self.exchange_manager,
                risk_manager=self.risk_manager,
            )
            strategy.on_start()
            self._strategies[strategy_id] = strategy
            self._save_state()

            return {
                "status": "ok",
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "params": merged_params,
                "message": f"Strategy '{strategy_name}' started with ID {strategy_id}.",
            }

    def stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Stop a running strategy."""
        with self._lock:
            strategy = self._strategies.get(strategy_id)
            if strategy is None:
                return {
                    "status": "error",
                    "message": f"Strategy '{strategy_id}' not found.",
                }
            strategy.on_stop()
            del self._strategies[strategy_id]
            self._save_state()
            return {
                "status": "ok",
                "strategy_id": strategy_id,
                "message": f"Strategy '{strategy_id}' stopped.",
            }

    def stop_all(self) -> List[Dict[str, Any]]:
        """Stop all running strategies."""
        results = []
        with self._lock:
            for sid in list(self._strategies.keys()):
                self._strategies[sid].on_stop()
                results.append({"strategy_id": sid, "status": "stopped"})
            self._strategies.clear()
            self._save_state()
        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self) -> List[Dict[str, Any]]:
        """Run evaluate() on all active strategies and collect signals."""
        all_signals = []
        for sid, strategy in list(self._strategies.items()):
            if not strategy.active:
                continue
            try:
                signals = strategy.evaluate()
                strategy.last_run = datetime.now(timezone.utc).isoformat()
                strategy.stats["signals_generated"] += len(signals)
                for signal in signals:
                    signal["strategy_id"] = sid
                    signal["strategy_name"] = strategy.name
                all_signals.extend(signals)
            except Exception as exc:
                logger.error("Strategy %s evaluate error: %s", sid, exc)
        return all_signals

    # ------------------------------------------------------------------
    # Status / Listing
    # ------------------------------------------------------------------

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Return status of all strategy instances."""
        return [s.to_dict() for s in self._strategies.values()]

    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Return status of a specific strategy."""
        strategy = self._strategies.get(strategy_id)
        return strategy.to_dict() if strategy else None

    def get_strategy_instance(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Return the live strategy object (for fill callbacks etc.)."""
        return self._strategies.get(strategy_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Save current strategy state to disk."""
        state = {
            "strategies": {
                sid: {**s.to_dict(), "state": s.get_state()}
                for sid, s in self._strategies.items()
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)

    def save_state(self) -> None:
        """Public alias so callers (e.g. the monitor daemon) can persist."""
        with self._lock:
            self._save_state()

    def _read_state_file(self) -> Optional[Dict[str, Any]]:
        if not self._state_path.exists():
            return None
        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt strategy state file at %s, ignoring.", self._state_path)
            return None

    def load_state(self) -> int:
        """Restore strategies persisted by previous processes.

        Must be called after all strategy classes are registered. Existing
        in-memory instances are kept as-is. Returns the number of strategies
        restored.
        """
        data = self._read_state_file()
        if not data:
            return 0

        restored = 0
        with self._lock:
            for sid, saved in data.get("strategies", {}).items():
                if sid in self._strategies:
                    continue
                strategy_class = self._registry.get(saved.get("name", ""))
                if strategy_class is None:
                    logger.warning(
                        "Cannot restore strategy %s: class '%s' not registered.",
                        sid, saved.get("name"),
                    )
                    continue
                strategy = strategy_class(
                    strategy_id=sid,
                    params=saved.get("params", {}),
                    exchange_manager=self.exchange_manager,
                    risk_manager=self.risk_manager,
                )
                strategy.active = saved.get("active", False)
                strategy.created_at = saved.get("created_at", strategy.created_at)
                strategy.last_run = saved.get("last_run")
                strategy.stats = saved.get("stats", strategy.stats)
                strategy.restore_state(saved.get("state", {}))
                self._strategies[sid] = strategy
                restored += 1

        if restored:
            logger.info("Restored %d strategy instance(s) from disk.", restored)
        return restored

    def sync_from_disk(self) -> None:
        """Reconcile in-memory strategies with the state file.

        Strategies stopped by another process are removed; strategies started
        by another process are added. In-memory runtime state of strategies
        that still exist is left untouched.
        """
        data = self._read_state_file()
        saved_ids = set((data or {}).get("strategies", {}).keys())

        with self._lock:
            for sid in list(self._strategies.keys()):
                if sid not in saved_ids:
                    logger.info("Strategy %s removed by another process.", sid)
                    self._strategies[sid].active = False
                    del self._strategies[sid]

        if data:
            self.load_state()
