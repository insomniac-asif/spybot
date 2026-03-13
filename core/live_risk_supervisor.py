"""
core/live_risk_supervisor.py — Single authority for all live order placement.

Nothing places a live order without passing authorize_entry().
Infrastructure-layer checks only (freshness, kill switch, runtime state).
Portfolio-level checks (daily loss, max open, etc.) remain in SimPortfolio.can_trade().
"""

import logging
import threading
import time
from typing import Tuple

logger = logging.getLogger(__name__)

# Freshness thresholds — override via .env if needed
import os
_BAR_STALE_SEC     = float(os.getenv("RISK_BAR_STALE_SEC",     "90"))
_QUOTE_STALE_SEC   = float(os.getenv("RISK_QUOTE_STALE_SEC",   "30"))
_ACCOUNT_STALE_SEC = float(os.getenv("RISK_ACCOUNT_STALE_SEC", "120"))
_BROKER_STALE_SEC  = float(os.getenv("RISK_BROKER_STALE_SEC",  "300"))


class LiveRiskSupervisor:
    """
    Final gate for any live order.
    Singleton is created in core/singletons.py and imported from there.
    """

    def __init__(self, runtime_state) -> None:
        self.runtime = runtime_state
        self._lock = threading.Lock()
        self._emergency_kill: bool = False
        self._kill_reason: str = ""
        self._last_bar_time: float = 0.0
        self._last_quote_time: float = 0.0
        self._last_account_time: float = 0.0
        self._last_broker_success: float = 0.0

    # ── Freshness stamps ───────────────────────────────────────────────────

    def update_bar_freshness(self, ts: float) -> None:
        with self._lock:
            self._last_bar_time = ts

    def update_quote_freshness(self, ts: float) -> None:
        with self._lock:
            self._last_quote_time = ts

    def update_account_freshness(self, ts: float) -> None:
        with self._lock:
            self._last_account_time = ts

    def update_broker_health(self, ts: float) -> None:
        with self._lock:
            self._last_broker_success = ts

    # ── Kill switch ────────────────────────────────────────────────────────

    def emergency_kill(self, reason: str) -> None:
        with self._lock:
            self._emergency_kill = True
            self._kill_reason = reason
        logger.critical("EMERGENCY_KILL: %s", reason)

    def clear_kill(self) -> None:
        with self._lock:
            self._emergency_kill = False
            self._kill_reason = ""
        logger.info("kill_switch_cleared")

    @property
    def is_killed(self) -> bool:
        return self._emergency_kill

    # ── Authorization ──────────────────────────────────────────────────────

    def authorize_entry(
        self,
        sim_id: str,
        direction: str,
        symbol: str,
        notional: float,
    ) -> Tuple[bool, str]:
        """
        The ONLY function that can greenlight a live entry order.
        Fail-fast — first failing check returns immediately.
        """
        now = time.time()

        with self._lock:
            if self._emergency_kill:
                return False, f"emergency_kill: {self._kill_reason}"

            bar_age = now - self._last_bar_time if self._last_bar_time > 0 else float("inf")
            quote_age = now - self._last_quote_time if self._last_quote_time > 0 else float("inf")
            acct_age = now - self._last_account_time if self._last_account_time > 0 else float("inf")
            broker_age = now - self._last_broker_success if self._last_broker_success > 0 else float("inf")

        if not self.runtime.can_enter_trades():
            return False, f"runtime_state={self.runtime.state.value}: {self.runtime.reason}"

        if bar_age > _BAR_STALE_SEC:
            return False, f"stale_bars: {bar_age:.0f}s (limit {_BAR_STALE_SEC:.0f}s)"
        if quote_age > _QUOTE_STALE_SEC:
            return False, f"stale_quotes: {quote_age:.0f}s (limit {_QUOTE_STALE_SEC:.0f}s)"
        if acct_age > _ACCOUNT_STALE_SEC:
            return False, f"stale_account: {acct_age:.0f}s (limit {_ACCOUNT_STALE_SEC:.0f}s)"
        if broker_age > _BROKER_STALE_SEC:
            return False, f"broker_unreachable: {broker_age:.0f}s (limit {_BROKER_STALE_SEC:.0f}s)"

        return True, "authorized"

    def authorize_exit(self, sim_id: str) -> Tuple[bool, str]:
        """PANIC_LOCKDOWN still allows exits (to close positions)."""
        if not self.runtime.can_exit_trades():
            return False, f"runtime_state={self.runtime.state.value}"
        return True, "authorized"

    # ── Status ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        now = time.time()

        def age(ts: float):
            return round(now - ts, 1) if ts > 0 else None

        with self._lock:
            return {
                "emergency_kill":      self._emergency_kill,
                "kill_reason":         self._kill_reason,
                "bar_age_seconds":     age(self._last_bar_time),
                "quote_age_seconds":   age(self._last_quote_time),
                "account_age_seconds": age(self._last_account_time),
                "broker_age_seconds":  age(self._last_broker_success),
            }
