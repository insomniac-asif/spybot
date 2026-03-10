"""
core/freshness_monitor.py — Auto-transitions RuntimeState on stale data.

Escalation:
  bars > 300s  → DEGRADED (entries blocked, exits OK)
  bars > 600s  → EXIT_ONLY + alert
  broker > 300s → PANIC_LOCKDOWN + alert

Recovery:
  bars < 150s AND state is DEGRADED → TRADING_ENABLED
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

_BAR_DEGRADE_SEC   = 300   # 5 min — genuine outage, not just a slow bar
_BAR_EXIT_ONLY_SEC = 600   # 10 min — extended outage
_BROKER_PANIC_SEC  = 300
_BAR_RECOVER_SEC   = 150   # recover once data is < 2.5 min old


class FreshnessMonitor:
    """
    Background task. Checks data freshness every 10s and triggers state
    transitions. Register via setup_hook() using the same safe_task pattern.
    """

    def __init__(self, runtime, risk_supervisor) -> None:
        self.runtime = runtime
        self.supervisor = risk_supervisor
        self._alerted: dict[str, bool] = {}

    async def run(self, send_alert_fn) -> None:
        """
        send_alert_fn: async callable(str) — posts to the alert Discord channel.
        """
        from core.runtime_state import SystemState
        while True:
            try:
                await asyncio.sleep(10)
                from core.market_clock import market_is_open
                if not market_is_open():
                    continue

                status = self.supervisor.get_status()
                bar_age    = status.get("bar_age_seconds")    or float("inf")
                broker_age = status.get("broker_age_seconds") or float("inf")

                current = self.runtime.state

                # ── Broker panic ──────────────────────────────────────────
                if (broker_age > _BROKER_PANIC_SEC
                        and current != SystemState.PANIC_LOCKDOWN
                        and current != SystemState.BOOTING
                        and current != SystemState.RECONCILING):
                    self.runtime.force_transition(
                        SystemState.PANIC_LOCKDOWN,
                        f"broker unreachable {broker_age:.0f}s",
                    )
                    if not self._alerted.get("panic"):
                        try:
                            await send_alert_fn(
                                f"⛔ **PANIC LOCKDOWN** — broker unreachable for {broker_age:.0f}s\n"
                                "All live entries blocked. Exits still allowed."
                            )
                        except Exception:
                            pass
                        self._alerted["panic"] = True
                    continue

                # ── Clear panic alert flag when no longer in PANIC_LOCKDOWN ─
                if current != SystemState.PANIC_LOCKDOWN and self._alerted.get("panic"):
                    self._alerted.pop("panic", None)

                # ── Bar EXIT_ONLY ─────────────────────────────────────────
                if bar_age > _BAR_EXIT_ONLY_SEC and current == SystemState.DEGRADED:
                    if self.runtime.transition(SystemState.EXIT_ONLY, f"bars {bar_age:.0f}s stale"):
                        if not self._alerted.get("exit_only"):
                            try:
                                await send_alert_fn(
                                    f"🔴 **EXIT ONLY** — market data stale for {bar_age:.0f}s\n"
                                    "Managing existing positions only. No new entries."
                                )
                            except Exception:
                                pass
                            self._alerted["exit_only"] = True
                    continue

                # ── Bar DEGRADED ──────────────────────────────────────────
                if bar_age > _BAR_DEGRADE_SEC and current == SystemState.TRADING_ENABLED:
                    self.runtime.add_degradation("stale_bars")
                    if self.runtime.transition(SystemState.DEGRADED, f"bars {bar_age:.0f}s stale"):
                        if not self._alerted.get("degraded"):
                            try:
                                await send_alert_fn(
                                    f"⚠️ **DEGRADED** — market data stale for {bar_age:.0f}s\n"
                                    "New entries blocked; exits still allowed."
                                )
                            except Exception:
                                pass
                            self._alerted["degraded"] = True
                    continue

                # ── READY → TRADING_ENABLED on market open + fresh data ──
                if bar_age < _BAR_RECOVER_SEC and current == SystemState.READY:
                    if self.runtime.transition(SystemState.TRADING_ENABLED, "market open, data fresh"):
                        self._alerted.clear()
                        try:
                            await send_alert_fn("✅ **TRADING ENABLED** — market open, data fresh.")
                        except Exception:
                            pass
                    continue

                # ── Recovery from DEGRADED ────────────────────────────────
                if bar_age < _BAR_RECOVER_SEC and current == SystemState.DEGRADED:
                    self.runtime.clear_degradation("stale_bars")
                    if not self.runtime.degradation_reasons():
                        if self.runtime.transition(SystemState.TRADING_ENABLED, "freshness_recovered"):
                            self._alerted.clear()
                            try:
                                await send_alert_fn("✅ **TRADING ENABLED** — data fresh, trading re-enabled.")
                            except Exception:
                                pass

            except Exception as e:
                logger.error("freshness_monitor_error: %s", e, exc_info=True)
                await asyncio.sleep(30)
