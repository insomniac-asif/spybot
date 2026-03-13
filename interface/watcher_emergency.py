# interface/watcher_emergency.py
#
# Emergency exit loop logic extracted from watchers.py.
# Contains: _run_emergency_exit_check

import asyncio
import logging
import os
from datetime import datetime

import pytz

# Trade IDs currently being closed by the emergency exit path.
# Checked by the normal exit runner to avoid double-close.
_emergency_closing: set[str] = set()


def is_emergency_closing(trade_id: str) -> bool:
    """Return True if this trade is currently being emergency-closed."""
    return trade_id in _emergency_closing


async def _run_emergency_exit_check() -> None:
    """
    Check SIM00 live positions for emergency exit conditions.

    Force-close conditions (any one triggers an immediate market close):
      1. PANIC_LOCKDOWN state → close ALL SIM00 live positions
      2. Same-day expiry past 15:50 ET (tighter than normal 15:55 cutoff)
      3. Option price dropped > 60% from entry (flash crash / data issue)
    """
    from core.market_clock import market_is_open
    if not market_is_open():
        return

    try:
        from core.singletons import RUNTIME, SystemState
        _state = RUNTIME.state
    except ImportError:
        return

    # ── Load SIM00 open trades (under sim lock) ────────────────
    from simulation.sim_portfolio import get_sim_lock
    _sim_lock = get_sim_lock("SIM00")
    await _sim_lock.acquire()
    try:
        try:
            import yaml as _yaml
            _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _cfg_path = os.path.join(_base, "simulation", "sim_config.yaml")
            def _load_sim00_config():
                with open(_cfg_path) as _f:
                    return _yaml.safe_load(_f) or {}
            _profiles = await asyncio.to_thread(_load_sim00_config)
            _sim00_profile = _profiles.get("SIM00", {})
            if not isinstance(_sim00_profile, dict):
                return
            if _sim00_profile.get("execution_mode") != "live":
                return
            from simulation.sim_portfolio import SimPortfolio
            _sim = SimPortfolio("SIM00", _sim00_profile)
            await asyncio.to_thread(_sim.load)
        except Exception:
            return

        if not _sim.open_trades:
            return

        eastern = pytz.timezone("America/New_York")
        now_et = datetime.now(eastern)
        _EOD_EMERGENCY = now_et.replace(hour=15, minute=50, second=0, microsecond=0).time()

        for _trade in list(_sim.open_trades):
            if not isinstance(_trade, dict):
                continue
            _option_sym = _trade.get("option_symbol")
            if not _option_sym:
                continue
            _trade_id = _trade.get("trade_id", "")

            _force = False
            _reason = ""

            # 1. Panic lockdown — force close everything
            if _state == SystemState.PANIC_LOCKDOWN:
                _force = True
                _reason = "panic_lockdown"

            # 2. Same-day expiry emergency cutoff (15:50)
            if not _force:
                try:
                    _expiry_raw = _trade.get("expiry")
                    if isinstance(_expiry_raw, str):
                        from datetime import datetime as _dt
                        _exp_date = _dt.fromisoformat(_expiry_raw).date()
                        if _exp_date == now_et.date() and now_et.time() >= _EOD_EMERGENCY:
                            _force = True
                            _reason = "emergency_expiry_cutoff"
                except Exception:
                    pass

            # 3. Price dropped > 60% from entry
            if not _force:
                try:
                    from execution.option_executor import get_option_price
                    _entry = float(_trade.get("entry_price", 0))
                    if _entry > 0:
                        _cur = get_option_price(_option_sym)
                        if _cur is not None and (_cur - _entry) / _entry <= -0.60:
                            _force = True
                            _reason = f"emergency_60pct_loss entry={_entry:.4f} cur={_cur:.4f}"
                except Exception:
                    pass

            if not _force:
                continue

            try:
                from execution.option_executor import close_option_position
                _qty = int(_trade.get("qty", 0))
                if _qty <= 0:
                    continue
                _emergency_closing.add(_trade_id)
                _close = await asyncio.to_thread(close_option_position, _option_sym, _qty)
                _filled = _close.get("filled_avg_price") if _close.get("ok") else None
                logging.error(
                    "emergency_exit_loop_closed: sim=SIM00 symbol=%s qty=%d reason=%s filled=%s",
                    _option_sym, _qty, _reason, _filled,
                )
                _exit_price = _filled if _filled is not None else _trade.get("entry_price", 0)
                _sim.record_close(_trade["trade_id"], {
                    "exit_price": _exit_price,
                    "exit_time": now_et.isoformat(),
                    "exit_reason": _reason,
                    "exit_price_source": "broker_fill" if _filled else "estimated_mid",
                    "exit_quote_model": "emergency",
                    "entry_price_source": _trade.get("entry_price_source", "live_fill"),
                    "time_in_trade_seconds": 0,
                    "spread_guard_bypassed": True,
                })
                _sim.save()
            except Exception as _ce:
                logging.error("emergency_exit_loop_close_failed: %s reason=%s", _ce, _reason)
            finally:
                _emergency_closing.discard(_trade_id)
    finally:
        _sim_lock.release()
