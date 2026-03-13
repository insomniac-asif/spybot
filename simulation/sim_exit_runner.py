"""
simulation/sim_exit_runner.py
Exit loop for paper and live simulation trades.
Extracted from sim_engine.py to reduce file size.
"""
import asyncio
import logging
import time as _time
from datetime import datetime, time

import pytz

from simulation.sim_portfolio import SimPortfolio, get_sim_lock
from simulation.sim_executor import sim_try_fill
from simulation.sim_live_router import manage_live_exit
from execution.option_executor import get_option_price
from simulation.sim_ml import record_sim_trade_close
from core.md_state import is_md_enabled

# Private helpers extracted to keep this file under 500 lines
from simulation.sim_exit_helpers import (
    _trade_grade,
    _sim_close_record,
    _evaluate_exit_conditions,
)


# ---------------------------------------------------------------------------
# Module-level constants — kept in sync with sim_engine
# ---------------------------------------------------------------------------

DAYTRADE_EOD_CUTOFF = time(15, 55)  # ET cutoff for day-trading sims to flatten
EXPIRY_EOD_CUTOFF = time(15, 55)    # ET cutoff for same-day expiries

# Track consecutive get_option_price failures per trade
_price_fail_counts: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_profiles():
    import simulation.sim_engine as _eng
    return _eng._PROFILES


def _get_chain_ts():
    import simulation.sim_engine as _eng
    return _eng._LAST_CHAIN_CALL_TS


def _set_chain_ts(val):
    import simulation.sim_engine as _eng
    _eng._LAST_CHAIN_CALL_TS = val


def _get_chain_interval():
    import simulation.sim_engine as _eng
    return _eng._CHAIN_CALL_MIN_INTERVAL


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_sim_exits() -> list[dict]:
    _PROFILES = _get_profiles()
    results = []
    if not _PROFILES:
        return [{"sim_id": None, "status": "error", "reason": "no_profiles_loaded"}]
    eastern = pytz.timezone("US/Eastern")

    for sim_id, profile in _PROFILES.items():
        _sim_lock = get_sim_lock(sim_id)
        await _sim_lock.acquire()
        try:
            sim = SimPortfolio(sim_id, profile)
            await asyncio.to_thread(sim.load)
        except Exception as e:
            logging.exception("run_sim_exits_load_error: %s", e)
            results.append({
                "sim_id": sim_id,
                "status": "error",
                "reason": str(e)
            })
            _sim_lock.release()
            continue
        if profile.get("execution_mode") == "live" and profile.get("enabled"):
            for trade in list(sim.open_trades):
                try:
                    live_result = await manage_live_exit(sim, trade)
                    if live_result and live_result.get("status") == "success":
                        results.append({
                            "sim_id": sim_id,
                            "trade_id": trade.get("trade_id"),
                            "status": "closed",
                            "exit_price": live_result.get("exit_price"),
                            "exit_reason": live_result.get("exit_reason"),
                            "exit_context": live_result.get("exit_context"),
                            "option_symbol": live_result.get("option_symbol"),
                            "strike": live_result.get("strike") or trade.get("strike"),
                            "expiry": live_result.get("expiry") or trade.get("expiry"),
                            "direction": live_result.get("direction") or trade.get("direction"),
                            "qty": live_result.get("qty"),
                            "entry_price": live_result.get("entry_price"),
                            "pnl": live_result.get("pnl"),
                            "mode": "LIVE",
                            "balance_after": live_result.get("balance_after"),
                            "time_in_trade_seconds": live_result.get("time_in_trade_seconds"),
                            "predicted_direction": live_result.get("predicted_direction"),
                            "prediction_confidence": live_result.get("prediction_confidence"),
                            "edge_prob": live_result.get("edge_prob"),
                            "direction_prob": live_result.get("direction_prob"),
                        })
                except Exception as e:
                    logging.exception("run_sim_exits_live_error: %s", e)
                    results.append({
                        "sim_id": sim_id,
                        "trade_id": trade.get("trade_id"),
                        "status": "error",
                        "reason": str(e)
                    })
            _sim_lock.release()
            continue
        for trade in list(sim.open_trades):
            try:
                now_et = datetime.now(eastern)
                entry_time_et = datetime.fromisoformat(trade["entry_time"])
                if entry_time_et.tzinfo is None:
                    entry_time_et = eastern.localize(entry_time_et)
                elapsed_seconds = (now_et - entry_time_et).total_seconds()
                elapsed = _time.monotonic() - _get_chain_ts()
                if elapsed < _get_chain_interval():
                    await asyncio.sleep(_get_chain_interval() - elapsed)
                _set_chain_ts(_time.monotonic())
                current_price = await asyncio.to_thread(get_option_price, trade["option_symbol"])
                if current_price is None:
                    _tid = trade.get("trade_id", "unknown")
                    _price_fail_counts[_tid] = _price_fail_counts.get(_tid, 0) + 1
                    _fc = _price_fail_counts[_tid]
                    logging.error(
                        "sim_exit_missing_quote: sim=%s trade=%s symbol=%s consecutive_fails=%d",
                        sim_id, _tid, trade.get("option_symbol"), _fc,
                    )
                    if _fc >= 5:
                        logging.error(
                            "CRITICAL: %d consecutive price failures for %s/%s. "
                            "Exit protection disabled for this trade.",
                            _fc, sim_id, trade.get("option_symbol"),
                        )
                    continue
                # Reset counter on successful price fetch
                _price_fail_counts.pop(trade.get("trade_id", ""), None)
                sim.update_open_trade_excursion(trade["trade_id"], current_price)
                # Force exit for same-day expiry (all sims) before market close
                expiry_date = None
                should_exit = False
                exit_reason = None
                exit_context = None
                spread_guard_bypass = False
                try:
                    expiry_raw = trade.get("expiry")
                    if isinstance(expiry_raw, str):
                        expiry_date = datetime.fromisoformat(expiry_raw).date()
                except Exception:
                    expiry_date = None
                if expiry_date == now_et.date() and now_et.time() >= EXPIRY_EOD_CUTOFF:
                    should_exit = True
                    exit_reason = "expiry_close"
                    spread_guard_bypass = True
                    expiry_text = expiry_date.isoformat() if expiry_date else "unknown"
                    exit_context = f"expiry={expiry_text} cutoff={EXPIRY_EOD_CUTOFF.strftime('%H:%M')}"

                # Force exit for day-trading sims before market close
                is_daytrade = int(profile.get("dte_max", 0)) == 0
                if not should_exit and is_daytrade and now_et.time() >= DAYTRADE_EOD_CUTOFF:
                    should_exit = True
                    exit_reason = "eod_daytrade_close"
                    spread_guard_bypass = True
                    exit_context = f"daytrade_cutoff={DAYTRADE_EOD_CUTOFF.strftime('%H:%M')}"
                if not should_exit and elapsed_seconds < trade["hold_min_seconds"]:
                    continue

                if not should_exit:
                    should_exit, exit_reason, exit_context, spread_guard_bypass = _evaluate_exit_conditions(
                        trade, profile, sim, current_price, elapsed_seconds, now_et
                    )

                if should_exit and exit_reason:
                    bid = current_price * 0.99
                    ask = current_price * 1.01
                    fill_result, err = sim_try_fill(
                        trade["option_symbol"],
                        qty=trade["qty"],
                        bid=bid,
                        ask=ask,
                        profile=profile,
                        side="exit"
                    )
                    if err and not spread_guard_bypass:
                        continue
                    if err and spread_guard_bypass:
                        fill_result = None
                    exit_price = fill_result["fill_price"] if fill_result else current_price
                    exit_price_source = (
                        fill_result.get("price_source", "mid_minus_slippage")
                        if fill_result
                        else "market_raw_price"
                    )
                    exit_data = {
                        "exit_price": exit_price,
                        "exit_time": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
                        "exit_reason": exit_reason,
                        "exit_context": exit_context,
                        "exit_price_source": exit_price_source,
                        "exit_quote_model": "market_raw_price" if (spread_guard_bypass and fill_result is None) else "synthetic_1pct",
                        "entry_price_source": trade.get("entry_price_source", "mid_plus_slippage"),
                        "time_in_trade_seconds": int(elapsed_seconds),
                        "spread_guard_bypassed": spread_guard_bypass and fill_result is None,
                        "mae": trade.get("mae_pct"),
                        "mfe": trade.get("mfe_pct"),
                        "regime_at_entry": trade.get("regime_at_entry"),
                        "time_of_day_bucket": trade.get("time_of_day_bucket"),
                    }
                    entry_price = trade.get("entry_price")
                    qty_val = trade.get("qty")
                    pnl_val = None
                    try:
                        pnl_val = (exit_price - float(entry_price)) * float(qty_val) * 100
                    except (TypeError, ValueError):
                        pnl_val = None
                    _sim_close_record(sim, trade, exit_data, pnl_val)
                    results.append({
                        "sim_id": sim_id,
                        "symbol": trade.get("symbol"),
                        "trade_id": trade["trade_id"],
                        "status": "closed",
                        "exit_price": exit_price,
                        "exit_reason": exit_data["exit_reason"],
                        "exit_context": exit_data.get("exit_context"),
                        "option_symbol": trade.get("option_symbol"),
                        "strike": trade.get("strike"),
                        "expiry": trade.get("expiry"),
                        "direction": trade.get("direction"),
                        "qty": qty_val,
                        "entry_price": entry_price,
                        "pnl": pnl_val,
                        "mae": trade.get("mae_pct"),
                        "mfe": trade.get("mfe_pct"),
                        "feature_snapshot": trade.get("feature_snapshot"),
                        "mode": "SIM",
                        "balance_after": sim.balance,
                        "time_in_trade_seconds": exit_data.get("time_in_trade_seconds"),
                        "predicted_direction": trade.get("predicted_direction") or trade.get("direction"),
                        "prediction_confidence": trade.get("prediction_confidence"),
                        "edge_prob": trade.get("edge_prob"),
                        "direction_prob": trade.get("direction_prob"),
                    })
                    continue
                # Hold-max force exit (final fallback only)
                if elapsed_seconds >= trade["hold_max_seconds"]:
                    bid = current_price * 0.99
                    ask = current_price * 1.01
                    fill_result, err = sim_try_fill(
                        trade["option_symbol"],
                        qty=trade["qty"],
                        bid=bid,
                        ask=ask,
                        profile=profile,
                        side="exit"
                    )
                    if err:
                        fill_result = None
                    exit_price = fill_result["fill_price"] if fill_result else current_price
                    exit_price_source = (
                        fill_result.get("price_source", "mid_minus_slippage")
                        if fill_result
                        else "market_raw_price"
                    )
                    exit_context = f"elapsed={int(elapsed_seconds)}s >= hold_max={int(trade['hold_max_seconds'])}s"
                    exit_data = {
                        "exit_price": exit_price,
                        "exit_time": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
                        "exit_reason": "hold_max_elapsed",
                        "exit_context": exit_context,
                        "exit_price_source": exit_price_source,
                        "exit_quote_model": "market_raw_price" if fill_result is None else "synthetic_1pct",
                        "entry_price_source": trade.get("entry_price_source", "mid_plus_slippage"),
                        "time_in_trade_seconds": int(elapsed_seconds),
                        "spread_guard_bypassed": fill_result is None,
                        "mae": trade.get("mae_pct"),
                        "mfe": trade.get("mfe_pct"),
                        "regime_at_entry": trade.get("regime_at_entry"),
                        "time_of_day_bucket": trade.get("time_of_day_bucket"),
                    }
                    entry_price = trade.get("entry_price")
                    qty_val = trade.get("qty")
                    pnl_val = None
                    try:
                        pnl_val = (exit_price - float(entry_price)) * float(qty_val) * 100
                    except (TypeError, ValueError):
                        pnl_val = None
                    _sim_close_record(sim, trade, exit_data, pnl_val)
                    results.append({
                        "sim_id": sim_id,
                        "trade_id": trade["trade_id"],
                        "status": "closed",
                        "exit_price": exit_price,
                        "exit_reason": "hold_max_elapsed",
                        "exit_context": exit_context,
                        "option_symbol": trade.get("option_symbol"),
                        "strike": trade.get("strike"),
                        "expiry": trade.get("expiry"),
                        "direction": trade.get("direction"),
                        "qty": qty_val,
                        "entry_price": entry_price,
                        "pnl": pnl_val,
                        "mae": trade.get("mae_pct"),
                        "mfe": trade.get("mfe_pct"),
                        "feature_snapshot": trade.get("feature_snapshot"),
                        "mode": "SIM",
                        "balance_after": sim.balance,
                        "time_in_trade_seconds": exit_data.get("time_in_trade_seconds"),
                        "predicted_direction": trade.get("predicted_direction") or trade.get("direction"),
                        "prediction_confidence": trade.get("prediction_confidence"),
                        "edge_prob": trade.get("edge_prob"),
                        "direction_prob": trade.get("direction_prob"),
                    })
            except Exception as e:
                logging.exception("run_sim_exits_error: %s", e)
                results.append({
                    "sim_id": sim_id,
                    "trade_id": trade.get("trade_id"),
                    "status": "error",
                    "reason": str(e)
                })
        _sim_lock.release()
    return results
