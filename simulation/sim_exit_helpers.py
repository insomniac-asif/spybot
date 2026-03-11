"""
simulation/sim_exit_helpers.py
Private helper functions for sim_exit_runner.py.
Extracted to keep sim_exit_runner.py under 500 lines.
"""
import logging
from datetime import datetime

import pytz

from simulation.sim_ml import record_sim_trade_close


def _trade_grade(tr) -> "float | None":
    """Best available confidence/edge score for a trade."""
    candidates = []
    for key in ("edge_prob", "prediction_confidence", "confidence", "ml_probability"):
        val = tr.get(key)
        if isinstance(val, (int, float)):
            candidates.append(float(val))
    return max(candidates) if candidates else None


def _sim_close_record(sim, trade, exit_data: dict, pnl_val):
    """Persist a paper trade close: portfolio record + ML close + strategy perf store."""
    sim.record_close(trade["trade_id"], exit_data)
    sim.save()
    record_sim_trade_close(trade, pnl_val)
    try:
        from analytics.strategy_performance import PERF_STORE
        entry_price = trade.get("entry_price")
        qty_val = trade.get("qty")
        _pnl_d = float(pnl_val or 0)
        _cost = float(entry_price or 0) * float(qty_val or 0) * 100
        _pnl_pct = _pnl_d / _cost if _cost > 0 else 0.0
        PERF_STORE.record_close(
            strategy=trade.get("signal_mode", ""),
            regime=trade.get("regime_at_entry", "UNKNOWN"),
            time_bucket=trade.get("time_of_day_bucket", "UNKNOWN"),
            pnl=_pnl_d,
            pnl_pct=_pnl_pct,
            hold_seconds=float(exit_data.get("time_in_trade_seconds") or 0),
            grade=trade.get("grade"),
            spread_pct=trade.get("spread_pct"),
        )
    except Exception:
        pass


def _evaluate_exit_conditions(trade, profile, sim, current_price, elapsed_seconds, now_et):
    """Evaluate all exit conditions for a paper trade.

    Returns (should_exit, exit_reason, exit_context, spread_guard_bypass).
    Does NOT perform the actual exit — only decides whether to exit and why.
    May mutate trade dict (peak_price, trailing_stop_high, tp2_activated, etc.)
    and call sim.save() when intermediate trade state is persisted.
    """
    should_exit = False
    exit_reason = None
    exit_context = None
    spread_guard_bypass = False

    from core.md_state import is_md_enabled

    stop_loss_pct = trade.get("stop_loss_pct", profile.get("stop_loss_pct"))
    if stop_loss_pct is not None and is_md_enabled() and trade.get("sim_id") != "SIM09":
        try:
            stop_loss_pct = max(float(stop_loss_pct) * 0.7, 0.05)
        except (TypeError, ValueError):
            pass
    if stop_loss_pct is not None:
        try:
            entry_price = float(trade.get("entry_price", 0))
            if entry_price > 0:
                peak_price = float(trade.get("peak_price") or 0)
                if current_price > entry_price and current_price > peak_price:
                    trade["peak_price"] = current_price
                    sim.save()
                    peak_price = current_price

                loss_pct = (current_price - entry_price) / entry_price
                effective_sl_pct = abs(float(stop_loss_pct))
                original_sl_price = entry_price * (1 - effective_sl_pct)
                effective_sl_price = max(original_sl_price, entry_price) if peak_price > entry_price else original_sl_price

                if current_price <= effective_sl_price:
                    should_exit = True
                    spread_guard_bypass = True
                    if effective_sl_price > original_sl_price:
                        exit_reason = "breakeven_stop"
                        exit_context = f"price={current_price:.4f} locked_floor={effective_sl_price:.4f} peak={peak_price:.4f}"
                    else:
                        exit_reason = "stop_loss"
                        exit_context = f"loss_pct={loss_pct:.3%} <= -{effective_sl_pct:.3%}"
        except (TypeError, ValueError):
            pass

    profit_target_pct = profile.get("profit_target_pct")
    entry_price = None
    gain_pct = None
    try:
        entry_price = float(trade.get("entry_price", 0))
        if entry_price > 0:
            gain_pct = (current_price - entry_price) / entry_price
    except (TypeError, ValueError):
        gain_pct = None

    # Greeks-aware exit: theta burn acceleration
    if not should_exit and gain_pct is not None and gain_pct <= 0.02:
        dte_at_entry = trade.get("dte_bucket")
        try:
            dte_val = int(dte_at_entry) if dte_at_entry is not None else None
        except (TypeError, ValueError):
            dte_val = None
        if dte_val is not None and dte_val <= 1:
            try:
                expiry_raw = trade.get("expiry")
                if isinstance(expiry_raw, str):
                    expiry_date = datetime.fromisoformat(expiry_raw).date()
                    if expiry_date == now_et.date():
                        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                        remaining_seconds = (market_close - now_et).total_seconds()
                        if 0 < remaining_seconds < 7200 and gain_pct <= 0.02:
                            should_exit = True
                            exit_reason = "theta_burn"
                            exit_context = f"gain_pct={gain_pct:.3%} remaining={remaining_seconds:.0f}s dte={dte_val}"
                            spread_guard_bypass = True
            except Exception:
                pass

    # Greeks-aware exit: IV crush detection
    if not should_exit and gain_pct is not None and gain_pct < 0:
        iv_at_entry = trade.get("iv_at_entry")
        if isinstance(iv_at_entry, (int, float)) and float(iv_at_entry) > 0:
            try:
                iv_entry = float(iv_at_entry)
                stop_pct = float(profile.get("stop_loss_pct", 0.30))
                if abs(gain_pct) > stop_pct * 0.3:
                    tightened_stop = stop_pct * 0.6
                    if gain_pct <= -abs(tightened_stop):
                        should_exit = True
                        exit_reason = "iv_crush_stop"
                        exit_context = f"gain_pct={gain_pct:.3%} iv_entry={iv_entry:.3f} tightened_stop={tightened_stop:.3%}"
                        spread_guard_bypass = True
            except (TypeError, ValueError):
                pass

    # Near-TP adaptive lock + TP2 (optional)
    if not should_exit and profit_target_pct is not None and gain_pct is not None:
        try:
            base_target = abs(float(profit_target_pct))
            if base_target > 0 and profile.get("tp2_enabled", True):
                near_ratio = float(profile.get("near_tp_trigger_ratio", 0.85))
                grade_min = float(profile.get("near_tp_grade_min", 0.6))
                tp2_mult = float(profile.get("tp2_multiplier", 1.3))
                grade = _trade_grade(trade)
                if grade is not None and grade >= grade_min and gain_pct >= base_target * near_ratio:
                    if not trade.get("tp2_activated"):
                        lock_pct = trade.get("lock_profit_pct")
                        if lock_pct is None:
                            lock_pct = max(0.05, min(base_target * 0.5, base_target - 0.02))
                        trade["lock_profit_pct"] = float(lock_pct)
                        trade["tp2_target_pct"] = float(base_target * tp2_mult)
                        trade["tp2_activated"] = True
                        sim.save()
        except (TypeError, ValueError):
            pass

    # Profit lock: exit if retrace below locked profit level after TP2 activation
    lock_pct = trade.get("lock_profit_pct")
    if not should_exit and gain_pct is not None and trade.get("tp2_activated") and isinstance(lock_pct, (int, float)):
        try:
            if gain_pct <= float(lock_pct):
                should_exit = True
                exit_reason = "profit_lock"
                exit_context = f"gain_pct={gain_pct:.3%} <= lock_pct={float(lock_pct):.3%}"
        except (TypeError, ValueError):
            pass

    effective_target = profit_target_pct
    if trade.get("tp2_activated") and trade.get("tp2_target_pct") is not None:
        effective_target = trade.get("tp2_target_pct")

    if not should_exit and effective_target is not None:
        try:
            if entry_price is None:
                entry_price = float(trade.get("entry_price", 0))
            if entry_price > 0:
                if gain_pct is None:
                    gain_pct = (current_price - entry_price) / entry_price
                target_val = abs(float(effective_target))
                if gain_pct >= target_val:
                    should_exit = True
                    exit_reason = "profit_target_2" if trade.get("tp2_activated") else "profit_target"
                    exit_context = f"gain_pct={gain_pct:.3%} >= {target_val:.3%}"
        except (TypeError, ValueError):
            pass

    trailing_activate = profile.get("trailing_stop_activate_pct")
    trailing_trail = profile.get("trailing_stop_trail_pct")
    if not should_exit and trailing_activate is not None and trailing_trail is not None:
        try:
            entry_price = float(trade.get("entry_price", 0))
            if entry_price > 0:
                gain_pct = (current_price - entry_price) / entry_price
                trailing_activate_f = abs(float(trailing_activate))
                trailing_trail_f = abs(float(trailing_trail))
                if not trade.get("trailing_stop_activated", False):
                    if gain_pct >= trailing_activate_f:
                        trade["trailing_stop_activated"] = True
                        trade["trailing_stop_high"] = current_price
                        sim.save()
                else:
                    if current_price > trade.get("trailing_stop_high", 0):
                        trade["trailing_stop_high"] = current_price
                        sim.save()
                    trail_high = float(trade.get("trailing_stop_high", 0))
                    if trail_high > 0:
                        drop_from_high = (current_price - trail_high) / trail_high
                        if drop_from_high <= -trailing_trail_f:
                            should_exit = True
                            exit_reason = "trailing_stop"
                            exit_context = f"drop_from_high={drop_from_high:.3%} <= -{trailing_trail_f:.3%} (high={trail_high:.4f})"
        except (TypeError, ValueError):
            pass

    # ── Configurable Greeks exits (opt-in per-sim via profile keys) ─────────
    # Run AFTER standard exits — a profitable trade will never be ejected here.

    # Helper: resolve threshold with adaptive overrides
    def _eff(key, default):
        try:
            from analytics.adaptive_tuning import get_effective_threshold
            return get_effective_threshold(trade.get("sim_id", ""), profile, key, default)
        except Exception:
            return float(profile.get(key, default))

    # 1. Enhanced theta burn: tighten or force exit on near-expiry losing trades
    if not should_exit and profile.get("theta_burn_enabled", False):
        try:
            dte_threshold = int(profile.get("theta_burn_dte_threshold", 1))
            tighten_pct   = _eff("theta_burn_stop_tighten_pct", 0.50)
            expiry_raw    = trade.get("expiry")
            if isinstance(expiry_raw, str):
                expiry_date = datetime.fromisoformat(expiry_raw).date()
                current_dte = (expiry_date - now_et.date()).days
                if current_dte <= dte_threshold and gain_pct is not None and gain_pct < 0:
                    if current_dte <= 0 and gain_pct < -0.05:
                        should_exit      = True
                        exit_reason      = "theta_burn_0dte"
                        exit_context     = f"gain_pct={gain_pct:.3%} current_dte={current_dte}"
                        spread_guard_bypass = True
                        logging.warning(
                            "greeks_exit theta_burn_0dte: trade_id=%s gain=%.3f",
                            trade.get("trade_id"), gain_pct,
                        )
                    else:
                        base_sl      = abs(float(profile.get("stop_loss_pct", 0.30)))
                        tightened_sl = base_sl * (1.0 - tighten_pct)
                        if gain_pct <= -tightened_sl:
                            should_exit      = True
                            exit_reason      = "theta_burn_tightened"
                            exit_context     = f"gain_pct={gain_pct:.3%} tightened_sl={tightened_sl:.3%} dte={current_dte}"
                            spread_guard_bypass = True
                            logging.warning(
                                "greeks_exit theta_burn_tightened: trade_id=%s gain=%.3f dte=%d",
                                trade.get("trade_id"), gain_pct, current_dte,
                            )
        except Exception:
            pass

    # 2. IV crush detection: option dropped more than vega_mult × entry_vega in dollars
    if not should_exit and profile.get("iv_crush_exit_enabled", False):
        vega_at_entry = trade.get("vega_at_entry")
        if (isinstance(vega_at_entry, (int, float))
                and float(vega_at_entry) > 0
                and gain_pct is not None
                and gain_pct < -0.15):
            try:
                vega_entry        = float(vega_at_entry)
                vega_mult         = _eff("iv_crush_vega_multiplier", 2.0)
                entry_price_val   = float(trade.get("entry_price", 0))
                if entry_price_val > 0:
                    option_drop       = abs(gain_pct) * entry_price_val
                    iv_crush_threshold = vega_entry * vega_mult
                    if option_drop > iv_crush_threshold:
                        should_exit      = True
                        exit_reason      = "iv_crush_exit"
                        exit_context     = (
                            f"gain_pct={gain_pct:.3%} option_drop={option_drop:.4f} "
                            f"threshold={iv_crush_threshold:.4f} vega={vega_entry:.4f}"
                        )
                        spread_guard_bypass = True
                        logging.warning(
                            "greeks_exit iv_crush: trade_id=%s gain=%.3f drop=%.4f threshold=%.4f",
                            trade.get("trade_id"), gain_pct, option_drop, iv_crush_threshold,
                        )
            except (TypeError, ValueError):
                pass

    # 3. Delta erosion guard: entry_delta ≥ entry_min, estimated current delta < current_max
    # Proxy: estimated delta = entry_delta × (current_price / entry_price), capped at 0
    if not should_exit and profile.get("delta_erosion_exit_enabled", False):
        delta_at_entry = trade.get("delta_at_entry")
        if isinstance(delta_at_entry, (int, float)) and gain_pct is not None and gain_pct < 0:
            try:
                entry_delta = abs(float(delta_at_entry))
                entry_min   = float(profile.get("delta_erosion_entry_min", 0.40))
                current_max = _eff("delta_erosion_current_max", 0.20)
                if entry_delta >= entry_min:
                    price_ratio = max(0.0, 1.0 + gain_pct)
                    est_delta   = entry_delta * price_ratio
                    if est_delta < current_max:
                        should_exit      = True
                        exit_reason      = "delta_erosion"
                        exit_context     = (
                            f"gain_pct={gain_pct:.3%} entry_delta={entry_delta:.3f} "
                            f"est_delta={est_delta:.3f} threshold={current_max}"
                        )
                        spread_guard_bypass = True
                        logging.warning(
                            "greeks_exit delta_erosion: trade_id=%s gain=%.3f "
                            "entry_delta=%.3f est_delta=%.3f",
                            trade.get("trade_id"), gain_pct, entry_delta, est_delta,
                        )
            except (TypeError, ValueError):
                pass

    return should_exit, exit_reason, exit_context, spread_guard_bypass
