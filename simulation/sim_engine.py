import os
import asyncio
import random
import uuid
import yaml
import pytz
import math
import logging
import time as _time
from datetime import datetime, time
from simulation.sim_portfolio import SimPortfolio
from simulation.sim_executor import sim_try_fill, sim_compute_risk_dollars, sim_should_trade_now
from simulation.sim_contract import select_sim_contract, select_sim_contract_with_reason, get_iv_series
from simulation.sim_live_router import sim_live_router, manage_live_exit
from execution.option_executor import get_option_price
from simulation.sim_signals import derive_sim_signal, get_signal_family
from core.market_clock import get_time_bucket
from simulation.sim_ml import predict_sim_trade, record_sim_trade_close
from analytics.sim_features import compute_sim_features
from core.md_state import is_md_enabled
from signals.volatility import volatility_state
from core.data_service import get_symbol_dataframe


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "simulation", "sim_config.yaml")


def _load_profiles() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
        return {k: v for k, v in raw.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return {}


def _load_global_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
        return raw.get("_global") or {}
    except Exception:
        return {}


_PROFILES = _load_profiles()
_GLOBAL_CONFIG = _load_global_config()
_LAST_CHAIN_CALL_TS = 0.0
_CHAIN_CALL_MIN_INTERVAL = 1.0  # seconds between Alpaca snapshot calls
DAYTRADE_EOD_CUTOFF = time(15, 55)  # ET cutoff for day-trading sims to flatten
EXPIRY_EOD_CUTOFF = time(15, 55)    # ET cutoff for same-day expiries


# ---------------------------------------------------------------------------
# Private helpers (extracted to reduce run_sim_entries / run_sim_exits size)
# ---------------------------------------------------------------------------

def _check_circuit_breaker(sim, profile, sim_id: str):
    """Check the circuit-breaker gate for a live sim.

    Returns (should_skip, result_dict).  When should_skip is True the caller
    should append result_dict and continue to the next iteration.
    """
    cb_config = profile.get("circuit_breaker")
    if not (isinstance(cb_config, dict) and cb_config.get("enabled")):
        return False, None

    cb_source_id = cb_config.get("source_sim")
    cb_window = int(cb_config.get("rolling_window", 20))
    cb_min_wr = float(cb_config.get("min_win_rate", 0.35))
    cb_min_exp = float(cb_config.get("min_expectancy", -50.0))
    cb_recover_wr = float(cb_config.get("recovery_win_rate", 0.45))
    cb_recover_exp = float(cb_config.get("recovery_expectancy", 0.0))

    if not cb_source_id:
        return False, None

    try:
        cb_src_profile = _PROFILES.get(cb_source_id, {})
        cb_src_sim = SimPortfolio(cb_source_id, cb_src_profile)
        cb_src_sim.load()
        cb_log = cb_src_sim.trade_log if isinstance(cb_src_sim.trade_log, list) else []
        cb_recent = cb_log[-cb_window:] if len(cb_log) >= cb_window else cb_log

        if len(cb_recent) >= cb_window:
            cb_wins = 0
            cb_pnl_total = 0.0
            for t in cb_recent:
                pnl = t.get("realized_pnl_dollars")
                if pnl is None:
                    continue
                try:
                    pnl_f = float(pnl)
                except (TypeError, ValueError):
                    continue
                cb_pnl_total += pnl_f
                if pnl_f > 0:
                    cb_wins += 1
            cb_wr = cb_wins / len(cb_recent) if cb_recent else 0
            cb_exp = cb_pnl_total / len(cb_recent) if cb_recent else 0

            was_tripped = sim.profile.get("_circuit_breaker_tripped", False)

            if not was_tripped:
                if cb_wr < cb_min_wr or cb_exp < cb_min_exp:
                    sim.profile["_circuit_breaker_tripped"] = True
                    return True, {
                        "sim_id": sim_id,
                        "status": "circuit_breaker_tripped",
                        "reason": "source_performance_degraded",
                        "source_sim": cb_source_id,
                        "source_wr": round(cb_wr, 3),
                        "source_exp": round(cb_exp, 2),
                        "threshold_wr": cb_min_wr,
                        "threshold_exp": cb_min_exp,
                        "window": cb_window,
                    }
            else:
                if cb_wr >= cb_recover_wr and cb_exp >= cb_recover_exp:
                    sim.profile["_circuit_breaker_tripped"] = False
                    return False, {
                        "sim_id": sim_id,
                        "status": "circuit_breaker_recovered",
                        "source_sim": cb_source_id,
                        "source_wr": round(cb_wr, 3),
                        "source_exp": round(cb_exp, 2),
                    }
                else:
                    return True, {
                        "sim_id": sim_id,
                        "status": "circuit_breaker_held",
                        "reason": "source_still_degraded",
                        "source_sim": cb_source_id,
                        "source_wr": round(cb_wr, 3),
                        "source_exp": round(cb_exp, 2),
                    }
    except Exception:
        logging.exception("circuit_breaker_check_error")

    return False, None


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

    return should_exit, exit_reason, exit_context, spread_guard_bypass


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


def _build_paper_trade_dict(
    sim_id, contract, fill_result, qty, direction, regime,
    time_of_day_bucket, signal_mode, signal_meta, ml_prediction,
    effective_profile, profile, feature_snapshot, underlying_price, df,
):
    """Build the trade dict that is recorded into SimPortfolio after a paper fill."""
    import re as _re
    _opt_sym = contract["option_symbol"] or ""
    _underlying = (_re.match(r'^([A-Z]{1,6})', _opt_sym) or [None, ''])[1] or 'SPY'

    trade = {
        "trade_id": f"{sim_id}__{uuid.uuid4()}",
        "sim_id": sim_id,
        "symbol": _underlying,
        "option_symbol": contract["option_symbol"],
        "entry_price": fill_result["fill_price"],
        "qty": qty,
        "entry_time": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
        "horizon": effective_profile.get("horizon", profile.get("horizon")),
        "dte_bucket": str(contract["dte"]),
        "otm_pct": contract["otm_pct_applied"],
        "direction": direction,
        "strike": contract["strike"],
        "expiry": contract["expiry"],
        "contract_type": contract["contract_type"],
        "hold_min_seconds": int(effective_profile.get("hold_min_seconds", profile.get("hold_min_seconds"))),
        "hold_max_seconds": int(effective_profile.get("hold_max_seconds", profile.get("hold_max_seconds"))),
        "entry_price_source": fill_result.get("price_source", "mid_plus_slippage"),
    }
    if feature_snapshot:
        trade["feature_snapshot"] = feature_snapshot
    if sim_id == "SIM09" or str(signal_mode).upper() == "OPPORTUNITY":
        vol_state = volatility_state(df)
        vol_stop_map = {
            "DEAD": 0.03,
            "LOW": 0.04,
            "NORMAL": 0.06,
            "HIGH": 0.08,
        }
        dynamic_stop = vol_stop_map.get(str(vol_state).upper(), 0.06)
        trade["stop_loss_pct"] = min(0.10, max(0.01, float(dynamic_stop)))
    trade["regime_at_entry"] = regime
    trade["time_of_day_bucket"] = time_of_day_bucket
    trade["signal_mode"] = signal_mode
    trade["strategy_family"] = get_signal_family(str(signal_mode).upper())
    if isinstance(signal_meta, dict):
        trade["structure_score"] = signal_meta.get("structure_score")
    trade["entry_context"] = None  # caller sets this
    trade["spread_pct"] = fill_result.get("spread_pct")
    trade["iv_at_entry"] = contract.get("iv")
    trade["delta_at_entry"] = contract.get("delta")
    trade["gamma_at_entry"] = contract.get("gamma")
    trade["theta_at_entry"] = contract.get("theta")
    trade["vega_at_entry"] = contract.get("vega")
    if isinstance(ml_prediction, dict):
        trade["predicted_direction"] = ml_prediction.get("predicted_direction")
        trade["prediction_confidence"] = ml_prediction.get("prediction_confidence")
        trade["direction_prob"] = ml_prediction.get("direction_prob")
        trade["edge_prob"] = ml_prediction.get("edge_prob")
        trade["regime"] = ml_prediction.get("regime")
        trade["volatility"] = ml_prediction.get("volatility")
        trade["conviction_score"] = ml_prediction.get("conviction_score")
        trade["impulse"] = ml_prediction.get("impulse")
        trade["follow_through"] = ml_prediction.get("follow_through")
        trade["setup"] = ml_prediction.get("setup")
        trade["style"] = ml_prediction.get("style")
        trade["confidence"] = ml_prediction.get("confidence")
        trade["ml_probability"] = ml_prediction.get("edge_prob")
    return trade


def _trade_grade(tr) -> float | None:
    """Best available confidence/edge score for a trade."""
    candidates = []
    for key in ("edge_prob", "prediction_confidence", "confidence", "ml_probability"):
        val = tr.get(key)
        if isinstance(val, (int, float)):
            candidates.append(float(val))
    return max(candidates) if candidates else None


def _count_directional_exposure(direction: str, symbol: str | None = None) -> int:
    """Count how many sims have open trades in the given direction (optionally per symbol)."""
    count = 0
    sym_upper = symbol.upper() if symbol else None
    for sid, prof in _PROFILES.items():
        if str(sid).startswith("_"):
            continue
        try:
            s = SimPortfolio(sid, prof)
            s.load()
            for t in s.open_trades:
                if not isinstance(t, dict) or t.get("direction") != direction:
                    continue
                if sym_upper is None or (t.get("symbol") or "").upper() == sym_upper:
                    count += 1
                    break
        except Exception:
            continue
    return count


def _count_family_directional_exposure(family: str, direction: str, symbol: str | None = None) -> int:
    """Count sims in the same strategy family with an open trade in the given direction (optionally per symbol)."""
    count = 0
    sym_upper = symbol.upper() if symbol else None
    for sid, prof in _PROFILES.items():
        if str(sid).startswith("_"):
            continue
        mode = str(prof.get("signal_mode", "")).upper()
        if get_signal_family(mode) != family:
            continue
        try:
            s = SimPortfolio(sid, prof)
            s.load()
            for t in s.open_trades:
                if not isinstance(t, dict) or t.get("direction") != direction:
                    continue
                if sym_upper is None or (t.get("symbol") or "").upper() == sym_upper:
                    count += 1
                    break
        except Exception:
            continue
    return count


# ---------------------------------------------------------------------------
# run_sim_entries / run_sim_exits — extracted to runner modules to reduce size.
# Re-exported here so existing callers continue to work unchanged.
# ---------------------------------------------------------------------------
from simulation.sim_entry_runner import run_sim_entries  # noqa: E402, F401
from simulation.sim_exit_runner import run_sim_exits     # noqa: E402, F401
