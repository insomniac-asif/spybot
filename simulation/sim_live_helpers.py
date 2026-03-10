"""
Pure helper functions for sim_live_router.py.
Extracted to keep sim_live_router.py concise.
"""

import os
import yaml
import pytz
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "simulation", "sim_config.yaml")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _trade_grade(tr) -> float | None:
    """Best available confidence/edge score for a trade."""
    candidates = []
    for key in ("edge_prob", "prediction_confidence", "confidence", "ml_probability"):
        val = tr.get(key)
        if isinstance(val, (int, float)):
            candidates.append(float(val))
    return max(candidates) if candidates else None


def _safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _load_profiles() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
        return {k: v for k, v in raw.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return {}


def _now_et_iso() -> str:
    return datetime.now(pytz.timezone("US/Eastern")).isoformat()


# ---------------------------------------------------------------------------
# Trade dict builder
# ---------------------------------------------------------------------------

def _build_live_trade_dict(
    sim_id: str,
    contract: dict,
    fill_result: dict,
    qty: int,
    profile: dict,
    direction: str,
    regime,
    time_of_day_bucket,
    signal_mode,
    entry_context,
    feature_snapshot,
    ml_prediction,
) -> dict:
    """Construct the trade dict that is recorded into the portfolio on entry."""
    trade = {
        "trade_id": f"{sim_id}__{os.urandom(8).hex()}",
        "sim_id": sim_id,
        "option_symbol": contract["option_symbol"],
        "entry_price": fill_result.get("fill_price"),
        "qty": qty,
        "entry_time": _now_et_iso(),
        "horizon": profile.get("horizon"),
        "dte_bucket": str(contract.get("dte")),
        "otm_pct": contract.get("otm_pct_applied"),
        "direction": direction,
        "strike": contract.get("strike"),
        "expiry": contract.get("expiry"),
        "contract_type": contract.get("contract_type"),
        "hold_min_seconds": int(profile.get("hold_min_seconds", 0)),
        "hold_max_seconds": int(profile.get("hold_max_seconds", 0)),
        "entry_price_source": "live_fill",
        "execution_mode": "live",
    }
    if feature_snapshot:
        trade["feature_snapshot"] = feature_snapshot
    trade["regime_at_entry"] = regime
    trade["time_of_day_bucket"] = time_of_day_bucket
    trade["signal_mode"] = signal_mode
    trade["entry_context"] = entry_context
    # greeks at entry (from contract snapshot)
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


# ---------------------------------------------------------------------------
# Exit-condition determination (extracted from manage_live_exit)
# ---------------------------------------------------------------------------
from core.md_state import is_md_enabled


def _determine_exit_condition(sim, trade, current_price, gain_pct, elapsed_seconds, profile, now_et):
    """
    Evaluate all exit conditions for a live trade.

    Returns (should_exit: bool, exit_reason: str|None, exit_context: str|None).
    force_exit / forced reasons passed in via `should_exit` / `exit_reason` when caller
    has already decided on a forced exit (e.g. expiry, EOD); caller should short-circuit
    if should_exit is already True before calling this.
    """
    should_exit = False
    exit_reason = None
    exit_context = None

    stop_loss_pct = profile.get("stop_loss_pct")
    if stop_loss_pct is not None and is_md_enabled() and sim.sim_id != "SIM09":
        try:
            stop_loss_pct = max(float(stop_loss_pct) * 0.7, 0.05)
        except (TypeError, ValueError):
            pass
    if stop_loss_pct is not None:
        try:
            entry_price = float(trade.get("entry_price", 0))
            if entry_price > 0:
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct <= -abs(float(stop_loss_pct)):
                    should_exit = True
                    exit_reason = "stop_loss"
                    exit_context = f"loss_pct={loss_pct:.3%} <= -{abs(float(stop_loss_pct)):.3%}"
        except (TypeError, ValueError):
            pass

    profit_target_pct = profile.get("profit_target_pct")

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
            except (TypeError, ValueError):
                pass

    # Near-TP adaptive lock + TP2
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

    # Profit lock check
    lock_pct = trade.get("lock_profit_pct")
    if not should_exit and gain_pct is not None and trade.get("tp2_activated") and isinstance(lock_pct, (int, float)):
        try:
            if gain_pct <= float(lock_pct):
                should_exit = True
                exit_reason = "profit_lock"
                exit_context = f"gain_pct={gain_pct:.3%} <= lock_pct={float(lock_pct):.3%}"
        except (TypeError, ValueError):
            pass

    # Effective target (TP2 overrides base)
    effective_target = profit_target_pct
    if trade.get("tp2_activated") and trade.get("tp2_target_pct") is not None:
        effective_target = trade.get("tp2_target_pct")

    if not should_exit and effective_target is not None:
        try:
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

    # Trailing stop
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

    # Hold max
    if not should_exit:
        hold_max_seconds = int(trade.get("hold_max_seconds", 0))
        if hold_max_seconds > 0 and elapsed_seconds >= hold_max_seconds:
            should_exit = True
            exit_reason = "hold_max_elapsed"
            exit_context = f"elapsed={int(elapsed_seconds)}s >= hold_max={int(hold_max_seconds)}s"

    # ── Configurable Greeks exits (opt-in per-sim via profile keys) ─────────
    # Run AFTER standard exits — a profitable trade will never be ejected here.

    # 1. Enhanced theta burn: tighten or force exit on near-expiry losing trades
    if not should_exit and profile.get("theta_burn_enabled", False):
        try:
            dte_threshold = int(profile.get("theta_burn_dte_threshold", 1))
            tighten_pct   = float(profile.get("theta_burn_stop_tighten_pct", 0.50))
            expiry_raw    = trade.get("expiry")
            if isinstance(expiry_raw, str):
                expiry_date = datetime.fromisoformat(expiry_raw).date()
                current_dte = (expiry_date - now_et.date()).days
                if current_dte <= dte_threshold and gain_pct is not None and gain_pct < 0:
                    if current_dte <= 0 and gain_pct < -0.05:
                        should_exit  = True
                        exit_reason  = "theta_burn_0dte"
                        exit_context = f"gain_pct={gain_pct:.3%} current_dte={current_dte}"
                        logging.warning(
                            "greeks_exit theta_burn_0dte: trade_id=%s gain=%.3f",
                            trade.get("trade_id"), gain_pct,
                        )
                    else:
                        base_sl      = abs(float(profile.get("stop_loss_pct", 0.30)))
                        tightened_sl = base_sl * (1.0 - tighten_pct)
                        if gain_pct <= -tightened_sl:
                            should_exit  = True
                            exit_reason  = "theta_burn_tightened"
                            exit_context = f"gain_pct={gain_pct:.3%} tightened_sl={tightened_sl:.3%} dte={current_dte}"
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
                vega_mult         = float(profile.get("iv_crush_vega_multiplier", 2.0))
                entry_price_val   = float(trade.get("entry_price", 0))
                if entry_price_val > 0:
                    option_drop        = abs(gain_pct) * entry_price_val
                    iv_crush_threshold = vega_entry * vega_mult
                    if option_drop > iv_crush_threshold:
                        should_exit  = True
                        exit_reason  = "iv_crush_exit"
                        exit_context = (
                            f"gain_pct={gain_pct:.3%} option_drop={option_drop:.4f} "
                            f"threshold={iv_crush_threshold:.4f} vega={vega_entry:.4f}"
                        )
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
                current_max = float(profile.get("delta_erosion_current_max", 0.20))
                if entry_delta >= entry_min:
                    price_ratio = max(0.0, 1.0 + gain_pct)
                    est_delta   = entry_delta * price_ratio
                    if est_delta < current_max:
                        should_exit  = True
                        exit_reason  = "delta_erosion"
                        exit_context = (
                            f"gain_pct={gain_pct:.3%} entry_delta={entry_delta:.3f} "
                            f"est_delta={est_delta:.3f} threshold={current_max}"
                        )
                        logging.warning(
                            "greeks_exit delta_erosion: trade_id=%s gain=%.3f "
                            "entry_delta=%.3f est_delta=%.3f",
                            trade.get("trade_id"), gain_pct, entry_delta, est_delta,
                        )
            except (TypeError, ValueError):
                pass

    return should_exit, exit_reason, exit_context
