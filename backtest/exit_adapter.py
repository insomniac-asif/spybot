"""
backtest/exit_adapter.py
Synchronous exit condition checker extracted from sim_exit_helpers._evaluate_exit_conditions().

Handles:
- stop_loss_pct: if option price drops X% from entry -> exit (with time-decay adjustment)
- profit_target_pct: if option price rises X% -> exit (with time-decay adjustment)
- trailing_stop: after profit lock, trail behind peak
- hold_max_seconds: time-based exit
- EOD: force close at 15:58 ET

Returns (should_exit: bool, reason: str, exit_price: float)
"""
from __future__ import annotations
from datetime import datetime, time as dt_time, date as dt_date
import pytz

EOD_FORCE_CLOSE = dt_time(15, 58)  # Force close all positions at 15:58 ET
MARKET_CLOSE_MINUTES = 390  # 9:30-16:00 = 390 minutes


def _compute_decay_factor(current_bar_ts, trade: dict, profile: dict) -> float:
    """Compute time-decay factor for TP/SL adjustment.

    For DTE <= 1: use minutes_to_close / 390 (intraday decay)
    For DTE > 1: use days_to_expiry / max_dte (multi-day decay)
    Returns 1.0 (no decay) if not applicable.
    """
    try:
        # Check DTE
        expiry_raw = trade.get("expiry")
        if not isinstance(expiry_raw, str):
            return 1.0

        expiry_date = datetime.fromisoformat(expiry_raw).date()
        if isinstance(current_bar_ts, datetime):
            if current_bar_ts.tzinfo is not None:
                bar_et = current_bar_ts.astimezone(pytz.timezone("America/New_York"))
            else:
                bar_et = current_bar_ts
            current_date = bar_et.date()
        else:
            return 1.0

        dte = (expiry_date - current_date).days

        if dte <= 1:
            # Intraday decay: 1.0 at open, 0.0 at close
            market_close = bar_et.replace(hour=16, minute=0, second=0, microsecond=0)
            minutes_remaining = max(0, (market_close - bar_et).total_seconds() / 60)
            return minutes_remaining / MARKET_CLOSE_MINUTES
        else:
            # Multi-day decay for longer DTE
            dte_max = int(profile.get("dte_max", 7))
            if dte_max <= 0:
                dte_max = 7
            return min(1.0, dte / dte_max)
    except Exception:
        return 1.0


def check_exit_conditions(
    trade: dict,
    profile: dict,
    current_price: float,
    elapsed_seconds: float,
    current_bar_ts: datetime,
) -> tuple:
    """
    Evaluate all exit conditions for a backtest trade at the current bar.

    Parameters
    ----------
    trade : dict
        Open trade record. Contains entry_price, stop_loss_pct, qty, trailing_stop_activated, etc.
    profile : dict
        Sim profile config (stop_loss_pct, profit_target_pct, trailing_stop_*, hold_max_seconds, etc.)
    current_price : float
        Current option mid price.
    elapsed_seconds : float
        Seconds since entry.
    current_bar_ts : datetime
        Timestamp of the current bar (ET, may be tz-aware or naive).

    Returns
    -------
    (should_exit, exit_reason, exit_price) : tuple[bool, str, float]
    """
    should_exit = False
    exit_reason = "still_open"

    entry_price = float(trade.get("entry_price", 0))
    if entry_price <= 0 or current_price is None or current_price <= 0:
        return False, "still_open", current_price

    gain_pct = (current_price - entry_price) / entry_price

    # ── Time-decay factor for TP/SL ──────────────────────────────────────────
    decay_factor = _compute_decay_factor(current_bar_ts, trade, profile)
    tp_decay_floor = float(profile.get("tp_decay_floor", 0.3))
    sl_decay_floor = float(profile.get("sl_decay_floor", 0.5))

    # ── EOD force close ──────────────────────────────────────────────────────
    bar_time = current_bar_ts
    if isinstance(bar_time, datetime):
        if bar_time.tzinfo is not None:
            eastern = pytz.timezone("America/New_York")
            bar_time = bar_time.astimezone(eastern)
            bar_et_time = bar_time.time()
        else:
            bar_et_time = bar_time.time()
        if bar_et_time >= EOD_FORCE_CLOSE:
            return True, "eod_close", current_price

    # ── Hold min check: don't exit before hold_min_seconds ──────────────────
    hold_min = float(trade.get("hold_min_seconds", profile.get("hold_min_seconds", 60)))
    if elapsed_seconds < hold_min:
        return False, "still_open", current_price

    # ── Stop loss (with time-decay) ──────────────────────────────────────────
    stop_loss_pct = trade.get("stop_loss_pct") or profile.get("stop_loss_pct")
    if stop_loss_pct is not None:
        try:
            effective_sl = abs(float(stop_loss_pct)) * max(sl_decay_floor, decay_factor)
            peak_price = float(trade.get("peak_price") or 0)
            # Update peak price tracking
            if current_price > entry_price and current_price > peak_price:
                trade["peak_price"] = current_price
                peak_price = current_price

            original_sl_price = entry_price * (1 - effective_sl)
            # Breakeven lock: once price moved up, floor stops at entry
            effective_sl_price = (
                max(original_sl_price, entry_price)
                if peak_price > entry_price
                else original_sl_price
            )

            if current_price <= effective_sl_price:
                should_exit = True
                if effective_sl_price > original_sl_price:
                    exit_reason = "breakeven_stop"
                else:
                    exit_reason = "stop_loss"
                return should_exit, exit_reason, current_price
        except (TypeError, ValueError):
            pass

    # ── Profit target (with time-decay) ──────────────────────────────────────
    profit_target_pct = profile.get("profit_target_pct")
    effective_target = trade.get("tp2_target_pct") if trade.get("tp2_activated") else profit_target_pct
    if effective_target is not None:
        try:
            target_val = abs(float(effective_target)) * max(tp_decay_floor, decay_factor)
            if gain_pct >= target_val:
                should_exit = True
                exit_reason = "profit_target_2" if trade.get("tp2_activated") else "profit_target"
                return should_exit, exit_reason, current_price
        except (TypeError, ValueError):
            pass

    # ── Trailing stop ────────────────────────────────────────────────────────
    trailing_activate = profile.get("trailing_stop_activate_pct")
    trailing_trail = profile.get("trailing_stop_trail_pct")
    if trailing_activate is not None and trailing_trail is not None:
        try:
            activate_f = abs(float(trailing_activate))
            trail_f = abs(float(trailing_trail))
            if not trade.get("trailing_stop_activated", False):
                if gain_pct >= activate_f:
                    trade["trailing_stop_activated"] = True
                    trade["trailing_stop_high"] = current_price
            else:
                if current_price > trade.get("trailing_stop_high", 0):
                    trade["trailing_stop_high"] = current_price
                trail_high = float(trade.get("trailing_stop_high", 0))
                if trail_high > 0:
                    drop_from_high = (current_price - trail_high) / trail_high
                    if drop_from_high <= -trail_f:
                        return True, "trailing_stop", current_price
        except (TypeError, ValueError):
            pass

    # ── Hold max elapsed ─────────────────────────────────────────────────────
    hold_max = float(trade.get("hold_max_seconds", profile.get("hold_max_seconds", 86400)))
    if elapsed_seconds >= hold_max:
        return True, "hold_max_elapsed", current_price

    # ── Theta burn: 0DTE near-expiry losing ─────────────────────────────────
    if profile.get("theta_burn_enabled", False) and gain_pct is not None and gain_pct < 0:
        try:
            dte_threshold = int(profile.get("theta_burn_dte_threshold", 1))
            tighten_pct = float(profile.get("theta_burn_stop_tighten_pct", 0.50))
            expiry_raw = trade.get("expiry")
            if isinstance(expiry_raw, str):
                expiry_date = datetime.fromisoformat(expiry_raw).date()
                if isinstance(current_bar_ts, datetime):
                    today = (current_bar_ts.date() if current_bar_ts.tzinfo is None
                             else current_bar_ts.astimezone(pytz.timezone("America/New_York")).date())
                else:
                    today = current_bar_ts.date() if hasattr(current_bar_ts, "date") else datetime.now().date()
                current_dte = (expiry_date - today).days
                if current_dte <= dte_threshold:
                    if current_dte <= 0 and gain_pct < -0.05:
                        return True, "theta_burn_0dte", current_price
                    else:
                        base_sl = abs(float(profile.get("stop_loss_pct", 0.30)))
                        tightened_sl = base_sl * (1.0 - tighten_pct)
                        if gain_pct <= -tightened_sl:
                            return True, "theta_burn_tightened", current_price
        except Exception:
            pass

    return False, "still_open", current_price
