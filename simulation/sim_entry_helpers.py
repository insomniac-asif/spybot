"""
simulation/sim_entry_helpers.py
Private helper functions for sim_entry_runner.py.
Extracted to keep sim_entry_runner.py under 500 lines.
"""
import asyncio
import logging
import math
import time as _time
import uuid
from datetime import datetime

import pytz

from simulation.sim_portfolio import SimPortfolio
from simulation.sim_signals import get_signal_family
from signals.volatility import volatility_state


def _get_profiles_and_global():
    """Lazily import _PROFILES / _GLOBAL_CONFIG from sim_engine."""
    import simulation.sim_engine as _eng
    return _eng._PROFILES, _eng._GLOBAL_CONFIG


def _trade_grade(tr) -> "float | None":
    """Best available confidence/edge score for a trade."""
    candidates = []
    for key in ("edge_prob", "prediction_confidence", "confidence", "ml_probability"):
        val = tr.get(key)
        if isinstance(val, (int, float)):
            candidates.append(float(val))
    return max(candidates) if candidates else None


def _count_global_open_trades() -> int:
    """Count total open trades across ALL sims."""
    _PROFILES, _ = _get_profiles_and_global()
    total = 0
    for sid, prof in _PROFILES.items():
        if str(sid).startswith("_"):
            continue
        try:
            s = SimPortfolio(sid, prof)
            s.load()
            total += len(s.open_trades)
        except Exception:
            continue
    return total


def _count_directional_exposure(direction: str, symbol: "str | None" = None) -> int:
    """Count how many sims have open trades in the given direction (optionally per symbol)."""
    _PROFILES, _ = _get_profiles_and_global()
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


def _count_family_directional_exposure(family: str, direction: str, symbol: "str | None" = None) -> int:
    """Count sims in the same strategy family with an open trade in the given direction."""
    _PROFILES, _ = _get_profiles_and_global()
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


def _check_circuit_breaker(sim, profile, sim_id: str):
    """Check the circuit-breaker gate for a live sim.

    Returns (should_skip, result_dict).  When should_skip is True the caller
    should append result_dict and continue to the next iteration.
    """
    _PROFILES, _ = _get_profiles_and_global()
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


async def _execute_live_entry(
    sim, sim_id, profile, _PROFILES, direction, underlying_price,
    ml_prediction, regime, time_of_day_bucket, signal_mode,
    entry_context, feature_snapshot, _trade_symbol,
    effective_profile, results,
):
    """Handle the live execution path for a single sim/symbol combo.

    Appends result(s) to `results` and returns True if the symbol loop
    should `continue` to the next symbol, False otherwise.
    """
    from simulation.sim_executor import sim_should_trade_now
    from simulation.sim_live_router import sim_live_router

    source_sim_id = sim.profile.get("source_sim")
    min_source_trades = 0
    try:
        min_source_trades = int(sim.profile.get("min_source_trades", 0))
    except (TypeError, ValueError):
        min_source_trades = 0
    if source_sim_id and min_source_trades > 0:
        src_trade_count = 0
        try:
            src_profile = _PROFILES.get(source_sim_id, {})
            src_sim = SimPortfolio(source_sim_id, src_profile)
            await asyncio.to_thread(src_sim.load)
            src_trade_count = len(src_sim.trade_log) if isinstance(src_sim.trade_log, list) else 0
        except Exception:
            src_trade_count = 0
        if src_trade_count < min_source_trades:
            results.append({
                "sim_id": sim_id,
                "status": "skipped",
                "reason": "insufficient_trade_history",
                "trade_count": src_trade_count,
                "min_trades_for_live": min_source_trades,
                "entry_context": entry_context,
                "signal_mode": signal_mode,
            })
            return True

    cb_skip, cb_result = _check_circuit_breaker(sim, sim.profile, sim_id)
    if cb_result is not None:
        results.append(cb_result)
    if cb_skip:
        return True

    if not sim.profile.get("enabled"):
        results.append({"sim_id": sim_id, "status": "skipped", "reason": "live_disabled"})
        return True
    ok, reason = sim_should_trade_now(effective_profile)
    if not ok:
        results.append({
            "sim_id": sim_id, "status": "skipped", "reason": reason,
            "entry_context": entry_context, "signal_mode": signal_mode,
        })
        return True
    ok, reason = sim.can_trade()
    if not ok:
        results.append({
            "sim_id": sim_id, "status": "skipped", "reason": reason,
            "entry_context": entry_context, "signal_mode": signal_mode,
        })
        return True

    def _get_chain_ts():
        import simulation.sim_engine as _eng
        return _eng._LAST_CHAIN_CALL_TS

    def _set_chain_ts(val):
        import simulation.sim_engine as _eng
        _eng._LAST_CHAIN_CALL_TS = val

    def _get_chain_interval():
        import simulation.sim_engine as _eng
        return _eng._CHAIN_CALL_MIN_INTERVAL

    elapsed = _time.monotonic() - _get_chain_ts()
    if elapsed < _get_chain_interval():
        await asyncio.sleep(_get_chain_interval() - elapsed)
    _set_chain_ts(_time.monotonic())

    live_result = await sim_live_router(
        sim_id=sim_id, direction=direction, price=underlying_price,
        ml_prediction=ml_prediction, regime=regime,
        time_of_day_bucket=time_of_day_bucket, signal_mode=signal_mode,
        entry_context=entry_context, feature_snapshot=feature_snapshot,
        symbol=_trade_symbol,
    )
    if not isinstance(live_result, dict) or live_result.get("status") != "success":
        results.append({
            "sim_id": sim_id, "status": "error",
            "reason": (live_result or {}).get("message", "live_order_failed"),
        })
        return True
    results.append({
        "sim_id": sim_id,
        "status": "live_submitted",
        "option_symbol": live_result.get("option_symbol"),
        "qty": live_result.get("qty"),
        "fill_price": live_result.get("fill_price"),
        "entry_price": live_result.get("fill_price"),
        "direction": direction,
        "risk_dollars": live_result.get("risk_dollars"),
        "strike": live_result.get("strike"),
        "expiry": live_result.get("expiry"),
        "dte": live_result.get("dte"),
        "spread_pct": live_result.get("spread_pct"),
        "regime": regime,
        "time_bucket": time_of_day_bucket,
        "mode": "LIVE",
        "balance": live_result.get("balance_after"),
        "entry_context": live_result.get("entry_context") or entry_context,
        "signal_mode": live_result.get("signal_mode") or signal_mode,
        "predicted_direction": live_result.get("predicted_direction") or ml_prediction.get("predicted_direction"),
        "prediction_confidence": live_result.get("prediction_confidence") or ml_prediction.get("prediction_confidence"),
        "edge_prob": live_result.get("edge_prob") or ml_prediction.get("edge_prob"),
        "direction_prob": live_result.get("direction_prob") or ml_prediction.get("direction_prob"),
    })
    return True


async def _execute_paper_entry(
    sim, sim_id, profile, direction, underlying_price,
    ml_prediction, regime, time_of_day_bucket, signal_mode,
    entry_context, feature_snapshot, _trade_symbol,
    effective_profile, df, results,
    structure_data=None, cross_asset_data=None, options_data=None,
):
    """Handle the paper execution path for a single sim/symbol combo.

    Appends result(s) to `results` and returns True if the symbol loop
    should `continue` to the next symbol, False otherwise.
    """
    from simulation.sim_executor import sim_try_fill, sim_compute_risk_dollars, sim_should_trade_now
    from simulation.sim_contract import select_sim_contract_with_reason

    ok, reason = sim_should_trade_now(effective_profile)
    if not ok:
        results.append({
            "sim_id": sim_id, "status": "skipped", "reason": reason,
            "entry_context": entry_context, "signal_mode": signal_mode,
        })
        return True

    ok, reason = sim.can_trade()
    if not ok:
        results.append({
            "sim_id": sim_id, "status": "skipped", "reason": reason,
            "entry_context": entry_context, "signal_mode": signal_mode,
        })
        return True

    def _get_chain_ts():
        import simulation.sim_engine as _eng
        return _eng._LAST_CHAIN_CALL_TS

    def _set_chain_ts(val):
        import simulation.sim_engine as _eng
        _eng._LAST_CHAIN_CALL_TS = val

    def _get_chain_interval():
        import simulation.sim_engine as _eng
        return _eng._CHAIN_CALL_MIN_INTERVAL

    elapsed = _time.monotonic() - _get_chain_ts()
    if elapsed < _get_chain_interval():
        await asyncio.sleep(_get_chain_interval() - elapsed)
    _set_chain_ts(_time.monotonic())

    contract, contract_reason = await asyncio.to_thread(
        select_sim_contract_with_reason,
        direction, underlying_price, {**effective_profile, "sim_id": sim_id},
        symbol=_trade_symbol,
    )
    if contract is None:
        results.append({
            "sim_id": sim_id, "status": "skipped",
            "reason": contract_reason or "no_contract",
            "entry_context": entry_context, "signal_mode": signal_mode,
        })
        return True

    bid = contract["bid"]
    ask = contract["ask"]

    # Probe fill at qty=1 to discover the realistic fill_price with slippage
    fill_result, err = sim_try_fill(
        contract["option_symbol"], qty=1, bid=bid, ask=ask, profile=profile, side="entry"
    )
    if err or fill_result is None:
        results.append({
            "sim_id": sim_id, "status": "skipped", "reason": err,
            "entry_context": entry_context, "signal_mode": signal_mode,
        })
        return True

    fill_price = fill_result["fill_price"]

    # ── Small-account sizing ──────────────────────────────────────────────
    # Use the small-account module when the profile enables small_account_mode,
    # or when balance_start indicates a small account (< $5,000).
    # Falls back to the legacy risk_dollars / fill_price formula otherwise.
    _use_small = (
        profile.get("small_account_mode", False)
        or float(profile.get("balance_start", 25000)) < 5000
    )
    if _use_small:
        from simulation.sim_account_mode import compute_small_account_qty
        qty, risk_dollars, block_reason = compute_small_account_qty(
            sim.balance, fill_price, profile
        )
        if block_reason is not None:
            results.append({
                "sim_id": sim_id, "status": "skipped", "reason": block_reason,
                "entry_context": entry_context, "signal_mode": signal_mode,
            })
            return True
    else:
        risk_dollars = sim_compute_risk_dollars(sim.balance, profile)
        qty = max(1, math.floor(risk_dollars / (fill_price * 100)))

    fill_result, err = sim_try_fill(
        contract["option_symbol"], qty=qty, bid=bid, ask=ask, profile=profile, side="entry"
    )
    if err or fill_result is None:
        results.append({"sim_id": sim_id, "status": "skipped", "reason": err})
        return True

    trade = _build_paper_trade_dict(
        sim_id=sim_id, contract=contract, fill_result=fill_result, qty=qty,
        direction=direction, regime=regime, time_of_day_bucket=time_of_day_bucket,
        signal_mode=signal_mode, signal_meta=None, ml_prediction=ml_prediction,
        effective_profile=effective_profile, profile=profile,
        feature_snapshot=feature_snapshot, underlying_price=underlying_price, df=df,
    )
    trade["entry_context"] = entry_context
    trade["structure_at_entry"] = structure_data if isinstance(structure_data, dict) else {}
    trade["cross_asset_at_entry"] = cross_asset_data if isinstance(cross_asset_data, dict) else {}
    trade["options_at_entry"] = options_data if isinstance(options_data, dict) else {}

    sim.record_open(trade)
    sim.save()

    try:
        from decision.candidate import Candidate
        from analytics.candidate_logger import log_candidate
        log_candidate(Candidate(
            sim_id=sim_id, strategy=signal_mode, symbol=_trade_symbol,
            direction=direction, fired=True,
            entry_ref=float(underlying_price) if underlying_price else None,
            regime=regime or "", time_bucket=time_of_day_bucket or "",
            traded=True, trade_id=trade.get("trade_id"),
            conviction=int(trade.get("conviction_score") or 0) or None,
            signal_params=profile.get("signal_params") or {},
        ))
    except Exception:
        pass

    results.append({
        "sim_id": sim_id, "status": "opened", "symbol": _trade_symbol,
        "trade_id": trade["trade_id"],
        "fill_price": trade["entry_price"],
        "entry_price": trade["entry_price"],
        "qty": qty,
        "option_symbol": trade["option_symbol"],
        "expiry": contract["expiry"],
        "strike": contract["strike"],
        "dte": contract["dte"],
        "spread_pct": contract["spread_pct"],
        "direction": direction,
        "risk_dollars": risk_dollars,
        "regime": regime,
        "time_bucket": time_of_day_bucket,
        "mode": "SIM",
        "balance": sim.balance,
        "entry_context": entry_context,
        "signal_mode": signal_mode,
        "predicted_direction": trade.get("predicted_direction"),
        "prediction_confidence": trade.get("prediction_confidence"),
        "edge_prob": trade.get("edge_prob"),
        "direction_prob": trade.get("direction_prob"),
    })
    return True
