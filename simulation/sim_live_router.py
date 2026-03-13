# LIVE PATH AUDIT
# Two paths can place live orders:
#   1. sim_live_router.py → SIM00 (sim graduation path) — ACTIVE
#   2. decision/trader.py → open_trade_if_valid() (auto_trader) — PAPER ONLY
#      (auto_trader routes through open_trade_if_valid which calls execute_option_entry,
#       but no sim with execution_mode=live feeds through trader.py in production.
#       Both paths are gated by core/singletons.py::RISK_SUPERVISOR.authorize_entry())
# Resolution: sim_live_router.py is the canonical live path for SIM00.
#             decision/trader.py manages the paper auto_trader account only.

import asyncio
import math
import logging
import pytz
from datetime import datetime

from simulation.sim_portfolio import SimPortfolio
from simulation.sim_contract import select_sim_contract, select_sim_contract_with_reason
from simulation.sim_ml import record_sim_trade_close
from execution.option_executor import execute_option_entry, close_option_position, get_option_price
from core.md_state import is_md_enabled
from simulation.sim_live_helpers import (
    _safe_float,
    _trade_grade,
    _safe_int,
    _load_profiles,
    _now_et_iso,
    _build_live_trade_dict,
    _determine_exit_condition,
)


_PROFILES = _load_profiles()
EOD_CUTOFF = datetime.strptime("15:55", "%H:%M").time()


def _is_option_expired(trade: dict) -> bool:
    """Check if the option in this trade has already expired."""
    from datetime import date
    today = date.today()
    expiry = trade.get("expiry")
    if isinstance(expiry, str):
        try:
            expiry_date = datetime.fromisoformat(expiry.replace("Z", "+00:00")).date()
            return today > expiry_date
        except ValueError:
            pass
    # Try OCC symbol format: ROOT YYMMDD C/P STRIKE
    option_symbol = trade.get("option_symbol", "")
    if len(option_symbol) >= 15:
        try:
            date_str = option_symbol[-15:-9]
            expiry_date = datetime.strptime(date_str, "%y%m%d").date()
            return today > expiry_date
        except (ValueError, IndexError):
            pass
    return False


async def sim_live_router(sim_id, direction, price, ml_prediction=None, regime=None, time_of_day_bucket=None, signal_mode=None, entry_context=None, feature_snapshot=None, symbol=None):
    try:
        try:
            from core.runtime_state import RUNTIME, SystemState
            _rs = RUNTIME.state
            if _rs == SystemState.PANIC_LOCKDOWN:
                return {"status": "error", "message": "panic_lockdown"}
            if _rs == SystemState.EXIT_ONLY:
                return {"status": "error", "message": "exit_only_mode"}
            if not RUNTIME.can_enter():
                return {"status": "error", "message": f"runtime_{_rs.value.lower()}"}
        except ImportError:
            pass

        if direction not in {"BULLISH", "BEARISH"}:
            return {"status": "error", "message": "invalid_direction"}
        try:
            price_val = float(price)
        except (TypeError, ValueError):
            return {"status": "error", "message": "invalid_price"}
        if price_val <= 0:
            return {"status": "error", "message": "invalid_price"}

        profiles = _PROFILES or _load_profiles()
        live_active = [
            key
            for key, prof in profiles.items()
            if isinstance(prof, dict)
            and prof.get("execution_mode") == "live"
            and prof.get("enabled")
        ]
        if len(live_active) > 1:
            logging.error("sim_live_router_multi_live_enabled: %s", live_active)
            return {"status": "error", "message": "multiple_live_sims"}

        profile = profiles.get(sim_id)
        if not isinstance(profile, dict):
            return {"status": "error", "message": "invalid_sim_id"}
        if profile.get("execution_mode") != "live":
            return {"status": "error", "message": "sim_not_live"}
        if not profile.get("enabled"):
            return {"status": "error", "message": "sim_disabled"}

        sim = SimPortfolio(sim_id, profile)
        await asyncio.to_thread(sim.load)

        daily_loss_limit = profile.get("daily_loss_limit")
        if daily_loss_limit is not None:
            try:
                # Include unrealized open-trade losses in daily loss check
                _current_prices = {}
                for _t in sim.open_trades:
                    _osym = _t.get("option_symbol", "") if isinstance(_t, dict) else ""
                    if _osym:
                        _cp = await asyncio.to_thread(get_option_price, _osym)
                        if _cp is not None:
                            _current_prices[_osym] = _cp
                _total_loss = sim.get_daily_loss_including_unrealized(_current_prices)
                if _total_loss >= float(daily_loss_limit):
                    return {"status": "error", "message": "daily_loss_limit"}
            except (TypeError, ValueError):
                pass

        max_open_trades = profile.get("max_open_trades")
        if max_open_trades is not None:
            try:
                if int(max_open_trades) > 0 and len(sim.open_trades) >= int(max_open_trades):
                    logging.info("sim_live_router_blocked_max_open_trades: %s", sim_id)
                    return {"status": "error", "message": "max_open_trades"}
            except (TypeError, ValueError):
                pass

        capital_limit = profile.get("capital_limit_dollars", 25000)
        capital_limit_val = _safe_float(capital_limit)
        if capital_limit_val is None or capital_limit_val <= 0:
            return {"status": "error", "message": "invalid_capital_limit"}

        # Resolve trade symbol: caller-supplied > profile "symbol" string > "SPY"
        # (SIM00 uses a "symbols" list, not "symbol" string, so profile.get("symbol") returns None)
        _symbols_list = profile.get("symbols")
        _profile_symbol = profile.get("symbol")
        if symbol:
            _trade_symbol = str(symbol).upper()
        elif _profile_symbol:
            _trade_symbol = str(_profile_symbol).upper()
        elif _symbols_list and isinstance(_symbols_list, list) and len(_symbols_list) > 0:
            _trade_symbol = str(_symbols_list[0]).upper()
        else:
            _trade_symbol = ""
        contract, contract_reason = select_sim_contract_with_reason(direction, price_val, profile, symbol=_trade_symbol)
        if contract is None:
            return {"status": "error", "message": contract_reason or "no_contract"}

        mid = contract.get("mid")
        mid_val = _safe_float(mid)
        if mid_val is None or mid_val <= 0:
            return {"status": "error", "message": "invalid_mid"}

        risk_pct = profile.get("risk_per_trade_pct")
        risk_pct_val = _safe_float(risk_pct)
        if risk_pct_val is None:
            return {"status": "error", "message": "invalid_risk_pct"}

        effective_balance = min(sim.balance, capital_limit_val)
        risk_dollars = effective_balance * risk_pct_val
        if risk_dollars < 50.0:
            risk_dollars = 50.0

        MAX_LIVE_CONTRACTS = 10  # absolute ceiling — never exceed regardless of math
        qty = max(1, math.floor(risk_dollars / (mid_val * 100)))
        if qty > MAX_LIVE_CONTRACTS:
            logging.error(
                "Large qty computed for %s: qty=%d (risk=$%.2f, mid=$%.4f). Capping at %d.",
                contract.get("option_symbol"), qty, risk_dollars, mid_val, MAX_LIVE_CONTRACTS,
            )
            qty = MAX_LIVE_CONTRACTS

        open_exposure = 0.0
        for t in sim.open_trades:
            if isinstance(t, dict):
                try:
                    entry_price = float(t.get("entry_price", 0.0))
                    qty_val = float(t.get("qty", 0.0))
                    open_exposure += entry_price * qty_val * 100
                except (TypeError, ValueError):
                    pass
        est_exposure = mid_val * qty * 100
        if open_exposure + est_exposure > capital_limit_val:
            return {"status": "error", "message": "capital_limit_reached"}

        try:
            from core.singletons import RISK_SUPERVISOR
            notional = mid_val * qty * 100
            _allowed, _deny_reason = RISK_SUPERVISOR.authorize_entry(
                sim_id=sim_id, direction=direction,
                symbol=_trade_symbol, notional=notional,
            )
            if not _allowed:
                logging.info("sim_live_router_blocked_supervisor: %s", _deny_reason)
                return {"status": "error", "message": f"risk_supervisor: {_deny_reason}"}
        except ImportError:
            pass

        fill_result, block = await execute_option_entry(
            contract["option_symbol"],
            qty,
            contract["bid"],
            contract["ask"],
        )
        if fill_result is None:
            return {"status": "error", "message": block or "order_not_filled"}

        trade = _build_live_trade_dict(
            sim_id=sim_id,
            contract=contract,
            fill_result=fill_result,
            qty=qty,
            profile=profile,
            direction=direction,
            regime=regime,
            time_of_day_bucket=time_of_day_bucket,
            signal_mode=signal_mode,
            entry_context=entry_context,
            feature_snapshot=feature_snapshot,
            ml_prediction=ml_prediction,
        )
        sim.record_open(trade)
        sim.save()

        # Log entry candidate for adaptive engine
        try:
            from decision.candidate import Candidate
            from analytics.candidate_logger import log_candidate
            from core.market_clock import get_time_bucket
            log_candidate(Candidate(
                sim_id=sim_id, strategy=signal_mode or "",
                symbol=_trade_symbol, direction=direction, fired=True,
                entry_ref=float(price_val), regime=regime or "",
                time_bucket=get_time_bucket(),
                traded=True, trade_id=trade.get("trade_id"),
            ))
        except Exception:
            pass

        return {
            "status": "success",
            "message": "live_trade_submitted",
            "option_symbol": contract["option_symbol"],
            "qty": qty,
            "fill_price": fill_result.get("fill_price"),
            "risk_dollars": risk_dollars,
            "strike": contract.get("strike"),
            "expiry": contract.get("expiry"),
            "dte": contract.get("dte"),
            "spread_pct": contract.get("spread_pct"),
            "balance_after": sim.balance,
            "entry_context": entry_context,
            "signal_mode": signal_mode,
            "predicted_direction": trade.get("predicted_direction"),
            "prediction_confidence": trade.get("prediction_confidence"),
            "edge_prob": trade.get("edge_prob"),
            "direction_prob": trade.get("direction_prob"),
        }
    except Exception as e:
        logging.exception("sim_live_router_error: %s", e)
        return {"status": "error", "message": str(e)}


async def manage_live_exit(sim, trade):
    try:
        if not isinstance(trade, dict):
            return {"status": "error", "message": "invalid_trade"}

        # Skip if emergency exit is already closing this trade
        _trade_id = trade.get("trade_id", "")
        try:
            from interface.watcher_emergency import is_emergency_closing
            if is_emergency_closing(_trade_id):
                return {"status": "skipped", "message": "emergency_close_in_progress"}
        except ImportError:
            pass

        option_symbol = trade.get("option_symbol")
        if not option_symbol:
            return {"status": "error", "message": "missing_symbol"}

        # Check if option has expired — write off as total loss
        if _is_option_expired(trade):
            logging.error(
                "EXPIRED_OPTION: sim=%s trade=%s symbol=%s — writing off as $0",
                sim.sim_id, trade.get("trade_id"), option_symbol,
            )
            entry_p = float(trade.get("entry_price", 0) or 0)
            qty_v = int(trade.get("qty", 1) or 1)
            pnl_dollars = -(entry_p * qty_v * 100)
            exit_data = {
                "exit_price": 0.0,
                "exit_time": _now_et_iso(),
                "exit_reason": "expired_worthless",
                "exit_context": f"option_symbol={option_symbol}",
                "exit_price_source": "expired",
                "exit_quote_model": "expired",
                "entry_price_source": trade.get("entry_price_source", "live_fill"),
                "time_in_trade_seconds": 0,
                "spread_guard_bypassed": True,
            }
            sim.record_close(trade.get("trade_id"), exit_data)
            sim.save()
            record_sim_trade_close(trade, pnl_dollars)
            return {
                "status": "success",
                "exit_price": 0.0,
                "exit_reason": "expired_worthless",
                "option_symbol": option_symbol,
                "pnl": pnl_dollars,
                "qty": qty_v,
                "entry_price": trade.get("entry_price"),
                "balance_after": sim.balance,
            }

        entry_time = trade.get("entry_time")
        if not entry_time:
            return {"status": "error", "message": "missing_entry_time"}

        eastern = pytz.timezone("US/Eastern")
        now_et = datetime.now(eastern)
        entry_time_et = datetime.fromisoformat(entry_time)
        if entry_time_et.tzinfo is None:
            entry_time_et = eastern.localize(entry_time_et)
        elapsed_seconds = (now_et - entry_time_et).total_seconds()

        # ── Forced exits that bypass hold_min ────────────────────
        force_exit = False
        force_reason = None
        force_context = None
        try:
            expiry_raw = trade.get("expiry")
            if isinstance(expiry_raw, str):
                expiry_date = datetime.fromisoformat(expiry_raw).date()
                if expiry_date == now_et.date() and now_et.time() >= EOD_CUTOFF:
                    force_exit = True
                    force_reason = "expiry_close"
                    force_context = f"expiry={expiry_date.isoformat()} cutoff=15:55"
        except Exception:
            pass

        profile = sim.profile
        is_daytrade = int(profile.get("dte_max", 0)) == 0
        if not force_exit and is_daytrade and now_et.time() >= EOD_CUTOFF:
            force_exit = True
            force_reason = "eod_daytrade_close"
            force_context = "daytrade_cutoff=15:55"

        hold_min_seconds = int(trade.get("hold_min_seconds", 0))
        if not force_exit and elapsed_seconds < hold_min_seconds:
            return {"status": "skipped", "message": "hold_min"}

        current_price = await asyncio.to_thread(get_option_price, option_symbol)
        if current_price is None:
            logging.error("sim_live_exit_missing_price: trade_id=%s symbol=%s", trade.get("trade_id"), option_symbol)
            return {"status": "error", "message": "missing_price"}

        sim.update_open_trade_excursion(trade.get("trade_id"), current_price)

        should_exit = force_exit
        exit_reason = force_reason
        exit_context = force_context

        gain_pct = None
        try:
            entry_price = float(trade.get("entry_price", 0))
            if entry_price > 0:
                gain_pct = (current_price - entry_price) / entry_price
        except (TypeError, ValueError):
            gain_pct = None

        if not should_exit:
            should_exit, exit_reason, exit_context = _determine_exit_condition(
                sim, trade, current_price, gain_pct, elapsed_seconds, profile, now_et
            )

        if not should_exit or not exit_reason:
            return {"status": "skipped", "message": "no_exit"}

        qty = trade.get("qty")
        qty_val = _safe_int(qty)
        if qty_val is None:
            return {"status": "error", "message": "invalid_qty"}
        if qty_val <= 0:
            return {"status": "error", "message": "invalid_qty"}

        close_result = await asyncio.to_thread(close_option_position, option_symbol, qty_val)
        if not close_result.get("ok"):
            logging.error(
                "sim_live_exit_failed: trade_id=%s symbol=%s error=%s order_id=%s",
                trade.get("trade_id"), option_symbol,
                close_result.get("error", "unknown"),
                close_result.get("order_id"),
            )
            return {"status": "error", "message": "exit_failed"}

        filled_avg_price = close_result.get("filled_avg_price")
        exit_price = filled_avg_price if filled_avg_price is not None else current_price
        exit_price_source = "broker_fill" if filled_avg_price is not None else "estimated_mid"

        exit_data = {
            "exit_price": exit_price,
            "exit_time": _now_et_iso(),
            "exit_reason": exit_reason,
            "exit_context": exit_context,
            "exit_price_source": exit_price_source,
            "exit_quote_model": "live_market",
            "entry_price_source": trade.get("entry_price_source", "live_fill"),
            "time_in_trade_seconds": int(elapsed_seconds),
            "spread_guard_bypassed": exit_price_source == "estimated_mid",
            "mae": trade.get("mae_pct"),
            "mfe": trade.get("mfe_pct"),
            "regime_at_entry": trade.get("regime_at_entry"),
            "time_of_day_bucket": trade.get("time_of_day_bucket"),
        }
        sim.record_close(trade.get("trade_id"), exit_data)
        sim.save()
        pnl_val = None
        try:
            entry_price_val = float(trade.get("entry_price", 0))
            if entry_price_val > 0:
                pnl_val = (exit_price - entry_price_val) * qty_val * 100
        except (TypeError, ValueError):
            pnl_val = None
        record_sim_trade_close(trade, pnl_val)

        # Strategy performance store
        try:
            from analytics.strategy_performance import PERF_STORE
            from core.market_clock import get_time_bucket
            _pnl_d = float(pnl_val or 0)
            _entry_f = float(trade.get("entry_price", 0) or 0)
            _qty_f = float(qty_val or 0)
            _cost = _entry_f * _qty_f * 100
            _pnl_pct = _pnl_d / _cost if _cost > 0 else 0.0
            _grade = _trade_grade(trade)
            _grade_str = str(_grade) if _grade is not None else None
            PERF_STORE.record_close(
                strategy=trade.get("signal_mode", ""),
                regime=trade.get("regime_at_entry", "UNKNOWN"),
                time_bucket=trade.get("time_of_day_bucket") or get_time_bucket(),
                pnl=_pnl_d, pnl_pct=_pnl_pct,
                hold_seconds=float(elapsed_seconds),
                grade=_grade_str,
                spread_pct=exit_data.get("spread_guard_bypassed") and None or trade.get("otm_pct"),
            )
        except Exception:
            pass

        return {
            "status": "success",
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "exit_context": exit_context,
            "option_symbol": option_symbol,
            "strike": trade.get("strike"),
            "expiry": trade.get("expiry"),
            "direction": trade.get("direction"),
            "qty": qty_val,
            "entry_price": trade.get("entry_price"),
            "pnl": pnl_val,
            "mae": trade.get("mae_pct"),
            "mfe": trade.get("mfe_pct"),
            "feature_snapshot": trade.get("feature_snapshot"),
            "balance_after": sim.balance,
            "time_in_trade_seconds": int(elapsed_seconds),
            "predicted_direction": trade.get("predicted_direction") or trade.get("direction"),
            "prediction_confidence": trade.get("prediction_confidence"),
            "edge_prob": trade.get("edge_prob"),
            "direction_prob": trade.get("direction_prob"),
        }
    except Exception as e:
        logging.exception("sim_live_exit_exception: %s", e)
        return {"status": "error", "message": str(e)}
