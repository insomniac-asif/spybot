# decision/trader.py
#
# ── LEGACY — PAPER AUTO-TRADER ONLY ──────────────────────────────────────────
# This module manages the paper auto_trader account (account.json).
# It is NOT the live trading path. Live trades go through:
#   simulation/sim_live_router.py → sim_live_router()
# This file should not be extended with new features. New strategy logic
# belongs in simulation/sim_engine.py and simulation/sim_signals.py.
# ─────────────────────────────────────────────────────────────────────────────
from core.md_state import is_md_enabled

from datetime import datetime
import pytz
import os
import uuid
import logging

from core.account_repository import (
    load_account,
    save_account,
)

from core.data_service import (
    get_latest_price,
    get_market_dataframe
)
from core.market_clock import market_is_open
from core.debug import debug_log
from core.decision_context import DecisionContext
from core.paths import DATA_DIR
from core.rate_limiter import rate_limit_sleep

from signals.predictor import make_prediction
from signals.regime import get_regime
from signals.volatility import volatility_state
from signals.setup_classifier import classify_trade
from signals.conviction import calculate_conviction

from execution.option_executor import execute_option_entry, get_option_price


# ── Extracted helpers ─────────────────────────────────────────────────────────
# All utility functions live in trader_utils; the filter function in
# trader_filters.  They are imported here so that callers can still do:
#   from decision.trader import get_ml_visibility_snapshot
# etc. without any change.
from decision.trader_utils import (
    _reset_daily_stats,
    _get_daily_trade_stats,
    _emit_daily_summary,
    _roll_daily_summary_if_needed,
    _record_signal_attempt,
    _record_gate_block,
    _category_for_block_reason,
    get_ml_visibility_snapshot,
    _track_confidence_distribution,
    load_models,
    _model_is_fresh,
    _feature_trade_count,
    maybe_retrain_models,
    build_ml_features,
    can_day_trade,
    select_style,
    _daily_stats,
)
from decision.trader_filters import apply_ml_and_edge_filters

# ── Extracted contract helpers ────────────────────────────────────────────────
from decision.trader_contracts import (
    _get_option_client,
    _parse_option_symbol,
    _select_option_contract,
    build_execution_plan,
)

# ── Extracted signal helpers ──────────────────────────────────────────────────
from decision.trader_signal import (
    pre_trade_checks,
    generate_signal,
    create_trade_object,
)

# ── Extracted exit helpers ────────────────────────────────────────────────────
from decision.trader_exit import (
    check_expectancy_exit,
    check_partial_logic,
    check_exit_conditions,
    calculate_pnl,
    finalize_trade,
    _finalize_reconstructed_trade,
    _manage_reconstructed_trades,
    _manage_reconstructed_advanced,
)


ALPACA_MIN_CALL_INTERVAL_SEC = float(os.getenv("ALPACA_MIN_CALL_INTERVAL_SEC", "0.5"))

MAX_OPEN_TRADES = 3


# =========================
# OPEN TRADE ENGINE
# =========================

async def open_trade_if_valid(ctx=None):
    if ctx is None:
        ctx = DecisionContext()

    try:
        from core.runtime_state import RUNTIME
        if not RUNTIME.can_enter():
            _rs = f"runtime_{RUNTIME.state.value.lower()}"
            ctx.set_block(_rs)
            return _rs
    except ImportError:
        pass

    _record_signal_attempt()

    acc = load_account()
    eastern = pytz.timezone("US/Eastern")
    now_eastern = datetime.now(eastern)
    _roll_daily_summary_if_needed(acc, now_eastern)

    # ----------------------------
    # 1️⃣ Pre-Trade Protection Layer
    # ----------------------------
    protection = pre_trade_checks(acc, ctx)
    if protection is not None:
        if ctx.block_reason is None:
            ctx.set_block(f"protection_{protection}")
        category = _category_for_block_reason(ctx.block_reason)
        if category:
            _record_gate_block(category)
        return protection

    # ----------------------------
    # 2️⃣ Signal Generation Layer
    # ----------------------------
    signal = generate_signal(acc, ctx)
    if signal is None:
        if ctx.block_reason is None:
            ctx.set_block("signal_none")
        category = _category_for_block_reason(ctx.block_reason)
        if category:
            _record_gate_block(category)
        return None

    df, regime, vol_state, direction, confidence, score, impulse, follow, price, setup_type = signal

    # ----------------------------
    # 3️⃣ ML + Edge Filtering Layer
    # ----------------------------
    allow, blended_score, style = apply_ml_and_edge_filters(
        acc,
        df,
        regime,
        vol_state,
        direction,
        confidence,
        score,
        impulse,
        follow,
        setup_type,
        ctx
    )

    if not allow:
        category = _category_for_block_reason(ctx.block_reason)
        if category:
            _record_gate_block(category)
        return None

    # ----------------------------
    # 4️⃣ Execution Plan Layer
    # ----------------------------
    execution = build_execution_plan(
        acc,
        df,
        regime,
        vol_state,
        direction,
        style,
        price,
        setup_type
    )

    if execution is None:
        ctx.set_block("execution_plan_none")
        return None
    if isinstance(execution, dict) and execution.get("block_reason"):
        ctx.set_block(execution["block_reason"])
        return None
    if not isinstance(execution, tuple) or len(execution) != 4:
        ctx.set_block("execution_plan_none")
        return None

    risk_dollars, trade_size, option, target_R = execution
    try:
        risk_dollars = float(risk_dollars)
        trade_size = int(trade_size)
        target_R = float(target_R)
    except (TypeError, ValueError):
        ctx.set_block("execution_plan_none")
        return None
    if not isinstance(option, dict):
        ctx.set_block("execution_plan_none")
        return None

    virtual_cap = acc.get("virtual_capital_limit", acc["balance"])
    open_trades = acc.get("open_trades", [])
    if not isinstance(open_trades, list):
        open_trades = []

    open_trades_updated = False
    for t in open_trades:
        if isinstance(t, dict) and t.get("trade_id") is None:
            t["trade_id"] = str(uuid.uuid4())
            open_trades_updated = True
    if open_trades_updated:
        acc["open_trades"] = open_trades
        save_account(acc)

    unique_ids = {
        t.get("trade_id")
        for t in open_trades
        if isinstance(t, dict) and t.get("trade_id") is not None
    }

    if len(unique_ids) >= MAX_OPEN_TRADES:
        ctx.set_block("max_open_trades_reached")
        return None
    unique_risk = {}
    for t in open_trades:
        if isinstance(t, dict):
            trade_id = t.get("trade_id")
            if trade_id:
                unique_risk[trade_id] = float(t.get("risk", 0))

    open_trade = acc.get("open_trade")
    if isinstance(open_trade, dict):
        trade_id = open_trade.get("trade_id")
        if trade_id and trade_id not in unique_risk:
            unique_risk[trade_id] = float(open_trade.get("risk", 0))

    total_open_risk = sum(unique_risk.values())
    if total_open_risk + risk_dollars > virtual_cap:
        ctx.set_block("capital_exposure_limit")
        return None

    option_symbol = option.get("symbol") if option else None
    bid = option.get("bid") if option else None
    ask = option.get("ask") if option else None
    if not option_symbol or bid is None or ask is None:
        ctx.set_block("execution_plan_none")
        return None
    try:
        bid = float(bid)
        ask = float(ask)
    except (TypeError, ValueError):
        ctx.set_block("execution_plan_none")
        return None
    if ask <= 0 or bid < 0:
        ctx.set_block("execution_plan_none")
        return None
    spread = ask - bid
    if spread < 0:
        ctx.set_block("execution_plan_none")
        return None
    spread_pct = spread / ask
    if spread_pct > 0.15:
        ctx.set_block("spread_too_wide")
        return None

    fill_result, exec_block = await execute_option_entry(option_symbol, trade_size, bid, ask, ctx=ctx, acc=acc)
    if fill_result is None:
        ctx.set_block(exec_block or "limit_not_filled")
        return None
    fill_price = fill_result.get("fill_price")
    filled_qty = fill_result.get("filled_qty")
    requested_qty = fill_result.get("requested_qty")
    fill_ratio = fill_result.get("fill_ratio")
    if fill_price is None or filled_qty is None or requested_qty is None:
        ctx.set_block("limit_not_filled")
        return None
    try:
        filled_qty = int(filled_qty)
        requested_qty = int(requested_qty)
        fill_ratio = float(fill_ratio) if fill_ratio is not None else None
    except (TypeError, ValueError):
        ctx.set_block("limit_not_filled")
        return None
    if filled_qty <= 0:
        ctx.set_block("limit_not_filled")
        return None
    if requested_qty <= 0:
        ctx.set_block("limit_not_filled")
        return None
    if fill_ratio is None:
        fill_ratio = filled_qty / requested_qty
    if fill_ratio < 0.5:
        ctx.set_block("partial_fill_below_threshold")
        return None
    if filled_qty < requested_qty:
        trade_size = filled_qty
        risk_dollars = risk_dollars * fill_ratio

    stop_loss_frac = 0.5
    if is_md_enabled():
        stop_loss_frac = 0.35
    stop = fill_price - (fill_price * stop_loss_frac)
    risk_per_contract = fill_price - stop              # option price units (per share)
    target = fill_price + (risk_per_contract * target_R)
    risk = trade_size * risk_per_contract * 100        # dollars: qty × $/share × 100 shares/contract

    # ----------------------------
    # 5️⃣ Create Trade Object
    # ----------------------------
    trade = create_trade_object(
        direction,
        style,
        fill_price,
        stop,
        target,
        risk,
        trade_size,
        confidence,
        regime,
        vol_state,
        score,
        impulse,
        follow,
        setup_type,
        blended_score,
        ctx
    )
    if option:
        import re as _re_und
        _opt_sym = option.get("symbol") or ""
        _und_match = _re_und.match(r'^([A-Z]{1,6})', _opt_sym)
        trade["underlying"] = _und_match.group(1) if _und_match else "SPY"
        trade["symbol"] = trade["underlying"]
        trade["option_symbol"] = option.get("symbol")
        trade["strike"] = option.get("strike")
        trade["expiry"] = option.get("expiry")
        trade["quantity"] = trade_size
        trade["entry_price"] = fill_price
        trade["stop"] = stop
        trade["initial_stop"] = stop
        trade["target"] = target
        trade["stop_price"] = stop
        trade["target_price"] = target

    acc["open_trade"] = trade
    save_account(acc)
    _daily_stats["trades_opened"] += 1
    ctx.set_opened()
    debug_log(
        "trade_opened",
        direction=trade["type"],
        entry=round(trade["entry_price"], 2),
        confidence=round(trade["confidence"], 3),
        blended=trade.get("ml_probability")
    )

    return trade


# =========================
# TRADE MANAGEMENT ENGINE
# =========================
# NOTE: build_execution_plan lives in decision.trader_contracts (imported above).
# NOTE: _finalize_reconstructed_trade, _manage_reconstructed_trades, and
#       _manage_reconstructed_advanced live in decision.trader_exit (imported above).


def manage_trade():

    acc = load_account()

    recon_result = _manage_reconstructed_trades(acc)
    if recon_result:
        return recon_result

    advanced_result = _manage_reconstructed_advanced(acc)
    if advanced_result:
        return advanced_result

    if acc["open_trade"] is None:
        return None

    trade = acc["open_trade"]

    # Use the live option price for all stop/target comparisons.
    # stop and target are expressed in option-price units, so comparing
    # them to the underlying stock price (get_latest_price) is wrong.
    option_symbol = trade.get("option_symbol") if isinstance(trade, dict) else None
    if not option_symbol:
        return None
    price = get_option_price(option_symbol)
    if price is None:
        return None

    # Expiry handling: close same-day expiry positions 5 minutes before close
    try:
        eastern = pytz.timezone("US/Eastern")
        now_et = datetime.now(eastern)
        expiry_raw = trade.get("expiry")
        expiry_date = None
        if isinstance(expiry_raw, str):
            expiry_date = datetime.fromisoformat(expiry_raw).date()
        if expiry_date == now_et.date() and now_et.time() >= datetime.strptime("15:55", "%H:%M").time():
            result = "win" if price >= trade["entry_price"] else "loss"
            pnl = calculate_pnl(trade, result, price)
            return finalize_trade(acc, trade, "expiry_close", pnl)
    except Exception:
        pass

    # 1️⃣ Expectancy Protection
    expectancy_exit = check_expectancy_exit(acc, trade, price)
    if expectancy_exit:
        return expectancy_exit

    # 2️⃣ Partial Logic
    partial = check_partial_logic(acc, trade, price)
    if partial is not None:
        return None

    # 3️⃣ Hard Exit Conditions
    exit_result = check_exit_conditions(trade, price)
    if exit_result is None:
        return None

    result = exit_result

    # 4️⃣ Calculate PnL
    pnl = calculate_pnl(trade, result, price)

    # 5️⃣ Finalize Trade
    return finalize_trade(acc, trade, result, pnl)
