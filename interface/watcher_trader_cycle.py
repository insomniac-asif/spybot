# interface/watcher_trader_cycle.py
#
# Auto-trader inner cycle extracted from watcher_helpers.py.
# Contains: _run_auto_trader_cycle

import asyncio
import logging
from datetime import datetime

import discord
import pytz

from core.debug import debug_log
from interface.watcher_utils import _format_et as _format_et_util
from interface.fmt import ab, A, pct_col


async def _run_auto_trader_cycle(
    bot,
    channel,
    df,
    eastern,
    _runtime_activated: bool,
    _send,
    _format_et,
    explain_block_reason,
    _record_decision_attempt,
    _last_spy_price,
) -> tuple[bool, bool]:
    """
    Execute one auto-trader evaluation cycle.

    Returns (should_continue, new_runtime_activated).
      - should_continue=True  → outer shell should `continue` (skip sleep)
      - should_continue=False → outer shell should proceed to sleep
    """
    from core.data_integrity import validate_market_dataframe
    from core.decision_context import DecisionContext
    from decision.trader import open_trade_if_valid, manage_trade
    from signals.regime import get_regime
    from analytics.setup_expectancy import calculate_setup_expectancy
    from analytics.risk_control import dynamic_risk_percent
    from analytics.conviction_stats import get_conviction_expectancy_stats
    from analytics.signal_logger import log_signal_attempt
    from analytics.blocked_signal_tracker import log_blocked_signal
    from signals.environment_filter import trader_environment_filter
    from interface.watcher_helpers import _build_skip_embed, _build_trade_open_embed, _build_trade_close_embed

    # Strict data integrity precheck before stale-data guard.
    validation = validate_market_dataframe(df)
    if not validation["valid"]:
        now = datetime.now(eastern)
        errors = validation.get("errors", [])
        debug_log("data_integrity_block", errors="; ".join(errors))
        was_invalid = getattr(bot, "data_integrity_state", False)
        last_warn = getattr(bot, "last_integrity_warning_time", None)
        allow_warn = (not was_invalid)
        if last_warn is None:
            allow_warn = True
        elif (now - last_warn).total_seconds() >= 300:
            allow_warn = True
        if allow_warn:
            top_error = errors[0] if errors else "unknown_integrity_error"
            await _send(
                channel,
                "⚠️ **data_integrity_block**\n\n"
                "Market data failed integrity validation.\n"
                f"Primary Reason: {top_error}\n"
                "Trading attempt skipped."
            )
            bot.last_integrity_warning_time = now
        bot.data_integrity_state = True
        return True, _runtime_activated

    bot.data_integrity_state = False

    if not _runtime_activated:
        try:
            from core.runtime_state import RUNTIME, SystemState
            if RUNTIME.transition(SystemState.TRADING_ENABLED, "first_clean_bar"):
                _runtime_activated = True
            elif RUNTIME.state not in {SystemState.BOOTING, SystemState.RECONCILING}:
                _runtime_activated = True  # already past startup
        except ImportError:
            _runtime_activated = True

    # Strict data freshness guard: block if latest candle is older than 2 minutes.
    last_ts = df.index[-1] if len(df.index) > 0 else None
    if last_ts is not None:
        ts = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
        if ts.tzinfo is None:
            ts = eastern.localize(ts)
        else:
            ts = ts.astimezone(eastern)
        now = datetime.now(eastern)
        age_seconds = (now - ts).total_seconds()
        if age_seconds > 300:
            debug_log(
                "data_stale_block",
                candle_time=ts.isoformat(),
                age_seconds=round(age_seconds, 1)
            )
            was_stale = getattr(bot, "data_stale_state", False)
            last_warn = getattr(bot, "last_stale_warning_time", None)
            allow_warn = (not was_stale)
            if last_warn is None:
                allow_warn = True
            elif (now - last_warn).total_seconds() >= 300:
                allow_warn = True
            if getattr(bot, "last_skip_reason", None) == "data_stale":
                allow_warn = False
            if allow_warn:
                stale_embed = discord.Embed(
                    title="⚠️ Market Data Stale — Trading Paused",
                    color=0xE67E22
                )
                stale_embed.add_field(name="⏱️ Data Age", value=ab(A(f"{age_seconds:.0f}s", "red", bold=True)), inline=True)
                stale_embed.add_field(name="⚠️ Threshold", value=ab(A("300s", "yellow", bold=True)), inline=True)
                stale_embed.add_field(name="📋 Action", value=ab(A("Bot will retry once data freshens. No trades will open until feed recovers.", "yellow")), inline=False)
                stale_embed.set_footer(text=f"{_format_et(now)}")
                await _send(channel, embed=stale_embed)
                bot.last_stale_warning_time = now
                bot.last_skip_reason = "data_stale"
                bot.last_skip_time = now
            bot.data_stale_state = True
            return True, _runtime_activated
        bot.data_stale_state = False

    decision_ctx = DecisionContext()
    trade = await open_trade_if_valid(decision_ctx)
    _record_decision_attempt(trade, decision_ctx)

    # --- data collection: log every signal cycle ---
    log_signal_attempt(decision_ctx, trade)
    if decision_ctx.outcome == "blocked":
        _spy_price = df.iloc[-1]["close"] if df is not None and len(df) > 0 else None
        log_blocked_signal(decision_ctx, _spy_price)

    if decision_ctx.outcome == "blocked":
        reason = decision_ctx.block_reason or "unknown"
        now = datetime.now(eastern)
        last_reason = getattr(bot, "last_skip_reason", None)
        if last_reason != reason:
            last_time = getattr(bot, "block_reason_last_time", {}).get(reason)
            if last_time is None or (now - last_time).total_seconds() >= 300:
                friendly_reason = explain_block_reason(reason)
                skip_embed = _build_skip_embed(reason, friendly_reason, decision_ctx, df, now)
                await _send(channel, embed=skip_embed)
                bot.block_reason_last_time[reason] = now
                bot.last_skip_reason = reason
                bot.last_skip_time = now

    if trade == "EQUITY_PROTECTION":
        debug_log("trade_gate", reason="EQUITY_PROTECTION")
        return True, _runtime_activated

    if trade == "EDGE_DECAY":
        debug_log("trade_gate", reason="EDGE_DECAY")
        return True, _runtime_activated

    if trade and isinstance(trade, dict):

        # ----------------------------
        # SETUP EXPECTANCY CHECK
        # ----------------------------
        setup_stats = calculate_setup_expectancy()
        current_setup = trade.get("setup")
        if setup_stats and current_setup in setup_stats:
            setup_avg_R = setup_stats[current_setup]["avg_R"]
            if setup_avg_R < 0:
                debug_log(
                    "trade_blocked",
                    gate="setup_expectancy",
                    setup=current_setup,
                    avg_R=round(setup_avg_R, 3)
                )
                return True, _runtime_activated

        # ----------------------------
        # RISK THROTTLE
        # ----------------------------
        risk_percent = dynamic_risk_percent()
        if risk_percent < 0.01:
            throttle_embed = discord.Embed(
                title="⚠️ Risk Throttled",
                description="System is in drawdown protection mode. Position sizing has been reduced.",
                color=0xF39C12
            )
            throttle_embed.add_field(name="📉 Current Risk/Trade", value=ab(pct_col(risk_percent, good_when_high=False, multiply=True)), inline=True)
            throttle_embed.add_field(name="🛡️ Normal Risk", value=ab(A("1.0%", "green", bold=True)), inline=True)
            throttle_embed.set_footer(text=f"{_format_et(datetime.now(pytz.timezone('America/New_York')))}")
            await _send(channel, embed=throttle_embed)

        # ----------------------------
        # ENVIRONMENT FILTER
        # ----------------------------
        expectancy = get_conviction_expectancy_stats()
        env = trader_environment_filter(
            df,
            trade["type"],
            trade["confidence"],
            expectancy,
            trade.get("regime", get_regime(df))
        )

        if not env["allow"]:
            debug_log(
                "trade_blocked",
                gate="environment_filter",
                adjusted_conf=round(env["adjusted_conf"], 3),
                reasons="; ".join(env["blocks"])
            )
        else:
            debug_log(
                "trade_opened",
                direction=trade["type"],
                entry=round(trade["entry_price"], 2),
                confidence=round(trade["confidence"], 3),
                regime=trade.get("regime")
            )
            open_embed = _build_trade_open_embed(
                trade, decision_ctx, df,
                datetime.now(pytz.timezone("America/New_York"))
            )
            await _send(channel, embed=open_embed)

    result = await asyncio.to_thread(manage_trade)
    if result:
        res, pnl, bal, trade = result
        close_embed = _build_trade_close_embed(
            res, pnl, bal, trade,
            datetime.now(pytz.timezone("America/New_York"))
        )
        await _send(channel, embed=close_embed)

    return False, _runtime_activated
