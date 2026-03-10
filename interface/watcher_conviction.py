# interface/watcher_conviction.py
#
# Conviction watcher cycle helper extracted from watchers.py.
# Contains: _run_conviction_cycle

import logging
from datetime import datetime

import discord
import pytz

from core.data_service import get_symbol_dataframe, _load_symbol_registry
from core.debug import debug_log
from core.md_state import record_md_decay, is_md_enabled, get_md_state, md_needs_warning, evaluate_md_auto
from signals.conviction import calculate_conviction, momentum_is_decaying
from signals.regime import get_regime
from signals.volatility import volatility_state
from analytics.setup_expectancy import calculate_setup_expectancy
from analytics.conviction_stats import log_conviction_signal, update_expectancy
from analytics.blocked_signal_tracker import update_blocked_outcomes
from interface.fmt import ab, A, lbl, dir_col, regime_col, vol_col, tier_col


async def _run_conviction_cycle(
    bot,
    channel,
    df,
    conviction_states: dict,
    last_upgrade_times: dict,
    last_decay_times: dict,
    DISCORD_OWNER_ID: int,
    _format_et,
    _send,
    _MD_TURNOFF_SUGGESTED_DATE,
) -> object:
    """
    Execute one conviction watcher cycle.

    Returns the (possibly updated) _MD_TURNOFF_SUGGESTED_DATE value.
    """
    setup_stats = calculate_setup_expectancy()
    profitable_setups, negative_setups = [], []
    if setup_stats:
        for sn, ss in setup_stats.items():
            if ss["avg_R"] > 0.5:
                profitable_setups.append(sn)
            if ss["avg_R"] < 0:
                negative_setups.append(sn)

    # ── Build per-symbol conviction data — all registry symbols uniformly ──
    try:
        _registry = _load_symbol_registry()
        _registry_syms = list(_registry.items())
    except Exception:
        _registry_syms = []

    # Fall back to the passed-in df as SPY when registry is unavailable
    if not _registry_syms:
        _registry_syms = [("SPY", {})]

    # Fetch all non-SPY dataframes concurrently to avoid blocking the event loop
    import asyncio as _asyncio
    _sym_names = [s.upper() for s, _ in _registry_syms]
    _fetch_tasks = [
        _asyncio.to_thread(get_symbol_dataframe, s) if s != "SPY" else _asyncio.sleep(0, result=df)
        for s in _sym_names
    ]
    _fetched_dfs = await _asyncio.gather(*_fetch_tasks, return_exceptions=True)

    _sym_data = []
    _io_coros = []  # CSV I/O tasks — run concurrently after the CPU loop
    for (_sym_upper, _sym_df_raw) in zip(_sym_names, _fetched_dfs):
        try:
            _sym_df = _sym_df_raw if not isinstance(_sym_df_raw, BaseException) else None
            if _sym_df is None or len(_sym_df) < 30:
                continue
            _s, _i, _f, _d = calculate_conviction(_sym_df)
            _r = get_regime(_sym_df)
            _v = volatility_state(_sym_df)
            _sym_data.append((_sym_upper, _sym_df, _s, _i, _f, _d, _r, _v))
            # Queue CSV I/O for concurrent execution (avoid blocking event loop)
            _io_coros.append(_asyncio.to_thread(log_conviction_signal, _sym_df, _d, _i, _f))
            _io_coros.append(_asyncio.to_thread(update_expectancy, _sym_df))
            if _sym_upper == "SPY":
                _io_coros.append(_asyncio.to_thread(update_blocked_outcomes, _sym_df))
        except Exception:
            pass
    # Run all CSV writes concurrently in one batch
    if _io_coros:
        await _asyncio.gather(*_io_coros, return_exceptions=True)

    now = datetime.now(pytz.timezone("America/New_York"))
    _any_high_decay = False   # any symbol at HIGH tier with decay this cycle
    _pending_upgrades = []    # collect (emoji, tier, prev_tier, sym, sc, imp, fol, dir, reg, vol)

    for _sym, _sym_df, _sc, _imp, _fol, _dir, _reg, _vol in _sym_data:

        # ── Tier calculation ──
        ts = _sc
        if _vol == "HIGH":
            ts += 1
        if _reg == "TREND":
            ts += 1
        if ts >= 6:
            tier = "HIGH";   emoji = "🔥"
        elif ts >= 4:
            tier = "MEDIUM"; emoji = "⚡"
        else:
            tier = "LOW";    emoji = "🟡"

        prev_tier = conviction_states.get(_sym, "LOW")

        # ── Conviction upgrade alert — collect for batch send ──
        if tier != prev_tier and tier in ["MEDIUM", "HIGH"]:
            last_up = last_upgrade_times.get(_sym)
            if last_up is None or (now - last_up).total_seconds() >= 1800:
                _pending_upgrades.append((emoji, tier, prev_tier, _sym, _sc, _imp, _fol, _dir, _reg, _vol))
                last_upgrade_times[_sym] = now
            conviction_states[_sym] = tier

        # ── Momentum decay — HIGH tier only, 20-min cooldown per symbol ──
        decay_detected = momentum_is_decaying(_sym_df)

        if tier == "HIGH" and decay_detected:
            _any_high_decay = True
            last_dec = last_decay_times.get(_sym)
            if last_dec is None or (now - last_dec).total_seconds() >= 1200:
                md_state = record_md_decay(level=tier)
                md_mode = str(md_state.get("mode", "manual")).upper()

                # AUTO mode: state update is enough — suppress per-symbol pings.
                # The system is already handling it; spamming 7+ symbols adds no value.
                if md_mode != "AUTO":
                    md_enabled = bool(md_state.get("enabled"))
                    decay_embed = discord.Embed(
                        title=f"⚠️ [{_sym}] Momentum Decay Detected",
                        description="Impulse is weakening while conviction was elevated. Risk management action recommended.",
                        color=0xE67E22
                    )
                    decay_embed.add_field(name="🧭 Regime", value=ab(regime_col(_reg)), inline=True)
                    decay_embed.add_field(name="⚡ Volatility", value=ab(vol_col(_vol)), inline=True)
                    decay_embed.add_field(name="📊 Conviction Level", value=ab(tier_col(tier)), inline=True)
                    md_text = A("ON", "green", bold=True) if md_enabled else A("OFF", "red", bold=True)
                    md_hint = A("Use `!md enable` to tighten stops.", "yellow") if not md_enabled else A("MD strict mode is active.", "green")
                    decay_embed.add_field(name="🧰 MD Strict", value=ab(f"{md_text}  {md_hint}"), inline=False)
                    decay_embed.add_field(name="💡 Suggested Action", value=ab(A("Tighten stops, reduce size, or stand aside until impulse recovers.", "yellow")), inline=False)
                    decay_embed.set_footer(text=f"{_format_et(now)}")
                    mention = f"<@{DISCORD_OWNER_ID}>" if DISCORD_OWNER_ID else None
                    await _send(channel, mention, embed=decay_embed)

                last_decay_times[_sym] = now
            conviction_states[_sym] = "LOW"

    # ── Evaluate MD auto mode once per cycle across all symbols ──
    evaluate_md_auto(_any_high_decay, "HIGH" if _any_high_decay else None)

    # ── Send batched conviction upgrades as a single embed ──
    if _pending_upgrades:
        has_high = any(u[1] == "HIGH" for u in _pending_upgrades)
        batch_color = 0xFF6B35 if has_high else 0xF39C12
        batch_embed = discord.Embed(
            title=f"Conviction Upgrades — {len(_pending_upgrades)} symbol{'s' if len(_pending_upgrades) != 1 else ''}",
            color=batch_color,
        )
        for _emoji, _tier, _prev, _sym, _sc, _imp, _fol, _dir, _reg, _vol in _pending_upgrades:
            follow_color = "green" if _fol >= 0.5 else "yellow" if _fol >= 0.3 else "red"
            batch_embed.add_field(
                name=f"{_emoji} {_sym} → {_tier}",
                value=ab(
                    A(f"{_dir.upper()} | Score {_sc} | {_imp:.2f}×", "yellow"),
                    A(f"FT {_fol*100:.0f}% | {_reg} | {_vol}", "white"),
                ),
                inline=True,
            )
        if profitable_setups:
            batch_embed.add_field(name="✅ Profitable Setups", value=ab(*[A(f"• {s}", "green") for s in profitable_setups]), inline=False)
        if negative_setups:
            batch_embed.add_field(name="⚠️ Negative Setups", value=ab(*[A(f"• {s}", "red") for s in negative_setups]), inline=False)
        batch_embed.set_footer(text=_format_et(now))
        await _send(channel, embed=batch_embed)

    # ── MD Turn-Off Suggestion (all symbols cleared, once per day) ──
    if not _any_high_decay and md_needs_warning():
        today = now.date()
        if _MD_TURNOFF_SUGGESTED_DATE != today:
            _MD_TURNOFF_SUGGESTED_DATE = today
            clear_embed = discord.Embed(
                title="✅ Momentum Conditions Cleared",
                description="MD strict mode is ON but no decay has been detected recently. Conditions look healthy.",
                color=0x2ECC71,
            )
            clear_embed.add_field(name="💡 Suggested Action", value=ab(A("Use `!md disable` to restore normal trade filtering.", "green")), inline=False)
            clear_embed.set_footer(text=f"{_format_et(now)}")
            mention = f"<@{DISCORD_OWNER_ID}>" if DISCORD_OWNER_ID else None
            await _send(channel, mention, embed=clear_embed)

    return _MD_TURNOFF_SUGGESTED_DATE
