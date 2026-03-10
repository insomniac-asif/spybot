"""
interface/cogs/market_helpers.py
Module-level helper functions for market_commands.py.
Extracted to reduce file size; no logic changes.
"""
import os
import asyncio
import logging
import time as _time

import discord
import matplotlib.pyplot as plt

from interface.fmt import ab, lbl, A
from interface.shared_state import (
    _send_embed,
    _append_footer,
    _add_field_icons,
    _schedule_delete,
    _CHART_TTL,
    _format_ts,
    _format_pct_signed,
    _get_status_banner,
    _add_trend_arrow,
    _tag_trade_mode,
    _safe_float,
)
from core.data_service import get_market_dataframe, get_symbol_csv_path


async def _symbol_snapshot(ctx, symbol: str):
    """Generic symbol snapshot: price, indicators, chart. Used by !spy, !iwm, !tsla, etc."""
    symbol = symbol.upper()
    import pandas_ta as _ta
    try:
        # Load candle data from CSV via registry
        from datetime import date as _date
        import pandas as _pd
        csv_path = get_symbol_csv_path(symbol)
        df = None
        if csv_path and os.path.exists(csv_path):
            df = _pd.read_csv(csv_path)
            df.columns = [c.lower() for c in df.columns]
            ts_col = next((c for c in ("timestamp", "time", "datetime") if c in df.columns), None)
            if ts_col:
                df[ts_col] = _pd.to_datetime(df[ts_col], errors="coerce")
                df = df.dropna(subset=[ts_col])
                if df[ts_col].dt.tz is not None:
                    df[ts_col] = df[ts_col].dt.tz_convert("US/Eastern").dt.tz_localize(None)
                df = df.set_index(ts_col).sort_index()
                # Filter to today only
                today = _pd.Timestamp(_date.today())
                df = df[df.index.date == today.date()]
        # For SPY fall back to get_market_dataframe
        if (df is None or df.empty) and symbol == "SPY":
            df = await asyncio.to_thread(get_market_dataframe)
        if df is None or df.empty:
            await _send_embed(ctx, f"No data for {symbol} — run `!backfill 5 {symbol.lower()}` first.")
            return

        # Compute indicators if missing
        if "ema9" not in df.columns and len(df) >= 9:
            df["ema9"] = df["close"].ewm(span=9).mean()
        if "ema20" not in df.columns and len(df) >= 20:
            df["ema20"] = df["close"].ewm(span=20).mean()
        if "vwap" not in df.columns and "volume" in df.columns:
            try:
                df["vwap"] = _ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            except Exception:
                df["vwap"] = df["close"]

        last = df.iloc[-1]
        recent = df.tail(120)
        high_price = recent["high"].max()
        low_price  = recent["low"].min()
        high_time  = recent["high"].idxmax()
        low_time   = recent["low"].idxmin()
        high_time_str = high_time.strftime('%H:%M') if hasattr(high_time, "strftime") else str(high_time)
        low_time_str  = low_time.strftime('%H:%M')  if hasattr(low_time,  "strftime") else str(low_time)

        # Generate chart (skip if cached copy is < 30 min old)
        os.makedirs("charts", exist_ok=True)
        chart_file = f"charts/snap_{symbol.lower()}.png"
        if not (os.path.exists(chart_file) and _time.time() - os.path.getmtime(chart_file) < _CHART_TTL):
            plt.figure(figsize=(8, 4))
            n = min(120, len(df))
            plt.plot(df.index[-n:], df["close"].iloc[-n:], label="Price")
            if "ema9"  in df.columns: plt.plot(df.index[-n:], df["ema9"].iloc[-n:],  label="EMA9")
            if "ema20" in df.columns: plt.plot(df.index[-n:], df["ema20"].iloc[-n:], label="EMA20")
            if "vwap"  in df.columns: plt.plot(df.index[-n:], df["vwap"].iloc[-n:],  label="VWAP")
            plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
            plt.savefig(chart_file); plt.close()

        def _v(col): return f"${last[col]:.2f}" if col in last.index and last[col] == last[col] else "N/A"
        embed = discord.Embed(title=f"📡 {symbol} Snapshot", color=0x3498DB)
        embed.add_field(name="💰 Price",         value=ab(A(_v("close"), "white", bold=True)), inline=True)
        embed.add_field(name="📊 VWAP",          value=ab(A(_v("vwap"),  "cyan")),             inline=True)
        embed.add_field(name="📈 EMA9",          value=ab(A(_v("ema9"),  "yellow")),           inline=True)
        embed.add_field(name="📉 EMA20",         value=ab(A(_v("ema20"), "yellow")),           inline=True)
        embed.add_field(name="⬆️ Session High",  value=ab(f"{A(f'${high_price:.2f}', 'green')} @ {A(high_time_str, 'white')}"), inline=True)
        embed.add_field(name="⬇️ Session Low",   value=ab(f"{A(f'${low_price:.2f}', 'red')} @ {A(low_time_str, 'white')}"),   inline=True)
        _append_footer(embed)
        await ctx.send(embed=embed)
        if os.path.exists(chart_file):
            await ctx.send(file=discord.File(chart_file))
            asyncio.create_task(_schedule_delete(chart_file))
    except Exception as e:
        logging.exception(f"!{symbol.lower()} snapshot failed: {e}")
        await _send_embed(ctx, f"⚠️ Chart error for {symbol} — data may still be warming up.")


def _safe_money(val, decimals=2):
    try:
        return f"${float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"

def _safe_float_local(val):
    try:
        if val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None

def _safe_r(val):
    try:
        return f"{float(val):.3f}R"
    except (TypeError, ValueError):
        return "N/A"

def _badge_from_pnl(val):
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "⚪"
    if num > 0:
        return "✅"
    if num < 0:
        return "❌"
    return "⚪"

def _format_duration_trades(seconds):
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if total < 0:
        return "N/A"
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


# ── Trades command helpers ────────────────────────────────────────────────────

def _build_trades_embed(
    page_num: int,
    total_pages: int,
    newest_first: list,
    per_page: int,
) -> "discord.Embed":
    page_num = max(1, min(page_num, total_pages))
    start = (page_num - 1) * per_page
    end = start + per_page
    page_trades = newest_first[start:end]

    lines = [f"📒 **Trade Log** (Page {page_num}/{total_pages})"]
    page_pnl = 0.0
    page_pnl_count = 0
    for idx, t in enumerate(page_trades, start=start + 1):
        trade_type = str(t.get("type", "unknown")).upper()
        style = t.get("style", "unknown")
        result = str(t.get("result", "unknown")).upper()
        entry_time = _format_ts(t.get("entry_time", "N/A"))
        exit_time = _format_ts(t.get("exit_time", "N/A"))
        exit_reason = t.get("result_reason") or t.get("exit_reason") or "N/A"

        risk = t.get("risk")
        pnl = t.get("pnl")
        r_mult = t.get("R")
        balance_after = t.get("balance_after")
        hold_text = _format_duration_trades(t.get("time_in_trade_seconds"))
        mode_tag = _tag_trade_mode(t)
        pnl_arrow = _add_trend_arrow(pnl, good_when_high=True)

        risk_text = _safe_money(risk)
        pnl_text = _safe_money(pnl)
        r_text = _safe_r(r_mult)
        bal_text = _safe_money(balance_after)
        badge = _badge_from_pnl(pnl)
        pnl_pct_text = "N/A"
        entry_price = _safe_float_local(t.get("entry_price"))
        exit_price = _safe_float_local(t.get("exit_price"))
        try:
            if entry_price is not None and exit_price is not None and entry_price != 0:
                pnl_pct = (float(exit_price) - float(entry_price)) / float(entry_price)
                pnl_pct_text = _format_pct_signed(pnl_pct)
        except (TypeError, ValueError):
            pnl_pct_text = "N/A"

        snapshot = t.get("decision_snapshot", {})
        delta = snapshot.get("threshold_delta") if isinstance(snapshot, dict) else None
        blended = snapshot.get("blended_score") if isinstance(snapshot, dict) else None
        delta_text = None
        try:
            if delta is not None and blended is not None:
                delta_val = float(delta)
                blended_val = float(blended)
                delta_text = f"Delta: {delta_val:+.4f} | Blended: {blended_val:.4f}"
        except (TypeError, ValueError):
            delta_text = None

        symbol = t.get("option_symbol")
        strike = t.get("strike")
        expiry = t.get("expiry")
        qty = t.get("quantity")

        contract_line = ""
        if symbol:
            contract_line = f"Contract: {symbol}\n"
        elif strike and expiry:
            contract_line = f"{strike} | Exp: {expiry} | Qty: {qty}\n"

        lines.append(
            f"\n{badge} **#{idx}** {trade_type} ({style}) - {result} | {mode_tag}\n"
            f"{contract_line}"
            f"Risk: {risk_text} | PnL: {pnl_text} ({pnl_pct_text}) {pnl_arrow} | R: {r_text}\n"
            f"{delta_text + chr(10) if delta_text else ''}"
            f"Entry: {entry_time}\n"
            f"Exit: {exit_time}\n"
            f"Hold: {hold_text} | Reason: {exit_reason}\n"
            f"Balance After: {bal_text}"
        )

        try:
            pnl_val = float(pnl) if pnl is not None else None
        except (TypeError, ValueError):
            pnl_val = None
        if pnl_val is not None:
            page_pnl += pnl_val
            page_pnl_count += 1

    final_message = "\n".join(lines)
    if page_pnl_count > 0:
        if page_pnl > 0:
            color = 0x2ECC71
        elif page_pnl < 0:
            color = 0xE74C3C
        else:
            color = 0x3498DB
    else:
        color = 0x3498DB

    embed = discord.Embed(
        title=f"📒 Trade Log (Page {page_num}/{total_pages})",
        description=final_message,
        color=color,
    )
    banner = _get_status_banner()
    if banner:
        embed.add_field(name="🧭 Status Banner", value=banner, inline=False)
    _append_footer(embed, extra=f"Page {page_num}/{total_pages}")
    return embed


async def _send_trades_paginated(
    ctx, bot, page_num: int, total_pages: int, newest_first: list, per_page: int
) -> None:
    page_num = max(1, min(page_num, total_pages))
    message = await ctx.send(
        embed=_build_trades_embed(page_num, total_pages, newest_first, per_page)
    )
    if total_pages <= 1:
        return
    try:
        for emoji in ("◀️", "▶️", "⏹️"):
            await message.add_reaction(emoji)
    except Exception:
        return

    def _check(reaction, user):
        return (
            user == ctx.author
            and reaction.message.id == message.id
            and str(reaction.emoji) in {"◀️", "▶️", "⏹️"}
        )

    while True:
        try:
            reaction, user = await bot.wait_for("reaction_add", timeout=60, check=_check)
        except asyncio.TimeoutError:
            try:
                await message.clear_reactions()
            except Exception:
                pass
            break

        emoji = str(reaction.emoji)
        if emoji == "⏹️":
            try:
                await message.clear_reactions()
            except Exception:
                pass
            break
        if emoji == "◀️":
            page_num = total_pages if page_num == 1 else page_num - 1
        elif emoji == "▶️":
            page_num = 1 if page_num == total_pages else page_num + 1

        try:
            await message.edit(
                embed=_build_trades_embed(page_num, total_pages, newest_first, per_page)
            )
        except Exception:
            pass
        try:
            await message.remove_reaction(reaction.emoji, user)
        except Exception:
            pass
