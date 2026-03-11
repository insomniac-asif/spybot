"""
interface/cogs/admin_commands.py
Discord commands for bot administration and operational maintenance.

Commands: help, helpplan, system, ratelimit, backfill, query,
          conviction_fix, features_reset, pred_reset, retrain
"""

import os
import csv
import asyncio
import logging

import discord
from discord.ext import commands

from interface.shared_state import (
    _send_embed,
    _append_footer,
    _add_field_icons,
    _format_ts,
)
from core.paths import DATA_DIR
from interface.cogs.admin_helpers import (
    COMMAND_LEVELS,
    COMMAND_GUIDES,
    build_help_page,
)
from interface.cogs.admin_embed_helpers import _build_system_status_embed


class AdminCommands(commands.Cog, name="Admin"):
    """Bot administration and maintenance commands."""

    def __init__(self, bot):
        self.bot = bot

    # ── help ──────────────────────────────────────────────────────────────────

    @commands.command(name="help")
    async def help_command(self, ctx, command_name: str | int | None = None):

        async def _send_help_paginated(start_page: int):
            pages_count = 3
            page = max(1, min(start_page, pages_count))
            message = await ctx.send(embed=build_help_page(page))
            if pages_count <= 1:
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
                    reaction, user = await self.bot.wait_for("reaction_add", timeout=60, check=_check)
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
                    page = pages_count if page == 1 else page - 1
                elif emoji == "▶️":
                    page = 1 if page == pages_count else page + 1

                try:
                    await message.edit(embed=build_help_page(page))
                except Exception:
                    pass
                try:
                    await message.remove_reaction(reaction.emoji, user)
                except Exception:
                    pass

        if command_name is None:
            await _send_help_paginated(1)
            return
        if isinstance(command_name, int):
            await _send_help_paginated(command_name)
            return
        if isinstance(command_name, str):
            page_text = command_name.strip().lower()
            if page_text.startswith("page"):
                page_text = page_text.replace("page", "").strip()
            if page_text.isdigit():
                await _send_help_paginated(int(page_text))
                return

        if not isinstance(command_name, str):
            await _send_embed(ctx, "Command name must be text.")
            return

        command_name = command_name.lower()

        if command_name in COMMAND_GUIDES:
            await _send_embed(ctx, COMMAND_GUIDES[command_name], title=f"!{command_name}")
        else:
            await _send_embed(
                ctx,
                "Unknown command.\n"
                "Type `!help` to view available commands.\n"
                "Type `!help <command>` for detailed usage."
            )

    # ── helpplan ──────────────────────────────────────────────────────────────

    @commands.command(name="helpplan")
    async def help_plan(self, ctx):
        await _send_embed(ctx, """
📋 **!plan Usage**

!plan <call/put> <strike> <premium> <contracts> <expiry YYYY-MM-DD>

Example:
!plan call 435 1.20 2 2026-02-14

Analyzes:
• Market structure
• Volatility
• Statistical alignment
• AI model grade (A–D)
""", title="Help")

    # ── system ────────────────────────────────────────────────────────────────

    @commands.command(name="system")
    async def system(self, ctx):
        import pytz
        from datetime import datetime
        from core.account_repository import load_account
        from interface.health_monitor import check_health
        from core.market_clock import market_is_open

        eastern = pytz.timezone("US/Eastern")
        now = datetime.now(eastern)

        try:
            acc = load_account()
        except Exception:
            acc = {}
        total_trades = len(acc.get("trade_log", []))

        try:
            status, report = check_health()
        except Exception:
            status = "UNKNOWN"
            report = "Health monitor unavailable."

        market_status = "🟢 OPEN" if market_is_open() else "🔴 CLOSED"
        embed = _build_system_status_embed(acc, status, report, market_status, now, total_trades)
        await ctx.send(embed=embed)

    # ── ratelimit ─────────────────────────────────────────────────────────────

    @commands.command(name="ratelimit")
    async def ratelimit(self, ctx):
        """Show API rate-limiter / circuit-breaker / cache stats."""
        from core.api_resilience import resilience_stats

        stats = resilience_stats()
        bucket = stats["bucket"]
        cache = stats["cache"]
        breaker = stats["breaker"]

        cb_state = breaker["state"]
        cb_color = 0x00CC44 if cb_state == "closed" else (0xFF8800 if cb_state == "half_open" else 0xFF3333)

        age_str = ""
        if breaker.get("opened_age_seconds") is not None:
            age_str = f" (open {breaker['opened_age_seconds']}s ago)"

        embed = discord.Embed(title="API Resilience Status", color=cb_color)
        embed.add_field(
            name="Token Bucket",
            value=(
                f"Available: **{bucket['tokens_available']}** / {int(bucket['max_tokens'])}\n"
                f"Refill rate: {bucket['refill_rate']}/s"
            ),
            inline=True,
        )
        embed.add_field(
            name="Response Cache",
            value=(
                f"Valid entries: **{cache['cached_entries']}**\n"
                f"Total slots: {cache['total_entries']}"
            ),
            inline=True,
        )
        cb_icon = "🟢" if cb_state == "closed" else ("🟡" if cb_state == "half_open" else "🔴")
        embed.add_field(
            name="Circuit Breaker",
            value=(
                f"{cb_icon} **{cb_state.upper()}**{age_str}\n"
                f"Failures: {breaker['failures']} / {breaker['threshold']}"
            ),
            inline=False,
        )
        _append_footer(embed)
        await ctx.send(embed=embed)

    # ── backfill ──────────────────────────────────────────────────────────────

    @commands.command(name="backfill")
    async def backfill(self, ctx, days: int = 30, symbol: str = "SPY"):
        """
        Backfill historical 1m candles from Alpaca.
        Usage: !backfill [days] [SPY|QQQ|all]
        Examples: !backfill 30 SPY  |  !backfill 7 QQQ  |  !backfill 30 all
        """
        from core.backfill import (
            run_backfill_async,
            run_backfill_all_symbols_async,
            backfill_status,
            _load_registered_symbols as _backfill_get_symbols,
        )

        if days < 1 or days > 365:
            await _send_embed(ctx, "Days must be between 1 and 365.")
            return

        backfill_all = symbol.lower() == "all"
        sym = symbol.upper()

        if backfill_all:
            registry = _backfill_get_symbols()
            sym_list = list(registry.keys()) if registry else ["SPY"]
            status_lines = []
            for s in sym_list:
                st = backfill_status(s)
                status_lines.append(f"**{s}**: {st['rows']:,} rows ({st['latest'] or 'no data'})")
            embed = discord.Embed(
                title=f"Backfilling {days} days — ALL symbols ({len(sym_list)})",
                description="\n".join(status_lines) + "\n\nRunning in background…",
                color=0x4444FF,
            )
        else:
            status = backfill_status(sym)
            embed = discord.Embed(
                title=f"Backfilling {days} days — {sym}",
                description=(
                    f"Current: **{status['rows']:,}** rows\n"
                    f"Earliest: {status['earliest'] or 'N/A'}\n"
                    f"Latest: {status['latest'] or 'N/A'}\n\n"
                    "Running in background — this may take a minute."
                ),
                color=0x4444FF,
            )
        _append_footer(embed)
        await ctx.send(embed=embed)

        messages: list[str] = []
        def progress(msg: str):
            messages.append(msg)

        if backfill_all:
            result = await run_backfill_all_symbols_async(days_back=days, progress_cb=progress)
        else:
            result = await run_backfill_async(days_back=days, symbol=sym, progress_cb=progress)

        if result.get("ok"):
            if backfill_all:
                lines = []
                for s, r in result.get("results", {}).items():
                    if r.get("ok"):
                        lines.append(f"**{s}**: +{r['added_rows']:,} rows (total {r['total_rows']:,}, {r['errors']} err)")
                    else:
                        lines.append(f"**{s}**: ❌ {r.get('error','failed')}")
                embed2 = discord.Embed(
                    title="Backfill Complete — All Symbols",
                    description="\n".join(lines) if lines else "No new rows added.",
                    color=0x00CC44,
                )
                embed2.add_field(
                    name="Totals",
                    value=f"Rows added: **{result['total_added']:,}** | Day-errors: {result['total_errors']}",
                    inline=False,
                )
            else:
                embed2 = discord.Embed(title=f"Backfill Complete — {sym}", color=0x00CC44)
                embed2.add_field(
                    name="Results",
                    value=(
                        f"Days fetched: **{result['fetched_days']}**\n"
                        f"Rows added: **{result['added_rows']:,}**\n"
                        f"Total rows: **{result['total_rows']:,}**\n"
                        f"Day-errors: {result['errors']}"
                    ),
                    inline=False,
                )
        else:
            embed2 = discord.Embed(
                title="Backfill Failed",
                description=result.get("error", "unknown error"),
                color=0xFF3333,
            )
        _append_footer(embed2)
        await ctx.send(embed=embed2)

    # ── query ─────────────────────────────────────────────────────────────────

    @commands.command(name="query")
    async def query(self, ctx, sim_id: str = None, page: int = 1):
        """Query trade journal DB. Usage: !query [SIM03] [page]"""
        from core.trade_db import query_trades, trade_count

        per_page = 10
        offset = (max(1, page) - 1) * per_page
        sid = sim_id.upper() if sim_id else None

        trades = query_trades(sim_id=sid, limit=per_page, offset=offset)
        total = trade_count(sim_id=sid)
        total_pages = max(1, (total + per_page - 1) // per_page)

        title = f"Trade Journal — {sid or 'ALL SIMS'} (page {page}/{total_pages})"
        embed = discord.Embed(title=title, color=0x4444AA)
        embed.description = f"Total records: **{total:,}**"

        if not trades:
            embed.description += "\nNo trades found."
        else:
            lines = []
            for t in trades:
                pnl = t.get("realized_pnl_dollars")
                pnl_str = f"${pnl:+.2f}" if pnl is not None else "?"
                sym = t.get("option_symbol") or "?"
                xt = (t.get("exit_time") or "")[:16]
                reason = t.get("exit_reason") or "?"
                sim = t.get("sim_id") or "?"
                lines.append(f"`{sim}` {sym} | {pnl_str} | {reason} | {xt}")
            embed.add_field(name="Trades", value="\n".join(lines), inline=False)

        if total_pages > 1:
            embed.set_footer(text=f"Use !query {sim_id or ''} {page + 1} for next page")
        else:
            _append_footer(embed)
        await ctx.send(embed=embed)

    # ── conviction_fix ────────────────────────────────────────────────────────

    @commands.command(name="conviction_fix")
    async def conviction_fix(self, ctx):
        try:
            from core.data_service import get_market_dataframe
            from analytics.conviction_stats import update_expectancy

            import asyncio as _asyncio
            df = await _asyncio.to_thread(get_market_dataframe)
            update_expectancy(df)
            await _send_embed(
                ctx,
                "Conviction expectancy backfill complete.\n"
                "fwd_5m / fwd_10m now updated where possible with price/time metadata.",
                title="Conviction Fix",
                color=0x2ecc71,
            )
        except Exception as e:
            logging.exception("conviction_fix_error: %s", e)
            await _send_embed(
                ctx,
                f"Conviction fix failed: {e}",
                title="Conviction Fix",
                color=0xe74c3c,
            )

    # ── features_reset ────────────────────────────────────────────────────────

    @commands.command(name="features_reset")
    async def features_reset(self, ctx):
        try:
            from analytics.feature_logger import FEATURE_FILE, FEATURE_HEADERS

            with open(FEATURE_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(FEATURE_HEADERS)
            await _send_embed(
                ctx,
                "trade_features.csv reset to clean header.",
                title="Features Reset",
                color=0x2ecc71,
            )
        except Exception as e:
            logging.exception("features_reset_error: %s", e)
            await _send_embed(
                ctx,
                f"Features reset failed: {e}",
                title="Features Reset",
                color=0xe74c3c,
            )

    # ── pred_reset ────────────────────────────────────────────────────────────

    @commands.command(name="pred_reset")
    async def pred_reset(self, ctx):
        try:
            from analytics.prediction_stats import PRED_FILE, PRED_HEADERS

            with open(PRED_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(PRED_HEADERS)
            await _send_embed(
                ctx,
                "predictions.csv reset to clean header.",
                title="Predictions Reset",
                color=0x2ecc71,
            )
        except Exception as e:
            logging.exception("pred_reset_error: %s", e)
            await _send_embed(
                ctx,
                f"Predictions reset failed: {e}",
                title="Predictions Reset",
                color=0xe74c3c,
            )

    # ── journal ─────────────────────────────────────────────────────────────

    @commands.command(name="journal")
    async def journal(self, ctx, date_str: str | None = None):
        """Daily trade journal. Usage: !journal [YYYY-MM-DD]"""
        import io
        from analytics.trade_journal import build_journal_summary, save_daily_journal

        try:
            path = await asyncio.to_thread(save_daily_journal, date_str)
            summary = await asyncio.to_thread(build_journal_summary, date_str)

            greeks_text = summary.get("greeks_exits", "0")
            embed = discord.Embed(
                title=f"Trade Journal -- {summary['date']}",
                color=0x3498DB,
            )
            embed.description = (
                f"**{summary['entries']}** entries | **{summary['exits']}** exits | "
                f"**{summary['win_rate']:.1f}%** WR\n"
                f"Net P&L: **${summary['net_pnl']:+.2f}**\n"
                f"Best: {summary['best_trade']}\n"
                f"Worst: {summary['worst_trade']}\n"
                f"Greeks exits: {greeks_text}\n\n"
                f"Full journal attached below"
            )
            _append_footer(embed)
            await ctx.send(embed=embed)

            # Attach the full markdown file
            with open(path, "rb") as f:
                file = discord.File(f, filename=os.path.basename(path))
                await ctx.send(file=file)

        except Exception as e:
            logging.exception("journal_error: %s", e)
            await _send_embed(ctx, f"Journal generation failed: {e}", color=0xFF3333)

    # ── weekjournal ────────────────────────────────────────────────────────

    @commands.command(name="weekjournal")
    async def weekjournal(self, ctx):
        """Weekly trade digest (last 5 trading days)."""
        import io
        from analytics.trade_journal import generate_weekly_digest

        try:
            digest = await asyncio.to_thread(generate_weekly_digest)
            # Send as file since it's too long for embed
            buf = io.BytesIO(digest.encode("utf-8"))
            buf.seek(0)
            file = discord.File(buf, filename="weekly_digest.md")
            embed = discord.Embed(
                title="Weekly Trade Digest",
                description="Full digest attached below.",
                color=0x3498DB,
            )
            _append_footer(embed)
            await ctx.send(embed=embed, file=file)
        except Exception as e:
            logging.exception("weekjournal_error: %s", e)
            await _send_embed(ctx, f"Weekly digest failed: {e}", color=0xFF3333)

    # ── retrain ───────────────────────────────────────────────────────────────

    @commands.command(name="retrain")
    async def retrain(self, ctx):

        feature_file = os.path.join(DATA_DIR, "trade_features.csv")

        if not os.path.exists(feature_file):
            await _send_embed(
                ctx,
                "No trade feature data found.\n"
                "You need at least 50 logged trades before retraining."
            )
            return

        await _send_embed(ctx, "🔄 Retraining models...")

        from research.train_ai import train_direction_model, train_edge_model
        train_direction_model()
        train_edge_model()

        await _send_embed(ctx, "✅ Models retrained successfully.")


async def setup(bot):
    await bot.add_cog(AdminCommands(bot))
