"""Sim commands cog -- 18 commands migrated from bot.py."""
import logging
import discord
from discord.ext import commands

from interface.fmt import ab, lbl, A, pnl_col, wr_col, exit_reason_col, balance_col, drawdown_col
from interface.shared_state import (
    _send_embed, _append_footer, _add_field_icons, _format_ts, _format_pct_signed,
    _format_duration_short, _get_data_freshness_text, _load_sim_profiles,
    _collect_sim_metrics, _safe_float, _tag_trade_mode, STRATEGY_INTENTS,
)
from simulation.sim_watcher import get_sim_last_skip_state
from interface.cogs.sim_helpers import (
    _format_money, _format_signed_money, _format_drawdown, _format_pct,
    _pnl_badge, _format_duration, _extract_reason, _format_feature_snapshot,
    _sim_path, _compute_breakdown, _ansi_breakdown, _parse_page,
    _gate_parts, _paginate, _add_lines_field,
)
from interface.cogs.sim_helpers2 import (
    handle_simstats, handle_simcompare, handle_simleaderboard, handle_simreport,
    handle_simdte, handle_simsetups, handle_simtrades, handle_simopen,
    handle_simreset, handle_simhealth, handle_siminfo,
)


class SimCommands(commands.Cog, name="Sims"):
    def __init__(self, bot): self.bot = bot

    # ── simstats ──────────────────────────────────────────────────────────
    @commands.command(name="simstats")
    async def simstats(self, ctx, sim_id: str | None = None):
        await handle_simstats(ctx, sim_id)

    # ── simcompare ────────────────────────────────────────────────────────
    @commands.command(name="simcompare")
    async def simcompare(self, ctx):
        await handle_simcompare(ctx)

    # ── simleaderboard ────────────────────────────────────────────────────
    @commands.command(name="simleaderboard")
    async def simleaderboard(self, ctx):
        await handle_simleaderboard(ctx)

    # ── simreport ─────────────────────────────────────────────────────────
    @commands.command(name="simreport")
    async def simreport(self, ctx, sim_id: str | None = None):
        if not sim_id:
            await ctx.send("Usage: `!simreport <SIM_ID>` e.g. `!simreport SIM05`")
            return
        await handle_simreport(ctx, sim_id.upper())

    # ── simstreaks ────────────────────────────────────────────────────────
    @commands.command(name="simstreaks")
    async def simstreaks(self, ctx):
        try:
            metrics, profiles = _collect_sim_metrics()
            if not profiles: await _send_embed(ctx, "No sim profiles found."); return
            if not metrics: await _send_embed(ctx, "No sim data available yet."); return
            eligible = [m for m in metrics if m.get("trades", 0) >= 3]
            if not eligible: await _send_embed(ctx, "Not enough trades to rank streaks (need 3+)."); return
            win_rank = sorted(eligible, key=lambda m: m.get("max_win_streak", 0), reverse=True)[:5]
            loss_rank = sorted(eligible, key=lambda m: m.get("max_loss_streak", 0), reverse=True)[:5]
            embed = discord.Embed(title="\U0001f501 Sim Streaks", color=0x9B59B6)
            wl = [f"{A(m['sim_id'], 'cyan', bold=True)} {A(str(m.get('max_win_streak', 0)), 'green', bold=True)} {lbl('WR')} {wr_col(m.get('win_rate', 0))}" for m in win_rank]
            ll = [f"{A(m['sim_id'], 'cyan', bold=True)} {A(str(m.get('max_loss_streak', 0)), 'red', bold=True)} {lbl('WR')} {wr_col(m.get('win_rate', 0))}" for m in loss_rank]
            embed.add_field(name="\u2705 Longest Win Streaks", value=ab(*wl) if wl else ab(A("N/A","gray")), inline=False)
            embed.add_field(name="\u274c Longest Loss Streaks", value=ab(*ll) if ll else ab(A("N/A","gray")), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("simstreaks_error")
            await _send_embed(ctx, "simstreaks failed due to an internal error.")

    # ── simregimes ────────────────────────────────────────────────────────
    @commands.command(name="simregimes")
    async def simregimes(self, ctx):
        try:
            metrics, profiles = _collect_sim_metrics()
            if not profiles: await _send_embed(ctx, "No sim profiles found."); return
            if not metrics: await _send_embed(ctx, "No sim data available yet."); return
            lines = []
            for reg in ["TREND", "RANGE", "VOLATILE", "UNKNOWN"]:
                best = None; best_wr = -1.0
                for m in metrics:
                    stats = m.get("regime_stats", {}).get(reg)
                    if not stats or stats.get("total", 0) < 3: continue
                    wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0.0
                    if wr > best_wr: best_wr = wr; best = (m["sim_id"], stats["wins"], stats["total"])
                if best:
                    sid, w, t = best
                    lines.append(f"{A(reg, 'cyan')} {A(sid, 'white', bold=True)} {A(f'{w}/{t}', 'white')} {wr_col(w/t)}")
            embed = discord.Embed(title="\U0001f9ed Best by Regime (Win Rate)", color=0x1ABC9C)
            embed.add_field(name="Regime Leaders", value=ab(*lines) if lines else ab(A("No regime data","gray")), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("simregimes_error")
            await _send_embed(ctx, "simregimes failed due to an internal error.")

    # ── simtimeofday ──────────────────────────────────────────────────────
    @commands.command(name="simtimeofday")
    async def simtimeofday(self, ctx):
        try:
            metrics, profiles = _collect_sim_metrics()
            if not profiles: await _send_embed(ctx, "No sim profiles found."); return
            if not metrics: await _send_embed(ctx, "No sim data available yet."); return
            lines = []
            for bucket in ["MORNING", "MIDDAY", "AFTERNOON", "CLOSE", "UNKNOWN"]:
                best = None; best_wr = -1.0
                for m in metrics:
                    stats = m.get("time_stats", {}).get(bucket)
                    if not stats or stats.get("total", 0) < 3: continue
                    wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0.0
                    if wr > best_wr: best_wr = wr; best = (m["sim_id"], stats["wins"], stats["total"])
                if best:
                    sid, w, t = best
                    lines.append(f"{A(bucket, 'cyan')} {A(sid, 'white', bold=True)} {A(f'{w}/{t}', 'white')} {wr_col(w/t)}")
            embed = discord.Embed(title="\U0001f552 Best by Time\u2011of\u2011Day (Win Rate)", color=0x2980B9)
            embed.add_field(name="Time\u2011of\u2011Day Leaders", value=ab(*lines) if lines else ab(A("No time\u2011bucket data","gray")), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("simtimeofday_error")
            await _send_embed(ctx, "simtimeofday failed due to an internal error.")

    # ── simpf ─────────────────────────────────────────────────────────────
    @commands.command(name="simpf")
    async def simpf(self, ctx):
        try:
            metrics, profiles = _collect_sim_metrics()
            if not profiles: await _send_embed(ctx, "No sim profiles found."); return
            eligible = [m for m in metrics if m.get("trades", 0) >= 3 and m.get("profit_factor") is not None]
            if not eligible: await _send_embed(ctx, "Not enough data for profit factor (need 3+ trades)."); return
            ranked = sorted(eligible, key=lambda m: m.get("profit_factor", 0), reverse=True)[:7]
            lines = []
            for m in ranked:
                pf = m.get("profit_factor")
                lines.append(f"{A(m['sim_id'], 'cyan', bold=True)} {lbl('PF')} {A(f'{pf:.2f}x', 'green' if pf >= 1 else 'red', bold=True)} {lbl('WR')} {wr_col(m.get('win_rate', 0))} {lbl('PnL')} {pnl_col(m.get('total_pnl'))}")
            embed = discord.Embed(title="\U0001f9ee Profit Factor Leaders", color=0x16A085)
            embed.add_field(name="Top Profit Factors", value=ab(*lines), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("simpf_error")
            await _send_embed(ctx, "simpf failed due to an internal error.")

    # ── simconsistency ────────────────────────────────────────────────────
    @commands.command(name="simconsistency")
    async def simconsistency(self, ctx):
        try:
            metrics, profiles = _collect_sim_metrics()
            if not profiles: await _send_embed(ctx, "No sim profiles found."); return
            eligible = [m for m in metrics if m.get("trades", 0) >= 3 and m.get("pnl_stdev") is not None]
            if not eligible: await _send_embed(ctx, "Not enough data for consistency (need 3+ trades)."); return
            ranked = sorted(eligible, key=lambda m: m.get("pnl_stdev", 0))[:7]
            lines = [f"{A(m['sim_id'], 'cyan', bold=True)} {lbl('\u03c3')} {pnl_col(m.get('pnl_stdev'))} {lbl('Median')} {pnl_col(m.get('pnl_median'))} {lbl('WR')} {wr_col(m.get('win_rate', 0))}" for m in ranked]
            embed = discord.Embed(title="\U0001f4cf Most Consistent Sims", color=0x8E44AD)
            embed.add_field(name="Lowest PnL Volatility", value=ab(*lines), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("simconsistency_error")
            await _send_embed(ctx, "simconsistency failed due to an internal error.")

    # ── simexits ──────────────────────────────────────────────────────────
    @commands.command(name="simexits")
    async def simexits(self, ctx):
        try:
            metrics, profiles = _collect_sim_metrics()
            if not profiles: await _send_embed(ctx, "No sim profiles found."); return
            if not metrics: await _send_embed(ctx, "No sim data available yet."); return
            lines = []
            for reason in ["profit_target", "trailing_stop", "stop_loss", "eod_daytrade_close", "hold_max_elapsed"]:
                best = None; best_rate = -1.0
                for m in metrics:
                    total = m.get("trades", 0)
                    if total < 3: continue
                    count = m.get("exit_counts", {}).get(reason, 0)
                    rate = count / total if total > 0 else 0.0
                    if rate > best_rate: best_rate = rate; best = (m["sim_id"], count, total)
                if best:
                    sid, c, t = best
                    lines.append(f"{A(reason, 'cyan')} {A(sid, 'white', bold=True)} {A(f'{c}/{t}', 'white')} {wr_col(c/t)}")
            embed = discord.Embed(title="\U0001f3af Best Exit Hit Rates", color=0xF39C12)
            embed.add_field(name="Exit Reason Leaders", value=ab(*lines) if lines else ab(A("No exit data","gray")), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("simexits_error")
            await _send_embed(ctx, "simexits failed due to an internal error.")

    # ── simhold ───────────────────────────────────────────────────────────
    @commands.command(name="simhold")
    async def simhold(self, ctx):
        try:
            metrics, profiles = _collect_sim_metrics()
            if not profiles: await _send_embed(ctx, "No sim profiles found."); return
            eligible = [m for m in metrics if m.get("trades", 0) >= 3 and m.get("avg_hold") is not None]
            if not eligible: await _send_embed(ctx, "Not enough data for hold\u2011time stats (need 3+ trades)."); return
            fastest = sorted(eligible, key=lambda m: m.get("avg_hold", 0))[:5]
            slowest = sorted(eligible, key=lambda m: m.get("avg_hold", 0), reverse=True)[:5]
            fl = [f"{A(m['sim_id'], 'cyan', bold=True)} {lbl('Avg Hold')} {A(_format_duration_short(m.get('avg_hold')), 'white')} {lbl('WR')} {wr_col(m.get('win_rate', 0))}" for m in fastest]
            sl = [f"{A(m['sim_id'], 'cyan', bold=True)} {lbl('Avg Hold')} {A(_format_duration_short(m.get('avg_hold')), 'white')} {lbl('WR')} {wr_col(m.get('win_rate', 0))}" for m in slowest]
            embed = discord.Embed(title="\u23f1 Sim Hold\u2011Time Leaders", color=0x2C3E50)
            embed.add_field(name="Fastest Average Holds", value=ab(*fl), inline=False)
            embed.add_field(name="Slowest Average Holds", value=ab(*sl), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("simhold_error")
            await _send_embed(ctx, "simhold failed due to an internal error.")

    # ── simdte ────────────────────────────────────────────────────────────
    @commands.command(name="simdte")
    async def simdte(self, ctx):
        await handle_simdte(ctx)

    # ── simsetups ─────────────────────────────────────────────────────────
    @commands.command(name="simsetups")
    async def simsetups(self, ctx):
        await handle_simsetups(ctx)

    # ── simtrades ─────────────────────────────────────────────────────────
    @commands.command(name="simtrades")
    async def simtrades(self, ctx, sim_id: str | None = None, page: str | int = 1):
        await handle_simtrades(ctx, self.bot, sim_id, page)

    # ── simopen ───────────────────────────────────────────────────────────
    @commands.command(name="simopen")
    async def simopen(self, ctx, sim_id: str | None = None, page: str | int = 1):
        await handle_simopen(ctx, self.bot, sim_id, page)

    # ── simreset ──────────────────────────────────────────────────────────
    @commands.command(name="simreset")
    async def simreset(self, ctx, sim_id: str | None = None):
        await handle_simreset(ctx, sim_id)

    # ── simhealth ─────────────────────────────────────────────────────────
    @commands.command(name="simhealth")
    async def simhealth(self, ctx, page: str | int | None = None):
        await handle_simhealth(ctx, self.bot, page)

    # ── siminfo ───────────────────────────────────────────────────────────
    @commands.command(name="siminfo")
    async def siminfo(self, ctx, sim_id: str | int | None = None):
        await handle_siminfo(ctx, sim_id)

    # ── simanalyze ────────────────────────────────────────────────────────
    @commands.command(name="simanalyze")
    async def simanalyze(self, ctx, sim_id: str | None = None, action: str | None = None):
        try:
            from simulation.trade_analyzer import (
                generate_entry_filters, analyze_sim_trades, analyze_all_sims,
                apply_filters_to_config,
            )

            if not sim_id:
                await _send_embed(ctx, "Usage: `!simanalyze <SIM_ID>` | `!simanalyze all` | `!simanalyze <SIM_ID> apply` | `!simanalyze <SIM_ID> reset`")
                return

            if sim_id.lower() == "all":
                # Batch summary across all sims
                all_results = analyze_all_sims()
                lines = []
                for sid in sorted(all_results.keys()):
                    r = all_results[sid]
                    if r.get("insufficient_data"):
                        lines.append(f"{A(sid, 'cyan', bold=True)} {A('< 15 trades', 'gray')}")
                        continue
                    wr = r.get("overall_win_rate", 0)
                    filters = r.get("filters", {})
                    filter_names = ", ".join(filters.keys()) if filters else "none"
                    impact = r.get("projected_impact", {})
                    reduction = impact.get("trade_reduction_pct")
                    imp_str = f" -{reduction*100:.0f}%" if reduction else ""
                    lines.append(
                        f"{A(sid, 'cyan', bold=True)} {lbl('WR')} {wr_col(wr)}"
                        f" {lbl('Filters')} {A(filter_names, 'white')}{A(imp_str, 'yellow')}"
                    )
                embed = discord.Embed(title="Sim Analyzer — All Sims", color=0x3498DB)
                # Paginate if needed (Discord field limit 1024 chars)
                chunk = lines[:15]
                embed.add_field(
                    name="Filter Recommendations",
                    value=ab(*chunk) if chunk else ab(A("No data", "gray")),
                    inline=False,
                )
                _append_footer(embed)
                await ctx.send(embed=embed)
                return

            sid = sim_id.upper()

            if action and action.lower() == "reset":
                ok = apply_filters_to_config(sid, {}, dry_run=False)
                msg = f"Quality filters removed for {sid}." if ok else f"Failed to reset {sid}."
                await _send_embed(ctx, msg)
                return

            result = generate_entry_filters(sid)
            total = result.get("total_trades", 0)
            analysis = result.get("analysis", {})

            embed = discord.Embed(
                title=f"Sim Analyzer — {sid}",
                description=f"{total} closed trades",
                color=0x3498DB,
            )

            if result.get("insufficient_data"):
                embed.add_field(
                    name="Status",
                    value=ab(A(result.get("message", "Insufficient data"), "yellow")),
                    inline=False,
                )
                _append_footer(embed)
                await ctx.send(embed=embed)
                return

            overall_wr = result.get("overall_win_rate", 0)

            # Grade distribution
            grade_dist = analysis.get("grade_dist", {})
            dist_parts = []
            for g in ["A", "B", "C", "D", "F"]:
                count = grade_dist.get(g, 0)
                col = "green" if g in ("A", "B") else ("yellow" if g == "C" else "red")
                dist_parts.append(f"{A(g, col, bold=True)}{A(f':{count}', 'white')}")
            embed.add_field(
                name=_add_field_icons("Grade Distribution"),
                value=ab(" ".join(dist_parts)) if dist_parts else ab(A("N/A", "gray")),
                inline=False,
            )

            # Grade stats (win rate per grade)
            grade_stats = analysis.get("grade_stats", {})
            stat_lines = []
            for g in ["A", "B", "C", "D", "F"]:
                s = grade_stats.get(g)
                if s and s.get("count", 0) > 0:
                    col = "green" if g in ("A", "B") else ("yellow" if g == "C" else "red")
                    stat_lines.append(
                        f"{A(g, col, bold=True)} {lbl('WR')} {wr_col(s['win_rate'])}"
                        f" {lbl('avg')} {pnl_col(s['avg_pnl'])}"
                    )
            if stat_lines:
                embed.add_field(
                    name=_add_field_icons("Win Rate by Grade"),
                    value=ab(*stat_lines),
                    inline=False,
                )

            # Worst dimensions
            worst = analysis.get("worst_dimensions", [])
            if worst:
                wdim_lines = [
                    f"{A(dim, 'yellow')} {A(f'{score:.0f}/100', 'white')}"
                    for dim, score in worst
                ]
                embed.add_field(
                    name=_add_field_icons("Weakest Dimensions"),
                    value=ab(*wdim_lines),
                    inline=False,
                )

            # Recommended filters
            filters = result.get("filters", {})
            if filters:
                filter_lines = []
                for k, v in filters.items():
                    filter_lines.append(f"{A(k, 'cyan')} {A(str(v), 'white')}")
                impact = result.get("projected_impact", {})
                if impact:
                    kept = impact.get("kept_count", 0)
                    orig = impact.get("original_count", total)
                    proj_wr = impact.get("projected_win_rate", overall_wr)
                    proj_exp = impact.get("projected_expectancy", 0)
                    filter_lines.append(
                        f"{lbl('Impact')} {A(f'{kept}/{orig} trades', 'white')}"
                        f" {lbl('WR')} {wr_col(proj_wr)}"
                        f" {lbl('Exp')} {pnl_col(proj_exp)}"
                    )
                embed.add_field(
                    name=_add_field_icons("Recommended Filters"),
                    value=ab(*filter_lines),
                    inline=False,
                )
            else:
                embed.add_field(
                    name=_add_field_icons("Filters"),
                    value=ab(A("No filters recommended (need more data or no strong patterns found)", "gray")),
                    inline=False,
                )

            embed.add_field(
                name=_add_field_icons("Overall Win Rate"),
                value=ab(wr_col(overall_wr)),
                inline=True,
            )

            if action and action.lower() == "apply":
                if filters:
                    def _check(reaction, user):
                        return user == ctx.author and str(reaction.emoji) in ("✅", "❌")

                    confirm_embed = discord.Embed(
                        title=f"Apply filters to {sid}?",
                        description="\n".join(f"`{k}: {v}`" for k, v in filters.items()),
                        color=0xF39C12,
                    )
                    confirm_embed.set_footer(text="React ✅ to apply, ❌ to cancel")
                    msg = await ctx.send(embed=confirm_embed)
                    await msg.add_reaction("✅")
                    await msg.add_reaction("❌")
                    import asyncio
                    try:
                        reaction, _ = await ctx.bot.wait_for("reaction_add", timeout=30.0, check=_check)
                        if str(reaction.emoji) == "✅":
                            ok = apply_filters_to_config(sid, filters, dry_run=False)
                            status_msg = f"Filters applied to {sid} in sim_config.yaml." if ok else "Failed to write config."
                            await ctx.send(status_msg)
                        else:
                            await ctx.send("Cancelled.")
                    except asyncio.TimeoutError:
                        await ctx.send("Timed out — no changes made.")
                else:
                    await ctx.send(f"No filters to apply for {sid}.")
                return

            _append_footer(embed)
            await ctx.send(embed=embed)
        except Exception:
            logging.exception("simanalyze_error")
            await _send_embed(ctx, "simanalyze failed due to an internal error.")

    # ── lastskip ──────────────────────────────────────────────────────────
    @commands.command(name="lastskip")
    async def lastskip(self, ctx):
        try:
            reason = getattr(self.bot, "last_skip_reason", None)
            ts = getattr(self.bot, "last_skip_time", None)
            ts_text = _format_ts(ts) if ts is not None else "N/A"
            sim_state = get_sim_last_skip_state()
            sim_lines = []
            for sid in sorted(sim_state.keys()):
                item = sim_state.get(sid, {})
                sr = item.get("reason") or "N/A"
                st = item.get("time")
                st_text = _format_ts(st) if st is not None else "N/A"
                sim_lines.append(f"{sid}: {sr} ({st_text})")
            sim_text = "\n".join(sim_lines) if sim_lines else "None"
            embed = discord.Embed(title="\u23f8 Last Skip Reasons", color=0xF39C12)
            trader_text = f"{reason} ({ts_text})" if reason else "None"
            embed.add_field(name=_add_field_icons("Trader"), value=ab(A(trader_text, "yellow")), inline=False)
            embed.add_field(name=_add_field_icons("Sims"), value=ab(A(sim_text, "yellow")), inline=False)
            _append_footer(embed); await ctx.send(embed=embed)
        except Exception:
            logging.exception("lastskip_error")
            await _send_embed(ctx, "lastskip failed due to an internal error.")


async def setup(bot):
    await bot.add_cog(SimCommands(bot))
