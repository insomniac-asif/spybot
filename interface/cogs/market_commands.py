"""
interface/cogs/market_commands.py
Market analysis, symbol snapshots, and account performance commands.

Commands:
  Symbol snapshots: !spy !qqq !iwm !vxx !tsla !aapl !nvda !msft !quote
  Market analysis:  !risk !regime !regimes !conviction !opportunity !predict
                    !plan !mlstats !importance !expectancy !replay !preopen !md
  Account perf:     !run !paperstats !career !equity !accuracy !analysis !attempts !trades
"""

import os
import asyncio
import logging

import discord
from discord.ext import commands

from interface.fmt import (
    ab, lbl, A, pnl_col, drawdown_col, wr_col,
)
from interface.shared_state import (
    _send_embed,
    _append_footer,
    _add_field_icons,
    _format_ts,
    _format_pct_signed,
    _get_data_freshness_text,
    _get_status_line,
    _schedule_delete,
    _safe_float,
)

from core.data_service import get_market_dataframe
from core.account_repository import load_account

from signals.conviction import calculate_conviction
from signals.opportunity import evaluate_opportunity
from signals.regime import get_regime

from analytics.prediction_stats import calculate_accuracy
from analytics.run_stats import get_run_stats
from analytics.performance import get_paper_stats, get_career_stats
from analytics.equity_curve import generate_equity_curve
from analytics.risk_metrics import calculate_r_metrics, calculate_drawdown
from analytics.regime_expectancy import calculate_regime_expectancy
from analytics.expectancy import calculate_expectancy
from analytics.feature_importance import get_feature_importance
from analytics.ml_accuracy import ml_rolling_accuracy
from analytics.decision_analysis import analyze_decision_quality

from interface.charting import generate_chart, generate_live_chart
from interface.watchers import get_decision_buffer_snapshot
from interface.cogs.market_helpers import (
    _symbol_snapshot,
    _safe_money,
    _safe_float_local,
    _safe_r,
    _badge_from_pnl,
    _format_duration_trades,
    _build_trades_embed,
    _send_trades_paginated,
)
from interface.cogs.market_helpers2 import (
    handle_md,
    handle_predict,
    handle_plan,
    handle_preopen,
)


class MarketCommands(commands.Cog, name="Market"):
    """Symbol snapshots, market analysis, and account performance commands."""

    def __init__(self, bot):
        self.bot = bot

    # ── Symbol Snapshot Commands ──────────────────────────────────────────────

    @commands.command()
    async def spy(self, ctx):
        await _symbol_snapshot(ctx,"SPY")

    @commands.command()
    async def qqq(self, ctx):
        await _symbol_snapshot(ctx,"QQQ")

    @commands.command()
    async def iwm(self, ctx):
        await _symbol_snapshot(ctx,"IWM")

    @commands.command()
    async def vxx(self, ctx):
        await _symbol_snapshot(ctx,"VXX")

    @commands.command()
    async def tsla(self, ctx):
        await _symbol_snapshot(ctx,"TSLA")

    @commands.command()
    async def aapl(self, ctx):
        await _symbol_snapshot(ctx,"AAPL")

    @commands.command()
    async def nvda(self, ctx):
        await _symbol_snapshot(ctx,"NVDA")

    @commands.command()
    async def msft(self, ctx):
        await _symbol_snapshot(ctx,"MSFT")

    @commands.command(name="quote")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def quote(self, ctx, symbol: str | None = None):
        """Generic: !quote <SYMBOL>"""
        if not symbol:
            await _send_embed(ctx, "Usage: `!quote <SYMBOL>` — e.g. `!quote tsla`")
            return
        await _symbol_snapshot(ctx,symbol)

    # ── Market Analysis Commands ──────────────────────────────────────────────

    @commands.command()
    async def risk(self, ctx):
        r = calculate_r_metrics()
        dd = calculate_drawdown()

        if not r:
            await _send_embed(ctx, "⚠️ Need at least 10 closed trades to calculate R metrics.\nClose more trades before evaluating performance.")
            return

        if not dd:
            await _send_embed(ctx, "⚠️ Drawdown metrics unavailable.\nClose at least 1 trade to calculate drawdown.")
            return

        risk_embed = discord.Embed(title="📊 Risk Metrics", color=0x3498DB)
        risk_embed.add_field(name="📊 Avg R",       value=ab(A(str(r['avg_R']), "yellow", bold=True)), inline=True)
        risk_embed.add_field(name="✅ Avg Win R",   value=ab(A(str(r['avg_win_R']), "green", bold=True)), inline=True)
        risk_embed.add_field(name="❌ Avg Loss R",  value=ab(A(str(r['avg_loss_R']), "red", bold=True)), inline=True)
        risk_embed.add_field(name="🏆 Max R",       value=ab(A(str(r['max_R']), "green")), inline=True)
        risk_embed.add_field(name="💀 Min R",       value=ab(A(str(r['min_R']), "red")), inline=True)
        risk_embed.add_field(name="📉 Max Drawdown", value=ab(drawdown_col(dd['max_drawdown_dollars'])), inline=True)
        _append_footer(risk_embed)
        await ctx.send(embed=risk_embed)

    @commands.command()
    async def md(self, ctx, action: str | None = None, level: str | None = None):
        await handle_md(ctx, action, level)

    @commands.command()
    async def expectancy(self, ctx):
        stats = calculate_expectancy()

        if not stats:
            await _send_embed(ctx, "Need at least 10 closed trades to calculate expectancy.")
            return

        exp_embed = discord.Embed(title="📊 Rolling Expectancy", color=0x3498DB)
        exp_embed.add_field(name="📊 Avg R",      value=ab(A(str(stats['avg_R']), "yellow", bold=True)), inline=True)
        exp_embed.add_field(name="🎯 Win Rate",   value=ab(wr_col(stats['winrate'] / 100.0)), inline=True)
        exp_embed.add_field(name="💰 Expectancy", value=ab(pnl_col(stats['expectancy'])), inline=True)
        exp_embed.add_field(name="📦 Trades",     value=ab(A(str(stats['samples']), "white")), inline=True)
        _append_footer(exp_embed)
        await ctx.send(embed=exp_embed)

    @commands.command(name="replay")
    async def replay(self, ctx, symbol: str | None = None):
        sym = symbol.upper() if symbol else None
        try:
            chart_path = generate_chart(sym)

            if chart_path:
                await _send_embed(ctx, f"📊 {sym} Recorded Session:")
                await ctx.send(file=discord.File(chart_path))
                asyncio.create_task(_schedule_delete(chart_path))
            else:
                await _send_embed(ctx, f"No recorded session data for {sym} — run `!backfill 5 {sym.lower()}` first.")

            live_path = generate_live_chart(sym)

            if live_path and os.path.exists(live_path):
                await _send_embed(ctx, f"📈 {sym} Live Market:")
                await ctx.send(file=discord.File(live_path))
                asyncio.create_task(_schedule_delete(live_path))
            else:
                await _send_embed(ctx, f"Market closed or no live data for {sym}.")

        except Exception as e:
            logging.exception(f"Replay error: {e}")
            await _send_embed(ctx, "Replay failed — check logs.")

    @commands.command()
    async def importance(self, ctx):
        data = get_feature_importance()

        if not data:
            await _send_embed(ctx, "Model not trained yet.\nRun `!retrain` after 50+ trades.")
            return

        message = "📊 Feature Importance (Edge Model)\n\n"

        for name, score in data:
            message += f"{name}: {round(score, 3)}\n"

        await _send_embed(ctx, message, title="Feature Importance")

    @commands.command()
    async def regime(self, ctx):
        import asyncio as _asyncio
        df = await _asyncio.to_thread(get_market_dataframe)
        if df is None:
            await _send_embed(ctx, "No data.")
            return
        await _send_embed(ctx, f"Market Regime: {get_regime(df)}", title="Regime")

    @commands.command()
    async def regimes(self, ctx):
        data = calculate_regime_expectancy()

        if not data:
            await _send_embed(ctx, "Need at least 10 closed trades to calculate meaningful expextancy metrics.")
            return

        message = "📊 Regime Expectancy (R-Multiple)\n\n"

        for regimes, stats in data.items():
            message += (
                f"{regimes}\n"
                f"Trades: {stats['trades']}\n"
                f"Avg R: {stats['avg_R']}\n"
                f"Winrate: {stats['winrate']}%\n\n"
            )

        await _send_embed(ctx, message, title="Regime Expectancy")

    @commands.command()
    async def conviction(self, ctx):
        import asyncio as _asyncio
        df = await _asyncio.to_thread(get_market_dataframe)
        if df is None:
            await _send_embed(ctx, "No data.")
            return

        score, impulse, follow, direction = calculate_conviction(df)

        await _send_embed(
            ctx,
            (
                f"Conviction Score: {score}\n"
                f"Direction: {direction}\n"
                f"Impulse: {impulse:.2f}\n"
                f"Follow: {follow*100:.0f}%"
            ),
            title="Conviction"
        )

    @commands.command()
    async def opportunity(self, ctx):
        import asyncio as _asyncio
        df = await _asyncio.to_thread(get_market_dataframe)
        if df is None:
            await _send_embed(ctx, "No data.")
            return

        result = evaluate_opportunity(df)

        if result:
            if not isinstance(result, (list, tuple)) or len(result) < 5:
                await _send_embed(ctx, "No opportunity right now.")
                return
            side = result[0]
            low = result[1]
            high = result[2]
            price = result[3]
            tp_low = tp_high = stop_loss = None
            if len(result) >= 8:
                tp_low = result[5]
                tp_high = result[6]
                stop_loss = result[7]
            lines = [
                f"{side} setup",
                f"Zone: {low:.2f}-{high:.2f}",
                f"Current: ${price:.2f}",
            ]
            if tp_low is not None and tp_high is not None:
                lines.append(f"Take-Profit: {tp_low:.2f}-{tp_high:.2f}")
            if stop_loss is not None:
                lines.append(f"Stop-Loss: {stop_loss:.2f}")
            await _send_embed(ctx, "\n".join(lines), title="Opportunity")
        else:
            await _send_embed(ctx, "No opportunity right now.")

    @commands.command()
    async def predict(self, ctx, minutes: str | None = None):
        await handle_predict(ctx, minutes)

    @commands.command()
    async def plan(self, ctx, side=None, strike=None, premium=None, contracts=None, expiry=None):
        await handle_plan(ctx, side, strike, premium, contracts, expiry)

    @commands.command()
    async def mlstats(self, ctx):
        acc = ml_rolling_accuracy()

        if acc is None:
            await _send_embed(ctx, "Not enough ML trade data.\nNeed at least 30 ML-evaluated trades.")
        else:
            await _send_embed(ctx, f"🧠 ML Rolling Accuracy (Last 30 Trades): {acc}%")

    @commands.command(name="preopen")
    async def preopen(self, ctx):
        await handle_preopen(ctx)

    # ── Account Performance Commands ──────────────────────────────────────────

    @commands.command()
    async def run(self, ctx):
        s = get_run_stats()
        await _send_embed(
            ctx,
            (
                f"Trades: {s['trades']}\n"
                f"Wins: {s['wins']}\n"
                f"Losses: {s['losses']}\n"
                f"Balance: ${s['current']}"
            ),
            title="Run Stats"
        )

    @commands.command()
    async def paperstats(self, ctx):
        s = get_paper_stats()
        status_line = _get_status_line()
        freshness_line = _get_data_freshness_text()
        pnl_badge = "⚪"
        try:
            pnl_val = float(s["pnl"])
            if pnl_val > 0:
                pnl_badge = "✅"
            elif pnl_val < 0:
                pnl_badge = "❌"
        except (TypeError, ValueError, KeyError):
            pnl_badge = "⚪"
        lines = [
            f"💰 Balance: ${s['balance']:.2f}",
            f"📈 PnL: {pnl_badge} ${s['pnl']:.2f}",
            f"🎯 Winrate: {s['winrate']}%",
        ]
        if status_line:
            lines.append(status_line)
        if freshness_line:
            lines.append(freshness_line)
        await _send_embed(ctx, "\n".join(lines), title="Paper Stats")

    @commands.command()
    async def career(self, ctx):
        c = get_career_stats()
        status_line = _get_status_line()
        freshness_line = _get_data_freshness_text()
        win_badge = "⚪"
        try:
            winrate_val = float(c["winrate"])
            if winrate_val >= 50:
                win_badge = "✅"
            else:
                win_badge = "❌"
        except (TypeError, ValueError, KeyError):
            win_badge = "⚪"
        lines = [
            f"📦 Total Trades: {c['total_trades']}",
            f"🎯 Winrate: {win_badge} {c['winrate']}%",
            f"🏆 Best Balance: ${c['best_balance']:.2f}",
        ]
        if status_line:
            lines.append(status_line)
        if freshness_line:
            lines.append(freshness_line)
        await _send_embed(ctx, "\n".join(lines), title="Career Stats")

    @commands.command()
    async def equity(self, ctx):
        chart = generate_equity_curve()
        if chart:
            embed = discord.Embed(title="📈 Equity Curve", color=0x3498DB)
            _append_footer(embed)
            await ctx.send(embed=embed, file=discord.File(chart))
        else:
            await _send_embed(ctx, "Need at least 1 closed trade(s) to see equity.")

    @commands.command()
    async def accuracy(self, ctx):
        stats = calculate_accuracy()
        if stats:
            await _send_embed(ctx, str(stats), title="Accuracy")
        else:
            await _send_embed(ctx, "Need at least 5 graded predictions before accuracy can be calculated.")

    @commands.command()
    async def analysis(self, ctx):
        try:
            acc = load_account()
        except Exception:
            await _send_embed(ctx, "Could not load account data.")
            return

        results = analyze_decision_quality(acc.get("trade_log", []))
        total = results.get("total_trades_analyzed", 0)
        corr_delta = results.get("corr_threshold_delta_vs_R")
        corr_blended = results.get("corr_blended_vs_R")

        corr_delta_text = "N/A" if corr_delta is None else f"{corr_delta:.4f}"
        corr_blended_text = "N/A" if corr_blended is None else f"{corr_blended:.4f}"

        lines = [
            "📊 Decision Analysis Summary",
            f"📦 Trades Analyzed: {total}",
            f"📈 Corr Delta vs R: {corr_delta_text}",
            f"🧠 Corr Blended vs R: {corr_blended_text}",
        ]

        exec_stats = acc.get("execution_stats")
        no_record = exec_stats.get("no_record_exits") if isinstance(exec_stats, dict) else None
        if isinstance(no_record, dict) and no_record:
            no_record_lines = []
            for key in sorted(no_record.keys()):
                no_record_lines.append(f"{key}: {no_record.get(key)}")
            stats_text = "\n".join(no_record_lines)
            lines.append("Execution No-Record Exits:")
            lines.append(stats_text)

        status_line = _get_status_line()
        freshness_line = _get_data_freshness_text()
        if status_line:
            lines.append(status_line)
        if freshness_line:
            lines.append(freshness_line)

        await _send_embed(ctx, "\n".join(lines)[:1500], title="Analysis")

    @commands.command()
    async def attempts(self, ctx):
        snap = get_decision_buffer_snapshot()
        avg_delta = snap.get("avg_delta")
        avg_delta_text = "N/A" if avg_delta is None else f"{avg_delta:.3f}"
        ml_weight = snap.get("ml_weight")
        ml_weight_text = "N/A" if ml_weight is None else f"{ml_weight:.3f}"

        status_line = _get_status_line()
        freshness_line = _get_data_freshness_text()
        lines = [
            "📊 **Decision Attempts (Runtime)**",
            f"Attempts: {snap.get('attempts', 0)}",
            f"Opened: {snap.get('opened', 0)}",
            f"Blocked: {snap.get('blocked', 0)}",
            f"Top Block Reason: {snap.get('top_block_reason', 'N/A')}",
            f"ML Weight: {ml_weight_text}",
            f"Avg Blended vs Threshold (Last 20): {avg_delta_text}",
        ]
        if status_line:
            lines.append(status_line)
        if freshness_line:
            lines.append(freshness_line)

        await _send_embed(ctx, "\n".join(lines), title="Attempts")

    @commands.command()
    async def trades(self, ctx, page: str | int = 1):
        try:
            acc = load_account()
        except Exception:
            await _send_embed(ctx, "Could not load account data.")
            return

        trade_log = acc.get("trade_log", [])
        if not trade_log:
            await _send_embed(ctx, "No closed trades yet.")
            return

        per_page = 5
        total = len(trade_log)
        total_pages = (total + per_page - 1) // per_page

        page_num = 1
        if isinstance(page, str):
            page_text = page.strip().lower()
            if page_text.startswith("page"):
                page_text = page_text.replace("page", "").strip()
            if page_text.isdigit():
                page_num = int(page_text)
        elif isinstance(page, int):
            page_num = int(page)
        if page_num < 1 or page_num > total_pages:
            await _send_embed(ctx, f"Invalid page. Use `!trades 1` to `!trades {total_pages}`.")
            return

        # Show newest trades first
        newest_first = list(reversed(trade_log))
        await _send_trades_paginated(ctx, self.bot, page_num, total_pages, newest_first, per_page)


async def setup(bot):
    await bot.add_cog(MarketCommands(bot))
