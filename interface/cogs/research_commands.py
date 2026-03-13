"""
interface/cogs/research_commands.py
AI coach and research Discord commands.

Commands: !ask  !askmore  !research (gaps, hypotheses, create, compare)
          !stratperf  !candidates
"""

import logging
import os
import re
import json
import pytz
import discord
from datetime import datetime, timedelta
from discord.ext import commands

from interface.shared_state import (
    _send_embed,
    _append_footer,
    _load_sim_profiles,
    ASK_CONTEXT_CACHE,
    STRATEGY_INTENTS,
)
from interface.fmt import ab, lbl, A, wr_col
from interface.ai_assistant import ask_ai
from analytics.performance import get_paper_stats, get_career_stats
from core.account_repository import load_account
from interface.cogs.research_helpers import (
    _norm_sim_id,
    _fmt_pct,
    _fmt_secs,
    _parse_trade_dt,
    _parse_strike,
    _extract_reason,
    _extract_target_sims,
    build_sim_context,
    build_sim_fallback_answer,
    resolve_sim_id,
    build_askmore_sim_fallback,
)
from interface.cogs.research_helpers2 import (
    handle_ask_trade,
    handle_research_gaps,
    handle_research_hypotheses,
    handle_research_create,
    handle_research_compare,
)

try:
    from research.behavior_divergence import find_behavior_gaps, generate_all_reports
    from research.hypothesis_builder import build_hypothesis, save_hypothesis, load_hypotheses
    from research.sim_generator import suggest_signal_mode
    _RESEARCH_AVAILABLE = True
except ImportError:
    _RESEARCH_AVAILABLE = False


class ResearchCommands(commands.Cog, name="Research"):
    """AI coach and research pipeline commands."""

    def __init__(self, bot):
        self.bot = bot

    # ── helper ────────────────────────────────────────────────────────────────

    async def _ask_trade_impl(self, ctx, symbol: str):
        """
        !ask <option_symbol>  — Find and analyze any trade matching the contract symbol.
        Example: !ask SPY260303C00670000
        """
        await handle_ask_trade(ctx, symbol)

    # ── !ask ──────────────────────────────────────────────────────────────────

    @commands.command(name="ask")
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def cmd_ask(self, ctx, *, question=None):

        if not question:
            await _send_embed(ctx,
                "Usage:\n"
                "\u2022 `!ask <option_contract>` \u2014 trade chart + AI analysis  "
                "(e.g. `!ask SPY260310C00592000`)\n"
                "\u2022 `!ask <question>` \u2014 AI reviews your performance  "
                "(e.g. `!ask Did I overtrade?`)"
            )
            return

        # If the argument looks like an OCC option symbol, route to trade analysis
        if re.match(r'^[A-Z]{1,6}\d{6}[CP]\d+$', question.strip().upper()):
            await self._ask_trade_impl(ctx, question.strip().upper())
            return

        strategy_intents = STRATEGY_INTENTS

        sim_context = build_sim_context(question, _load_sim_profiles, strategy_intents)

        try:
            answer = ask_ai(
                question,
                "Live market snapshot",
                get_paper_stats(),
                get_career_stats(),
                load_account().get("trade_log", [])[-5:],
                sim_context,
                prior_context=None,
            )
        except Exception:
            answer = None

        # If AI returns an empty/weak/non-sim-specific reply for SIM questions, provide deterministic local analysis.
        answer_text = str(answer).strip() if answer is not None else ""
        weak_for_sim = (
            sim_context is not None
            and (
                not answer_text
                or len(answer_text) < 120
                or "sim" not in answer_text.lower()
            )
        )
        if weak_for_sim:
            fallback = build_sim_fallback_answer(question, _load_sim_profiles, strategy_intents)
            if fallback:
                answer = fallback

        if answer is None or not str(answer).strip():
            answer = "No response available."
        else:
            answer = str(answer).strip()
            if "!askmore" not in answer.lower():
                answer = (
                    f"{answer}\n\n"
                    "Follow-up: use `!askmore <question>` to continue from this exact context."
                )
            # Discord embed description hard limit is 4096; keep headroom for status lines.
            if len(answer) > 3400:
                answer = answer[:3400].rstrip() + "\n\n[truncated]"

        ASK_CONTEXT_CACHE[str(ctx.author.id)] = {
            "question": str(question),
            "answer": str(answer),
            "sim_context": sim_context,
            "timestamp": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
        }

        await _send_embed(ctx, answer, title="AI Coach")

    # ── !askmore ──────────────────────────────────────────────────────────────

    @commands.command(name="askmore")
    async def cmd_askmore(self, ctx, *, question=None):
        if not question:
            await _send_embed(ctx, "Usage: `!askmore <follow-up question>`")
            return

        cache_key = str(ctx.author.id)
        last_ctx = ASK_CONTEXT_CACHE.get(cache_key)
        if not isinstance(last_ctx, dict):
            await _send_embed(ctx, "No prior `!ask` context found. Use `!ask` first.")
            return

        paper = get_paper_stats()
        career = get_career_stats()
        acc = load_account()

        prev_q = str(last_ctx.get("question") or "")
        prev_a = str(last_ctx.get("answer") or "")
        prev_ts = str(last_ctx.get("timestamp") or "N/A")
        sim_context = last_ctx.get("sim_context")

        prior_context = (
            f"Previous question: {prev_q}\n"
            f"Previous answer: {prev_a}\n"
            f"Previous timestamp ET: {prev_ts}\n"
            "Continue from this context and directly answer the follow-up."
        )

        try:
            answer = ask_ai(
                question,
                "Live market snapshot",
                paper,
                career,
                acc.get("trade_log", [])[-5:],
                sim_context,
                prior_context=prior_context,
            )
        except Exception:
            answer = None

        answer_text = str(answer).strip() if answer is not None else ""
        weak_for_sim = (
            sim_context is not None
            and (
                not answer_text
                or len(answer_text) < 120
                or "sim" not in answer_text.lower()
            )
        )

        if weak_for_sim:
            fallback = build_askmore_sim_fallback(question, prev_q, sim_context, _load_sim_profiles)
            if fallback:
                answer = fallback
            else:
                answer = (
                    "I could not produce a full continuation from model output this turn. "
                    f"Prior question was: `{prev_q}`. "
                    "Try `!askmore break down the last 3 trades with entry context and regime`."
                )
        elif answer is None or not str(answer).strip() or len(str(answer).strip()) < 40:
            answer = (
                "I could not produce a full continuation from model output this turn. "
                f"Prior question was: `{prev_q}`. "
                "Try a narrower follow-up like `!askmore break down the last 3 trades and exit reasons`."
            )
        else:
            answer = str(answer).strip()
            if "!askmore" not in answer.lower():
                answer = (
                    f"{answer}\n\n"
                    "Follow-up: use `!askmore <question>` to keep this same context."
                )
            if len(answer) > 3400:
                answer = answer[:3400].rstrip() + "\n\n[truncated]"

        ASK_CONTEXT_CACHE[cache_key] = {
            "question": str(question),
            "answer": str(answer),
            "sim_context": sim_context,
            "timestamp": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
        }
        await _send_embed(ctx, answer, title="AI Coach")

    # ── !research ─────────────────────────────────────────────────────────────

    @commands.command(name="research")
    @commands.cooldown(1, 30, commands.BucketType.user)
    async def cmd_research(self, ctx, subcmd: str | None = None, *args):
        if not _RESEARCH_AVAILABLE:
            embed = discord.Embed(title="Research Unavailable", description="Research modules failed to import.", color=0xE74C3C)
            _append_footer(embed)
            await ctx.send(embed=embed)
            return

        subcmd = (subcmd or "").lower()

        if subcmd == "gaps":
            await handle_research_gaps(ctx, args, find_behavior_gaps, generate_all_reports)

        elif subcmd == "hypotheses":
            await handle_research_hypotheses(ctx, load_hypotheses)

        elif subcmd == "create":
            await handle_research_create(ctx, args, build_hypothesis, save_hypothesis, suggest_signal_mode)

        elif subcmd == "compare":
            await handle_research_compare(ctx, args, find_behavior_gaps)

        else:
            embed = discord.Embed(
                title="!research \u2014 Subcommands",
                description=ab(
                    f"{A('!research gaps <SIM_ID>', 'cyan')} \u2014 behavior gap analysis for one sim",
                    f"{A('!research gaps all', 'cyan')} \u2014 gap summary for all sims",
                    f"{A('!research compare <SIM_A> <SIM_B>', 'cyan')} \u2014 side-by-side behavior comparison",
                    f"{A('!research hypotheses', 'cyan')} \u2014 list saved hypotheses",
                    f'{A("!research create", "cyan")} <source> <claim> <counter> <features> \u2014 create hypothesis',
                ),
                color=0x3498DB,
            )
            _append_footer(embed)
            await ctx.send(embed=embed)


    @commands.command(name="stratperf")
    async def cmd_stratperf(self, ctx, strategy: str = None, regime: str = None):
        """Strategy performance breakdown. Usage: !stratperf <strategy> [regime]"""
        if strategy is None:
            await _send_embed(ctx, "Usage: `!stratperf <strategy> [regime]`\nExample: `!stratperf TREND_PULLBACK` or `!stratperf TREND_PULLBACK TREND`")
            return
        try:
            from analytics.strategy_performance import PERF_STORE
            strat = strategy.upper()
            if regime:
                reg = regime.upper()
                data = PERF_STORE._data.get(strat, {}).get(reg, {})
                if not data:
                    await _send_embed(ctx, f"No data for `{strat}` in regime `{reg}` yet.")
                    return
                embed = discord.Embed(title=f"{strat} / {reg}", color=0x3498DB)
                lines = []
                for bucket, stats in sorted(data.items()):
                    wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
                    _pnl = stats["total_pnl"]
                    _pnl_str = f"${_pnl:.2f}"
                    lines.append(
                        f"{lbl(bucket)} {A(str(stats['trades']), 'white', bold=True)} trades  "
                        f"{A(f'{wr:.0f}%', 'green' if wr >= 50 else 'red')} WR  "
                        f"{A(_pnl_str, 'green' if _pnl >= 0 else 'red')}"
                    )
                embed.add_field(name="Bucket breakdown", value=ab(*lines) if lines else "_none_", inline=False)
            else:
                summary = PERF_STORE.get_strategy_summary(strat)
                if not summary:
                    await _send_embed(ctx, f"No data for `{strat}` yet.")
                    return
                embed = discord.Embed(title=f"{strat} — Performance Summary", color=0x3498DB)
                embed.add_field(name="Trades",    value=str(summary["trades"]),                               inline=True)
                embed.add_field(name="Win Rate",  value=f"{summary['win_rate'] * 100:.0f}%",                  inline=True)
                embed.add_field(name="Total PnL", value=f"${summary['total_pnl']:.2f}",                       inline=True)
                embed.add_field(name="Regimes",   value=", ".join(summary["regimes"]) or "—",                 inline=False)
            _append_footer(embed)
            await ctx.send(embed=embed)
        except Exception as exc:
            logging.exception("!stratperf failed: %s", exc)
            await _send_embed(ctx, "⚠️ Error loading strategy performance data.")

    @commands.command(name="candidates")
    async def cmd_candidates(self, ctx, date: str = None):
        """Candidate signal summary for a day. Usage: !candidates [YYYY-MM-DD]"""
        try:
            date_str = date or datetime.now(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d")
            path = os.path.join("data", "candidates", f"{date_str}.jsonl")
            if not os.path.exists(path):
                await _send_embed(ctx, f"No candidate data for `{date_str}`.")
                return

            fired = blocked = traded = total = 0
            by_strategy: dict = {}
            with open(path) as f:
                for line in f:
                    try:
                        c = json.loads(line)
                    except Exception:
                        continue
                    total += 1
                    if c.get("fired"):     fired += 1
                    if c.get("blocked"):   blocked += 1
                    if c.get("traded"):    traded += 1
                    s = c.get("strategy") or "?"
                    entry = by_strategy.setdefault(s, {"fired": 0, "blocked": 0, "traded": 0})
                    if c.get("fired"):   entry["fired"] += 1
                    if c.get("blocked"): entry["blocked"] += 1
                    if c.get("traded"):  entry["traded"] += 1

            embed = discord.Embed(title=f"📊 Candidates — {date_str}", color=0x3498DB)
            embed.add_field(name="Evaluations", value=str(total),   inline=True)
            embed.add_field(name="Fired",        value=str(fired),   inline=True)
            embed.add_field(name="Blocked",      value=str(blocked), inline=True)
            embed.add_field(name="Traded",       value=str(traded),  inline=True)

            top = sorted(by_strategy.items(), key=lambda x: x[1]["fired"], reverse=True)[:8]
            lines = [
                f"{lbl(s)} {A(str(d['fired']), 'white', bold=True)} fired  "
                f"{A(str(d['blocked']), 'red')} blocked  "
                f"{A(str(d['traded']), 'green')} traded"
                for s, d in top
            ]
            if lines:
                embed.add_field(name="By Strategy (top 8)", value=ab(*lines), inline=False)
            _append_footer(embed)
            await ctx.send(embed=embed)
        except Exception as exc:
            logging.exception("!candidates failed: %s", exc)
            await _send_embed(ctx, "⚠️ Error loading candidate data.")


async def setup(bot):
    await bot.add_cog(ResearchCommands(bot))
