"""
interface/cogs/live_commands.py
Control-plane Discord commands for SIM00 live trading operations.

Commands: !kill  !unkill  !reconcile  !status
"""

import discord
from discord.ext import commands


class LiveCommands(commands.Cog, name="Live"):
    """Control-plane commands for the live trading system."""

    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="kill")
    @commands.is_owner()
    async def cmd_kill(self, ctx, *, reason: str = "operator triggered"):
        """Activate the emergency kill switch — blocks all new live entries."""
        try:
            from core.singletons import RUNTIME, RISK_SUPERVISOR, SystemState
            RISK_SUPERVISOR.emergency_kill(reason)
            RUNTIME.transition(SystemState.EXIT_ONLY, f"kill: {reason}")
            embed = discord.Embed(
                title="🔴 KILL SWITCH ACTIVATED",
                description=(
                    f"Reason: **{reason}**\n\n"
                    "All new live entries are blocked. Existing positions still managed.\n"
                    "Use `!unkill` to restore."
                ),
                color=0xFF0000,
            )
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Kill switch error: {e}")

    @commands.command(name="unkill")
    @commands.is_owner()
    async def cmd_unkill(self, ctx):
        """Clear the emergency kill switch."""
        try:
            from core.singletons import RUNTIME, RISK_SUPERVISOR, SystemState
            RISK_SUPERVISOR.clear_kill()
            if RUNTIME.state == SystemState.EXIT_ONLY:
                RUNTIME.transition(SystemState.READY, "kill_switch_cleared")
            embed = discord.Embed(
                title="🟢 Kill Switch Cleared",
                description=f"System state: **{RUNTIME.state.value}**",
                color=0x00FF00,
            )
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Unkill error: {e}")

    @commands.command(name="reconcile")
    @commands.is_owner()
    async def cmd_reconcile(self, ctx):
        """Force a broker vs internal state reconciliation check."""
        await ctx.send("Running reconciliation…")
        try:
            from core.reconciliation import reconcile_live_positions
            from core.singletons import RUNTIME, SystemState
            result = await reconcile_live_positions()
            embed = discord.Embed(
                title="Reconciliation Report",
                description=f"```{result.summary()}```",
                color=0x00FF00 if result.clean else 0xFF0000,
            )
            if result.orphaned_broker:
                embed.add_field(
                    name="⚠️ Broker Orphans",
                    value="\n".join(
                        f"`{p['symbol']}` qty={p.get('qty')}"
                        for p in result.orphaned_broker[:10]
                    ),
                    inline=False,
                )
            if result.orphaned_internal:
                embed.add_field(
                    name="⚠️ Internal Orphans",
                    value="\n".join(
                        f"`{t['symbol']}` qty={t.get('qty')}"
                        for t in result.orphaned_internal[:10]
                    ),
                    inline=False,
                )
            if result.mismatched:
                embed.add_field(
                    name="⚠️ Qty Mismatches",
                    value="\n".join(
                        f"`{m['symbol']}` broker={m.get('broker_qty')} vs internal={m.get('internal_qty')}"
                        for m in result.mismatched[:10]
                    ),
                    inline=False,
                )
            await ctx.send(embed=embed)
            if result.clean and RUNTIME.state == SystemState.EXIT_ONLY:
                RUNTIME.transition(SystemState.READY, "manual_reconciliation_passed")
                await ctx.send("✅ System cleared — transitioned to READY")
        except Exception as e:
            await ctx.send(f"Reconciliation error: {e}")

    @commands.command(name="status")
    async def cmd_system_status(self, ctx):
        """Show current runtime state, freshness, and risk supervisor status."""
        try:
            from core.singletons import RUNTIME, RISK_SUPERVISOR
            rt = RUNTIME.get_status_dict()
            rs = RISK_SUPERVISOR.get_status()
            _emoji = {
                "BOOTING": "🔄", "RECONCILING": "🔍", "READY": "🟡",
                "TRADING_ENABLED": "🟢", "DEGRADED": "🟠",
                "EXIT_ONLY": "🔴", "PANIC_LOCKDOWN": "⛔",
            }
            state_name = rt["state"]
            color = (
                0x00FF00 if state_name == "TRADING_ENABLED"
                else (0xFF0000 if state_name in ("PANIC_LOCKDOWN", "EXIT_ONLY") else 0xFF9900)
            )
            embed = discord.Embed(
                title=f"{_emoji.get(state_name, '❓')} Runtime: {state_name}",
                description=rt["reason"],
                color=color,
            )
            lines = []
            for key, label in [
                ("bar_age_seconds", "Bars"),
                ("quote_age_seconds", "Quotes"),
                ("account_age_seconds", "Account"),
                ("broker_age_seconds", "Broker"),
            ]:
                v = rs.get(key)
                if v is None:
                    lines.append(f"**{label}:** never updated")
                elif v > 120:
                    lines.append(f"**{label}:** ⚠️ {v:.0f}s stale")
                else:
                    lines.append(f"**{label}:** ✅ {v:.0f}s")
            embed.add_field(name="Data Freshness", value="\n".join(lines), inline=False)
            if rs["emergency_kill"]:
                embed.add_field(
                    name="🔴 Kill Switch Active",
                    value=rs["kill_reason"] or "no reason given",
                    inline=False,
                )
            if rt.get("degradation_reasons"):
                embed.add_field(
                    name="Degradation Reasons",
                    value=", ".join(rt["degradation_reasons"]),
                    inline=False,
                )
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Status error: {e}")


async def setup(bot):
    await bot.add_cog(LiveCommands(bot))
