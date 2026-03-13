# INTERFACE LAYER — Entry Point

import os, asyncio, json, logging, signal, discord
from logging.handlers import RotatingFileHandler
from discord.ext import commands
from dotenv import load_dotenv
from interface.health_monitor import start_heartbeat
from interface.watchers import (
    auto_trader, conviction_watcher, forecast_watcher, heart_monitor,
    prediction_grader, opportunity_watcher, preopen_check_loop,
    eod_open_trade_report_loop, option_chain_health_loop, emergency_exit_loop,
    backfill_watcher, weight_reoptimizer_loop,
    journal_auto_save_loop, adaptive_tuning_loop,
)
from simulation.sim_watcher import (
    sim_entry_loop, sim_exit_loop, sim_eod_report_loop, sim_daily_summary_loop,
    sim_weekly_leaderboard_loop, sim_weekly_behavior_report_loop,
    sim_session_leaderboard_loop, set_sim_bot,
    correlation_tracker_loop, get_correlation_state,
)
from core.structured_logger import setup_structured_logging
from analytics.feature_logger import ensure_feature_file
from logs.recorder import start_recorder_background
from core.reconciler import full_reconcile
from interface.startup_checks import run_startup_phase_gates
from interface.shared_state import (
    PAPER_CHANNEL_ID, ALERT_CHANNEL_ID, FORECAST_CHANNEL_ID,
    HEART_CHANNEL_ID, EOD_REPORT_CHANNEL_ID, _send_embed,
)

_sys_handler = RotatingFileHandler(
    "system.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=3,              # keep system.log.1, .2, .3
)
_sys_handler.setLevel(logging.ERROR)
_sys_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(message)s"))
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger().addHandler(_sys_handler)
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN or not DISCORD_TOKEN.strip():
    raise RuntimeError("DISCORD_TOKEN is missing or empty.")

intents = discord.Intents.default()
intents.message_content = True

_COG_MODULES = [
    "interface.cogs.live_commands", "interface.cogs.sim_commands",
    "interface.cogs.market_commands", "interface.cogs.research_commands",
    "interface.cogs.admin_commands",
]


class QQQBot(commands.Bot):

    async def safe_task(self, coro_func, *args):
        while True:
            try:
                await coro_func(*args)
            except Exception as e:
                logging.error(f"[CRASH] Background task {coro_func.__name__} crashed: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def setup_hook(self):
        setup_structured_logging()
        try:
            from core.runtime_state import RUNTIME, SystemState
            RUNTIME.transition(SystemState.RECONCILING, "setup_hook")
        except ImportError:
            pass
        try:
            from core.trade_db import sync_from_sim_jsons, create_tables
            create_tables()
            print(f"Trade DB synced {sync_from_sim_jsons()} records from sim JSONs.")
        except Exception as exc:
            logging.warning("trade_db_startup_sync_failed: %s", exc)
        print("Launching background systems...")
        # Auto-backfill all symbol CSVs before recorder starts
        try:
            from core.data_service import startup_backfill_all
            print("Running startup backfill for all symbols...")
            backfill_results = await asyncio.to_thread(startup_backfill_all)
            if backfill_results:
                print(f"Startup backfill complete: {backfill_results}")
            else:
                print("Startup backfill: all CSVs up to date")
        except Exception as e:
            logging.warning("startup_backfill_failed: %s", e)
            print(f"Startup backfill failed (non-fatal): {e}")
        # Backfill missed predictions for today's gap
        try:
            from simulation.prediction_backfill import backfill_missed_predictions
            print("Checking for prediction backfill...")
            bf_result = await asyncio.to_thread(backfill_missed_predictions)
            if bf_result.get("predictions_generated", 0) > 0:
                print(f"Prediction backfill: {bf_result['predictions_generated']} predictions across {bf_result['total_slots']} slots")
            else:
                print(f"Prediction backfill: no gap ({bf_result.get('reason', 'up_to_date')})")
        except Exception as e:
            print(f"Prediction backfill failed (non-fatal): {e}")
        if not hasattr(self, "recorder_thread"):
            self.recorder_thread = start_recorder_background()
        self._init_state()
        try:
            ensure_feature_file(reset_if_invalid=True)
        except Exception as e:
            logging.exception("feature_file_init_error: %s", e)
        await full_reconcile(self)
        await self._broker_reconcile()
        self._register_background_tasks()
        self._start_freshness_monitor()
        for cog in _COG_MODULES:
            try:
                await self.load_extension(cog)
            except Exception as exc:
                logging.warning("cog_load_failed: %s — %s", cog, exc)

    def _init_state(self):
        self.startup_errors = run_startup_phase_gates()
        self.trading_enabled = not self.startup_errors
        self.startup_notice_sent = False
        self.paper_channel_id = PAPER_CHANNEL_ID
        self.data_stale_state = self.data_integrity_state = False
        self.predictor_drift_state = self.recorder_stalled_state = False
        self.last_stale_warning_time = self.last_integrity_warning_time = None
        self.last_predictor_drift_warning_time = self.last_recorder_stall_warning_time = None
        self.last_recorder_thread_dead_warning_time = self.last_holiday_notice_date = None
        self.predictor_winrate_history, self.predictor_baseline_winrate = [], None
        self.block_reason_last_time = {}
        for err in self.startup_errors:
            logging.error(f"startup_error: {err}")

    async def _broker_reconcile(self):
        try:
            from core.reconciliation import reconcile_live_positions
            from core.runtime_state import RUNTIME, SystemState as _SS
            _recon = await reconcile_live_positions()
            if _recon.clean:
                RUNTIME.transition(_SS.READY, "reconciliation_clean")
            else:
                RUNTIME.transition(_SS.EXIT_ONLY, "reconciliation_mismatch")
                logging.error("startup_reconciliation_failed:\n%s", _recon.summary())
        except Exception as exc:
            logging.warning("reconciliation_startup_error: %s", exc)
            try:
                from core.runtime_state import RUNTIME, SystemState as _SS
                RUNTIME.transition(_SS.READY, "reconciliation_skipped")
            except Exception:
                pass

    def _start_dashboard(self):
        """Start dashboard as a subprocess, restart if it dies."""
        import subprocess, sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        python_exe = sys.executable

        async def _dashboard_watchdog():
            port = os.environ.get("DASHBOARD_PORT", "8090")
            proc = None
            while True:
                try:
                    if proc is None or proc.poll() is not None:
                        proc = subprocess.Popen(
                            [python_exe, "-m", "uvicorn", "dashboard.app:app",
                             "--host", "0.0.0.0", "--port", port],
                            cwd=project_root,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        logging.error(f"Dashboard started (PID: {proc.pid}, port: {port})")
                except Exception as e:
                    logging.error(f"Dashboard start failed: {e}")
                await asyncio.sleep(30)

        self.loop.create_task(self.safe_task(_dashboard_watchdog))

    def _register_background_tasks(self):
        T, S = self.loop.create_task, self.safe_task
        self._start_dashboard()
        T(S(auto_trader, self, PAPER_CHANNEL_ID))
        T(S(conviction_watcher, self, ALERT_CHANNEL_ID))
        T(S(forecast_watcher, self, FORECAST_CHANNEL_ID))
        T(S(heart_monitor, self, HEART_CHANNEL_ID))
        T(S(prediction_grader, self, PAPER_CHANNEL_ID))
        T(S(start_heartbeat)); T(S(opportunity_watcher, self, ALERT_CHANNEL_ID))
        set_sim_bot(self)
        T(S(sim_entry_loop)); T(S(sim_exit_loop))
        T(S(sim_eod_report_loop, EOD_REPORT_CHANNEL_ID))
        T(S(sim_daily_summary_loop, EOD_REPORT_CHANNEL_ID))
        T(S(sim_weekly_leaderboard_loop, EOD_REPORT_CHANNEL_ID))
        T(S(sim_weekly_behavior_report_loop, EOD_REPORT_CHANNEL_ID))
        T(S(sim_session_leaderboard_loop, 1478675253780807795))
        T(S(correlation_tracker_loop, ALERT_CHANNEL_ID))
        T(S(preopen_check_loop, self, EOD_REPORT_CHANNEL_ID))
        T(S(eod_open_trade_report_loop, self, EOD_REPORT_CHANNEL_ID))
        T(S(option_chain_health_loop, self, EOD_REPORT_CHANNEL_ID))
        T(S(emergency_exit_loop))
        T(S(backfill_watcher))
        T(S(weight_reoptimizer_loop))
        T(S(journal_auto_save_loop))
        T(S(adaptive_tuning_loop, self, EOD_REPORT_CHANNEL_ID))

    def _start_freshness_monitor(self):
        try:
            from core.freshness_monitor import FreshnessMonitor
            from core.singletons import RUNTIME, RISK_SUPERVISOR
            fm = FreshnessMonitor(RUNTIME, RISK_SUPERVISOR)
            ch_id = ALERT_CHANNEL_ID
            async def _fm_run():
                await self.wait_until_ready()
                ch = self.get_channel(ch_id)
                async def _send_alert(msg: str):
                    if ch:
                        try: await ch.send(msg)
                        except Exception: pass
                await fm.run(_send_alert)
            self.loop.create_task(self.safe_task(_fm_run))
        except Exception as exc:
            logging.warning("freshness_monitor_init_failed: %s", exc)


bot = QQQBot(command_prefix="!", intents=intents, help_command=None)

CLASSROOM_URL = os.getenv("CLASSROOM_URL", "")

@bot.command(name="classroom")
async def classroom(ctx):
    """Post a link to the SPY Sim Classroom dashboard."""
    if not CLASSROOM_URL or CLASSROOM_URL.startswith("https://your-tunnel"):
        embed = discord.Embed(
            title="Classroom URL not configured",
            description="Set `CLASSROOM_URL` in `.env` to enable this command.",
            color=0xFF6633,
        )
    else:
        embed = discord.Embed(
            title="📚 Trader Sim Classroom",
            url=CLASSROOM_URL,
            description="Live dashboard — click to open",
            color=0x8B4513,
        )
        embed.set_footer(text="36 sim strategies · live positions · leaderboard")
    await ctx.send(embed=embed)


@bot.command(name="correlation")
async def correlation(ctx, sub: str = ""):
    """Show current directional crowding state. Use `!correlation history` for recent alerts."""
    state = get_correlation_state()
    if sub.lower() == "history":
        hist = state["alert_history"]
        if not hist:
            embed = discord.Embed(
                title="Correlation History",
                description="No alerts fired yet this session.",
                color=0x95A5A6,
            )
        else:
            embed = discord.Embed(title=f"Correlation Alert History ({len(hist)} entries)", color=0xE67E22)
            for entry in reversed(hist[-10:]):
                ts_str = entry["ts"].strftime("%H:%M ET") if hasattr(entry["ts"], "strftime") else str(entry["ts"])
                sims_str = "  ".join(entry["sims"])
                fams_str = "  ·  ".join(entry["families"])
                embed.add_field(
                    name=f"{ts_str} — {entry['direction']} ({entry['count']} sims)",
                    value=f"`{sims_str}`\n{fams_str}",
                    inline=False,
                )
    else:
        fresh = state["window_entries"]
        calls = [e["sim_id"] for e in fresh if e["direction"] == "call"]
        puts = [e["sim_id"] for e in fresh if e["direction"] == "put"]
        cooldown = int(state["cooldown_remaining"])
        embed = discord.Embed(title="⚠️ Correlation Tracker — 10-min Window", color=0xE67E22)
        embed.add_field(
            name="Calls (last 10 min)",
            value=f"`{', '.join(calls)}`" if calls else "_none_",
            inline=True,
        )
        embed.add_field(
            name="Puts (last 10 min)",
            value=f"`{', '.join(puts)}`" if puts else "_none_",
            inline=True,
        )
        embed.add_field(
            name="Status",
            value=f"Cooldown: {cooldown}s remaining" if cooldown > 0 else "Ready to alert",
            inline=False,
        )
        embed.set_footer(text="Use !correlation history for past alerts")
    await ctx.send(embed=embed)


_SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sims")


@bot.command(name="tradereplay")
async def tradereplay(ctx, sim_id: str = None, trade_index: int = 1):
    """Show a trade replay chart. Usage: !tradereplay <sim_id> [N] (N=1 is most recent)."""
    if sim_id is None:
        await _send_embed(ctx, "Usage: `!tradereplay <sim_id> [N]`\nExample: `!tradereplay SIM03` or `!tradereplay SIM03 2`")
        return
    sid = sim_id.strip().upper()
    try:
        sim_path = os.path.join(_SIM_DIR, f"{sid}.json")
        if not os.path.exists(sim_path):
            await _send_embed(ctx, f"Sim not found: `{sid}`")
            return
        with open(sim_path) as _f:
            sim_data = json.load(_f)
        trades = [t for t in (sim_data.get("trade_log") or []) if t.get("exit_time")]
        if not trades:
            await _send_embed(ctx, f"No closed trades found for `{sid}`.")
            return
        if trade_index < 1 or trade_index > len(trades):
            await _send_embed(ctx, f"Only {len(trades)} closed trade(s) for `{sid}`. Use 1–{len(trades)}.")
            return
        trade = trades[-trade_index]

        from interface.charting import generate_trade_replay
        chart_path = await asyncio.to_thread(generate_trade_replay, sid, trade)

        pnl = float(trade.get("realized_pnl_dollars") or 0)
        pnl_sign = "+" if pnl >= 0 else ""
        hold_s = int(trade.get("time_in_trade_seconds") or 0)
        hold_str = f"{hold_s // 60}m {hold_s % 60}s" if hold_s else "—"
        entry_p = trade.get("entry_price")
        exit_p  = trade.get("exit_price")

        embed = discord.Embed(
            title=f"Trade Replay — {sid}  (#{trade_index} most recent)",
            color=0x2ECC71 if pnl >= 0 else 0xE74C3C,
        )
        embed.add_field(name="Direction",    value=trade.get("direction", "—"),    inline=True)
        embed.add_field(name="Signal Mode",  value=trade.get("signal_mode", "—"),  inline=True)
        embed.add_field(name="Symbol",       value=trade.get("option_symbol", "—"), inline=True)
        embed.add_field(name="Entry",  value=f"${entry_p:.4f}" if entry_p is not None else "—", inline=True)
        embed.add_field(name="Exit",   value=f"${exit_p:.4f}"  if exit_p  is not None else "—", inline=True)
        embed.add_field(name="P&L",    value=f"{pnl_sign}${pnl:.2f}",  inline=True)
        embed.add_field(name="Exit Reason", value=trade.get("exit_reason", "—"), inline=True)
        embed.add_field(name="Hold Time",   value=hold_str,                         inline=True)
        embed.add_field(name="Trades",      value=f"{len(trades)} closed",          inline=True)

        if chart_path and os.path.exists(chart_path):
            embed.set_image(url="attachment://replay.png")
            await ctx.send(embed=embed, file=discord.File(chart_path, filename="replay.png"))
        else:
            embed.add_field(
                name="Chart",
                value="Unavailable — price data not found for this trade's time range",
                inline=False,
            )
            await ctx.send(embed=embed)
    except Exception as _exc:
        logging.exception("!tradereplay failed: %s", _exc)
        await _send_embed(ctx, f"⚠️ Error generating trade replay for `{sid}`.")


async def _shutdown_handler():
    """Force-exit SIM00 live positions on shutdown."""
    logging.error("SHUTDOWN: Graceful shutdown initiated. Closing SIM00 positions...")
    try:
        import yaml
        from simulation.sim_portfolio import SimPortfolio, get_sim_lock
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _cfg_path = os.path.join(_base, "simulation", "sim_config.yaml")
        with open(_cfg_path) as _f:
            _profiles = yaml.safe_load(_f) or {}
        _sim00_profile = _profiles.get("SIM00", {})
        if not isinstance(_sim00_profile, dict) or _sim00_profile.get("execution_mode") != "live":
            logging.error("SHUTDOWN: SIM00 not live-mode. Nothing to close.")
            return
        lock = get_sim_lock("SIM00")
        async with lock:
            sim = SimPortfolio("SIM00", _sim00_profile)
            sim.load()
            if not sim.open_trades:
                logging.error("SHUTDOWN: No open SIM00 positions. Clean exit.")
                return
            from execution.option_executor import close_option_position
            for trade in list(sim.open_trades):
                option_symbol = trade.get("option_symbol")
                qty = int(trade.get("qty", 1) or 1)
                if option_symbol and qty > 0:
                    logging.error("SHUTDOWN: Closing %s (qty=%d)", option_symbol, qty)
                    try:
                        result = await asyncio.to_thread(close_option_position, option_symbol, qty)
                        logging.error("SHUTDOWN: Close result for %s: ok=%s", option_symbol, result.get("ok"))
                    except Exception as e:
                        logging.error("SHUTDOWN: Failed to close %s: %s", option_symbol, e)
    except Exception as e:
        logging.error("SHUTDOWN: Error during graceful shutdown: %s", e)
    logging.error("SHUTDOWN: Complete.")


def _setup_signal_handlers(loop):
    """Register SIGTERM/SIGINT handlers for graceful shutdown."""
    def handle_signal(sig):
        logging.error("Received signal %s. Initiating graceful shutdown.", sig.name)
        loop.create_task(_shutdown_handler())
        loop.call_later(15, lambda: loop.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handle_signal, sig)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler


@bot.event
async def on_ready():
    _setup_signal_handlers(asyncio.get_event_loop())
    print(f"Bot online as {bot.user}")

print("TOKEN LOADED: ", DISCORD_TOKEN is not None)

@bot.event
async def on_error(event, *args, **kwargs):
    print(f"An error occurred: {event}")

@bot.event
async def on_command_error(ctx, error):
    logging.exception(f"[COMMAND ERROR] {error}")
    if isinstance(error, commands.MissingRequiredArgument):
        await _send_embed(ctx, "Missing required argument. Use `!help <command>`.")
        return
    if isinstance(error, commands.BadArgument):
        await _send_embed(ctx, "Invalid argument type. Use `!help <command>`.")
        return
    if isinstance(error, commands.CommandNotFound):
        await _send_embed(ctx, "Unknown command. Type `!help`.")
        return
    await _send_embed(ctx, "⚠️ Internal error occurred. Logged for review.")

bot.run(DISCORD_TOKEN)
