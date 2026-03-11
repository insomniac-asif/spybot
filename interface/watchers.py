# interface/watchers.py

import asyncio
import os
import yaml
import logging
import time
from datetime import datetime, time as dtime
import pytz
import discord
try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None
import pandas as pd
from pandas.errors import EmptyDataError

from core.market_clock import market_is_open
from core.data_service import get_market_dataframe, backfill_symbol_csvs
from core.structured_logger import slog_critical
from core.reconciler import write_heartbeat
from core.paths import DATA_DIR
from analytics.feature_drift import detect_feature_drift
from analytics.grader import check_predictions
from signals.opportunity import evaluate_opportunity
from signals.regime import get_regime
from signals.volatility import volatility_state

from interface.watcher_helpers import (
    _run_auto_trader_cycle,
    explain_block_reason,
    _record_decision_attempt,
    get_decision_buffer_snapshot,
)
from interface.watcher_preopen import _run_preopen_checks
from interface.watcher_health import _run_heart_monitor_cycle
from interface.watcher_utils import (
    _format_et,
    _last_spy_price,
    _get_data_age_text,
    _send_embed_message,
    _send,
)
from interface.watcher_conviction import _run_conviction_cycle
from interface.watcher_forecast import _run_forecast_cycle
from interface.watcher_loops import _run_eod_report, _build_chain_health_embed
from interface.watcher_emergency import _run_emergency_exit_check
from interface.watcher_grader import _run_grader_cycle
from interface.watcher_opportunity import _build_opp_embed_from_parts

# Track uptime
START_TIME = time.time()

DISCORD_OWNER_ID = int(os.getenv("DISCORD_OWNER_ID", "0") or "0")
_MD_TURNOFF_SUGGESTED_DATE = None


# =========================================================
# OPPORTUNITY WATCHER
# =========================================================
print("Opportunity watcher started")

async def opportunity_watcher(bot, alert_channel_id):
    await bot.wait_until_ready()
    channel = bot.get_channel(alert_channel_id)
    last_alert: dict = {}  # keyed by symbol

    while not bot.is_closed():
        if not market_is_open():
            await asyncio.sleep(120)
            continue

        # Load all registered symbols' dataframes concurrently
        from core.data_service import get_symbol_dataframe, _load_symbol_registry
        try:
            _registry = _load_symbol_registry()
            _sym_names = [s.upper() for s in _registry]
        except Exception:
            _sym_names = ["SPY"]

        _fetch_tasks = [
            asyncio.to_thread(get_symbol_dataframe, s) if s != "SPY"
            else asyncio.to_thread(get_market_dataframe)
            for s in _sym_names
        ]
        _fetched = await asyncio.gather(*_fetch_tasks, return_exceptions=True)

        for sym, sym_df in zip(_sym_names, _fetched):
            if isinstance(sym_df, BaseException) or sym_df is None:
                continue
            result = evaluate_opportunity(sym_df)
            if result and result != last_alert.get(sym):
                side = result[0]
                low = result[1]
                high = result[2]
                price = result[3]
                conviction_score = result[4]
                tp_low = result[5] if len(result) > 5 else None
                tp_high = result[6] if len(result) > 6 else None
                stop_loss = result[7] if len(result) > 7 else None
                vol = volatility_state(sym_df)
                regime = get_regime(sym_df)
                opp_embed = _build_opp_embed_from_parts(
                    side, low, high, price, conviction_score,
                    tp_low, tp_high, stop_loss, vol, regime, _format_et,
                    symbol=sym,
                )
                await _send(channel, embed=opp_embed)
                last_alert[sym] = result

        await asyncio.sleep(120)


# =========================================================
# AUTO TRADER (Detailed + Structured)
# =========================================================
print("Auto trader started")
async def auto_trader(bot, channel_id):
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id)
    nyse_calendar = mcal.get_calendar("NYSE") if mcal is not None else None
    _runtime_activated = False
    while not bot.is_closed():
        if not getattr(bot, "trading_enabled", True):
            if not getattr(bot, "startup_notice_sent", False):
                startup_errors = getattr(bot, "startup_errors", [])
                err_text = "\n".join(f"• {e}" for e in startup_errors) if startup_errors else "• unknown"
                await _send(channel,
                    "🛑 **Trading Disabled – Startup Phase Gate Failed**\n\n"
                    f"{err_text}"
                )
                bot.startup_notice_sent = True
            await asyncio.sleep(60)
            continue

        eastern = pytz.timezone("America/New_York")
        now_eastern = datetime.now(eastern)
        today = now_eastern.date()
        if nyse_calendar is not None and today.weekday() < 5:
            today_schedule = nyse_calendar.schedule(
                start_date=today.isoformat(),
                end_date=today.isoformat()
            )
            if today_schedule.empty:
                notice_date = getattr(bot, "last_holiday_notice_date", None)
                if notice_date != today.isoformat():
                    await _send(channel,
                        "🗓️ **NYSE Holiday Closure**\n\n"
                        "Market is fully closed today. Auto trading attempts paused."
                    )
                    bot.last_holiday_notice_date = today.isoformat()
                await asyncio.sleep(60)
                continue

        if not market_is_open():
            await asyncio.sleep(60)
            continue

        df = await asyncio.to_thread(get_market_dataframe)
        if df is None:
            await asyncio.sleep(60)
            continue

        should_continue, _runtime_activated = await _run_auto_trader_cycle(
            bot=bot,
            channel=channel,
            df=df,
            eastern=eastern,
            _runtime_activated=_runtime_activated,
            _send=_send,
            _format_et=_format_et,
            explain_block_reason=explain_block_reason,
            _record_decision_attempt=_record_decision_attempt,
            _last_spy_price=_last_spy_price,
        )
        if should_continue:
            await asyncio.sleep(60)
            continue

        await asyncio.sleep(60)


# =========================================================
# PREDICTION GRADER LOOP
# =========================================================
print("Prediction grader started")
async def prediction_grader(bot, channel_id=None):
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id) if channel_id is not None else None
    pred_file = f"{DATA_DIR}/predictions.csv"

    while not bot.is_closed():
        await asyncio.to_thread(check_predictions)

        try:
            preds = await asyncio.to_thread(pd.read_csv, pred_file)
        except (FileNotFoundError, EmptyDataError):
            preds = None
        except Exception:
            preds = None

        await _run_grader_cycle(bot, channel, preds)
        await asyncio.sleep(300)


# =========================================================
# CONVICTION WATCHER (Detailed + Setup Intelligence)
# =========================================================
print("Conviction watcher started")

async def conviction_watcher(bot, alert_channel_id):
    await bot.wait_until_ready()
    channel = bot.get_channel(alert_channel_id)

    conviction_states  = {}
    last_upgrade_times = {}
    last_decay_times   = {}

    drift = detect_feature_drift()
    if drift:
        severity = drift["severity"]
        features = "\n".join(drift["features"])
        await _send(channel,
            f"⚠️ **Feature Drift Detected**\n\n"
            f"Severity: {severity}\n"
            f"{features}"
        )

    while not bot.is_closed():
        try:
            if not market_is_open():
                await asyncio.sleep(120)
                continue

            df = await asyncio.to_thread(get_market_dataframe)
            if df is None:
                await asyncio.sleep(120)
                continue

            global _MD_TURNOFF_SUGGESTED_DATE
            _MD_TURNOFF_SUGGESTED_DATE = await _run_conviction_cycle(
                bot=bot,
                channel=channel,
                df=df,
                conviction_states=conviction_states,
                last_upgrade_times=last_upgrade_times,
                last_decay_times=last_decay_times,
                DISCORD_OWNER_ID=DISCORD_OWNER_ID,
                _format_et=_format_et,
                _send=_send,
                _MD_TURNOFF_SUGGESTED_DATE=_MD_TURNOFF_SUGGESTED_DATE,
            )

        except Exception:
            logging.exception("conviction_watcher_error")
        await asyncio.sleep(120)


# =========================================================
# FORECAST WATCHER (Detailed + Clean Logging)
# =========================================================
print("Forecast watcher started")
async def forecast_watcher(bot, forecast_channel_id):
    await bot.wait_until_ready()
    channel = bot.get_channel(forecast_channel_id)
    last_logged_slot = None

    while not bot.is_closed():
        try:
            if not market_is_open():
                await asyncio.sleep(60)
                continue

            eastern = pytz.timezone("US/Eastern")
            now = datetime.now(eastern)

            slot_minute = (now.minute // 10) * 10
            slot_time = now.replace(minute=slot_minute, second=0, microsecond=0)
            if last_logged_slot is None or slot_time > last_logged_slot:
                df = await asyncio.to_thread(get_market_dataframe)
                if df is None:
                    await asyncio.sleep(60)
                    continue

                try:
                    await _run_forecast_cycle(
                        channel=channel,
                        df=df,
                        slot_time=slot_time,
                        _format_et=_format_et,
                        _send=_send,
                    )
                except Exception:
                    logging.exception("forecast_prediction_error")
                    await asyncio.sleep(60)
                    continue

                last_logged_slot = slot_time

            await asyncio.sleep(20)
        except Exception as _fw_exc:
            logging.exception("forecast_watcher_error")
            slog_critical("forecast_watcher_error", error=str(_fw_exc))
            await asyncio.sleep(60)


# ── HEART MONITOR ──────────────────────────────────────
async def heart_monitor(bot, channel_id):
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id)
    last_health_emit = None
    last_reconcile_time = None

    while not bot.is_closed():
        try:
            from core.account_repository import load_account
            eastern = pytz.timezone("US/Eastern")
            now = datetime.now(eastern)
            acc = load_account()

            last_health_emit, last_reconcile_time = await _run_heart_monitor_cycle(
                bot=bot,
                channel=channel,
                acc=acc,
                now_et=now,
                eastern=eastern,
                start_time=START_TIME,
                last_health_emit=last_health_emit,
                last_reconcile_time=last_reconcile_time,
                _send=_send,
            )

        except Exception as e:
            await _send(channel, "⚠️ Health monitor encountered an error.")
            print("Health monitor error:", e)

        write_heartbeat()
        await asyncio.sleep(60)


# ── EMERGENCY EXIT LOOP (Phase 6) ──────────────────────
async def emergency_exit_loop():
    """
    Force-close conditions (any one triggers an immediate market close):
      1. PANIC_LOCKDOWN state → close ALL SIM00 live positions
      2. Same-day expiry past 15:50 ET (tighter than normal 15:55 cutoff)
      3. Option price dropped > 60% from entry (flash crash / data issue)
    """
    while True:
        try:
            await asyncio.sleep(15)
            await _run_emergency_exit_check()
        except Exception as _e:
            logging.error("emergency_exit_loop_error: %s", _e, exc_info=True)
            await asyncio.sleep(30)


async def backfill_watcher():
    """
    Runs once per trading day shortly after market open (09:35 ET).
    Checks all symbol CSVs for sparse/missing data from the previous session
    and backfills from Alpaca if needed.
    """
    last_run_date = None
    eastern = pytz.timezone("America/New_York")
    target_time = dtime(9, 35)
    while True:
        try:
            now_et = datetime.now(eastern)
            if now_et.weekday() > 4:
                await asyncio.sleep(300)
                continue
            if now_et.time() < target_time:
                await asyncio.sleep(30)
                continue
            if last_run_date == now_et.date():
                await asyncio.sleep(300)
                continue

            last_run_date = now_et.date()
            results = await asyncio.to_thread(backfill_symbol_csvs)
            if results:
                logging.error("backfill_watcher_done: %s", results)
                # Generate predictions for corrected dates
                try:
                    corrected_syms = list(results.keys())
                    corrected_dates = [v["date"] for v in results.values() if isinstance(v, dict) and "date" in v]
                    if corrected_dates:
                        earliest = str(min(corrected_dates))
                        from scripts.backfill_predictions import (
                            backfill_all_symbols_chronological, _load_existing_keys,
                        )
                        from analytics.prediction_stats import PRED_FILE, PRED_HEADERS
                        existing = await asyncio.to_thread(_load_existing_keys, PRED_FILE)
                        new_rows = await asyncio.to_thread(
                            backfill_all_symbols_chronological,
                            corrected_syms, earliest, existing,
                        )
                        if new_rows:
                            import csv
                            with open(PRED_FILE, "a", newline="") as f:
                                writer = csv.DictWriter(f, fieldnames=PRED_HEADERS)
                                writer.writerows(new_rows)
                            logging.error("backfill_predictions_done: %d predictions for %s from %s",
                                          len(new_rows), corrected_syms, earliest)
                except Exception as e:
                    logging.error("backfill_predictions_error: %s", e, exc_info=True)
        except Exception as e:
            logging.error("backfill_watcher_error: %s", e, exc_info=True)
        await asyncio.sleep(30)


async def preopen_check_loop(bot, channel_id: int):
    """
    Auto pre-open readiness check. Runs once per trading day around 09:25 ET.
    Posts a summary embed to the given channel.
    """
    last_run_date = None
    eastern = pytz.timezone("America/New_York")
    target_time = dtime(9, 25)
    while True:
        try:
            now_et = datetime.now(eastern)
            if now_et.weekday() > 4:
                await asyncio.sleep(300)
                continue
            window_start = dtime(9, 10)
            window_end = dtime(9, 40)
            if now_et.time() < window_start or now_et.time() > window_end:
                await asyncio.sleep(600)
                continue
            if now_et.time() < target_time:
                await asyncio.sleep(30)
                continue
            if last_run_date == now_et.date():
                await asyncio.sleep(300)
                continue

            channel = bot.get_channel(channel_id) if bot else None
            if channel is None:
                await asyncio.sleep(300)
                continue

            df = await asyncio.to_thread(get_market_dataframe)
            if df is None or df.empty:
                await _send_embed_message(channel, "Market data unavailable.", title="Pre-Open Check")
                last_run_date = now_et.date()
                continue

            profile_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "simulation", "sim_config.yaml")
            )
            profile = None
            try:
                with open(profile_path, "r") as f:
                    profiles = yaml.safe_load(f) or {}
                    profile = profiles.get("SIM03") or profiles.get("SIM01")
            except Exception:
                profile = None

            await _run_preopen_checks(bot, channel, df, now_et, profile)
            last_run_date = now_et.date()
        except Exception:
            logging.exception("preopen_check_loop_error")
        await asyncio.sleep(30)


async def eod_open_trade_report_loop(bot, channel_id: int):
    """
    End-of-day open trade report for live trades (main trader).
    Posts summary of open trades at 16:00 ET.
    """
    last_run_date = None
    eastern = pytz.timezone("America/New_York")
    cutoff = dtime(16, 0)
    while True:
        try:
            now_et = datetime.now(eastern)
            if now_et.weekday() > 4:
                await asyncio.sleep(300)
                continue
            if now_et.time() < cutoff:
                await asyncio.sleep(30)
                continue
            if last_run_date == now_et.date():
                await asyncio.sleep(300)
                continue

            channel = bot.get_channel(channel_id) if bot else None
            if channel is None:
                await asyncio.sleep(300)
                continue

            from core.account_repository import load_account
            acc = load_account()
            embed = _run_eod_report(acc)
            _age_text = await asyncio.to_thread(_get_data_age_text)
            embed.set_footer(text=f"Time: {_format_et(now_et)} | {_age_text or 'Data age: N/A'}")
            await channel.send(embed=embed)
            last_run_date = now_et.date()
        except Exception:
            logging.exception("eod_open_trade_report_error")
        await asyncio.sleep(30)


async def option_chain_health_loop(bot, channel_id: int):
    """
    Hourly during market hours: report chain/snapshot errors for sims.
    """
    last_run_hour = None
    eastern = pytz.timezone("America/New_York")
    while True:
        try:
            now_et = datetime.now(eastern)
            if now_et.weekday() > 4 or not market_is_open():
                await asyncio.sleep(300)
                continue
            if last_run_hour == (now_et.date(), now_et.hour):
                await asyncio.sleep(30)
                continue
            if now_et.minute != 0:
                await asyncio.sleep(30)
                continue

            embed = _build_chain_health_embed(now_et)
            _age_text = await asyncio.to_thread(_get_data_age_text)
            embed.set_footer(text=f"Time: {_format_et(now_et)} | {_age_text or 'Data age: N/A'}")

            if bot is not None and channel_id:
                channel = bot.get_channel(channel_id)
                if channel is not None:
                    await channel.send(embed=embed)

            last_run_hour = (now_et.date(), now_et.hour)
        except Exception:
            logging.exception("option_chain_health_loop_error")
        await asyncio.sleep(30)


async def journal_auto_save_loop():
    """
    Auto-save daily trade journal at 16:20 ET each trading day.
    """
    last_run_date = None
    eastern = pytz.timezone("America/New_York")
    target_time = dtime(16, 20)
    while True:
        try:
            now_et = datetime.now(eastern)
            if now_et.weekday() > 4:
                await asyncio.sleep(300)
                continue
            if now_et.time() < target_time:
                await asyncio.sleep(30)
                continue
            if last_run_date == now_et.date():
                await asyncio.sleep(300)
                continue

            last_run_date = now_et.date()
            from analytics.trade_journal import save_daily_journal
            path = await asyncio.to_thread(save_daily_journal)
            logging.error("journal_auto_save: saved %s", path)
        except Exception:
            logging.exception("journal_auto_save_error")
        await asyncio.sleep(30)


async def adaptive_tuning_loop(bot=None, channel_id=None):
    """
    Run adaptive Greeks tuning at 16:25 ET each trading day.
    Posts summary of changes to Discord if bot/channel provided.
    """
    last_run_date = None
    eastern = pytz.timezone("America/New_York")
    target_time = dtime(16, 25)
    while True:
        try:
            now_et = datetime.now(eastern)
            if now_et.weekday() > 4:
                await asyncio.sleep(300)
                continue
            if now_et.time() < target_time:
                await asyncio.sleep(30)
                continue
            if last_run_date == now_et.date():
                await asyncio.sleep(300)
                continue

            last_run_date = now_et.date()
            from analytics.adaptive_tuning import run_all_adaptive_tuning
            all_changes = await asyncio.to_thread(run_all_adaptive_tuning)

            if all_changes:
                logging.error("adaptive_tuning: changes=%s", all_changes)
                # Post to Discord
                if bot and channel_id:
                    channel = bot.get_channel(channel_id)
                    if channel:
                        lines = []
                        for sim_id, changes in all_changes.items():
                            for c in changes:
                                lines.append(
                                    f"**{sim_id}** {c['trigger']}: {c['old']} -> {c['new']} ({c['reason']})"
                                )
                        embed = discord.Embed(
                            title="Adaptive Tuning Update",
                            description="\n".join(lines[:20]),
                            color=0x3498DB,
                        )
                        embed.set_footer(text=_format_et(now_et))
                        await channel.send(embed=embed)
            else:
                logging.error("adaptive_tuning: no changes today")
        except Exception:
            logging.exception("adaptive_tuning_loop_error")
        await asyncio.sleep(30)


async def weight_reoptimizer_loop():
    """
    Post-market weight reoptimizer. Runs once per trading day at 16:15 ET.
    If the total graded prediction count is divisible by 250, recompute
    predictor weights from the full graded history.
    """
    last_run_date = None
    last_optimized_count = None
    eastern = pytz.timezone("America/New_York")
    target_time = dtime(16, 15)
    while True:
        try:
            now_et = datetime.now(eastern)
            if now_et.weekday() > 4:
                await asyncio.sleep(300)
                continue
            if now_et.time() < target_time:
                await asyncio.sleep(30)
                continue
            if last_run_date == now_et.date():
                await asyncio.sleep(300)
                continue

            last_run_date = now_et.date()

            pred_file = os.path.join(DATA_DIR, "predictions.csv")
            if not os.path.exists(pred_file):
                continue

            df = await asyncio.to_thread(
                pd.read_csv, pred_file, usecols=["checked"],
            )
            total_graded = int((df["checked"] == True).sum())

            if total_graded > 0 and total_graded % 250 == 0 and total_graded != last_optimized_count:
                from analytics.predictor_optimizer import update_predictor_weights
                weights = await asyncio.to_thread(update_predictor_weights)
                last_optimized_count = total_graded
                wr = weights.get("meta", {}).get("overall_wr", 0) if weights else 0
                logging.error(
                    "weight_reoptimizer: reoptimized at %d graded predictions, WR=%.1f%%",
                    total_graded, wr * 100,
                )
            else:
                logging.error(
                    "weight_reoptimizer: %d graded (not divisible by 250, skipping)",
                    total_graded,
                )
        except Exception:
            logging.exception("weight_reoptimizer_error")
        await asyncio.sleep(30)
