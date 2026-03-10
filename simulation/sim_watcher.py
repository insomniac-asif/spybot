# simulation/sim_watcher.py
import asyncio
import logging
import os
import yaml
import pytz
import discord
from datetime import datetime, time
from core.data_service import get_market_dataframe
from simulation.sim_report_helpers import (
    _format_et,
    _parse_strike_from_symbol,
    _format_option_symbol,
    _format_entry_time,
    _format_context_parts,
    _format_exit_context,
    _get_data_age_text,
    _format_skip_reason,
    _format_feature_snapshot,
    _build_entry_embed,
    _build_exit_embed,
    _build_skip_embed,
)
from simulation.sim_watcher_loops import (
    _build_eod_open_positions_embed,
    _build_eod_combined_embed,
    _build_graduation_embed,
    _build_daily_sim_embed,
    _build_daily_combined_embed,
    _collect_weekly_sim_data,
    _build_weekly_leaderboard_embed,
    _collect_session_sim_rows,
    _build_session_leaderboard_page,
    SKIP_DEDUP_REASONS,
    _skip_dedup_key,
    _build_circuit_breaker_embed,
    _build_trade_history_ready_embed,
)
from simulation.sim_watcher_corr import (
    _record_corr_entry,
    _maybe_fire_corr_alert,
    set_corr_channel,
    correlation_tracker_loop,
    get_correlation_state,
)
from simulation.sim_watcher_sched import (
    sim_eod_report_loop,
    sim_daily_summary_loop,
    sim_weekly_leaderboard_loop,
    sim_weekly_behavior_report_loop,
    sim_session_leaderboard_loop,
)
from interface.fmt import (
    ab,
    A,
    lbl,
    pnl_col,
    conf_col,
    dir_col,
    regime_col,
    vol_col,
    delta_col,
    ml_col,
    exit_reason_col,
    balance_col,
    wr_col,
    tier_col,
    drawdown_col,
    pct_col,
)

try:
    from workspace.workspace_logger import log_event as _ws_log_event
except ImportError:
    _ws_log_event = None

ENTRY_INTERVAL_SECONDS = 60
EXIT_INTERVAL_SECONDS = 30
EOD_REPORT_TIME_ET = time(16, 0)
DAILY_SUMMARY_TIME_ET = time(16, 5)
WEEKLY_LEADERBOARD_TIME_ET = time(16, 10)
SESSION_LEADERBOARD_SIMS_PER_PAGE = 4

SIM_CHANNEL_MAP = {
    "SIM00": 1477023293545648258,  # Live version of SIM03
    "SIM01": 1476794016019386460,
    "SIM02": 1478257480952975525,
    "SIM03": 1476794039067218102,
    "SIM04": 1478257645696843939,
    "SIM05": 1476794065793323120,
    "SIM06": 1478257843894485145,
    "SIM07": 1476794956826935339,
    "SIM08": 1476795166751854654,
    "SIM09": 1477017498451705968,
    "SIM10": 1478200317035417641,
    "SIM11": 1478200298466971679,
    "SIM12": 1480410929589129266,
    "SIM13": 1480410951143522325,
    "SIM14": 1480410977857175703,
    "SIM15": 1480411008928579595,
    "SIM16": 1480411067212632124,
    "SIM17": 1480411143234392154,
    "SIM18": 1480425205384871977,
    "SIM19": 1480425237639201000,
    "SIM20": 1480425269729955930,
    "SIM21": 1480425299048140922,
    "SIM22": 1480425327959474226,
    "SIM23": 1480425354840641569,
    "SIM24": 1480709591913201675,
    "SIM25": 1480709609881866454,
    "SIM26": 1480709627170525285,
    "SIM27": 1480709642735718440,
    "SIM28": 1480709656660672648,
    "SIM29": 1480727396918366359,
    "SIM30": 1480727662120140931,
    "SIM31": 1480728633696981062,
    "SIM32": 1480727817275838536,
    "SIM33": 1480733191596802108,
    "SIM34": 1480737061458939985,
    "SIM35": 1480737079616077844,
}

_SIM_BOT = None
_SIM_LAST_SKIP_REASON = {}
_SIM_LAST_SKIP_TIME = {}
_SIM_LAST_DATA_AGE = None
_SIM_EOD_REPORT_DATE = None
_SIM_PREOPEN_REPORT_DATE = None
_SIM_DAILY_SUMMARY_DATE = None
_SIM_WEEKLY_LEADERBOARD_DATE = None
_SIM_TRADE_PROGRESS_DATE: dict = {}   # sim_id -> date of last morning progress post
_SIM_TRADE_HISTORY_UNLOCKED: set = set()  # sims announced as ready to go live
_SIM_SESSION_LEADERBOARD_DATE = None  # last date the session leaderboard was posted

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "simulation", "sim_config.yaml")


def _load_profiles() -> dict:
    try:
        with open(CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f) or {}
        return {k: v for k, v in raw.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return {}


_SIM_PROFILES = _load_profiles()


def _now_et() -> datetime:
    return datetime.now(pytz.timezone("US/Eastern"))


def set_sim_bot(bot) -> None:
    global _SIM_BOT
    _SIM_BOT = bot


def get_sim_last_skip_state() -> dict:
    state = {}
    for sim_id, reason in _SIM_LAST_SKIP_REASON.items():
        state[sim_id] = {
            "reason": reason,
            "time": _SIM_LAST_SKIP_TIME.get(sim_id),
        }
    return state


async def _post_sim_event(sim_id: str, embed: "discord.Embed") -> None:
    if _SIM_BOT is None:
        return
    channel_id = SIM_CHANNEL_MAP.get(sim_id)
    if channel_id is None:
        logging.warning("post_sim_event_no_channel: sim=%s", sim_id)
        return
    channel = _SIM_BOT.get_channel(channel_id)
    if channel is None:
        logging.warning("post_sim_event_channel_not_found: sim=%s channel_id=%s", sim_id, channel_id)
        return
    try:
        await channel.send(embed=embed)
    except Exception:
        logging.exception("post_sim_event_send_error: sim=%s", sim_id)
        return


async def _post_main_skip(embed: "discord.Embed") -> None:
    if _SIM_BOT is None:
        return
    channel_id = getattr(_SIM_BOT, "paper_channel_id", None)
    if channel_id is None:
        return
    channel = _SIM_BOT.get_channel(channel_id)
    if channel is None:
        return
    try:
        main_embed = discord.Embed(
            title=embed.title,
            description=embed.description,
            color=embed.color.value if embed.color else 0xF1C40F
        )
        for field in embed.fields:
            main_embed.add_field(name=field.name, value=field.value, inline=field.inline)
        if embed.footer and embed.footer.text:
            main_embed.set_footer(text=embed.footer.text)
        await channel.send(embed=main_embed)
    except Exception:
        return


async def _post_main_embed(embed: "discord.Embed") -> None:
    if _SIM_BOT is None:
        return
    channel_id = getattr(_SIM_BOT, "paper_channel_id", None)
    if channel_id is None:
        return
    channel = _SIM_BOT.get_channel(channel_id)
    if channel is None:
        return
    try:
        await channel.send(embed=embed)
    except Exception:
        return


def _is_market_hours() -> bool:
    eastern = pytz.timezone("US/Eastern")
    now_et = datetime.now(eastern)
    return (
        now_et.weekday() <= 4
        and (now_et.hour, now_et.minute) >= (9, 30)
        and (now_et.hour, now_et.minute) <= (16, 0)
    )


def _derive_regime(df):
    """
    Classify market regime from indicator columns.
    Returns "VOLATILE", "TREND", "RANGE", or None if data insufficient.
    """
    TREND_SEPARATION_THRESHOLD = 0.001
    ATR_LOOKBACK = 50
    ATR_VOLATILE_PERCENTILE = 75
    MIN_BARS_REQUIRED = 2

    try:
        if df is None or len(df) < MIN_BARS_REQUIRED:
            return None
        last = df.iloc[-1]
        close = float(last.get("close", 0))
        if close <= 0:
            return None
        atr_col = None
        for candidate in ["atr", "ATR", "atr14", "ATR14"]:
            if candidate in df.columns:
                atr_col = candidate
                break
        if atr_col is not None:
            lookback = min(ATR_LOOKBACK, len(df))
            atr_series = df[atr_col].iloc[-lookback:].dropna()
            if len(atr_series) >= 2:
                current_atr = float(atr_series.iloc[-1])
                pct_rank = (atr_series < current_atr).sum() / len(atr_series) * 100
                if pct_rank > ATR_VOLATILE_PERCENTILE:
                    ema9_col = None
                    for c in ["ema9", "EMA9", "ema_9"]:
                        if c in df.columns:
                            ema9_col = c
                            break
                    if ema9_col and len(df) >= 2:
                        ema9_now = float(df[ema9_col].iloc[-1])
                        ema9_prev = float(df[ema9_col].iloc[-2])
                        if ema9_now < ema9_prev:
                            return "VOLATILE"
        ema9_val = None
        ema20_val = None
        for c in ["ema9", "EMA9", "ema_9"]:
            if c in df.columns:
                ema9_val = float(last[c])
                break
        for c in ["ema20", "EMA20", "ema_20"]:
            if c in df.columns:
                ema20_val = float(last[c])
                break
        if ema9_val is not None and ema20_val is not None:
            separation = abs(ema9_val - ema20_val) / close
            if separation > TREND_SEPARATION_THRESHOLD:
                return "TREND"
        return "RANGE"
    except Exception:
        return None


def _get_time_of_day_bucket():
    """Classify current ET time into a trading session bucket."""
    try:
        now_et = datetime.now(pytz.timezone("US/Eastern"))
        t = now_et.time()
        if t < time(10, 30):
            return "MORNING"
        elif t < time(12, 0):
            return "MIDDAY"
        elif t < time(14, 0):
            return "AFTERNOON"
        else:
            return "CLOSE"
    except Exception:
        return None


async def sim_entry_loop() -> None:
    eastern = pytz.timezone("US/Eastern")
    while True:
        iter_start = datetime.now(eastern)
        try:
            if _is_market_hours():
                from simulation.sim_engine import run_sim_entries
                df = await asyncio.to_thread(get_market_dataframe)
                global _SIM_LAST_DATA_AGE
                _SIM_LAST_DATA_AGE = _get_data_age_text(df)

                # Build trader_signal context for SIM09 Opportunity Ranker.
                # Captures the main trader's current prediction state without
                # blocking the event loop or importing execution/interface modules.
                _trader_signal = None
                try:
                    from signals.predictor import make_prediction
                    from signals.regime import get_regime as _get_regime_fn
                    _pred_15 = make_prediction(15, df)
                    _pred_60 = make_prediction(60, df)
                    _regime_now = _get_regime_fn(df) if df is not None else None
                    _dir_15 = _pred_15.get("direction") if isinstance(_pred_15, dict) else None
                    _conf_15 = _pred_15.get("confidence") if isinstance(_pred_15, dict) else None
                    _dir_60 = _pred_60.get("direction") if isinstance(_pred_60, dict) else None
                    _conf_60 = _pred_60.get("confidence") if isinstance(_pred_60, dict) else None
                    _ml_pred = _pred_60 if isinstance(_pred_60, dict) else None
                    _und_price = None
                    if df is not None and len(df) > 0:
                        try:
                            _und_price = float(df.iloc[-1].get("close") or 0) or None
                        except Exception:
                            _und_price = None
                    # Fetch real conviction from evaluate_opportunity() (pure function, no side effects).
                    # Falls back to ML confidence (0.0-1.0) when no clear setup is detected.
                    _primary_dir = _dir_60 or _dir_15
                    _real_conviction = None
                    _environment_passed = None
                    try:
                        from signals.opportunity import evaluate_opportunity
                        _opp_result = evaluate_opportunity(df)
                        if _opp_result is not None:
                            # Normalize int conviction score (3-7) to 0.0-1.0 range
                            _real_conviction = min(1.0, float(_opp_result[4]) / 7.0)
                            _primary_dir = "BULLISH" if _opp_result[0] == "CALLS" else "BEARISH"
                            _environment_passed = True
                        else:
                            # No clear setup: use ML confidence as proxy, no env bonus/penalty
                            _real_conviction = _conf_60
                    except Exception:
                        _real_conviction = _conf_60
                    _trader_signal = {
                        "direction": _primary_dir,
                        "underlying_price": _und_price,
                        "conviction_score": _real_conviction,
                        "ml_prediction": _ml_pred,
                        "confidence_15m": _conf_15,
                        "confidence_60m": _conf_60,
                        "environment_passed": _environment_passed,
                        "setup_expectancy": None,
                    }
                except Exception:
                    _trader_signal = None

                results = await run_sim_entries(df, regime=_derive_regime(df), trader_signal=_trader_signal)
                # Sims that opened a trade this cycle — suppress skip
                # notifications for those sims (e.g. empty_chain from VXX
                # after SPY successfully opened).
                _opened_sims = {
                    r.get("sim_id") for r in results
                    if r.get("status") in {"opened", "live_submitted"} and r.get("sim_id")
                }
                for result in results:
                    logging.debug("sim_entry_result: %s", result)
                    try:
                        sim_id = result.get("sim_id")
                        status = result.get("status")
                        if sim_id and status in {"opened", "live_submitted"}:
                            entry_embed = _build_entry_embed(sim_id, result)
                            await _post_sim_event(sim_id, entry_embed)
                            _SIM_LAST_SKIP_REASON.pop(sim_id, None)
                            if _ws_log_event:
                                try:
                                    _ws_log_event("trade_entry", {
                                        "sim_id": sim_id,
                                        "direction": result.get("direction"),
                                        "price": result.get("fill_price"),
                                    })
                                except Exception:
                                    pass
                            direction = result.get("direction")
                            if direction:
                                _record_corr_entry(sim_id, str(direction).lower())
                                await _maybe_fire_corr_alert()
                        elif sim_id and status == "skipped":
                            reason = result.get("reason") or "unknown"
                            if reason == "insufficient_trade_history":
                                trade_count = result.get("trade_count", 0)
                                min_trades = result.get("min_trades_for_live", 50)
                                today = _now_et().date()
                                if isinstance(trade_count, int) and isinstance(min_trades, int) and trade_count >= min_trades:
                                    if sim_id not in _SIM_TRADE_HISTORY_UNLOCKED:
                                        _SIM_TRADE_HISTORY_UNLOCKED.add(sim_id)
                                        ready_embed = _build_trade_history_ready_embed(
                                            sim_id, trade_count, min_trades, _SIM_LAST_DATA_AGE
                                        )
                                        await _post_sim_event(sim_id, ready_embed)
                                elif _SIM_TRADE_PROGRESS_DATE.get(sim_id) != today:
                                    _SIM_TRADE_PROGRESS_DATE[sim_id] = today
                                    skip_embed = _build_skip_embed(sim_id, result)
                                    await _post_sim_event(sim_id, skip_embed)
                            elif reason in SKIP_DEDUP_REASONS:
                                # Suppress if this sim already opened a trade
                                # this cycle (e.g. empty_chain from VXX after
                                # SPY succeeded).
                                if sim_id in _opened_sims:
                                    pass
                                else:
                                    last_reason = _SIM_LAST_SKIP_REASON.get(sim_id)
                                    # Use bucket keys so chain-failure variants
                                    # (empty_chain ↔ no_contract ↔ no_snapshot)
                                    # don't retrigger the notification.
                                    if _skip_dedup_key(last_reason or "") != _skip_dedup_key(reason):
                                        _SIM_LAST_SKIP_REASON[sim_id] = reason
                                        _SIM_LAST_SKIP_TIME[sim_id] = _now_et()
                                        skip_embed = _build_skip_embed(sim_id, result)
                                        await _post_sim_event(sim_id, skip_embed)
                        elif sim_id and status == "circuit_breaker_tripped":
                            await _post_sim_event(
                                sim_id, _build_circuit_breaker_embed(sim_id, result, tripped=True)
                            )
                        elif sim_id and status == "circuit_breaker_recovered":
                            await _post_sim_event(
                                sim_id, _build_circuit_breaker_embed(sim_id, result, tripped=False)
                            )
                    except Exception:
                        logging.exception("sim_entry_notify_error: sim=%s status=%s", result.get("sim_id"), result.get("status"))
        except Exception as e:
            logging.exception("sim_entry_loop_error: %s", e)
        elapsed = (datetime.now(eastern) - iter_start).total_seconds()
        await asyncio.sleep(max(0.0, ENTRY_INTERVAL_SECONDS - elapsed))


async def sim_exit_loop() -> None:
    eastern = pytz.timezone("US/Eastern")
    while True:
        iter_start = datetime.now(eastern)
        try:
            if not _is_market_hours():
                await asyncio.sleep(60)
                continue
            from simulation.sim_engine import run_sim_exits
            global _SIM_LAST_DATA_AGE
            _SIM_LAST_DATA_AGE = _get_data_age_text()
            results = await run_sim_exits()
            for result in results:
                logging.debug("sim_exit_result: %s", result)
                try:
                    sim_id = result.get("sim_id")
                    status = result.get("status")
                    if sim_id and status == "closed":
                        exit_embed = _build_exit_embed(sim_id, result)
                        await _post_sim_event(sim_id, exit_embed)
                        if _ws_log_event:
                            try:
                                _ws_log_event("trade_exit", {
                                    "sim_id": sim_id,
                                    "exit_reason": result.get("exit_reason"),
                                    "pnl": result.get("pnl"),
                                })
                            except Exception:
                                pass
                        try:
                            from analytics.grader import check_predictions
                            check_predictions(result)
                        except Exception:
                            logging.exception("sim_prediction_grader_error")
                except Exception:
                    pass
        except Exception as e:
            logging.exception("sim_exit_loop_error: %s", e)
        elapsed = (datetime.now(eastern) - iter_start).total_seconds()
        await asyncio.sleep(max(0.0, EXIT_INTERVAL_SECONDS - elapsed))
