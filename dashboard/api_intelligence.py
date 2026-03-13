"""
dashboard/api_intelligence.py

Intelligence endpoints for the dashboard — strategy rankings, P&L summaries,
predictor analytics, and trade narratives.
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytz
from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SIM_DIR = os.path.join(DATA_DIR, "sims")
DB_PATH = os.path.join(DATA_DIR, "analytics.db")
RANKINGS_PATH = os.path.join(DATA_DIR, "strategy_rankings.json")
HEARTBEAT_PATH = os.path.join(DATA_DIR, "heartbeat.json")

ET = pytz.timezone("US/Eastern")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_sim_trades() -> list[dict]:
    """Load closed trades from all sim JSON files."""
    trades = []
    if not os.path.isdir(SIM_DIR):
        return trades
    for fname in os.listdir(SIM_DIR):
        if not fname.endswith(".json") or not fname.startswith("SIM"):
            continue
        try:
            with open(os.path.join(SIM_DIR, fname), "r") as f:
                data = json.load(f)
            sim_id = data.get("sim_id", fname.replace(".json", ""))
            for t in data.get("trade_log", []):
                if isinstance(t, dict):
                    t.setdefault("sim_id", sim_id)
                    trades.append(t)
        except Exception:
            continue
    return trades


def _load_all_sim_summaries() -> dict:
    """Load balance, open_trades count, is_dead from all sim JSON files."""
    summaries = {}
    if not os.path.isdir(SIM_DIR):
        return summaries
    for fname in os.listdir(SIM_DIR):
        if not fname.endswith(".json") or not fname.startswith("SIM"):
            continue
        try:
            with open(os.path.join(SIM_DIR, fname), "r") as f:
                data = json.load(f)
            sim_id = data.get("sim_id", fname.replace(".json", ""))
            summaries[sim_id] = {
                "balance": float(data.get("balance", 0)),
                "open_trades": len(data.get("open_trades", [])),
                "is_dead": bool(data.get("is_dead", False)),
                "trade_count": len(data.get("trade_log", [])),
            }
        except Exception:
            continue
    return summaries


def _build_trade_narrative(trade: dict) -> str:
    """Generate a human-readable sentence explaining a trade."""
    symbol = trade.get("symbol") or trade.get("trade_symbol") or "SPY"
    direction = str(trade.get("direction", "")).upper()
    signal_mode = str(trade.get("signal_mode", "")).replace("_", " ").title()
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price")
    pnl = trade.get("realized_pnl_dollars")
    exit_reason = str(trade.get("exit_reason", ""))
    hold_seconds = trade.get("hold_seconds", 0)

    option_type = "call" if direction == "BULLISH" else "put"

    if exit_price is None or pnl is None:
        try:
            return f"Open trade: {symbol} {option_type} using {signal_mode}. Entered at ${float(entry_price):.2f}."
        except (TypeError, ValueError):
            return f"Open trade: {symbol} {option_type} using {signal_mode}."

    try:
        pnl_val = float(pnl)
        result = "won" if pnl_val > 0 else "lost"
        pnl_str = f"${abs(pnl_val):.2f}"
    except (TypeError, ValueError):
        return f"{symbol} {option_type} using {signal_mode}: completed."

    try:
        minutes = float(hold_seconds) / 60
        hold_str = f" in {minutes:.0f} minutes"
    except (TypeError, ValueError):
        hold_str = ""

    exit_desc = {
        "stop_loss": "hit the stop-loss",
        "profit_target": "hit the profit target",
        "trailing_stop": "was caught by the trailing stop",
        "hold_max": "reached max hold time",
        "eod_daytrade_close": "was closed at end of day",
        "expiry_close": "was closed before expiry",
        "exit_calc_error": "was force-closed due to a data error",
        "expired_worthless": "expired worthless",
    }.get(exit_reason, f"exited ({exit_reason})" if exit_reason else "was closed")

    return (
        f"{symbol} {option_type} using {signal_mode}: "
        f"{result} {pnl_str}{hold_str}. "
        f"The trade {exit_desc}."
    )


def _compute_pnl_summary(trades: list[dict]) -> dict:
    """Compute P&L summaries from trade list."""
    now = datetime.now(ET)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=now.weekday())
    month_start = today_start.replace(day=1)

    today_pnl = 0.0
    week_pnl = 0.0
    month_pnl = 0.0
    all_pnl = 0.0
    trades_today = 0

    for t in trades:
        pnl = t.get("realized_pnl_dollars")
        if pnl is None:
            continue
        try:
            pnl_val = float(pnl)
        except (TypeError, ValueError):
            continue
        all_pnl += pnl_val

        exit_time_str = t.get("exit_time")
        if not exit_time_str:
            continue
        try:
            et = datetime.fromisoformat(str(exit_time_str))
            if et.tzinfo is None:
                et = ET.localize(et)
        except Exception:
            continue

        if et >= today_start:
            today_pnl += pnl_val
            trades_today += 1
        if et >= week_start:
            week_pnl += pnl_val
        if et >= month_start:
            month_pnl += pnl_val

    direction = "up" if today_pnl >= 0 else "down"
    sentence = f"Today you're {direction} ${abs(today_pnl):.2f} across {trades_today} trades."

    return {
        "today": round(today_pnl, 2),
        "this_week": round(week_pnl, 2),
        "this_month": round(month_pnl, 2),
        "all_time": round(all_pnl, 2),
        "trades_today": trades_today,
        "today_sentence": sentence,
    }


def _get_highlights(trades: list[dict]) -> list[str]:
    """Get today's best/worst trades as plain-English highlights."""
    now = datetime.now(ET)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_trades = []

    for t in trades:
        pnl = t.get("realized_pnl_dollars")
        if pnl is None:
            continue
        exit_time_str = t.get("exit_time")
        if not exit_time_str:
            continue
        try:
            et = datetime.fromisoformat(str(exit_time_str))
            if et.tzinfo is None:
                et = ET.localize(et)
            if et >= today_start:
                today_trades.append(t)
        except Exception:
            continue

    if not today_trades:
        return ["No trades completed today yet."]

    highlights = []
    best = max(today_trades, key=lambda t: float(t.get("realized_pnl_dollars", 0)))
    worst = min(today_trades, key=lambda t: float(t.get("realized_pnl_dollars", 0)))

    best_pnl = float(best.get("realized_pnl_dollars", 0))
    if best_pnl > 0:
        sym = best.get("symbol") or "SPY"
        mode = str(best.get("signal_mode", "")).replace("_", " ").title()
        highlights.append(f"Best trade: {best.get('sim_id', '?')} +${best_pnl:.2f} on {sym} ({mode})")

    worst_pnl = float(worst.get("realized_pnl_dollars", 0))
    if worst_pnl < 0:
        sym = worst.get("symbol") or "SPY"
        mode = str(worst.get("signal_mode", "")).replace("_", " ").title()
        highlights.append(f"Worst trade: {worst.get('sim_id', '?')} -${abs(worst_pnl):.2f} on {sym} ({mode})")

    highlights.append(f"{len(today_trades)} trades completed today.")
    return highlights


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/summary")
async def intelligence_summary():
    """Top-level intelligence summary for dashboard banner."""
    try:
        trades = await asyncio.to_thread(_load_all_sim_trades)
        summaries = await asyncio.to_thread(_load_all_sim_summaries)

        pnl = _compute_pnl_summary(trades)

        # Health
        active_sims = sum(1 for s in summaries.values() if not s.get("is_dead"))
        open_trades = sum(s.get("open_trades", 0) for s in summaries.values())
        uptime_str = "unknown"
        try:
            if os.path.exists(HEARTBEAT_PATH):
                with open(HEARTBEAT_PATH) as f:
                    hb = json.load(f)
                started = hb.get("started_at") or hb.get("timestamp")
                if started:
                    started_dt = datetime.fromisoformat(str(started))
                    if started_dt.tzinfo is None:
                        started_dt = ET.localize(started_dt)
                    delta = datetime.now(ET) - started_dt
                    hours = int(delta.total_seconds() // 3600)
                    mins = int((delta.total_seconds() % 3600) // 60)
                    uptime_str = f"{hours // 24}d {hours % 24}h {mins}m" if hours >= 24 else f"{hours}h {mins}m"
        except Exception:
            pass

        # Predictor
        predictor_accuracy = None
        predictor_sentence = "Predictor data unavailable."
        try:
            from analytics.ml_accuracy import ml_rolling_accuracy
            ml_data = ml_rolling_accuracy(200)
            if isinstance(ml_data, dict):
                predictor_accuracy = ml_data.get("accuracy")
                if predictor_accuracy is not None:
                    predictor_sentence = f"The directional predictor is calling {predictor_accuracy*100:.1f}% of trades correctly."
        except Exception:
            pass

        # Market
        regime = "UNKNOWN"
        try:
            from signals.regime import get_regime
            from core.data_service import get_market_dataframe
            df = get_market_dataframe()
            if df is not None and len(df) > 30:
                regime = str(get_regime(df) or "UNKNOWN")
        except Exception:
            pass

        # Strategy optimizer top pick
        strategy_optimizer = {"current_pick": None, "score": None, "sentence": "No strategy rankings available."}
        try:
            if os.path.exists(RANKINGS_PATH):
                with open(RANKINGS_PATH) as f:
                    rdata = json.load(f)
                rankings = rdata.get("rankings", [])
                if rankings:
                    top = rankings[0]
                    mode = top["signal_mode"]
                    score = top["score"]
                    wr = top.get("win_rate", 0)
                    n = top.get("trade_count", 0)
                    runner = rankings[1]["signal_mode"] if len(rankings) > 1 else "none"
                    runner_score = rankings[1]["score"] if len(rankings) > 1 else 0
                    strategy_optimizer = {
                        "current_pick": mode,
                        "score": score,
                        "sentence": f"SIM09 is using {mode} (score: {score}/100, {wr:.0%} WR, {n} trades). Runner-up: {runner} ({runner_score}).",
                        "rankings": rankings[:5],
                    }
        except Exception:
            pass

        highlights = _get_highlights(trades)

        return {
            "timestamp": datetime.now(ET).isoformat(),
            "pnl": pnl,
            "health": {
                "status": "HEALTHY",
                "uptime": uptime_str,
                "active_sims": active_sims,
                "dead_sims": len(summaries) - active_sims,
                "open_trades": open_trades,
            },
            "intelligence": {
                "predictor_accuracy": predictor_accuracy,
                "predictor_sentence": predictor_sentence,
            },
            "market": {
                "regime": regime,
            },
            "strategy_optimizer": strategy_optimizer,
            "highlights": highlights,
        }
    except Exception as e:
        logger.error("intelligence_summary_error: %s", e, exc_info=True)
        return {"error": str(e)}


@router.get("/strategy-rankings")
async def strategy_rankings():
    """Current strategy performance rankings."""
    try:
        if os.path.exists(RANKINGS_PATH):
            def _read():
                with open(RANKINGS_PATH) as f:
                    return json.load(f)
            return await asyncio.to_thread(_read)
    except Exception:
        pass

    # Fallback: compute from evaluator
    try:
        from simulation.strategy_evaluator import evaluate_strategies
        rankings = await asyncio.to_thread(evaluate_strategies)
        return {
            "timestamp": datetime.now(ET).isoformat(),
            "rankings": rankings,
        }
    except Exception as e:
        return {"rankings": [], "timestamp": None, "error": str(e)}


@router.get("/predictor-stats")
async def predictor_stats():
    """Predictor accuracy breakdown by hour, confidence, and trend."""
    def _query():
        if not os.path.exists(DB_PATH):
            return {"error": "analytics.db not found"}

        conn = sqlite3.connect(DB_PATH)
        try:
            # Overall accuracy
            overall = pd.read_sql(
                "SELECT AVG(correct) as accuracy, COUNT(*) as total FROM predictions WHERE correct IS NOT NULL",
                conn,
            )

            # By hour — `time` column contains ISO timestamp strings
            by_hour = pd.read_sql("""
                SELECT CAST(substr(time, 12, 2) AS INTEGER) as hour,
                       AVG(correct) as accuracy, COUNT(*) as n
                FROM predictions
                WHERE correct IS NOT NULL AND direction IN ('bullish', 'bearish')
                GROUP BY hour HAVING n >= 20
                ORDER BY hour
            """, conn)

            # By confidence band
            by_conf = pd.read_sql("""
                SELECT CASE
                    WHEN confidence < 0.40 THEN 'Low (<0.40)'
                    WHEN confidence < 0.55 THEN 'Mid (0.40-0.55)'
                    WHEN confidence < 0.70 THEN 'High (0.55-0.70)'
                    ELSE 'Very High (>=0.70)'
                END as band, AVG(correct) as accuracy, COUNT(*) as n
                FROM predictions
                WHERE correct IS NOT NULL
                GROUP BY band
                ORDER BY band
            """, conn)

            # Recent trend
            recent = pd.read_sql(
                "SELECT AVG(correct) as accuracy FROM predictions WHERE time >= datetime('now', '-7 days') AND correct IS NOT NULL",
                conn,
            )
            previous = pd.read_sql(
                "SELECT AVG(correct) as accuracy FROM predictions WHERE time >= datetime('now', '-14 days') AND time < datetime('now', '-7 days') AND correct IS NOT NULL",
                conn,
            )

            recent_acc = float(recent.iloc[0]["accuracy"]) if not recent.empty and recent.iloc[0]["accuracy"] is not None else None
            prev_acc = float(previous.iloc[0]["accuracy"]) if not previous.empty and previous.iloc[0]["accuracy"] is not None else None

            trend = "stable"
            if recent_acc is not None and prev_acc is not None:
                diff = recent_acc - prev_acc
                if diff > 0.02:
                    trend = "improving"
                elif diff < -0.02:
                    trend = "declining"

            return {
                "overall_accuracy": float(overall.iloc[0]["accuracy"]) if overall.iloc[0]["accuracy"] is not None else None,
                "total_predictions": int(overall.iloc[0]["total"]),
                "by_hour": by_hour.to_dict(orient="records"),
                "by_confidence": by_conf.to_dict(orient="records"),
                "recent_7d_accuracy": recent_acc,
                "previous_7d_accuracy": prev_acc,
                "trend": trend,
            }
        finally:
            conn.close()

    try:
        return await asyncio.to_thread(_query)
    except Exception as e:
        return {"error": str(e)}


@router.get("/trade-narrative")
async def trade_narrative(limit: int = Query(default=15, le=50)):
    """Recent trades with plain-English explanations."""
    try:
        all_trades = await asyncio.to_thread(_load_all_sim_trades)

        # Sort by exit_time descending
        def _sort_key(t):
            et = t.get("exit_time") or t.get("entry_time") or ""
            return str(et)

        all_trades.sort(key=_sort_key, reverse=True)

        result = []
        for trade in all_trades[:limit]:
            narrative = _build_trade_narrative(trade)
            pnl = trade.get("realized_pnl_dollars")
            try:
                pnl_val = float(pnl) if pnl is not None else None
            except (TypeError, ValueError):
                pnl_val = None

            result.append({
                "sim_id": trade.get("sim_id", "?"),
                "symbol": trade.get("symbol") or trade.get("trade_symbol") or "SPY",
                "direction": str(trade.get("direction", "")).upper(),
                "signal_mode": str(trade.get("signal_mode", "")),
                "pnl": pnl_val,
                "exit_reason": trade.get("exit_reason", ""),
                "exit_time": trade.get("exit_time", ""),
                "entry_time": trade.get("entry_time", ""),
                "narrative": narrative,
            })

        return {"trades": result}
    except Exception as e:
        return {"trades": [], "error": str(e)}
