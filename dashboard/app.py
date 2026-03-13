"""
SpyBot Classroom Dashboard
--------------------------
Run from repo root:
    python -m dashboard.app
or:
    uvicorn dashboard.app:app --host 0.0.0.0 --port 8080 --reload

Access at: http://<laptop-ip>:8080
"""

import asyncio
import json
import os
import re
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from dashboard.app_helpers import (
    _cleanup_trade_files,
    _load_config,
    _load_sim,
    _compute_stats,
    _is_sim_disabled,
    _safe_float,
    _parse_underlying,
    _to_naive_et,
    _get_candle_window,
    _get_trade_by_id,
    _parse_occ,
    _handle_get_chart,
    _handle_get_recent_trades,
    _handle_get_trade_history,
    NARRATIVES_DIR,
    CHARTS_DIR,
)
from dashboard.app_helpers2 import _handle_get_sim
from dashboard.api_intelligence import router as intelligence_router

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(BASE_DIR, "data", "sims")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "data", "predictions.csv")
CANDLES_CSV = os.path.join(BASE_DIR, "data")  # per-symbol CSVs in data/ dir
CONFIG_PATH = os.path.join(BASE_DIR, "simulation", "sim_config.yaml")
HEARTBEAT_PATH = os.path.join(BASE_DIR, "data", "heartbeat.json")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(_cleanup_trade_files())
    yield
    task.cancel()

app = FastAPI(title="SpyBot Dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
app.include_router(intelligence_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def cache_and_security_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path

    # Security headers on all responses
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Vary"] = "Accept-Encoding"

    # Skip websocket upgrades
    if request.headers.get("upgrade", "").lower() == "websocket":
        return response

    # API routes — never cache
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["CDN-Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    # Static assets — cache aggressively (cache-busted via ?v= query param)
    elif path.startswith("/static/") and not path.endswith(".html"):
        response.headers["Cache-Control"] = "public, max-age=86400, immutable"
        response.headers["CDN-Cache-Control"] = "public, max-age=604800"
    # HTML pages — always revalidate
    elif path == "/" or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        response.headers["CDN-Cache-Control"] = "no-cache"
    # Fallback
    else:
        response.headers["Cache-Control"] = "no-cache"

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/sims")
async def list_sims():
    config = _load_config()
    results = []
    for sim_id, profile in config.items():
        if str(sim_id).startswith("_") or not isinstance(profile, dict):
            continue
        if not re.match(r'^SIM\d+$', str(sim_id).upper()):
            continue
        data = _load_sim(sim_id)
        _cfg_syms = profile.get("symbols") or ([profile.get("symbol")] if profile.get("symbol") else [])
        if data is None:
            results.append({
                "sim_id": sim_id,
                "name": profile.get("name", sim_id),
                "signal_mode": profile.get("signal_mode", ""),
                "features_enabled": bool(profile.get("features_enabled")),
                "horizon": profile.get("horizon", ""),
                "dte_min": profile.get("dte_min"),
                "dte_max": profile.get("dte_max"),
                "symbols": _cfg_syms,
                "balance": float(profile.get("balance_start", 25000)),
                "balance_start": float(profile.get("balance_start", 25000)),
                "pnl_dollars": 0.0,
                "pnl_pct": 0.0,
                "total_trades": 0,
                "win_rate": None,
                "avg_pnl": None,
                "total_pnl": 0.0,
                "daily_loss": 0.0,
                "open_count": 0,
                "open_trade": None,
                "best_trade": None,
                "worst_trade": None,
                "max_drawdown_pct": 0.0,
                "symbol_stats": {},
                "no_data": True,
                "is_disabled": _is_sim_disabled(profile),
                "stop_loss_pct": profile.get("stop_loss_pct"),
                "profit_target_pct": profile.get("profit_target_pct"),
                "trailing_stop_activate_pct": profile.get("trailing_stop_activate_pct"),
                "trailing_stop_trail_pct": profile.get("trailing_stop_trail_pct"),
                "risk_per_trade_pct": profile.get("risk_per_trade_pct"),
                "daily_loss_limit_pct": profile.get("daily_loss_limit_pct"),
                "regime_filter": profile.get("regime_filter"),
                "hold_min_seconds": profile.get("hold_min_seconds"),
                "hold_max_seconds": profile.get("hold_max_seconds"),
                "cutoff_time_et": profile.get("cutoff_time_et"),
                "otm_pct": profile.get("otm_pct"),
                "session_filter": profile.get("session_filter"),
                "blocked_sessions": profile.get("blocked_sessions"),
                "execution_mode": profile.get("execution_mode"),
                "enabled": profile.get("enabled", True),
            })
        else:
            s = _compute_stats(sim_id, data, profile)
            results.append(s)

    results.sort(key=lambda x: int(x["sim_id"].replace("SIM", ""))
                 if x["sim_id"].replace("SIM", "").isdigit() else 0)
    return results


@app.get("/api/sim/{sim_id}")
async def get_sim(sim_id: str):
    sim_id = sim_id.upper()
    result = _handle_get_sim(sim_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Sim not found")
    return result


@app.get("/api/chart")
async def get_chart(symbol: str = None, bars: int = 480):
    """Return last N 1-min candles for a symbol using live data_service (never stale CSV)."""
    return await _handle_get_chart(symbol, bars)


@app.get("/api/predictions")
async def get_predictions(symbol: str = None):
    """Return last 30 minutes of predictions from SQLite, optionally filtered by symbol."""
    try:
        from core.analytics_db import read_df as _db_read_df
        import asyncio

        # Only load today's predictions (or last available day) — never the full 160K+ table
        _et_tz = pytz.timezone("America/New_York")
        _today_str = datetime.now(_et_tz).strftime("%Y-%m-%d")

        if symbol:
            df = await asyncio.to_thread(
                _db_read_df,
                "SELECT * FROM predictions WHERE UPPER(symbol) = ? AND time >= ?",
                [symbol.upper(), _today_str],
            )
            # Fall back to last available day if no predictions today
            if df.empty:
                df = await asyncio.to_thread(
                    _db_read_df,
                    "SELECT * FROM predictions WHERE UPPER(symbol) = ? ORDER BY time DESC LIMIT 200",
                    [symbol.upper()],
                )
        else:
            df = await asyncio.to_thread(
                _db_read_df,
                "SELECT * FROM predictions WHERE time >= ?",
                [_today_str],
            )
            if df.empty:
                df = await asyncio.to_thread(
                    _db_read_df,
                    "SELECT * FROM predictions ORDER BY time DESC LIMIT 500",
                )

        if df.empty:
            return {"predictions": [], "latest": None}

        ts_col = "time"
        df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", errors="coerce")
        df = df.dropna(subset=[ts_col])
        df = df.sort_values(ts_col)

        if df.empty:
            return {"predictions": [], "latest": None}

        # Use today's ET date as cutoff — fall back to last day with predictions if today is empty
        _et_tz = pytz.timezone("America/New_York")
        _today_date = datetime.now(_et_tz).date()
        cutoff = pd.Timestamp(_today_date)

        recent = df[df[ts_col] >= cutoff]

        # If no predictions today, fall back to the last day that has data
        if recent.empty and not df.empty:
            _last_date = df[ts_col].dt.date.iloc[-1]
            _last_cutoff = pd.Timestamp(_last_date)
            recent = df[df[ts_col] >= _last_cutoff]

        # Deduplicate: backfill writes predictions at exact :00.000000 timestamps
        # from potentially stale CSV data, while the live watcher writes at
        # fractional-second timestamps with fresh prices. For today's predictions,
        # drop backfill rows entirely when live rows exist for the same symbol.
        # For historical dates, keep backfill rows (they're the only source).
        if not recent.empty and "symbol" in recent.columns:
            recent = recent.copy()
            _today = pd.Timestamp(datetime.now(pytz.timezone("America/New_York")).date())
            _is_today = recent[ts_col] >= _today
            _is_backfill = recent[ts_col].dt.microsecond == 0
            # For today: if any live (non-backfill) rows exist for a symbol, drop all backfill rows for it
            if _is_today.any():
                _today_df = recent[_is_today]
                _live_symbols = set(_today_df.loc[~_is_backfill[_is_today], "symbol"]) if "symbol" in _today_df.columns else set()
                if _live_symbols:
                    _drop_mask = _is_today & _is_backfill & recent["symbol"].isin(_live_symbols)
                    recent = recent[~_drop_mask]

        preds = []
        for _, row in recent.iterrows():
            ts_val = row[ts_col]
            if pd.notnull(ts_val) and ts_val.tzinfo is None:
                ts_val = ts_val.tz_localize(pytz.timezone("America/New_York"), ambiguous=True, nonexistent="shift_forward")
            entry = {"time": ts_val.isoformat()}
            for col in df.columns:
                if col in (ts_col, "id"):
                    continue
                val = row[col]
                try:
                    entry[col] = float(val) if pd.notna(val) and val != "" else None
                except Exception:
                    entry[col] = str(val) if pd.notna(val) else None
            preds.append(entry)

        latest = preds[-1] if preds else None
        return {"predictions": preds, "latest": latest}
    except Exception as e:
        return {"predictions": [], "latest": None, "error": str(e)}


@app.get("/api/status")
async def get_status():
    """Return bot heartbeat and market status."""
    alive = False
    age_seconds = None
    pid = None
    last_heartbeat = None
    try:
        if os.path.exists(HEARTBEAT_PATH):
            with open(HEARTBEAT_PATH, "r") as f:
                hb = json.load(f)
            ts = hb.get("ts")
            pid = hb.get("pid")
            if ts:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = pytz.timezone("US/Eastern").localize(dt)
                age_seconds = round((datetime.now(pytz.utc) - dt.astimezone(pytz.utc)).total_seconds())
                alive = age_seconds < 180
                last_heartbeat = ts
    except Exception:
        pass

    # Market open check (ET 9:30–16:00 weekdays)
    et = datetime.now(pytz.timezone("US/Eastern"))
    weekday = et.weekday()  # 0=Mon, 4=Fri
    market_open = (
        weekday < 5
        and (et.hour, et.minute) >= (9, 30)
        and et.hour < 16
    )
    return {
        "alive": alive,
        "age_seconds": age_seconds,
        "pid": pid,
        "last_heartbeat": last_heartbeat,
        "market_open": market_open,
        "market_time": et.strftime("%H:%M ET"),
        "market_date": et.strftime("%a %b %d"),
    }


@app.get("/api/trades/recent")
async def get_recent_trades(
    limit: int = 200,
    sim_id: str = Query(None),
    symbol: str = Query(None),
    max_entry_price: float = Query(None),
):
    """Return closed + open trades across all sims. Open trades appear first."""
    return await _handle_get_recent_trades(limit, sim_id, symbol, max_entry_price)


@app.get("/api/symbols")
async def get_symbols():
    """Return all available symbols (from registry + convention-based CSV discovery)."""
    try:
        import sys, glob as _glob
        sys.path.insert(0, BASE_DIR)
        from core.data_service import _load_symbol_registry
        result = dict(_load_symbol_registry())
        # Discover symbols from data/*_1m.csv convention
        data_dir = os.path.join(BASE_DIR, "data")
        for csv_path in sorted(_glob.glob(os.path.join(data_dir, "*_1m.csv"))):
            sym = os.path.basename(csv_path).replace("_1m.csv", "").upper()
            if sym not in result:
                result[sym] = {"data_file": os.path.relpath(csv_path, BASE_DIR)}
        return result
    except Exception:
        return {}


@app.get("/api/trades/{sim_id}/history")
async def get_trade_history(
    sim_id: str,
    date: str = Query(None),
    direction: str = Query(None),
    result: str = Query(None),
    symbol: str = Query(None),
    page: int = Query(1),
    per_page: int = Query(50),
):
    """Full paginated trade log with optional filters."""
    sim_id = sim_id.upper()
    result_data = await _handle_get_trade_history(sim_id, date, direction, result, symbol, page, per_page)
    if result_data is None:
        raise HTTPException(status_code=404, detail="Sim not found")
    return result_data


@app.get("/api/trades/{sim_id}/{trade_id}/chart")
async def get_trade_chart(sim_id: str, trade_id: str, refresh: bool = False):
    """Return annotated trade chart as PNG."""
    sim_id = sim_id.upper()
    safe   = re.sub(r"[^\w\-]", "_", trade_id)
    cache  = os.path.join(CHARTS_DIR, f"{sim_id}_{safe}.png")

    if os.path.exists(cache) and not refresh:
        age = _time.time() - os.path.getmtime(cache)
        if age < 1800:
            return Response(content=open(cache, "rb").read(), media_type="image/png")

    trade, profile = _get_trade_by_id(sim_id, trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")

    candle_data = _get_candle_window(
        trade.get("entry_time", ""),
        trade.get("exit_time", ""),
        symbol=trade.get("symbol") or _parse_underlying(trade.get("option_symbol", "")),
    )

    # Load narrative if cached (for S/R lines)
    narrative = None
    narr_path = os.path.join(NARRATIVES_DIR, f"{sim_id}_{safe}.json")
    if os.path.exists(narr_path):
        try:
            narrative = json.loads(open(narr_path).read())
        except Exception:
            pass

    try:
        import sys
        sys.path.insert(0, BASE_DIR)
        from charts.trade_chart import generate_trade_chart
        png = generate_trade_chart(
            trade, candle_data, narrative=narrative,
            force_refresh=refresh,
        )
        if isinstance(png, str):
            png = open(png, "rb").read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {e}")

    return Response(content=png, media_type="image/png")


@app.get("/api/sim/{sim_id}/replay/{trade_index}")
async def get_replay_chart(sim_id: str, trade_index: int):
    """Return trade replay chart PNG. trade_index is 1-based from most recent closed trade."""
    sim_id = sim_id.upper()
    data = _load_sim(sim_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Sim not found: {sim_id}")

    trades = [t for t in (data.get("trade_log") or []) if t.get("exit_time")]
    if not trades:
        raise HTTPException(status_code=404, detail="No closed trades")
    if trade_index < 1 or trade_index > len(trades):
        raise HTTPException(status_code=404, detail=f"Trade index out of range (1–{len(trades)})")

    trade = trades[-trade_index]
    try:
        import sys
        sys.path.insert(0, BASE_DIR)
        from interface.charting import generate_trade_replay

        chart_path = await asyncio.to_thread(generate_trade_replay, sim_id, trade)
        if chart_path and os.path.exists(chart_path):
            return Response(content=open(chart_path, "rb").read(), media_type="image/png")

        # Fallback placeholder if candle data unavailable
        from charts.trade_chart import _render_placeholder
        placeholder = _render_placeholder(
            f"Price data unavailable for this trade's time range\n"
            f"({trade.get('entry_time', '')[:10]})",
            (800, 400),
        )
        return Response(content=placeholder, media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {exc}")


@app.get("/api/trades/{sim_id}/{trade_id}/narrative")
async def get_narrative(sim_id: str, trade_id: str):
    """Return cached GPT narrative JSON, or 404 if not yet generated."""
    sim_id = sim_id.upper()
    safe   = re.sub(r"[^\w\-]", "_", trade_id)
    path   = os.path.join(NARRATIVES_DIR, f"{sim_id}_{safe}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Narrative not yet generated")
    try:
        return json.loads(open(path).read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trades/{sim_id}/{trade_id}/narrate")
async def trigger_narrative(sim_id: str, trade_id: str, force: bool = False):
    """Trigger GPT narrative generation. Returns narrative JSON."""
    sim_id = sim_id.upper()
    trade, profile = _get_trade_by_id(sim_id, trade_id)
    if trade is None:
        raise HTTPException(status_code=404, detail="Trade not found")

    candle_data = _get_candle_window(
        trade.get("entry_time", ""),
        trade.get("exit_time", ""),
        symbol=trade.get("symbol") or _parse_underlying(trade.get("option_symbol", "")),
    )

    try:
        import sys
        sys.path.insert(0, BASE_DIR)
        from analytics.trade_narrator import narrate_trade
        narrative = await narrate_trade(trade, candle_data, profile, force_refresh=force)
        # Invalidate chart cache so it re-renders with new S/R lines
        safe  = re.sub(r"[^\w\-]", "_", trade_id)
        chart = os.path.join(CHARTS_DIR, f"{sim_id}_{safe}.png")
        if force and os.path.exists(chart):
            os.remove(chart)
        return narrative
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategy_performance")
async def api_strategy_performance():
    """Return strategy performance store data for dashboard heatmaps."""
    try:
        import sys
        sys.path.insert(0, BASE_DIR)
        from analytics.strategy_performance import PERF_STORE
        PERF_STORE._load()   # reload from disk in case exits happened since process start
        return PERF_STORE._data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Backtest endpoints
# ---------------------------------------------------------------------------

@app.get("/api/backtest/results")
async def get_backtest_results():
    path = os.path.join(BASE_DIR, "backtest", "results", "dashboard_data.json")
    if not os.path.exists(path):
        return {"error": "No backtest results found. Run: python -m backtest.runner --start YYYY-MM-DD --end YYYY-MM-DD"}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load backtest results: {e}"}


@app.get("/api/backtest/results/{sim_id}")
async def get_backtest_results_sim(sim_id: str):
    all_data = await get_backtest_results()
    if "error" in all_data:
        return all_data
    sim_id_upper = sim_id.upper()
    if sim_id_upper in all_data:
        return all_data[sim_id_upper]
    return {"error": f"No backtest data for {sim_id}"}


@app.get("/api/backtest/patterns/{sim_id}")
async def get_backtest_patterns(sim_id: str):
    path = os.path.join(BASE_DIR, "backtest", "results", f"patterns_{sim_id.upper()}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/api/backtest/growth/{sim_id}")
async def get_backtest_growth(sim_id: str):
    path = os.path.join(BASE_DIR, "backtest", "results", f"growth_{sim_id.upper()}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/api/backtest/optimizer/{sim_id}")
async def get_backtest_optimizer_results(sim_id: str):
    path = os.path.join(BASE_DIR, "backtest", "results", f"optimizer_{sim_id.upper()}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@app.get("/api/backtest/optimize/{sim_id}")
async def get_backtest_optimization(sim_id: str):
    """Return optimizer analysis for a sim. Runs analysis on-the-fly from backtest data."""
    try:
        from backtest.optimizer import analyze_sim, result_to_dict
        result = analyze_sim(sim_id.upper())
        if not result:
            return {"error": f"No backtest data to optimize for {sim_id}"}
        return result_to_dict(result)
    except Exception as e:
        return {"error": f"Optimization failed: {e}"}


# ---------------------------------------------------------------------------
# Pattern endpoints
# ---------------------------------------------------------------------------

PATTERNS_DIR = os.path.join(BASE_DIR, "research", "patterns")


@app.get("/api/patterns")
async def list_patterns():
    """Return list of sim IDs that have pattern data."""
    if not os.path.isdir(PATTERNS_DIR):
        return []
    sims = []
    for fname in sorted(os.listdir(PATTERNS_DIR)):
        if fname.endswith("_patterns.json") and fname.startswith("SIM"):
            sims.append(fname.replace("_patterns.json", ""))
    return sims


@app.get("/api/patterns/{sim_id}")
async def get_patterns(sim_id: str):
    """Return pattern data for a sim."""
    sim_id = sim_id.upper()
    path = os.path.join(PATTERNS_DIR, f"{sim_id}_patterns.json")
    if not os.path.exists(path):
        return {"error": "No pattern data. Run backtest first."}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Greeks Analytics endpoints
# ---------------------------------------------------------------------------

@app.get("/api/greeks/overview")
async def greeks_overview():
    """Aggregate Greeks exit stats across all sims."""
    try:
        cfg = _load_config()
        profiles = {k: v for k, v in cfg.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        profiles = {}

    by_trigger = {
        "theta_burn": {"count": 0, "pnl_sum": 0, "saved": 0},
        "iv_crush": {"count": 0, "pnl_sum": 0, "saved": 0},
        "delta_erosion": {"count": 0, "pnl_sum": 0, "saved": 0},
    }
    total_trades = 0
    daily_trend = {}  # date_str -> {theta, iv, delta}

    theta_reasons = {"theta_burn", "theta_burn_0dte", "theta_burn_tightened"}
    iv_reasons = {"iv_crush_stop", "iv_crush_exit"}
    delta_reasons = {"delta_erosion"}
    all_greeks = theta_reasons | iv_reasons | delta_reasons

    for sim_id, profile in profiles.items():
        try:
            from simulation.sim_portfolio import SimPortfolio
            sim = SimPortfolio(sim_id, profile)
            sim.load()
            trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []
            total_trades += len(trade_log)

            for t in trade_log:
                reason = t.get("exit_reason", "")
                if reason not in all_greeks:
                    continue

                pnl = _safe_float(t.get("realized_pnl_dollars"), 0)
                exit_time = t.get("exit_time", "")
                day = str(exit_time)[:10]

                if reason in theta_reasons:
                    cat = "theta_burn"
                elif reason in iv_reasons:
                    cat = "iv_crush"
                else:
                    cat = "delta_erosion"

                by_trigger[cat]["count"] += 1
                by_trigger[cat]["pnl_sum"] += pnl
                if pnl <= 0:
                    by_trigger[cat]["saved"] += 1

                if day and len(day) == 10:
                    if day not in daily_trend:
                        daily_trend[day] = {"date": day, "theta": 0, "iv": 0, "delta": 0}
                    if cat == "theta_burn":
                        daily_trend[day]["theta"] += 1
                    elif cat == "iv_crush":
                        daily_trend[day]["iv"] += 1
                    else:
                        daily_trend[day]["delta"] += 1
        except Exception:
            continue

    total_greeks = sum(v["count"] for v in by_trigger.values())
    result_triggers = {}
    for trig, data in by_trigger.items():
        c = data["count"]
        result_triggers[trig] = {
            "count": c,
            "avg_pnl": round(data["pnl_sum"] / c, 2) if c > 0 else 0,
            "saved_pct": round(data["saved"] / c * 100, 1) if c > 0 else 0,
        }

    return {
        "total_greeks_exits": total_greeks,
        "total_trades": total_trades,
        "by_trigger": result_triggers,
        "greeks_exits_vs_total": f"{total_greeks}/{total_trades} ({total_greeks/total_trades*100:.1f}%)" if total_trades > 0 else "0/0",
        "daily_trend": sorted(daily_trend.values(), key=lambda x: x["date"])[-14:],
    }


@app.get("/api/greeks/sim/{sim_id}")
async def greeks_sim(sim_id: str):
    """Per-sim Greeks detail."""
    sim_id = sim_id.upper()
    try:
        cfg = _load_config()
        profile = cfg.get(sim_id, {})
    except Exception:
        profile = {}

    try:
        from analytics.adaptive_tuning import (
            evaluate_greeks_effectiveness, get_effective_threshold,
            _load_overrides, _load_tuning_log,
        )
        effectiveness = evaluate_greeks_effectiveness(sim_id, profile)
    except Exception:
        effectiveness = {}

    triggers_enabled = []
    thresholds = {}
    for trig, cfg_key, default in [
        ("theta_burn", "theta_burn_stop_tighten_pct", 0.50),
        ("iv_crush", "iv_crush_vega_multiplier", 2.0),
        ("delta_erosion", "delta_erosion_current_max", 0.20),
    ]:
        enabled_key = {"theta_burn": "theta_burn_enabled", "iv_crush": "iv_crush_exit_enabled", "delta_erosion": "delta_erosion_exit_enabled"}[trig]
        if profile.get(enabled_key, False):
            triggers_enabled.append(trig)
        try:
            thresholds[cfg_key] = get_effective_threshold(sim_id, profile, cfg_key, default)
        except Exception:
            thresholds[cfg_key] = float(profile.get(cfg_key, default))

    # Get adaptive history from tuning log
    adaptive_history = []
    try:
        log = _load_tuning_log()
        adaptive_history = [e for e in log if e.get("sim_id") == sim_id][-20:]
    except Exception:
        pass

    # Recent Greeks exits from trade log
    recent_greeks = []
    theta_reasons = {"theta_burn", "theta_burn_0dte", "theta_burn_tightened"}
    iv_reasons = {"iv_crush_stop", "iv_crush_exit"}
    delta_reasons = {"delta_erosion"}
    all_greeks = theta_reasons | iv_reasons | delta_reasons

    try:
        from simulation.sim_portfolio import SimPortfolio
        sim = SimPortfolio(sim_id, profile)
        sim.load()
        for t in reversed(sim.trade_log if isinstance(sim.trade_log, list) else []):
            reason = t.get("exit_reason", "")
            if reason in all_greeks:
                cat = "theta_burn" if reason in theta_reasons else ("iv_crush" if reason in iv_reasons else "delta_erosion")
                pnl = _safe_float(t.get("realized_pnl_dollars"), 0)
                recent_greeks.append({
                    "time": str(t.get("exit_time", ""))[:16],
                    "trigger": cat,
                    "exit_reason": reason,
                    "pnl": round(pnl, 2),
                    "saved": pnl <= 0,
                })
            if len(recent_greeks) >= 20:
                break
    except Exception:
        pass

    return {
        "sim_id": sim_id,
        "triggers_enabled": triggers_enabled,
        "current_thresholds": thresholds,
        "effectiveness": effectiveness,
        "adaptive_history": adaptive_history,
        "recent_greeks_exits": recent_greeks,
    }


@app.get("/api/greeks/tuning-log")
async def greeks_tuning_log():
    """Return full adaptive tuning log."""
    try:
        from analytics.adaptive_tuning import _load_tuning_log
        return _load_tuning_log()
    except Exception:
        return []


@app.get("/api/greeks/heatmap")
async def greeks_heatmap():
    """Per-sim heatmap data: trigger counts, save rates, composite score."""
    try:
        cfg = _load_config()
        profiles = {k: v for k, v in cfg.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return []

    theta_reasons = {"theta_burn", "theta_burn_0dte", "theta_burn_tightened"}
    iv_reasons = {"iv_crush_stop", "iv_crush_exit"}
    delta_reasons = {"delta_erosion"}
    all_greeks = theta_reasons | iv_reasons | delta_reasons

    rows = []
    for sim_id, profile in sorted(profiles.items()):
        if sim_id == "SIM00":
            continue
        try:
            from simulation.sim_portfolio import SimPortfolio
            sim = SimPortfolio(sim_id, profile)
            sim.load()
            trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []

            counts = {"theta_burn": 0, "iv_crush": 0, "delta_erosion": 0}
            saved_counts = {"theta_burn": 0, "iv_crush": 0, "delta_erosion": 0}

            for t in trade_log:
                reason = t.get("exit_reason", "")
                if reason in theta_reasons:
                    cat = "theta_burn"
                elif reason in iv_reasons:
                    cat = "iv_crush"
                elif reason in delta_reasons:
                    cat = "delta_erosion"
                else:
                    continue
                counts[cat] += 1
                if _safe_float(t.get("realized_pnl_dollars"), 0) <= 0:
                    saved_counts[cat] += 1

            total_greeks = sum(counts.values())
            total_saved = sum(saved_counts.values())

            # Get composite score
            score = None
            try:
                from analytics.composite_score import compute_composite_score
                cs = compute_composite_score(sim_id, profile)
                score = cs.get("composite_score")
            except Exception:
                pass

            rows.append({
                "sim_id": sim_id,
                "theta_count": counts["theta_burn"],
                "iv_count": counts["iv_crush"],
                "delta_count": counts["delta_erosion"],
                "total_greeks": total_greeks,
                "saved_pct": round(total_saved / total_greeks * 100, 1) if total_greeks > 0 else 0,
                "composite_score": score,
                "total_trades": len(trade_log),
            })
        except Exception:
            continue

    return rows


# ---------------------------------------------------------------------------
# Intelligence endpoints
# ---------------------------------------------------------------------------

@app.get("/api/intelligence/decision-gates")
async def intel_decision_gates():
    """Return current decision gate adjustments for all sims."""
    try:
        cfg = _load_config()
        profiles = {k: v for k, v in cfg.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return {"sims": [], "total_adjustments": 0}

    results = []
    total_adj = 0
    for sim_id, profile in sorted(profiles.items()):
        try:
            from analytics.decision_gates import get_analytics_adjustments
            adj = get_analytics_adjustments(sim_id, profile)
            reasons = adj.get("reasons", [])
            if reasons:
                total_adj += len(reasons)
                results.append({
                    "sim_id": sim_id,
                    "loosen_filters": adj.get("loosen_filters", {}),
                    "predictor_override": adj.get("predictor_override"),
                    "size_multiplier": adj.get("size_multiplier", 1.0),
                    "reasons": reasons,
                })
        except Exception:
            continue

    return {"sims": results, "total_adjustments": total_adj}


@app.get("/api/intelligence/blocked-signals")
async def intel_blocked_signals():
    """Return blocked signal stats grouped by block reason."""
    try:
        from core.analytics_db import read_df as _db_read_df
        df = await asyncio.to_thread(
            _db_read_df,
            "SELECT block_reason, fwd_5m, fwd_15m, fwd_5m_status FROM blocked_signals ORDER BY id DESC LIMIT 500"
        )
        if df.empty:
            return {"reasons": [], "total": 0}

        reasons = []
        for reason, group in df.groupby("block_reason"):
            if reason is None or str(reason).strip() == "":
                continue
            count = len(group)
            filled = group[group["fwd_5m_status"] == "filled"]
            fwd_5m_vals = pd.to_numeric(filled["fwd_5m"], errors="coerce").dropna()
            fwd_15m_vals = pd.to_numeric(filled["fwd_15m"], errors="coerce").dropna()
            would_win = len(fwd_5m_vals[fwd_5m_vals > 0]) if len(fwd_5m_vals) > 0 else 0
            win_pct = round(would_win / len(fwd_5m_vals) * 100, 1) if len(fwd_5m_vals) > 0 else None
            reasons.append({
                "reason": str(reason),
                "count": count,
                "would_win_pct": win_pct,
                "avg_fwd_5m": round(float(fwd_5m_vals.mean()), 4) if len(fwd_5m_vals) > 0 else None,
                "avg_fwd_15m": round(float(fwd_15m_vals.mean()), 4) if len(fwd_15m_vals) > 0 else None,
                "filled_count": len(filled),
                "pending_count": count - len(filled),
            })

        reasons.sort(key=lambda x: x["count"], reverse=True)
        return {"reasons": reasons, "total": len(df)}
    except Exception as e:
        return {"reasons": [], "total": 0, "error": str(e)}


@app.get("/api/intelligence/ml-accuracy")
async def intel_ml_accuracy():
    """Return ML predictor rolling accuracy."""
    try:
        from analytics.ml_accuracy import ml_rolling_accuracy
        result = await asyncio.to_thread(ml_rolling_accuracy, 200)
        if result is None:
            return {"status": "insufficient_data", "accuracy": None, "samples": 0}
        health = "healthy" if result["accuracy"] >= 40 else ("warning" if result["accuracy"] >= 28 else "critical")
        return {
            "status": health,
            "accuracy": result["accuracy"],
            "samples": result["samples"],
            "confident_accuracy": result.get("confident_accuracy"),
        }
    except Exception as e:
        return {"status": "error", "accuracy": None, "samples": 0, "error": str(e)}


@app.get("/api/intelligence/feature-drift")
async def intel_feature_drift():
    """Return feature drift detection results."""
    try:
        from analytics.feature_drift import detect_feature_drift
        result = await asyncio.to_thread(detect_feature_drift)
        if result is None:
            return {"detected": False, "severity": 0, "features": []}
        return {
            "detected": True,
            "severity": result.get("severity", 0),
            "features": result.get("features", []),
        }
    except Exception as e:
        return {"detected": False, "severity": 0, "features": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Trade Grading
# ---------------------------------------------------------------------------

@app.post("/api/grade-trade")
async def grade_trade(body: dict):
    """Grade a user-submitted trade."""
    try:
        symbol = (body.get("symbol") or "").upper()
        direction = body.get("direction", "call")
        entry_price = body.get("entry_price")
        exit_price = body.get("exit_price")
        if not symbol or entry_price is None or exit_price is None:
            return {"error": "symbol, entry_price, and exit_price are required"}

        entry_price = float(entry_price)
        exit_price = float(exit_price)
        multiplier = 1 if direction == "call" else -1
        pnl_pct = multiplier * (exit_price - entry_price) / entry_price * 100

        # Simple grading logic
        if pnl_pct >= 30:
            grade, feedback = "A", "Excellent trade — strong profit capture."
        elif pnl_pct >= 10:
            grade, feedback = "B", "Good trade — solid execution."
        elif pnl_pct >= 0:
            grade, feedback = "C", "Break-even or marginal gain. Consider tighter entries."
        elif pnl_pct >= -15:
            grade, feedback = "D", "Small loss. Review entry timing and stop placement."
        else:
            grade, feedback = "F", "Significant loss. Re-evaluate thesis and risk management."

        return {"grade": grade, "feedback": feedback, "pnl_pct": round(pnl_pct, 2)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Last Price Fallback
# ---------------------------------------------------------------------------

@app.get("/api/last-price")
async def last_price(symbol: str = Query("SPY")):
    """Return last known close price from CSV — never calls Alpaca."""
    try:
        from dashboard.app_helpers3 import _load_symbol_csv_only
        df = await asyncio.to_thread(_load_symbol_csv_only, symbol.upper())
        if df is not None and not df.empty:
            close_col = next((c for c in df.columns if c.lower() == 'close'), None)
            if close_col:
                last_close = float(df[close_col].iloc[-1])
                return {"symbol": symbol.upper(), "price": round(last_close, 2)}
        return {"symbol": symbol.upper(), "price": None}
    except Exception as e:
        return {"symbol": symbol.upper(), "price": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Equity Curve (Aggregate Portfolio)
# ---------------------------------------------------------------------------

@app.get("/api/equity-curve")
async def equity_curve():
    """Aggregate equity curve across all alive sims."""
    config = _load_config()
    points = {}  # date_str -> total_balance

    for sid, profile in config.items():
        if str(sid).startswith("_") or not isinstance(profile, dict):
            continue
        if not re.match(r'^SIM\d+$', str(sid).upper()):
            continue
        data = _load_sim(sid)
        if not data:
            continue
        if data.get("is_dead"):
            continue

        balance = data.get("balance", 0)
        trade_log = data.get("trade_log") or []

        # Walk trade log to build balance-over-time series
        running = float(profile.get("balance_start", 3000))
        for t in trade_log:
            pnl = t.get("realized_pnl_dollars")
            exit_time = t.get("exit_time", "")
            if pnl is None or not exit_time:
                continue
            day = exit_time[:10]  # YYYY-MM-DD
            running += float(pnl)
            points[day] = points.get(day, 0) + float(pnl)

    # Compute current total balance across alive sims (always, even with no trades)
    total_balance = 0
    alive_count = 0
    for sid, profile in config.items():
        if str(sid).startswith("_") or not isinstance(profile, dict):
            continue
        if not re.match(r'^SIM\d+$', str(sid).upper()):
            continue
        data = _load_sim(sid)
        if data and not data.get("is_dead"):
            total_balance += data.get("balance", 0)
            alive_count += 1

    # Build cumulative series
    if not points:
        return {"series": [], "total_balance": round(total_balance, 2),
                "alive_count": alive_count}

    sorted_days = sorted(points.keys())
    cumulative = 0
    series = []
    for day in sorted_days:
        cumulative += points[day]
        series.append({"date": day, "pnl": round(cumulative, 2)})

    return {"series": series, "total_balance": round(total_balance, 2),
            "alive_count": alive_count}


# ---------------------------------------------------------------------------
# Expel (Disable) a SIM
# ---------------------------------------------------------------------------

@app.post("/api/sim/{sim_id}/expel")
async def expel_sim(sim_id: str):
    """Disable a dead sim by setting blocked_sessions to all sessions in YAML config."""
    sim_id = sim_id.upper()
    data = _load_sim(sim_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"{sim_id} not found")

    config = _load_config()
    profile = config.get(sim_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"{sim_id} not in config")

    # Mark as expelled: add blocked_sessions covering all time slots
    all_sessions = ["PREMARKET", "OPENING_HOUR", "MIDDAY", "POWER_HOUR", "CLOSING"]
    profile["blocked_sessions"] = all_sessions

    # Write updated config back to YAML
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "simulation", "sim_config.yaml"
    )
    try:
        import yaml
        def _write_config():
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        await asyncio.to_thread(_write_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config: {e}")

    return {
        "status": "expelled",
        "sim_id": sim_id,
        "message": f"{sim_id} has been expelled. All trading sessions are now blocked.",
    }


# ---------------------------------------------------------------------------
# System Health
# ---------------------------------------------------------------------------

@app.get("/api/system-health")
async def system_health():
    """Return system health metrics: CPU, RAM, DB size, etc."""
    import shutil

    health = {}

    # CPU & RAM via /proc (no psutil dependency)
    try:
        pid = os.getpid()
        # RSS from /proc/self/status
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    health["ram_mb"] = round(int(line.split()[1]) / 1024, 1)
                    break
        # System RAM from /proc/meminfo
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:"):
                    meminfo[parts[0]] = int(parts[1])
            if "MemTotal:" in meminfo:
                health["ram_total_mb"] = round(meminfo["MemTotal:"] / 1024, 1)
                if "MemAvailable:" in meminfo:
                    used = meminfo["MemTotal:"] - meminfo["MemAvailable:"]
                    health["ram_used_pct"] = round(used / meminfo["MemTotal:"] * 100, 1)
        # CPU load average (1-min)
        load1 = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        health["cpu_pct"] = round(load1 / cpu_count * 100, 1)
    except Exception:
        health["ram_mb"] = None
        health["cpu_pct"] = None

    # Database size
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "analytics.db"
    )
    try:
        if os.path.exists(db_path):
            health["db_size_mb"] = round(os.path.getsize(db_path) / 1024 / 1024, 1)
        else:
            health["db_size_mb"] = 0
    except Exception:
        health["db_size_mb"] = None

    # SQLite WAL size
    wal_path = db_path + "-wal"
    try:
        if os.path.exists(wal_path):
            health["db_wal_mb"] = round(os.path.getsize(wal_path) / 1024 / 1024, 1)
        else:
            health["db_wal_mb"] = 0
    except Exception:
        health["db_wal_mb"] = 0

    # Disk space
    try:
        usage = shutil.disk_usage(os.path.dirname(db_path))
        health["disk_free_gb"] = round(usage.free / 1024 / 1024 / 1024, 1)
        health["disk_total_gb"] = round(usage.total / 1024 / 1024 / 1024, 1)
    except Exception:
        health["disk_free_gb"] = None

    # Alpaca API rate limit heuristic: check last call timestamp
    try:
        from core.data_service import _last_alpaca_call_ts
        import time
        elapsed = time.time() - _last_alpaca_call_ts if _last_alpaca_call_ts else None
        health["alpaca_last_call_ago_sec"] = round(elapsed, 1) if elapsed else None
    except Exception:
        health["alpaca_last_call_ago_sec"] = None

    # Sim state summary
    config = _load_config()
    alive = 0
    dead = 0
    total_balance = 0
    for sid, profile in config.items():
        if str(sid).startswith("_") or not isinstance(profile, dict):
            continue
        if not re.match(r'^SIM\d+$', str(sid).upper()):
            continue
        d = _load_sim(sid)
        if d:
            if d.get("is_dead"):
                dead += 1
            else:
                alive += 1
                total_balance += d.get("balance", 0)
    health["sims_alive"] = alive
    health["sims_dead"] = dead
    health["total_balance"] = round(total_balance, 2)

    return health


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    print(f"Starting SpyBot Dashboard on http://0.0.0.0:{port}")
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=port, reload=False)
