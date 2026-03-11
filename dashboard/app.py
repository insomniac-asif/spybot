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

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(BASE_DIR, "data", "sims")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "data", "predictions.csv")
CANDLES_CSV = os.path.join(BASE_DIR, "data", "spy_1m.csv")
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
        _cfg_syms = profile.get("symbols") or ([profile.get("symbol")] if profile.get("symbol") else ["SPY"])
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
async def get_chart(symbol: str = "SPY", bars: int = 480):
    """Return last N 1-min candles for a symbol using live data_service (never stale CSV)."""
    return await _handle_get_chart(symbol, bars)


@app.get("/api/predictions")
async def get_predictions(symbol: str = None):
    """Return last 30 minutes of predictions from predictions.csv, optionally filtered by symbol."""
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        df.columns = [c.lower() for c in df.columns]
        ts_col = next((c for c in ("time", "timestamp", "datetime") if c in df.columns), None)
        if ts_col is None:
            return {"predictions": [], "latest": None}

        df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", errors="coerce")
        df = df.dropna(subset=[ts_col])
        df = df.sort_values(ts_col)

        # Symbol filter — rows without "symbol" column are treated as SPY (legacy)
        if symbol:
            sym_upper = symbol.upper()
            if "symbol" in df.columns:
                df = df[df["symbol"].fillna("SPY").str.upper() == sym_upper]
            elif sym_upper != "SPY":
                # No symbol column yet and requesting non-SPY → no results
                return {"predictions": [], "latest": None, "symbol": sym_upper}

        # Use today's ET date as cutoff — fall back to last day with predictions if today is empty
        _et_tz = pytz.timezone("America/New_York")
        _today_date = datetime.now(_et_tz).date()
        if df[ts_col].dt.tz is not None:
            cutoff = pd.Timestamp(_today_date, tz=_et_tz)
        else:
            cutoff = pd.Timestamp(_today_date)

        recent = df[df[ts_col] >= cutoff]

        # If no predictions today, fall back to the last day that has data
        if recent.empty and not df.empty:
            _last_date = df[ts_col].dt.date.iloc[-1]
            if df[ts_col].dt.tz is not None:
                _last_cutoff = pd.Timestamp(_last_date, tz=_et_tz)
            else:
                _last_cutoff = pd.Timestamp(_last_date)
            recent = df[df[ts_col] >= _last_cutoff]

        preds = []
        for _, row in recent.iterrows():
            ts_val = row[ts_col]
            if pd.notnull(ts_val) and ts_val.tzinfo is None:
                ts_val = ts_val.tz_localize(pytz.timezone("America/New_York"), ambiguous=True, nonexistent="shift_forward")
            entry = {"time": ts_val.isoformat()}
            for col in df.columns:
                if col == ts_col:
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    print(f"Starting SpyBot Dashboard on http://0.0.0.0:{port}")
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=port, reload=False)
