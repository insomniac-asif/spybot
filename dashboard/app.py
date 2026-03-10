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
async def get_chart(symbol: str = "SPY", bars: int = 60):
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

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
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

        recent = df[df[ts_col] >= cutoff].tail(30)

        # If no predictions today, fall back to the last day that has data
        if recent.empty and not df.empty:
            _last_date = df[ts_col].dt.date.iloc[-1]
            if df[ts_col].dt.tz is not None:
                _last_cutoff = pd.Timestamp(_last_date, tz=_et_tz)
            else:
                _last_cutoff = pd.Timestamp(_last_date)
            recent = df[df[ts_col] >= _last_cutoff].tail(30)

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
    """Return the symbol registry from sim_config.yaml."""
    try:
        import sys
        sys.path.insert(0, BASE_DIR)
        from core.data_service import _load_symbol_registry
        return _load_symbol_registry()
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    print(f"Starting SpyBot Dashboard on http://0.0.0.0:{port}")
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=port, reload=False)
