"""
dashboard/app_helpers3.py
--------------------------
Overflow helpers for dashboard/app.py — extracted route bodies too large to
fit in app_helpers.py without exceeding the 500-line limit.
"""
import asyncio
import os
import re
from typing import Optional

import pandas as pd

from dashboard.app_helpers import (
    _load_config,
    _load_sim,
    _safe_float,
    _parse_underlying,
    _parse_occ,
    _to_naive_et,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NARRATIVES_DIR = os.path.join(BASE_DIR, "data", "trade_narratives")


async def _handle_get_chart(symbol: str, bars: int) -> dict:
    """Body of GET /api/chart — returns candle dict."""
    sym = symbol.upper()
    try:
        from core.data_service import get_symbol_dataframe
        df = await asyncio.to_thread(get_symbol_dataframe, sym)

        if df is None or df.empty:
            return {"candles": [], "symbol": sym, "error": f"no_data_{sym}"}

        df = df.reset_index()
        if "index" in df.columns and df["index"].dtype != "datetime64[ns]" and not hasattr(df["index"].dtype, "tz"):
            df = df.drop(columns=["index"])
        ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
        if ts_col is None:
            return {"candles": [], "symbol": sym, "error": "no_timestamp_col"}

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        import pytz as _pytz
        _et = _pytz.timezone("America/New_York")
        if df[ts_col].dt.tz is None:
            df[ts_col] = df[ts_col].dt.tz_localize(_et, ambiguous="infer", nonexistent="shift_forward")
        else:
            df[ts_col] = df[ts_col].dt.tz_convert(_et)

        _now_et = pd.Timestamp.now(tz=_et)
        _today_et = _now_et.date()
        _today_start = pd.Timestamp(_today_et, tz=_et)
        _today_end   = _today_start + pd.Timedelta(days=1)
        df_today = df[(df[ts_col] >= _today_start) & (df[ts_col] < _today_end)]
        if len(df_today) >= 10:
            df = df_today
        else:
            _last_date = df[ts_col].dt.date.iloc[-1]
            _last_start = pd.Timestamp(_last_date, tz=_et)
            _last_end   = _last_start + pd.Timedelta(days=1)
            df_last = df[(df[ts_col] >= _last_start) & (df[ts_col] < _last_end)]
            if len(df_last) >= 10:
                df = df_last
            else:
                # Neither today nor last date had enough data — use tail as fallback
                df = df.tail(max(bars, 1))
        candles = []
        for _, row in df.iterrows():
            candles.append({
                "t": row[ts_col].isoformat(),
                "o": round(_safe_float(row.get("open")), 4),
                "h": round(_safe_float(row.get("high")), 4),
                "l": round(_safe_float(row.get("low")), 4),
                "c": round(_safe_float(row.get("close")), 4),
                "v": int(_safe_float(row.get("volume", 0))),
            })
        return {"candles": candles, "symbol": sym}
    except Exception as e:
        return {"candles": [], "symbol": sym, "error": str(e)}


async def _handle_get_recent_trades(
    limit: int,
    sim_id_filter: Optional[str],
    symbol_filter: Optional[str],
    max_entry_price: Optional[float],
) -> dict:
    """Body of GET /api/trades/recent — returns trade list dict."""
    config = _load_config()
    closed_trades, open_trades_out = [], []

    for sid, profile in config.items():
        if str(sid).startswith("_") or not isinstance(profile, dict):
            continue
        if not re.match(r'^SIM\d+$', str(sid).upper()):
            continue
        if sim_id_filter and sid.upper() != sim_id_filter.upper():
            continue
        data = _load_sim(sid)
        if not data:
            continue
        _sl_pct = _safe_float(profile.get("stop_loss_pct"), 0)
        _tp_pct = _safe_float(profile.get("profit_target_pct"), 0)

        def _make_record(t, status):
            sym = (t.get("symbol") or _parse_underlying(t.get("option_symbol", ""))).upper()
            if symbol_filter and sym != symbol_filter.upper():
                return None
            ep = t.get("entry_price")
            if max_entry_price is not None and ep is not None and float(ep) > max_entry_price:
                return None
            opt_sym = t.get("option_symbol", "")
            parsed = _parse_occ(opt_sym)
            sl_price = round(ep * (1 - _sl_pct), 4) if ep and _sl_pct else None
            tp_price = round(ep * (1 + _tp_pct), 4) if ep and _tp_pct else None
            return {
                "status": status,
                "sim_id": sid,
                "trade_id": t.get("trade_id", ""),
                "symbol": sym,
                "direction": t.get("direction", ""),
                "option_symbol": opt_sym,
                "strike": parsed.get("strike") or t.get("strike"),
                "contract_type": parsed.get("contract_type") or t.get("contract_type", ""),
                "expiry": parsed.get("expiry") or t.get("expiry", ""),
                "entry_price": ep,
                "exit_price": t.get("exit_price"),
                "sl_price": sl_price,
                "tp_price": tp_price,
                "qty": t.get("qty"),
                "pnl": t.get("realized_pnl_dollars"),
                "pnl_pct": t.get("realized_pnl_pct"),
                "entry_time": t.get("entry_time", ""),
                "exit_time": t.get("exit_time", ""),
                "exit_reason": t.get("exit_reason", ""),
                "exit_context": t.get("exit_context", ""),
                "regime": t.get("regime_at_entry", ""),
                "signal_mode": t.get("signal_mode", profile.get("signal_mode", "")),
                "mae_pct": t.get("mae_pct"),
                "mfe_pct": t.get("mfe_pct"),
            }

        for t in (data.get("open_trades") or []):
            rec = _make_record(t, "open")
            if rec:
                open_trades_out.append(rec)

        for t in (data.get("trade_log") or []):
            if t.get("exit_time") and t.get("realized_pnl_dollars") is not None:
                rec = _make_record(t, "closed")
                if rec:
                    closed_trades.append(rec)

    open_trades_out.sort(key=lambda x: x.get("entry_time") or "", reverse=True)
    closed_trades.sort(key=lambda x: x.get("exit_time") or "", reverse=True)
    all_trades = open_trades_out + closed_trades

    all_sims = sorted({t["sim_id"] for t in all_trades})
    all_syms = sorted({t["symbol"] for t in all_trades if t["symbol"]})

    return {
        "trades": all_trades[:limit],
        "total": len(all_trades),
        "open_count": len(open_trades_out),
        "sims": all_sims,
        "symbols": all_syms,
    }


async def _handle_get_trade_history(
    sim_id: str,
    date: Optional[str],
    direction: Optional[str],
    result: Optional[str],
    symbol: Optional[str],
    page: int,
    per_page: int,
) -> dict:
    """Body of GET /api/trades/{sim_id}/history — returns paginated trade log dict."""
    import json
    data = _load_sim(sim_id)
    if data is None:
        return None  # caller raises 404

    trade_log = list(reversed(data.get("trade_log") or []))  # newest first

    if date:
        trade_log = [t for t in trade_log if (t.get("entry_time") or "").startswith(date)]
    if direction and direction.upper() != "ALL":
        d = direction.upper()
        trade_log = [t for t in trade_log if (t.get("direction") or "").upper() == d]
    if symbol and symbol.upper() != "ALL":
        sym = symbol.upper()
        trade_log = [t for t in trade_log
                     if (t.get("symbol") or _parse_underlying(t.get("option_symbol", ""))).upper() == sym]
    if result and result.lower() != "all":
        if result.lower() == "win":
            trade_log = [t for t in trade_log if _safe_float(t.get("realized_pnl_dollars", 0)) > 0]
        elif result.lower() == "loss":
            trade_log = [t for t in trade_log if _safe_float(t.get("realized_pnl_dollars", 0)) <= 0]

    total = len(trade_log)
    start = (page - 1) * per_page
    page_trades = trade_log[start:start + per_page]

    config = _load_config()
    _prof = config.get(sim_id) or {}
    _sl_pct = _safe_float(_prof.get("stop_loss_pct"), 0)
    _tp_pct = _safe_float(_prof.get("profit_target_pct"), 0)

    try:
        from analytics.grader import grade_trade as _grade_trade
    except Exception:
        _grade_trade = None

    trades_out = []
    for t in page_trades:
        safe_id = re.sub(r"[^\w\-]", "_", t.get("trade_id", ""))
        grade = ""
        narr_path = os.path.join(NARRATIVES_DIR, f"{sim_id}_{safe_id}.json")
        if os.path.exists(narr_path):
            try:
                narr_data = json.loads(open(narr_path).read())
                grade = narr_data.get("grade", "") or ""
            except Exception:
                pass
        if not grade and _grade_trade and t.get("exit_price") is not None:
            grade = _grade_trade(t)
        ep = t.get("entry_price")
        sl_price = round(ep * (1 - _sl_pct), 4) if ep and _sl_pct else None
        tp_price = round(ep * (1 + _tp_pct), 4) if ep and _tp_pct else None
        trades_out.append({
            "trade_id":    t.get("trade_id", ""),
            "symbol":      (t.get("symbol") or _parse_underlying(t.get("option_symbol", ""))).upper(),
            "option_symbol": t.get("option_symbol", ""),
            "direction":   t.get("direction", ""),
            "entry_price": ep,
            "exit_price":  t.get("exit_price"),
            "sl_price":    sl_price,
            "tp_price":    tp_price,
            "entry_time":  t.get("entry_time", ""),
            "exit_time":   t.get("exit_time", ""),
            "qty":         t.get("qty"),
            "pnl":         t.get("realized_pnl_dollars"),
            "pnl_pct":     t.get("realized_pnl_pct"),
            "exit_reason": t.get("exit_reason", ""),
            "exit_context": t.get("exit_context", ""),
            "signal_mode": t.get("signal_mode", ""),
            "regime":      t.get("regime_at_entry", ""),
            "time_bucket": t.get("time_of_day_bucket", ""),
            "strike":      t.get("strike"),
            "expiry":      t.get("expiry", ""),
            "contract_type": t.get("contract_type", ""),
            "mae_pct":     t.get("mae_pct") or t.get("mae"),
            "mfe_pct":     t.get("mfe_pct") or t.get("mfe"),
            "structure_score": t.get("structure_score"),
            "grade":       grade,
            "has_narrative": os.path.exists(narr_path),
        })

    return {"sim_id": sim_id, "total": total, "page": page, "per_page": per_page, "trades": trades_out}
