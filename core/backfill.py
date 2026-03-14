"""
Historical 1-minute bar backfill from Alpaca.

Fetches 1m bars for a given symbol (or all registered symbols) for a date
range in daily chunks and merges into each symbol's CSV file.

Usage:
    from core.backfill import run_backfill, run_backfill_all_symbols, backfill_status

    # Single symbol (synchronous)
    result = run_backfill(symbol="AAPL", days_back=30)

    # All registered symbols (synchronous)
    results = run_backfill_all_symbols(days_back=30)

    # Async wrapper — use from bot commands
    result  = await run_backfill_async(symbol="AAPL", days_back=30, progress_cb=cb)
    results = await run_backfill_all_symbols_async(days_back=30, progress_cb=cb)
"""

import asyncio
try:
    import fcntl
except ImportError:
    fcntl = None
    import msvcrt
import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import pytz
import yaml
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv

from core.paths import DATA_DIR
from core.rate_limiter import rate_limit_sleep

load_dotenv()

_ET = pytz.timezone("US/Eastern")
_ALPACA_MIN_INTERVAL = float(os.getenv("ALPACA_MIN_CALL_INTERVAL_SEC", "0.5"))
_INTER_CHUNK_SLEEP = 1.0
_MIN_BARS_WARNING = 200

# ── Symbol registry helpers ────────────────────────────────────────────────

def _load_registered_symbols() -> dict:
    """Return {SYMBOL: {data_file: ..., type: ...}, ...} from sim_config.yaml."""
    base = os.path.dirname(DATA_DIR)
    cfg_path = os.path.join(base, "simulation", "sim_config.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("symbols") or {}
    except Exception as e:
        logging.warning("backfill_registry_load_failed: %s", e)
        return {}


def _get_data_file(symbol: str) -> str:
    """Return the absolute CSV path for a symbol."""
    sym = symbol.upper()
    registry = _load_registered_symbols()
    entry = registry.get(sym)
    if entry:
        rel = entry.get("data_file", "")
        if rel:
            base = os.path.dirname(DATA_DIR)
            return os.path.join(base, rel) if not os.path.isabs(rel) else rel
    # Fallback: data/{sym.lower()}_1m.csv
    return os.path.join(DATA_DIR, f"{sym.lower()}_1m.csv")


# ── Alpaca helpers ─────────────────────────────────────────────────────────

def _get_client() -> Optional[StockHistoricalDataClient]:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    return StockHistoricalDataClient(api_key, secret_key)


def _is_market_day(d: date) -> bool:
    return d.weekday() < 5


def _fetch_day(
    client: StockHistoricalDataClient,
    day: date,
    symbol: str,
) -> Optional[pd.DataFrame]:
    """Fetch 1m bars for a single trading day for any symbol."""
    market_open_et  = _ET.localize(datetime(day.year, day.month, day.day, 9, 30, 0))
    market_close_et = _ET.localize(datetime(day.year, day.month, day.day, 16, 0, 0))
    start_utc = market_open_et.astimezone(pytz.UTC)
    end_utc   = market_close_et.astimezone(pytz.UTC)

    try:
        rate_limit_sleep("alpaca_backfill", _ALPACA_MIN_INTERVAL)
        req = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=TimeFrame(1, TimeFrameUnit("Min")),
            start=start_utc,
            end=end_utc,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        df = getattr(bars, "df", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]
    except Exception as exc:
        logging.warning("backfill_fetch_day_error: symbol=%s day=%s err=%s", symbol, day, exc)
        return None


def _load_csv(data_file: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(data_file):
        return None
    try:
        with open(data_file, "r", newline="") as f:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            else:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                df = pd.read_csv(f)
            finally:
                if fcntl:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                else:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLOCK, 1)
        return df if not df.empty else None
    except Exception:
        return None


def _save_csv(df: pd.DataFrame, data_file: str) -> None:
    os.makedirs(os.path.dirname(data_file) or DATA_DIR, exist_ok=True)
    tmp = data_file + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, data_file)


def _merge(existing: Optional[pd.DataFrame], new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new_rows
    else:
        merged = pd.concat([existing, new_rows], ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.dropna(subset=["timestamp"])
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["timestamp"] = merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return merged


# ── Public API ─────────────────────────────────────────────────────────────

def backfill_status(symbol: str = None) -> dict:
    """Return basic info about a symbol's CSV."""
    data_file = _get_data_file(symbol)
    df = _load_csv(data_file)
    if df is None or df.empty:
        return {"symbol": symbol, "rows": 0, "earliest": None, "latest": None,
                "sparse": True, "file": data_file}
    rows = len(df)
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    earliest = ts.min().isoformat() if not ts.empty else None
    latest   = ts.max().isoformat() if not ts.empty else None
    return {
        "symbol":   symbol,
        "rows":     rows,
        "earliest": earliest,
        "latest":   latest,
        "sparse":   rows < _MIN_BARS_WARNING,
        "file":     data_file,
    }


def run_backfill(
    days_back: int = 30,
    symbol: str = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Synchronous backfill for a single symbol.
    Returns {ok, symbol, fetched_days, added_rows, total_rows, errors}.
    """
    sym = symbol.upper()
    data_file = _get_data_file(sym)

    client = _get_client()
    if client is None:
        return {"ok": False, "symbol": sym, "error": "alpaca_keys_missing"}

    today = date.today()
    days: list[date] = sorted(
        [today - timedelta(days=i) for i in range(1, days_back + 1) if _is_market_day(today - timedelta(days=i))]
    )

    existing = _load_csv(data_file)
    existing_ts_set: set = set()
    if existing is not None and not existing.empty and "timestamp" in existing.columns:
        existing_ts_set = set(existing["timestamp"].astype(str).tolist())

    all_new: list[pd.DataFrame] = []
    fetched_days = error_days = 0

    for day in days:
        day_prefix = day.strftime("%Y-%m-%d 09:30")
        if day_prefix in existing_ts_set:
            continue
        if progress_cb:
            progress_cb(f"[{sym}] Fetching {day.isoformat()}…")

        day_df = _fetch_day(client, day, sym)
        if day_df is None or day_df.empty:
            error_days += 1
        else:
            all_new.append(day_df)
            fetched_days += 1
            for ts in day_df["timestamp"].astype(str).tolist():
                existing_ts_set.add(ts)

        time.sleep(_INTER_CHUNK_SLEEP)

    if not all_new:
        total = len(existing) if existing is not None else 0
        return {"ok": True, "symbol": sym, "fetched_days": 0, "added_rows": 0,
                "total_rows": total, "errors": error_days}

    new_df  = pd.concat(all_new, ignore_index=True)
    merged  = _merge(existing, new_df)
    added   = len(merged) - (len(existing) if existing is not None else 0)
    _save_csv(merged, data_file)

    return {
        "ok":           True,
        "symbol":       sym,
        "fetched_days": fetched_days,
        "added_rows":   max(added, 0),
        "total_rows":   len(merged),
        "errors":       error_days,
    }


def run_backfill_all_symbols(
    days_back: int = 30,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Synchronous backfill for ALL registered symbols.
    Returns {ok, results: {SYMBOL: {...}, ...}, total_added, total_errors}.
    """
    registry = _load_registered_symbols()
    symbols  = list(registry.keys()) if registry else []

    results = {}
    total_added = total_errors = 0

    for sym in symbols:
        if progress_cb:
            progress_cb(f"Starting {sym}…")
        r = run_backfill(days_back=days_back, symbol=sym, progress_cb=progress_cb)
        results[sym] = r
        if r.get("ok"):
            total_added  += r.get("added_rows", 0)
            total_errors += r.get("errors", 0)

    return {
        "ok":           True,
        "results":      results,
        "symbols":      symbols,
        "total_added":  total_added,
        "total_errors": total_errors,
    }


async def run_backfill_async(
    days_back: int = 30,
    symbol: str = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Async wrapper for run_backfill."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_backfill(days_back=days_back, symbol=symbol, progress_cb=progress_cb),
    )


async def run_backfill_all_symbols_async(
    days_back: int = 30,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Async wrapper for run_backfill_all_symbols."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_backfill_all_symbols(days_back=days_back, progress_cb=progress_cb),
    )
