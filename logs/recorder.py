# logs/recorder.py
import sys
import os
import time
import csv
import fcntl
import yaml
from datetime import datetime, timedelta
from collections import deque

import pytz
import pandas as pd

from core.paths import DATA_DIR
from core.market_clock import market_is_open
from core.data_service import get_client
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Legacy single-file path (kept for backward compat)
FILE = os.path.join(DATA_DIR, "spy_1m.csv")

_RECENT_TS_MAX = 300

# Per-symbol state (keyed by symbol string)
_last_saved_ts: dict[str, str | None] = {}
_recent_ts: dict[str, set] = {}
_recent_ts_queue: dict[str, deque] = {}


# ── Registry helpers ───────────────────────────────────────────────────────

def _load_symbol_registry() -> dict:
    """Load symbols section from sim_config.yaml."""
    base = os.path.dirname(DATA_DIR)
    cfg_path = os.path.join(base, "simulation", "sim_config.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("symbols") or {}
    except Exception:
        return {}


def _get_csv_file(symbol: str) -> str:
    """Return the CSV file path for a symbol."""
    sym = symbol.upper()
    registry = _load_symbol_registry()
    entry = registry.get(sym)
    if entry:
        rel = entry.get("data_file", "")
        if rel:
            base = os.path.dirname(DATA_DIR)
            return os.path.join(base, rel) if not os.path.isabs(rel) else rel
    return os.path.join(DATA_DIR, f"{sym.lower()}_1m.csv")


# ── File helpers ───────────────────────────────────────────────────────────

def _get_last_saved_timestamp(file_path: str) -> str | None:
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.tell() > 0:
                    if f.read(1) == b"\n":
                        break
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode("utf-8", errors="ignore").strip()
        if not last_line or last_line.lower().startswith("timestamp"):
            return None
        return last_line.split(",")[0].strip()
    except Exception:
        return None


def append_candle_to(file_path: str, row: dict):
    """Append a candle row to any CSV file, writing header if new."""
    os.makedirs(os.path.dirname(file_path) or DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            last_ts = _get_last_saved_timestamp(file_path)
            if last_ts and row.get("timestamp") == last_ts:
                return
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "open", "high", "low", "close", "volume"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _dedupe_file(file_path: str):
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path, "r", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                df = pd.read_csv(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        if df.empty or "timestamp" not in df.columns:
            return
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        with open(file_path, "w", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                df.to_csv(f, index=False)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"Recorder dedupe error for {file_path}: {e}")


# ── Alpaca fetch ───────────────────────────────────────────────────────────

def get_latest_candle_for(symbol: str):
    """Fetch the most recent 1-min candle for any symbol from Alpaca."""
    try:
        client = get_client()
        if client is None:
            return None
        eastern = pytz.timezone("US/Eastern")
        end   = eastern.localize(datetime.now())
        start = end - timedelta(minutes=5)
        request = StockBarsRequest(
            symbol_or_symbols=symbol.upper(),
            timeframe=TimeFrame(1, TimeFrameUnit("Min")),
            start=start.astimezone(pytz.UTC),
            end=end.astimezone(pytz.UTC),
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request)
        bars_df = getattr(bars, "df", None)
        if not isinstance(bars_df, pd.DataFrame) or bars_df.empty:
            return None
        df = bars_df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        return df.iloc[-1]
    except Exception as e:
        print(f"Error fetching candle for {symbol}: {e}")
        return None


# ── Save logic ─────────────────────────────────────────────────────────────

def save_candle_for(symbol: str):
    """Fetch and save the latest 1-min candle for a symbol to its CSV."""
    sym = symbol.upper()
    candle = get_latest_candle_for(sym)
    if candle is None:
        return

    ts_str    = candle["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    file_path = _get_csv_file(sym)

    # Init per-symbol state on first call
    if sym not in _last_saved_ts:
        _last_saved_ts[sym] = _get_last_saved_timestamp(file_path)
    if sym not in _recent_ts:
        _recent_ts[sym]       = set()
        _recent_ts_queue[sym] = deque()

    if ts_str == _last_saved_ts.get(sym):
        return
    if ts_str in _recent_ts[sym]:
        return

    _last_saved_ts[sym] = ts_str
    _recent_ts[sym].add(ts_str)
    _recent_ts_queue[sym].append(ts_str)
    while len(_recent_ts_queue[sym]) > _RECENT_TS_MAX:
        old = _recent_ts_queue[sym].popleft()
        _recent_ts[sym].discard(old)

    row = {
        "timestamp": ts_str,
        "open":      candle["open"],
        "high":      candle["high"],
        "low":       candle["low"],
        "close":     candle["close"],
        "volume":    candle["volume"],
    }
    append_candle_to(file_path, row)
    print(f"Saved {sym}: {ts_str}")


# ── Recorder loop ──────────────────────────────────────────────────────────

def run_recorder():
    """
    Continuous recorder loop — records ALL symbols from the registry
    or discovered via data/*_1m.csv convention.
    """
    import glob as _glob
    registry = _load_symbol_registry()
    symbols  = list(registry.keys()) if registry else []
    # Discover symbols from convention-based CSVs
    for csv_path in _glob.glob(os.path.join(DATA_DIR, "*_1m.csv")):
        sym = os.path.basename(csv_path).replace("_1m.csv", "").upper()
        if sym not in symbols:
            symbols.append(sym)
    if not symbols:
        symbols = ["SPY"]
    print(f"Recorder started for: {', '.join(sorted(symbols))}")

    # Dedupe all files at startup
    for sym in symbols:
        _dedupe_file(_get_csv_file(sym))

    while True:
        if market_is_open():
            for sym in symbols:
                try:
                    save_candle_for(sym)
                except Exception as e:
                    print(f"Recorder error for {sym}: {e}")
        time.sleep(60)


def start_recorder_background():
    import threading

    def _run():
        while True:
            try:
                run_recorder()
            except Exception as e:
                print(f"Recorder crashed, restarting in 5s: {e}")
                time.sleep(5)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


# ── Legacy compatibility wrappers (single-symbol, SPY) ────────────────────

def get_latest_candle():
    """Legacy: fetch SPY candle."""
    return get_latest_candle_for("SPY")


def append_candle(row: dict):
    """Legacy: append to SPY CSV."""
    append_candle_to(FILE, row)


def save_candle():
    """Legacy: save SPY candle."""
    save_candle_for("SPY")
