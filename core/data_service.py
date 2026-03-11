# core/data_service.py

import os
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv
from core.market_clock import market_is_open
from core.rate_limiter import rate_limit_sleep

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_MIN_CALL_INTERVAL_SEC = float(os.getenv("ALPACA_MIN_CALL_INTERVAL_SEC", "0.5"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "spy_1m.csv")

_client = None


def get_client():
    global _client
    if _client is None:
        if not API_KEY or not SECRET_KEY:
            logging.error("alpaca_keys_missing: data_service client not initialized")
            return None
        _client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    return _client


def get_market_dataframe():
    """Returns full SPY dataframe with indicators. Thin wrapper around get_symbol_dataframe."""
    df = get_symbol_dataframe("SPY")
    if df is None:
        return None
    if len(df) < 200:
        logging.warning(
            "data_service_sparse_csv: only %d rows — run: python scripts/backfill_candles.py",
            len(df),
        )
    try:
        open_now = market_is_open()
        df.attrs["market_open"] = open_now
        df.attrs["market_status"] = "open" if open_now else "closed"
    except Exception:
        pass
    return df

def get_recent_candles(n=60):
    df = get_market_dataframe()
    if df is None:
        return None
    return df.tail(n)


def get_latest_price():
    df = get_market_dataframe()
    if df is None:
        return None
    return df.iloc[-1]["close"]


def get_price_at(timestamp):
    df = get_market_dataframe()
    if df is None:
        return None

    target = pd.to_datetime(timestamp)
    df["diff"] = abs(df.index - target)
    row = df.loc[df["diff"].idxmin()]
    return row

def _prepare_dataframe(df):

    # -----------------------------
    # Validate timestamp
    # -----------------------------
    if "timestamp" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        return None

    # -----------------------------
    # Set index properly
    # -----------------------------
    df = df.set_index("timestamp")

    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="last")]

    # Strict sort
    df = df.sort_index()

    # Ensure ordered DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return None

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # -----------------------------
    # MINIMUM SAFETY SIZE
    # -----------------------------
    if len(df) < 5:
        # Not enough for indicators yet
        return df

    # -----------------------------
    # SAFE INDICATORS
    # -----------------------------

    # EMA safe
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI safe
    try:
        df["rsi"] = ta.rsi(df["close"], length=14)
    except:
        df["rsi"] = None

    # ATR safe
    try:
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    except:
        df["atr"] = None

    # VWAP SAFE (critical fix)
    try:
        if len(df) > 10:
            df["vwap"] = ta.vwap(
                df["high"],
                df["low"],
                df["close"],
                df["volume"]
            )
        else:
            df["vwap"] = None
    except:
        df["vwap"] = None

    # Do NOT dropna() globally anymore
    # That was silently killing small datasets

    return df


def _fetch_from_alpaca(symbol: str = "SPY"):
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    if now.weekday() == 5:
        now -= timedelta(days=1)
    elif now.weekday() == 6:
        now -= timedelta(days=2)

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=0, second=0, microsecond=0)

    start_utc = market_open.astimezone(pytz.UTC)
    end_utc = end.astimezone(pytz.UTC)

    client = get_client()
    if client is None:
        return None

    request = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame(1, TimeFrameUnit("Min")),
        start=start_utc,
        end=end_utc,
        feed=DataFeed.IEX
    )

    # Rate-limit Alpaca calls
    rate_limit_sleep("alpaca_stock_bars", ALPACA_MIN_CALL_INTERVAL_SEC)
    bars = client.get_stock_bars(request)

    # Broker responded without exception — mark health even if bars are empty.
    # (Empty bars during volatile/post-market periods still means broker is reachable.)
    try:
        from core.singletons import RISK_SUPERVISOR
        import time as _t
        RISK_SUPERVISOR.update_broker_health(_t.time())
    except ImportError:
        pass

    df = getattr(bars, "df", None)

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.reset_index()
    # Multi-symbol responses include a 'symbol' column; drop it if present
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    try:
        from core.singletons import RISK_SUPERVISOR
        import time as _t
        RISK_SUPERVISOR.update_bar_freshness(_t.time())
    except ImportError:
        pass

    return df


def get_symbol_dataframe(symbol: str):
    """
    Like get_market_dataframe() but for any registered symbol.
    Reads from the symbol's CSV, refreshes from Alpaca if stale.
    Returns None if data unavailable.
    """
    symbol = symbol.upper()
    csv_path = get_symbol_csv_path(symbol)
    if not csv_path:
        return None

    open_now = market_is_open()
    df = None

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None

    if df is None or df.empty:
        fresh = _fetch_from_alpaca(symbol)
        if fresh is not None:
            fresh.attrs["source"] = "alpaca"
            df = fresh
    else:
        try:
            temp = df.copy()
            if "timestamp" in temp.columns:
                temp["timestamp"] = pd.to_datetime(temp["timestamp"], errors="coerce")
                temp = temp.dropna(subset=["timestamp"])
                if not temp.empty:
                    last_ts = temp["timestamp"].iloc[-1]
                    if isinstance(last_ts, pd.Timestamp):
                        if last_ts.tzinfo is None:
                            last_ts = last_ts.tz_localize("US/Eastern")
                        else:
                            last_ts = last_ts.tz_convert("US/Eastern")
                    now = datetime.now(pytz.timezone("US/Eastern"))
                    age_seconds = (now - last_ts).total_seconds()
                    if age_seconds > 110 and open_now:
                        fresh = _fetch_from_alpaca(symbol)
                        if fresh is not None and not fresh.empty:
                            if len(fresh) < len(df):
                                try:
                                    df_ts = pd.to_datetime(df["timestamp"], errors="coerce")
                                    fresh_ts = pd.to_datetime(fresh["timestamp"], errors="coerce")
                                    cutoff = fresh_ts.min()
                                    historical = df[df_ts < cutoff]
                                    if not historical.empty:
                                        fresh = pd.concat([historical, fresh], ignore_index=True)
                                except Exception:
                                    pass
                            fresh.attrs["source"] = "alpaca"
                            df = fresh
                            # Write-back: persist fresh data to CSV so charts/snapshots stay current
                            try:
                                _cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in fresh.columns]
                                _to_save = fresh[_cols].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                                _tmp = csv_path + ".tmp"
                                _to_save.to_csv(_tmp, index=False)
                                os.replace(_tmp, csv_path)
                            except Exception:
                                pass
                        else:
                            df.attrs["source"] = "csv_stale"
                    else:
                        df.attrs["source"] = "csv"
        except Exception:
            if df is not None:
                df.attrs["source"] = "csv"

    if df is None or len(df) == 0:
        return None

    df = _prepare_dataframe(df)
    return df if df is not None and len(df) > 0 else None


# ---------------------------------------------------------------------------
# Multi-symbol candle data
# ---------------------------------------------------------------------------

_SYMBOL_REGISTRY = None
_REGISTRY_MTIME  = 0.0

def _load_symbol_registry() -> dict:
    """Load the symbols: section from sim_config.yaml. Cached with mtime check."""
    global _SYMBOL_REGISTRY, _REGISTRY_MTIME
    import yaml
    cfg_path = os.path.join(BASE_DIR, "simulation", "sim_config.yaml")
    try:
        mtime = os.path.getmtime(cfg_path)
        if _SYMBOL_REGISTRY is not None and mtime == _REGISTRY_MTIME:
            return _SYMBOL_REGISTRY
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        _SYMBOL_REGISTRY = cfg.get("symbols") or {}
        _REGISTRY_MTIME  = mtime
        return _SYMBOL_REGISTRY
    except Exception as e:
        logging.warning("symbol_registry_load_failed: %s", e)
        return {}


def get_symbol_csv_path(symbol: str) -> str | None:
    """Return absolute path to the per-symbol candle CSV, or None."""
    registry = _load_symbol_registry()
    entry = registry.get(symbol.upper())
    if entry:
        rel = entry.get("data_file", "")
        if rel:
            return os.path.join(BASE_DIR, rel) if not os.path.isabs(rel) else rel
    # Convention fallback: data/{symbol}_1m.csv
    fallback = os.path.join(DATA_DIR, f"{symbol.lower()}_1m.csv")
    if os.path.exists(fallback):
        return fallback
    return None


def get_candle_data(symbol: str, start: "datetime", end: "datetime") -> list[dict]:
    """
    Return 1-minute OHLCV candles for `symbol` between `start` and `end` (naive ET).

    Sources tried in order:
      1. Per-symbol CSV from the symbol registry
      2. Alpaca StockBarsRequest (historical)

    Returns list of dicts: [{t, o, h, l, c, v}, ...] or [] on failure.
    Never raises.
    """
    symbol = symbol.upper()
    result = []

    # ── 1. Try CSV ──
    csv_path = get_symbol_csv_path(symbol)
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.lower() for c in df.columns]
            ts_col = next((c for c in ("timestamp", "time", "datetime") if c in df.columns), None)
            if ts_col:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
                df = df.dropna(subset=[ts_col])
                if df[ts_col].dt.tz is not None:
                    df[ts_col] = df[ts_col].dt.tz_convert("US/Eastern").dt.tz_localize(None)
                mask = (df[ts_col] >= pd.Timestamp(start)) & (df[ts_col] <= pd.Timestamp(end))
                window = df[mask].sort_values(ts_col)
                if not window.empty:
                    for _, row in window.iterrows():
                        try:
                            result.append({
                                "t": str(row[ts_col]),
                                "o": round(float(row.get("open",  0)), 4),
                                "h": round(float(row.get("high",  0)), 4),
                                "l": round(float(row.get("low",   0)), 4),
                                "c": round(float(row.get("close", 0)), 4),
                                "v": int(float(row.get("volume", 0))),
                            })
                        except Exception:
                            pass
                    if result:
                        return result
        except Exception as e:
            logging.warning("get_candle_data_csv_failed symbol=%s: %s", symbol, e)

    # ── 2. Fallback: Alpaca historical ──
    try:
        client = get_client()
        if client is None:
            return []
        eastern = pytz.timezone("US/Eastern")
        start_utc = eastern.localize(start).astimezone(pytz.UTC) if start.tzinfo is None else start.astimezone(pytz.UTC)
        end_utc   = eastern.localize(end).astimezone(pytz.UTC)   if end.tzinfo   is None else end.astimezone(pytz.UTC)

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed

        rate_limit_sleep("alpaca_stock_bars", ALPACA_MIN_CALL_INTERVAL_SEC)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit("Min")),
            start=start_utc,
            end=end_utc,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request)
        df   = getattr(bars, "df", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
        df = df.sort_values("timestamp")
        for _, row in df.iterrows():
            try:
                result.append({
                    "t": str(row["timestamp"]),
                    "o": round(float(row.get("open",  0)), 4),
                    "h": round(float(row.get("high",  0)), 4),
                    "l": round(float(row.get("low",   0)), 4),
                    "c": round(float(row.get("close", 0)), 4),
                    "v": int(float(row.get("volume", 0))),
                })
            except Exception:
                pass
        return result
    except Exception as e:
        logging.warning("get_candle_data_alpaca_failed symbol=%s: %s", symbol, e)
        return []


def startup_backfill_all():
    """
    Auto-backfill all symbol CSVs at bot startup.
    For each symbol CSV, checks the last timestamp. If there's a gap
    (data older than today during market hours, or older than last trading day),
    fetches fresh bars from Alpaca and appends them.
    Returns dict of {symbol: bars_added}.
    """
    import glob as _glob
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)
    today = now.date()
    results = {}

    # Discover all symbol CSVs
    symbol_paths = {}
    for csv_path in sorted(_glob.glob(os.path.join(DATA_DIR, "*_1m.csv"))):
        sym = os.path.basename(csv_path).replace("_1m.csv", "").upper()
        symbol_paths[sym] = csv_path

    if not symbol_paths:
        return results

    client = get_client()
    if client is None:
        print("startup_backfill: No Alpaca client, skipping")
        return results

    for sym, csv_path in symbol_paths.items():
        try:
            if not os.path.exists(csv_path):
                continue

            # Read last timestamp from CSV (fast: read last line)
            last_ts_str = None
            try:
                with open(csv_path, "rb") as f:
                    try:
                        f.seek(-2, os.SEEK_END)
                        while f.tell() > 0:
                            if f.read(1) == b"\n":
                                break
                            f.seek(-2, os.SEEK_CUR)
                    except OSError:
                        f.seek(0)
                    last_line = f.readline().decode("utf-8", errors="ignore").strip()
                if last_line and not last_line.lower().startswith("timestamp"):
                    last_ts_str = last_line.split(",")[0].strip()
            except Exception:
                continue

            if not last_ts_str:
                continue

            last_ts = pd.Timestamp(last_ts_str)
            if last_ts.tzinfo is None:
                last_ts = eastern.localize(last_ts)
            else:
                last_ts = last_ts.astimezone(eastern)

            last_date = last_ts.date()

            # Skip if CSV already has today's data
            if last_date >= today:
                continue

            # Fetch from last_date+1 through now
            fetch_start = eastern.localize(
                datetime.combine(last_date + timedelta(days=1), datetime.min.time()).replace(hour=4, minute=0)
            )
            fetch_end = now

            if fetch_start >= fetch_end:
                continue

            print(f"  Backfilling {sym}: {last_date} -> {today}...")
            rate_limit_sleep("alpaca_stock_bars", ALPACA_MIN_CALL_INTERVAL_SEC)
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(1, TimeFrameUnit("Min")),
                start=fetch_start.astimezone(pytz.UTC),
                end=fetch_end.astimezone(pytz.UTC),
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
            fresh_df = getattr(bars, "df", None)
            if fresh_df is None or fresh_df.empty:
                continue

            fresh_df = fresh_df.reset_index()
            if "symbol" in fresh_df.columns:
                fresh_df = fresh_df.drop(columns=["symbol"])
            fresh_df["timestamp"] = pd.to_datetime(fresh_df["timestamp"], utc=True)
            fresh_df["timestamp"] = fresh_df["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)

            csv_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            fresh_df = fresh_df[[c for c in csv_cols if c in fresh_df.columns]]

            # Read existing, append, dedupe, write
            existing = pd.read_csv(csv_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], errors="coerce")
            combined = pd.concat([existing[csv_cols], fresh_df], ignore_index=True)
            combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

            tmp = csv_path + ".tmp"
            combined.to_csv(tmp, index=False)
            os.replace(tmp, csv_path)
            results[sym] = len(fresh_df)
            print(f"  {sym}: +{len(fresh_df)} bars (now {len(combined)} total, last={combined['timestamp'].iloc[-1]})")

        except Exception as e:
            print(f"  startup_backfill_failed: {sym} — {e}")

    return results


def backfill_symbol_csvs(min_bars: int = 100):
    """
    Check all symbols in the registry. For each, look at the last trading day
    in the CSV. If it has fewer than `min_bars` bars, fetch that day from Alpaca
    and replace the sparse data. Also backfills SPY.

    Returns dict of {symbol: bars_written} for symbols that were backfilled.
    """
    import glob as _glob
    eastern = pytz.timezone("US/Eastern")
    results = {}

    # Build list from registry + convention-based discovery
    symbol_paths = {}
    registry = _load_symbol_registry()
    for sym, entry in registry.items():
        rel = entry.get("data_file", "")
        if rel:
            symbol_paths[sym.upper()] = os.path.join(BASE_DIR, rel) if not os.path.isabs(rel) else rel
    for csv_path in _glob.glob(os.path.join(DATA_DIR, "*_1m.csv")):
        sym = os.path.basename(csv_path).replace("_1m.csv", "").upper()
        if sym not in symbol_paths:
            symbol_paths[sym] = csv_path

    for sym, csv_path in symbol_paths.items():
        try:
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            if df.empty:
                continue

            # Find the last date and count its bars
            last_date = df["timestamp"].dt.date.iloc[-1]
            day_bars = df[df["timestamp"].dt.date == last_date]
            if len(day_bars) >= min_bars:
                continue  # enough data, skip

            # Fetch from Alpaca for that date
            start_et = eastern.localize(datetime.combine(last_date, datetime.min.time()).replace(hour=9, minute=30))
            end_et = eastern.localize(datetime.combine(last_date, datetime.min.time()).replace(hour=16, minute=0))

            client = get_client()
            if client is None:
                continue

            rate_limit_sleep("alpaca_stock_bars", ALPACA_MIN_CALL_INTERVAL_SEC)
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(1, TimeFrameUnit("Min")),
                start=start_et.astimezone(pytz.UTC),
                end=end_et.astimezone(pytz.UTC),
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
            fresh_df = getattr(bars, "df", None)
            if fresh_df is None or fresh_df.empty:
                continue

            fresh_df = fresh_df.reset_index()
            if "symbol" in fresh_df.columns:
                fresh_df = fresh_df.drop(columns=["symbol"])
            fresh_df["timestamp"] = pd.to_datetime(fresh_df["timestamp"], utc=True)
            fresh_df["timestamp"] = fresh_df["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)

            # Keep only columns matching the CSV
            csv_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            fresh_df = fresh_df[[c for c in csv_cols if c in fresh_df.columns]]

            # Remove sparse day, append fresh
            before = df[df["timestamp"].dt.date < last_date]
            result = pd.concat([before, fresh_df], ignore_index=True).sort_values("timestamp")
            result.to_csv(csv_path, index=False)
            results[sym] = {"bars": len(fresh_df), "date": last_date}
            logging.error("backfill_complete: %s got %d bars for %s (was %d)", sym, len(fresh_df), last_date, len(day_bars))

        except Exception as e:
            logging.error("backfill_failed: %s — %s", sym, e)

    return results
