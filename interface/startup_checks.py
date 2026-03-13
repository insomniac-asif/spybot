# Startup phase-gate validation (extracted from bot.py)

import os
import csv
import json
import logging
import pytz
import joblib
from datetime import datetime, timedelta

from core.paths import DATA_DIR
from analytics.prediction_stats import PRED_HEADERS
from interface.shared_state import BOT_TIMEZONE

CONVICTION_HEADERS = [
    "time", "direction", "impulse", "follow", "price",
    "fwd_5m", "fwd_10m", "fwd_5m_price", "fwd_5m_time", "fwd_5m_status",
    "fwd_10m_price", "fwd_10m_time", "fwd_10m_status",
]
LEGACY_CONVICTION_HEADERS = [
    "time", "direction", "impulse", "follow", "price", "fwd_5m", "fwd_10m"
]
PREDICTION_REQUIRED_HEADERS = [
    "time", "timeframe", "direction", "confidence", "high", "low",
    "regime", "volatility", "session", "actual", "correct", "checked"
]
ACCOUNT_REQUIRED_KEYS = [
    "balance", "starting_balance", "open_trade", "trade_log", "wins",
    "losses", "day_trades", "risk_per_trade", "max_trade_size",
    "daily_loss", "max_daily_loss", "last_trade_day"
]

SYMBOL_CSV_REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}
SYMBOL_CSV_MIN_ROWS = 200
# Warn if newest bar is older than this many calendar days (covers weekends + market holidays)
SYMBOL_CSV_STALE_DAYS = 5
# Symbols where a missing/stale CSV is a hard error (blocks trading)
SYMBOL_CSV_CRITICAL = {"SPY", "QQQ"}


def check_symbol_csvs() -> tuple[list[str], list[str]]:
    """
    Validate all registered symbol CSV files.

    Returns
    -------
    errors   : list[str]  — hard errors (critical symbols missing/corrupt)
    warnings : list[str]  — soft warnings (stale data, low row count, non-critical)
    """
    errors: list[str] = []
    warnings: list[str] = []

    import glob as _glob
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    now = datetime.now(pytz.timezone("US/Eastern"))
    stale_cutoff = now - timedelta(days=SYMBOL_CSV_STALE_DAYS)

    # Build symbol -> csv_path map from registry + convention-based discovery
    symbol_csv_map: dict[str, str] = {}
    try:
        import yaml
        cfg_path = os.path.join(_base, "simulation", "sim_config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        registry: dict = cfg.get("symbols") or {}
        for symbol, entry in registry.items():
            if isinstance(entry, dict):
                rel_path = entry.get("data_file", "")
                if rel_path:
                    symbol_csv_map[symbol.upper()] = os.path.join(_base, rel_path) if not os.path.isabs(rel_path) else rel_path
    except Exception as e:
        warnings.append(f"symbol_registry_load_failed:{e}")

    # Convention fallback: discover data/*_1m.csv
    data_dir = os.path.join(_base, "data")
    for csv_file in sorted(_glob.glob(os.path.join(data_dir, "*_1m.csv"))):
        sym = os.path.basename(csv_file).replace("_1m.csv", "").upper()
        if sym not in symbol_csv_map:
            symbol_csv_map[sym] = csv_file

    if not symbol_csv_map:
        warnings.append("symbol_csvs_none_found")
        return errors, warnings

    for symbol, csv_path in symbol_csv_map.items():

        if not os.path.exists(csv_path):
            msg = f"symbol_csv_missing:{symbol}:{rel_path}"
            (errors if symbol in SYMBOL_CSV_CRITICAL else warnings).append(msg)
            logging.error("startup_symbol_csv_missing: %s -> %s", symbol, csv_path)
            continue

        # Read headers and first data row
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                # Count rows cheaply (iterate without loading full CSV)
                row_count = sum(1 for _ in reader)
        except Exception as e:
            msg = f"symbol_csv_read_error:{symbol}:{e}"
            (errors if symbol in SYMBOL_CSV_CRITICAL else warnings).append(msg)
            continue

        if headers is None:
            msg = f"symbol_csv_empty:{symbol}"
            (errors if symbol in SYMBOL_CSV_CRITICAL else warnings).append(msg)
            continue

        missing_cols = SYMBOL_CSV_REQUIRED_COLS - set(headers)
        if missing_cols:
            msg = f"symbol_csv_bad_columns:{symbol}:missing={','.join(sorted(missing_cols))}"
            (errors if symbol in SYMBOL_CSV_CRITICAL else warnings).append(msg)
            logging.error("startup_symbol_csv_bad_columns: %s missing=%s", symbol, missing_cols)
            continue

        if row_count < SYMBOL_CSV_MIN_ROWS:
            msg = f"symbol_csv_low_rows:{symbol}:{row_count}<{SYMBOL_CSV_MIN_ROWS}"
            warnings.append(msg)
            logging.warning("startup_symbol_csv_low_rows: %s has %d rows (need %d)", symbol, row_count, SYMBOL_CSV_MIN_ROWS)

        # Check staleness via last line timestamp
        try:
            with open(csv_path, "r", newline="") as f:
                last_line = None
                for last_line in csv.reader(f):
                    pass
            if last_line and len(last_line) >= 1:
                ts_str = last_line[0]
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = pytz.timezone("US/Eastern").localize(ts)
                if ts < stale_cutoff:
                    msg = f"symbol_csv_stale:{symbol}:last={ts_str}"
                    warnings.append(msg)
                    logging.warning("startup_symbol_csv_stale: %s last bar %s", symbol, ts_str)
        except Exception:
            pass  # stale check is best-effort

    return errors, warnings


def _read_csv_headers(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def run_startup_phase_gates():
    errors = []

    try:
        tz = pytz.timezone(BOT_TIMEZONE)
        if tz.zone != BOT_TIMEZONE:
            errors.append(f"timezone_invalid:{tz.zone}")
    except Exception as e:
        errors.append(f"timezone_error:{e}")

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        errors.append("alpaca_api_keys_missing")
    else:
        # Verify Alpaca connectivity with a real API call
        try:
            from alpaca.trading.client import TradingClient
            _client = TradingClient(api_key, secret_key, paper=True)
            _acct = _client.get_account()
            logging.error(
                "Alpaca connectivity OK: status=%s equity=$%s",
                _acct.status, _acct.equity,
            )
        except Exception as e:
            errors.append(f"alpaca_connectivity_failed:{e}")
            logging.error("ALPACA CONNECTIVITY FAILED: %s", e)

    # Initialize SQLite analytics DB (create tables, migrate CSV data)
    try:
        from core.analytics_db import init_db
        init_db()
    except Exception as e:
        errors.append(f"analytics_db_init_error:{e}")

    direction_model_path = os.path.join(DATA_DIR, "direction_model.pkl")
    edge_model_path = os.path.join(DATA_DIR, "edge_model.pkl")
    for model_path, model_name in [
        (direction_model_path, "direction_model"),
        (edge_model_path, "edge_model"),
    ]:
        if not os.path.exists(model_path):
            continue
        try:
            joblib.load(model_path)
        except Exception as e:
            errors.append(f"{model_name}_load_error:{e}")

    account_file = os.path.join(DATA_DIR, "account.json")
    if not os.path.exists(account_file):
        errors.append("account_missing")
    else:
        try:
            with open(account_file, "r") as f:
                acc = json.load(f)
            missing_keys = [k for k in ACCOUNT_REQUIRED_KEYS if k not in acc]
            if missing_keys:
                errors.append(f"account_missing_keys:{','.join(missing_keys)}")
        except Exception as e:
            errors.append(f"account_read_error:{e}")

    csv_errors, csv_warnings = check_symbol_csvs()
    errors.extend(csv_errors)
    for w in csv_warnings:
        logging.warning("startup_symbol_csv_warning: %s", w)

    return errors
