# core/analytics_db.py
"""
SQLite-backed analytics store.  Replaces the scattered CSV files
(predictions.csv, blocked_signals.csv, conviction_expectancy.csv,
signal_log.csv, contract_selection_log.csv, execution_quality_log.csv,
trade_features.csv) with a single WAL-mode database at data/analytics.db.

Thread-safe: each call opens its own connection (cheap for SQLite).
"""

import os
import sqlite3
import logging
from contextlib import contextmanager

import pandas as pd
from pandas.errors import EmptyDataError

from core.paths import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "analytics.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = {
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            symbol TEXT,
            timeframe TEXT,
            direction TEXT,
            confidence REAL,
            high REAL,
            low REAL,
            regime TEXT,
            volatility TEXT,
            session TEXT,
            actual TEXT,
            correct INTEGER DEFAULT 0,
            checked INTEGER DEFAULT 0,
            high_hit INTEGER DEFAULT 0,
            low_hit INTEGER DEFAULT 0,
            price_at_check REAL,
            close_at_check REAL,
            confidence_band TEXT
        )
    """,
    "blocked_signals": """
        CREATE TABLE IF NOT EXISTS blocked_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            spy_price REAL,
            regime TEXT,
            volatility TEXT,
            direction TEXT,
            confidence REAL,
            blended_score REAL,
            threshold REAL,
            threshold_delta REAL,
            block_reason TEXT,
            fwd_5m REAL,
            fwd_15m REAL,
            fwd_5m_price REAL,
            fwd_15m_price REAL,
            fwd_5m_status TEXT DEFAULT 'pending',
            fwd_15m_status TEXT DEFAULT 'pending'
        )
    """,
    "conviction_expectancy": """
        CREATE TABLE IF NOT EXISTS conviction_expectancy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            direction TEXT,
            impulse REAL,
            follow REAL,
            price REAL,
            fwd_5m REAL,
            fwd_10m REAL,
            fwd_5m_price REAL,
            fwd_5m_time TEXT,
            fwd_5m_status TEXT,
            fwd_10m_price REAL,
            fwd_10m_time TEXT,
            fwd_10m_status TEXT
        )
    """,
    "signal_log": """
        CREATE TABLE IF NOT EXISTS signal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            outcome TEXT,
            block_reason TEXT,
            regime TEXT,
            volatility TEXT,
            direction_60m TEXT,
            confidence_60m REAL,
            direction_15m TEXT,
            confidence_15m REAL,
            dual_alignment TEXT,
            conviction_score REAL,
            impulse REAL,
            follow_through REAL,
            blended_score REAL,
            threshold REAL,
            threshold_delta REAL,
            ml_weight REAL,
            regime_samples INTEGER,
            expectancy_samples INTEGER,
            regime_transition TEXT,
            regime_transition_severity REAL,
            spy_price REAL
        )
    """,
    "contract_selection_log": """
        CREATE TABLE IF NOT EXISTS contract_selection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            direction TEXT,
            underlying_price REAL,
            expiry TEXT,
            dte TEXT,
            strike REAL,
            result TEXT,
            reason TEXT,
            bid REAL,
            ask REAL,
            mid REAL,
            spread_pct REAL,
            iv REAL,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL
        )
    """,
    "execution_quality_log": """
        CREATE TABLE IF NOT EXISTS execution_quality_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            option_symbol TEXT,
            side TEXT,
            order_type TEXT,
            qty_requested INTEGER,
            qty_filled INTEGER,
            fill_ratio REAL,
            expected_mid REAL,
            fill_price REAL,
            slippage_pct REAL,
            bid_at_order REAL,
            ask_at_order REAL,
            spread_at_order_pct REAL
        )
    """,
    "trade_features": """
        CREATE TABLE IF NOT EXISTS trade_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regime_encoded INTEGER,
            volatility_encoded INTEGER,
            conviction_score REAL,
            impulse REAL,
            follow_through REAL,
            confidence REAL,
            style_encoded INTEGER,
            setup_encoded INTEGER,
            session_encoded INTEGER,
            setup_raw_avg_R REAL,
            regime_raw_avg_R REAL,
            ml_probability REAL,
            predicted_won INTEGER,
            won INTEGER
        )
    """,
}

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pred_checked ON predictions(checked)",
    "CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(time)",
    "CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_blocked_fwd ON blocked_signals(fwd_5m_status, fwd_15m_status)",
    "CREATE INDEX IF NOT EXISTS idx_conv_fwd ON conviction_expectancy(fwd_5m_status, fwd_10m_status)",
    "CREATE INDEX IF NOT EXISTS idx_signal_ts ON signal_log(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_contract_ts ON contract_selection_log(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_exec_ts ON execution_quality_log(timestamp)",
]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_conn():
    """Return a new SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextmanager
def transaction():
    """Context manager that commits on success, rolls back on error."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def read_df(sql, params=None):
    """Execute a SELECT and return a pandas DataFrame."""
    conn = get_conn()
    try:
        return pd.read_sql_query(sql, conn, params=params or [])
    finally:
        conn.close()


def scalar(sql, params=None):
    """Execute a query and return a single scalar value."""
    conn = get_conn()
    try:
        row = conn.execute(sql, params or ()).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def insert(table, data: dict):
    """Insert a single row."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join("?" * len(data))
    with transaction() as conn:
        conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            list(data.values()),
        )


def insert_many(table, rows: list[dict]):
    """Insert multiple rows in a single transaction."""
    if not rows:
        return
    cols = list(rows[0].keys())
    col_str = ", ".join(cols)
    placeholders = ", ".join("?" * len(cols))
    with transaction() as conn:
        conn.executemany(
            f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})",
            [[r.get(c) for c in cols] for r in rows],
        )


def update(table, updates: dict, where: str, params: tuple = ()):
    """Update rows matching WHERE clause."""
    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    with transaction() as conn:
        conn.execute(
            f"UPDATE {table} SET {set_clause} WHERE {where}",
            list(updates.values()) + list(params),
        )


def delete_all(table):
    """Delete all rows from a table."""
    with transaction() as conn:
        conn.execute(f"DELETE FROM {table}")


def row_count(table, where=None, params=None):
    """Count rows, optionally filtered."""
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    return scalar(sql, params)


def last_write_time(table, time_col="time"):
    """Return the most recent timestamp in a table (as string)."""
    return scalar(f"SELECT MAX({time_col}) FROM {table}")


# ---------------------------------------------------------------------------
# Schema + migration
# ---------------------------------------------------------------------------

def ensure_schema():
    """Create all tables and indexes if they don't exist."""
    with transaction() as conn:
        for ddl in _SCHEMA.values():
            conn.execute(ddl)
        for idx in _INDEXES:
            conn.execute(idx)


def _safe_float(val):
    """Convert a value to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(val):
    """Convert a value to int, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        # Handle "True"/"False" strings
        if isinstance(val, str):
            if val.lower() == "true":
                return 1
            if val.lower() == "false":
                return 0
        f = float(val)
        if pd.isna(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


def _safe_str(val):
    """Convert a value to string, returning None for NaN/empty."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _migrate_csv_to_table(conn, table, csv_path, col_types):
    """
    Import an existing CSV file into a table if the table is empty.
    col_types: dict mapping column name to converter (_safe_float, _safe_int, _safe_str).
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0

    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if count > 0:
        return 0  # already migrated

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except (EmptyDataError, Exception):
        return 0

    if df.empty:
        return 0

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Build rows using the type converters
    cols = list(col_types.keys())
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in cols:
            raw = row.get(col)
            converter = col_types[col]
            values.append(converter(raw))
        rows.append(values)

    if not rows:
        return 0

    col_str = ", ".join(cols)
    placeholders = ", ".join("?" * len(cols))
    conn.executemany(
        f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


# Column type maps for migration
_PRED_TYPES = {
    "time": _safe_str, "symbol": _safe_str, "timeframe": _safe_str,
    "direction": _safe_str, "confidence": _safe_float, "high": _safe_float,
    "low": _safe_float, "regime": _safe_str, "volatility": _safe_str,
    "session": _safe_str, "actual": _safe_str, "correct": _safe_int,
    "checked": _safe_int, "high_hit": _safe_int, "low_hit": _safe_int,
    "price_at_check": _safe_float, "close_at_check": _safe_float,
    "confidence_band": _safe_str,
}

_BLOCKED_TYPES = {
    "timestamp": _safe_str, "spy_price": _safe_float, "regime": _safe_str,
    "volatility": _safe_str, "direction": _safe_str, "confidence": _safe_float,
    "blended_score": _safe_float, "threshold": _safe_float,
    "threshold_delta": _safe_float, "block_reason": _safe_str,
    "fwd_5m": _safe_float, "fwd_15m": _safe_float,
    "fwd_5m_price": _safe_float, "fwd_15m_price": _safe_float,
    "fwd_5m_status": _safe_str, "fwd_15m_status": _safe_str,
}

_CONVICTION_TYPES = {
    "time": _safe_str, "direction": _safe_str, "impulse": _safe_float,
    "follow": _safe_float, "price": _safe_float,
    "fwd_5m": _safe_float, "fwd_10m": _safe_float,
    "fwd_5m_price": _safe_float, "fwd_5m_time": _safe_str,
    "fwd_5m_status": _safe_str, "fwd_10m_price": _safe_float,
    "fwd_10m_time": _safe_str, "fwd_10m_status": _safe_str,
}

_SIGNAL_TYPES = {
    "timestamp": _safe_str, "outcome": _safe_str, "block_reason": _safe_str,
    "regime": _safe_str, "volatility": _safe_str,
    "direction_60m": _safe_str, "confidence_60m": _safe_float,
    "direction_15m": _safe_str, "confidence_15m": _safe_float,
    "dual_alignment": _safe_str, "conviction_score": _safe_float,
    "impulse": _safe_float, "follow_through": _safe_float,
    "blended_score": _safe_float, "threshold": _safe_float,
    "threshold_delta": _safe_float, "ml_weight": _safe_float,
    "regime_samples": _safe_int, "expectancy_samples": _safe_int,
    "regime_transition": _safe_str, "regime_transition_severity": _safe_float,
    "spy_price": _safe_float,
}

_CONTRACT_TYPES = {
    "timestamp": _safe_str, "source": _safe_str, "direction": _safe_str,
    "underlying_price": _safe_float, "expiry": _safe_str, "dte": _safe_str,
    "strike": _safe_float, "result": _safe_str, "reason": _safe_str,
    "bid": _safe_float, "ask": _safe_float, "mid": _safe_float,
    "spread_pct": _safe_float, "iv": _safe_float, "delta": _safe_float,
    "gamma": _safe_float, "theta": _safe_float, "vega": _safe_float,
}

_EXEC_TYPES = {
    "timestamp": _safe_str, "option_symbol": _safe_str, "side": _safe_str,
    "order_type": _safe_str, "qty_requested": _safe_int,
    "qty_filled": _safe_int, "fill_ratio": _safe_float,
    "expected_mid": _safe_float, "fill_price": _safe_float,
    "slippage_pct": _safe_float, "bid_at_order": _safe_float,
    "ask_at_order": _safe_float, "spread_at_order_pct": _safe_float,
}

_FEATURE_TYPES = {
    "regime_encoded": _safe_int, "volatility_encoded": _safe_int,
    "conviction_score": _safe_float, "impulse": _safe_float,
    "follow_through": _safe_float, "confidence": _safe_float,
    "style_encoded": _safe_int, "setup_encoded": _safe_int,
    "session_encoded": _safe_int, "setup_raw_avg_R": _safe_float,
    "regime_raw_avg_R": _safe_float, "ml_probability": _safe_float,
    "predicted_won": _safe_int, "won": _safe_int,
}


def migrate_from_csv():
    """One-time import of existing CSV data into SQLite tables."""
    csv_map = {
        "predictions": (os.path.join(DATA_DIR, "predictions.csv"), _PRED_TYPES),
        "blocked_signals": (os.path.join(DATA_DIR, "blocked_signals.csv"), _BLOCKED_TYPES),
        "conviction_expectancy": (os.path.join(DATA_DIR, "conviction_expectancy.csv"), _CONVICTION_TYPES),
        "signal_log": (os.path.join(DATA_DIR, "signal_log.csv"), _SIGNAL_TYPES),
        "contract_selection_log": (os.path.join(DATA_DIR, "contract_selection_log.csv"), _CONTRACT_TYPES),
        "execution_quality_log": (os.path.join(DATA_DIR, "execution_quality_log.csv"), _EXEC_TYPES),
        "trade_features": (os.path.join(DATA_DIR, "trade_features.csv"), _FEATURE_TYPES),
    }

    with transaction() as conn:
        for table, (csv_path, col_types) in csv_map.items():
            try:
                n = _migrate_csv_to_table(conn, table, csv_path, col_types)
                if n > 0:
                    logging.error("csv_migrated: %s -> %s (%d rows)", csv_path, table, n)
            except Exception as e:
                logging.error("csv_migration_failed: %s -> %s: %s", csv_path, table, e)


def init_db():
    """Ensure schema exists and migrate CSV data on first run."""
    ensure_schema()
    migrate_from_csv()


def backup_analytics_db(db_path: str = DB_PATH, backup_dir: str | None = None) -> str | None:
    """Create a daily backup of analytics.db using SQLite's backup API.

    Keeps last 7 backups.  Returns the backup path on success, None on failure.
    """
    from datetime import datetime as _dt
    from pathlib import Path

    if backup_dir is None:
        backup_dir = os.path.join(DATA_DIR, "backups")
    bp = Path(backup_dir)
    bp.mkdir(parents=True, exist_ok=True)

    date_str = _dt.now().strftime("%Y%m%d")
    dest = bp / f"analytics_{date_str}.db"

    if dest.exists():
        return str(dest)

    try:
        source = sqlite3.connect(db_path)
        target = sqlite3.connect(str(dest))
        source.backup(target)
        target.close()
        source.close()
        logging.error("Analytics DB backed up to %s", dest)

        # Rotate: keep last 7 backups
        backups = sorted(bp.glob("analytics_*.db"))
        while len(backups) > 7:
            oldest = backups.pop(0)
            oldest.unlink()
            logging.error("Rotated old backup: %s", oldest)

        return str(dest)
    except Exception as e:
        logging.error("Analytics DB backup FAILED: %s", e)
        return None
