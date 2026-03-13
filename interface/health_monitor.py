# interface/health_monitor.py

import os
import time
import logging
import asyncio
import pandas as pd
from datetime import datetime
import pytz

from core.paths import DATA_DIR
from core.market_clock import market_is_open
from core.analytics_db import last_write_time, DB_PATH

DATA_FILE = os.path.join(DATA_DIR, "spy_1m.csv")
ACCOUNT_FILE = os.path.join(DATA_DIR, "account.json")


def check_health():

    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    report = []
    healthy = True

    # Market status
    if market_is_open():
        report.append("Market: OPEN")
    else:
        report.append("Market: CLOSED")

    def _fmt_age(seconds: float) -> str:
        total = int(seconds)
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def _file_age(path: str):
        if not os.path.exists(path):
            return None
        try:
            return time.time() - os.path.getmtime(path)
        except Exception:
            return None

    def _file_status(label: str, path: str, stale_after: int | None = None, critical: bool = False):
        nonlocal healthy
        if not os.path.exists(path):
            report.append(f"{label}: MISSING")
            if critical:
                healthy = False
            return
        if os.path.getsize(path) == 0:
            report.append(f"{label}: EMPTY")
            if critical:
                healthy = False
            return
        age = _file_age(path)
        if age is None:
            report.append(f"{label}: ERROR READING")
            if critical:
                healthy = False
            return
        if stale_after is not None and market_is_open() and age > stale_after:
            report.append(f"{label}: STALE ({_fmt_age(age)} old)")
            if critical:
                healthy = False
        else:
            report.append(f"{label}: OK ({_fmt_age(age)} old)")

    def _db_table_status(label: str, table: str, time_col: str, stale_after: int | None = None):
        """Check freshness of a SQLite table by its most recent timestamp."""
        try:
            last_ts = last_write_time(table, time_col)
            if last_ts is None:
                report.append(f"{label}: EMPTY")
                return
            last_dt = pd.to_datetime(last_ts, errors="coerce")
            if pd.isna(last_dt):
                report.append(f"{label}: OK (has data)")
                return
            if last_dt.tzinfo is None:
                last_dt = eastern.localize(last_dt)
            age = (now - last_dt).total_seconds()
            if stale_after is not None and market_is_open() and age > stale_after:
                report.append(f"{label}: STALE ({_fmt_age(age)} old)")
            else:
                report.append(f"{label}: OK ({_fmt_age(age)} old)")
        except Exception:
            report.append(f"{label}: OK (db)")

    # --- Recorder Check ---
    if not os.path.exists(DATA_FILE):
        report.append("Recorder: FILE MISSING")
        healthy = False

    else:
        try:
            df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

            if df.empty:
                report.append("Recorder: EMPTY FILE")
                healthy = False

            else:
                last_time = pd.to_datetime(df["timestamp"].iloc[-1])

                # Make last_time tz-aware in Eastern
                if last_time.tzinfo is None:
                    last_time = pytz.timezone("US/Eastern").localize(last_time)

                age = (now - last_time).total_seconds()

                if not market_is_open():
                    report.append("Recorder: Idle (market closed)")

                elif age > 300:
                    report.append(f"Recorder: STALLED ({int(age)}s old)")
                    healthy = False

                else:
                    report.append(f"Recorder: OK ({int(age)}s old)")

        except Exception as e:
            report.append(f"Recorder: ERROR READING FILE: {e}")
            healthy = False

    # --- Account File Check ---
    _file_status("Account File", ACCOUNT_FILE, critical=True)

    # --- Analytics DB checks (non-critical) ---
    _db_table_status("Predictions", "predictions", "time", stale_after=3600)
    _db_table_status("Conviction", "conviction_expectancy", "time", stale_after=300)
    _db_table_status("Trade Features", "trade_features", "id", stale_after=None)
    _db_table_status("Signal Log", "signal_log", "timestamp", stale_after=300)
    _db_table_status("Blocked Signals", "blocked_signals", "timestamp", stale_after=900)
    _db_table_status("Contract Log", "contract_selection_log", "timestamp", stale_after=900)
    _db_table_status("Execution Log", "execution_quality_log", "timestamp", stale_after=900)

    status = "HEALTHY" if healthy else "ATTENTION NEEDED"

    return status, "\n".join(report)

# ==============================
# HEARTBEAT SYSTEM
# ==============================

async def start_heartbeat():
    _start = time.time()
    while True:
        try:
            uptime_s = int(time.time() - _start)
            hours, rem = divmod(uptime_s, 3600)
            mins = rem // 60
            logging.error("HEARTBEAT: Bot alive | uptime=%dh%dm", hours, mins)
        except Exception as e:
            logging.exception(f"Heartbeat error: {e}")
        await asyncio.sleep(600)  # log every 10 minutes
