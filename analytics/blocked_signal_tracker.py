# analytics/blocked_signal_tracker.py
#
# Logs every blocked signal with SPY price at decision time, then
# retrospectively fills in forward SPY returns (+5m, +15m) so you can
# measure the "cost" of each gate — i.e. would the trade have worked?
#
# Call log_blocked_signal() immediately after a blocked decision.
# Call update_blocked_outcomes(df) periodically (e.g. from conviction_watcher)
# to fill in the forward-return columns once market data is available.

import os
from datetime import datetime, timedelta

import pandas as pd
import pytz

from core.paths import DATA_DIR
from core.analytics_db import insert, read_df, transaction

FILE = os.path.join(DATA_DIR, "blocked_signals.csv")
HEADERS = [
    "timestamp",
    "spy_price",
    "regime",
    "volatility",
    "direction",          # direction_60m (primary bias)
    "confidence",         # confidence_60m
    "blended_score",
    "threshold",
    "threshold_delta",
    "block_reason",
    # forward return columns — filled by update_blocked_outcomes()
    "fwd_5m",
    "fwd_15m",
    "fwd_5m_price",
    "fwd_15m_price",
    "fwd_5m_status",      # "filled" | "estimated" | "pending"
    "fwd_15m_status",
]


def _ensure_file() -> None:
    """No-op: schema is ensured at startup via analytics_db.init_db()."""
    pass


def _safe_round(val, decimals=4):
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None


def log_blocked_signal(ctx, spy_price) -> None:
    """
    Write one row per blocked auto_trader cycle.

    Parameters
    ----------
    ctx       : DecisionContext after open_trade_if_valid() returned blocked
    spy_price : float — current SPY price at decision time (pass df.iloc[-1]["close"])
    """
    try:
        blended = getattr(ctx, "blended_score", None)
        threshold = getattr(ctx, "threshold", None)
        delta = (
            _safe_round(blended - threshold, 6)
            if blended is not None and threshold is not None
            else None
        )

        insert("blocked_signals", {
            "timestamp": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
            "spy_price": _safe_round(spy_price),
            "regime": getattr(ctx, "regime", None) or None,
            "volatility": getattr(ctx, "volatility", None) or None,
            "direction": getattr(ctx, "direction_60m", None) or None,
            "confidence": _safe_round(getattr(ctx, "confidence_60m", None)),
            "blended_score": _safe_round(blended),
            "threshold": _safe_round(threshold),
            "threshold_delta": delta,
            "block_reason": getattr(ctx, "block_reason", None) or None,
            "fwd_5m": None,
            "fwd_15m": None,
            "fwd_5m_price": None,
            "fwd_15m_price": None,
            "fwd_5m_status": "pending",
            "fwd_15m_status": "pending",
        })
    except Exception:
        pass


def update_blocked_outcomes(df=None) -> None:
    """
    Fill in fwd_5m / fwd_15m columns for rows that are still "pending".
    Call this from conviction_watcher (or any periodic task) so the update
    happens in the same cycle that already has a fresh market DataFrame.

    Only rows whose status is not already "filled" are updated.
    Writes back only when at least one row changed.
    """
    try:
        if df is None:
            from core.data_service import get_market_dataframe
            df = get_market_dataframe()
        if df is None:
            return

        signals = read_df(
            "SELECT * FROM blocked_signals WHERE fwd_5m_status != 'filled' OR fwd_15m_status != 'filled'"
        )

        if signals.empty:
            return

        signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")
        signals = signals.dropna(subset=["timestamp"])
        if signals.empty:
            return

        # Build aligned market series
        mdf = df.reset_index()
        if "timestamp" not in mdf.columns and "index" in mdf.columns:
            mdf.rename(columns={"index": "timestamp"}, inplace=True)
        mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], errors="coerce")
        mdf = (
            mdf.dropna(subset=["timestamp", "close"])
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
        )
        if mdf.empty:
            return

        # Strip tz so numpy comparison works uniformly
        try:
            if signals["timestamp"].dt.tz is not None:
                signals["timestamp"] = (
                    signals["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
                )
        except Exception:
            pass
        try:
            if mdf["timestamp"].dt.tz is not None:
                mdf["timestamp"] = (
                    mdf["timestamp"].dt.tz_convert("US/Eastern").dt.tz_localize(None)
                )
        except Exception:
            pass

        market_ts = mdf["timestamp"].to_numpy(dtype="datetime64[ns]")
        market_close = mdf["close"].to_numpy()

        def _lookup(target_time):
            """Return (price, status) for the candle closest to target_time."""
            if len(market_ts) == 0:
                return None, "no_market_data"
            try:
                t64 = pd.Timestamp(target_time).to_datetime64()
            except Exception:
                return None, "invalid_time"
            idx = market_ts.searchsorted(t64)
            candidate = min(idx, len(market_ts) - 1)
            delta_sec = abs(
                (pd.Timestamp(market_ts[candidate]) - pd.Timestamp(t64)).total_seconds()
            )
            price = float(market_close[candidate])
            if delta_sec <= 120:
                return price, "filled"
            return price, "estimated"

        with transaction() as conn:
            for _, row in signals.iterrows():
                try:
                    base_price = float(row["spy_price"])
                except (TypeError, ValueError):
                    continue

                ts = row["timestamp"]
                row_id = int(row["id"])
                updates = {}

                # 5-minute forward return
                if row.get("fwd_5m_status") != "filled":
                    p5, s5 = _lookup(ts + timedelta(minutes=5))
                    if p5 is not None:
                        updates["fwd_5m_price"] = round(p5, 4)
                        updates["fwd_5m"] = round(p5 - base_price, 4)
                        updates["fwd_5m_status"] = s5

                # 15-minute forward return
                if row.get("fwd_15m_status") != "filled":
                    p15, s15 = _lookup(ts + timedelta(minutes=15))
                    if p15 is not None:
                        updates["fwd_15m_price"] = round(p15, 4)
                        updates["fwd_15m"] = round(p15 - base_price, 4)
                        updates["fwd_15m_status"] = s15

                if updates:
                    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                    conn.execute(
                        f"UPDATE blocked_signals SET {set_clause} WHERE id = ?",
                        list(updates.values()) + [row_id],
                    )
    except Exception:
        pass
