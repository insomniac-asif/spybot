"""
simulation/prediction_backfill.py

Late-start prediction backfill: when the bot starts after market open,
replays the prediction pipeline for each missed 10-minute slot so the
prediction log in SQLite has no gaps.

Also provides a prediction lock mechanism that stops new predictions
from being written after 16:00 ET (market close).
"""

import logging
import os
from datetime import datetime, date, timedelta

import pytz

ET = pytz.timezone("US/Eastern")

# ---------------------------------------------------------------------------
# Prediction Lock — prevents writes after market close
# ---------------------------------------------------------------------------

_PREDICTION_LOCK_DATE: date | None = None


def is_prediction_locked() -> bool:
    """Returns True if predictions are locked for today (after 16:00 ET)."""
    global _PREDICTION_LOCK_DATE
    now_et = datetime.now(ET)
    today = now_et.date()

    if _PREDICTION_LOCK_DATE == today:
        return True

    if now_et.hour >= 16:
        _PREDICTION_LOCK_DATE = today
        logging.error("predictions_locked: date=%s", today.isoformat())
        return True

    return False


def reset_prediction_lock() -> None:
    """Reset the lock (for testing or new-day reset)."""
    global _PREDICTION_LOCK_DATE
    _PREDICTION_LOCK_DATE = None


# ---------------------------------------------------------------------------
# Backfill missed predictions
# ---------------------------------------------------------------------------

def backfill_missed_predictions() -> dict:
    """
    Detect gap between market open and now, replay predictions for missed
    10-minute slots. Uses the same make_prediction() + log_prediction() path
    as the live forecast_watcher.

    Returns summary dict: {total_slots, predictions_generated, symbols}
    """
    try:
        now_et = datetime.now(ET)

        # Only backfill during market hours (9:30 - 16:00 ET)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

        if now_et < market_open or now_et > market_close:
            return {"total_slots": 0, "predictions_generated": 0, "reason": "outside_market_hours"}

        # If bot started within first 10 minutes, no backfill needed
        if now_et <= market_open + timedelta(minutes=10):
            return {"total_slots": 0, "predictions_generated": 0, "reason": "near_market_open"}

        # Find last prediction timestamp in SQLite for today
        last_pred_time = _get_last_prediction_time_today()

        if last_pred_time is not None:
            gap_start = last_pred_time + timedelta(minutes=10)
        else:
            # No predictions today — start from first 10-min slot after open
            gap_start = market_open + timedelta(minutes=10)

        # Round gap_start up to next 10-minute boundary
        gap_start = gap_start.replace(
            minute=(gap_start.minute // 10) * 10,
            second=0,
            microsecond=0,
        )

        # Don't backfill into the future
        gap_end = now_et - timedelta(minutes=1)
        if gap_start >= gap_end:
            return {"total_slots": 0, "predictions_generated": 0, "reason": "no_gap"}

        # Load symbol registry
        from core.data_service import _load_symbol_registry, get_symbol_dataframe
        registry = _load_symbol_registry()
        if not registry:
            return {"total_slots": 0, "predictions_generated": 0, "reason": "no_symbol_registry"}

        symbols = [s.upper() for s in registry]

        # Load dataframes for all symbols
        sym_dfs = {}
        for sym in symbols:
            try:
                sym_df = get_symbol_dataframe(sym)
                if sym_df is not None and len(sym_df) > 30:
                    sym_dfs[sym] = sym_df
            except Exception:
                continue

        if not sym_dfs:
            return {"total_slots": 0, "predictions_generated": 0, "reason": "no_dataframes"}

        # Generate 10-minute slots to backfill
        from signals.predictor import make_prediction
        from signals.regime import get_regime
        from signals.volatility import volatility_state
        from analytics.prediction_stats import log_prediction

        slots = []
        current = gap_start
        while current <= gap_end:
            if market_open <= current <= market_close:
                slots.append(current)
            current += timedelta(minutes=10)

        total_predictions = 0

        for slot_time in slots:
            for sym, sym_df in sym_dfs.items():
                try:
                    # Slice df up to this slot time (simulate having data only up to that point)
                    if "timestamp" in sym_df.columns:
                        ts_col = sym_df["timestamp"]
                        if hasattr(ts_col.iloc[0], "tzinfo") and ts_col.iloc[0].tzinfo is not None:
                            slot_aware = slot_time if slot_time.tzinfo else ET.localize(slot_time)
                            mask = ts_col <= slot_aware
                        else:
                            mask = ts_col <= slot_time.replace(tzinfo=None)
                        sliced = sym_df[mask]
                    else:
                        sliced = sym_df

                    if sliced is None or len(sliced) < 30:
                        continue

                    pred = make_prediction(10, sliced)
                    if pred is None:
                        continue

                    # Override prediction time with the slot time
                    pred["time"] = slot_time.isoformat()

                    regime = get_regime(sliced)
                    vola = volatility_state(sliced)

                    log_prediction(pred, regime, vola, symbol=sym)
                    total_predictions += 1
                except Exception:
                    continue

        logging.error(
            "prediction_backfill_complete: slots=%d predictions=%d symbols=%d gap=%s_to_%s",
            len(slots), total_predictions, len(sym_dfs),
            gap_start.strftime("%H:%M"), gap_end.strftime("%H:%M"),
        )

        return {
            "total_slots": len(slots),
            "predictions_generated": total_predictions,
            "symbols": list(sym_dfs.keys()),
            "gap_start": gap_start.isoformat(),
            "gap_end": gap_end.isoformat(),
        }

    except Exception:
        logging.exception("prediction_backfill_error")
        return {"total_slots": 0, "predictions_generated": 0, "reason": "error"}


def _get_last_prediction_time_today() -> datetime | None:
    """Query SQLite for the most recent prediction timestamp today."""
    try:
        from core.analytics_db import get_conn
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        conn = get_conn()
        try:
            cursor = conn.execute(
                "SELECT MAX(time) FROM predictions WHERE time LIKE ?",
                (f"{today_str}%",),
            )
            row = cursor.fetchone()
            if row and row[0]:
                dt = datetime.fromisoformat(str(row[0]))
                if dt.tzinfo is None:
                    dt = ET.localize(dt)
                return dt
        finally:
            conn.close()
    except Exception:
        pass
    return None
