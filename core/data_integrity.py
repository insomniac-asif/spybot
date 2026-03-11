import calendar
from datetime import date

import pandas as pd
import pytz

# US equity early-close dates (market closes at 1:00 PM ET).
_EARLY_CLOSE_DATES = set()
for _y in range(2024, 2028):
    _EARLY_CLOSE_DATES.add(date(_y, 7, 3))    # Day before Independence Day
    _EARLY_CLOSE_DATES.add(date(_y, 12, 24))  # Christmas Eve
    # Black Friday = day after Thanksgiving (4th Thursday of November)
    _nov_cal = calendar.monthcalendar(_y, 11)
    _thurs = [w[calendar.THURSDAY] for w in _nov_cal if w[calendar.THURSDAY] != 0]
    if len(_thurs) >= 4:
        _EARLY_CLOSE_DATES.add(date(_y, 11, _thurs[3] + 1))


def validate_market_dataframe(df):
    """
    Validate market dataframe integrity before trading decisions.

    Returns:
        {
            "valid": bool,
            "errors": [str, ...]
        }

    Inline test ideas:
    - Missing minute scenario:
      A dataframe with 09:30, 09:31, 09:33 in the same RTH session should fail continuity.
    - Duplicate timestamp scenario:
      A dataframe with duplicated 10:15 timestamp should fail duplicate check.
    - Valid dataframe scenario:
      Continuous 1-minute RTH bars, unique/monotonic index, sane OHLC should pass.
    """
    errors = []

    # A) Empty or insufficient length
    if df is None:
        return {"valid": False, "errors": ["df_none"]}

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("index_not_datetimeindex")
        return {"valid": False, "errors": errors}

    # Dynamic minimum length: allow early-session trading without requiring 30 bars
    required_len = 30
    try:
        eastern = pytz.timezone("America/New_York")
        last_ts = df.index[-1] if len(df.index) > 0 else None
        if last_ts is not None:
            ts = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
            if ts.tzinfo is None:
                ts = eastern.localize(ts)
            else:
                ts = ts.astimezone(eastern)
            session_open = ts.replace(hour=9, minute=30, second=0, microsecond=0)
            session_close = ts.replace(hour=16, minute=0, second=0, microsecond=0)
            if session_open <= ts <= session_close:
                minutes_since_open = int((ts - session_open).total_seconds() // 60)
                required_len = max(2, min(30, minutes_since_open + 1))
    except Exception:
        required_len = 30

    if len(df) < required_len:
        errors.append(f"insufficient_length_lt_{required_len}")

    # C) Duplicate timestamps
    if df.index.has_duplicates:
        errors.append("duplicate_timestamps")

    # D) Monotonic index (strictly increasing)
    if not df.index.is_monotonic_increasing:
        errors.append("index_not_monotonic_increasing")

    # B) Timestamp continuity during RTH (09:30-16:00 America/New_York)
    eastern = pytz.timezone("America/New_York")
    if df.index.tz is None:
        # Naive timestamps in this codebase are ET (America/New_York), not UTC
        idx_eastern = df.index.tz_localize(eastern, ambiguous="infer", nonexistent="shift_forward")
    else:
        idx_eastern = df.index.tz_convert(eastern)

    df_eastern = df.copy()
    df_eastern.index = idx_eastern

    _today = pd.Timestamp.now(tz=eastern).date()
    rth_df = df_eastern.between_time("09:30", "16:00")
    if not rth_df.empty:
        for session_date, session in rth_df.groupby(rth_df.index.date):
            # Skip today (session still in progress) and early-close days
            if session_date >= _today or session_date in _EARLY_CLOSE_DATES:
                continue
            session_idx = session.index.sort_values()
            if session_idx.empty:
                continue
            expected = pd.date_range(
                start=session_idx[0],
                end=session_idx[-1],
                freq="1min",
                tz=session_idx.tz
            )
            missing = expected.difference(session_idx)
            if len(missing) > 5:
                errors.append(
                    f"missing_rth_minutes:{session_date}:count={len(missing)}"
                )

    # E) OHLC sanity
    required_cols = ["open", "high", "low", "close"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"missing_ohlc_columns:{','.join(missing_cols)}")
    else:
        o = pd.Series(pd.to_numeric(df["open"], errors="coerce"), index=df.index)
        h = pd.Series(pd.to_numeric(df["high"], errors="coerce"), index=df.index)
        l = pd.Series(pd.to_numeric(df["low"], errors="coerce"), index=df.index)
        c = pd.Series(pd.to_numeric(df["close"], errors="coerce"), index=df.index)

        invalid_nan = o.isna() | h.isna() | l.isna() | c.isna()
        if invalid_nan.any():
            errors.append("ohlc_non_numeric_or_nan")

        bad_high = h < pd.concat([o, c], axis=1).max(axis=1)
        bad_low = l > pd.concat([o, c], axis=1).min(axis=1)
        bad_range = h < l
        if bad_high.any() or bad_low.any() or bad_range.any():
            errors.append("ohlc_sanity_violation")

    return {"valid": len(errors) == 0, "errors": errors}
