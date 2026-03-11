# simulation/sim_signal_funcs_smc2.py
#
# Second half of SMC signal functions, split from sim_signal_funcs_smc.py
# to keep both files under 500 lines.
# Imports shared helpers and aggregation utilities from sim_signal_funcs_smc.

from simulation.sim_signal_funcs import _find_col
from simulation.sim_signal_funcs_smc import _aggregate_bars


def _signal_multi_tf_confirm(df) -> tuple:
    """
    Multi-timeframe EMA alignment: 5m EMA9>EMA20 AND 15m EMA9>EMA20 AND close>VWAP
    for BULLISH; reverse for BEARISH.  EMAs are computed from aggregated 1m bars.
    Requires at least 150 1m bars (10 × 15m bars with EMA warmup).
    """
    MIN_BARS = 150
    MIN_5M_BARS = 20   # need 20 5m bars for EMA20
    MIN_15M_BARS = 10  # need 10 15m bars minimum

    def _ema_last(prices, span):
        k = 2.0 / (span + 1)
        val = float(prices[0])
        for p in prices[1:]:
            val = float(p) * k + val * (1 - k)
        return val

    try:
        if df is None or len(df) < MIN_BARS:
            return None, None, {"reason": "insufficient_bars"}

        close_col = _find_col(df, ["close", "Close"])
        high_col  = _find_col(df, ["high",  "High"])
        low_col   = _find_col(df, ["low",   "Low"])
        open_col  = _find_col(df, ["open",  "Open"])
        vol_col   = _find_col(df, ["volume", "Volume"])
        vwap_col  = _find_col(df, ["vwap",  "VWAP"])

        if close_col is None or vwap_col is None:
            return None, None, {"reason": "missing_cols"}
        if high_col is None or low_col is None or open_col is None:
            return None, None, {"reason": "missing_ohlc_cols"}

        close = float(df.iloc[-1][close_col])
        vwap  = float(df.iloc[-1][vwap_col])

        bars_5m  = _aggregate_bars(df, close_col, high_col, low_col, open_col, vol_col, 5)
        bars_15m = _aggregate_bars(df, close_col, high_col, low_col, open_col, vol_col, 15)

        if len(bars_5m) < MIN_5M_BARS or len(bars_15m) < MIN_15M_BARS:
            return None, None, {"reason": "insufficient_tf_bars"}

        closes_5m  = [b["close"] for b in bars_5m]
        closes_15m = [b["close"] for b in bars_15m]

        ema9_5m   = _ema_last(closes_5m,  9)
        ema20_5m  = _ema_last(closes_5m,  20)
        ema9_15m  = _ema_last(closes_15m, 9)
        ema20_15m = _ema_last(closes_15m, 20)

        bull_5m  = ema9_5m  > ema20_5m
        bull_15m = ema9_15m > ema20_15m

        if bull_5m and bull_15m and close > vwap:
            return "BULLISH", close, {
                "reason": "multi_tf_bull",
                "ema9_5m":  round(ema9_5m,  4),
                "ema20_5m": round(ema20_5m, 4),
                "ema9_15m": round(ema9_15m,  4),
            }

        if (not bull_5m) and (not bull_15m) and close < vwap:
            return "BEARISH", close, {
                "reason": "multi_tf_bear",
                "ema9_5m":  round(ema9_5m,  4),
                "ema20_5m": round(ema20_5m, 4),
                "ema9_15m": round(ema9_15m,  4),
            }

        return None, None, {"reason": "no_multi_tf_align"}
    except Exception:
        return None, None, {"reason": "multi_tf_confirm_error"}


def _signal_gap_fade(df) -> tuple:
    """
    Fade an opening gap: if the bar 30 bars ago opened >0.3% away from the
    bar before it, and price has not yet filled the gap, signal a reversion.
    Fires only when price is still within ~30 bars of the gap event.
    """
    GAP_MIN_PCT = 0.0015  # was 0.003; allow smaller intraday gaps
    WINDOW = 15  # was 30; check sooner after gap forms

    try:
        if df is None or len(df) < WINDOW + 2:
            return None, None, {"reason": "insufficient_bars"}

        close_col = _find_col(df, ["close", "Close"])
        open_col  = _find_col(df, ["open",  "Open"])

        if close_col is None or open_col is None:
            return None, None, {"reason": "missing_cols"}

        # Bar at session open (WINDOW bars ago) and the bar just before it
        session_open_bar = df.iloc[-(WINDOW + 1)]
        pre_gap_bar      = df.iloc[-(WINDOW + 2)]
        curr             = df.iloc[-1]

        prev_close        = float(pre_gap_bar[close_col])
        session_open_px   = float(session_open_bar[open_col])
        curr_close        = float(curr[close_col])

        if prev_close <= 0:
            return None, None, {"reason": "invalid_prev_close"}

        gap_pct = (session_open_px - prev_close) / prev_close

        if abs(gap_pct) < GAP_MIN_PCT:
            return None, None, {"reason": "gap_too_small"}

        # Gap up → fade bearish (price hasn't yet returned below prev_close)
        if gap_pct > GAP_MIN_PCT and curr_close > prev_close:
            return "BEARISH", curr_close, {
                "reason":       "gap_fade_down",
                "gap_pct":      round(gap_pct, 4),
                "prev_close":   round(prev_close, 2),
                "session_open": round(session_open_px, 2),
            }

        # Gap down → fade bullish (price hasn't yet returned above prev_close)
        if gap_pct < -GAP_MIN_PCT and curr_close < prev_close:
            return "BULLISH", curr_close, {
                "reason":       "gap_fade_up",
                "gap_pct":      round(gap_pct, 4),
                "prev_close":   round(prev_close, 2),
                "session_open": round(session_open_px, 2),
            }

        return None, None, {"reason": "gap_already_filled"}
    except Exception:
        return None, None, {"reason": "gap_fade_error"}


def _signal_vpoc_reversion(df) -> tuple:
    """
    Volume Point of Control (VPOC) reversion: build a $0.50-bucket volume
    profile over the last 30 bars, find the highest-volume price level,
    then signal reversion when price is >0.5% from VPOC and RSI diverges.
    """
    BUCKET_SIZE      = 0.50
    MIN_BARS         = 30
    RSI_OVERSOLD     = 40
    RSI_OVERBOUGHT   = 60
    DIST_THRESHOLD   = 0.005   # 0.5% away from VPOC

    try:
        if df is None or len(df) < MIN_BARS:
            return None, None, {"reason": "insufficient_bars"}

        close_col = _find_col(df, ["close", "Close"])
        vol_col   = _find_col(df, ["volume", "Volume"])
        rsi_col   = _find_col(df, ["rsi", "RSI", "rsi14", "RSI14"])

        if close_col is None or vol_col is None or rsi_col is None:
            return None, None, {"reason": "missing_cols"}

        recent = df.tail(MIN_BARS)

        # Build $0.50-bucket volume profile
        vol_by_bucket: dict = {}
        for _, row in recent.iterrows():
            bucket = round(float(row[close_col]) / BUCKET_SIZE) * BUCKET_SIZE
            vol_by_bucket[bucket] = vol_by_bucket.get(bucket, 0.0) + float(row[vol_col])

        if not vol_by_bucket:
            return None, None, {"reason": "no_volume_data"}

        vpoc = max(vol_by_bucket, key=vol_by_bucket.__getitem__)

        curr_close = float(recent.iloc[-1][close_col])
        curr_rsi   = float(recent.iloc[-1][rsi_col])

        if vpoc <= 0:
            return None, None, {"reason": "invalid_vpoc"}

        dist_pct = (curr_close - vpoc) / vpoc

        if dist_pct > DIST_THRESHOLD and curr_rsi > RSI_OVERBOUGHT:
            return "BEARISH", curr_close, {
                "reason":   "vpoc_reversion_short",
                "vpoc":     round(vpoc, 2),
                "dist_pct": round(dist_pct, 4),
                "rsi":      round(curr_rsi, 1),
            }

        if dist_pct < -DIST_THRESHOLD and curr_rsi < RSI_OVERSOLD:
            return "BULLISH", curr_close, {
                "reason":   "vpoc_reversion_long",
                "vpoc":     round(vpoc, 2),
                "dist_pct": round(dist_pct, 4),
                "rsi":      round(curr_rsi, 1),
            }

        return None, None, {"reason": "no_vpoc_reversion"}
    except Exception:
        return None, None, {"reason": "vpoc_reversion_error"}


def _signal_opening_range_reclaim(df) -> tuple:
    """
    Detects a reclaim of the opening range after a flush-through.
    - Bullish: prev bar closed below or_low, current bar reclaims above it,
      close > vwap, and RSI < 60 (hasn't already ripped).
    - Bearish: prev bar closed above or_high, current bar fails back inside,
      close < vwap, and RSI > 40.
    Opening range = first 6 bars of the dataframe.  Requires 30+ bars total.
    """
    try:
        if df is None or len(df) < 30:
            return None, None, {"reason": "insufficient_bars"}

        close_col = _find_col(df, ["close", "Close"])
        high_col  = _find_col(df, ["high",  "High"])
        low_col   = _find_col(df, ["low",   "Low"])
        rsi_col   = _find_col(df, ["rsi",   "RSI", "rsi14", "RSI14"])
        vwap_col  = _find_col(df, ["vwap",  "VWAP"])

        if None in (close_col, high_col, low_col, rsi_col, vwap_col):
            return None, None, {"reason": "missing_cols"}

        or_high = float(df[high_col].iloc[:6].max())
        or_low  = float(df[low_col].iloc[:6].min())

        # Don't evaluate while the opening range is still forming
        if len(df) <= 6:
            return None, None, {"reason": "in_range_formation"}

        prev_close = float(df.iloc[-2][close_col])
        curr       = df.iloc[-1]
        close      = float(curr[close_col])
        rsi        = float(curr[rsi_col])
        vwap       = float(curr[vwap_col])

        if prev_close < or_low and close > or_low and close > vwap and rsi < 60:
            return "BULLISH", close, {
                "reason": "orr_bull_reclaim",
                "or_low": round(or_low, 2),
                "rsi":    round(rsi, 1),
            }

        if prev_close > or_high and close < or_high and close < vwap and rsi > 40:
            return "BEARISH", close, {
                "reason": "orr_bear_reclaim",
                "or_high": round(or_high, 2),
                "rsi":     round(rsi, 1),
            }

        return None, None, {"reason": "no_orr_signal"}
    except Exception:
        return None, None, {"reason": "opening_range_reclaim_error"}


def _signal_vol_compression_breakout(df) -> tuple:
    """
    Detects a volatility compression-to-expansion event.
    Previous bar's ATR must be below 60% of the 20-bar ATR mean (compression),
    and current bar's ATR must have recovered to 90%+ of the mean (expansion).
    Direction is determined by close vs EMA9.
    """
    COMPRESSION_LOOKBACK  = 20
    COMPRESSION_THRESHOLD = 0.6

    try:
        if df is None or len(df) < COMPRESSION_LOOKBACK + 2:
            return None, None, {"reason": "insufficient_bars"}

        atr_col = _find_col(df, ["atr", "ATR"])
        if atr_col is None:
            return None, None, {"reason": "missing_cols"}

        atr_vals  = df[atr_col].values
        window    = atr_vals[-(COMPRESSION_LOOKBACK + 1):-1]
        atr_mean  = float(sum(window) / len(window))
        atr_prev  = float(atr_vals[-2])
        atr_now   = float(atr_vals[-1])

        if atr_mean <= 0:
            return None, None, {"reason": "zero_atr_mean"}

        # Check compression in prev 2 bars (was strictly prev bar only)
        atr_prev2 = float(atr_vals[-3]) if len(atr_vals) >= 3 else atr_prev
        compressed = atr_prev < atr_mean * COMPRESSION_THRESHOLD or atr_prev2 < atr_mean * COMPRESSION_THRESHOLD
        if not compressed:
            return None, None, {"reason": "no_compression"}

        if not (atr_now > atr_mean * 0.75):  # was 0.9; lowered for realistic expansion
            return None, None, {"reason": "no_expansion"}

        close_col = _find_col(df, ["close", "Close"])
        ema9_col  = _find_col(df, ["ema9",  "EMA9", "ema_9"])

        if close_col is None or ema9_col is None:
            return None, None, {"reason": "missing_cols"}

        close = float(df.iloc[-1][close_col])
        ema9  = float(df.iloc[-1][ema9_col])

        if close > ema9:
            return "BULLISH", close, {
                "reason":    "vcb_bull",
                "atr_mean":  round(atr_mean, 4),
                "atr_ratio": round(atr_now / atr_mean, 3),
            }

        if close < ema9:
            return "BEARISH", close, {
                "reason":    "vcb_bear",
                "atr_mean":  round(atr_mean, 4),
                "atr_ratio": round(atr_now / atr_mean, 3),
            }

        return None, None, {"reason": "no_direction"}
    except Exception:
        return None, None, {"reason": "vol_compression_breakout_error"}


def _signal_vol_spike_fade(df) -> tuple:
    """
    Fades intraday volatility spikes when RSI is stretched and price is far
    from VWAP.  Complements VOL_COMPRESSION_BREAKOUT: that signal trades
    WITH the expansion; this one trades AGAINST it when it's overextended.
    - Current ATR must be 1.5x the 19-bar ATR mean (spike detected).
    - abs(close - vwap) must exceed 1.5x the ATR mean (meaningful dislocation).
    - Direction: close > vwap + RSI > 70 → BEARISH; close < vwap + RSI < 30 → BULLISH.
    """
    SPIKE_ATR_MULTIPLIER   = 1.5
    VWAP_DIST_ATR_MULT     = 1.5

    try:
        if df is None or len(df) < 20:
            return None, None, {"reason": "insufficient_bars"}

        close_col = _find_col(df, ["close",  "Close"])
        vwap_col  = _find_col(df, ["vwap",   "VWAP"])
        atr_col   = _find_col(df, ["atr",    "ATR"])
        rsi_col   = _find_col(df, ["rsi",    "RSI", "rsi14", "RSI14"])

        if None in (close_col, vwap_col, atr_col, rsi_col):
            return None, None, {"reason": "missing_cols"}

        atr_vals  = df[atr_col].values
        window    = atr_vals[-20:-1]          # 19 bars, excluding current
        atr_mean  = float(sum(window) / len(window))
        atr_now   = float(atr_vals[-1])

        if atr_mean <= 0:
            return None, None, {"reason": "zero_atr_mean"}

        if atr_now <= atr_mean * SPIKE_ATR_MULTIPLIER:
            return None, None, {"reason": "no_vol_spike"}

        curr  = df.iloc[-1]
        close = float(curr[close_col])
        vwap  = float(curr[vwap_col])
        rsi   = float(curr[rsi_col])

        vwap_dist = abs(close - vwap)
        if vwap_dist <= atr_mean * VWAP_DIST_ATR_MULT:
            return None, None, {"reason": "price_near_vwap"}

        if close > vwap and rsi > 70:
            return "BEARISH", close, {
                "reason":    "vsf_fade_bear",
                "rsi":       round(rsi, 1),
                "atr_ratio": round(atr_now / atr_mean, 3),
            }

        if close < vwap and rsi < 30:
            return "BULLISH", close, {
                "reason":    "vsf_fade_bull",
                "rsi":       round(rsi, 1),
                "atr_ratio": round(atr_now / atr_mean, 3),
            }

        return None, None, {"reason": "no_vsf_signal"}
    except Exception:
        return None, None, {"reason": "vol_spike_fade_error"}
