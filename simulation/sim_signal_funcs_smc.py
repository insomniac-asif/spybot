# simulation/sim_signal_funcs_smc.py
#
# SMC/advanced signal functions extracted from sim_signal_funcs.py.
# Imports shared helpers (_find_col, _ctx_float, _safe_float) from
# sim_signal_funcs to avoid duplication.

from simulation.sim_signal_funcs import _find_col, _ctx_float, _safe_float, _signal_trend_pullback

# ---------------------------------------------------------------------------
# SMC signal helpers — shared state, aggregation, FVG detection, cooldowns
# ---------------------------------------------------------------------------

# Per-session cooldown state: resets on process restart (intentional)
_signal_cooldowns: dict = {}

_SMC_COOLDOWN = _signal_cooldowns  # alias used externally (if any)


def _check_cooldown(mode: str, direction: str, current_bar_count: int, cooldown_bars: int) -> bool:
    key = (mode, direction)
    last = _signal_cooldowns.get(key, -999999)
    return (current_bar_count - last) >= cooldown_bars


def _set_cooldown(mode: str, direction: str, current_bar_count: int) -> None:
    _signal_cooldowns[(mode, direction)] = current_bar_count


def _aggregate_bars(df, close_col, high_col, low_col, open_col, vol_col, n_minutes):
    """Aggregate 1m OHLCV into n-minute bars. Returns list of dicts."""
    rows = len(df)
    bars = []
    for i in range(0, rows - n_minutes + 1, n_minutes):
        chunk = df.iloc[i:i + n_minutes]
        bar = {
            "open":      float(chunk[open_col].iloc[0]),
            "high":      float(chunk[high_col].max()),
            "low":       float(chunk[low_col].min()),
            "close":     float(chunk[close_col].iloc[-1]),
            "volume":    float(chunk[vol_col].sum()) if vol_col else 0.0,
            "bar_index": i + n_minutes - 1,
        }
        bars.append(bar)
    return bars


def _detect_fvg_zones(bars, min_gap_pct, max_age_bars):
    """Scan aggregated bars for FVG patterns. Returns list of unfilled FVG zone dicts."""
    zones = []
    if len(bars) < 3:
        return zones
    scan_start = max(0, len(bars) - 3 - max_age_bars)
    for i in range(scan_start, len(bars) - 2):
        c1, c2, c3 = bars[i], bars[i + 1], bars[i + 2]
        mid_price = c2["close"]
        if mid_price <= 0:
            continue
        # Bullish FVG: gap above candle 1's high, below candle 3's low
        if c1["high"] < c3["low"]:
            gap_size = (c3["low"] - c1["high"]) / mid_price
            if gap_size >= min_gap_pct:
                zones.append({
                    "direction":   "BULLISH",
                    "zone_top":    c3["low"],
                    "zone_bottom": c1["high"],
                    "bar_index":   c2["bar_index"],
                    "filled":      False,
                })
        # Bearish FVG: gap below candle 1's low, above candle 3's high
        if c1["low"] > c3["high"]:
            gap_size = (c1["low"] - c3["high"]) / mid_price
            if gap_size >= min_gap_pct:
                zones.append({
                    "direction":   "BEARISH",
                    "zone_top":    c1["low"],
                    "zone_bottom": c3["high"],
                    "bar_index":   c2["bar_index"],
                    "filled":      False,
                })
    # Mark filled zones
    bar_index_map = {b["bar_index"]: bi for bi, b in enumerate(bars)}
    for zone in zones:
        zone_pos = bar_index_map.get(zone["bar_index"])
        if zone_pos is not None:
            for b in bars[zone_pos + 2:]:
                if b["high"] >= zone["zone_top"] and b["low"] <= zone["zone_bottom"]:
                    zone["filled"] = True
                    break
    return [z for z in zones if not z["filled"]]


# ---------------------------------------------------------------------------
# SMC Signal constants
# ---------------------------------------------------------------------------

FVG_4H_AGG_MINUTES   = 240
FVG_4H_MIN_GAP_PCT   = 0.0005
FVG_4H_MAX_AGE_BARS  = 12
FVG_4H_COOLDOWN_BARS = 60

FVG_5M_AGG_MINUTES      = 5
FVG_5M_MIN_GAP_PCT      = 0.0001   # was 0.0002; detect smaller FVG gaps
FVG_5M_MAX_AGE_BARS     = 30       # was 20; look further back for unfilled zones
FVG_5M_COOLDOWN_BARS    = 5        # was 10; allow faster re-entry
FVG_5M_VOL_MULTIPLIER   = 0.3      # was 0.5; accept lower-volume zones

SWEEP_SWING_LOOKBACK      = 3     # was 5; detect more swing points
SWEEP_CONFIRMATION_BARS   = 7     # was 5; wider confirmation window
SWEEP_MIN_BODY_RATIO      = 0.2   # was 0.3; accept thinner reversal candles
SWEEP_COOLDOWN_BARS       = 8     # was 15; faster re-entry
SWEEP_MIN_BREAK_PCT       = 0.00005  # was 0.0001; smaller break threshold

COMBO_SWEEP_WINDOW   = 3
COMBO_COOLDOWN_BARS  = 10   # was 20; faster re-entry

FLOW_VWAP_SLOPE_LOOKBACK   = 10
FLOW_VOL_SMA_LOOKBACK      = 20
FLOW_VOL_SPIKE_THRESHOLD   = 1.0   # was 1.2; any above-average volume qualifies
FLOW_MOMENTUM_LOOKBACK     = 10
FLOW_MOMENTUM_MIN_ATR      = 0.15  # was 0.3; less extreme momentum needed
FLOW_COOLDOWN_BARS         = 15    # was 30; faster re-entry


# ---------------------------------------------------------------------------
# SMC Signal functions
# ---------------------------------------------------------------------------

def _signal_fvg_4h(df) -> tuple:
    try:
        if len(df) < FVG_4H_AGG_MINUTES * 2:  # was *4 (960); now *2 (480) — fits within a day
            return None, None, {"reason": "insufficient_data"}
        close_col = _find_col(df, ["close", "Close"])
        high_col  = _find_col(df, ["high", "High"])
        low_col   = _find_col(df, ["low", "Low"])
        open_col  = _find_col(df, ["open", "Open"])
        vol_col   = _find_col(df, ["volume", "Volume", "vol"])
        if not all([close_col, high_col, low_col, open_col]):
            return None, None, {"reason": "missing_columns"}
        bars = _aggregate_bars(df, close_col, high_col, low_col, open_col, vol_col, FVG_4H_AGG_MINUTES)
        if len(bars) < 3:
            return None, None, {"reason": "insufficient_bars"}
        zones = _detect_fvg_zones(bars, FVG_4H_MIN_GAP_PCT, FVG_4H_MAX_AGE_BARS)
        current_close = float(df[close_col].iloc[-1])
        # Compute ATR for proximity check
        atr = sum(float(df[high_col].iloc[i]) - float(df[low_col].iloc[i])
                  for i in range(-min(14, len(df)), 0)) / min(14, len(df))
        n = len(df)
        for zone in reversed(zones):
            zone_mid = (zone["zone_top"] + zone["zone_bottom"]) / 2
            # Signal on proximity: within zone OR within 1 ATR of zone boundary
            in_zone = zone["zone_bottom"] <= current_close <= zone["zone_top"]
            near_zone = abs(current_close - zone_mid) <= (zone["zone_top"] - zone["zone_bottom"]) / 2 + atr
            if in_zone or near_zone:
                d = zone["direction"]
                if _check_cooldown("FVG_4H", d, n, FVG_4H_COOLDOWN_BARS):
                    _set_cooldown("FVG_4H", d, n)
                    return d, current_close, {"signal": "fvg_4h", "zone": zone["direction"]}
        return None, None, {"reason": "no_fvg_zone"}
    except Exception:
        return None, None, {"reason": "fvg_4h_error"}


def _signal_fvg_5m(df) -> tuple:
    try:
        if len(df) < FVG_5M_AGG_MINUTES * 6:
            return None, None, {"reason": "insufficient_data"}
        close_col = _find_col(df, ["close", "Close"])
        high_col  = _find_col(df, ["high", "High"])
        low_col   = _find_col(df, ["low", "Low"])
        open_col  = _find_col(df, ["open", "Open"])
        vol_col   = _find_col(df, ["volume", "Volume", "vol"])
        if not all([close_col, high_col, low_col, open_col]):
            return None, None, {"reason": "missing_columns"}
        bars = _aggregate_bars(df, close_col, high_col, low_col, open_col, vol_col, FVG_5M_AGG_MINUTES)
        if len(bars) < 3:
            return None, None, {"reason": "insufficient_bars"}
        vol_sma = sum(b["volume"] for b in bars[-20:]) / min(len(bars), 20) if vol_col else 0.0
        zones = _detect_fvg_zones(bars, FVG_5M_MIN_GAP_PCT, FVG_5M_MAX_AGE_BARS)
        if vol_col and vol_sma > 0:
            bar_vol_map = {b["bar_index"]: b["volume"] for b in bars}
            zones = [z for z in zones if bar_vol_map.get(z["bar_index"], 0) >= vol_sma * FVG_5M_VOL_MULTIPLIER]
        current_close = float(df[close_col].iloc[-1])
        # ATR for proximity
        atr = sum(float(df[high_col].iloc[i]) - float(df[low_col].iloc[i])
                  for i in range(-min(14, len(df)), 0)) / min(14, len(df))
        n = len(df)
        for zone in reversed(zones):
            zone_mid = (zone["zone_top"] + zone["zone_bottom"]) / 2
            in_zone = zone["zone_bottom"] <= current_close <= zone["zone_top"]
            near_zone = abs(current_close - zone_mid) <= (zone["zone_top"] - zone["zone_bottom"]) / 2 + atr * 0.5
            if in_zone or near_zone:
                d = zone["direction"]
                if _check_cooldown("FVG_5M", d, n, FVG_5M_COOLDOWN_BARS):
                    _set_cooldown("FVG_5M", d, n)
                    return d, current_close, {"signal": "fvg_5m", "zone": zone["direction"]}
        return None, None, {"reason": "no_fvg_zone"}
    except Exception:
        return None, None, {"reason": "fvg_5m_error"}


def _signal_liquidity_sweep(df) -> tuple:
    try:
        if len(df) < 60:
            return None, None, {"reason": "insufficient_data"}
        close_col = _find_col(df, ["close", "Close"])
        high_col  = _find_col(df, ["high", "High"])
        low_col   = _find_col(df, ["low", "Low"])
        open_col  = _find_col(df, ["open", "Open"])
        if not all([close_col, high_col, low_col, open_col]):
            return None, None, {"reason": "missing_columns"}
        closes = df[close_col].values
        highs  = df[high_col].values
        lows   = df[low_col].values
        opens  = df[open_col].values
        n = len(df)
        # Identify swing highs/lows in last 60 bars
        swing_highs, swing_lows = [], []
        scan_start = max(SWEEP_SWING_LOOKBACK, n - 60)
        for i in range(scan_start, n - SWEEP_SWING_LOOKBACK):
            left_h  = highs[i - SWEEP_SWING_LOOKBACK:i]
            right_h = highs[i + 1:i + 1 + SWEEP_SWING_LOOKBACK]
            if len(left_h) > 0 and len(right_h) > 0:
                if highs[i] > max(left_h) and highs[i] > max(right_h):
                    swing_highs.append((i, float(highs[i])))
            left_l  = lows[i - SWEEP_SWING_LOOKBACK:i]
            right_l = lows[i + 1:i + 1 + SWEEP_SWING_LOOKBACK]
            if len(left_l) > 0 and len(right_l) > 0:
                if lows[i] < min(left_l) and lows[i] < min(right_l):
                    swing_lows.append((i, float(lows[i])))
        # Check buyside sweep: broke above swing high, closed back below
        for sh_idx, sh_level in reversed(swing_highs):
            window_start = max(sh_idx + 1, n - SWEEP_CONFIRMATION_BARS - 2)
            for j in range(window_start, n):
                if float(highs[j]) > sh_level * (1 + SWEEP_MIN_BREAK_PCT):
                    for k in range(j + 1, min(j + SWEEP_CONFIRMATION_BARS + 1, n)):
                        if float(closes[k]) < sh_level:
                            body = abs(float(closes[k]) - float(opens[k]))
                            rng  = float(highs[k]) - float(lows[k])
                            if rng > 0 and body / rng >= SWEEP_MIN_BODY_RATIO:
                                if _check_cooldown("SWEEP", "BEARISH", n, SWEEP_COOLDOWN_BARS):
                                    _set_cooldown("SWEEP", "BEARISH", n)
                                    return "BEARISH", float(closes[-1]), {"signal": "sweep_buyside"}
                    break
        # Check sellside sweep: broke below swing low, closed back above
        for sl_idx, sl_level in reversed(swing_lows):
            window_start = max(sl_idx + 1, n - SWEEP_CONFIRMATION_BARS - 2)
            for j in range(window_start, n):
                if float(lows[j]) < sl_level * (1 - SWEEP_MIN_BREAK_PCT):
                    for k in range(j + 1, min(j + SWEEP_CONFIRMATION_BARS + 1, n)):
                        if float(closes[k]) > sl_level:
                            body = abs(float(closes[k]) - float(opens[k]))
                            rng  = float(highs[k]) - float(lows[k])
                            if rng > 0 and body / rng >= SWEEP_MIN_BODY_RATIO:
                                if _check_cooldown("SWEEP", "BULLISH", n, SWEEP_COOLDOWN_BARS):
                                    _set_cooldown("SWEEP", "BULLISH", n)
                                    return "BULLISH", float(closes[-1]), {"signal": "sweep_sellside"}
                    break
        return None, None, {"reason": "no_sweep"}
    except Exception:
        return None, None, {"reason": "sweep_error"}


def _signal_fvg_sweep_combo(df) -> tuple:
    try:
        sweep_dir, sweep_price, sweep_ctx = _signal_liquidity_sweep(df)
        if sweep_dir is None:
            return None, None, {"reason": "no_sweep_for_combo"}
        if len(df) < FVG_5M_AGG_MINUTES * 6:
            return None, None, {"reason": "insufficient_data"}
        close_col = _find_col(df, ["close", "Close"])
        high_col  = _find_col(df, ["high", "High"])
        low_col   = _find_col(df, ["low", "Low"])
        open_col  = _find_col(df, ["open", "Open"])
        vol_col   = _find_col(df, ["volume", "Volume", "vol"])
        if not all([close_col, high_col, low_col, open_col]):
            return None, None, {"reason": "missing_columns"}
        bars  = _aggregate_bars(df, close_col, high_col, low_col, open_col, vol_col, FVG_5M_AGG_MINUTES)
        zones = _detect_fvg_zones(bars, FVG_5M_MIN_GAP_PCT, FVG_5M_MAX_AGE_BARS)
        current_close = float(df[close_col].iloc[-1])
        atr = sum(float(df[high_col].iloc[i]) - float(df[low_col].iloc[i])
                  for i in range(-min(14, len(df)), 0)) / min(14, len(df))
        n = len(df)
        for zone in reversed(zones):
            if zone["direction"] != sweep_dir:
                continue
            zone_mid = (zone["zone_top"] + zone["zone_bottom"]) / 2
            in_zone = zone["zone_bottom"] <= current_close <= zone["zone_top"]
            near_zone = abs(current_close - zone_mid) <= (zone["zone_top"] - zone["zone_bottom"]) / 2 + atr * 0.5
            if in_zone or near_zone:
                if _check_cooldown("COMBO", sweep_dir, n, COMBO_COOLDOWN_BARS):
                    _set_cooldown("COMBO", sweep_dir, n)
                    return sweep_dir, current_close, {"signal": "fvg_sweep_combo"}
        return None, None, {"reason": "no_fvg_confluence"}
    except Exception:
        return None, None, {"reason": "combo_error"}


def _signal_flow_divergence(df) -> tuple:
    try:
        min_rows = max(FLOW_VOL_SMA_LOOKBACK, FLOW_MOMENTUM_LOOKBACK) + 5
        if len(df) < min_rows:
            return None, None, {"reason": "insufficient_data"}
        close_col = _find_col(df, ["close", "Close"])
        high_col  = _find_col(df, ["high", "High"])
        low_col   = _find_col(df, ["low", "Low"])
        vol_col   = _find_col(df, ["volume", "Volume", "vol"])
        vwap_col  = _find_col(df, ["vwap", "VWAP"])
        if not all([close_col, high_col, low_col]):
            return None, None, {"reason": "missing_columns"}
        closes = df[close_col].values
        highs  = df[high_col].values
        lows   = df[low_col].values
        n = len(df)
        current_close = float(closes[-1])
        # Get baseline signal direction (optional — enhances but no longer required)
        std_dir, _, _ = _signal_trend_pullback(df)
        # If no baseline signal, infer direction from EMA slope
        if std_dir is None:
            ema9_col = _find_col(df, ["ema9", "EMA9", "ema_9"])
            ema20_col = _find_col(df, ["ema20", "EMA20", "ema_20"])
            if ema9_col and ema20_col:
                ema9 = float(df[ema9_col].iloc[-1])
                ema20 = float(df[ema20_col].iloc[-1])
                std_dir = "BULLISH" if ema9 > ema20 else "BEARISH"
            else:
                # Fallback: use price momentum direction
                std_dir = "BULLISH" if current_close > float(closes[-11]) else "BEARISH"
        # VWAP slope
        vwap_slope_bullish = None
        if vwap_col:
            vwaps = df[vwap_col].values
            if len(vwaps) >= FLOW_VWAP_SLOPE_LOOKBACK + 1:
                vwap_slope_bullish = float(vwaps[-1]) > float(vwaps[-FLOW_VWAP_SLOPE_LOOKBACK - 1])
        # Volume spike
        vol_spike = False
        if vol_col:
            vols = df[vol_col].values
            vol_current = float(vols[-1])
            vol_sma = sum(float(v) for v in vols[-FLOW_VOL_SMA_LOOKBACK:]) / FLOW_VOL_SMA_LOOKBACK
            if vol_sma > 0:
                vol_spike = vol_current >= vol_sma * FLOW_VOL_SPIKE_THRESHOLD
        # Price momentum normalized by ATR
        price_momentum = float(closes[-1]) - float(closes[-FLOW_MOMENTUM_LOOKBACK - 1])
        atr_vals = [float(highs[i]) - float(lows[i]) for i in range(n - 14, n)]
        atr = sum(atr_vals) / len(atr_vals) if atr_vals else 1.0
        norm_momentum = price_momentum / atr if atr > 0 else 0.0
        momentum_bullish = norm_momentum > FLOW_MOMENTUM_MIN_ATR
        momentum_bearish = norm_momentum < -FLOW_MOMENTUM_MIN_ATR
        # Divergence: standard says BULLISH but flow says BEARISH
        if std_dir == "BULLISH" and momentum_bearish and vol_spike:
            vwap_ok = (vwap_slope_bullish is None) or (not vwap_slope_bullish)
            if vwap_ok and _check_cooldown("FLOW", "BEARISH", n, FLOW_COOLDOWN_BARS):
                _set_cooldown("FLOW", "BEARISH", n)
                return "BEARISH", current_close, {"signal": "flow_divergence", "override": "bearish"}
        # Divergence: standard says BEARISH but flow says BULLISH
        if std_dir == "BEARISH" and momentum_bullish and vol_spike:
            vwap_ok = (vwap_slope_bullish is None) or vwap_slope_bullish
            if vwap_ok and _check_cooldown("FLOW", "BULLISH", n, FLOW_COOLDOWN_BARS):
                _set_cooldown("FLOW", "BULLISH", n)
                return "BULLISH", current_close, {"signal": "flow_divergence", "override": "bullish"}
        return None, None, {"reason": "no_divergence"}
    except Exception:
        return None, None, {"reason": "flow_divergence_error"}


# ---------------------------------------------------------------------------
# Re-exports from sim_signal_funcs_smc2 (split to keep file under 500 lines)
# ---------------------------------------------------------------------------
from simulation.sim_signal_funcs_smc2 import (  # noqa: E402
    _signal_multi_tf_confirm,
    _signal_gap_fade,
    _signal_vpoc_reversion,
    _signal_opening_range_reclaim,
    _signal_vol_compression_breakout,
    _signal_vol_spike_fade,
)

__all__ = [
    "_check_cooldown", "_set_cooldown", "_aggregate_bars", "_detect_fvg_zones",
    "_signal_fvg_4h", "_signal_fvg_5m", "_signal_liquidity_sweep",
    "_signal_fvg_sweep_combo", "_signal_flow_divergence",
    "_signal_multi_tf_confirm", "_signal_gap_fade", "_signal_vpoc_reversion",
    "_signal_opening_range_reclaim", "_signal_vol_compression_breakout",
    "_signal_vol_spike_fade",
]
