import logging

from simulation.sim_signal_funcs import (
    _find_col, _ctx_float, _safe_float,
    _signal_mean_reversion, _signal_breakout, _signal_trend_pullback,
    _signal_orb_breakout, _signal_swing_trend, _signal_opportunity,
    _signal_vwap_reversion, _signal_zscore_bounce,
    _signal_failed_breakout_reversal, _signal_vwap_continuation,
    _signal_opening_drive, _signal_afternoon_breakout, _signal_trend_reclaim,
    _signal_extreme_extension_fade,
)
from simulation.sim_signal_funcs_smc import (
    _signal_fvg_4h, _signal_fvg_5m,
    _signal_liquidity_sweep, _signal_fvg_sweep_combo, _signal_flow_divergence,
    _signal_multi_tf_confirm, _signal_gap_fade, _signal_vpoc_reversion,
    _signal_opening_range_reclaim, _signal_vol_compression_breakout,
    _signal_vol_spike_fade,
)
from analytics.fair_value_gaps import detect_fvgs

# ---------------------------------------------------------------------------
# Signal mode registry
# ---------------------------------------------------------------------------

_KNOWN_SIGNAL_MODES = frozenset({
    "MEAN_REVERSION",
    "BREAKOUT",
    "TREND_PULLBACK",
    "SWING_TREND",
    "OPPORTUNITY",
    "ORB_BREAKOUT",
    "VWAP_REVERSION",
    "ZSCORE_BOUNCE",
    "FAILED_BREAKOUT_REVERSAL",
    "VWAP_CONTINUATION",
    "OPENING_DRIVE",
    "AFTERNOON_BREAKOUT",
    "TREND_RECLAIM",
    "EXTREME_EXTENSION_FADE",
    "FVG_4H",
    "FVG_5M",
    "LIQUIDITY_SWEEP",
    "FVG_SWEEP_COMBO",
    "FLOW_DIVERGENCE",
    "MULTI_TF_CONFIRM",
    "GAP_FADE",
    "VPOC_REVERSION",
    "OPENING_RANGE_RECLAIM",
    "VOL_COMPRESSION_BREAKOUT",
    "VOL_SPIKE_FADE",
    "STRUCTURE_FADE",
    "GEX_FLOW",
    "FVG_FILL",
})

_SIGNAL_MODE_FAMILY = {
    "MEAN_REVERSION":           "reversal",
    "BREAKOUT":                 "breakout",
    "TREND_PULLBACK":           "trend",
    "SWING_TREND":              "trend",
    "OPPORTUNITY":              "adaptive",
    "ORB_BREAKOUT":             "breakout",
    "VWAP_REVERSION":           "reversal",
    "ZSCORE_BOUNCE":            "reversal",
    "FAILED_BREAKOUT_REVERSAL": "reversal",
    "VWAP_CONTINUATION":        "trend",
    "OPENING_DRIVE":            "breakout",
    "AFTERNOON_BREAKOUT":       "breakout",
    "TREND_RECLAIM":            "reclaim",
    "EXTREME_EXTENSION_FADE":   "fade",
    "FVG_4H":                   "reversal",
    "FVG_5M":                   "reversal",
    "LIQUIDITY_SWEEP":          "reversal",
    "FVG_SWEEP_COMBO":          "reversal",
    "FLOW_DIVERGENCE":          "fade",
    "MULTI_TF_CONFIRM":         "trend",
    "GAP_FADE":                 "fade",
    "VPOC_REVERSION":           "reversal",
    "OPENING_RANGE_RECLAIM":    "structure",
    "VOL_COMPRESSION_BREAKOUT": "volatility",
    "VOL_SPIKE_FADE":           "volatility",
    "STRUCTURE_FADE":           "structure",
    "GEX_FLOW":                 "flow",
    "FVG_FILL":                 "reversal",
}


def get_signal_family(signal_mode: str) -> str:
    """Return the strategy family for a given signal mode."""
    return _SIGNAL_MODE_FAMILY.get(str(signal_mode).upper(), "unknown")


def is_known_signal_mode(signal_mode: str) -> bool:
    return str(signal_mode).upper() in _KNOWN_SIGNAL_MODES


def derive_sim_signal(
    df,
    signal_mode,
    context: dict | None = None,
    feature_snapshot: dict | None = None,
    profile: dict | None = None,
    signal_params: dict | None = None,
    structure_data: dict | None = None,
    options_data: dict | None = None,
):
    """
    Dispatch entry-signal generation by mode.

    Always returns a 3-tuple: (direction, price, context_dict).
    direction is "BULLISH", "BEARISH", or None.
    price is float or None.
    context_dict is a dict or None.

    signal_params: optional per-sim dict from YAML signal_params key.
    Merged (at lower priority) into context before dispatch so signal
    functions can read tuning constants via _ctx_float.
    """
    # Merge signal_params (lower priority) under context (higher priority)
    if signal_params:
        context = {**signal_params, **(context or {})}

    try:
        mode = str(signal_mode).upper()

        if mode == "MEAN_REVERSION":
            direction, price, ctx = _signal_mean_reversion(df, context)
            if direction is not None and isinstance(options_data, dict) and isinstance(ctx, dict):
                if options_data.get("gex_positive") is False:
                    ctx["gex_warning"] = "negative_gamma_fade_risk"
            return direction, price, ctx

        elif mode == "BREAKOUT":
            direction, price, ctx = _signal_breakout(df, context)
            if direction is None:
                return None, None, ctx
            vol_z_min = _ctx_float(context, "vol_z_min", None)
            if vol_z_min is not None:
                if feature_snapshot is None:
                    return None, None, {"reason": "features_required"}
                vol_z = feature_snapshot.get("vol_z")
                try:
                    if vol_z is None or float(vol_z) < float(vol_z_min):
                        return None, None, {"reason": "vol_z_filter"}
                except Exception:
                    return None, None, {"reason": "vol_z_invalid"}
            # Options-aware metadata for breakouts
            if isinstance(ctx, dict) and isinstance(options_data, dict):
                if options_data.get("in_low_gamma_zone"):
                    ctx["gamma_boost"] = True
                if direction == "BULLISH":
                    _cwd = options_data.get("call_wall_distance_pct")
                    if _cwd is not None and _cwd < 0.003:
                        ctx["approaching_call_wall"] = True
                if direction == "BEARISH":
                    _pwd = options_data.get("put_wall_distance_pct")
                    if _pwd is not None and _pwd < 0.003:
                        ctx["approaching_put_wall"] = True
            return direction, price, ctx

        elif mode == "TREND_PULLBACK":
            direction, price, ctx = _signal_trend_pullback(df, context)
            if direction is None:
                return None, None, ctx
            min_exp = _ctx_float(context, "atr_expansion_min", None)
            if min_exp is not None:
                if feature_snapshot is None:
                    return None, None, {"reason": "features_required"}
                atr_exp = feature_snapshot.get("atr_expansion")
                try:
                    if atr_exp is None or float(atr_exp) < min_exp:
                        return None, None, {"reason": "atr_expansion_filter"}
                except Exception:
                    return None, None, {"reason": "atr_expansion_invalid"}
            vol_z_min = _ctx_float(context, "vol_z_min", None)
            if vol_z_min is not None:
                if feature_snapshot is None:
                    return None, None, {"reason": "features_required"}
                vol_z = feature_snapshot.get("vol_z")
                try:
                    if vol_z is None or float(vol_z) < vol_z_min:
                        return None, None, {"reason": "vol_z_filter"}
                except Exception:
                    return None, None, {"reason": "vol_z_invalid"}
            iv_rank_max = _ctx_float(context, "iv_rank_max", None)
            if iv_rank_max is not None:
                if feature_snapshot is None:
                    return None, None, {"reason": "features_required"}
                iv_rank = feature_snapshot.get("iv_rank_proxy")
                try:
                    if iv_rank is None or float(iv_rank) > iv_rank_max:
                        return None, None, {"reason": "iv_rank_filter"}
                except Exception:
                    return None, None, {"reason": "iv_rank_invalid"}
            # Structure-aware suppression (optional)
            if direction == "BULLISH" and isinstance(structure_data, dict):
                _dist_r = structure_data.get("distance_to_resistance_pct")
                if _dist_r is not None and _dist_r < 0.0015:
                    logging.debug("trend_pullback_suppressed_near_resistance: dist=%.4f", _dist_r)
                    return None, None, {"reason": "suppressed_near_resistance", "distance_to_resistance_pct": _dist_r}
            if direction == "BEARISH" and isinstance(structure_data, dict):
                _dist_s = structure_data.get("distance_to_support_pct")
                if _dist_s is not None and _dist_s < 0.0015:
                    logging.debug("trend_pullback_suppressed_near_support: dist=%.4f", _dist_s)
                    return None, None, {"reason": "suppressed_near_support", "distance_to_support_pct": _dist_s}
            return direction, price, ctx

        elif mode == "SWING_TREND":
            return _signal_swing_trend(df)

        elif mode == "OPPORTUNITY":
            return _signal_opportunity(df, context)

        elif mode == "ORB_BREAKOUT":
            if feature_snapshot is None:
                return None, None, {"reason": "features_required"}
            direction, price, reason = _signal_orb_breakout(feature_snapshot, context or {})
            if direction is None:
                return None, None, {"reason": reason or "orb_no_signal"}
            return direction, price, {"reason": reason or "orb_break"}

        elif mode == "VWAP_REVERSION":
            return _signal_vwap_reversion(df, context, feature_snapshot)

        elif mode == "ZSCORE_BOUNCE":
            return _signal_zscore_bounce(df, context, feature_snapshot)

        elif mode == "FAILED_BREAKOUT_REVERSAL":
            return _signal_failed_breakout_reversal(df, context)

        elif mode == "VWAP_CONTINUATION":
            return _signal_vwap_continuation(df, profile)

        elif mode == "OPENING_DRIVE":
            return _signal_opening_drive(df, profile)

        elif mode == "AFTERNOON_BREAKOUT":
            return _signal_afternoon_breakout(df, profile)

        elif mode == "TREND_RECLAIM":
            return _signal_trend_reclaim(df, profile)

        elif mode == "EXTREME_EXTENSION_FADE":
            return _signal_extreme_extension_fade(df, context, profile, feature_snapshot)

        elif mode == "FVG_4H":
            return _signal_fvg_4h(df)

        elif mode == "FVG_5M":
            return _signal_fvg_5m(df)

        elif mode == "LIQUIDITY_SWEEP":
            return _signal_liquidity_sweep(df)

        elif mode == "FVG_SWEEP_COMBO":
            return _signal_fvg_sweep_combo(df)

        elif mode == "FLOW_DIVERGENCE":
            return _signal_flow_divergence(df)

        elif mode == "MULTI_TF_CONFIRM":
            return _signal_multi_tf_confirm(df)

        elif mode == "GAP_FADE":
            return _signal_gap_fade(df)

        elif mode == "VPOC_REVERSION":
            return _signal_vpoc_reversion(df)

        elif mode == "OPENING_RANGE_RECLAIM":
            return _signal_opening_range_reclaim(df)

        elif mode == "VOL_COMPRESSION_BREAKOUT":
            return _signal_vol_compression_breakout(df)

        elif mode == "VOL_SPIKE_FADE":
            return _signal_vol_spike_fade(df)

        elif mode == "STRUCTURE_FADE":
            return _signal_structure_fade(df, structure_data, options_data)

        elif mode == "GEX_FLOW":
            return _signal_gex_flow(df, structure_data, options_data)

        elif mode == "FVG_FILL":
            return _signal_fvg_fill(df, context, feature_snapshot)

        else:
            return None, None, {"reason": f"unknown_signal_mode:{signal_mode}"}

    except Exception:
        return None, None, {"reason": "dispatch_error"}


def _compute_structure_levels(df, close_col, high_col, low_col, lookback=100):
    """Derive support/resistance from swing highs/lows when external structure_data unavailable."""
    highs = df[high_col].values
    lows = df[low_col].values
    close = float(df.iloc[-1][close_col])
    n = len(df)
    start = max(3, n - lookback)
    swing_highs, swing_lows = [], []
    for i in range(start, n - 3):
        if highs[i] > max(highs[i-3:i]) and highs[i] > max(highs[i+1:i+4]):
            swing_highs.append(float(highs[i]))
        if lows[i] < min(lows[i-3:i]) and lows[i] < min(lows[i+1:i+4]):
            swing_lows.append(float(lows[i]))
    nearest_support = max((s for s in swing_lows if s < close), default=None)
    nearest_resistance = min((r for r in swing_highs if r > close), default=None)
    dist_sup = (close - nearest_support) / close if nearest_support else None
    dist_res = (nearest_resistance - close) / close if nearest_resistance else None
    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "distance_to_support_pct": dist_sup,
        "distance_to_resistance_pct": dist_res,
    }


def _signal_structure_fade(df, structure_data, options_data) -> tuple:
    """Mean-reversion trade off confirmed structure levels (support/resistance bounce)."""
    try:
        close_col = _find_col(df, ["close", "Close"])
        high_col = _find_col(df, ["high", "High"])
        low_col = _find_col(df, ["low", "Low"])
        rsi_col = _find_col(df, ["rsi", "RSI", "rsi14"])
        if close_col is None or rsi_col is None or df is None or len(df) < 20:
            return None, None, {"reason": "missing_data"}

        close = float(df.iloc[-1][close_col])
        rsi = float(df.iloc[-1][rsi_col])

        # Use provided structure_data or compute from price bars
        if not isinstance(structure_data, dict):
            if high_col is None or low_col is None:
                return None, None, {"reason": "no_structure_data"}
            structure_data = _compute_structure_levels(df, close_col, high_col, low_col)

        nearest_support = structure_data.get("nearest_support")
        nearest_resistance = structure_data.get("nearest_resistance")
        dist_to_support = structure_data.get("distance_to_support_pct")
        dist_to_resistance = structure_data.get("distance_to_resistance_pct")

        # BULLISH: price near support, RSI < 40
        if nearest_support and dist_to_support is not None and dist_to_support < 0.003:
            if rsi < 40:
                return "BULLISH", close, {
                    "reason": "structure_fade_support",
                    "entry_context": f"structure_fade at {nearest_support:.2f}",
                    "level_type": "support",
                    "level_price": nearest_support,
                    "rsi": rsi,
                }

        # BEARISH: price near resistance, RSI > 60
        if nearest_resistance and dist_to_resistance is not None and dist_to_resistance < 0.003:
            if rsi > 60:
                return "BEARISH", close, {
                    "reason": "structure_fade_resistance",
                    "entry_context": f"structure_fade at {nearest_resistance:.2f}",
                    "level_type": "resistance",
                    "level_price": nearest_resistance,
                    "rsi": rsi,
                }

        return None, None, {"reason": "no_structure_fade"}
    except Exception:
        return None, None, {"reason": "structure_fade_error"}


def _signal_gex_flow(df, structure_data, options_data) -> tuple:
    """Trade in the direction of mechanical dealer gamma flows."""
    try:
        if not isinstance(options_data, dict) or not options_data.get("options_data_available"):
            return None, None, {"reason": "no_options_data"}

        close_col = _find_col(df, ["close", "Close"])
        if close_col is None or df is None or len(df) < 2:
            return None, None, {"reason": "missing_data"}

        close = float(df.iloc[-1][close_col])

        gex_positive = options_data.get("gex_positive")
        gex_flip = options_data.get("gex_flip_strike")
        max_pain = options_data.get("max_pain_strike")
        nearest_call_wall = options_data.get("nearest_call_wall")
        nearest_put_wall = options_data.get("nearest_put_wall")

        if gex_positive is None or gex_flip is None:
            return None, None, {"reason": "incomplete_gex_data"}

        # BULLISH: positive gamma + above flip + between max_pain and call wall
        # (price being pulled toward call wall by delta hedging)
        if gex_positive is True and close > gex_flip:
            if max_pain is not None and nearest_call_wall is not None:
                if max_pain <= close <= nearest_call_wall:
                    return "BULLISH", close, {
                        "reason": "gex_flow_bullish",
                        "entry_context": f"gex_flow: positive gamma, target call wall {nearest_call_wall:.0f}",
                        "gex_positive": True,
                        "gex_flip_strike": gex_flip,
                        "target_wall": nearest_call_wall,
                    }

        # BEARISH: negative gamma + below flip + between put wall and max_pain
        # (price being pushed down by dealer selling)
        if gex_positive is False and close < gex_flip:
            if max_pain is not None and nearest_put_wall is not None:
                if nearest_put_wall <= close <= max_pain:
                    return "BEARISH", close, {
                        "reason": "gex_flow_bearish",
                        "entry_context": f"gex_flow: negative gamma, target put wall {nearest_put_wall:.0f}",
                        "gex_positive": False,
                        "gex_flip_strike": gex_flip,
                        "target_wall": nearest_put_wall,
                    }

        return None, None, {"reason": "no_gex_flow_setup"}
    except Exception:
        return None, None, {"reason": "gex_flow_error"}


def _signal_fvg_fill(df, context, feature_snapshot) -> tuple:
    """Trade into unmitigated FVG zones expecting price to fill the gap."""
    try:
        close_col = _find_col(df, ["close", "Close"])
        if close_col is None or df is None or len(df) < 5:
            return None, None, {"reason": "fvg_no_close"}

        close = float(df.iloc[-1][close_col])

        gaps = detect_fvgs(df, max_gaps=20)
        if not gaps:
            return None, None, {"reason": "fvg_no_active_gap"}

        # Filter by age
        fvg_max_age = int(_ctx_float(context, "fvg_max_age_bars", 50) or 50)
        last_idx = len(df) - 1
        gaps = [g for g in gaps if not g.get("mitigated") and (last_idx - g["bar_idx"]) <= fvg_max_age]

        if not gaps:
            return None, None, {"reason": "fvg_no_active_gap"}

        # Optional RSI filter
        rsi_col = _find_col(df, ["rsi", "RSI", "rsi14"])
        rsi = None
        if rsi_col is not None:
            try:
                rsi = float(df.iloc[-1][rsi_col])
            except Exception:
                rsi = None

        # Check bull FVGs (close inside gap zone → expect fill upward)
        for gap in gaps:
            if gap["type"] != "bull":
                continue
            if gap["bottom"] <= close <= gap["top"]:
                if rsi is not None and rsi > 70:
                    continue
                return "BULLISH", close, {
                    "reason": "fvg_fill_long",
                    "gap_top": gap["top"],
                    "gap_bottom": gap["bottom"],
                    "gap_age": last_idx - gap["bar_idx"],
                }

        # Check bear FVGs (close inside gap zone → expect fill downward)
        for gap in gaps:
            if gap["type"] != "bear":
                continue
            if gap["bottom"] <= close <= gap["top"]:
                if rsi is not None and rsi < 30:
                    continue
                return "BEARISH", close, {
                    "reason": "fvg_fill_short",
                    "gap_top": gap["top"],
                    "gap_bottom": gap["bottom"],
                    "gap_age": last_idx - gap["bar_idx"],
                }

        return None, None, {"reason": "fvg_no_active_gap"}
    except Exception:
        return None, None, {"reason": "fvg_fill_error"}


def derive_opportunity_signal(df, sim_states, regime, trader_signal=None):
    """
    OPPORTUNITY mode signal dispatcher.
    Uses OpportunityRanker to evaluate all signal candidates and pick the best.

    Returns (direction, underlying_price, signal_meta) — standard 3-tuple.
    direction is "BULLISH", "BEARISH", or None.
    signal_meta is a dict with ranker context, or None.
    """
    from simulation.sim_opportunity_ranker import OpportunityRanker
    ranker = OpportunityRanker()
    result = ranker.rank_opportunities(df, sim_states, regime, trader_signal=trader_signal)
    if result is None:
        return None, None, None
    signal_meta = {
        "winning_mode": result.signal_mode,
        "composite_score": result.composite_score,
        "breakdown": result.breakdown,
        "competing_candidates": result.competing_candidates,
        "recommended_dte_min": result.recommended_dte_min,
        "recommended_dte_max": result.recommended_dte_max,
        "recommended_hold_max_minutes": result.recommended_hold_max_minutes,
    }
    return result.direction, result.underlying_price, signal_meta


def _signal_opportunity(df, context=None):
    """
    Legacy stub for OPPORTUNITY mode.
    The live path now uses derive_opportunity_signal() via sim_engine.
    Kept for backward compatibility.
    """
    from simulation.sim_signal_funcs import _signal_opportunity as _old_opp
    return _old_opp(df, context)
