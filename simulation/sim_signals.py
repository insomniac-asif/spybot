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
            return _signal_mean_reversion(df, context)

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

        else:
            return None, None, {"reason": f"unknown_signal_mode:{signal_mode}"}

    except Exception:
        return None, None, {"reason": "dispatch_error"}


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
