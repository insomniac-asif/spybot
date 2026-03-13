import logging
from signals.opportunity import evaluate_opportunity
from signals.volatility import volatility_state
from signals.predictor import make_prediction

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_col(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ctx_float(context, key, default):
    if not isinstance(context, dict):
        return default
    val = context.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Existing signal functions (preserved, updated to 3-tuple returns)
# ---------------------------------------------------------------------------

def _signal_mean_reversion(df, context=None) -> tuple:
    RSI_OVERSOLD = _ctx_float(context, "rsi_oversold", 30)
    RSI_OVERBOUGHT = _ctx_float(context, "rsi_overbought", 70)
    MIN_BARS_REQUIRED = 2
    try:
        if df is None or len(df) < MIN_BARS_REQUIRED:
            return None, None, {"reason": "insufficient_bars"}
        close_col = _find_col(df, ["close", "Close"])
        rsi_col = _find_col(df, ["rsi", "RSI", "rsi14", "RSI14"])
        vwap_col = _find_col(df, ["vwap", "VWAP"])
        if close_col is None or rsi_col is None or vwap_col is None:
            return None, None, {"reason": "missing_cols"}
        last = df.iloc[-1]
        close = float(last[close_col])
        rsi = float(last[rsi_col])
        vwap = float(last[vwap_col])
        if rsi < RSI_OVERSOLD and close < vwap:
            return "BULLISH", close, {"reason": "mean_reversion_long", "rsi": rsi}
        if rsi > RSI_OVERBOUGHT and close > vwap:
            return "BEARISH", close, {"reason": "mean_reversion_short", "rsi": rsi}
        return None, None, {"reason": "no_mean_reversion"}
    except Exception:
        return None, None, {"reason": "mean_reversion_error"}


def _signal_breakout(df, context=None) -> tuple:
    BREAKOUT_LOOKBACK = int(_ctx_float(context, "breakout_lookback", 20))
    try:
        if df is None or len(df) < BREAKOUT_LOOKBACK + 1:
            return None, None, {"reason": "insufficient_bars"}
        close_col = _find_col(df, ["close", "Close"])
        high_col = _find_col(df, ["high", "High"])
        low_col = _find_col(df, ["low", "Low"])
        if close_col is None or high_col is None or low_col is None:
            return None, None, {"reason": "missing_cols"}
        close = float(df.iloc[-1][close_col])
        highs = df[high_col].iloc[-(BREAKOUT_LOOKBACK + 1):-1].dropna()
        lows = df[low_col].iloc[-(BREAKOUT_LOOKBACK + 1):-1].dropna()
        if len(highs) < 1 or len(lows) < 1:
            return None, None, {"reason": "insufficient_data"}
        recent_high = max(highs)
        recent_low = min(lows)
        if close > recent_high:
            return "BULLISH", close, {"reason": "breakout_high", "ref_high": recent_high}
        if close < recent_low:
            return "BEARISH", close, {"reason": "breakout_low", "ref_low": recent_low}
        return None, None, {"reason": "no_breakout"}
    except Exception:
        return None, None, {"reason": "breakout_error"}


def _signal_trend_pullback(df, context=None) -> tuple:
    PULLBACK_TOLERANCE = _ctx_float(context, "pullback_tolerance", 0.001)
    MIN_BARS_REQUIRED = 2
    try:
        if df is None or len(df) < MIN_BARS_REQUIRED:
            return None, None, {"reason": "insufficient_bars"}
        close_col = _find_col(df, ["close", "Close"])
        ema9_col = _find_col(df, ["ema9", "EMA9", "ema_9"])
        ema20_col = _find_col(df, ["ema20", "EMA20", "ema_20"])
        if close_col is None or ema9_col is None or ema20_col is None:
            return None, None, {"reason": "missing_cols"}
        last = df.iloc[-1]
        close = float(last[close_col])
        ema9 = float(last[ema9_col])
        ema20 = float(last[ema20_col])
        if ema9 > ema20:
            if close <= ema9 * (1 + PULLBACK_TOLERANCE) and close >= ema9 * (1 - PULLBACK_TOLERANCE):
                return "BULLISH", close, {"reason": "trend_pullback_bull"}
        if ema9 < ema20:
            if close >= ema9 * (1 - PULLBACK_TOLERANCE) and close <= ema9 * (1 + PULLBACK_TOLERANCE):
                return "BEARISH", close, {"reason": "trend_pullback_bear"}
        return None, None, {"reason": "no_pullback"}
    except Exception:
        return None, None, {"reason": "trend_pullback_error"}


def _signal_orb_breakout(feature_snapshot: dict, profile: dict) -> tuple:
    try:
        close = feature_snapshot.get("close")
        orb_high = feature_snapshot.get("orb_high")
        orb_low = feature_snapshot.get("orb_low")
        vol_z = feature_snapshot.get("vol_z")
        ema_spread = feature_snapshot.get("ema_spread")
        if close is None or orb_high is None or orb_low is None:
            return None, None, "orb_unavailable"
        vol_z_min = float(profile.get("vol_z_min", 0.0))
        if vol_z is None or float(vol_z) < vol_z_min:
            return None, None, "vol_z_filter"
        require_trend_bias = bool(profile.get("require_trend_bias", False))
        if close > orb_high:
            if require_trend_bias and (ema_spread is None or float(ema_spread) <= 0.0):
                return None, None, "trend_bias_fail"
            return "BULLISH", float(close), "orb_break_high"
        if close < orb_low:
            if require_trend_bias and (ema_spread is None or float(ema_spread) >= 0.0):
                return None, None, "trend_bias_fail"
            return "BEARISH", float(close), "orb_break_low"
        return None, None, "no_orb_break"
    except Exception:
        return None, None, "orb_error"


def _signal_swing_trend(df) -> tuple:
    SWING_SLOPE_LOOKBACK = 10
    try:
        if df is None or len(df) < SWING_SLOPE_LOOKBACK + 1:
            return None, None, {"reason": "insufficient_bars"}
        close_col = _find_col(df, ["close", "Close"])
        ema20_col = _find_col(df, ["ema20", "EMA20", "ema_20"])
        if close_col is None or ema20_col is None:
            return None, None, {"reason": "missing_cols"}
        last = df.iloc[-1]
        close = float(last[close_col])
        ema20_now = float(last[ema20_col])
        ema20_past = float(df[ema20_col].iloc[-(SWING_SLOPE_LOOKBACK + 1)])
        ema20_slope_positive = ema20_now > ema20_past
        if ema20_slope_positive and close > ema20_now:
            return "BULLISH", close, {"reason": "swing_trend_bull"}
        if (not ema20_slope_positive) and close < ema20_now:
            return "BEARISH", close, {"reason": "swing_trend_bear"}
        return None, None, {"reason": "no_swing_trend"}
    except Exception:
        return None, None, {"reason": "swing_trend_error"}


def _pick_opportunity_horizon(conviction_score, vol_state):
    try:
        score = float(conviction_score) if conviction_score is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    if score >= 6:
        return "WEEKLY"
    if score <= 3:
        return "DAYTRADE"
    if vol_state == "HIGH":
        return "DAYTRADE"
    if vol_state == "LOW":
        return "WEEKLY"
    return "SWING"


def _signal_opportunity(df, context: dict | None = None):
    try:
        result = evaluate_opportunity(df)
        if not result:
            return None, None, None
        side = result[0]
        entry_low = result[1]
        entry_high = result[2]
        price = result[3]
        conviction_score = result[4]
        tp_low = result[5] if len(result) > 5 else None
        tp_high = result[6] if len(result) > 6 else None
        stop_loss = result[7] if len(result) > 7 else None

        direction = "BULLISH" if str(side).upper() in {"CALL", "CALLS"} else "BEARISH"
        underlying_price = float(price)
        vol_state = volatility_state(df)
        horizon_type = _pick_opportunity_horizon(conviction_score, vol_state)

        trade_count = 0
        if isinstance(context, dict):
            try:
                trade_count = int(context.get("trade_count", 0))
            except (TypeError, ValueError):
                trade_count = 0
        optimize_ready = trade_count >= 50

        pred_minutes_map = {
            "DAYTRADE": 30,
            "SWING": 120,
            "WEEKLY": 390,
        }
        pred_minutes = pred_minutes_map.get(horizon_type, 60)
        pred = make_prediction(pred_minutes, df)
        pred_dir = pred.get("direction") if isinstance(pred, dict) else None
        pred_conf = pred.get("confidence") if isinstance(pred, dict) else None
        pred_dir_up = pred_dir.upper() if isinstance(pred_dir, str) else None

        # Predictor veto: respect predictor_mode from config
        _pred_mode = "veto_only"
        try:
            import yaml, os
            _cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation", "sim_config.yaml")
            with open(_cfg_path, "r") as _f:
                _raw = yaml.safe_load(_f) or {}
            _pred_mode = str((_raw.get("_global") or {}).get("predictor_mode", "veto_only")).lower()
        except Exception:
            pass

        if _pred_mode != "disabled" and optimize_ready:
            # Veto if predictor opposes signal direction at >70% confidence
            if pred_dir_up in {"BULLISH", "BEARISH"} and pred_conf is not None:
                try:
                    if float(pred_conf) > 0.70:
                        if (direction == "BULLISH" and pred_dir_up == "BEARISH") or (
                            direction == "BEARISH" and pred_dir_up == "BULLISH"
                        ):
                            return None, None, None
                except (TypeError, ValueError):
                    pass
            # RANGE predictions ignored — let the signal through

        horizon_map = {
            "DAYTRADE": {
                "dte_min": 0,
                "dte_max": 0,
                "hold_min_seconds": 300,
                "hold_max_seconds": 3600,
                "horizon": "scalp",
                "cutoff_time_et": "15:30",
            },
            "SWING": {
                "dte_min": 1,
                "dte_max": 5,
                "hold_min_seconds": 1800,
                "hold_max_seconds": 86400,
                "horizon": "intraday",
                "cutoff_time_et": "15:45",
            },
            "WEEKLY": {
                "dte_min": 7,
                "dte_max": 21,
                "hold_min_seconds": 3600,
                "hold_max_seconds": 604800,
                "horizon": "swing",
                "cutoff_time_et": "15:45",
            },
        }
        meta = dict(horizon_map.get(horizon_type, {}))
        if optimize_ready:
            try:
                meta["hold_max_seconds"] = max(int(meta["hold_min_seconds"]), int(pred_minutes * 60))
            except Exception:
                pass
        entry_ctx = []
        try:
            entry_ctx.append(f"opp_entry={float(entry_low):.2f}-{float(entry_high):.2f}")
        except Exception:
            pass
        if tp_low is not None and tp_high is not None:
            try:
                entry_ctx.append(f"tp={float(tp_low):.2f}-{float(tp_high):.2f}")
            except Exception:
                pass
        if stop_loss is not None:
            try:
                entry_ctx.append(f"sl={float(stop_loss):.2f}")
            except Exception:
                pass
        if horizon_type:
            entry_ctx.append(f"horizon={horizon_type}")
        if pred_dir_up:
            entry_ctx.append(f"pred={pred_dir_up}@{pred_minutes}m")
        if pred_conf is not None:
            try:
                entry_ctx.append(f"pred_conf={float(pred_conf):.2f}")
            except (TypeError, ValueError):
                pass
        entry_ctx.append(f"opt_ready={int(optimize_ready)}")
        entry_ctx.append(f"cutoff={meta.get('cutoff_time_et', 'N/A')}")
        if entry_ctx:
            meta["entry_context"] = " | ".join(entry_ctx)
        meta["opportunity_type"] = horizon_type
        meta["take_profit_low"] = tp_low
        meta["take_profit_high"] = tp_high
        meta["stop_loss"] = stop_loss
        meta["horizon_type"] = horizon_type
        meta["predicted_direction"] = pred_dir_up
        meta["prediction_confidence"] = pred_conf
        meta["prediction_timeframe"] = pred_minutes
        meta["optimize_ready"] = optimize_ready

        return direction, underlying_price, meta
    except Exception:
        return None, None, None


def _signal_vwap_reversion(df, context: dict | None = None, feature_snapshot: dict | None = None) -> tuple:
    try:
        if feature_snapshot is None:
            return None, None, {"reason": "features_required"}
        close = feature_snapshot.get("close")
        vwap_z = feature_snapshot.get("vwap_z")
        rsi = feature_snapshot.get("rsi")
        if close is None or vwap_z is None:
            return None, None, {"reason": "vwap_z_unavailable"}
        vwap_z_min = _ctx_float(context, "vwap_z_min", 2.0)
        vwap_z_val = float(vwap_z)
        if vwap_z_val <= -vwap_z_min:
            if rsi is not None:
                try:
                    if float(rsi) > 60:
                        return None, None, {"reason": "rsi_conflict"}
                except (TypeError, ValueError):
                    pass
            return "BULLISH", float(close), {
                "reason": "vwap_reversion_long",
                "vwap_z": vwap_z_val,
                "entry_context": f"vwap_z={vwap_z_val:.2f} threshold={vwap_z_min}",
            }
        if vwap_z_val >= vwap_z_min:
            if rsi is not None:
                try:
                    if float(rsi) < 40:
                        return None, None, {"reason": "rsi_conflict"}
                except (TypeError, ValueError):
                    pass
            return "BEARISH", float(close), {
                "reason": "vwap_reversion_short",
                "vwap_z": vwap_z_val,
                "entry_context": f"vwap_z={vwap_z_val:.2f} threshold={vwap_z_min}",
            }
        return None, None, {"reason": "vwap_z_below_threshold"}
    except Exception:
        return None, None, {"reason": "vwap_reversion_error"}


def _signal_zscore_bounce(df, context: dict | None = None, feature_snapshot: dict | None = None) -> tuple:
    try:
        if feature_snapshot is None:
            return None, None, {"reason": "features_required"}
        close = feature_snapshot.get("close")
        close_z = feature_snapshot.get("close_z")
        rsi = feature_snapshot.get("rsi")
        ema_spread = feature_snapshot.get("ema_spread")
        if close is None or close_z is None:
            return None, None, {"reason": "close_z_unavailable"}
        close_z_min = _ctx_float(context, "close_z_min", 2.0)
        close_z_val = float(close_z)
        if close_z_val <= -close_z_min:
            if rsi is not None:
                try:
                    if float(rsi) > 55:
                        return None, None, {"reason": "rsi_divergence"}
                except (TypeError, ValueError):
                    pass
            if ema_spread is not None:
                try:
                    if float(ema_spread) < -0.003:
                        return None, None, {"reason": "strong_downtrend"}
                except (TypeError, ValueError):
                    pass
            return "BULLISH", float(close), {
                "reason": "zscore_bounce_long",
                "close_z": close_z_val,
                "entry_context": f"close_z={close_z_val:.2f} threshold={close_z_min}",
            }
        if close_z_val >= close_z_min:
            if rsi is not None:
                try:
                    if float(rsi) < 45:
                        return None, None, {"reason": "rsi_divergence"}
                except (TypeError, ValueError):
                    pass
            if ema_spread is not None:
                try:
                    if float(ema_spread) > 0.003:
                        return None, None, {"reason": "strong_uptrend"}
                except (TypeError, ValueError):
                    pass
            return "BEARISH", float(close), {
                "reason": "zscore_bounce_short",
                "close_z": close_z_val,
                "entry_context": f"close_z={close_z_val:.2f} threshold={close_z_min}",
            }
        return None, None, {"reason": "close_z_below_threshold"}
    except Exception:
        return None, None, {"reason": "zscore_bounce_error"}


# ---------------------------------------------------------------------------
# Re-exports from sim_signal_funcs2 (split to keep file under 500 lines)
# ---------------------------------------------------------------------------
from simulation.sim_signal_funcs2 import (  # noqa: E402
    _signal_failed_breakout_reversal,
    _signal_vwap_continuation,
    _signal_opening_drive,
    _signal_afternoon_breakout,
    _signal_trend_reclaim,
    _signal_extreme_extension_fade,
)

__all__ = [
    "_find_col", "_ctx_float", "_safe_float",
    "_signal_mean_reversion", "_signal_breakout", "_signal_trend_pullback",
    "_signal_orb_breakout", "_signal_swing_trend", "_signal_opportunity",
    "_signal_vwap_reversion", "_signal_zscore_bounce",
    "_signal_failed_breakout_reversal", "_signal_vwap_continuation",
    "_signal_opening_drive", "_signal_afternoon_breakout",
    "_signal_trend_reclaim", "_signal_extreme_extension_fade",
]
