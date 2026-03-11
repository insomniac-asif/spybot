# signals/predictor.py

import json
import math
import os
from datetime import datetime

import pytz

from core.data_service import get_market_dataframe
from signals.regime import get_regime
from signals.session_classifier import classify_session

# ── learned-weight loader (file-stat cache) ───────────────────────────────────
_WEIGHTS_FILE  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictor_weights.json")
_weights_cache: dict = {}
_weights_mtime: float = 0.0


def _load_weights() -> dict:
    """Return cached predictor weights, reloading when the file changes."""
    global _weights_cache, _weights_mtime
    try:
        mtime = os.path.getmtime(_WEIGHTS_FILE)
        if mtime != _weights_mtime:
            with open(_WEIGHTS_FILE) as f:
                _weights_cache = json.load(f)
            _weights_mtime = mtime
    except Exception:
        pass
    return _weights_cache


def _apply_biases(
    bullish_score: float,
    bearish_score: float,
    range_score: float,
    regime: str,
    session: str,
    volatility: str,
    weights: dict,
) -> tuple[float, float, float]:
    """
    Add learned bias corrections to pre-softmax scores.

    Priority order (most → least specific):
      1. regime × session combo  (if available, replaces regime + session)
      2. regime + session separately
      3. volatility (always added at half weight as a supplementary signal)
    """
    combo_key = f"{regime}_{session}"
    combo     = weights.get("regime_session", {}).get(combo_key)

    if combo:
        # Use the more-specific combined bias
        bullish_score += combo.get("bullish", 0.0)
        bearish_score += combo.get("bearish", 0.0)
        range_score   += combo.get("range",   0.0)
    else:
        # Fall back to independent regime + session biases
        r_bias = weights.get("regime",  {}).get(regime,  {})
        s_bias = weights.get("session", {}).get(session, {})
        bullish_score += r_bias.get("bullish", 0.0) + s_bias.get("bullish", 0.0)
        bearish_score += r_bias.get("bearish", 0.0) + s_bias.get("bearish", 0.0)
        range_score   += r_bias.get("range",   0.0) + s_bias.get("range",   0.0)

    # Volatility bias always layered on top (half weight — supplementary)
    v_bias = weights.get("volatility", {}).get(volatility, {})
    bullish_score += v_bias.get("bullish", 0.0) * 0.5
    bearish_score += v_bias.get("bearish", 0.0) * 0.5
    range_score   += v_bias.get("range",   0.0) * 0.5

    return bullish_score, bearish_score, range_score


def _detect_volatility(df) -> str:
    """Lightweight volatility bucket from last 30 bars."""
    try:
        recent = df.tail(30)
        vol = float(recent["high"].max()) - float(recent["low"].min())
        if vol < 0.35:   return "DEAD"
        if vol < 0.75:   return "LOW"
        if vol < 1.50:   return "NORMAL"
        return "HIGH"
    except Exception:
        return "NORMAL"


def make_prediction(minutes=60, df=None):

    if df is None:
        df = get_market_dataframe()

    if df is None or len(df) < 30:
        return None

    recent = df.tail(30)

    last = recent.iloc[-1]

    # FIX: defensive numeric parsing to avoid crashing the forecast loop
    try:
        price = float(last["close"])
    except Exception:
        return None

    vwap = last.get("vwap", price)
    try:
        vwap = float(vwap)
    except Exception:
        vwap = price
    if not math.isfinite(vwap):
        vwap = price

    try:
        high30 = float(recent["high"].max())
        low30 = float(recent["low"].min())
    except Exception:
        return None

    try:
        trend = float(recent["close"].iloc[-1]) - float(recent["close"].iloc[0])
    except Exception:
        return None

    vol = (high30 - low30) / price if price != 0 else 0

    # ── Additional indicators ──────────────────────────────────────────────────
    # RSI-14 on close prices
    closes = recent["close"].astype(float)
    deltas = closes.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = (-deltas.clip(upper=0))
    avg_gain = gains.rolling(14, min_periods=14).mean().iloc[-1] if len(deltas) >= 14 else 0.0
    avg_loss = losses.rolling(14, min_periods=14).mean().iloc[-1] if len(deltas) >= 14 else 0.0
    rsi = 50.0
    if avg_gain + avg_loss > 0:
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # ── Evidence scoring ─────────────────────────────────────────────────────
    # Goal: maximize directional accuracy when we DO predict direction,
    # and use "range" as the honest "no edge" bucket.
    bullish_score = 0.15
    bearish_score = 0.15
    range_score = 0.15
    reasons = []

    vwap_dist = (price - vwap) / price if price else 0
    loc_strength = min(abs(vwap_dist) / 0.004, 1.0)
    trend_strength = min(abs(trend) / price / 0.004, 1.0) if price else 0.0

    # VWAP location
    if price > vwap:
        bullish_score += 0.9 * loc_strength
        reasons.append("Price above VWAP")
    else:
        bearish_score += 0.9 * loc_strength
        reasons.append("Price below VWAP")

    # 30-bar trend
    if trend > 0:
        bullish_score += 0.9 * trend_strength
        reasons.append("Upward momentum")
    elif trend < 0:
        bearish_score += 0.9 * trend_strength
        reasons.append("Downward momentum")
    else:
        range_score += 0.15
        reasons.append("Flat momentum")

    # RSI — mean-reversion at extremes overrides trend
    if rsi > 70:
        bearish_score += 0.7
        bullish_score -= 0.3  # dampen the trend-following bull signal
        reasons.append(f"RSI overbought ({rsi:.0f})")
    elif rsi < 30:
        bullish_score += 0.7
        bearish_score -= 0.3
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi > 60:
        bullish_score += 0.2
        reasons.append(f"RSI bullish ({rsi:.0f})")
    elif rsi < 40:
        bearish_score += 0.2
        reasons.append(f"RSI bearish ({rsi:.0f})")

    # Volatility
    if vol < 0.0018:
        range_score += 1.0
        bullish_score += 0.1
        bearish_score += 0.1
        reasons.append("Low volatility (range favored)")
    elif vol > 0.004:
        bullish_score += 0.3
        bearish_score += 0.3
        reasons.append("High volatility (directional expansion)")
    else:
        bullish_score += 0.15
        bearish_score += 0.15
        range_score += 0.05
        reasons.append("Normal volatility")

    # ── Apply learned bias corrections ───────────────────────────────────────
    weights = _load_weights()
    if weights:
        regime    = get_regime(df)
        now_et    = datetime.now(pytz.timezone("US/Eastern"))
        session   = classify_session(now_et.isoformat())
        vol_state = _detect_volatility(df)

        bullish_score, bearish_score, range_score = _apply_biases(
            bullish_score, bearish_score, range_score,
            regime, session, vol_state, weights,
        )
        reasons.append(f"Bias: {regime}/{session}/{vol_state}")
    # ─────────────────────────────────────────────────────────────────────────

    # Temperature keeps probabilities realistic and avoids overconfidence.
    temperature = 1.35
    exp_bull  = math.exp(bullish_score / temperature)
    exp_bear  = math.exp(bearish_score / temperature)
    exp_range = math.exp(range_score   / temperature)
    total     = exp_bull + exp_bear + exp_range
    bullish    = exp_bull  / total
    bearish    = exp_bear  / total
    range_prob = exp_range / total

    direction = max(
        [("bullish", bullish), ("bearish", bearish), ("range", range_prob)],
        key=lambda x: x[1]
    )[0]

    confidence = max(bullish, bearish, range_prob)

    # Target high/low: use recent volatility to set realistic bands
    # The grading logic: bullish = high_hit AND NOT low_hit
    # So we want: directional target reachable, opposite target NOT reachable
    expected_move = price * (vol * (minutes / 30))
    if expected_move < 0.01:
        expected_move = (high30 - low30) * 0.5

    pred_high = price + expected_move / 2
    pred_low  = price - expected_move / 2

    return {
        "time":      datetime.now(pytz.timezone("US/Eastern")),
        "timeframe": minutes,
        "direction": direction,
        "confidence": round(confidence, 3),
        "high":      round(pred_high, 2),
        "low":       round(pred_low,  2),
        "reasons":   reasons,
    }
