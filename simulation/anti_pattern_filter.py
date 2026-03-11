"""
simulation/anti_pattern_filter.py
Data-driven anti-pattern filter for sim entries.
Reads per-sim anti-patterns from research/patterns/{sim_id}_patterns.json
and blocks entries that match known losing condition combos.
"""
import json
import logging
import os
from datetime import datetime

import pytz

_PATTERNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "research", "patterns",
)

# Cache: sim_id -> (mtime, anti_patterns list)
_CACHE: dict[str, tuple[float, list]] = {}

# Minimum score to treat as a blocking anti-pattern
MIN_ANTI_PATTERN_SCORE = 20.0


def _load_anti_patterns(sim_id: str) -> list[dict]:
    """Load anti-patterns for a sim, with file-mtime caching."""
    path = os.path.join(_PATTERNS_DIR, f"{sim_id}_patterns.json")
    if not os.path.exists(path):
        return []
    try:
        mtime = os.path.getmtime(path)
        cached = _CACHE.get(sim_id)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(path) as f:
            data = json.load(f)
        patterns = data.get("anti_patterns", [])
        # Only keep patterns above score threshold
        patterns = [p for p in patterns if p.get("score", 0) >= MIN_ANTI_PATTERN_SCORE]
        _CACHE[sim_id] = (mtime, patterns)
        return patterns
    except Exception as exc:
        logging.debug("anti_pattern_load_error: %s — %s", sim_id, exc)
        return []


def _compute_current_tags(df, direction: str, regime: str | None) -> dict:
    """Compute market condition tags from the current dataframe.

    Mirrors the tagging logic in research/pattern_pipeline.py:tag_trade()
    so that anti-pattern conditions can be matched at entry time.
    """
    tags = {}
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)

    tags["hour_bucket"] = now_et.hour
    tags["day_of_week"] = now_et.strftime("%A")
    tags["regime_at_entry"] = regime or "UNKNOWN"

    if df is None or df.empty:
        return tags

    # ATR state (20-bar)
    if len(df) >= 20 and "high" in df.columns and "low" in df.columns:
        tr = df["high"] - df["low"]
        atr_current = tr.iloc[-1]
        atr_sma = tr.tail(20).mean()
        if atr_sma > 0:
            ratio = atr_current / atr_sma
            if ratio < 0.8:
                tags["atr_state"] = "COMPRESSING"
            elif ratio > 1.2:
                tags["atr_state"] = "EXPANDING"
            else:
                tags["atr_state"] = "NORMAL"
        else:
            tags["atr_state"] = "UNKNOWN"
    else:
        tags["atr_state"] = "UNKNOWN"

    # Trend alignment (50-bar SMA slope vs direction)
    if len(df) >= 51 and "close" in df.columns:
        sma50 = df["close"].tail(50).mean()
        sma50_prev = df["close"].iloc[-51:-1].mean()
        sma_slope = "UP" if sma50 > sma50_prev else "DOWN"
        if direction in ("BULLISH", "CALL"):
            tags["trend_alignment"] = "YES" if sma_slope == "UP" else "NO"
        elif direction in ("BEARISH", "PUT"):
            tags["trend_alignment"] = "YES" if sma_slope == "DOWN" else "NO"
        else:
            tags["trend_alignment"] = "UNKNOWN"
    else:
        tags["trend_alignment"] = "UNKNOWN"

    # Volume state (20-bar)
    if len(df) >= 20 and "volume" in df.columns:
        vol_current = df["volume"].iloc[-1]
        vol_sma = df["volume"].tail(20).mean()
        if vol_sma > 0:
            ratio = vol_current / vol_sma
            if ratio < 0.7:
                tags["volume_state"] = "LOW"
            elif ratio > 1.3:
                tags["volume_state"] = "HIGH"
            else:
                tags["volume_state"] = "NORMAL"
        else:
            tags["volume_state"] = "UNKNOWN"
    else:
        tags["volume_state"] = "UNKNOWN"

    # VIX level from VXX data
    try:
        from core.data_service import get_market_dataframe
        vxx_df = get_market_dataframe(symbol="VXX")
        if vxx_df is not None and not vxx_df.empty:
            vxx_price = vxx_df["close"].iloc[-1]
            if vxx_price < 36:
                tags["vix_level"] = "LOW"
            elif vxx_price < 45:
                tags["vix_level"] = "NORMAL"
            elif vxx_price < 53:
                tags["vix_level"] = "ELEVATED"
            else:
                tags["vix_level"] = "HIGH"
        else:
            tags["vix_level"] = "UNKNOWN"
    except Exception:
        tags["vix_level"] = "UNKNOWN"

    return tags


def _matches_pattern(tags: dict, conditions: dict) -> bool:
    """Check if all conditions in an anti-pattern match current tags."""
    for key, required_val in conditions.items():
        current_val = tags.get(key)
        if current_val is None or current_val == "UNKNOWN":
            return False  # Can't confirm match — don't block
        # hour_bucket is int in tags, may be int or str in conditions
        if key == "hour_bucket":
            if int(current_val) != int(required_val):
                return False
        elif str(current_val) != str(required_val):
            return False
    return True


def check_anti_patterns(sim_id: str, df, direction: str,
                        regime: str | None = None) -> str | None:
    """Check if current conditions match any anti-pattern for this sim.

    Returns None if entry is OK, or a skip reason string if blocked.
    """
    patterns = _load_anti_patterns(sim_id)
    if not patterns:
        return None

    tags = _compute_current_tags(df, direction, regime)

    for pattern in patterns:
        conditions = pattern.get("conditions", {})
        if not conditions:
            continue
        if _matches_pattern(tags, conditions):
            desc = pattern.get("description", "anti-pattern match")
            score = pattern.get("score", 0)
            logging.info(
                "anti_pattern_blocked: %s — %s (score=%.1f, tags=%s)",
                sim_id, desc, score, tags,
            )
            return f"anti_pattern:{desc}"

    return None
