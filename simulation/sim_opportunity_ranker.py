"""
simulation/sim_opportunity_ranker.py

SIM09 Opportunity Ranker — evaluates all signal modes as candidates and picks
the highest-composite-score winner above a minimum threshold.

NO imports from decision/, execution/, or interface/.
"""
from __future__ import annotations

import logging
import os
import yaml
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Config loading (mirrors pattern in sim_engine.py)
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_BASE_DIR, "simulation", "sim_config.yaml")


_SIM_CONFIG_CACHE: dict | None = None


def _load_sim_config() -> dict:
    global _SIM_CONFIG_CACHE
    if _SIM_CONFIG_CACHE is not None:
        return _SIM_CONFIG_CACHE
    try:
        with open(_CONFIG_PATH, "r") as f:
            _SIM_CONFIG_CACHE = yaml.safe_load(f) or {}
    except Exception:
        _SIM_CONFIG_CACHE = {}
    return _SIM_CONFIG_CACHE


# ---------------------------------------------------------------------------
# Mode → regime alignment matrix
# ---------------------------------------------------------------------------

# Regime values from get_regime(): "TREND", "RANGE", "VOLATILE", "COMPRESSION", "NO_DATA"
# Map to matrix keys: TREND→TRENDING_UP/DOWN (use direction), RANGE→RANGE_BOUND,
# VOLATILE→VOLATILE, COMPRESSION→RANGE_BOUND, NO_DATA→None

_REGIME_ALIGNMENT_MATRIX = {
    "TREND_PULLBACK": {
        "TRENDING_UP": 90, "TRENDING_DOWN": 90,
        "RANGE_BOUND": 20, "VOLATILE": 40, None: 50,
    },
    "BREAKOUT": {
        "TRENDING_UP": 70, "TRENDING_DOWN": 70,
        "RANGE_BOUND": 30, "VOLATILE": 80, None: 50,
    },
    "MEAN_REVERSION": {
        "TRENDING_UP": 30, "TRENDING_DOWN": 30,
        "RANGE_BOUND": 90, "VOLATILE": 50, None: 50,
    },
    "ORB_BREAKOUT": {
        "TRENDING_UP": 60, "TRENDING_DOWN": 60,
        "RANGE_BOUND": 40, "VOLATILE": 70, None: 50,
    },
    "SWING_TREND": {
        "TRENDING_UP": 85, "TRENDING_DOWN": 85,
        "RANGE_BOUND": 15, "VOLATILE": 35, None: 50,
    },
    "TRADER_CONVICTION": {
        "TRENDING_UP": 80, "TRENDING_DOWN": 80,
        "RANGE_BOUND": 40, "VOLATILE": 50, None: 50,
    },
}

# Hardcoded fallback timeframe parameters per mode
_MODE_TIMEFRAMES_FALLBACK = {
    "MEAN_REVERSION":    {"dte_min": 0, "dte_max": 1, "hold_max_minutes": 45},
    "BREAKOUT":          {"dte_min": 0, "dte_max": 1, "hold_max_minutes": 60},
    "TREND_PULLBACK":    {"dte_min": 0, "dte_max": 1, "hold_max_minutes": 90},
    "ORB_BREAKOUT":      {"dte_min": 0, "dte_max": 0, "hold_max_minutes": 15},
    "SWING_TREND":       {"dte_min": 1, "dte_max": 7, "hold_max_minutes": 1440},
    "TRADER_CONVICTION": {"dte_min": 0, "dte_max": 1, "hold_max_minutes": 90},
}

# Ranker only evaluates these core signal modes (avoids sparse/exotic modes)
_RANKABLE_MODES = [
    "MEAN_REVERSION",
    "BREAKOUT",
    "TREND_PULLBACK",
    "ORB_BREAKOUT",
    "SWING_TREND",
    "TRADER_CONVICTION",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpportunityResult:
    signal_mode: str
    direction: str
    underlying_price: float
    composite_score: float
    breakdown: dict
    recommended_dte_min: int
    recommended_dte_max: int
    recommended_hold_max_minutes: int
    competing_candidates: list = field(default_factory=list)


@dataclass
class CandidateScore:
    mode: str
    direction: str
    underlying_price: float
    signal_strength: float
    historical: float
    regime: float
    risk_reward: float
    composite: float


# ---------------------------------------------------------------------------
# Helper: normalise a value into [0, 100] clamped range
# ---------------------------------------------------------------------------

def _normalise(value: float, low: float, high: float) -> float:
    """Map value from [low, high] → [0, 100], clamped."""
    if high <= low:
        return 50.0
    result = (value - low) / (high - low) * 100.0
    return max(0.0, min(100.0, result))


# ---------------------------------------------------------------------------
# Helper: map raw regime string to matrix key
# ---------------------------------------------------------------------------

def _map_regime_key(regime: str | None, direction: str | None) -> str | None:
    """
    Map get_regime() output to the alignment matrix keys.
    Returns None for unknown/no-data, or one of:
      TRENDING_UP, TRENDING_DOWN, RANGE_BOUND, VOLATILE
    """
    if regime is None:
        return None
    r = str(regime).upper()
    if r == "TREND":
        # Use direction to distinguish up/down
        d = str(direction or "").upper()
        if d == "BEARISH":
            return "TRENDING_DOWN"
        return "TRENDING_UP"
    if r in ("RANGE", "COMPRESSION"):
        return "RANGE_BOUND"
    if r == "VOLATILE":
        return "VOLATILE"
    # NO_DATA or anything else
    return None


# ---------------------------------------------------------------------------
# Helper: compute historical performance score for a mode
# ---------------------------------------------------------------------------

def _historical_score(sim_states: dict | None, mode: str) -> float:
    """
    Score 0–100 from recent trades of sims using `mode`.
    If fewer than 5 trades available, return 40.
    Formula: win_rate*40 + normalise(profit_factor,0,3)*35 + normalise(avg_pnl,-100,200)*25
    """
    if not sim_states:
        return 40.0

    trades = []
    for sim_id, state in sim_states.items():
        try:
            sim_mode = (state.get("signal_mode") or "").upper()
            if sim_mode != mode.upper():
                continue
            trade_log = state.get("trade_log")
            if not isinstance(trade_log, list):
                continue
            for t in trade_log:
                if not isinstance(t, dict):
                    continue
                pnl = t.get("realized_pnl_dollars")
                if pnl is not None:
                    try:
                        trades.append(float(pnl))
                    except (TypeError, ValueError):
                        pass
        except Exception:
            continue

    # Use last 20
    recent = trades[-20:] if len(trades) >= 20 else trades
    if len(recent) < 5:
        return 40.0

    wins = sum(1 for p in recent if p > 0)
    win_rate = wins / len(recent)

    gross_profit = sum(p for p in recent if p > 0) or 0.0
    gross_loss = abs(sum(p for p in recent if p < 0)) or 1.0
    profit_factor = gross_profit / gross_loss

    avg_pnl = sum(recent) / len(recent)

    score = (
        win_rate * 40.0
        + _normalise(profit_factor, 0.0, 3.0) * 0.35
        + _normalise(avg_pnl, -100.0, 200.0) * 0.25
    )
    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Helper: compute regime alignment score
# ---------------------------------------------------------------------------

def _regime_score(mode: str, regime: str | None, direction: str | None) -> float:
    matrix = _REGIME_ALIGNMENT_MATRIX.get(mode)
    if matrix is None:
        return 50.0
    regime_key = _map_regime_key(regime, direction)
    return float(matrix.get(regime_key, matrix.get(None, 50.0)))


# ---------------------------------------------------------------------------
# Helper: compute risk/reward score
# ---------------------------------------------------------------------------

def _risk_reward_score(df, mode: str) -> float:
    """
    Default 60.0. Adjustments:
    - ATR percentile over 50-bar range: +/-20 for breakout/trend vs mean-rev
    - ORB: morning time bonus +10
    - Spread: N/A at ranker level (no option chain access)
    """
    base = 60.0
    try:
        if df is None or len(df) < 10:
            return base

        # Compute ATR percentile
        atr_col = None
        for c in ("atr", "ATR", "atr14"):
            if c in df.columns:
                atr_col = c
                break

        if atr_col is not None:
            lookback = min(50, len(df))
            atr_series = df[atr_col].iloc[-lookback:].dropna()
            if len(atr_series) >= 5:
                current_atr = float(atr_series.iloc[-1])
                pct_rank = float((atr_series < current_atr).sum()) / len(atr_series) * 100.0

                mode_upper = mode.upper()
                if mode_upper in ("BREAKOUT", "TREND_PULLBACK", "ORB_BREAKOUT", "SWING_TREND", "TRADER_CONVICTION"):
                    # High ATR good for breakout/trend modes
                    if pct_rank > 75:
                        base += 20.0
                    elif pct_rank < 25:
                        base -= 20.0
                elif mode_upper == "MEAN_REVERSION":
                    # Low ATR good for mean reversion (range environment)
                    if pct_rank < 25:
                        base += 20.0
                    elif pct_rank > 75:
                        base -= 20.0

        # ORB: morning bonus
        if mode.upper() == "ORB_BREAKOUT":
            try:
                import pytz
                from datetime import datetime, time as dtime
                now_et = datetime.now(pytz.timezone("US/Eastern"))
                if dtime(9, 30) <= now_et.time() <= dtime(10, 30):
                    base += 10.0
            except Exception:
                pass

    except Exception:
        pass

    return max(0.0, min(100.0, base))


# ---------------------------------------------------------------------------
# Helper: compute signal strength for each mode
# ---------------------------------------------------------------------------

def _signal_strength_from_result(mode: str, direction: str | None, price: float | None,
                                  signal_meta: dict | None) -> float:
    """Derive signal strength 0–100 from the signal's natural metrics."""
    if direction is None or price is None:
        return 0.0

    base = 60.0
    try:
        if isinstance(signal_meta, dict):
            # Use structure_score if present
            sc = signal_meta.get("structure_score")
            if isinstance(sc, (int, float)):
                base = _normalise(float(sc), 0.0, 5.0) * 0.7 + 30.0

            # Boost for specific reasons
            reason = str(signal_meta.get("reason", "")).lower()
            if "breakout" in reason:
                base = min(100.0, base + 10.0)
            elif "pullback" in reason:
                base = min(100.0, base + 5.0)

            # RSI extremes boost for mean reversion
            rsi = signal_meta.get("rsi")
            if isinstance(rsi, (int, float)):
                rsi_f = float(rsi)
                if rsi_f < 25 or rsi_f > 75:
                    base = min(100.0, base + 15.0)
                elif rsi_f < 30 or rsi_f > 70:
                    base = min(100.0, base + 8.0)

    except Exception:
        pass

    return max(0.0, min(100.0, base))


def _trader_conviction_strength(trader_signal: dict | None) -> tuple[float, str | None, float | None]:
    """
    Return (signal_strength, direction, underlying_price) for TRADER_CONVICTION mode.
    If trader_signal is None or has no conviction, returns (0, None, None).
    """
    if not isinstance(trader_signal, dict):
        return 0.0, None, None

    direction = trader_signal.get("direction")
    if isinstance(direction, str):
        direction = direction.upper()  # normalize "bullish" → "BULLISH"
    if direction not in ("BULLISH", "BEARISH"):
        return 0.0, None, None

    underlying_price = trader_signal.get("underlying_price")
    if not isinstance(underlying_price, (int, float)) or underlying_price <= 0:
        return 0.0, None, None

    conviction_score = trader_signal.get("conviction_score")
    try:
        cs = float(conviction_score) if conviction_score is not None else 0.5
    except (TypeError, ValueError):
        cs = 0.5

    base_strength = cs * 100.0

    # +10 if environment_passed AND ml_prediction agrees with direction
    environment_passed = trader_signal.get("environment_passed")
    ml_prediction = trader_signal.get("ml_prediction")

    if environment_passed:
        ml_dir = None
        if isinstance(ml_prediction, dict):
            ml_dir = ml_prediction.get("predicted_direction") or ml_prediction.get("direction")
            if isinstance(ml_dir, str):
                ml_dir = ml_dir.upper()
        if ml_dir is not None and ml_dir == direction:
            base_strength = min(100.0, base_strength + 10.0)
    elif environment_passed is False:
        base_strength = max(0.0, base_strength - 15.0)

    return min(100.0, base_strength), direction, float(underlying_price)


# ---------------------------------------------------------------------------
# Main ranker class
# ---------------------------------------------------------------------------

class OpportunityRanker:
    """
    Stateless per-call ranker.  Evaluates all rankable signal modes and returns
    the best OpportunityResult above the composite threshold, or None.
    """

    COMPOSITE_THRESHOLD = 55.0

    def __init__(self):
        self._config = _load_sim_config()
        self._signal_to_sims = self._build_signal_to_sims()
        self._mode_timeframes = self._build_mode_timeframes()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _build_signal_to_sims(self) -> dict[str, list[str]]:
        """Group sim_ids by signal_mode from config. TRADER_CONVICTION → SIM00+SIM03."""
        mapping: dict[str, list[str]] = {}
        for sim_id, profile in self._config.items():
            if str(sim_id).startswith("_") or not isinstance(profile, dict):
                continue
            mode = str(profile.get("signal_mode", "")).upper()
            if not mode:
                continue
            mapping.setdefault(mode, []).append(str(sim_id))
        mapping["TRADER_CONVICTION"] = ["SIM00", "SIM03"]
        return mapping

    def _build_mode_timeframes(self) -> dict[str, dict]:
        """
        Build DTE/hold parameters per mode from median of sim profiles.
        Fall back to _MODE_TIMEFRAMES_FALLBACK for any missing mode.
        """
        from statistics import median

        mode_dte_min: dict[str, list] = {}
        mode_dte_max: dict[str, list] = {}
        mode_hold_max: dict[str, list] = {}

        for sim_id, profile in self._config.items():
            if str(sim_id).startswith("_") or not isinstance(profile, dict):
                continue
            mode = str(profile.get("signal_mode", "")).upper()
            if not mode:
                continue
            dte_min = profile.get("dte_min")
            dte_max = profile.get("dte_max")
            hold_max = profile.get("hold_max_seconds")
            if isinstance(dte_min, (int, float)):
                mode_dte_min.setdefault(mode, []).append(int(dte_min))
            if isinstance(dte_max, (int, float)):
                mode_dte_max.setdefault(mode, []).append(int(dte_max))
            if isinstance(hold_max, (int, float)):
                mode_hold_max.setdefault(mode, []).append(int(hold_max) // 60)

        result = {}
        for mode in set(list(mode_dte_min) + list(mode_dte_max) + list(mode_hold_max)):
            fb = _MODE_TIMEFRAMES_FALLBACK.get(mode, {"dte_min": 0, "dte_max": 1, "hold_max_minutes": 60})
            dmin_vals = mode_dte_min.get(mode, [])
            dmax_vals = mode_dte_max.get(mode, [])
            hmax_vals = mode_hold_max.get(mode, [])
            result[mode] = {
                "dte_min": int(median(dmin_vals)) if dmin_vals else fb["dte_min"],
                "dte_max": int(median(dmax_vals)) if dmax_vals else fb["dte_max"],
                "hold_max_minutes": int(median(hmax_vals)) if hmax_vals else fb["hold_max_minutes"],
            }

        # Ensure all fallback modes are covered
        for mode, fb in _MODE_TIMEFRAMES_FALLBACK.items():
            if mode not in result:
                result[mode] = dict(fb)

        return result

    def _get_timeframe(self, mode: str) -> dict:
        return self._mode_timeframes.get(
            mode.upper(),
            _MODE_TIMEFRAMES_FALLBACK.get(mode.upper(), {"dte_min": 0, "dte_max": 1, "hold_max_minutes": 60})
        )

    # ------------------------------------------------------------------
    # Build sim_states summary from SimPortfolio trade logs
    # ------------------------------------------------------------------

    def _build_sim_states_summary(self, sim_states_raw) -> dict:
        """
        Accept either:
        - dict of sim_id → SimPortfolio object (has .trade_log, .profile)
        - dict of sim_id → plain dict (from tests or callers that pre-built it)
        Returns normalised dict keyed by sim_id with signal_mode + trade_log.
        """
        result = {}
        if not sim_states_raw:
            return result

        for sim_id, state in sim_states_raw.items():
            try:
                if hasattr(state, "profile") and hasattr(state, "trade_log"):
                    # SimPortfolio object
                    result[sim_id] = {
                        "signal_mode": state.profile.get("signal_mode", ""),
                        "trade_log": state.trade_log if isinstance(state.trade_log, list) else [],
                    }
                elif isinstance(state, dict):
                    result[sim_id] = state
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Core ranking
    # ------------------------------------------------------------------

    def rank_opportunities(
        self,
        df,
        sim_states,
        regime: str | None,
        trader_signal: dict | None = None,
    ) -> OpportunityResult | None:
        """
        Evaluate all rankable signal modes, return the best OpportunityResult
        above COMPOSITE_THRESHOLD, or None.
        """
        # Lazily import signal functions here (not at module level) to avoid
        # circular imports and to keep this module free of interface/ imports.
        from simulation.sim_signal_funcs import (
            _signal_mean_reversion,
            _signal_breakout,
            _signal_trend_pullback,
            _signal_swing_trend,
            _signal_orb_breakout,
        )

        normalised_states = self._build_sim_states_summary(sim_states or {})
        candidates: list[CandidateScore] = []

        for mode in _RANKABLE_MODES:
            try:
                if mode == "TRADER_CONVICTION":
                    strength, direction, price = _trader_conviction_strength(trader_signal)
                    if direction is None:
                        continue
                    signal_meta = None
                else:
                    # Call each signal function with try/except
                    try:
                        if mode == "MEAN_REVERSION":
                            direction, price, signal_meta = _signal_mean_reversion(df)
                        elif mode == "BREAKOUT":
                            direction, price, signal_meta = _signal_breakout(df)
                        elif mode == "TREND_PULLBACK":
                            direction, price, signal_meta = _signal_trend_pullback(df)
                        elif mode == "SWING_TREND":
                            direction, price, signal_meta = _signal_swing_trend(df)
                        elif mode == "ORB_BREAKOUT":
                            # ORB requires feature_snapshot — skip without it
                            direction, price, signal_meta = None, None, {"reason": "no_feature_snapshot"}
                        else:
                            direction, price, signal_meta = None, None, None
                    except Exception:
                        direction, price, signal_meta = None, None, None

                    if direction is None or price is None:
                        continue

                    strength = _signal_strength_from_result(mode, direction, price, signal_meta)

                hist = _historical_score(normalised_states, mode)
                reg = _regime_score(mode, regime, direction)
                rr = _risk_reward_score(df, mode)

                composite = (
                    strength * 0.25
                    + hist * 0.35
                    + reg * 0.20
                    + rr * 0.20
                )

                candidates.append(CandidateScore(
                    mode=mode,
                    direction=direction,
                    underlying_price=float(price),
                    signal_strength=round(strength, 1),
                    historical=round(hist, 1),
                    regime=round(reg, 1),
                    risk_reward=round(rr, 1),
                    composite=round(composite, 2),
                ))

            except Exception:
                continue

        if not candidates:
            logging.info("SIM09 Ranker: no candidates fired")
            return None

        # Sort by composite score descending
        candidates.sort(key=lambda c: c.composite, reverse=True)
        winner = candidates[0]

        if winner.composite < self.COMPOSITE_THRESHOLD:
            logging.info(
                "SIM09 Ranker: no candidate above threshold (best=%s at %.1f)",
                winner.mode, winner.composite,
            )
            return None

        # Build competing list (all except winner)
        competing = [
            {
                "mode": c.mode,
                "composite_score": c.composite,
                "breakdown": {
                    "signal_strength": c.signal_strength,
                    "historical": c.historical,
                    "regime": c.regime,
                    "risk_reward": c.risk_reward,
                },
            }
            for c in candidates[1:]
        ]

        logging.info(
            "SIM09 Ranker: %d candidates fired, winner=%s score=%.1f",
            len(candidates), winner.mode, winner.composite,
        )

        tf = self._get_timeframe(winner.mode)

        return OpportunityResult(
            signal_mode=winner.mode,
            direction=winner.direction,
            underlying_price=winner.underlying_price,
            composite_score=winner.composite,
            breakdown={
                "signal_strength": winner.signal_strength,
                "historical": winner.historical,
                "regime": winner.regime,
                "risk_reward": winner.risk_reward,
            },
            recommended_dte_min=tf["dte_min"],
            recommended_dte_max=tf["dte_max"],
            recommended_hold_max_minutes=tf["hold_max_minutes"],
            competing_candidates=competing,
        )
