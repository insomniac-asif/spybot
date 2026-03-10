"""simulation/trade_analyzer.py — A-Tier Entry Filter System.

Mine closed trade history to identify what separates winners from losers,
produce multi-dimensional quality grades, and generate concrete entry filters.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Path helpers ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SIM_DIR = os.path.join(_ROOT, "data", "sims")
_CONFIG_PATH = os.path.join(_HERE, "sim_config.yaml")


# ── Regime preference maps ─────────────────────────────────────────────────────
_REGIME_PREF: dict[str, dict[str, float]] = {
    "TREND_PULLBACK":       {"TREND": 100, "RANGE": 40, "VOLATILE": 20, "SIDEWAYS": 40},
    "MEAN_REVERSION":       {"RANGE": 100, "SIDEWAYS": 90, "VOLATILE": 70, "TREND": 30},
    "BREAKOUT":             {"VOLATILE": 90, "TREND": 70, "RANGE": 50},
    "ORB_BREAKOUT":         {"VOLATILE": 90, "TREND": 70, "RANGE": 50},
    "SWING_TREND":          {"TREND": 100, "VOLATILE": 50, "RANGE": 30},
    "VOLATILITY_EXPANSION": {"VOLATILE": 100, "TREND": 60, "RANGE": 20},
    "REVERSAL":             {"RANGE": 90, "SIDEWAYS": 90, "VOLATILE": 60, "TREND": 40},
    "CONTRA_TREND":         {"RANGE": 90, "SIDEWAYS": 90, "VOLATILE": 60, "TREND": 40},
    "FADE":                 {"VOLATILE": 90, "RANGE": 70, "TREND": 40},
    "OPPORTUNITY":          {},  # neutral — always 60
}

_BREAKOUT_FAMILY = frozenset({"BREAKOUT", "ORB_BREAKOUT", "VOLATILITY_EXPANSION"})
_TREND_FAMILY    = frozenset({"TREND_PULLBACK", "SWING_TREND"})
_REVERSAL_FAMILY = frozenset({"MEAN_REVERSION", "REVERSAL", "CONTRA_TREND", "FADE"})


# ── Default weights by signal family ─────────────────────────────────────────
_WEIGHTS: dict[str, dict[str, float]] = {
    "breakout": {
        "ml_confidence": 0.20, "regime_alignment": 0.15, "timing_quality": 0.20,
        "entry_efficiency": 0.15, "exit_efficiency": 0.10, "risk_reward_actual": 0.10,
        "hold_efficiency": 0.05, "spread_quality": 0.03, "contract_quality": 0.02,
    },
    "trend": {
        "ml_confidence": 0.25, "regime_alignment": 0.20, "timing_quality": 0.10,
        "entry_efficiency": 0.15, "exit_efficiency": 0.10, "risk_reward_actual": 0.10,
        "hold_efficiency": 0.05, "spread_quality": 0.03, "contract_quality": 0.02,
    },
    "reversal": {
        "ml_confidence": 0.20, "regime_alignment": 0.20, "timing_quality": 0.15,
        "entry_efficiency": 0.20, "exit_efficiency": 0.10, "risk_reward_actual": 0.08,
        "hold_efficiency": 0.04, "spread_quality": 0.02, "contract_quality": 0.01,
    },
    "default": {
        "ml_confidence": 0.25, "regime_alignment": 0.15, "timing_quality": 0.15,
        "entry_efficiency": 0.15, "exit_efficiency": 0.10, "risk_reward_actual": 0.10,
        "hold_efficiency": 0.05, "spread_quality": 0.03, "contract_quality": 0.02,
    },
}


def _get_weight_family(signal_mode: Optional[str]) -> str:
    sm = (signal_mode or "").upper()
    if sm in _BREAKOUT_FAMILY:
        return "breakout"
    if sm in _TREND_FAMILY:
        return "trend"
    if sm in _REVERSAL_FAMILY:
        return "reversal"
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_profile(sim_id: str) -> dict:
    """Load sim profile from sim_config.yaml. Return {} on failure."""
    try:
        import yaml
        with open(_CONFIG_PATH, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        return dict(cfg.get(sim_id, {}))
    except Exception as exc:
        logger.debug("_load_profile error: %s", exc)
        return {}


def load_sim_trades(sim_id: str) -> list:
    """Load closed trades from data/sims/{sim_id}.json. Return [] on failure."""
    try:
        path = os.path.join(_SIM_DIR, f"{sim_id}.json")
        if not os.path.exists(path):
            return []
        with open(path, "r") as fh:
            data = json.load(fh)
        trades = data.get("trade_log", [])
        closed = [
            t for t in trades
            if isinstance(t, dict)
            and (t.get("exit_price") is not None or t.get("realized_pnl_dollars") is not None)
        ]
        return closed
    except Exception as exc:
        logger.debug("load_sim_trades error sim=%s: %s", sim_id, exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trade_grade_compat(trade: dict) -> Optional[float]:
    """Return best ML confidence score from a trade dict.

    Tries to import from simulation.sim_engine first, falls back to inline logic.
    """
    try:
        from simulation.sim_engine import _trade_grade
        return _trade_grade(trade)
    except Exception:
        pass
    # Fallback: replicate logic inline
    try:
        candidates = []
        for key in ("edge_prob", "prediction_confidence", "confidence", "ml_probability"):
            val = trade.get(key)
            if isinstance(val, (int, float)):
                candidates.append(float(val))
        return max(candidates) if candidates else None
    except Exception:
        return None


def _score_regime_alignment(
    regime: Optional[str],
    signal_mode: Optional[str],
    profile: Optional[dict],
) -> Optional[float]:
    """Score 0–100 for how well the regime matched this signal_mode.

    If profile has regime_filter (whitelist), that is the ground truth.
    """
    try:
        if regime is None:
            return None

        regime = (regime or "").upper()
        sm = (signal_mode or "").upper()

        # Profile has explicit regime whitelist — use it as ground truth
        regime_filter = (profile or {}).get("regime_filter")
        if regime_filter is not None:
            if isinstance(regime_filter, list):
                allowed = [str(r).upper() for r in regime_filter]
            elif isinstance(regime_filter, str):
                # Handle string shortcuts
                _MAP = {
                    "TREND_ONLY": ["TREND"],
                    "RANGE_ONLY": ["RANGE"],
                    "VOLATILE_ONLY": ["VOLATILE"],
                }
                allowed = _MAP.get(regime_filter.upper(), [regime_filter.upper()])
            else:
                allowed = []
            return 100.0 if regime in allowed else 20.0

        # Use regime preference map
        pref = _REGIME_PREF.get(sm)
        if pref is None:
            # Unknown signal mode → neutral
            return 50.0
        if not pref:
            # OPPORTUNITY → always neutral
            return 60.0
        return float(pref.get(regime, 50))
    except Exception as exc:
        logger.debug("_score_regime_alignment error: %s", exc)
        return None


def _score_timing_quality(
    time_bucket: Optional[str],
    bucket_win_rates: dict,
) -> Optional[float]:
    """Score 0–100 based on historical win rate for this time bucket.

    Returns None if bucket has < 5 trades.
    """
    try:
        if time_bucket is None:
            return None
        info = bucket_win_rates.get(time_bucket)
        if not info:
            return None
        count = info.get("count", 0)
        if count < 5:
            return None
        win_rate = info.get("win_rate", 0.5)
        return float(win_rate * 100)
    except Exception as exc:
        logger.debug("_score_timing_quality error: %s", exc)
        return None


def _score_entry_efficiency(
    mae_pct: Optional[float],
    stop_loss_pct: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> Optional[float]:
    """Score 0–100 based on Maximum Adverse Excursion.

    mae_pct is negative (e.g. -0.118 = 11.8% adverse). We score relative to the
    stop_loss_pct so that the worst case (hit stop) = 0 and no adverse move = 100.

    Formula: max(0, (1 - abs(mae_pct) / worst_case)) * 100
    """
    try:
        if mae_pct is None:
            return None
        mag = abs(float(mae_pct))
        # Use stop_loss_pct as the worst-case normalizer when available.
        # Fall back to a hard default of 0.25 (25%) so options-scale MAE values
        # produce meaningful scores rather than always collapsing to 0.
        worst = float(stop_loss_pct) if stop_loss_pct and float(stop_loss_pct) > 0 else 0.25
        return float(max(0.0, (1.0 - mag / worst) * 100.0))
    except Exception as exc:
        logger.debug("_score_entry_efficiency error: %s", exc)
        return None


def _score_exit_efficiency(
    mfe_pct: Optional[float],
    realized_pnl_dollars: Optional[float],
    entry_price: Optional[float],
    qty: Optional[int] = None,
) -> Optional[float]:
    """Score 0–100 based on how much of the MFE opportunity was captured.

    mfe_dollars is scaled by qty * 100 (options contract multiplier) to match
    the scale of realized_pnl_dollars which is already in full position terms.
    """
    try:
        if mfe_pct is None or realized_pnl_dollars is None or entry_price is None:
            return None
        mfe_pct_f = float(mfe_pct)
        pnl = float(realized_pnl_dollars)
        ep = float(entry_price)
        if ep <= 0:
            return None

        _qty = float(qty) if qty and float(qty) > 0 else 1.0
        # Scale by qty * 100 so mfe_dollars is in the same units as realized_pnl_dollars
        mfe_dollars = mfe_pct_f * ep * _qty * 100.0
        if mfe_pct_f <= 0:
            # No positive MFE excursion at all
            if pnl < 0:
                return 10.0
            return 30.0

        if mfe_dollars <= 0:
            return 5.0

        capture_ratio = pnl / mfe_dollars

        if pnl < 0 and mfe_pct_f > 0:
            # Had opportunity but gave it back → especially bad
            return float(max(0.0, 15.0 * (1 + capture_ratio)))

        if capture_ratio >= 0.8:
            return 100.0
        if capture_ratio >= 0.5:
            return float(60.0 + (capture_ratio - 0.5) / 0.3 * 40.0)
        if capture_ratio >= 0.2:
            return float(20.0 + (capture_ratio - 0.2) / 0.3 * 40.0)
        return float(max(0.0, capture_ratio / 0.2 * 20.0))
    except Exception as exc:
        logger.debug("_score_exit_efficiency error: %s", exc)
        return None


def _score_risk_reward(
    realized_pnl_dollars: Optional[float],
    max_risk: Optional[float],
) -> Optional[float]:
    """Score 0–100 based on actual R-multiple achieved."""
    try:
        if realized_pnl_dollars is None or max_risk is None or max_risk <= 0:
            return None
        r = float(realized_pnl_dollars) / float(max_risk)
        if r >= 2.0:
            return 100.0
        if r >= 1.0:
            return float(70.0 + (r - 1.0) * 30.0)
        if r >= 0.5:
            return float(50.0 + (r - 0.5) * 40.0)
        if r >= 0.0:
            return float(30.0 + r / 0.5 * 20.0)
        # r < 0 (loss): scale 0–30
        return float(max(0.0, 30.0 + r * 15.0))  # r=-2 → 0, r=0 → 30
    except Exception as exc:
        logger.debug("_score_risk_reward error: %s", exc)
        return None


def _score_hold_efficiency(
    hold_seconds: Optional[float],
    profile: Optional[dict],
    realized_pnl_dollars: Optional[float],
) -> Optional[float]:
    """Score 0–100 based on hold time relative to profile bounds."""
    try:
        if hold_seconds is None or profile is None:
            return None
        hold_min = profile.get("hold_min_seconds") or profile.get("hold_min")
        hold_max = profile.get("hold_max_seconds") or profile.get("hold_max")
        if hold_min is None or hold_max is None:
            return None
        hold_min = float(hold_min)
        hold_max = float(hold_max)
        if hold_max <= hold_min:
            return None

        hold_seconds = float(hold_seconds)
        pnl = float(realized_pnl_dollars) if realized_pnl_dollars is not None else None
        is_win = pnl is not None and pnl > 0

        # Ratio of hold within [hold_min, hold_max]
        near_min = hold_seconds <= hold_min * 1.5
        near_max = hold_seconds >= hold_max * 0.85

        if is_win and near_min:
            return 90.0   # quick win = great
        if is_win and near_max:
            return 60.0   # held a long time but won = ok
        if is_win:
            return 75.0   # moderate hold, won = good
        if not is_win and near_max:
            return 10.0   # held loser to max = bad
        if not is_win and near_min:
            return 50.0   # quick cut = ok
        return 35.0       # held loser somewhere in middle
    except Exception as exc:
        logger.debug("_score_hold_efficiency error: %s", exc)
        return None


def _score_spread_quality(
    spread_at_entry: Optional[float],
    spread_guard_pct: Optional[float],
) -> Optional[float]:
    """Score 0–100 based on spread relative to max allowed spread."""
    try:
        if spread_at_entry is None or spread_guard_pct is None:
            return None
        if float(spread_guard_pct) <= 0:
            return None
        ratio = float(spread_at_entry) / float(spread_guard_pct)
        if ratio <= 0.3:
            return 100.0
        if ratio <= 0.7:
            return float(70.0 + (0.7 - ratio) / 0.4 * 30.0)
        if ratio <= 1.0:
            return float(40.0 + (1.0 - ratio) / 0.3 * 30.0)
        # ratio > 1.0: shouldn't have entered
        return float(max(0.0, 40.0 - (ratio - 1.0) * 40.0))
    except Exception as exc:
        logger.debug("_score_spread_quality error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive trade grade
# ─────────────────────────────────────────────────────────────────────────────

def comprehensive_trade_grade(
    trade: dict,
    profile: Optional[dict] = None,
    bucket_win_rates: Optional[dict] = None,
) -> dict:
    """Multi-dimensional trade quality assessment.

    Returns a dict with composite_score, letter_grade, dimensions, flags, weights_used.
    """
    try:
        signal_mode = trade.get("signal_mode")
        fam = _get_weight_family(signal_mode)
        base_weights = dict(_WEIGHTS[fam])

        # ── Compute each dimension ────────────────────────────────────────────

        # ML confidence: use _trade_grade_compat → map 0.5–1.0 to 0–100.
        # Cold-start detection: if ALL available ML fields are exactly 0.5, the
        # model has never been trained — treat as unavailable (None) rather than
        # scoring it as 0 (which would unfairly drag down composite grades).
        ml_grade_raw = _trade_grade_compat(trade)
        _ml_keys = ("edge_prob", "prediction_confidence", "ml_probability")
        _ml_present = [trade.get(k) for k in _ml_keys if trade.get(k) is not None]
        _ml_all_half = bool(_ml_present) and all(v == 0.5 for v in _ml_present)
        if _ml_all_half:
            ml_score: Optional[float] = None  # cold-start default — exclude from weighting
        elif ml_grade_raw is not None:
            # 0.5 is random baseline → 0 score; 1.0 → 100 score
            ml_score = float(max(0.0, min(100.0, (ml_grade_raw - 0.5) * 200.0)))
        else:
            ml_score = None

        regime = trade.get("regime_at_entry")
        regime_score = _score_regime_alignment(regime, signal_mode, profile)

        time_bucket = trade.get("time_of_day_bucket")
        bwr = bucket_win_rates or {}
        timing_score = _score_timing_quality(time_bucket, bwr)

        # Resolve stop_loss_pct and qty once — used by multiple scorers below
        stop_loss_pct = None
        try:
            _sl = trade.get("stop_loss_pct") or (profile or {}).get("stop_loss_pct")
            if _sl is not None:
                stop_loss_pct = float(_sl)
        except Exception:
            pass
        qty = trade.get("qty", 1) or 1

        mae_pct = trade.get("mae_pct")
        entry_eff_score = _score_entry_efficiency(mae_pct, stop_loss_pct=stop_loss_pct)

        mfe_pct = trade.get("mfe_pct")
        realized_pnl = trade.get("realized_pnl_dollars")
        entry_price = trade.get("entry_price")
        exit_eff_score = _score_exit_efficiency(mfe_pct, realized_pnl, entry_price, qty=qty)

        # Risk-reward: try to derive max_risk from trade
        max_risk: Optional[float] = None
        try:
            mr = trade.get("max_risk") or trade.get("stop_loss_dollars")
            if mr is not None:
                max_risk = abs(float(mr))
            elif entry_price is not None and stop_loss_pct is not None:
                max_risk = float(entry_price) * float(qty) * 100.0 * stop_loss_pct
        except Exception:
            max_risk = None
        rr_score = _score_risk_reward(realized_pnl, max_risk)

        hold_seconds = trade.get("time_in_trade_seconds")
        hold_eff_score = _score_hold_efficiency(hold_seconds, profile, realized_pnl)

        # Spread quality: use spread_pct captured at entry (now stored since the
        # _build_paper_trade_dict fix). Fall back to spread_guard_bypassed flag.
        spread_score: Optional[float] = None
        _spread_pct = trade.get("spread_pct")
        _spread_guard = (profile or {}).get("max_spread_pct") if profile else None
        if _spread_pct is not None and _spread_guard is not None:
            spread_score = _score_spread_quality(_spread_pct, _spread_guard)
        elif trade.get("spread_guard_bypassed") is True:
            spread_score = 25.0

        contract_score: Optional[float] = None  # future placeholder

        # ── Assemble dimension dict ───────────────────────────────────────────
        dimensions = {
            "ml_confidence":    ml_score,
            "regime_alignment": regime_score,
            "timing_quality":   timing_score,
            "entry_efficiency": entry_eff_score,
            "exit_efficiency":  exit_eff_score,
            "risk_reward_actual": rr_score,
            "hold_efficiency":  hold_eff_score,
            "spread_quality":   spread_score,
            "contract_quality": contract_score,
        }

        # ── Weight redistribution for missing dimensions ──────────────────────
        available = {k: v for k, v in dimensions.items() if v is not None}
        total_base_weight = sum(base_weights[k] for k in available)
        if total_base_weight <= 0:
            # No data at all
            return {
                "composite_score": 50.0,
                "letter_grade": "C",
                "dimensions": dimensions,
                "flags": ["no_data"],
                "weights_used": {},
            }

        weights_used = {
            k: base_weights[k] / total_base_weight
            for k in available
        }

        composite = sum(available[k] * weights_used[k] for k in available)

        # ── Letter grade ──────────────────────────────────────────────────────
        if composite >= 75:
            letter = "A"
        elif composite >= 60:
            letter = "B"
        elif composite >= 45:
            letter = "C"
        elif composite >= 30:
            letter = "D"
        else:
            letter = "F"

        # ── Flags ─────────────────────────────────────────────────────────────
        flags = []
        if regime is None:
            flags.append("no_regime_data")
        if regime_score is not None and regime_score < 40:
            flags.append("poor_regime_alignment")
        if time_bucket == "CLOSE":
            flags.append("entered_close_session")
        if spread_score is not None and spread_score < 40:
            flags.append("high_spread_entry")
        if hold_eff_score is not None and hold_eff_score < 25 and realized_pnl is not None and realized_pnl < 0:
            flags.append("held_too_long_loser")

        return {
            "composite_score": float(composite),
            "letter_grade": letter,
            "dimensions": dimensions,
            "flags": flags,
            "weights_used": weights_used,
        }
    except Exception as exc:
        logger.debug("comprehensive_trade_grade error: %s", exc)
        return {
            "composite_score": 50.0,
            "letter_grade": "C",
            "dimensions": {},
            "flags": ["grade_error"],
            "weights_used": {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Historical edge computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_historical_edges(sim_id: str) -> dict:
    """Compute per-feature win rates from trade history.

    Returns dict with regime, time_bucket, direction, ml_confidence_bucket stats.
    """
    try:
        trades = load_sim_trades(sim_id)
        if not trades:
            return {"sim_id": sim_id, "total_trades": 0, "insufficient_data": True}

        total = len(trades)

        def _is_win(t):
            pnl = t.get("realized_pnl_dollars")
            return isinstance(pnl, (int, float)) and pnl > 0

        def _group_stats(grouped: dict) -> dict:
            result = {}
            for key, bucket_trades in grouped.items():
                n = len(bucket_trades)
                wins = sum(1 for t in bucket_trades if _is_win(t))
                avg_pnl = (
                    sum(t.get("realized_pnl_dollars", 0) or 0 for t in bucket_trades) / n
                    if n > 0 else 0.0
                )
                result[key] = {
                    "count": n,
                    "wins": wins,
                    "win_rate": wins / n if n > 0 else 0.0,
                    "avg_pnl": avg_pnl,
                    "sufficient": n >= 5,
                }
            return result

        # ── Regime ──
        regime_groups: dict = {}
        for t in trades:
            r = t.get("regime_at_entry") or "UNKNOWN"
            regime_groups.setdefault(r, []).append(t)

        # ── Time bucket ──
        bucket_groups: dict = {}
        for t in trades:
            b = t.get("time_of_day_bucket") or "UNKNOWN"
            bucket_groups.setdefault(b, []).append(t)

        # ── Direction ──
        dir_groups: dict = {}
        for t in trades:
            d = t.get("direction") or "UNKNOWN"
            dir_groups.setdefault(d, []).append(t)

        # ── ML confidence buckets ──
        conf_groups: dict = {}
        for t in trades:
            grade = _trade_grade_compat(t)
            if grade is None:
                bkt = "unknown"
            elif grade < 0.45:
                bkt = "<0.45"
            elif grade < 0.50:
                bkt = "0.45-0.50"
            elif grade < 0.55:
                bkt = "0.50-0.55"
            elif grade < 0.60:
                bkt = "0.55-0.60"
            elif grade < 0.65:
                bkt = "0.60-0.65"
            elif grade < 0.70:
                bkt = "0.65-0.70"
            else:
                bkt = ">=0.70"
            conf_groups.setdefault(bkt, []).append(t)

        # Overall win rate
        all_wins = sum(1 for t in trades if _is_win(t))
        overall_win_rate = all_wins / total if total > 0 else 0.0
        overall_avg_pnl = (
            sum(t.get("realized_pnl_dollars", 0) or 0 for t in trades) / total
            if total > 0 else 0.0
        )

        return {
            "sim_id": sim_id,
            "total_trades": total,
            "overall_win_rate": overall_win_rate,
            "overall_avg_pnl": overall_avg_pnl,
            "insufficient_data": total < 15,
            "regime": _group_stats(regime_groups),
            "time_bucket": _group_stats(bucket_groups),
            "direction": _group_stats(dir_groups),
            "ml_confidence": _group_stats(conf_groups),
        }
    except Exception as exc:
        logger.debug("compute_historical_edges error sim=%s: %s", sim_id, exc)
        return {"sim_id": sim_id, "total_trades": 0, "insufficient_data": True, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Full sim analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sim_trades(sim_id: str) -> dict:
    """Run comprehensive_trade_grade on every closed trade for a sim."""
    try:
        trades = load_sim_trades(sim_id)
        if not trades:
            return {"sim_id": sim_id, "total_trades": 0, "grades": [], "grade_dist": {}}

        profile = _load_profile(sim_id)
        edges = compute_historical_edges(sim_id)
        bucket_win_rates = edges.get("time_bucket", {})

        graded = []
        for t in trades:
            g = comprehensive_trade_grade(t, profile=profile, bucket_win_rates=bucket_win_rates)
            g["trade_id"] = t.get("trade_id")
            g["realized_pnl_dollars"] = t.get("realized_pnl_dollars")
            g["signal_mode"] = t.get("signal_mode")
            g["regime_at_entry"] = t.get("regime_at_entry")
            g["time_of_day_bucket"] = t.get("time_of_day_bucket")
            graded.append(g)

        # Grade distribution
        grade_dist: dict = {}
        for g in graded:
            lg = g.get("letter_grade", "?")
            grade_dist[lg] = grade_dist.get(lg, 0) + 1

        # Per-grade win rates
        grade_pnl: dict = {}
        for g in graded:
            lg = g.get("letter_grade", "?")
            pnl = g.get("realized_pnl_dollars")
            if pnl is not None:
                grade_pnl.setdefault(lg, []).append(float(pnl))

        grade_stats = {}
        for lg, pnls in grade_pnl.items():
            wins = sum(1 for p in pnls if p > 0)
            grade_stats[lg] = {
                "count": len(pnls),
                "wins": wins,
                "win_rate": wins / len(pnls) if pnls else 0.0,
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            }

        # Worst dimensions (those with lowest average score across all trades)
        dim_totals: dict = {}
        dim_counts: dict = {}
        for g in graded:
            for dim, score in (g.get("dimensions") or {}).items():
                if score is not None:
                    dim_totals[dim] = dim_totals.get(dim, 0.0) + score
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1

        dim_avgs = {
            d: dim_totals[d] / dim_counts[d]
            for d in dim_totals
            if dim_counts.get(d, 0) > 0
        }
        worst_dimensions = sorted(dim_avgs.items(), key=lambda x: x[1])[:3]

        # Common flags
        all_flags: dict = {}
        for g in graded:
            for f in (g.get("flags") or []):
                all_flags[f] = all_flags.get(f, 0) + 1

        return {
            "sim_id": sim_id,
            "total_trades": len(trades),
            "grades": graded,
            "grade_dist": grade_dist,
            "grade_stats": grade_stats,
            "worst_dimensions": worst_dimensions,
            "dimension_averages": dim_avgs,
            "common_flags": all_flags,
            "historical_edges": edges,
        }
    except Exception as exc:
        logger.debug("analyze_sim_trades error sim=%s: %s", sim_id, exc)
        return {"sim_id": sim_id, "total_trades": 0, "grades": [], "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Filter generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_entry_filters(sim_id: str, min_confidence: float = 0.6) -> dict:
    """Full filter recommendation based on historical trade analysis."""
    try:
        analysis = analyze_sim_trades(sim_id)
        total = analysis.get("total_trades", 0)
        if total < 15:
            return {
                "sim_id": sim_id,
                "filters": {},
                "insufficient_data": True,
                "total_trades": total,
                "message": f"Need >= 15 closed trades; found {total}",
            }

        grades = analysis.get("grades", [])
        edges = analysis.get("historical_edges", {})
        overall_wr = edges.get("overall_win_rate", 0.5)

        # Split trades into A/B (good) vs D/F (bad)
        good_grades = frozenset({"A", "B"})
        bad_grades = frozenset({"D", "F"})

        good_trades = [g for g in grades if g.get("letter_grade") in good_grades]
        bad_trades = [g for g in grades if g.get("letter_grade") in bad_grades]

        filters: dict = {}
        filter_evidence: dict = {}

        # ── Regime filter ─────────────────────────────────────────────────────
        try:
            regime_info = edges.get("regime", {})
            # Only consider regimes with sufficient data
            sufficient_regimes = {
                r: info for r, info in regime_info.items()
                if info.get("sufficient", False)
            }

            if sufficient_regimes and len(good_trades) >= 5 and len(bad_trades) >= 5:
                def _regime_dist(trade_list):
                    counts: dict = {}
                    for g in trade_list:
                        r = g.get("regime_at_entry") or "UNKNOWN"
                        counts[r] = counts.get(r, 0) + 1
                    total_g = len(trade_list) or 1
                    return {r: c / total_g for r, c in counts.items()}

                good_regime_dist = _regime_dist(good_trades)
                bad_regime_dist = _regime_dist(bad_trades)

                # Regimes strongly associated with good trades
                good_regime_set = {
                    r for r, pct in good_regime_dist.items()
                    if pct > 0.6 and r in sufficient_regimes
                }
                # Regimes strongly associated with bad trades (and not in good set)
                bad_regime_set = {
                    r for r, pct in bad_regime_dist.items()
                    if pct > 0.6 and r in sufficient_regimes and r not in good_regime_set
                }

                if good_regime_set and bad_regime_set:
                    # Suggest whitelist if good set is meaningful
                    proposed = list(good_regime_set)
                    # Guard: don't filter >60% of trades
                    trades_in_proposed = sum(
                        regime_info.get(r, {}).get("count", 0) for r in proposed
                    )
                    if trades_in_proposed / total >= 0.40:  # keep at least 40%
                        filters["regime_whitelist"] = proposed
                        filter_evidence["regime_whitelist"] = {
                            "good_distribution": good_regime_dist,
                            "bad_distribution": bad_regime_dist,
                            "proposed": proposed,
                        }
        except Exception as _e:
            logger.debug("regime filter generation error: %s", _e)

        # ── Time bucket filter ────────────────────────────────────────────────
        try:
            bucket_info = edges.get("time_bucket", {})
            bad_buckets = []
            for bucket, info in bucket_info.items():
                if not info.get("sufficient", False):
                    continue
                bwr = info.get("win_rate", 0.5)
                # If overall WR > 50%: blacklist buckets with WR < 35%
                # If overall WR <= 50%: blacklist buckets with WR < 25%
                threshold = 0.35 if overall_wr > 0.50 else 0.25
                if bwr < threshold:
                    bad_buckets.append(bucket)

            if bad_buckets:
                # Guard: don't filter too many trades
                trades_in_bad = sum(
                    bucket_info.get(b, {}).get("count", 0) for b in bad_buckets
                )
                if trades_in_bad / total <= 0.60:
                    filters["time_bucket_blacklist"] = bad_buckets
                    filter_evidence["time_bucket_blacklist"] = {
                        "blacklisted": bad_buckets,
                        "overall_win_rate": overall_wr,
                    }
        except Exception as _e:
            logger.debug("time bucket filter generation error: %s", _e)

        # ── Direction bias ─────────────────────────────────────────────────────
        try:
            dir_info = edges.get("direction", {})
            bull = dir_info.get("BULLISH")
            bear = dir_info.get("BEARISH")
            if bull and bear and bull.get("count", 0) >= 10 and bear.get("count", 0) >= 10:
                bull_wr = bull.get("win_rate", 0.5)
                bear_wr = bear.get("win_rate", 0.5)
                diff = abs(bull_wr - bear_wr)
                if diff >= 0.20:
                    better_dir = "BULLISH" if bull_wr > bear_wr else "BEARISH"
                    worse_dir = "BEARISH" if better_dir == "BULLISH" else "BULLISH"
                    worse_count = dir_info[worse_dir].get("count", 0)
                    if worse_count >= 10:
                        # Guard: don't filter >60% of trades
                        if worse_count / total <= 0.60:
                            filters["direction_bias"] = better_dir
                            filter_evidence["direction_bias"] = {
                                "better": better_dir,
                                "worse": worse_dir,
                                "better_wr": bull_wr if better_dir == "BULLISH" else bear_wr,
                                "worse_wr": bear_wr if better_dir == "BULLISH" else bull_wr,
                                "diff": diff,
                            }
        except Exception as _e:
            logger.debug("direction bias filter generation error: %s", _e)

        # ── ML confidence filter ───────────────────────────────────────────────
        try:
            # Test thresholds: find where below-threshold WR < 35% and above > overall_wr
            all_conf_trades = []
            for t in load_sim_trades(sim_id):
                grade = _trade_grade_compat(t)
                pnl = t.get("realized_pnl_dollars")
                if grade is not None and pnl is not None:
                    all_conf_trades.append((grade, float(pnl) > 0))

            best_threshold = None
            best_evidence: dict = {}

            for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
                below = [(g, w) for g, w in all_conf_trades if g < thresh]
                above = [(g, w) for g, w in all_conf_trades if g >= thresh]
                if len(below) < 5 or len(above) < 5:
                    continue
                below_wr = sum(1 for _, w in below if w) / len(below)
                above_wr = sum(1 for _, w in above if w) / len(above)
                if below_wr < 0.35 and above_wr > overall_wr:
                    # Guard: don't filter >60%
                    if len(below) / total <= 0.60:
                        best_threshold = thresh
                        best_evidence = {
                            "threshold": thresh,
                            "below_count": len(below),
                            "below_wr": below_wr,
                            "above_count": len(above),
                            "above_wr": above_wr,
                        }
                        # Prefer the tightest threshold that passes
                        break

            if best_threshold is not None:
                filters["min_ml_confidence"] = best_threshold
                filter_evidence["min_ml_confidence"] = best_evidence
        except Exception as _e:
            logger.debug("ml confidence filter generation error: %s", _e)

        # ── Project impact ─────────────────────────────────────────────────────
        impact = _project_filter_impact(filters, sim_id, edges, overall_wr, total)

        return {
            "sim_id": sim_id,
            "filters": filters,
            "insufficient_data": False,
            "total_trades": total,
            "overall_win_rate": overall_wr,
            "analysis": {
                "grade_dist": analysis.get("grade_dist"),
                "grade_stats": analysis.get("grade_stats"),
                "worst_dimensions": analysis.get("worst_dimensions"),
                "common_flags": analysis.get("common_flags"),
                "dimension_averages": analysis.get("dimension_averages"),
            },
            "filter_evidence": filter_evidence,
            "projected_impact": impact,
        }
    except Exception as exc:
        logger.debug("generate_entry_filters error sim=%s: %s", sim_id, exc)
        return {"sim_id": sim_id, "filters": {}, "error": str(exc)}


def _project_filter_impact(
    filters: dict,
    sim_id: str,
    edges: dict,
    overall_wr: float,
    total: int,
) -> dict:
    """Estimate what the filters would do to trade count, WR, and expectancy."""
    try:
        if not filters or total == 0:
            return {}

        trades = load_sim_trades(sim_id)
        if not trades:
            return {}

        kept = []
        for t in trades:
            grade = _trade_grade_compat(t)
            regime = t.get("regime_at_entry") or "UNKNOWN"
            direction = t.get("direction") or "UNKNOWN"
            bucket = t.get("time_of_day_bucket") or "UNKNOWN"

            skip = False

            if "regime_whitelist" in filters:
                if regime not in filters["regime_whitelist"]:
                    skip = True

            if not skip and "time_bucket_blacklist" in filters:
                if bucket in filters["time_bucket_blacklist"]:
                    skip = True

            if not skip and "direction_bias" in filters:
                if direction != filters["direction_bias"]:
                    skip = True

            if not skip and "min_ml_confidence" in filters:
                if grade is None or grade < filters["min_ml_confidence"]:
                    skip = True

            if not skip:
                kept.append(t)

        if not kept:
            return {"trade_reduction_pct": 1.0, "warning": "all_trades_filtered"}

        kept_wins = sum(1 for t in kept if (t.get("realized_pnl_dollars") or 0) > 0)
        kept_wr = kept_wins / len(kept)
        kept_pnls = [t.get("realized_pnl_dollars") or 0 for t in kept]
        kept_expectancy = sum(kept_pnls) / len(kept_pnls) if kept_pnls else 0.0

        all_pnls = [t.get("realized_pnl_dollars") or 0 for t in trades]
        all_expectancy = sum(all_pnls) / len(all_pnls) if all_pnls else 0.0

        return {
            "original_count": total,
            "kept_count": len(kept),
            "trade_reduction_pct": (total - len(kept)) / total,
            "original_win_rate": overall_wr,
            "projected_win_rate": kept_wr,
            "original_expectancy": all_expectancy,
            "projected_expectancy": kept_expectancy,
        }
    except Exception as exc:
        logger.debug("_project_filter_impact error: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Real-time gate checks (fast, no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def check_quality_gate(
    quality_filters: dict,
    direction: Optional[str],
    regime: Optional[str],
    time_bucket: Optional[str],
    ml_grade: Optional[float],
    signal_mode: Optional[str] = None,
) -> Optional[str]:
    """Real-time entry gate. Fast, no I/O.

    Returns skip reason string if blocked, None if all pass.
    """
    try:
        if not quality_filters:
            return None

        # 1. regime_blacklist
        blacklist = quality_filters.get("regime_blacklist")
        if blacklist and regime and regime in blacklist:
            return f"regime_blacklisted:{regime}"

        # 2. regime_whitelist
        whitelist = quality_filters.get("regime_whitelist")
        if whitelist and regime and regime not in whitelist:
            return f"regime_not_whitelisted:{regime}"

        # 3. time_bucket_blacklist
        tb_blacklist = quality_filters.get("time_bucket_blacklist")
        if tb_blacklist and time_bucket and time_bucket in tb_blacklist:
            return f"time_bucket_blacklisted:{time_bucket}"

        # 4. time_bucket_whitelist
        tb_whitelist = quality_filters.get("time_bucket_whitelist")
        if tb_whitelist and time_bucket and time_bucket not in tb_whitelist:
            return f"time_bucket_not_whitelisted:{time_bucket}"

        # 5. direction_bias
        dir_bias = quality_filters.get("direction_bias")
        if dir_bias and direction and direction != dir_bias:
            return f"direction_filtered:{direction}"

        # 6. min_ml_confidence
        min_conf = quality_filters.get("min_ml_confidence")
        if min_conf is not None and ml_grade is not None:
            if ml_grade < float(min_conf):
                return f"ml_confidence_too_low:{ml_grade:.3f}"

        return None
    except Exception as exc:
        logger.debug("check_quality_gate error: %s", exc)
        return None


def check_post_contract_gate(
    quality_filters: dict,
    contract: dict,
    underlying_price: float,
) -> Optional[str]:
    """Check contract-level quality gates (OTM pct, spread). Fast, no I/O."""
    try:
        if not quality_filters or not contract:
            return None

        max_otm_pct = quality_filters.get("max_otm_pct")
        if max_otm_pct is not None:
            otm = contract.get("otm_pct")
            if otm is not None and abs(float(otm)) > float(max_otm_pct):
                return f"otm_too_high:{otm:.4f}"

        max_spread_pct = quality_filters.get("max_spread_pct")
        if max_spread_pct is not None and underlying_price and underlying_price > 0:
            bid = contract.get("bid") or 0
            ask = contract.get("ask") or 0
            spread = abs(ask - bid)
            spread_pct = spread / float(underlying_price)
            if spread_pct > float(max_spread_pct):
                return f"spread_too_wide:{spread_pct:.4f}"

        return None
    except Exception as exc:
        logger.debug("check_post_contract_gate error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Backfill grades
# ─────────────────────────────────────────────────────────────────────────────

def backfill_grades(sim_id: Optional[str] = None) -> None:
    """Grade all closed trades and write quality_grade back to trade dict. Idempotent."""
    try:
        import glob as _glob
        if sim_id:
            sim_ids = [sim_id]
        else:
            paths = _glob.glob(os.path.join(_SIM_DIR, "SIM*.json"))
            sim_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]

        for sid in sim_ids:
            try:
                path = os.path.join(_SIM_DIR, f"{sid}.json")
                if not os.path.exists(path):
                    continue
                with open(path, "r") as fh:
                    data = json.load(fh)

                profile = _load_profile(sid)
                edges = compute_historical_edges(sid)
                bucket_win_rates = edges.get("time_bucket", {})

                trade_log = data.get("trade_log", [])
                changed = 0
                for t in trade_log:
                    if not isinstance(t, dict):
                        continue
                    if t.get("exit_price") is None and t.get("realized_pnl_dollars") is None:
                        continue
                    g = comprehensive_trade_grade(t, profile=profile, bucket_win_rates=bucket_win_rates)
                    t["quality_grade"] = g.get("letter_grade")
                    t["quality_score"] = round(g.get("composite_score", 50.0), 2)
                    changed += 1

                if changed > 0:
                    data["trade_log"] = trade_log
                    with open(path, "w") as fh:
                        json.dump(data, fh, indent=2, default=str)
                    print(f"{sid}: graded {changed} closed trades")
            except Exception as exc:
                print(f"{sid}: backfill_grades error: {exc}")
    except Exception as exc:
        logger.debug("backfill_grades error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Analyze all sims
# ─────────────────────────────────────────────────────────────────────────────

def analyze_all_sims() -> dict:
    """Run generate_entry_filters for all sims with >= 15 closed trades."""
    try:
        import glob as _glob
        paths = _glob.glob(os.path.join(_SIM_DIR, "SIM*.json"))
        results = {}

        print(f"\n{'SIM':<8} {'Trades':>7} {'WR':>6} {'Filters':<40} {'Impact':>10}")
        print("-" * 75)

        for path in sorted(paths):
            sid = os.path.splitext(os.path.basename(path))[0]
            try:
                result = generate_entry_filters(sid)
                results[sid] = result

                total = result.get("total_trades", 0)
                if result.get("insufficient_data"):
                    print(f"{sid:<8} {total:>7}   —    insufficient data")
                    continue

                wr = result.get("overall_win_rate", 0)
                filters = result.get("filters", {})
                filter_str = ", ".join(filters.keys()) if filters else "none"
                impact = result.get("projected_impact", {})
                reduction = impact.get("trade_reduction_pct")
                impact_str = f"-{reduction*100:.0f}% trades" if reduction is not None else ""
                print(f"{sid:<8} {total:>7} {wr*100:>5.1f}%  {filter_str:<40} {impact_str:>10}")
            except Exception as exc:
                print(f"{sid:<8}  error: {exc}")

        return results
    except Exception as exc:
        logger.debug("analyze_all_sims error: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# apply_filters_to_config helper
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters_to_config(sim_id: str, filters: dict, dry_run: bool = True) -> bool:
    """Write quality_filters to sim_config.yaml for a given sim."""
    try:
        import yaml
        with open(_CONFIG_PATH, "r") as fh:
            raw = fh.read()
        cfg = yaml.safe_load(raw) or {}

        if sim_id not in cfg:
            print(f"ERROR: {sim_id} not found in sim_config.yaml")
            return False

        if not filters:
            # Remove quality_filters if present
            if "quality_filters" in cfg[sim_id]:
                del cfg[sim_id]["quality_filters"]
                action = "removed quality_filters"
            else:
                print(f"{sim_id}: no quality_filters to remove")
                return True
        else:
            cfg[sim_id]["quality_filters"] = filters
            action = f"set quality_filters: {list(filters.keys())}"

        if dry_run:
            print(f"[dry-run] Would {action} for {sim_id}")
            return True

        with open(_CONFIG_PATH, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"{sim_id}: {action}")
        return True
    except Exception as exc:
        print(f"apply_filters_to_config error: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trade quality analyzer")
    parser.add_argument("--sim", help="Analyze single sim (e.g. SIM03)")
    parser.add_argument("--all", action="store_true", help="Analyze all sims")
    parser.add_argument("--grade", help="Grade a single sim's trades (e.g. SIM03)")
    parser.add_argument("--apply", action="store_true", help="Write filters to sim_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--min-grade", default="C", help="Minimum grade to keep (unused, future)")
    parser.add_argument("--backfill", action="store_true", help="Backfill grades to trade dicts")
    args = parser.parse_args()

    if args.backfill:
        backfill_grades(args.sim)

    elif args.grade:
        sid = args.grade.upper()
        result = analyze_sim_trades(sid)
        total = result.get("total_trades", 0)
        print(f"\n=== {sid} Trade Analysis — {total} closed trades ===")
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            dist = result.get("grade_dist", {})
            print("\nGrade distribution:")
            for g in ["A", "B", "C", "D", "F"]:
                count = dist.get(g, 0)
                bar = "#" * count
                print(f"  {g}: {count:3d}  {bar}")

            stats = result.get("grade_stats", {})
            print("\nGrade performance:")
            for g in ["A", "B", "C", "D", "F"]:
                s = stats.get(g)
                if s:
                    print(f"  {g}: WR={s['win_rate']*100:.1f}%  avg_pnl=${s['avg_pnl']:.2f}")

            worst = result.get("worst_dimensions", [])
            if worst:
                print("\nWorst dimensions (lowest avg score):")
                for dim, score in worst:
                    print(f"  {dim}: {score:.1f}")

            flags = result.get("common_flags", {})
            if flags:
                print("\nCommon flags:")
                for f, count in sorted(flags.items(), key=lambda x: -x[1]):
                    print(f"  {f}: {count}")

    elif args.all:
        analyze_all_sims()

    elif args.sim:
        sid = args.sim.upper()
        result = generate_entry_filters(sid, args.min_confidence)
        total = result.get("total_trades", 0)
        print(f"\n=== {sid} Entry Filter Recommendations — {total} closed trades ===")

        if result.get("insufficient_data"):
            print(result.get("message", "Insufficient data"))
        elif result.get("error"):
            print(f"Error: {result['error']}")
        else:
            wr = result.get("overall_win_rate", 0)
            print(f"Overall win rate: {wr*100:.1f}%")

            filters = result.get("filters", {})
            if not filters:
                print("No filters recommended (insufficient evidence).")
            else:
                print("\nRecommended filters:")
                for k, v in filters.items():
                    print(f"  {k}: {v}")

                impact = result.get("projected_impact", {})
                if impact:
                    print(f"\nProjected impact:")
                    print(f"  Trades kept: {impact.get('kept_count')}/{impact.get('original_count')} "
                          f"({(1-impact.get('trade_reduction_pct',0))*100:.0f}%)")
                    print(f"  Win rate: {impact.get('original_win_rate',0)*100:.1f}% → "
                          f"{impact.get('projected_win_rate',0)*100:.1f}%")
                    print(f"  Expectancy: ${impact.get('original_expectancy',0):.2f} → "
                          f"${impact.get('projected_expectancy',0):.2f}")

                if args.apply or args.dry_run:
                    apply_filters_to_config(sid, filters, dry_run=not args.apply)

            analysis = result.get("analysis", {})
            worst = analysis.get("worst_dimensions", [])
            if worst:
                print("\nWorst dimensions:")
                for dim, score in worst:
                    print(f"  {dim}: {score:.1f}")
    else:
        parser.print_help()
