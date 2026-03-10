"""
analytics/composite_score.py

Composite ranking score for simulation profiles.
Computes a 0-100 score from 5 weighted components.
No Discord dependencies — importable standalone.
"""
from __future__ import annotations

from datetime import datetime

# ── Scoring thresholds (tune here) ───────────────────────────────────────────
PROFITABILITY_MIDPOINT   = 0.00   # 0 % return  → 50 pts
PROFITABILITY_MAX_RETURN = 0.20   # +20% return → 100 pts

WIN_RATE_ZERO_PTS = 0.40          # ≤ 40% WR → 0 pts
WIN_RATE_FULL_PTS = 0.60          # ≥ 60% WR → 100 pts

PF_ZERO_PTS  = 0.50               # PF ≤ 0.50 → 0 pts
PF_MID_PTS   = 1.00               # PF  1.00  → 30 pts
PF_FULL_PTS  = 2.00               # PF ≥ 2.00 → 100 pts
PF_MID_SCORE = 30.0

CONSISTENCY_ZERO_PCT = 0.30       # 30 % green days → 0 pts
CONSISTENCY_FULL_PCT = 0.70       # 70 % green days → 100 pts

DD_FULL_PCT = 0.50                # 50 % max DD → 0 pts

MIN_TRADES_FOR_RANKING = 10

WEIGHTS: dict[str, float] = {
    "profitability": 0.25,
    "win_rate":      0.20,
    "risk_adjusted": 0.25,
    "consistency":   0.20,
    "drawdown":      0.10,
}

# (threshold, grade, emoji)
GRADE_THRESHOLDS = [
    (90, "A+", "🏆"),
    (80, "A",  "⭐"),
    (70, "B",  "📈"),
    (60, "C",  "📊"),
    (50, "D",  "📉"),
    (0,  "F",  "💀"),
]


# ── Component scorers ─────────────────────────────────────────────────────────

def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def _score_profitability(return_pct: float) -> float:
    """return_pct as fraction, e.g. 0.12 for 12 % gain."""
    slope = 50.0 / PROFITABILITY_MAX_RETURN
    return _clamp(50.0 + return_pct * slope)


def _score_win_rate(win_rate: float) -> float:
    rng = WIN_RATE_FULL_PTS - WIN_RATE_ZERO_PTS
    return _clamp((win_rate - WIN_RATE_ZERO_PTS) / rng * 100.0)


def _score_risk_adjusted(pf: float) -> float:
    """Piecewise linear: 0.5→0, 1.0→30, 2.0→100."""
    if pf >= PF_FULL_PTS:
        return 100.0
    if pf >= PF_MID_PTS:
        slope = (100.0 - PF_MID_SCORE) / (PF_FULL_PTS - PF_MID_PTS)
        return _clamp(PF_MID_SCORE + (pf - PF_MID_PTS) * slope)
    if pf >= PF_ZERO_PTS:
        slope = PF_MID_SCORE / (PF_MID_PTS - PF_ZERO_PTS)
        return _clamp((pf - PF_ZERO_PTS) * slope)
    return 0.0


def _score_consistency(green_day_pct: float) -> float:
    rng = CONSISTENCY_FULL_PCT - CONSISTENCY_ZERO_PCT
    return _clamp((green_day_pct - CONSISTENCY_ZERO_PCT) / rng * 100.0)


def _score_drawdown(max_dd_pct: float) -> float:
    """max_dd_pct as fraction, e.g. 0.10 for 10 % drawdown."""
    return _clamp((1.0 - max_dd_pct / DD_FULL_PCT) * 100.0)


def _letter_grade(score: float) -> tuple[str, str]:
    for threshold, grade, emoji in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade, emoji
    return "F", "💀"


def _compute_green_day_pct(trade_log: list) -> float:
    """Fraction of trading days where total closed PnL > 0."""
    day_pnl: dict[str, float] = {}
    for t in trade_log:
        if not isinstance(t, dict):
            continue
        try:
            pnl = float(t.get("realized_pnl_dollars") or 0)
        except (TypeError, ValueError):
            continue
        exit_time = t.get("exit_time") or t.get("entry_time")
        try:
            day = str(exit_time)[:10]
        except Exception:
            continue
        day_pnl[day] = day_pnl.get(day, 0.0) + pnl
    if not day_pnl:
        return 0.0
    green = sum(1 for v in day_pnl.values() if v > 0)
    return green / len(day_pnl)


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_composite_score(sim_id: str, profile: dict | None = None) -> dict:
    """
    Compute composite score (0-100) for a simulation.

    Parameters
    ----------
    sim_id  : e.g. "SIM05"
    profile : sim config dict; loaded from sim_config.yaml if None

    Returns dict with:
      sim_id, unranked, unranked_reason, composite_score, grade, emoji,
      components (per-component raw/score/weight), total_trades, days_active,
      return_pct, win_rate, profit_factor, green_day_pct, max_dd_pct
    """
    import os
    from simulation.sim_portfolio import SimPortfolio

    if profile is None:
        import yaml
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(base, "simulation", "sim_config.yaml")
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            profile = cfg.get(sim_id, {})
        except Exception:
            profile = {}

    try:
        sim = SimPortfolio(sim_id, profile)
        sim.load()
    except Exception:
        return _unranked(sim_id, 0, "load_error")

    trade_log   = sim.trade_log if isinstance(sim.trade_log, list) else []
    total_trades = len(trade_log)

    if total_trades < MIN_TRADES_FOR_RANKING:
        return _unranked(
            sim_id, total_trades,
            f"insufficient_trades ({total_trades} < {MIN_TRADES_FOR_RANKING})",
        )

    # ── Compute raw metrics ───────────────────────────────────────────────────
    pnl_vals: list[float] = []
    for t in trade_log:
        try:
            pnl_vals.append(float(t.get("realized_pnl_dollars") or 0))
        except (TypeError, ValueError):
            pass

    total_pnl  = sum(pnl_vals)
    wins       = sum(1 for p in pnl_vals if p > 0)
    win_rate   = wins / total_trades
    win_sum    = sum(p for p in pnl_vals if p > 0)
    loss_sum   = abs(sum(p for p in pnl_vals if p < 0))
    # If no losses, treat PF as 2.0 (capped best); if no wins and no losses, 1.0
    pf = win_sum / loss_sum if loss_sum > 0 else (2.0 if win_sum > 0 else 1.0)

    start_balance = float(profile.get("balance_start") or 500.0)
    return_pct    = total_pnl / start_balance if start_balance > 0 else 0.0

    peak_balance  = float(getattr(sim, "peak_balance", start_balance) or start_balance)
    balance       = float(sim.balance or start_balance)
    max_dd_dollars = max(0.0, peak_balance - balance)
    max_dd_pct    = max_dd_dollars / start_balance if start_balance > 0 else 0.0

    green_day_pct = _compute_green_day_pct(trade_log)

    # Days active
    times: list[datetime] = []
    for t in trade_log:
        try:
            ts_str = t.get("exit_time") or t.get("entry_time")
            if ts_str:
                times.append(datetime.fromisoformat(str(ts_str)))
        except Exception:
            pass
    days_active = (
        max((max(times) - min(times)).total_seconds() / 86400.0, 1 / 24)
        if len(times) >= 2 else 1.0
    )

    # ── Score each component ──────────────────────────────────────────────────
    s_profit  = _score_profitability(return_pct)
    s_wr      = _score_win_rate(win_rate)
    s_ra      = _score_risk_adjusted(pf)
    s_consist = _score_consistency(green_day_pct)
    s_dd      = _score_drawdown(max_dd_pct)

    composite = round(
        s_profit  * WEIGHTS["profitability"] +
        s_wr      * WEIGHTS["win_rate"] +
        s_ra      * WEIGHTS["risk_adjusted"] +
        s_consist * WEIGHTS["consistency"] +
        s_dd      * WEIGHTS["drawdown"],
        1,
    )

    grade, emoji = _letter_grade(composite)

    return {
        "sim_id":          sim_id,
        "unranked":        False,
        "unranked_reason": None,
        "composite_score": composite,
        "grade":           grade,
        "emoji":           emoji,
        "components": {
            "profitability": {
                "raw": round(return_pct * 100, 2),
                "score": round(s_profit, 1),
                "weight": WEIGHTS["profitability"],
            },
            "win_rate": {
                "raw": round(win_rate * 100, 2),
                "score": round(s_wr, 1),
                "weight": WEIGHTS["win_rate"],
            },
            "risk_adjusted": {
                "raw": round(pf, 3),
                "score": round(s_ra, 1),
                "weight": WEIGHTS["risk_adjusted"],
            },
            "consistency": {
                "raw": round(green_day_pct * 100, 1),
                "score": round(s_consist, 1),
                "weight": WEIGHTS["consistency"],
            },
            "drawdown": {
                "raw": round(max_dd_pct * 100, 2),
                "score": round(s_dd, 1),
                "weight": WEIGHTS["drawdown"],
            },
        },
        "total_trades":  total_trades,
        "days_active":   round(days_active, 1),
        "return_pct":    round(return_pct * 100, 2),
        "win_rate":      round(win_rate * 100, 2),
        "profit_factor": round(pf, 3),
        "green_day_pct": round(green_day_pct * 100, 1),
        "max_dd_pct":    round(max_dd_pct * 100, 2),
        "signal_mode":   profile.get("signal_mode", ""),
    }


def _unranked(sim_id: str, total_trades: int, reason: str) -> dict:
    return {
        "sim_id":          sim_id,
        "unranked":        True,
        "unranked_reason": reason,
        "composite_score": None,
        "grade":           None,
        "emoji":           None,
        "components":      {},
        "total_trades":    total_trades,
        "days_active":     0,
        "return_pct":      None,
        "win_rate":        None,
        "profit_factor":   None,
        "green_day_pct":   None,
        "max_dd_pct":      None,
        "signal_mode":     "",
    }
