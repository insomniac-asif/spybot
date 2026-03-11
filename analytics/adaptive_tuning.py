"""
analytics/adaptive_tuning.py

Adaptive Greeks exit threshold tuning per-sim based on rolling trade results.
Stores overrides in data/adaptive_overrides.json (never modifies sim_config.yaml).
Logs all changes to data/adaptive_tuning_log.json.

No Discord dependencies -- importable standalone.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import pytz

from core.paths import DATA_DIR
from simulation.sim_portfolio import SimPortfolio

_ET = pytz.timezone("US/Eastern")

_OVERRIDES_PATH = os.path.join(DATA_DIR, "adaptive_overrides.json")
_TUNING_LOG_PATH = os.path.join(DATA_DIR, "adaptive_tuning_log.json")

# Greeks exit reason sets
_GREEKS_THETA = {"theta_burn", "theta_burn_0dte", "theta_burn_tightened"}
_GREEKS_IV = {"iv_crush_stop", "iv_crush_exit"}
_GREEKS_DELTA = {"delta_erosion"}

# Tuning thresholds
MIN_EXITS_FOR_TUNING = 5
SAVED_MONEY_THRESHOLD = 0.70    # >= 70% saved -> tighten
CUT_WINNERS_THRESHOLD = 0.50    # >= 50% cut winners -> loosen

# Hard floors / ceilings
THETA_BURN_MIN = 0.25
THETA_BURN_MAX = 0.70
THETA_BURN_STEP = 0.05

IV_CRUSH_MULT_MIN = 1.2
IV_CRUSH_MULT_MAX = 3.0
IV_CRUSH_MULT_STEP = 0.2

DELTA_EROSION_MIN = 0.10
DELTA_EROSION_MAX = 0.30
DELTA_EROSION_STEP = 0.03

# Composite score regression threshold for auto-revert
SCORE_REGRESSION_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Override storage
# ---------------------------------------------------------------------------

def _load_overrides() -> dict:
    if os.path.exists(_OVERRIDES_PATH):
        try:
            with open(_OVERRIDES_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_overrides(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_OVERRIDES_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_tuning_log() -> list:
    if os.path.exists(_TUNING_LOG_PATH):
        try:
            with open(_TUNING_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_tuning_log(entries: list) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    # Keep last 500 entries to prevent unbounded growth
    with open(_TUNING_LOG_PATH, "w") as f:
        json.dump(entries[-500:], f, indent=2, default=str)


def _log_change(sim_id: str, trigger: str, field: str, old_val, new_val, reason: str) -> None:
    entries = _load_tuning_log()
    entries.append({
        "timestamp": datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S"),
        "sim_id": sim_id,
        "trigger": trigger,
        "field": field,
        "old_value": old_val,
        "new_value": new_val,
        "reason": reason,
    })
    _save_tuning_log(entries)


# ---------------------------------------------------------------------------
# Effective threshold resolution: adaptive override > sim config > default
# ---------------------------------------------------------------------------

def get_effective_threshold(sim_id: str, profile: dict, key: str, default) -> float:
    """Get effective threshold: adaptive override > sim config > default."""
    overrides = _load_overrides()
    sim_overrides = overrides.get(sim_id, {}).get("thresholds", {})
    if key in sim_overrides:
        return float(sim_overrides[key])
    return float(profile.get(key, default))


# ---------------------------------------------------------------------------
# Greeks exit effectiveness analysis
# ---------------------------------------------------------------------------

def evaluate_greeks_effectiveness(sim_id: str, profile: dict, lookback_trades: int = 20) -> dict:
    """
    Analyze the last N trades where each Greeks trigger fired.
    Returns per-trigger stats.
    """
    try:
        sim = SimPortfolio(sim_id, profile)
        sim.load()
    except Exception:
        return {}

    trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []
    if not trade_log:
        return {}

    results = {}

    for trigger_name, reason_set in [
        ("theta_burn", _GREEKS_THETA),
        ("iv_crush", _GREEKS_IV),
        ("delta_erosion", _GREEKS_DELTA),
    ]:
        # Get trades where this trigger fired (most recent first)
        trigger_trades = [
            t for t in reversed(trade_log)
            if t.get("exit_reason", "") in reason_set
        ][:lookback_trades]

        if not trigger_trades:
            results[trigger_name] = {
                "count": 0,
                "enabled": _is_trigger_enabled(profile, trigger_name),
                "status": "no_data",
            }
            continue

        # Analyze: was the exit "saving money" or "cutting a winner"?
        saved_count = 0
        cut_winner_count = 0
        pnls = []

        for t in trigger_trades:
            pnl = _safe_float(t.get("realized_pnl_dollars"))
            pnls.append(pnl)

            # A Greeks exit "saved money" if the trade was already losing
            # (exit prevented further loss)
            if pnl <= 0:
                saved_count += 1
            else:
                cut_winner_count += 1

        total = len(trigger_trades)
        saved_pct = saved_count / total if total > 0 else 0
        avg_pnl = sum(pnls) / total if total > 0 else 0

        # Check how many were near the stop loss anyway
        near_stop = 0
        for t in trigger_trades:
            stop_loss_pct = float(profile.get("stop_loss_pct", 0.40))
            pnl_pct = _safe_float(t.get("realized_pnl_pct"))
            if pnl_pct <= 0 and abs(pnl_pct) >= stop_loss_pct * 0.8:
                near_stop += 1

        results[trigger_name] = {
            "count": total,
            "enabled": _is_trigger_enabled(profile, trigger_name),
            "saved_count": saved_count,
            "cut_winner_count": cut_winner_count,
            "saved_pct": round(saved_pct * 100, 1),
            "avg_pnl": round(avg_pnl, 2),
            "near_stop_count": near_stop,
            "status": "sufficient" if total >= MIN_EXITS_FOR_TUNING else "insufficient",
        }

    return results


def _is_trigger_enabled(profile: dict, trigger_name: str) -> bool:
    mapping = {
        "theta_burn": "theta_burn_enabled",
        "iv_crush": "iv_crush_exit_enabled",
        "delta_erosion": "delta_erosion_exit_enabled",
    }
    return bool(profile.get(mapping.get(trigger_name, ""), False))


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Tuning logic
# ---------------------------------------------------------------------------

def run_adaptive_tuning(sim_id: str, profile: dict) -> list[dict]:
    """
    Run adaptive tuning for a single sim. Returns list of changes made.
    SIM00 is NEVER auto-tuned.
    """
    if sim_id == "SIM00":
        return []

    if not profile.get("adaptive_tuning_enabled", False):
        # Check global default
        import yaml
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(base, "simulation", "sim_config.yaml")
        try:
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}
            global_cfg = raw.get("_global", {})
            if not global_cfg.get("adaptive_tuning_enabled", False):
                return []
        except Exception:
            return []

    effectiveness = evaluate_greeks_effectiveness(sim_id, profile)
    if not effectiveness:
        return []

    overrides = _load_overrides()
    sim_data = overrides.setdefault(sim_id, {
        "thresholds": {},
        "original_thresholds": {},
        "last_tuning_date": None,
        "baseline_composite_score": None,
    })

    # Max 1 adjustment per trigger per day
    today = datetime.now(_ET).date().isoformat()
    if sim_data.get("last_tuning_date") == today:
        return []

    changes = []

    for trigger_name, stats in effectiveness.items():
        if not stats.get("enabled") or stats.get("status") != "sufficient":
            continue

        saved_pct = stats.get("saved_pct", 0) / 100.0
        cut_pct = stats.get("cut_winner_count", 0) / max(stats.get("count", 1), 1)

        if trigger_name == "theta_burn":
            key = "theta_burn_stop_tighten_pct"
            current = get_effective_threshold(sim_id, profile, key, 0.50)

            # Store original if not already
            if key not in sim_data.get("original_thresholds", {}):
                sim_data.setdefault("original_thresholds", {})[key] = current

            if saved_pct >= SAVED_MONEY_THRESHOLD:
                new_val = max(THETA_BURN_MIN, round(current - THETA_BURN_STEP, 2))
                if new_val != current:
                    sim_data["thresholds"][key] = new_val
                    reason = f"saved {stats['saved_pct']}% -> tighten"
                    _log_change(sim_id, trigger_name, key, current, new_val, reason)
                    changes.append({"trigger": trigger_name, "field": key, "old": current, "new": new_val, "reason": reason})

            elif cut_pct >= CUT_WINNERS_THRESHOLD:
                new_val = min(THETA_BURN_MAX, round(current + THETA_BURN_STEP, 2))
                if new_val != current:
                    sim_data["thresholds"][key] = new_val
                    reason = f"cut winners {cut_pct*100:.0f}% -> loosen"
                    _log_change(sim_id, trigger_name, key, current, new_val, reason)
                    changes.append({"trigger": trigger_name, "field": key, "old": current, "new": new_val, "reason": reason})

        elif trigger_name == "iv_crush":
            key = "iv_crush_vega_multiplier"
            current = get_effective_threshold(sim_id, profile, key, 2.0)

            if key not in sim_data.get("original_thresholds", {}):
                sim_data.setdefault("original_thresholds", {})[key] = current

            if saved_pct >= SAVED_MONEY_THRESHOLD:
                new_val = max(IV_CRUSH_MULT_MIN, round(current - IV_CRUSH_MULT_STEP, 1))
                if new_val != current:
                    sim_data["thresholds"][key] = new_val
                    reason = f"saved {stats['saved_pct']}% -> tighten"
                    _log_change(sim_id, trigger_name, key, current, new_val, reason)
                    changes.append({"trigger": trigger_name, "field": key, "old": current, "new": new_val, "reason": reason})

            elif cut_pct >= CUT_WINNERS_THRESHOLD:
                new_val = min(IV_CRUSH_MULT_MAX, round(current + IV_CRUSH_MULT_STEP, 1))
                if new_val != current:
                    sim_data["thresholds"][key] = new_val
                    reason = f"cut winners {cut_pct*100:.0f}% -> loosen"
                    _log_change(sim_id, trigger_name, key, current, new_val, reason)
                    changes.append({"trigger": trigger_name, "field": key, "old": current, "new": new_val, "reason": reason})

        elif trigger_name == "delta_erosion":
            key = "delta_erosion_current_max"
            current = get_effective_threshold(sim_id, profile, key, 0.20)

            if key not in sim_data.get("original_thresholds", {}):
                sim_data.setdefault("original_thresholds", {})[key] = current

            if saved_pct >= SAVED_MONEY_THRESHOLD:
                # Tighten = increase max (triggers more easily)
                new_val = min(DELTA_EROSION_MAX, round(current + DELTA_EROSION_STEP, 2))
                if new_val != current:
                    sim_data["thresholds"][key] = new_val
                    reason = f"saved {stats['saved_pct']}% -> tighten"
                    _log_change(sim_id, trigger_name, key, current, new_val, reason)
                    changes.append({"trigger": trigger_name, "field": key, "old": current, "new": new_val, "reason": reason})

            elif cut_pct >= CUT_WINNERS_THRESHOLD:
                # Loosen = decrease max (triggers less easily)
                new_val = max(DELTA_EROSION_MIN, round(current - DELTA_EROSION_STEP, 2))
                if new_val != current:
                    sim_data["thresholds"][key] = new_val
                    reason = f"cut winners {cut_pct*100:.0f}% -> loosen"
                    _log_change(sim_id, trigger_name, key, current, new_val, reason)
                    changes.append({"trigger": trigger_name, "field": key, "old": current, "new": new_val, "reason": reason})

    if changes:
        sim_data["last_tuning_date"] = today
    _save_overrides(overrides)

    return changes


def check_score_regression(sim_id: str, profile: dict) -> bool:
    """
    Check if composite score has dropped > threshold after adaptive changes.
    If so, revert to original thresholds and return True.
    """
    overrides = _load_overrides()
    sim_data = overrides.get(sim_id)
    if not sim_data or not sim_data.get("thresholds"):
        return False

    try:
        from analytics.composite_score import compute_composite_score
        current = compute_composite_score(sim_id, profile)
        current_score = current.get("composite_score")
        if current_score is None:
            return False

        baseline = sim_data.get("baseline_composite_score")
        if baseline is None:
            # Set baseline on first check
            sim_data["baseline_composite_score"] = current_score
            _save_overrides(overrides)
            return False

        if baseline - current_score > SCORE_REGRESSION_THRESHOLD:
            # Revert to original thresholds
            originals = sim_data.get("original_thresholds", {})
            for key, val in originals.items():
                _log_change(sim_id, "regression_revert", key, sim_data["thresholds"].get(key, "?"), val,
                           f"score dropped {baseline:.1f} -> {current_score:.1f}")
            sim_data["thresholds"] = {}
            sim_data["baseline_composite_score"] = None
            _save_overrides(overrides)
            logging.warning("adaptive_tuning_reverted: %s score_drop=%.1f", sim_id, baseline - current_score)
            return True
    except Exception as e:
        logging.warning("adaptive_tuning_regression_check_error: %s %s", sim_id, e)

    return False


def run_all_adaptive_tuning() -> dict:
    """
    Run adaptive tuning for all eligible sims. Returns {sim_id: [changes]}.
    """
    import yaml
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base, "simulation", "sim_config.yaml")
    try:
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        profiles = {k: v for k, v in raw.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return {}

    all_changes = {}
    for sim_id, profile in sorted(profiles.items()):
        if sim_id == "SIM00":
            continue
        try:
            # Check regression first
            check_score_regression(sim_id, profile)
            # Then run tuning
            changes = run_adaptive_tuning(sim_id, profile)
            if changes:
                all_changes[sim_id] = changes
        except Exception as e:
            logging.warning("adaptive_tuning_error: %s %s", sim_id, e)

    return all_changes


def get_tuning_status(sim_id: str) -> dict:
    """Get current adaptive tuning status for a sim (for Discord command)."""
    import yaml
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base, "simulation", "sim_config.yaml")
    try:
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        profile = raw.get(sim_id, {})
    except Exception:
        profile = {}

    overrides = _load_overrides()
    sim_data = overrides.get(sim_id, {})
    thresholds = sim_data.get("thresholds", {})
    originals = sim_data.get("original_thresholds", {})
    last_date = sim_data.get("last_tuning_date", "never")

    effectiveness = evaluate_greeks_effectiveness(sim_id, profile)

    triggers = {}
    for trigger_name, cfg_key in [
        ("theta_burn", "theta_burn_stop_tighten_pct"),
        ("iv_crush", "iv_crush_vega_multiplier"),
        ("delta_erosion", "delta_erosion_current_max"),
    ]:
        enabled = _is_trigger_enabled(profile, trigger_name)
        if not enabled:
            triggers[trigger_name] = {"status": "DISABLED", "enabled": False}
            continue

        current_val = get_effective_threshold(sim_id, profile, cfg_key, None)
        original_val = originals.get(cfg_key)
        override_val = thresholds.get(cfg_key)

        stats = effectiveness.get(trigger_name, {})

        if override_val is not None and original_val is not None and override_val != original_val:
            if override_val < original_val:
                status = "TIGHTENED"
            else:
                status = "LOOSENED"
        else:
            status = "UNCHANGED"

        triggers[trigger_name] = {
            "status": status,
            "enabled": True,
            "current_value": current_val,
            "original_value": original_val,
            "exit_count": stats.get("count", 0),
            "saved_pct": stats.get("saved_pct", 0),
        }

    return {
        "sim_id": sim_id,
        "triggers": triggers,
        "last_tuning_date": last_date,
        "adaptive_enabled": profile.get("adaptive_tuning_enabled", False),
    }
