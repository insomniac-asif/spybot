"""
simulation/strategy_evaluator.py

Ranks signal_modes by recent performance across all sims.
Used by SIM09's OpportunityRanker to make data-driven strategy selection,
and by the dashboard for strategy leaderboard display.

Usage:
    from simulation.strategy_evaluator import evaluate_strategies, get_sim_states_for_ranker
    ranking = evaluate_strategies(lookback_days=14)
    sim_states = get_sim_states_for_ranker()
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from statistics import stdev

import pytz

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SIM_DIR = os.path.join(DATA_DIR, "sims")
RANKINGS_PATH = os.path.join(DATA_DIR, "strategy_rankings.json")

# Scoring weights (sum to 1.0)
W_WINRATE = 0.25
W_EXPECTANCY = 0.30
W_PROFIT_FACTOR = 0.20
W_CONSISTENCY = 0.15
W_REGIME_FIT = 0.10

# Cache to avoid repeated file I/O during entry loop
_cache: dict = {}
_cache_time: float = 0.0
_CACHE_TTL = 60.0  # seconds


def _get_current_regime() -> str:
    try:
        from signals.regime import get_regime
        from core.data_service import get_market_dataframe
        df = get_market_dataframe()
        if df is not None and len(df) > 30:
            return str(get_regime(df) or "UNKNOWN").upper()
    except Exception:
        pass
    return "UNKNOWN"


def _load_sim_configs() -> dict:
    """Load sim configs from YAML."""
    try:
        import yaml
        config_path = os.path.join(BASE_DIR, "simulation", "sim_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return {k: v for k, v in config.items() if k.startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return {}


def _load_all_trade_logs() -> dict:
    """Load trade_log from every sim JSON file. Returns {sim_id: {signal_mode, trade_log}}."""
    configs = _load_sim_configs()
    result = {}
    if not os.path.isdir(SIM_DIR):
        return result
    for sim_id, profile in configs.items():
        path = os.path.join(SIM_DIR, f"{sim_id}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            trade_log = data.get("trade_log", [])
            if not isinstance(trade_log, list):
                trade_log = []
            result[sim_id] = {
                "signal_mode": str(profile.get("signal_mode", "")).upper(),
                "trade_log": trade_log,
            }
        except Exception:
            continue
    return result


def get_sim_states_for_ranker() -> dict:
    """Build sim_states dict that OpportunityRanker._historical_score() expects."""
    return _load_all_trade_logs()


def _load_perf_store_data() -> dict:
    """Load aggregated strategy_performance.json data."""
    try:
        from analytics.strategy_performance import PERF_STORE
        PERF_STORE._load()
        return dict(PERF_STORE._data)
    except Exception:
        perf_path = os.path.join(DATA_DIR, "strategy_performance.json")
        if os.path.exists(perf_path):
            try:
                with open(perf_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _aggregate_mode_from_perf(mode: str, perf_data: dict) -> dict:
    """Aggregate stats for a signal_mode from strategy_performance.json."""
    mode_data = perf_data.get(mode, {})
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    pnl_list = []
    regime_pnl: dict[str, float] = {}
    regime_trades: dict[str, int] = {}

    for regime, buckets in mode_data.items():
        if not isinstance(buckets, dict):
            continue
        for bucket_data in buckets.values():
            if not isinstance(bucket_data, dict):
                continue
            n = int(bucket_data.get("trades", 0))
            w = int(bucket_data.get("wins", 0))
            pnl = float(bucket_data.get("total_pnl", 0.0))
            total_trades += n
            total_wins += w
            total_pnl += pnl
            regime_pnl[regime] = regime_pnl.get(regime, 0.0) + pnl
            regime_trades[regime] = regime_trades.get(regime, 0) + n
            # Approximate individual trade PnL from aggregates
            if n > 0:
                avg_pnl = pnl / n
                pnl_list.extend([avg_pnl] * n)

    return {
        "trades": total_trades,
        "wins": total_wins,
        "total_pnl": total_pnl,
        "pnl_list": pnl_list,
        "regime_pnl": regime_pnl,
        "regime_trades": regime_trades,
    }


def _aggregate_mode_from_logs(mode: str, sim_states: dict, lookback_days: int) -> dict:
    """Aggregate stats for a signal_mode from sim trade_logs."""
    cutoff = datetime.now(pytz.timezone("US/Eastern")) - timedelta(days=lookback_days)
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    pnl_list = []
    regime_pnl: dict[str, float] = {}
    regime_trades: dict[str, int] = {}
    best_sim = None
    best_sim_wr = -1.0
    sim_wr: dict[str, tuple[int, int]] = {}

    for sim_id, state in sim_states.items():
        if state.get("signal_mode", "").upper() != mode.upper():
            continue
        for trade in state.get("trade_log", []):
            if not isinstance(trade, dict):
                continue
            pnl = trade.get("realized_pnl_dollars")
            if pnl is None:
                continue
            # Filter by lookback
            exit_time_str = trade.get("exit_time")
            if exit_time_str and lookback_days < 9999:
                try:
                    et = datetime.fromisoformat(str(exit_time_str))
                    if et.tzinfo is None:
                        et = pytz.timezone("US/Eastern").localize(et)
                    if et < cutoff:
                        continue
                except Exception:
                    pass
            pnl_val = float(pnl)
            total_trades += 1
            if pnl_val > 0:
                total_wins += 1
            total_pnl += pnl_val
            pnl_list.append(pnl_val)

            regime = str(trade.get("regime_at_entry") or trade.get("regime") or "UNKNOWN").upper()
            regime_pnl[regime] = regime_pnl.get(regime, 0.0) + pnl_val
            regime_trades[regime] = regime_trades.get(regime, 0) + 1

            # Track per-sim win rate for best_sim
            wins, total = sim_wr.get(sim_id, (0, 0))
            sim_wr[sim_id] = (wins + (1 if pnl_val > 0 else 0), total + 1)

    # Find best sim
    for sid, (w, t) in sim_wr.items():
        if t >= 3:
            wr = w / t
            if wr > best_sim_wr:
                best_sim_wr = wr
                best_sim = sid

    return {
        "trades": total_trades,
        "wins": total_wins,
        "total_pnl": total_pnl,
        "pnl_list": pnl_list,
        "regime_pnl": regime_pnl,
        "regime_trades": regime_trades,
        "best_sim": best_sim,
    }


def _compute_score(agg: dict, current_regime: str) -> dict:
    """Compute composite score from aggregated stats."""
    trades = agg["trades"]
    if trades == 0:
        return {"score": 0, "win_rate": 0, "expectancy": 0, "profit_factor": 0}

    win_rate = agg["wins"] / trades
    expectancy = agg["total_pnl"] / trades

    gross_profit = sum(p for p in agg["pnl_list"] if p > 0) if agg["pnl_list"] else 0.0
    gross_loss = abs(sum(p for p in agg["pnl_list"] if p < 0)) if agg["pnl_list"] else 0.0
    profit_factor = gross_profit / max(gross_loss, 0.01)

    # Consistency: 1 - (stdev / mean_abs), clipped to [0, 1]
    consistency = 0.5
    if len(agg["pnl_list"]) >= 3:
        try:
            sd = stdev(agg["pnl_list"])
            mean_abs = sum(abs(p) for p in agg["pnl_list"]) / len(agg["pnl_list"])
            if mean_abs > 0:
                consistency = max(0.0, min(1.0, 1.0 - sd / mean_abs))
        except Exception:
            pass

    # Regime fit: 1.0 if current regime matches strategy's best regime, 0.5 otherwise
    regime_fit = 0.5
    if agg.get("regime_pnl") and current_regime != "UNKNOWN":
        best_regime = max(agg["regime_pnl"], key=lambda r: agg["regime_pnl"][r] / max(agg["regime_trades"].get(r, 1), 1))
        if best_regime.upper() == current_regime.upper():
            regime_fit = 1.0
        elif current_regime.upper() in agg["regime_pnl"]:
            # Partial credit if current regime has positive expectancy
            regime_trades = agg["regime_trades"].get(current_regime.upper(), 0)
            if regime_trades > 0:
                regime_exp = agg["regime_pnl"].get(current_regime.upper(), 0) / regime_trades
                regime_fit = 0.8 if regime_exp > 0 else 0.3

    # Normalize to 0-1
    wr_norm = win_rate
    exp_norm = max(0.0, min(1.0, (expectancy + 50.0) / 100.0))  # [-50, 50] → [0, 1]
    pf_norm = max(0.0, min(1.0, profit_factor / 3.0))

    score = (
        W_WINRATE * wr_norm
        + W_EXPECTANCY * exp_norm
        + W_PROFIT_FACTOR * pf_norm
        + W_CONSISTENCY * consistency
        + W_REGIME_FIT * regime_fit
    ) * 100

    return {
        "score": round(score, 1),
        "win_rate": round(win_rate, 4),
        "expectancy": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2),
        "consistency": round(consistency, 3),
        "regime_fit": round(regime_fit, 2),
    }


def evaluate_strategies(lookback_days: int = 14, min_trades: int = 3) -> list[dict]:
    """
    Evaluate all signal_modes and return ranked list.

    Returns list of dicts sorted by score descending:
    [{"signal_mode": "TREND_PULLBACK", "score": 82.3, "win_rate": 0.62, ...}, ...]
    """
    global _cache, _cache_time
    cache_key = f"{lookback_days}:{min_trades}"
    if time.time() - _cache_time < _CACHE_TTL and cache_key in _cache:
        return _cache[cache_key]

    sim_states = _load_all_trade_logs()
    perf_data = _load_perf_store_data()
    current_regime = _get_current_regime()

    # Collect all signal modes from both sources
    all_modes: set[str] = set()
    for state in sim_states.values():
        mode = state.get("signal_mode", "")
        if mode:
            all_modes.add(mode.upper())
    for mode in perf_data:
        if mode and not mode.startswith("_"):
            all_modes.add(mode.upper())

    # Skip meta-modes
    all_modes.discard("OPPORTUNITY")

    rankings = []
    configs = _load_sim_configs()

    for mode in sorted(all_modes):
        # Aggregate from trade logs first (preferred), then perf store
        log_agg = _aggregate_mode_from_logs(mode, sim_states, lookback_days)
        perf_agg = _aggregate_mode_from_perf(mode, perf_data)

        # Use whichever has more data; merge if both have some
        if log_agg["trades"] >= min_trades:
            agg = log_agg
        elif perf_agg["trades"] >= min_trades:
            agg = perf_agg
        elif log_agg["trades"] + perf_agg["trades"] >= min_trades:
            # Merge
            agg = {
                "trades": log_agg["trades"] + perf_agg["trades"],
                "wins": log_agg["wins"] + perf_agg["wins"],
                "total_pnl": log_agg["total_pnl"] + perf_agg["total_pnl"],
                "pnl_list": log_agg["pnl_list"] + perf_agg["pnl_list"],
                "regime_pnl": {**perf_agg.get("regime_pnl", {}), **log_agg.get("regime_pnl", {})},
                "regime_trades": {**perf_agg.get("regime_trades", {}), **log_agg.get("regime_trades", {})},
                "best_sim": log_agg.get("best_sim"),
            }
        else:
            continue

        metrics = _compute_score(agg, current_regime)
        if metrics["score"] <= 0:
            continue

        # Find best_sim and its params
        best_sim = agg.get("best_sim")
        recommended_params = {}
        if best_sim and best_sim in configs:
            prof = configs[best_sim]
            for key in ("dte_min", "dte_max", "stop_loss_pct", "profit_target_pct",
                        "hold_min_seconds", "hold_max_seconds",
                        "trailing_stop_activate_pct", "trailing_stop_trail_pct"):
                if key in prof:
                    recommended_params[key] = prof[key]

        rankings.append({
            "signal_mode": mode,
            "trade_count": agg["trades"],
            "best_sim": best_sim,
            "recommended_params": recommended_params,
            **metrics,
        })

    rankings.sort(key=lambda r: r["score"], reverse=True)

    _cache[cache_key] = rankings
    _cache_time = time.time()

    return rankings


def persist_rankings(rankings: list[dict] | None = None) -> None:
    """Persist the latest strategy ranking for dashboard consumption."""
    if rankings is None:
        rankings = evaluate_strategies()
    output = {
        "timestamp": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
        "rankings": rankings,
    }
    tmp = RANKINGS_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(output, f, indent=2, default=str)
        os.replace(tmp, RANKINGS_PATH)
    except Exception:
        logger.error("strategy_rankings_persist_failed", exc_info=True)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
