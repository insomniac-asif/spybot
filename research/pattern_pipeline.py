#!/usr/bin/env python3
"""
research/pattern_pipeline.py
Comprehensive pattern discovery pipeline (Phases 2-3-6 combined).

1. Runs backtests for all sims, captures full trade data
2. Tags each trade with market conditions at entry
3. Discovers winning/losing patterns (1-2-3 factor combos)
4. Validates pattern consistency across runs
5. Runs optimized engine on test data (skip anti-patterns)
6. Saves results to research/patterns/

Usage:
    python research/pattern_pipeline.py                    # All sims
    python research/pattern_pipeline.py --sim SIM03        # Single sim
    python research/pattern_pipeline.py --tag-only         # Re-tag existing trades
    python research/pattern_pipeline.py --analyze-only     # Re-analyze existing tagged trades
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine

# ── Date splits (527 trading days: 2024-02-01 to 2026-03-10) ────────────
TRAIN_START = "2024-02-01"
TRAIN_END = "2025-07-22"
VAL_START = "2025-07-23"
VAL_END = "2025-11-11"
TEST_START = "2025-11-12"
TEST_END = "2026-03-10"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "simulation", "sim_config.yaml")
PATTERNS_DIR = os.path.join(os.path.dirname(__file__), "patterns")
TRADES_DIR = os.path.join(PATTERNS_DIR, "trades")

os.makedirs(PATTERNS_DIR, exist_ok=True)
os.makedirs(TRADES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Run backtests and collect trade data
# ═══════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run_backtest_collect_trades(sim_id: str, profile: dict,
                                start: str, end: str) -> list[dict]:
    """Run backtest, return full trade list with all metadata."""
    bt_profile = copy.deepcopy(profile)
    bt_profile["symbols"] = ["SPY"]  # SPY-only for consistency

    engine = BacktestEngine(
        profile_id=sim_id,
        profile=bt_profile,
        start_date=start,
        end_date=end,
        max_runs=10,  # Allow multiple runs (blowups restart at $500)
        verbose=False,
        adaptive=False,
    )
    summary = engine.run()

    trades = []
    for r in summary.runs:
        if isinstance(r, dict):
            trades.extend(r.get("trades", []))
        else:
            tlist = r.trades if hasattr(r, "trades") and isinstance(r.trades, list) else []
            trades.extend(tlist)

    # Convert BacktestTrade objects to dicts if needed
    result = []
    for t in trades:
        if isinstance(t, dict):
            result.append(t)
        elif hasattr(t, "__dict__"):
            result.append(vars(t))
        else:
            result.append({"pnl": getattr(t, "pnl", 0)})

    return result


def collect_all_trades(sim_id: str, profile: dict) -> dict:
    """Collect trades for train, validate, and test periods."""
    print(f"  Collecting trades: TRAIN ({TRAIN_START} to {TRAIN_END})...", flush=True)
    train_trades = run_backtest_collect_trades(sim_id, profile, TRAIN_START, TRAIN_END)
    print(f"    -> {len(train_trades)} trades", flush=True)

    print(f"  Collecting trades: VALIDATE ({VAL_START} to {VAL_END})...", flush=True)
    val_trades = run_backtest_collect_trades(sim_id, profile, VAL_START, VAL_END)
    print(f"    -> {len(val_trades)} trades", flush=True)

    print(f"  Collecting trades: TEST ({TEST_START} to {TEST_END})...", flush=True)
    test_trades = run_backtest_collect_trades(sim_id, profile, TEST_START, TEST_END)
    print(f"    -> {len(test_trades)} trades", flush=True)

    return {
        "sim_id": sim_id,
        "signal_mode": profile.get("signal_mode", "?"),
        "train_trades": train_trades,
        "val_trades": val_trades,
        "test_trades": test_trades,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6a: Tag every trade with market conditions
# ═══════════════════════════════════════════════════════════════════════════

# Pre-load symbol data for tagging
_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _load_symbol_df(symbol: str) -> pd.DataFrame:
    """Load and cache a symbol's 1-min CSV."""
    if symbol not in _DATA_CACHE:
        csv_path = os.path.join(BASE_DIR, "data", f"{symbol.lower()}_1m.csv")
        if not os.path.exists(csv_path):
            _DATA_CACHE[symbol] = pd.DataFrame()
            return _DATA_CACHE[symbol]
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        _DATA_CACHE[symbol] = df
    return _DATA_CACHE[symbol]


def _get_bars_at(symbol: str, ts: pd.Timestamp, lookback: int = 60) -> pd.DataFrame:
    """Get the lookback bars ending at or before timestamp."""
    df = _load_symbol_df(symbol)
    if df.empty:
        return pd.DataFrame()
    mask = df.index <= ts
    return df.loc[mask].tail(lookback)


def _compute_regime(bars: pd.DataFrame) -> str:
    """Replicate the bot's regime detection on 60 bars."""
    if len(bars) < 60:
        return "NO_DATA"
    recent = bars.tail(60)
    price_change = recent["close"].iloc[-1] - recent["close"].iloc[0]
    high = recent["high"].max()
    low = recent["low"].min()
    total_range = high - low
    avg_candle = (recent["high"] - recent["low"]).mean()

    if avg_candle < 0.08:
        return "COMPRESSION"

    directionality = abs(price_change) / total_range if total_range > 0 else 0

    # VWAP approximation: volume-weighted close
    if "volume" in recent.columns and recent["volume"].sum() > 0:
        vwap = (recent["close"] * recent["volume"]).sum() / recent["volume"].sum()
    else:
        vwap = recent["close"].mean()

    above_vwap = (recent["close"] > vwap).sum()
    below_vwap = (recent["close"] < vwap).sum()

    if directionality > 0.6 and abs(above_vwap - below_vwap) > 30:
        return "TREND"
    if total_range > 1.20:
        return "VOLATILE"
    return "RANGE"


def _get_vix_level(ts: pd.Timestamp) -> str:
    """Get VIX level from VXX data at timestamp.
    VXX ranges ~25-94 in our data (median ~45). Use quartile-based buckets.
    """
    bars = _get_bars_at("VXX", ts, lookback=5)
    if bars.empty:
        return "UNKNOWN"
    vxx_price = bars["close"].iloc[-1]
    if vxx_price < 36:
        return "LOW"
    elif vxx_price < 45:
        return "NORMAL"
    elif vxx_price < 53:
        return "ELEVATED"
    else:
        return "HIGH"


def tag_trade(trade: dict) -> dict:
    """Add condition tags to a single trade."""
    entry_time_str = trade.get("entry_time", "")
    if not entry_time_str:
        trade["tags"] = {}
        return trade

    try:
        ts = pd.Timestamp(entry_time_str)
    except Exception:
        trade["tags"] = {}
        return trade

    symbol = trade.get("symbol", "SPY")
    direction = trade.get("direction", "")
    pnl = trade.get("pnl") or trade.get("realized_pnl_dollars", 0)

    # Get bars at entry time
    bars = _get_bars_at(symbol, ts, lookback=60)

    tags = {}

    # Hour bucket
    tags["hour_bucket"] = ts.hour

    # Day of week
    tags["day_of_week"] = ts.strftime("%A")

    # Regime at entry (recompute from bars for accuracy)
    if len(bars) >= 60:
        tags["regime_at_entry"] = _compute_regime(bars)
    else:
        tags["regime_at_entry"] = trade.get("regime", "UNKNOWN")

    # VIX level
    tags["vix_level"] = _get_vix_level(ts)

    # ATR state
    if len(bars) >= 20:
        tr = bars["high"] - bars["low"]
        atr_current = tr.iloc[-1]
        atr_sma = tr.tail(20).mean()
        if atr_sma > 0:
            atr_ratio = atr_current / atr_sma
            if atr_ratio < 0.8:
                tags["atr_state"] = "COMPRESSING"
            elif atr_ratio > 1.2:
                tags["atr_state"] = "EXPANDING"
            else:
                tags["atr_state"] = "NORMAL"
        else:
            tags["atr_state"] = "UNKNOWN"
    else:
        tags["atr_state"] = "UNKNOWN"

    # Trend alignment (signal direction aligned with 50-bar SMA slope)
    if len(bars) >= 50:
        sma50 = bars["close"].tail(50).mean()
        sma50_prev = bars["close"].iloc[-51:-1].mean() if len(bars) >= 51 else sma50
        sma_slope = "UP" if sma50 > sma50_prev else "DOWN"
        if direction in ("BULLISH", "CALL"):
            tags["trend_alignment"] = "YES" if sma_slope == "UP" else "NO"
        elif direction in ("BEARISH", "PUT"):
            tags["trend_alignment"] = "YES" if sma_slope == "DOWN" else "NO"
        else:
            tags["trend_alignment"] = "UNKNOWN"
    else:
        tags["trend_alignment"] = "UNKNOWN"

    # Volume state
    if len(bars) >= 20 and "volume" in bars.columns:
        vol_current = bars["volume"].iloc[-1]
        vol_sma = bars["volume"].tail(20).mean()
        if vol_sma > 0:
            vol_ratio = vol_current / vol_sma
            if vol_ratio < 0.7:
                tags["volume_state"] = "LOW"
            elif vol_ratio > 1.3:
                tags["volume_state"] = "HIGH"
            else:
                tags["volume_state"] = "NORMAL"
        else:
            tags["volume_state"] = "UNKNOWN"
    else:
        tags["volume_state"] = "UNKNOWN"

    # Recent momentum (last 5 bars net move vs signal direction)
    if len(bars) >= 5:
        net_move = bars["close"].iloc[-1] - bars["close"].iloc[-5]
        if direction in ("BULLISH", "CALL"):
            tags["recent_momentum"] = "WITH_SIGNAL" if net_move > 0 else "AGAINST_SIGNAL"
        elif direction in ("BEARISH", "PUT"):
            tags["recent_momentum"] = "WITH_SIGNAL" if net_move < 0 else "AGAINST_SIGNAL"
        else:
            tags["recent_momentum"] = "UNKNOWN"
    else:
        tags["recent_momentum"] = "UNKNOWN"

    # Win/loss
    tags["win"] = pnl > 0

    trade["tags"] = tags
    return trade


def tag_all_trades(trades: list[dict]) -> list[dict]:
    """Tag all trades with market conditions."""
    for i, t in enumerate(trades):
        tag_trade(t)
        if (i + 1) % 200 == 0:
            print(f"    Tagged {i+1}/{len(trades)} trades...", flush=True)
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6b-c: Pattern discovery with consistency checks
# ═══════════════════════════════════════════════════════════════════════════

TAG_DIMENSIONS = [
    "hour_bucket", "day_of_week", "regime_at_entry", "vix_level",
    "atr_state", "trend_alignment", "volume_state", "recent_momentum",
]


def _trade_pnl(t: dict) -> float:
    return t.get("pnl") or t.get("realized_pnl_dollars", 0)


def _compute_pattern_stats(matching_trades: list[dict], all_trades: list[dict]) -> dict:
    """Compute statistics for a set of matching trades."""
    n = len(matching_trades)
    if n == 0:
        return {}

    wins = [t for t in matching_trades if _trade_pnl(t) > 0]
    losses = [t for t in matching_trades if _trade_pnl(t) <= 0]
    win_rate = len(wins) / n

    avg_win_pct = np.mean([t.get("pnl_pct", 0) for t in wins]) if wins else 0
    avg_loss_pct = np.mean([abs(t.get("pnl_pct", 0)) for t in losses]) if losses else 0
    total_pnl = sum(_trade_pnl(t) for t in matching_trades)
    expectancy = total_pnl / n

    # Profit factor
    gross_profit = sum(_trade_pnl(t) for t in wins)
    gross_loss = abs(sum(_trade_pnl(t) for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Overall stats for comparison
    overall_wr = sum(1 for t in all_trades if _trade_pnl(t) > 0) / len(all_trades) if all_trades else 0

    # Kelly criterion
    if avg_loss_pct > 0:
        b = avg_win_pct / avg_loss_pct
        kelly = (b * win_rate - (1 - win_rate)) / b if b > 0 else 0
    else:
        kelly = 0
    half_kelly = max(0, kelly * 0.5)

    # Frequency (trades per trading day, assuming ~527 days)
    # Estimate from train+val (447 days)
    frequency = n / 447.0

    return {
        "sample_size": n,
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(float(avg_win_pct), 4),
        "avg_loss_pct": round(float(avg_loss_pct), 4),
        "expectancy_per_trade": round(expectancy, 2),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(min(pf, 999), 2),
        "half_kelly": round(half_kelly, 4),
        "frequency_per_day": round(frequency, 3),
        "overall_win_rate": round(overall_wr, 4),
        "edge_vs_baseline": round(win_rate - overall_wr, 4),
    }


def _check_consistency(matching_trades: list[dict], min_per_run: int = 3) -> float:
    """Check pattern consistency across runs. Returns consistency score 0-1."""
    runs = defaultdict(list)
    for t in matching_trades:
        run_num = t.get("run_number") or t.get("_run_number", 0)
        runs[run_num].append(t)

    if not runs:
        return 0.0

    # Group by month as proxy for "runs" if all in same run
    if len(runs) <= 1:
        # Use monthly grouping instead
        runs = defaultdict(list)
        for t in matching_trades:
            entry = t.get("entry_time", "")
            if entry:
                month_key = entry[:7]  # "2024-02"
                runs[month_key].append(t)

    qualifying_runs = 0
    winning_runs = 0

    for _, run_trades in runs.items():
        if len(run_trades) < min_per_run:
            continue
        qualifying_runs += 1
        run_wr = sum(1 for t in run_trades if _trade_pnl(t) > 0) / len(run_trades)
        if run_wr > 0.5:  # Majority winners
            winning_runs += 1

    if qualifying_runs == 0:
        return 0.0

    return round(winning_runs / qualifying_runs, 2)


def _get_tag_value(trade: dict, dim: str):
    """Get tag value for a dimension from a trade."""
    tags = trade.get("tags", {})
    return tags.get(dim)


def _matches_conditions(trade: dict, conditions: dict) -> bool:
    """Check if trade matches ALL conditions."""
    tags = trade.get("tags", {})
    for dim, val in conditions.items():
        if tags.get(dim) != val:
            return False
    return True


def find_patterns(trades: list[dict], min_sample: int = 10) -> list[dict]:
    """
    Find winning condition combinations using EXPECTANCY as primary metric.
    Many strategies have <50% WR but profit from large winners.
    A pattern is "winning" if it has positive expectancy significantly above baseline.
    """
    if not trades:
        return []

    overall_wr = sum(1 for t in trades if _trade_pnl(t) > 0) / len(trades)
    overall_expect = sum(_trade_pnl(t) for t in trades) / len(trades)
    patterns = []

    # Get all unique values per dimension
    dim_values = {}
    for dim in TAG_DIMENSIONS:
        vals = set()
        for t in trades:
            v = _get_tag_value(t, dim)
            if v is not None and v != "UNKNOWN":
                vals.add(v)
        dim_values[dim] = vals

    def _evaluate(conditions, min_n=min_sample, factor_count=1):
        matching = [t for t in trades if _matches_conditions(t, conditions)]
        if len(matching) < min_n:
            return None
        stats = _compute_pattern_stats(matching, trades)
        # Pattern must have positive expectancy AND beat baseline by meaningful margin
        if stats["expectancy_per_trade"] <= 0:
            return None
        # Must beat baseline: either +$1/trade better OR +10pp WR
        expect_edge = stats["expectancy_per_trade"] - overall_expect
        wr_edge = stats["win_rate"] - overall_wr
        if expect_edge < 0.5 and wr_edge < 0.08:
            return None
        consistency = _check_consistency(matching)
        return {
            "factors": factor_count,
            "conditions": conditions,
            "consistency": consistency,
            "expect_edge": round(expect_edge, 2),
            **stats,
        }

    # ── 1-factor patterns ──
    for dim in TAG_DIMENSIONS:
        for val in dim_values.get(dim, []):
            r = _evaluate({dim: val}, factor_count=1)
            if r:
                patterns.append(r)

    # ── 2-factor patterns ──
    for dim1, dim2 in combinations(TAG_DIMENSIONS, 2):
        for v1 in dim_values.get(dim1, []):
            for v2 in dim_values.get(dim2, []):
                r = _evaluate({dim1: v1, dim2: v2}, factor_count=2)
                if r:
                    patterns.append(r)

    # ── 3-factor patterns (only from promising 2-factor) ──
    promising_2f = [p for p in patterns if p["factors"] == 2 and p["expectancy_per_trade"] > 0]
    for p2 in promising_2f[:30]:  # Limit search space
        used_dims = set(p2["conditions"].keys())
        for dim3 in TAG_DIMENSIONS:
            if dim3 in used_dims:
                continue
            for v3 in dim_values.get(dim3, []):
                conditions = {**p2["conditions"], dim3: v3}
                r = _evaluate(conditions, min_n=max(min_sample, 15), factor_count=3)
                if r and r["consistency"] >= 0.50:
                    patterns.append(r)

    # Score: expectancy_edge * sqrt(sample_size) * consistency
    for p in patterns:
        p["score"] = round(
            max(0, p["expect_edge"]) * math.sqrt(p["sample_size"]) * max(p["consistency"], 0.3),
            3
        )

    patterns.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate: remove patterns that are subsets of higher-scored patterns
    filtered = []
    for p in patterns:
        is_subset = False
        for existing in filtered:
            if (existing["factors"] > p["factors"] and
                all(existing["conditions"].get(k) == v for k, v in p["conditions"].items())):
                is_subset = True
                break
        if not is_subset:
            filtered.append(p)

    return filtered[:20]


def find_anti_patterns(trades: list[dict], min_sample: int = 20) -> list[dict]:
    """
    Find condition combinations with STRONGLY negative expectancy.
    Conservative: only flag patterns with deeply negative expectancy,
    high sample size, and consistent losses. Max 5 anti-patterns to
    avoid over-filtering.
    """
    if not trades:
        return []

    overall_wr = sum(1 for t in trades if _trade_pnl(t) > 0) / len(trades)
    overall_expect = sum(_trade_pnl(t) for t in trades) / len(trades)
    anti_patterns = []

    dim_values = {}
    for dim in TAG_DIMENSIONS:
        vals = set()
        for t in trades:
            v = _get_tag_value(t, dim)
            if v is not None and v != "UNKNOWN":
                vals.add(v)
        dim_values[dim] = vals

    def _evaluate_anti(conditions, min_n=min_sample, factor_count=1):
        matching = [t for t in trades if _matches_conditions(t, conditions)]
        if len(matching) < min_n:
            return None
        stats = _compute_pattern_stats(matching, trades)
        # Must have NEGATIVE expectancy
        if stats["expectancy_per_trade"] >= 0:
            return None
        # Must be significantly worse than baseline (at least $1/trade worse)
        expect_gap = overall_expect - stats["expectancy_per_trade"]
        if expect_gap < 1.0:
            return None
        # Must not be the majority of all trades (max 30% of total)
        if len(matching) > len(trades) * 0.30:
            return None
        consistency = _check_consistency(matching)
        loss_consistency = 1.0 - consistency  # High = consistently losing
        if loss_consistency < 0.60:
            return None
        return {
            "factors": factor_count,
            "conditions": conditions,
            "consistency": loss_consistency,
            "expect_gap": round(expect_gap, 2),
            **stats,
        }

    # 1-factor anti-patterns
    for dim in TAG_DIMENSIONS:
        for val in dim_values.get(dim, []):
            r = _evaluate_anti({dim: val}, factor_count=1)
            if r:
                anti_patterns.append(r)

    # 2-factor anti-patterns
    for dim1, dim2 in combinations(TAG_DIMENSIONS, 2):
        for v1 in dim_values.get(dim1, []):
            for v2 in dim_values.get(dim2, []):
                r = _evaluate_anti({dim1: v1, dim2: v2}, min_n=25, factor_count=2)
                if r:
                    anti_patterns.append(r)

    # Score: negative expectancy * sqrt(sample) * consistency
    for ap in anti_patterns:
        ap["score"] = round(
            abs(ap["expectancy_per_trade"]) * math.sqrt(ap["sample_size"]) * ap["consistency"],
            3
        )

    anti_patterns.sort(key=lambda x: x["score"], reverse=True)

    # Keep max 5 non-overlapping anti-patterns to avoid over-filtering
    filtered = []
    for ap in anti_patterns:
        # Check overlap: don't add if it would push skip rate too high
        combined_conditions = [f["conditions"] for f in filtered]
        if len(filtered) >= 5:
            break
        # Check this doesn't overlap >50% with existing
        this_trades = set(
            t.get("entry_time", "") for t in trades if _matches_conditions(t, ap["conditions"])
        )
        overlap_count = 0
        for existing_cond in combined_conditions:
            existing_trades = set(
                t.get("entry_time", "") for t in trades if _matches_conditions(t, existing_cond)
            )
            overlap_count += len(this_trades & existing_trades)
        if overlap_count > len(this_trades) * 0.5 and filtered:
            continue
        filtered.append(ap)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6d: Optimized backtest engine (skip anti-patterns)
# ═══════════════════════════════════════════════════════════════════════════

def run_optimized_test(test_trades: list[dict], anti_patterns: list[dict],
                       balance_start: float = 500, death_threshold: float = 25) -> dict:
    """
    Re-simulate test trades, skipping those matching anti-patterns.
    Returns comparison metrics.
    """
    # Tag test trades if not already tagged
    for t in test_trades:
        if "tags" not in t:
            tag_trade(t)

    # Only skip trades matching very strong anti-patterns (consistency >= 0.70, score >= 5)
    strong_anti = [ap for ap in anti_patterns
                   if ap.get("consistency", 0) >= 0.70 and ap.get("score", 0) >= 3.0]

    kept_trades = []
    skipped_trades = []
    for t in test_trades:
        skip = False
        for ap in strong_anti:
            if _matches_conditions(t, ap["conditions"]):
                skip = True
                break
        if skip:
            skipped_trades.append(t)
        else:
            kept_trades.append(t)

    # Simulate the kept trades
    def _simulate(trades_list):
        balance = balance_start
        peak = balance_start
        wins = 0
        total = 0
        total_pnl = 0
        gross_profit = 0
        gross_loss = 0
        blown = 0

        for t in trades_list:
            if balance <= death_threshold:
                blown += 1
                balance = balance_start
                peak = balance_start

            pnl = _trade_pnl(t)
            balance += pnl
            total_pnl += pnl
            total += 1

            if pnl > 0:
                wins += 1
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)

            if balance > peak:
                peak = balance

        pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)
        return {
            "trades": total,
            "wins": wins,
            "win_rate": round(wins / total, 4) if total else 0,
            "total_pnl": round(total_pnl, 2),
            "expectancy": round(total_pnl / total, 2) if total else 0,
            "profit_factor": round(min(pf, 999), 2),
            "peak_balance": round(peak, 2),
            "final_balance": round(balance, 2),
            "blown_count": blown,
        }

    base_result = _simulate(test_trades)
    opt_result = _simulate(kept_trades)

    return {
        "base": base_result,
        "optimized": opt_result,
        "trades_skipped": len(skipped_trades),
        "anti_patterns_used": len(strong_anti),
        "skip_rate": round(len(skipped_trades) / len(test_trades), 3) if test_trades else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_sim_pipeline(sim_id: str, profile: dict, tag_only: bool = False,
                     analyze_only: bool = False) -> dict:
    """Full pipeline for one sim."""
    print(f"\n{'='*60}")
    print(f"  {sim_id}: {profile.get('signal_mode', '?')}")
    print(f"{'='*60}")

    trades_file = os.path.join(TRADES_DIR, f"{sim_id}_trades.json")
    patterns_file = os.path.join(PATTERNS_DIR, f"{sim_id}_patterns.json")

    # Step 1: Collect trades (or load from cache)
    if analyze_only and os.path.exists(trades_file):
        print(f"  Loading cached trades from {trades_file}...", flush=True)
        with open(trades_file) as f:
            trade_data = json.load(f)
    else:
        trade_data = collect_all_trades(sim_id, profile)

        # Save raw trades
        with open(trades_file, "w") as f:
            json.dump(trade_data, f, default=str)
        print(f"  Saved trades to {trades_file}", flush=True)

    train_trades = trade_data.get("train_trades", [])
    val_trades = trade_data.get("val_trades", [])
    test_trades = trade_data.get("test_trades", [])

    total_trades = len(train_trades) + len(val_trades) + len(test_trades)
    if total_trades < 20:
        print(f"  SKIP: Only {total_trades} total trades (need >=20)", flush=True)
        result = {
            "sim_id": sim_id,
            "signal_mode": profile.get("signal_mode", "?"),
            "skip_reason": f"too_few_trades:{total_trades}",
            "total_trades": total_trades,
        }
        with open(patterns_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    # Step 2: Tag trades
    print(f"  Tagging train+val trades ({len(train_trades) + len(val_trades)})...", flush=True)
    discovery_trades = tag_all_trades(train_trades + val_trades)
    print(f"  Tagging test trades ({len(test_trades)})...", flush=True)
    test_trades = tag_all_trades(test_trades)

    if tag_only:
        # Save tagged trades and return
        trade_data["train_trades"] = train_trades
        trade_data["val_trades"] = val_trades
        trade_data["test_trades"] = test_trades
        with open(trades_file, "w") as f:
            json.dump(trade_data, f, default=str)
        print(f"  Tag-only mode: saved tagged trades", flush=True)
        return {"sim_id": sim_id, "tagged": True}

    # Step 3: Discover patterns on train+val
    overall_wr = sum(1 for t in discovery_trades if _trade_pnl(t) > 0) / len(discovery_trades) if discovery_trades else 0
    print(f"  Overall WR (train+val): {overall_wr:.1%} ({len(discovery_trades)} trades)", flush=True)

    print(f"  Finding winning patterns...", flush=True)
    patterns = find_patterns(discovery_trades)
    print(f"  Found {len(patterns)} winning patterns", flush=True)

    print(f"  Finding anti-patterns...", flush=True)
    anti_patterns = find_anti_patterns(discovery_trades)
    print(f"  Found {len(anti_patterns)} anti-patterns", flush=True)

    # Step 4: Run optimized engine on test data
    print(f"  Running optimized test...", flush=True)
    comparison = run_optimized_test(test_trades, anti_patterns)

    base = comparison["base"]
    opt = comparison["optimized"]
    print(f"\n  === {sim_id} COMPARISON: BASE vs OPTIMIZED (TEST SET) ===")
    print(f"  {'':20} {'BASE':>10} {'OPTIMIZED':>10}")
    print(f"  {'Trades:':20} {base['trades']:>10} {opt['trades']:>10}")
    print(f"  {'Win rate:':20} {base['win_rate']:>9.1%} {opt['win_rate']:>9.1%}")
    print(f"  {'Expectancy:':20} ${base['expectancy']:>8.2f} ${opt['expectancy']:>8.2f}")
    print(f"  {'Total PnL:':20} ${base['total_pnl']:>8.2f} ${opt['total_pnl']:>8.2f}")
    print(f"  {'Profit Factor:':20} {base['profit_factor']:>10.2f} {opt['profit_factor']:>10.2f}")
    print(f"  {'Final Balance:':20} ${base['final_balance']:>8.2f} ${opt['final_balance']:>8.2f}")
    print(f"  {'Skipped:':20} {'':>10} {comparison['trades_skipped']:>10}")

    # Step 5: Compile and save results
    # Add descriptions to patterns
    for i, p in enumerate(patterns):
        conds = " + ".join(f"{k}={v}" for k, v in p["conditions"].items())
        p["rank"] = i + 1
        p["description"] = conds

    for i, ap in enumerate(anti_patterns):
        conds = " + ".join(f"{k}={v}" for k, v in ap["conditions"].items())
        ap["rank"] = i + 1
        ap["description"] = f"AVOID: {conds}"

    result = {
        "sim_id": sim_id,
        "signal_mode": profile.get("signal_mode", "?"),
        "data_used": f"train+validate ({TRAIN_START} to {VAL_END})",
        "overall_win_rate": round(overall_wr, 4),
        "total_trades_analyzed": len(discovery_trades),
        "train_trades": len(train_trades),
        "val_trades": len(val_trades),
        "test_trades": len(test_trades),
        "patterns": patterns,
        "anti_patterns": anti_patterns,
        "optimized_test_results": opt,
        "base_test_results": base,
        "comparison": comparison,
    }

    with open(patterns_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved to {patterns_file}", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Pattern discovery pipeline")
    parser.add_argument("--sim", type=str, help="Run only this sim")
    parser.add_argument("--tag-only", action="store_true", help="Only tag trades, skip analysis")
    parser.add_argument("--analyze-only", action="store_true", help="Re-analyze cached trades")
    parser.add_argument("--dedup", action="store_true", help="One sim per signal_mode")
    args = parser.parse_args()

    config = load_config()

    # Pre-load VXX and SPY data for tagging
    print("Pre-loading symbol data for tagging...", flush=True)
    _load_symbol_df("SPY")
    _load_symbol_df("VXX")
    print("  Loaded SPY and VXX data", flush=True)

    # Select sims
    mode_groups = {}
    for sim_id in sorted(config.keys()):
        if not sim_id.startswith("SIM") or sim_id == "SIM00":
            continue
        if not isinstance(config[sim_id], dict):
            continue
        mode = config[sim_id].get("signal_mode", "?")
        mode_groups.setdefault(mode, []).append(sim_id)

    sim_ids = []
    if args.sim:
        sim_ids = [args.sim.upper()]
    elif args.dedup:
        for mode, sims in sorted(mode_groups.items()):
            sim_ids.append(sims[0])
    else:
        for sim_id in sorted(config.keys()):
            if not sim_id.startswith("SIM") or sim_id == "SIM00":
                continue
            if not isinstance(config[sim_id], dict):
                continue
            sim_ids.append(sim_id)

    print(f"\nPattern Discovery Pipeline")
    print(f"Train:    {TRAIN_START} to {TRAIN_END} (368 days)")
    print(f"Validate: {VAL_START} to {VAL_END} (79 days)")
    print(f"Test:     {TEST_START} to {TEST_END} (80 days)")
    print(f"Sims:     {len(sim_ids)}")
    print(f"Mode:     {'tag-only' if args.tag_only else 'analyze-only' if args.analyze_only else 'full pipeline'}")
    sys.stdout.flush()

    all_results = {}
    for sim_id in sim_ids:
        profile = config[sim_id]
        try:
            r = run_sim_pipeline(sim_id, profile,
                                 tag_only=args.tag_only,
                                 analyze_only=args.analyze_only)
            all_results[sim_id] = r
        except Exception as e:
            print(f"\n  ERROR on {sim_id}: {e}")
            traceback.print_exc()
            all_results[sim_id] = {"sim_id": sim_id, "error": str(e)}

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  PATTERN DISCOVERY SUMMARY")
    print(f"{'='*80}")
    print(f"{'SIM':<8} {'Mode':<25} {'Patterns':>8} {'Anti':>5} {'Base WR':>8} {'Opt WR':>8} {'Base E$':>8} {'Opt E$':>8}")
    print(f"{'-'*80}")

    for sim_id in sim_ids:
        r = all_results.get(sim_id, {})
        if "error" in r or "skip_reason" in r:
            reason = r.get("error", r.get("skip_reason", "?"))
            print(f"{sim_id:<8} {reason[:60]}")
            continue

        mode = r.get("signal_mode", "?")[:24]
        n_pat = len(r.get("patterns", []))
        n_anti = len(r.get("anti_patterns", []))
        base = r.get("base_test_results", {})
        opt = r.get("optimized_test_results", {})

        print(f"{sim_id:<8} {mode:<25} {n_pat:>8} {n_anti:>5} "
              f"{base.get('win_rate', 0):>7.1%} {opt.get('win_rate', 0):>7.1%} "
              f"${base.get('expectancy', 0):>6.2f} ${opt.get('expectancy', 0):>6.2f}")

    # Save combined summary
    summary_path = os.path.join(PATTERNS_DIR, "all_patterns_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
