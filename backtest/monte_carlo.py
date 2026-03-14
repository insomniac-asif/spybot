"""
backtest/monte_carlo.py
GPU-accelerated Monte Carlo simulation for strategy stress-testing.

Uses completed backtest trade data to calibrate a statistical model,
then generates millions of synthetic trade sequences on the GPU to
produce probability distributions of outcomes.

Usage:
    python -m backtest.monte_carlo --sim SIM06
    python -m backtest.monte_carlo --sim SIM06 --paths 2000000 --days 548
    python -m backtest.monte_carlo --all

Requires: cupy (pip install cupy-cuda12x nvidia-curand-cu12)
Falls back to NumPy CPU if CuPy unavailable (slower but functional).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# GPU acceleration — graceful fallback to CPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np  # type: ignore
    GPU_AVAILABLE = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "simulation", "sim_config.yaml")


# ── Calibration ──────────────────────────────────────────────────────────


@dataclass
class TradeModel:
    """Statistical model calibrated from historical backtest trades."""
    sim_id: str
    signal_mode: str
    n_calibration_trades: int

    # Win probability
    win_rate: float

    # PnL distributions (log-normal params for wins, losses separately)
    win_pnl_mean: float      # mean of winning trade PnL %
    win_pnl_std: float       # std of winning trade PnL %
    loss_pnl_mean: float     # mean of losing trade PnL % (negative)
    loss_pnl_std: float      # std of losing trade PnL %

    # Position sizing
    avg_risk_pct: float      # average % of balance risked per trade
    avg_qty_dollars: float   # average dollar notional per trade

    # Trade frequency
    trades_per_day: float    # average trades per trading day

    # Streak clustering (autocorrelation of wins/losses)
    streak_autocorr: float   # positive = streaky, 0 = independent

    # Account rules
    start_capital: float
    death_threshold: float
    target_balance: float


def calibrate_from_trades(sim_id: str, trades: list[dict],
                          start_capital: float = 3000.0,
                          death_threshold: float = 150.0,
                          target_balance: float = 5000.0) -> TradeModel:
    """Build a statistical model from completed backtest trades."""
    if not trades:
        raise ValueError(f"No trades for {sim_id}")

    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]

    win_rate = len(wins) / len(trades) if trades else 0.5

    # PnL percentage distributions
    win_pcts = [t["pnl_pct"] for t in wins] if wins else [0.05]
    loss_pcts = [abs(t["pnl_pct"]) for t in losses] if losses else [0.05]

    win_pnl_mean = float(np.mean(win_pcts))
    win_pnl_std = max(float(np.std(win_pcts)), 0.001)
    loss_pnl_mean = float(np.mean(loss_pcts))
    loss_pnl_std = max(float(np.std(loss_pcts)), 0.001)

    # Risk sizing from balance_before
    risk_pcts = []
    for t in trades:
        bb = t.get("balance_before", start_capital)
        if bb > 0:
            risk_pcts.append(abs(t.get("pnl", 0)) / bb)
    avg_risk_pct = float(np.mean(risk_pcts)) if risk_pcts else 0.02

    # Average dollar notional
    qty_dollars = [t.get("entry_price", 1.0) * t.get("qty", 1) * 100
                   for t in trades]
    avg_qty_dollars = float(np.mean(qty_dollars)) if qty_dollars else 500.0

    # Trade frequency: trades per trading day
    dates = set()
    for t in trades:
        dt_str = t.get("date", t.get("entry_time", ""))[:10]
        if dt_str:
            dates.add(dt_str)
    trading_days = max(len(dates), 1)
    trades_per_day = len(trades) / trading_days

    # Streak autocorrelation: are wins/losses clustered?
    if len(trades) > 10:
        outcomes = np.array([1.0 if t.get("pnl", 0) > 0 else 0.0
                             for t in trades])
        outcomes_centered = outcomes - outcomes.mean()
        if outcomes_centered.std() > 0:
            autocorr = float(np.corrcoef(outcomes_centered[:-1],
                                          outcomes_centered[1:])[0, 1])
        else:
            autocorr = 0.0
    else:
        autocorr = 0.0

    signal_mode = trades[0].get("signal_mode", "UNKNOWN") if trades else "UNKNOWN"

    return TradeModel(
        sim_id=sim_id,
        signal_mode=signal_mode,
        n_calibration_trades=len(trades),
        win_rate=win_rate,
        win_pnl_mean=win_pnl_mean,
        win_pnl_std=win_pnl_std,
        loss_pnl_mean=loss_pnl_mean,
        loss_pnl_std=loss_pnl_std,
        avg_risk_pct=avg_risk_pct,
        avg_qty_dollars=avg_qty_dollars,
        trades_per_day=trades_per_day,
        streak_autocorr=max(min(autocorr, 0.5), -0.5),  # clamp
        start_capital=start_capital,
        death_threshold=death_threshold,
        target_balance=target_balance,
    )


# ── GPU Simulation Engine ────────────────────────────────────────────────


def simulate(model: TradeModel, n_paths: int = 1_000_000,
             n_days: int = 548, seed: int = 42) -> dict:
    """
    Run Monte Carlo simulation on GPU.

    Generates n_paths independent trade sequences, each spanning n_days
    of trading. Returns probability distributions of outcomes.
    """
    xp = cp  # cupy if GPU, numpy if fallback
    backend = "GPU (CuPy)" if GPU_AVAILABLE else "CPU (NumPy)"

    print(f"  Monte Carlo: {n_paths:,} paths x {n_days} days on {backend}")
    print(f"  Model: WR={model.win_rate:.1%}, "
          f"AvgWin={model.win_pnl_mean:.1%}, AvgLoss=-{model.loss_pnl_mean:.1%}, "
          f"{model.trades_per_day:.1f} trades/day")

    t0 = time.perf_counter()

    # Total trades per path
    avg_total_trades = int(model.trades_per_day * n_days)
    if avg_total_trades < 1:
        avg_total_trades = 1

    # Seed the RNG
    xp.random.seed(seed)
    shape = (n_paths, avg_total_trades)

    # Win/loss determination with streak autocorrelation
    # Use correlated Bernoulli via thresholded AR(1) process
    if abs(model.streak_autocorr) > 0.01:
        # AR(1) latent process -> threshold for win/loss
        innovations = xp.random.standard_normal(shape).astype(xp.float32)
        latent = xp.zeros(shape, dtype=xp.float32)
        latent[:, 0] = innovations[:, 0]
        rho = xp.float32(model.streak_autocorr)
        scale = xp.float32(math.sqrt(1 - model.streak_autocorr ** 2))
        for t in range(1, avg_total_trades):
            latent[:, t] = rho * latent[:, t - 1] + scale * innovations[:, t]
        # Threshold: if latent < quantile(win_rate), it's a win
        from scipy.stats import norm as _norm
        threshold = xp.float32(_norm.ppf(model.win_rate))
        is_win = latent < threshold
    else:
        # Independent wins
        is_win = xp.random.random(shape).astype(xp.float32) < model.win_rate

    # PnL percentages: separate distributions for wins and losses
    # Use absolute normal (half-normal) to avoid negative wins / positive losses
    win_pnl = xp.abs(xp.random.normal(
        model.win_pnl_mean, model.win_pnl_std, shape
    ).astype(xp.float32))

    loss_pnl = -xp.abs(xp.random.normal(
        model.loss_pnl_mean, model.loss_pnl_std, shape
    ).astype(xp.float32))

    # Combined PnL per trade (as fraction of risked amount)
    trade_pnl_pct = xp.where(is_win, win_pnl, loss_pnl)

    # ── Balance simulation (vectorized across paths) ──
    # Each trade's PnL = balance * risk_pct * trade_pnl_pct
    # We track balance as a running product for efficiency

    balance = xp.full(n_paths, model.start_capital, dtype=xp.float32)
    peak_balance = balance.copy()
    max_drawdown = xp.zeros(n_paths, dtype=xp.float32)

    # Track milestones
    hit_1k = xp.zeros(n_paths, dtype=xp.int32)
    hit_2500 = xp.zeros(n_paths, dtype=xp.int32)
    hit_target = xp.zeros(n_paths, dtype=xp.int32)
    death_count = xp.zeros(n_paths, dtype=xp.int32)
    alive = xp.ones(n_paths, dtype=xp.bool_)

    # Track trade-by-trade for milestone timing
    day_of_1k = xp.full(n_paths, -1, dtype=xp.int32)
    day_of_2500 = xp.full(n_paths, -1, dtype=xp.int32)
    day_of_target = xp.full(n_paths, -1, dtype=xp.int32)

    trades_per_day = max(1, int(round(model.trades_per_day)))
    risk_pct = xp.float32(model.avg_risk_pct)

    # Process day by day for death/restart logic
    trade_idx = 0
    for day in range(n_days):
        if trade_idx >= avg_total_trades:
            break

        day_trades = min(trades_per_day, avg_total_trades - trade_idx)

        for _ in range(day_trades):
            if trade_idx >= avg_total_trades:
                break

            # PnL for this trade
            pnl = balance * risk_pct * trade_pnl_pct[:, trade_idx]
            balance = balance + pnl
            trade_idx += 1

            # Death check
            newly_dead = alive & (balance <= model.death_threshold)
            death_count += newly_dead.astype(xp.int32)
            # Restart from start_capital
            balance = xp.where(newly_dead, xp.float32(model.start_capital),
                               balance)

            # Peak and drawdown
            peak_balance = xp.maximum(peak_balance, balance)
            dd = (peak_balance - balance) / xp.maximum(peak_balance,
                                                        xp.float32(1.0))
            max_drawdown = xp.maximum(max_drawdown, dd)

        # Milestone checks (once per day for efficiency)
        newly_1k = alive & (hit_1k == 0) & (balance >= 1000)
        day_of_1k = xp.where(newly_1k, xp.int32(day), day_of_1k)
        hit_1k = xp.where(newly_1k, xp.int32(1), hit_1k)

        newly_2500 = alive & (hit_2500 == 0) & (balance >= 2500)
        day_of_2500 = xp.where(newly_2500, xp.int32(day), day_of_2500)
        hit_2500 = xp.where(newly_2500, xp.int32(1), hit_2500)

        newly_target = alive & (hit_target == 0) & (
            balance >= model.target_balance)
        day_of_target = xp.where(newly_target, xp.int32(day), day_of_target)
        hit_target = xp.where(newly_target, xp.int32(1), hit_target)

    # Sync GPU
    if GPU_AVAILABLE:
        cp.cuda.Stream.null.synchronize()

    elapsed = time.perf_counter() - t0
    print(f"  Simulation complete in {elapsed:.1f}s")

    # ── Collect results (transfer to CPU) ──
    if GPU_AVAILABLE:
        balance_cpu = cp.asnumpy(balance)
        peak_cpu = cp.asnumpy(peak_balance)
        max_dd_cpu = cp.asnumpy(max_drawdown)
        death_cpu = cp.asnumpy(death_count)
        hit_1k_cpu = cp.asnumpy(hit_1k)
        hit_2500_cpu = cp.asnumpy(hit_2500)
        hit_target_cpu = cp.asnumpy(hit_target)
        day_1k_cpu = cp.asnumpy(day_of_1k)
        day_2500_cpu = cp.asnumpy(day_of_2500)
        day_target_cpu = cp.asnumpy(day_of_target)
    else:
        balance_cpu = balance
        peak_cpu = peak_balance
        max_dd_cpu = max_drawdown
        death_cpu = death_count
        hit_1k_cpu = hit_1k
        hit_2500_cpu = hit_2500
        hit_target_cpu = hit_target
        day_1k_cpu = day_of_1k
        day_2500_cpu = day_of_2500
        day_target_cpu = day_of_target

    # ── Statistics ──
    results = {
        "sim_id": model.sim_id,
        "signal_mode": model.signal_mode,
        "n_paths": n_paths,
        "n_days": n_days,
        "backend": backend,
        "elapsed_seconds": round(elapsed, 2),
        "calibration_trades": model.n_calibration_trades,
        "model": {
            "win_rate": round(model.win_rate, 4),
            "win_pnl_mean": round(model.win_pnl_mean, 4),
            "win_pnl_std": round(model.win_pnl_std, 4),
            "loss_pnl_mean": round(model.loss_pnl_mean, 4),
            "loss_pnl_std": round(model.loss_pnl_std, 4),
            "avg_risk_pct": round(model.avg_risk_pct, 4),
            "trades_per_day": round(model.trades_per_day, 2),
            "streak_autocorr": round(model.streak_autocorr, 4),
            "start_capital": model.start_capital,
            "death_threshold": model.death_threshold,
            "target_balance": model.target_balance,
        },

        # Final balance distribution
        "final_balance": {
            "mean": round(float(np.mean(balance_cpu)), 2),
            "median": round(float(np.median(balance_cpu)), 2),
            "std": round(float(np.std(balance_cpu)), 2),
            "p5": round(float(np.percentile(balance_cpu, 5)), 2),
            "p10": round(float(np.percentile(balance_cpu, 10)), 2),
            "p25": round(float(np.percentile(balance_cpu, 25)), 2),
            "p75": round(float(np.percentile(balance_cpu, 75)), 2),
            "p90": round(float(np.percentile(balance_cpu, 90)), 2),
            "p95": round(float(np.percentile(balance_cpu, 95)), 2),
            "min": round(float(np.min(balance_cpu)), 2),
            "max": round(float(np.max(balance_cpu)), 2),
        },

        # Peak balance
        "peak_balance": {
            "mean": round(float(np.mean(peak_cpu)), 2),
            "median": round(float(np.median(peak_cpu)), 2),
            "p95": round(float(np.percentile(peak_cpu, 95)), 2),
        },

        # Drawdown
        "max_drawdown_pct": {
            "mean": round(float(np.mean(max_dd_cpu) * 100), 2),
            "median": round(float(np.median(max_dd_cpu) * 100), 2),
            "p95": round(float(np.percentile(max_dd_cpu, 95) * 100), 2),
        },

        # Deaths
        "deaths": {
            "mean": round(float(np.mean(death_cpu)), 2),
            "median": round(float(np.median(death_cpu)), 2),
            "zero_deaths_pct": round(
                float(np.mean(death_cpu == 0) * 100), 2),
            "max": int(np.max(death_cpu)),
        },

        # Milestone probabilities
        "milestones": {
            "$1K": {
                "probability_pct": round(
                    float(np.mean(hit_1k_cpu) * 100), 2),
                "median_days": int(np.median(
                    day_1k_cpu[day_1k_cpu >= 0])) if np.any(
                    day_1k_cpu >= 0) else None,
                "p25_days": int(np.percentile(
                    day_1k_cpu[day_1k_cpu >= 0], 25)) if np.any(
                    day_1k_cpu >= 0) else None,
                "p75_days": int(np.percentile(
                    day_1k_cpu[day_1k_cpu >= 0], 75)) if np.any(
                    day_1k_cpu >= 0) else None,
            },
            "$2.5K": {
                "probability_pct": round(
                    float(np.mean(hit_2500_cpu) * 100), 2),
                "median_days": int(np.median(
                    day_2500_cpu[day_2500_cpu >= 0])) if np.any(
                    day_2500_cpu >= 0) else None,
            },
            "$5K_target": {
                "probability_pct": round(
                    float(np.mean(hit_target_cpu) * 100), 2),
                "median_days": int(np.median(
                    day_target_cpu[day_target_cpu >= 0])) if np.any(
                    day_target_cpu >= 0) else None,
            },
        },

        # Probability of profit
        "profitable_pct": round(
            float(np.mean(balance_cpu > model.start_capital) * 100), 2),

        # Risk of ruin (ending below death threshold)
        "ruin_pct": round(
            float(np.mean(balance_cpu <= model.death_threshold) * 100), 2),

        # Histogram buckets for charting
        "balance_histogram": _build_histogram(balance_cpu,
                                               model.start_capital),
    }

    return results


def _build_histogram(balances: np.ndarray, start: float,
                     n_bins: int = 50) -> list[dict]:
    """Build histogram buckets for frontend charting."""
    # Clip extreme outliers for readable histogram
    p99 = float(np.percentile(balances, 99.5))
    clipped = np.clip(balances, 0, p99)
    counts, edges = np.histogram(clipped, bins=n_bins)
    total = len(balances)
    buckets = []
    for i in range(len(counts)):
        buckets.append({
            "bin_start": round(float(edges[i]), 2),
            "bin_end": round(float(edges[i + 1]), 2),
            "count": int(counts[i]),
            "pct": round(float(counts[i] / total * 100), 2),
        })
    return buckets


# ── Load trades from backtest results ────────────────────────────────────


def _load_trades(sim_id: str) -> tuple[list[dict], float, float, float]:
    """Load trades from backtest summary JSON. Returns (trades, start, death, target)."""
    import yaml

    # Load account rules from sim_config
    start_capital = 3000.0
    death_threshold = 150.0
    target_balance = 5000.0

    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        g = cfg.get("_global", {})
        start_capital = g.get("default_balance_start", 3000.0)
        death_threshold = g.get("death_threshold", 150.0)
    except Exception:
        pass

    # Try dashboard_data.json first (has all sims)
    dashboard_path = os.path.join(RESULTS_DIR, "dashboard_data.json")
    if os.path.exists(dashboard_path):
        with open(dashboard_path) as f:
            all_data = json.load(f)
        if sim_id in all_data:
            sim_data = all_data[sim_id]
            runs = sim_data.get("runs", [])
            trades = []
            for run in runs:
                trades.extend(run.get("trades", []))
            if trades:
                return trades, start_capital, death_threshold, target_balance

    # Fallback: individual summary file
    summary_path = os.path.join(RESULTS_DIR, f"{sim_id}_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        runs = summary.get("runs", [])
        trades = []
        for run in runs:
            trades.extend(run.get("trades", []))
        if trades:
            return trades, start_capital, death_threshold, target_balance

    raise FileNotFoundError(
        f"No backtest data for {sim_id}. Run backtest first.")


# ── Pretty printer ───────────────────────────────────────────────────────


def print_results(r: dict) -> None:
    """Print Monte Carlo results in a readable format."""
    sim = r["sim_id"]
    fb = r["final_balance"]
    ml = r["milestones"]
    dd = r["max_drawdown_pct"]
    deaths = r["deaths"]

    print(f"\n{'=' * 70}")
    print(f"  MONTE CARLO RESULTS: {sim} ({r['signal_mode']})")
    print(f"  {r['n_paths']:,} paths x {r['n_days']} days | "
          f"{r['backend']} | {r['elapsed_seconds']}s")
    print(f"  Calibrated from {r['calibration_trades']} historical trades")
    print(f"{'=' * 70}")

    print(f"\n  FINAL BALANCE DISTRIBUTION:")
    print(f"    Median: ${fb['median']:,.0f}  |  Mean: ${fb['mean']:,.0f}")
    print(f"    5th %%:  ${fb['p5']:,.0f}  |  95th %%: ${fb['p95']:,.0f}")
    print(f"    Min:    ${fb['min']:,.0f}  |  Max:    ${fb['max']:,.0f}")

    print(f"\n  PROBABILITIES:")
    print(f"    Profitable (>${r['model']['start_capital']:,.0f}): "
          f"{r['profitable_pct']:.1f}%")
    print(f"    Ruin (<=${r['model']['death_threshold']:,.0f}):      "
          f"{r['ruin_pct']:.1f}%")

    print(f"\n  MILESTONES:")
    for name, m in ml.items():
        prob = m["probability_pct"]
        days = m.get("median_days")
        day_str = f" in ~{days}d" if days else ""
        marker = "*" if prob >= 50 else "-"
        print(f"    {marker} {name:12s}: {prob:5.1f}% chance{day_str}")

    print(f"\n  RISK:")
    print(f"    Max DD (median): {dd['median']:.1f}%  |  "
          f"95th %%: {dd['p95']:.1f}%")
    print(f"    Deaths (mean):   {deaths['mean']:.1f}  |  "
          f"Zero deaths: {deaths['zero_deaths_pct']:.1f}%")

    print(f"\n{'=' * 70}\n")


# ── Save results ─────────────────────────────────────────────────────────


def save_results(results: dict) -> str:
    """Save Monte Carlo results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"montecarlo_{results['sim_id']}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU Monte Carlo simulation for strategy stress-testing")
    parser.add_argument("--sim", help="Single sim ID (e.g., SIM06)")
    parser.add_argument("--all", action="store_true",
                        help="Run for all sims with backtest data")
    parser.add_argument("--paths", type=int, default=1_000_000,
                        help="Number of simulation paths (default: 1M)")
    parser.add_argument("--days", type=int, default=548,
                        help="Trading days to simulate (default: 548 = ~2yr)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.all:
        # Find all sims with backtest data
        dashboard_path = os.path.join(RESULTS_DIR, "dashboard_data.json")
        if not os.path.exists(dashboard_path):
            print("No dashboard_data.json found. Run backtests first.")
            return
        with open(dashboard_path) as f:
            all_data = json.load(f)
        sim_ids = sorted(all_data.keys())
    elif args.sim:
        sim_ids = [args.sim.upper()]
    else:
        parser.print_help()
        return

    all_results = []
    for sim_id in sim_ids:
        print(f"\n{'-' * 70}")
        print(f"  {sim_id}: Loading trade data...")

        try:
            trades, start_cap, death_thresh, target = _load_trades(sim_id)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        if len(trades) < 20:
            print(f"  SKIP: Only {len(trades)} trades (need >= 20)")
            continue

        model = calibrate_from_trades(
            sim_id, trades, start_cap, death_thresh, target)

        results = simulate(model, n_paths=args.paths, n_days=args.days,
                           seed=args.seed)
        print_results(results)

        path = save_results(results)
        print(f"  Saved: {path}")
        all_results.append(results)

    # Summary comparison if multiple sims
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON (sorted by target hit probability)")
        print(f"{'=' * 70}")
        ranked = sorted(all_results,
                        key=lambda r: r["milestones"]["$5K_target"][
                            "probability_pct"],
                        reverse=True)
        print(f"  {'SIM':<8} {'Median$':>8} {'P(profit)':>10} "
              f"{'P($5K)':>8} {'P(ruin)':>8} {'AvgDeaths':>10} "
              f"{'MedDD':>7}")
        for r in ranked:
            print(
                f"  {r['sim_id']:<8} "
                f"${r['final_balance']['median']:>7,.0f} "
                f"{r['profitable_pct']:>9.1f}% "
                f"{r['milestones']['$5K_target']['probability_pct']:>7.1f}% "
                f"{r['ruin_pct']:>7.1f}% "
                f"{r['deaths']['mean']:>9.1f} "
                f"{r['max_drawdown_pct']['median']:>6.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
