#!/usr/bin/env python3
"""
research/walk_forward.py
Walk-forward validation runner for all sim profiles.

Splits data into train (70%) / validate (15%) / test (15%) by date.
Runs the existing BacktestEngine on each split.
For sweeps: runs engine once to capture entries, then replays exit params.

Usage:
    python research/walk_forward.py                    # Run all sims baseline
    python research/walk_forward.py --sim SIM03        # Run one sim
    python research/walk_forward.py --sweep            # Sweep + optimize
    python research/walk_forward.py --sweep --sim SIM03
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import traceback
from itertools import product

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

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "simulation", "sim_config.yaml")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.json")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get(r, key, default=0):
    return r.get(key, default) if isinstance(r, dict) else getattr(r, key, default)


def _pnl(t):
    if isinstance(t, dict):
        return t.get("pnl") or t.get("realized_pnl_dollars", 0)
    return getattr(t, "pnl", 0)


def extract_metrics(summary) -> dict:
    """Extract metrics from a BacktestSummary."""
    all_trades = []
    for r in summary.runs:
        if isinstance(r, dict):
            trades_list = r.get("trades", [])
        else:
            trades_list = r.trades if isinstance(r.trades, list) else []
        all_trades.extend(trades_list if isinstance(trades_list, list) else [])

    total = len(all_trades)
    if total == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "expectancy": 0, "profit_factor": 0,
            "max_drawdown_pct": 0, "peak_balance": 500, "final_balance": 500,
        }

    wins = [t for t in all_trades if _pnl(t) > 0]
    losses = [t for t in all_trades if _pnl(t) <= 0]

    total_pnl = sum(_pnl(t) for t in all_trades)
    gross_profit = sum(_pnl(t) for t in wins)
    gross_loss = abs(sum(_pnl(t) for t in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    best_run = max(summary.runs, key=lambda r: _get(r, "peak_balance", 0))

    return {
        "trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total, 4) if total else 0,
        "total_pnl": round(total_pnl, 2),
        "expectancy": round(total_pnl / total, 2) if total else 0,
        "profit_factor": round(min(pf, 999), 2),
        "max_drawdown_pct": round(_get(best_run, "max_drawdown_pct", 0), 4),
        "peak_balance": round(_get(best_run, "peak_balance", 500), 2),
        "final_balance": round(_get(best_run, "final_balance", 500), 2),
    }


def run_single_backtest(sim_id: str, profile: dict, start: str, end: str,
                        verbose: bool = False, single_symbol: str = "SPY") -> dict:
    """Run a single backtest and return metrics dict.
    Uses single_symbol only to keep memory low (~1.5GB vs 4GB for 3 symbols).
    """
    bt_profile = copy.deepcopy(profile)
    if single_symbol:
        bt_profile["symbols"] = [single_symbol]

    engine = BacktestEngine(
        profile_id=sim_id,
        profile=bt_profile,
        start_date=start,
        end_date=end,
        max_runs=1,
        verbose=verbose,
        adaptive=False,
    )
    summary = engine.run()
    return extract_metrics(summary)


def replay_exits(trades: list[dict], sl: float, tp: float,
                 trail_act: float, trail_pct: float,
                 balance_start: float = 500, death_threshold: float = 25) -> dict:
    """
    Fast exit-param replay over pre-captured trades.

    Each trade dict has: entry_price, exit_price, pnl_pct, qty, direction, symbol,
    plus we use entry_price and exit_price ratio to estimate the option price path.

    For exit sweep, we recalculate PnL based on where the new SL/TP/trailing would have
    hit relative to the original entry_price and the actual pnl_pct (which tells us the
    full price trajectory).

    Since we don't have the full bar-by-bar option price path, we use a simplified model:
    - If the trade's max favorable excursion (MFE) >= TP -> profit target hit
    - If the trade's max adverse excursion (MAE) >= SL -> stop loss hit
    - Otherwise use the original exit

    We estimate MFE/MAE from the original pnl_pct and exit_reason.
    """
    balance = balance_start
    peak_balance = balance_start
    wins = 0
    total = 0
    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades:
        if balance <= death_threshold:
            break

        entry_price = t.get("entry_price", 0)
        orig_exit_price = t.get("exit_price", 0)
        qty = t.get("qty", 1)
        if entry_price <= 0 or qty <= 0:
            continue

        orig_pnl_pct = t.get("pnl_pct", 0)
        exit_reason = t.get("exit_reason", "")

        # Estimate max favorable/adverse excursion from exit info
        # If original exit was profit_target: MFE >= original TP
        # If stop_loss: MAE >= original SL
        # If trailing_stop: MFE >= trail activation, then gave back trail_pct
        # For simplicity: estimate MFE and MAE
        if orig_pnl_pct > 0:
            mfe = max(orig_pnl_pct * 1.1, orig_pnl_pct)  # at least hit exit price
            mae = min(0.02, abs(orig_pnl_pct) * 0.3)  # small dip before winning
        else:
            mfe = max(0, orig_pnl_pct + abs(orig_pnl_pct) * 0.3)  # small bounce before losing
            mae = abs(orig_pnl_pct)

        # Apply new exit rules
        if mae >= sl:
            # Stop loss would have triggered
            new_pnl_pct = -sl
        elif mfe >= tp:
            # Profit target would have triggered
            new_pnl_pct = tp
        elif trail_act > 0 and mfe >= trail_act:
            # Trailing stop activated, give back trail_pct from peak
            new_pnl_pct = max(mfe * (1 - trail_pct) - trail_pct, mfe - trail_pct)
            new_pnl_pct = max(new_pnl_pct, 0)  # trailing shouldn't lose if activated in profit
        else:
            # No exit rule triggered before original exit
            new_pnl_pct = orig_pnl_pct

        # Apply slippage (1% each way already baked into entry/exit prices)
        pnl_dollars = entry_price * new_pnl_pct * qty * 100
        balance += pnl_dollars
        total_pnl += pnl_dollars
        total += 1

        if pnl_dollars > 0:
            wins += 1
            gross_profit += pnl_dollars
        else:
            gross_loss += abs(pnl_dollars)

        if balance > peak_balance:
            peak_balance = balance

    pf = (gross_profit / gross_loss) if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    return {
        "trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total, 4) if total else 0,
        "total_pnl": round(total_pnl, 2),
        "expectancy": round(total_pnl / total, 2) if total else 0,
        "profit_factor": round(min(pf, 999), 2),
        "peak_balance": round(peak_balance, 2),
        "final_balance": round(balance, 2),
    }


def generate_param_grid(profile: dict) -> list[dict]:
    """Generate a compact parameter sweep grid."""
    horizon = profile.get("horizon", "intraday")
    hold_max = float(profile.get("hold_max_seconds", 3600))

    if horizon == "scalp" or hold_max <= 600:
        sl_range = [0.06, 0.08, 0.10, 0.15]
        tp_range = [0.12, 0.15, 0.20, 0.30]
        trail_act = [0.04, 0.06, 0.08]
        trail_pct = [0.02, 0.03, 0.04]
    elif horizon == "swing" or hold_max > 7200:
        sl_range = [0.15, 0.20, 0.25, 0.30]
        tp_range = [0.30, 0.40, 0.50, 0.60]
        trail_act = [0.08, 0.12, 0.15]
        trail_pct = [0.04, 0.06, 0.08]
    else:  # intraday
        sl_range = [0.08, 0.10, 0.12, 0.15, 0.20]
        tp_range = [0.15, 0.20, 0.25, 0.30, 0.40]
        trail_act = [0.05, 0.06, 0.08, 0.10]
        trail_pct = [0.03, 0.04, 0.05]

    grid = []
    for sl, tp, ta, tt in product(sl_range, tp_range, trail_act, trail_pct):
        if tp <= sl:
            continue
        if ta >= tp:
            continue
        grid.append({
            "stop_loss_pct": sl,
            "profit_target_pct": tp,
            "trailing_stop_activate_pct": ta,
            "trailing_stop_trail_pct": tt,
        })

    if len(grid) > 150:
        import random
        random.seed(42)
        grid = random.sample(grid, 150)

    return grid


def score_result(result: dict) -> float:
    """Score: expectancy * sqrt(trade_count). Higher = better."""
    trades = result.get("trades", 0)
    expectancy = result.get("expectancy", 0)
    if trades < 5 or expectancy <= 0:
        return -999
    return expectancy * math.sqrt(trades)


def run_sim_walkforward(sim_id: str, profile: dict, sweep: bool = False,
                        verbose: bool = False) -> dict:
    """Run walk-forward validation for a single sim."""
    print(f"\n{'='*60}")
    print(f"  {sim_id}: {profile.get('signal_mode', '?')}")
    print(f"{'='*60}")

    result = {"sim_id": sim_id, "signal_mode": profile.get("signal_mode", "?")}

    # ── Baseline run on all three splits ──
    print(f"  Running TRAIN ({TRAIN_START} to {TRAIN_END})...", flush=True)
    train_result = run_single_backtest(sim_id, profile, TRAIN_START, TRAIN_END, verbose)
    result["train_baseline"] = train_result
    print(f"    Train: {train_result['trades']} trades, WR={train_result['win_rate']:.1%}, "
          f"PnL=${train_result['total_pnl']:.0f}, E[t]=${train_result['expectancy']:.2f}", flush=True)

    print(f"  Running VALIDATE ({VAL_START} to {VAL_END})...", flush=True)
    val_result = run_single_backtest(sim_id, profile, VAL_START, VAL_END, verbose)
    result["validate_baseline"] = val_result
    print(f"    Val:   {val_result['trades']} trades, WR={val_result['win_rate']:.1%}, "
          f"PnL=${val_result['total_pnl']:.0f}, E[t]=${val_result['expectancy']:.2f}", flush=True)

    print(f"  Running TEST ({TEST_START} to {TEST_END})...", flush=True)
    test_result = run_single_backtest(sim_id, profile, TEST_START, TEST_END, verbose)
    result["test_baseline"] = test_result
    print(f"    Test:  {test_result['trades']} trades, WR={test_result['win_rate']:.1%}, "
          f"PnL=${test_result['total_pnl']:.0f}, E[t]=${test_result['expectancy']:.2f}", flush=True)

    if not sweep:
        result["optimized"] = False
        return result

    # ── Fast exit-param sweep using trade replay ──
    if train_result["trades"] < 10:
        print(f"  SKIP sweep: too few train trades ({train_result['trades']})")
        result["optimized"] = False
        result["skip_reason"] = "too_few_train_trades"
        return result

    # Extract trades from baseline runs for fast exit-param replay
    # We already have the baseline results, but need the raw trade lists
    # Re-run once to capture trades (uses cached stock/option data, fast)
    def _extract_trades(start, end):
        bt_profile = copy.deepcopy(profile)
        bt_profile["symbols"] = ["SPY"]
        eng = BacktestEngine(
            profile_id=sim_id, profile=bt_profile,
            start_date=start, end_date=end,
            max_runs=1, verbose=False, adaptive=False,
        )
        s = eng.run()
        trades = []
        for r in s.runs:
            if isinstance(r, dict):
                trades.extend(r.get("trades", []))
            else:
                trades.extend(r.trades if isinstance(r.trades, list) else [])
        return trades

    print(f"  Extracting trade data for replay...", flush=True)
    train_trades = _extract_trades(TRAIN_START, TRAIN_END)
    val_trades = _extract_trades(VAL_START, VAL_END)
    test_trades = _extract_trades(TEST_START, TEST_END)

    grid = generate_param_grid(profile)
    print(f"  Sweeping {len(grid)} param combos via replay...", flush=True)

    baseline_score = score_result(train_result)
    best_score = baseline_score
    best_params = None
    best_train = train_result

    for i, params in enumerate(grid):
        sweep_result = replay_exits(
            train_trades,
            sl=params["stop_loss_pct"],
            tp=params["profit_target_pct"],
            trail_act=params["trailing_stop_activate_pct"],
            trail_pct=params["trailing_stop_trail_pct"],
        )

        s = score_result(sweep_result)
        if s > best_score:
            best_score = s
            best_params = params
            best_train = sweep_result

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(grid)}] best: "
                  f"WR={best_train['win_rate']:.1%} E=${best_train['expectancy']:.2f}", flush=True)

    if best_params is None:
        print(f"  No improvement found over baseline.")
        result["optimized"] = False
        return result

    print(f"  Best params: SL={best_params['stop_loss_pct']} TP={best_params['profit_target_pct']} "
          f"Trail={best_params['trailing_stop_activate_pct']}/{best_params['trailing_stop_trail_pct']}")
    print(f"  Train: WR={best_train['win_rate']:.1%}, E=${best_train['expectancy']:.2f}, "
          f"PnL=${best_train['total_pnl']:.0f}", flush=True)

    # Validate best params via replay
    opt_val = replay_exits(
        val_trades,
        sl=best_params["stop_loss_pct"],
        tp=best_params["profit_target_pct"],
        trail_act=best_params["trailing_stop_activate_pct"],
        trail_pct=best_params["trailing_stop_trail_pct"],
    )
    print(f"    Val (opt): WR={opt_val['win_rate']:.1%}, E=${opt_val['expectancy']:.2f}, "
          f"PnL=${opt_val['total_pnl']:.0f}", flush=True)

    # Check for overfit
    overfit = False
    if best_train["win_rate"] > 0:
        wr_drop = (best_train["win_rate"] - opt_val["win_rate"]) / best_train["win_rate"]
        if wr_drop > 0.30:
            overfit = True
            print(f"  WARNING OVERFIT: train WR={best_train['win_rate']:.1%} -> "
                  f"val WR={opt_val['win_rate']:.1%} (dropped {wr_drop:.0%})")

    # Test with optimized params
    opt_test = replay_exits(
        test_trades,
        sl=best_params["stop_loss_pct"],
        tp=best_params["profit_target_pct"],
        trail_act=best_params["trailing_stop_activate_pct"],
        trail_pct=best_params["trailing_stop_trail_pct"],
    )
    print(f"    Test (opt): WR={opt_test['win_rate']:.1%}, E=${opt_test['expectancy']:.2f}, "
          f"PnL=${opt_test['total_pnl']:.0f}", flush=True)

    # Run FULL backtest with optimized params on validate+test (ground truth)
    print(f"  Running FULL validate backtest with optimized params...", flush=True)
    opt_profile = copy.deepcopy(profile)
    opt_profile.update(best_params)
    full_val = run_single_backtest(sim_id, opt_profile, VAL_START, VAL_END, verbose=False, single_symbol="SPY")
    print(f"    Val (full): WR={full_val['win_rate']:.1%}, E=${full_val['expectancy']:.2f}, "
          f"PnL=${full_val['total_pnl']:.0f}", flush=True)

    full_test = run_single_backtest(sim_id, opt_profile, TEST_START, TEST_END, verbose=False, single_symbol="SPY")
    print(f"    Test (full): WR={full_test['win_rate']:.1%}, E=${full_test['expectancy']:.2f}, "
          f"PnL=${full_test['total_pnl']:.0f}", flush=True)

    # Re-check overfit with full numbers
    if best_train["win_rate"] > 0 and full_val["trades"] > 0:
        wr_drop = (best_train["win_rate"] - full_val["win_rate"]) / best_train["win_rate"]
        if wr_drop > 0.30:
            overfit = True

    result["optimized"] = True
    result["overfit"] = overfit
    result["best_params"] = best_params
    result["train_optimized"] = best_train
    result["validate_optimized"] = full_val
    result["test_optimized"] = full_test

    return result


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtester")
    parser.add_argument("--sim", type=str, help="Run only this sim (e.g. SIM03)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--verbose", action="store_true", help="Verbose backtest output")
    parser.add_argument("--dedup", action="store_true", help="Only test one sim per unique signal_mode")
    args = parser.parse_args()

    config = load_config()

    # Group sims by signal_mode
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
        # One representative per signal_mode
        for mode, sims in sorted(mode_groups.items()):
            sim_ids.append(sims[0])
    else:
        for sim_id in sorted(config.keys()):
            if not sim_id.startswith("SIM") or sim_id == "SIM00":
                continue
            if not isinstance(config[sim_id], dict):
                continue
            sim_ids.append(sim_id)

    print(f"Walk-Forward Validation")
    print(f"Train:    {TRAIN_START} to {TRAIN_END}")
    print(f"Validate: {VAL_START} to {VAL_END}")
    print(f"Test:     {TEST_START} to {TEST_END}")
    print(f"Sims:     {len(sim_ids)}")
    print(f"Sweep:    {'YES' if args.sweep else 'NO'}")
    sys.stdout.flush()

    results = {}
    for sim_id in sim_ids:
        profile = config[sim_id]
        try:
            r = run_sim_walkforward(sim_id, profile, sweep=args.sweep, verbose=args.verbose)
            results[sim_id] = r
        except Exception as e:
            print(f"\n  ERROR on {sim_id}: {e}")
            traceback.print_exc()
            results[sim_id] = {"sim_id": sim_id, "error": str(e)}

        # Save after each sim
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [saved]", flush=True)

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'SIM':<8} {'Mode':<25} {'Tr.WR':>6} {'Val.WR':>6} {'Tst.WR':>6} {'Trades':>6} {'Expect':>7} {'Overfit':>7}")
    print(f"{'-'*70}")

    for sim_id in sim_ids:
        r = results.get(sim_id, {})
        if "error" in r:
            print(f"{sim_id:<8} ERROR: {r['error'][:50]}")
            continue

        mode = r.get("signal_mode", "?")[:24]
        if r.get("optimized"):
            tr = r.get("train_optimized", {})
            va = r.get("validate_optimized", {})
            te = r.get("test_optimized", {})
            overfit = "YES" if r.get("overfit") else "no"
        else:
            tr = r.get("train_baseline", {})
            va = r.get("validate_baseline", {})
            te = r.get("test_baseline", {})
            overfit = "-"

        print(f"{sim_id:<8} {mode:<25} {tr.get('win_rate',0):>5.1%} {va.get('win_rate',0):>5.1%} "
              f"{te.get('win_rate',0):>5.1%} {te.get('trades',0):>6} {te.get('expectancy',0):>7.2f} {overfit:>7}")


if __name__ == "__main__":
    main()
