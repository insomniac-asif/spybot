"""
backtest/runner.py
CLI runner for the backtesting engine.

Usage examples:
  # Run all sims for a date range
  python -m backtest.runner --start 2024-01-01 --end 2024-12-31

  # Run specific sims
  python -m backtest.runner --sims SIM03 SIM11 --start 2024-06-01 --end 2024-12-31

  # Run a single sim, single symbol override
  python -m backtest.runner --sims SIM03 --symbol SPY --start 2024-01-01 --end 2024-12-31

  # Run with custom max_runs and be verbose
  python -m backtest.runner --sims SIM03 --max-runs 100 --start 2024-01-01 --end 2024-12-31 --verbose

  # Pre-fetch all required stock data first (useful for large date ranges)
  python -m backtest.runner --prefetch --start 2024-01-01 --end 2024-12-31

Options:
  --start         Start date (YYYY-MM-DD), required
  --end           End date (YYYY-MM-DD), required
  --sims          Sim IDs to run (e.g. SIM03 SIM11). Default: all sims.
  --symbol        Override symbol (default: use profile's first symbol)
  --max-runs      Maximum runs per sim (default: 50)
  --prefetch      Pre-download all stock data before running
  --no-verbose    Suppress per-bar progress output
"""
from __future__ import annotations
import argparse
import os
import sys
import time

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_profiles() -> dict:
    """Load sim profiles from sim_config.yaml, skipping _global and non-SIM keys."""
    import yaml
    cfg_path = os.path.join(_PROJECT_ROOT, "simulation", "sim_config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    profiles = {}
    for key, val in cfg.items():
        if str(key).startswith("_"):
            continue
        if not isinstance(val, dict):
            continue
        k = str(key).upper()
        if k.startswith("SIM") and k[3:].isdigit():
            profiles[k] = val
    return profiles


def _get_symbols_for_profile(profile: dict) -> list:
    """Return the list of tradeable symbols for a profile."""
    symbols_raw = profile.get("symbols")
    if symbols_raw and isinstance(symbols_raw, list):
        return [str(s).upper() for s in symbols_raw]
    if profile.get("symbol"):
        return [str(profile["symbol"]).upper()]
    if profile.get("underlying"):
        return [str(profile["underlying"]).upper()]
    return []


def main():
    parser = argparse.ArgumentParser(description="QQQBot Historical Backtest Runner")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--sims", nargs="*", help="Sim IDs to run (default: all)")
    parser.add_argument("--symbol", help="Override symbol (default: profile's first symbol)")
    parser.add_argument("--max-runs", type=int, default=0, help="Max runs per sim (0 = unlimited)")
    parser.add_argument("--prefetch", action="store_true", help="Pre-download stock data first")
    parser.add_argument("--no-verbose", action="store_true", help="Suppress verbose output")
    parser.add_argument("--skip-live", action="store_true", default=True,
                        help="Skip SIM00 (live execution sim). Default: True.")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive optimization (learn from each run)")
    parser.add_argument("--patterns", action="store_true",
                        help="Run pattern scanner on completed backtest trades")
    parser.add_argument("--growth", action="store_true",
                        help="Run growth analysis (milestones, PDT, streaks, Sharpe)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run parameter optimizer with walk-forward validation")
    parser.add_argument("--objective", default="growth",
                        choices=["growth", "winrate", "balanced", "expectancy"],
                        help="Optimizer objective function (default: growth)")
    args = parser.parse_args()

    verbose = not args.no_verbose
    start_date = args.start
    end_date = args.end

    mode_str = "ADAPTIVE" if args.adaptive else "STATIC"
    print(f"QQQBot Backtest Runner ({mode_str})")
    print(f"  Period: {start_date} -> {end_date}")
    print(f"  Max runs per sim: {'unlimited' if args.max_runs == 0 else args.max_runs}")
    if args.adaptive:
        print(f"  Adaptive mode: ON — filters evolve between runs")

    # Load profiles
    try:
        all_profiles = _load_profiles()
    except Exception as e:
        print(f"ERROR loading sim_config.yaml: {e}")
        sys.exit(1)

    # Filter to requested sims
    if args.sims:
        requested = [s.upper() for s in args.sims]
        profiles = {k: v for k, v in all_profiles.items() if k in requested}
        missing = [s for s in requested if s not in profiles]
        if missing:
            print(f"WARNING: Sims not found in config: {missing}")
    else:
        profiles = all_profiles

    # Skip live sim by default
    if args.skip_live and "SIM00" in profiles:
        print("  Skipping SIM00 (live sim). Use --sims to include it explicitly.")
        del profiles["SIM00"]

    if not profiles:
        print("No sims to run.")
        sys.exit(0)

    print(f"  Running {len(profiles)} sims: {list(profiles.keys())}")

    # Pre-fetch stock data if requested
    if args.prefetch:
        from backtest.data_fetcher import prefetch_stock_data
        # Collect all unique symbols
        all_symbols = set()
        for pid, prof in profiles.items():
            syms = _get_symbols_for_profile(prof)
            all_symbols.update(syms[:3])  # Only fetch first 3 per sim to limit API calls
        if args.symbol:
            all_symbols = {args.symbol.upper()}
        print(f"\nPre-fetching stock data for: {sorted(all_symbols)}")
        prefetch_stock_data(sorted(all_symbols), start_date, end_date)
        print("Pre-fetch complete.\n")

    # ── Optimize mode: run grid-search instead of normal backtest ──────
    if args.optimize:
        from backtest.optimizer import SimOptimizer
        total_start = time.time()
        for sim_id in profiles:
            optimizer = SimOptimizer(sim_id, start_date, end_date,
                                    objective=args.objective)
            result = optimizer.run()

            verdict = result.get("verdict", "?")
            verdict_reason = result.get("verdict_reason", "")
            print(f"\n  {'=' * 60}")
            print(f"  Verdict: {verdict}")
            print(f"  {verdict_reason}")
            print(f"\n  Top 10 Parameter Sets for {sim_id} "
                  f"(objective={args.objective}):")
            print(f"  {'Rank':>4} {'TP':>6} {'SL':>6} {'HoldMax':>8} "
                  f"{'TrainScore':>11} {'TestScore':>10} "
                  f"{'Consistency':>12} {'TestTrades':>11} {'TestPnL':>9} {'Overfit?':>9}")
            print(f"  {'-'*4} {'-'*6} {'-'*6} {'-'*8} "
                  f"{'-'*11} {'-'*10} {'-'*12} {'-'*11} {'-'*9} {'-'*9}")
            for entry in result.get("top_10", []):
                p = entry["params"]
                ovf = "YES" if entry["overfit_flag"] else "no"
                print(f"  {entry['rank']:>4} {p['tp']:>6.2f} {p['sl']:>6.2f} "
                      f"{p['hold_max']:>6}m "
                      f"{entry['avg_train_score']:>11.2f} "
                      f"{entry['avg_test_score']:>10.2f} "
                      f"{entry['consistency']*100:>10.0f}% "
                      f"{entry.get('total_test_trades', '?'):>11} "
                      f"${entry.get('avg_test_pnl', 0):>7.0f} "
                      f"{ovf:>9}")

            # Print fold details for the top combo
            top = result.get("top_10", [{}])
            if top and top[0].get("fold_details"):
                print(f"\n  Fold details for rank #1:")
                for fd in top[0]["fold_details"]:
                    pf_tag = "+" if fd.get("test_profitable") else "-"
                    print(f"    Fold {fd['fold']}: train={fd['train_trades']} trades "
                          f"(score {fd['train_score']:.2f}), "
                          f"test={fd['test_trades']} trades "
                          f"(score {fd['test_score']:.2f}, "
                          f"WR {fd.get('test_win_rate', 0)*100:.0f}%, "
                          f"PnL ${fd.get('test_pnl', 0):.0f}) [{pf_tag}]")

            bl = result.get("baseline_params", {})
            print(f"\n  Baseline: TP={bl.get('tp')} SL={bl.get('sl')} "
                  f"HoldMax={bl.get('hold_max')}m")
            print(f"  Saved to: backtest/results/optimizer_{sim_id}.json")

        total_elapsed = time.time() - total_start
        print(f"\nTotal optimization time: {total_elapsed:.1f}s")
        print("Done.")
        sys.exit(0)

    from backtest.engine import BacktestEngine
    from backtest.save_results import save_sim_summary, save_dashboard_data
    if args.patterns:
        from backtest.pattern_scanner import PatternScanner
    if args.growth:
        from backtest.growth_simulator import GrowthSimulator

    all_summaries = []
    total_start = time.time()

    for sim_id, profile in profiles.items():
        t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"Running {sim_id}: {profile.get('name', sim_id)}")
        print(f"  Signal mode: {profile.get('signal_mode', 'TREND_PULLBACK')}")

        # Apply symbol override if specified
        if args.symbol:
            profile = dict(profile)
            profile["symbols"] = [args.symbol.upper()]

        try:
            engine = BacktestEngine(
                profile_id=sim_id,
                profile=profile,
                start_date=start_date,
                end_date=end_date,
                max_runs=args.max_runs,
                verbose=verbose,
                adaptive=args.adaptive,
            )
            summary = engine.run()
            all_summaries.append(summary)

            # Save per-sim file
            sim_path = save_sim_summary(summary)
            elapsed = time.time() - t0

            print(f"\n  Results for {sim_id}:")
            print(f"    Runs completed: {summary.total_runs}")
            print(f"    Blown accounts: {summary.blown_count}")
            print(f"    Target hits:    {summary.target_hit_count}")
            print(f"    Avg win rate:   {summary.avg_win_rate * 100:.1f}%")
            print(f"    Avg drawdown:   {summary.avg_max_drawdown * 100:.1f}%")
            print(f"    Saved to:       {sim_path}")
            print(f"    Time:           {elapsed:.1f}s")

            if args.adaptive and engine.adapt_filters:
                af = engine.adapt_filters
                dn = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
                print(f"    Adaptive gen:   {af.generation} | Best peak: ${af.best_run_peak:.0f}")
                if af.blocked_hours: print(f"    Blocked hours:  {sorted(af.blocked_hours)}")
                if af.allowed_hours is not None: print(f"    Allowed hours:  {sorted(af.allowed_hours)}")
                if af.blocked_days: print(f"    Blocked days:   {[dn.get(d,d) for d in sorted(af.blocked_days)]}")
                if af.allowed_days is not None: print(f"    Allowed days:   {[dn.get(d,d) for d in sorted(af.allowed_days)]}")
                if af.blocked_direction: print(f"    Blocked dir:    {af.blocked_direction}")
                if af.required_direction: print(f"    Required dir:   {af.required_direction}")
                if af.max_hold_seconds: print(f"    Max hold:       {af.max_hold_seconds}s ({af.max_hold_seconds//60}min)")

            # Pattern scanning
            if args.patterns:
                all_trades = []
                for r in (summary.runs if hasattr(summary, "runs") else []):
                    trades_list = r.get("trades", []) if isinstance(r, dict) else (r.trades if hasattr(r, "trades") else [])
                    for t in trades_list:
                        td = dict(t) if isinstance(t, dict) else {
                            "entry_time": t.entry_time, "exit_time": t.exit_time,
                            "direction": t.direction, "realized_pnl_dollars": t.pnl,
                            "holding_seconds": t.holding_seconds, "symbol": t.symbol,
                            "signal_mode": t.signal_mode, "exit_reason": t.exit_reason,
                        }
                        td.setdefault("realized_pnl_dollars", td.get("pnl"))
                        all_trades.append(td)

                scanner = PatternScanner()
                patterns = scanner.scan(all_trades, sim_id)
                if patterns:
                    print(f"\n  Top Patterns for {sim_id} ({len(patterns)} found):")
                    print(f"  {'Rank':<5} {'Pattern':<40} {'WinRate':>8} {'PF':>6} {'Trades':>7} {'AvgPnL':>8} {'Freq/wk':>8} {'Recurrence':<22}")
                    print(f"  {'-'*5} {'-'*40} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*22}")
                    for i, p in enumerate(patterns[:10], 1):
                        desc = p["description"][:40]
                        ra = p.get("recurrence_analysis", {})
                        interval = ra.get("primary_interval", "?")
                        reg = ra.get("regularity_score", 0)
                        recur_str = f"{interval} ({reg:.2f})"
                        print(f"  {i:<5} {desc:<40} {p['win_rate']*100:>7.1f}% {p['profit_factor']:>6.2f} {p['total_trades']:>7} {p['avg_pnl']:>7.2f} {p['trades_per_week']:>7.2f} {recur_str:<22}")
                    print(f"  Saved to: backtest/results/patterns_{sim_id}.json")
                else:
                    print(f"\n  No patterns meeting thresholds for {sim_id}")

            # Growth analysis
            if args.growth:
                all_trades_g = []
                all_equity = []
                for r in (summary.runs if hasattr(summary, "runs") else []):
                    trades_list = r.get("trades", []) if isinstance(r, dict) else (r.trades if hasattr(r, "trades") else [])
                    eq_curve = r.get("equity_curve", []) if isinstance(r, dict) else (r.equity_curve if hasattr(r, "equity_curve") else [])
                    for t in trades_list:
                        td = dict(t) if isinstance(t, dict) else {
                            "entry_time": t.entry_time, "exit_time": t.exit_time,
                            "direction": t.direction, "realized_pnl_dollars": t.pnl,
                            "holding_seconds": t.holding_seconds, "symbol": t.symbol,
                            "signal_mode": t.signal_mode, "exit_reason": t.exit_reason,
                            "run_number": t.run_number,
                        }
                        td.setdefault("realized_pnl_dollars", td.get("pnl"))
                        td.setdefault("run_number", r.get("run_number", 1) if isinstance(r, dict) else (r.run_number if hasattr(r, "run_number") else 1))
                        all_trades_g.append(td)
                    all_equity.extend(eq_curve)

                gs = GrowthSimulator()
                growth = gs.analyze(summary, all_trades_g, all_equity)

                ec = growth["end_capital"]
                print(f"\n  Growth Analysis for {sim_id}:")
                print(f"    ${growth['start_capital']:.0f} -> ${ec:,.0f} | {growth['total_trades']} trades | {growth['win_rate']*100:.0f}% WR | {growth['deaths']} deaths")

                ms_parts = []
                for m_val in [1000, 2500, 5000]:
                    m_data = growth["milestones"].get(str(m_val))
                    if m_data:
                        ms_parts.append(f"${m_val//1000}K in {m_data['days']}d ({m_data['trades']} trades)")
                    else:
                        ms_parts.append(f"${m_val//1000}K: NOT REACHED")
                print(f"    Milestones: {', '.join(ms_parts)}")

                pdt = growth["pdt_violations"]
                pdt_str = f"{pdt} (would need cash account or PDT awareness)" if pdt > 0 else "0"
                print(f"    PDT violations: {pdt_str}")
                print(f"    Max DD: {growth['max_drawdown_pct']*100:.1f}% | Sharpe: {growth['daily_sharpe']:.1f} | Best streak: {growth['best_win_streak']}W | Worst: {growth['worst_loss_streak']}L")

                bd = growth["best_day"]
                wd = growth["worst_day"]
                if bd["date"]:
                    print(f"    Best day: {bd['date']} (${bd['pnl']:+.2f}) | Worst day: {wd['date']} (${wd['pnl']:+.2f})")
                print(f"    Saved to: backtest/results/growth_{sim_id}.json")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"  ERROR running {sim_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save combined dashboard data
    if all_summaries:
        try:
            dashboard_path = save_dashboard_data(all_summaries)
            print(f"\n{'=' * 60}")
            print(f"Dashboard data saved to: {dashboard_path}")
            print(f"View on dashboard (Backtest tab)")
        except Exception as e:
            print(f"ERROR saving dashboard data: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f"\nTotal backtest time: {total_elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
