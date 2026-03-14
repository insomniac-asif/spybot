"""
backtest/optimizer.py
Analyzes backtest results to find patterns that separate winning from losing trades,
then generates optimized entry filters and strategy recommendations.

Usage:
    python -m backtest.optimizer --sim SIM03
    python -m backtest.optimizer --all
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime as _dt, timedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Grade thresholds (same as dashboard JS) ──────────────────────────────
def grade_trade(pnl_pct: float, exit_reason: str = "") -> tuple[str, str]:
    """Return (letter, tier) for a trade based on PnL%."""
    p = pnl_pct * 100  # convert to percentage
    reason = exit_reason.lower()
    if p >= 80:  return "A+", "A"
    if p >= 40:  return "A",  "A"
    if p >= 15 and ("profit" in reason or "tp" in reason): return "B+", "B"
    if p >= 5:   return "B",  "B"
    if p >= 0:   return "C",  "C"
    if p >= -10: return "C-", "C"
    if p >= -25: return "D",  "D"
    if p >= -50: return "D-", "D"
    return "F", "F"


@dataclass
class DimensionStats:
    """Stats for one value within a dimension (e.g., hour=10, day=Monday)."""
    label: str
    total: int = 0
    wins: int = 0
    losses: int = 0
    a_trades: int = 0
    f_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    a_rate: float = 0.0
    f_rate: float = 0.0
    avg_holding_mins: float = 0.0


@dataclass
class OptimizationResult:
    """Full optimization analysis for one sim."""
    sim_id: str
    signal_mode: str
    symbol: str
    total_trades: int
    overall_win_rate: float
    overall_a_rate: float
    overall_f_rate: float

    # Per-dimension analysis
    by_hour: list[DimensionStats] = field(default_factory=list)
    by_day: list[DimensionStats] = field(default_factory=list)
    by_direction: list[DimensionStats] = field(default_factory=list)
    by_regime: list[DimensionStats] = field(default_factory=list)
    by_exit_reason: list[DimensionStats] = field(default_factory=list)
    by_time_slot: list[DimensionStats] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    suggested_filters: dict = field(default_factory=dict)

    # A-trade profile
    a_trade_profile: dict = field(default_factory=dict)


def _load_backtest_data(sim_id: str) -> dict | None:
    """Load backtest dashboard data for a sim."""
    path = os.path.join(RESULTS_DIR, "dashboard_data.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(sim_id.upper())


def _compute_dimension_stats(trades: list[dict], key_fn, label_fn=None) -> list[DimensionStats]:
    """Group trades by a key function and compute stats for each group."""
    groups = defaultdict(list)
    for t in trades:
        k = key_fn(t)
        if k is not None:
            groups[k].append(t)

    results = []
    for k in sorted(groups.keys()):
        group = groups[k]
        label = label_fn(k) if label_fn else str(k)
        total = len(group)
        wins = sum(1 for t in group if (t.get("realized_pnl_dollars") or t.get("pnl") or 0) > 0)
        losses = total - wins
        a_count = sum(1 for t in group if grade_trade(t.get("pnl_pct", 0), t.get("exit_reason", ""))[1] == "A")
        f_count = sum(1 for t in group if grade_trade(t.get("pnl_pct", 0), t.get("exit_reason", ""))[1] == "F")
        total_pnl = sum(t.get("realized_pnl_dollars") or t.get("pnl") or 0 for t in group)
        avg_pnl_pct = sum(t.get("pnl_pct", 0) for t in group) / total if total else 0
        avg_hold = sum(t.get("holding_seconds", 0) for t in group) / total / 60 if total else 0

        results.append(DimensionStats(
            label=label,
            total=total,
            wins=wins,
            losses=losses,
            a_trades=a_count,
            f_trades=f_count,
            total_pnl=round(total_pnl, 2),
            avg_pnl_pct=round(avg_pnl_pct, 4),
            win_rate=round(wins / total, 4) if total else 0,
            a_rate=round(a_count / total, 4) if total else 0,
            f_rate=round(f_count / total, 4) if total else 0,
            avg_holding_mins=round(avg_hold, 1),
        ))
    return results


def _time_slot(hour: int, minute: int = 0) -> str:
    """Categorize time into slots."""
    t = hour * 60 + minute
    if t < 600:   return "pre-market"
    if t < 630:   return "open-rush (9:30-10:00)"
    if t < 690:   return "morning (10:00-11:30)"
    if t < 780:   return "midday (11:30-13:00)"
    if t < 870:   return "afternoon (13:00-14:30)"
    if t < 945:   return "power-hour (14:30-15:45)"
    return "close (15:45-16:00)"


def analyze_sim(sim_id: str) -> OptimizationResult | None:
    """Run full optimization analysis on a sim's backtest data."""
    entry = _load_backtest_data(sim_id)
    if not entry:
        return None

    summary = entry.get("summary", {})
    trades = entry.get("trade_log", [])
    if not trades:
        return None

    signal_mode = summary.get("signal_mode", "")
    symbol = summary.get("symbol", "")
    total = len(trades)
    wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
    a_trades = sum(1 for t in trades if grade_trade(t.get("pnl_pct", 0), t.get("exit_reason", ""))[1] == "A")
    f_trades = sum(1 for t in trades if grade_trade(t.get("pnl_pct", 0), t.get("exit_reason", ""))[1] == "F")

    # ── Dimension analysis ────────────────────────────────────────────────
    by_hour = _compute_dimension_stats(
        trades,
        key_fn=lambda t: t.get("entry_hour", _parse_hour(t.get("entry_time", ""))),
        label_fn=lambda h: f"{h}:00-{h}:59",
    )

    by_day = _compute_dimension_stats(
        trades,
        key_fn=lambda t: t.get("day_of_week", _parse_dow(t.get("entry_time", ""))),
        label_fn=lambda d: DAY_NAMES[d] if 0 <= d <= 6 else str(d),
    )

    by_direction = _compute_dimension_stats(
        trades,
        key_fn=lambda t: (t.get("direction") or "").upper(),
        label_fn=lambda d: d,
    )

    by_regime = _compute_dimension_stats(
        trades,
        key_fn=lambda t: t.get("regime") or "UNKNOWN",
        label_fn=lambda r: r,
    )

    by_exit_reason = _compute_dimension_stats(
        trades,
        key_fn=lambda t: (t.get("exit_reason") or "unknown").replace("_", " "),
        label_fn=lambda r: r,
    )

    by_time_slot = _compute_dimension_stats(
        trades,
        key_fn=lambda t: _time_slot(
            t.get("entry_hour", _parse_hour(t.get("entry_time", ""))),
            t.get("entry_minute", 0),
        ),
        label_fn=lambda s: s,
    )

    # ── Generate recommendations ──────────────────────────────────────────
    recs = []
    suggested = {}
    overall_wr = wins / total if total else 0

    # Hour analysis: find best and worst hours
    profitable_hours = [h for h in by_hour if h.win_rate > overall_wr + 0.10 and h.total >= 5]
    losing_hours = [h for h in by_hour if h.win_rate < overall_wr - 0.10 and h.total >= 5]

    if profitable_hours:
        best = max(profitable_hours, key=lambda h: h.win_rate)
        recs.append(f"BEST HOUR: {best.label} — {best.win_rate*100:.0f}% WR ({best.total} trades, {best.a_trades} A-grades)")

    if losing_hours:
        worst = min(losing_hours, key=lambda h: h.win_rate)
        recs.append(f"WORST HOUR: {worst.label} — {worst.win_rate*100:.0f}% WR ({worst.total} trades, {worst.f_trades} F-grades). Consider blocking.")
        suggested["block_hours"] = [h.label.split(":")[0] for h in losing_hours if h.win_rate < 0.25]

    # Time slot analysis
    good_slots = [s for s in by_time_slot if s.win_rate > overall_wr + 0.08 and s.total >= 10]
    bad_slots = [s for s in by_time_slot if s.win_rate < overall_wr - 0.08 and s.total >= 10]
    if good_slots:
        recs.append(f"BEST TIME SLOTS: {', '.join(s.label for s in good_slots)} (avg WR: {sum(s.win_rate for s in good_slots)/len(good_slots)*100:.0f}%)")
    if bad_slots:
        recs.append(f"AVOID TIME SLOTS: {', '.join(s.label for s in bad_slots)} (avg WR: {sum(s.win_rate for s in bad_slots)/len(bad_slots)*100:.0f}%)")
        suggested["block_time_slots"] = [s.label for s in bad_slots]

    # Day analysis
    good_days = [d for d in by_day if d.win_rate > overall_wr + 0.10 and d.total >= 10]
    bad_days = [d for d in by_day if d.win_rate < overall_wr - 0.10 and d.total >= 10]
    if bad_days:
        recs.append(f"WORST DAYS: {', '.join(d.label for d in bad_days)} — avg WR {sum(d.win_rate for d in bad_days)/len(bad_days)*100:.0f}%. Consider skipping.")
        suggested["block_days"] = [d.label for d in bad_days]
    if good_days:
        recs.append(f"BEST DAYS: {', '.join(d.label for d in good_days)} — avg WR {sum(d.win_rate for d in good_days)/len(good_days)*100:.0f}%")

    # Direction analysis
    for ds in by_direction:
        if ds.total >= 10:
            if ds.win_rate < 0.20 and ds.f_rate > 0.40:
                recs.append(f"DIRECTION {ds.label}: only {ds.win_rate*100:.0f}% WR with {ds.f_rate*100:.0f}% F-rate. Consider blocking {ds.label} signals.")
                suggested["block_direction"] = ds.label
            elif ds.win_rate > overall_wr + 0.15:
                recs.append(f"DIRECTION {ds.label}: strong at {ds.win_rate*100:.0f}% WR. Consider {ds.label}-only mode.")

    # Regime analysis
    for rs in by_regime:
        if rs.total >= 10:
            if rs.win_rate < 0.20 and rs.f_rate > 0.40:
                recs.append(f"REGIME {rs.label}: {rs.win_rate*100:.0f}% WR, {rs.f_rate*100:.0f}% F-rate. Block this regime.")
                suggested.setdefault("block_regimes", []).append(rs.label)
            elif rs.win_rate > overall_wr + 0.15:
                recs.append(f"REGIME {rs.label}: {rs.win_rate*100:.0f}% WR — favorable regime.")

    # Exit reason analysis
    for er in by_exit_reason:
        if er.total >= 5 and er.avg_pnl_pct < -0.15:
            recs.append(f"EXIT '{er.label}': avg PnL {er.avg_pnl_pct*100:.1f}% across {er.total} trades. Review this exit trigger.")

    # Holding time analysis
    winning_trades = [t for t in trades if (t.get("pnl") or 0) > 0]
    losing_trades = [t for t in trades if (t.get("pnl") or 0) <= 0]
    if winning_trades and losing_trades:
        avg_win_hold = sum(t.get("holding_seconds", 0) for t in winning_trades) / len(winning_trades)
        avg_loss_hold = sum(t.get("holding_seconds", 0) for t in losing_trades) / len(losing_trades)
        if avg_win_hold > 0 and avg_loss_hold > 0:
            recs.append(f"HOLDING TIME: Winners avg {avg_win_hold/60:.0f}min, Losers avg {avg_loss_hold/60:.0f}min.")
            if avg_loss_hold > avg_win_hold * 1.5:
                recs.append(f"  → Losers held {avg_loss_hold/avg_win_hold:.1f}x longer than winners. Consider tighter hold_max.")
                suggested["hold_max_seconds"] = int(avg_win_hold * 1.5)

    # ── A-trade profile ───────────────────────────────────────────────────
    a_trade_list = [t for t in trades if grade_trade(t.get("pnl_pct", 0), t.get("exit_reason", ""))[1] == "A"]
    a_profile = {}
    if a_trade_list:
        a_hours = defaultdict(int)
        a_days = defaultdict(int)
        a_dirs = defaultdict(int)
        a_regimes = defaultdict(int)
        a_slots = defaultdict(int)
        for t in a_trade_list:
            h = t.get("entry_hour", _parse_hour(t.get("entry_time", "")))
            a_hours[h] += 1
            d = t.get("day_of_week", _parse_dow(t.get("entry_time", "")))
            a_days[DAY_NAMES[d] if 0 <= d <= 6 else str(d)] += 1
            a_dirs[(t.get("direction") or "").upper()] += 1
            a_regimes[t.get("regime") or "UNKNOWN"] += 1
            slot = _time_slot(h, t.get("entry_minute", 0))
            a_slots[slot] += 1

        a_count = len(a_trade_list)
        a_profile = {
            "count": a_count,
            "pct_of_total": round(a_count / total * 100, 1),
            "avg_pnl_pct": round(sum(t.get("pnl_pct", 0) for t in a_trade_list) / a_count * 100, 1),
            "top_hours": sorted(a_hours.items(), key=lambda x: -x[1])[:3],
            "top_days": sorted(a_days.items(), key=lambda x: -x[1])[:3],
            "direction_split": dict(a_dirs),
            "top_regimes": sorted(a_regimes.items(), key=lambda x: -x[1])[:3],
            "top_time_slots": sorted(a_slots.items(), key=lambda x: -x[1])[:3],
            "avg_holding_mins": round(sum(t.get("holding_seconds", 0) for t in a_trade_list) / a_count / 60, 1),
        }

        recs.append("")
        recs.append(f"═══ A-TRADE PROFILE ({a_count} trades, {a_profile['pct_of_total']}% of total) ═══")
        recs.append(f"  Avg PnL: +{a_profile['avg_pnl_pct']}%")
        recs.append(f"  Avg hold: {a_profile['avg_holding_mins']} min")
        recs.append(f"  Best hours: {', '.join(f'{h}:00 ({c}x)' for h, c in a_profile['top_hours'])}")
        recs.append(f"  Best days: {', '.join(f'{d} ({c}x)' for d, c in a_profile['top_days'])}")
        recs.append(f"  Direction: {', '.join(f'{d}: {c}' for d, c in a_profile['direction_split'].items())}")
        recs.append(f"  Best slots: {', '.join(f'{s} ({c}x)' for s, c in a_profile['top_time_slots'])}")

    # ── What-if: A-only filter estimate ───────────────────────────────────
    if a_trade_list:
        a_total_pnl = sum(t.get("pnl") or 0 for t in a_trade_list)
        all_total_pnl = sum(t.get("pnl") or 0 for t in trades)
        recs.append("")
        recs.append(f"═══ WHAT-IF: A-TRADES ONLY ═══")
        recs.append(f"  Would take {a_count}/{total} trades ({a_count/total*100:.0f}%)")
        recs.append(f"  Total PnL: ${a_total_pnl:.2f} (vs ${all_total_pnl:.2f} taking all)")
        recs.append(f"  Improvement: {'+' if a_total_pnl > all_total_pnl else ''}{a_total_pnl - all_total_pnl:.2f}")

    if not recs:
        recs.append("Not enough data for meaningful recommendations.")

    return OptimizationResult(
        sim_id=sim_id,
        signal_mode=signal_mode,
        symbol=symbol,
        total_trades=total,
        overall_win_rate=round(overall_wr, 4),
        overall_a_rate=round(a_trades / total, 4) if total else 0,
        overall_f_rate=round(f_trades / total, 4) if total else 0,
        by_hour=by_hour,
        by_day=by_day,
        by_direction=by_direction,
        by_regime=by_regime,
        by_exit_reason=by_exit_reason,
        by_time_slot=by_time_slot,
        recommendations=recs,
        suggested_filters=suggested,
        a_trade_profile=a_profile,
    )


def generate_yaml_filters(result: OptimizationResult) -> str:
    """Generate sim_config.yaml quality_filters block from optimization results."""
    lines = [f"  # Auto-generated filters for {result.sim_id} ({result.signal_mode})"]
    lines.append(f"  # Based on {result.total_trades} backtest trades, {result.overall_win_rate*100:.1f}% WR")
    lines.append(f"  quality_filters:")

    sf = result.suggested_filters

    if sf.get("block_hours"):
        hours = sf["block_hours"]
        lines.append(f"    block_hours: [{', '.join(str(h) for h in hours)}]  # hours with <25% WR")

    if sf.get("block_days"):
        days = sf["block_days"]
        lines.append(f"    block_days: [{', '.join(d for d in days)}]")

    if sf.get("block_direction"):
        lines.append(f"    block_direction: {sf['block_direction']}  # direction with poor WR")

    if sf.get("block_regimes"):
        regimes = sf["block_regimes"]
        lines.append(f"    block_regimes: [{', '.join(r for r in regimes)}]")

    if sf.get("block_time_slots"):
        slots = sf["block_time_slots"]
        lines.append(f"    # Consider blocking: {', '.join(slots)}")

    if sf.get("hold_max_seconds"):
        lines.append(f"    hold_max_seconds: {sf['hold_max_seconds']}  # based on winner avg hold time")

    return "\n".join(lines)


def result_to_dict(result: OptimizationResult) -> dict:
    """Convert result to JSON-serializable dict for dashboard."""
    def _dim_list(dims):
        return [
            {
                "label": d.label, "total": d.total, "wins": d.wins,
                "a_trades": d.a_trades, "f_trades": d.f_trades,
                "total_pnl": d.total_pnl, "avg_pnl_pct": d.avg_pnl_pct,
                "win_rate": d.win_rate, "a_rate": d.a_rate, "f_rate": d.f_rate,
                "avg_holding_mins": d.avg_holding_mins,
            }
            for d in dims
        ]

    return {
        "sim_id": result.sim_id,
        "signal_mode": result.signal_mode,
        "symbol": result.symbol,
        "total_trades": result.total_trades,
        "overall_win_rate": result.overall_win_rate,
        "overall_a_rate": result.overall_a_rate,
        "overall_f_rate": result.overall_f_rate,
        "by_hour": _dim_list(result.by_hour),
        "by_day": _dim_list(result.by_day),
        "by_direction": _dim_list(result.by_direction),
        "by_regime": _dim_list(result.by_regime),
        "by_exit_reason": _dim_list(result.by_exit_reason),
        "by_time_slot": _dim_list(result.by_time_slot),
        "recommendations": result.recommendations,
        "suggested_filters": result.suggested_filters,
        "a_trade_profile": result.a_trade_profile,
    }


def save_optimization_results(results: list[OptimizationResult]) -> str:
    """Save optimization results for dashboard consumption."""
    output = {}
    for r in results:
        output[r.sim_id.upper()] = result_to_dict(r)

    path = os.path.join(RESULTS_DIR, "optimization_data.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    return path


# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_hour(entry_time: str) -> int:
    """Extract hour from entry_time string."""
    try:
        parts = entry_time.split(" ")
        time_part = parts[1] if len(parts) > 1 else parts[0]
        return int(time_part.split(":")[0])
    except (IndexError, ValueError):
        return 0


def _parse_dow(entry_time: str) -> int:
    """Extract day of week from entry_time string."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(entry_time)
        return dt.weekday()
    except (ValueError, TypeError):
        return 0


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Optimizer")
    parser.add_argument("--sim", help="Single sim ID (e.g., SIM03)")
    parser.add_argument("--all", action="store_true", help="Analyze all sims with backtest data")
    args = parser.parse_args()

    # Load dashboard data to find available sims
    path = os.path.join(RESULTS_DIR, "dashboard_data.json")
    if not os.path.exists(path):
        print("No backtest data found. Run: python -m backtest.runner first")
        return

    with open(path) as f:
        all_data = json.load(f)

    if args.sim:
        sim_ids = [args.sim.upper()]
    elif args.all:
        sim_ids = sorted(all_data.keys())
    else:
        sim_ids = sorted(all_data.keys())

    all_results = []
    for sim_id in sim_ids:
        if sim_id not in all_data:
            print(f"No backtest data for {sim_id}")
            continue

        result = analyze_sim(sim_id)
        if not result:
            print(f"No analyzable data for {sim_id}")
            continue

        all_results.append(result)

        print(f"\n{'='*70}")
        print(f"  {sim_id} — {result.signal_mode} ({result.symbol})")
        print(f"  {result.total_trades} trades | WR: {result.overall_win_rate*100:.1f}% | "
              f"A-rate: {result.overall_a_rate*100:.1f}% | F-rate: {result.overall_f_rate*100:.1f}%")
        print(f"{'='*70}")

        # Hour breakdown
        print("\n  BY HOUR:")
        for h in result.by_hour:
            bar = "█" * int(h.win_rate * 20) + "░" * (20 - int(h.win_rate * 20))
            a_tag = f" A:{h.a_trades}" if h.a_trades else ""
            f_tag = f" F:{h.f_trades}" if h.f_trades else ""
            print(f"    {h.label:12s} {bar} {h.win_rate*100:5.1f}% WR  ({h.total:3d} trades, ${h.total_pnl:>8.2f}){a_tag}{f_tag}")

        # Day breakdown
        print("\n  BY DAY:")
        for d in result.by_day:
            bar = "█" * int(d.win_rate * 20) + "░" * (20 - int(d.win_rate * 20))
            print(f"    {d.label:12s} {bar} {d.win_rate*100:5.1f}% WR  ({d.total:3d} trades, ${d.total_pnl:>8.2f})")

        # Time slot breakdown
        print("\n  BY TIME SLOT:")
        for s in result.by_time_slot:
            bar = "█" * int(s.win_rate * 20) + "░" * (20 - int(s.win_rate * 20))
            print(f"    {s.label:30s} {bar} {s.win_rate*100:5.1f}% WR  ({s.total:3d} trades)")

        # Direction
        print("\n  BY DIRECTION:")
        for d in result.by_direction:
            print(f"    {d.label:10s} WR: {d.win_rate*100:.1f}% | A-rate: {d.a_rate*100:.1f}% | F-rate: {d.f_rate*100:.1f}% | ({d.total} trades)")

        # Regime
        if result.by_regime:
            print("\n  BY REGIME:")
            for r in result.by_regime:
                if r.total >= 3:
                    print(f"    {r.label:15s} WR: {r.win_rate*100:.1f}% | ({r.total} trades, ${r.total_pnl:.2f})")

        # Exit reasons
        print("\n  BY EXIT REASON:")
        for e in sorted(result.by_exit_reason, key=lambda x: -x.total):
            if e.total >= 3:
                pnl_tag = f"avg {e.avg_pnl_pct*100:+.1f}%"
                print(f"    {e.label:20s} {e.total:3d} trades | WR: {e.win_rate*100:.1f}% | {pnl_tag}")

        # Recommendations
        print("\n  RECOMMENDATIONS:")
        for rec in result.recommendations:
            if rec:
                print(f"    {rec}")

        # YAML filters
        yaml_block = generate_yaml_filters(result)
        print(f"\n  SUGGESTED YAML CONFIG:")
        for line in yaml_block.split("\n"):
            print(f"    {line}")

    # Save all results
    if all_results:
        out_path = save_optimization_results(all_results)
        print(f"\n{'='*70}")
        print(f"Optimization data saved to: {out_path}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Grid-Search Optimizer with Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════

# Sweep ranges
_TP_VALUES = [0.03, 0.05, 0.08, 0.12, 0.15, 0.20]
_SL_VALUES = [0.02, 0.03, 0.05, 0.08, 0.10]
_HOLD_MAX_MINUTES = [15, 30, 60, 120, 240]
_HOLD_MAX_SECONDS = [m * 60 for m in _HOLD_MAX_MINUTES]


def _wf_get_trade_pnl(t) -> float:
    if isinstance(t, dict):
        return float(t.get("realized_pnl_dollars") or t.get("pnl") or 0)
    return float(getattr(t, "pnl", 0))


def _wf_run_attr(r, key, default=0):
    """Get attribute from a run (dict or dataclass)."""
    if isinstance(r, dict):
        return r.get(key, default)
    return getattr(r, key, default)


def _wf_total_trades(summary) -> int:
    return sum(_wf_run_attr(r, "total_trades", 0)
               for r in summary.runs) if summary.runs else 0


def _wf_profit_factor(summary) -> float:
    gp, gl = 0.0, 0.0
    for r in summary.runs:
        for t in _wf_run_attr(r, "trades", []):
            pnl = _wf_get_trade_pnl(t)
            if pnl > 0:
                gp += pnl
            else:
                gl += abs(pnl)
    return gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)


def _wf_daily_sharpe(summary) -> float:
    daily_pnl: dict = {}
    for r in summary.runs:
        for t in _wf_run_attr(r, "trades", []):
            d = t.get("date", "") if isinstance(t, dict) else getattr(t, "date", "")
            daily_pnl[d] = daily_pnl.get(d, 0.0) + _wf_get_trade_pnl(t)
    vals = list(daily_pnl.values())
    if len(vals) < 2:
        return 0.0
    avg = sum(vals) / len(vals)
    var = sum((v - avg) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    return (avg / std) * math.sqrt(252) if std > 0 else 0.0


def _wf_win_rate(summary) -> float:
    """Compute overall win rate across all runs."""
    total_wins = 0
    total_trades = 0
    for r in summary.runs:
        total_wins += _wf_run_attr(r, "wins", 0)
        total_trades += _wf_run_attr(r, "total_trades", 0)
    return total_wins / total_trades if total_trades > 0 else 0.0


def _wf_avg_win_loss(summary) -> tuple[float, float]:
    """Compute average win and average loss across all runs."""
    wins_list = []
    loss_list = []
    for r in summary.runs:
        for t in _wf_run_attr(r, "trades", []):
            pnl = _wf_get_trade_pnl(t)
            if pnl > 0:
                wins_list.append(pnl)
            elif pnl < 0:
                loss_list.append(abs(pnl))
    avg_win = sum(wins_list) / len(wins_list) if wins_list else 0.0
    avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
    return avg_win, avg_loss


def _wf_total_pnl(summary) -> float:
    """Compute total PnL across all runs."""
    return sum(_wf_run_attr(r, "total_pnl", 0) for r in summary.runs) if summary.runs else 0.0


def _wf_score(summary, objective: str) -> float:
    """Score a BacktestSummary for optimizer ranking.

    Objectives:
    - growth: total_return / max_drawdown (Calmar-like), penalizes blow-ups
    - expectancy: (win_rate * avg_win - loss_rate * avg_loss) * sqrt(trade_count)
    - winrate: win_rate * profit_factor (requires PF > 1)
    - balanced: daily Sharpe ratio (requires >= 30 trades)
    """
    if not summary.runs:
        return 0.0
    total = _wf_total_trades(summary)
    if total == 0:
        return 0.0

    if objective == "growth":
        # Use total return across ALL runs, not just the last run's end balance
        total_pnl = _wf_total_pnl(summary)
        start_bal = _wf_run_attr(summary.runs[0], "starting_balance", 3000)
        total_return = total_pnl / start_bal if start_bal > 0 else 0.0
        max_dd = max(summary.avg_max_drawdown, 0.05)
        # Negative returns produce negative scores — enables differentiation
        return total_return / max_dd

    elif objective == "expectancy":
        if total < 10:
            return 0.0
        wr = _wf_win_rate(summary)
        avg_win, avg_loss = _wf_avg_win_loss(summary)
        if avg_loss == 0:
            return avg_win * (total ** 0.5) if avg_win > 0 else 0.0
        expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)
        return expectancy * (total ** 0.5)

    elif objective == "winrate":
        pf = _wf_profit_factor(summary)
        if pf < 1.0:
            return 0.0
        return summary.avg_win_rate * min(pf, 99.0)

    elif objective == "balanced":
        if total < 30:
            return 0.0
        return _wf_daily_sharpe(summary)

    return 0.0


def _wf_run_fold(work_unit: dict) -> dict:
    """Worker: run train + test for one (combo, fold). Module-level for pickling."""
    proj_root = work_unit["project_root"]
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    import logging
    logging.disable(logging.WARNING)

    from backtest.engine import BacktestEngine

    profile = dict(work_unit["profile"])
    params = work_unit["params"]
    profile["profit_target_pct"] = params["profit_target_pct"]
    profile["stop_loss_pct"] = params["stop_loss_pct"]
    profile["hold_max_seconds"] = params["hold_max_seconds"]

    sid = work_unit["sim_id"]
    obj = work_unit["objective"]

    # Train
    te = BacktestEngine(sid, profile, work_unit["train_start"],
                        work_unit["train_end"], max_runs=0, verbose=False)
    ts = te.run()
    train_score = _wf_score(ts, obj)
    train_trades = _wf_total_trades(ts)

    # Test
    te2 = BacktestEngine(sid, profile, work_unit["test_start"],
                         work_unit["test_end"], max_runs=0, verbose=False)
    ts2 = te2.run()
    test_score = _wf_score(ts2, obj)
    test_trades = _wf_total_trades(ts2)
    test_pnl = _wf_total_pnl(ts2)
    test_wr = _wf_win_rate(ts2)
    test_avg_win, test_avg_loss = _wf_avg_win_loss(ts2)

    return {
        "combo_idx": work_unit["combo_idx"],
        "fold_num": work_unit["fold_num"],
        "params": params,
        "train_score": train_score,
        "test_score": test_score,
        "train_trades": train_trades,
        "test_trades": test_trades,
        "test_profitable": test_pnl > 0,
        "test_pnl": round(test_pnl, 2),
        "test_win_rate": round(test_wr, 4),
        "test_avg_win": round(test_avg_win, 2),
        "test_avg_loss": round(test_avg_loss, 2),
    }


class SimOptimizer:
    """Grid-search parameter optimizer with walk-forward validation."""

    def __init__(self, sim_id: str, start_date: str, end_date: str,
                 objective: str = "growth"):
        if sim_id.upper() == "SIM00":
            raise ValueError("SIM00 (live sim) must never be optimized.")
        self.sim_id = sim_id
        self.start_date = start_date
        self.end_date = end_date
        self.objective = objective

    def _load_profile(self) -> dict:
        import yaml
        cfg_path = os.path.join(_PROJECT_ROOT, "simulation", "sim_config.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        key = self.sim_id.upper()
        for k, v in cfg.items():
            if str(k).upper() == key and isinstance(v, dict):
                return v
        raise ValueError(f"Profile {self.sim_id} not found in sim_config.yaml")

    def _build_folds(self) -> list[dict]:
        """Build 3 walk-forward folds with non-overlapping test windows.

        Splits the date range into 5 equal chunks:
          Fold 1: train [0..2], test [2..3]  — early test
          Fold 2: train [0..3], test [3..4]  — middle test
          Fold 3: train [0..4], test [4..5]  — late test (expanding train)
        """
        start = _dt.strptime(self.start_date, "%Y-%m-%d")
        end = _dt.strptime(self.end_date, "%Y-%m-%d")
        total_days = (end - start).days
        chunk = timedelta(days=total_days // 5)

        boundaries = [start + i * chunk for i in range(5)] + [end]

        return [
            {
                "train_start": boundaries[0].strftime("%Y-%m-%d"),
                "train_end": boundaries[2].strftime("%Y-%m-%d"),
                "test_start": boundaries[2].strftime("%Y-%m-%d"),
                "test_end": boundaries[3].strftime("%Y-%m-%d"),
            },
            {
                "train_start": boundaries[0].strftime("%Y-%m-%d"),
                "train_end": boundaries[3].strftime("%Y-%m-%d"),
                "test_start": boundaries[3].strftime("%Y-%m-%d"),
                "test_end": boundaries[4].strftime("%Y-%m-%d"),
            },
            {
                "train_start": boundaries[0].strftime("%Y-%m-%d"),
                "train_end": boundaries[4].strftime("%Y-%m-%d"),
                "test_start": boundaries[4].strftime("%Y-%m-%d"),
                "test_end": boundaries[5].strftime("%Y-%m-%d"),
            },
        ]

    def run(self) -> dict:
        profile = self._load_profile()
        folds = self._build_folds()

        baseline = {
            "tp": profile.get("profit_target_pct", 0.2),
            "sl": profile.get("stop_loss_pct", 0.1),
            "hold_max": int(profile.get("hold_max_seconds", 3600)) // 60,
        }

        # Build combos
        combos = []
        for tp in _TP_VALUES:
            for sl in _SL_VALUES:
                for hm in _HOLD_MAX_SECONDS:
                    combos.append({
                        "profit_target_pct": tp,
                        "stop_loss_pct": sl,
                        "hold_max_seconds": hm,
                    })

        total_units = len(combos) * len(folds)
        max_workers = min(max(1, (os.cpu_count() or 2) - 1), 4)

        print(f"\n  Optimizer: {self.sim_id} | objective={self.objective}")
        print(f"  Baseline: TP={baseline['tp']} SL={baseline['sl']} "
              f"HoldMax={baseline['hold_max']}min")
        print(f"  Grid: {len(combos)} combos x {len(folds)} folds "
              f"= {total_units} engine runs")
        for i, f in enumerate(folds, 1):
            print(f"    Fold {i}: train [{f['train_start']} -> {f['train_end']}] "
                  f"test [{f['test_start']} -> {f['test_end']}]")
        print(f"  Workers: {max_workers}")

        # Build work units
        work_units = []
        for ci, params in enumerate(combos):
            for fi, fold in enumerate(folds, 1):
                work_units.append({
                    "sim_id": self.sim_id,
                    "profile": profile,
                    "params": params,
                    "objective": self.objective,
                    "combo_idx": ci,
                    "fold_num": fi,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "project_root": _PROJECT_ROOT,
                })

        # Execute in parallel
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_wf_run_fold, wu): wu for wu in work_units}
            done = 0
            for future in as_completed(futures):
                done += 1
                if done % 10 == 0 or done == total_units:
                    print(f"  Progress: {done}/{total_units} runs complete...",
                          flush=True)
                try:
                    results.append(future.result())
                except Exception as e:
                    wu = futures[future]
                    print(f"  ERROR combo {wu['combo_idx']} "
                          f"fold {wu['fold_num']}: {e}")

        # Aggregate by combo
        by_combo: dict[int, list] = {}
        for r in results:
            by_combo.setdefault(r["combo_idx"], []).append(r)

        ranked = []
        for ci, fold_results in by_combo.items():
            if len(fold_results) < len(folds):
                continue
            p = fold_results[0]["params"]
            train_scores = [r["train_score"] for r in fold_results]
            test_scores = [r["test_score"] for r in fold_results]
            avg_train = sum(train_scores) / len(train_scores)
            avg_test = sum(test_scores) / len(test_scores)
            consistency = sum(
                1 for r in fold_results if r["test_profitable"]
            ) / len(fold_results)
            overfit = avg_test < avg_train * 0.6 if avg_train > 0 else False
            total_test_trades = sum(r["test_trades"] for r in fold_results)
            avg_test_pnl = sum(r.get("test_pnl", 0) for r in fold_results) / len(fold_results)

            ranked.append({
                "params": {
                    "tp": p["profit_target_pct"],
                    "sl": p["stop_loss_pct"],
                    "hold_max": p["hold_max_seconds"] // 60,
                },
                "avg_train_score": round(avg_train, 4),
                "avg_test_score": round(avg_test, 4),
                "consistency": round(consistency, 2),
                "overfit_flag": overfit,
                "total_test_trades": total_test_trades,
                "avg_test_pnl": round(avg_test_pnl, 2),
                "fold_details": sorted(
                    [
                        {
                            "fold": r["fold_num"],
                            "train_score": round(r["train_score"], 4),
                            "test_score": round(r["test_score"], 4),
                            "train_trades": r["train_trades"],
                            "test_trades": r["test_trades"],
                            "test_profitable": r["test_profitable"],
                            "test_pnl": r.get("test_pnl", 0),
                            "test_win_rate": r.get("test_win_rate", 0),
                        }
                        for r in fold_results
                    ],
                    key=lambda x: x["fold"],
                ),
            })

        ranked.sort(key=lambda x: x["avg_test_score"], reverse=True)
        top_10 = ranked[:10]
        for i, entry in enumerate(top_10, 1):
            entry["rank"] = i

        # Determine verdict
        if not ranked:
            verdict = "NO_RESULTS"
            verdict_reason = "No parameter combos produced results."
        elif all(e["avg_test_score"] <= 0 for e in top_10):
            verdict = "NO_VIABLE_PARAMS"
            verdict_reason = ("All parameter combos produced negative OOS results. "
                              "Strategy may lack fundamental edge.")
        elif all(e["consistency"] == 0 for e in top_10):
            # Check if all scores are effectively the same
            scores = [e["avg_test_score"] for e in top_10]
            score_range = max(scores) - min(scores) if scores else 0
            if score_range < 0.01:
                verdict = "NO_DIFFERENTIATION"
                verdict_reason = ("All combos scored identically OOS. "
                                  "Parameters not reaching trade logic or test periods too short.")
            else:
                verdict = "WEAK"
                verdict_reason = ("No combo was profitable in any fold, "
                                  "but scores vary — some params are less bad than others.")
        else:
            best = top_10[0]
            if best["consistency"] >= 0.67:
                verdict = "VIABLE"
                verdict_reason = (f"Top combo profitable in {best['consistency']*100:.0f}% "
                                  f"of folds with avg OOS score {best['avg_test_score']:.2f}.")
            else:
                verdict = "MARGINAL"
                verdict_reason = (f"Some combos show positive OOS scores "
                                  f"but consistency is low ({best['consistency']*100:.0f}%).")

        output = {
            "sim_id": self.sim_id,
            "objective": self.objective,
            "date_range": [self.start_date, self.end_date],
            "baseline_params": baseline,
            "total_combos": len(combos),
            "total_runs": total_units,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "folds": folds,
            "top_10": top_10,
        }

        path = os.path.join(RESULTS_DIR, f"optimizer_{self.sim_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        return output
