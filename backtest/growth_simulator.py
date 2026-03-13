"""
backtest/growth_simulator.py
Post-backtest growth analysis wrapper.

Analyzes completed backtest results for milestone tracking,
PDT compliance, streak analysis, and compound statistics.
Does NOT re-run the engine -- works on already-completed output.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_START_CAPITAL = 3000.0
DEFAULT_TARGET = 5000.0
DEATH_THRESHOLD = 150.0


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        cleaned = str(s).replace("T", " ")
        for tz in (" EDT", " EST", " ET", "+00:00", "-04:00", "-05:00"):
            cleaned = cleaned.replace(tz, "")
        if "." in cleaned:
            base, frac = cleaned.split(".", 1)
            cleaned = base + "." + frac[:6]
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


class GrowthSimulator:
    """Post-process completed backtest results with growth analysis."""

    def __init__(self, start_capital: float = DEFAULT_START_CAPITAL,
                 target: float = DEFAULT_TARGET):
        self.start_capital = start_capital
        self.target = target
        self.milestones = [1000, 2500, 5000]

    def analyze(self, summary, trade_log: list[dict],
                equity_curve: list[dict]) -> dict:
        """
        Analyze completed backtest output.

        Args:
            summary: BacktestSummary object or dict
            trade_log: Flat list of trade dicts across all runs (chronological)
            equity_curve: Flat list of equity curve points across all runs

        Returns:
            Growth analysis dict
        """
        sim_id = (summary.sim_profile if hasattr(summary, "sim_profile")
                  else str(summary.get("sim_profile", "")))

        # ── Walk through trades, track running balance ──────────────
        running_balance = self.start_capital
        milestone_results = {str(m): None for m in self.milestones}
        start_dt = None

        trade_entries = []
        prev_run = None

        for t in trade_log:
            run_num = t.get("run_number") or t.get("_run_number") or 1

            # Detect run reset (death happened, engine restarted at $500)
            if prev_run is not None and run_num != prev_run:
                running_balance = self.start_capital
            prev_run = run_num

            pnl = float(t.get("realized_pnl_dollars") or t.get("pnl") or 0)
            running_balance += pnl

            entry_dt = _parse_dt(t.get("entry_time", ""))
            exit_dt = _parse_dt(t.get("exit_time", ""))

            if entry_dt and start_dt is None:
                start_dt = entry_dt

            trade_entries.append({
                "entry_dt": entry_dt,
                "exit_dt": exit_dt,
                "entry_date": entry_dt.date() if entry_dt else None,
                "exit_date": exit_dt.date() if exit_dt else None,
                "pnl": pnl,
                "balance_after": running_balance,
                "run_number": run_num,
                "is_win": pnl > 0,
            })

            # Milestone check
            for m in self.milestones:
                key = str(m)
                if milestone_results[key] is None and running_balance >= m:
                    days_from_start = ((entry_dt - start_dt).days
                                       if entry_dt and start_dt else 0)
                    milestone_results[key] = {
                        "date": str(entry_dt.date()) if entry_dt else None,
                        "trades": len(trade_entries),
                        "days": days_from_start,
                    }

        total_trades = len(trade_entries)
        end_balance = running_balance if trade_entries else self.start_capital

        # ── Deaths: count blown runs ────────────────────────────────
        deaths = 0
        runs = (summary.runs if hasattr(summary, "runs")
                else summary.get("runs", []) if isinstance(summary, dict)
                else [])
        for r in runs:
            outcome = r.outcome if hasattr(r, "outcome") else r.get("outcome", "")
            if outcome == "BLOWN":
                deaths += 1

        # ── PDT audit ───────────────────────────────────────────────
        pdt_violations = self._count_pdt_violations(trade_entries)

        # ── Streak analysis ─────────────────────────────────────────
        best_win_streak, worst_loss_streak = self._streak_analysis(trade_entries)

        # ── Best/worst day ──────────────────────────────────────────
        best_day, worst_day = self._best_worst_days(trade_entries)

        # ── Win rate ────────────────────────────────────────────────
        wins = sum(1 for t in trade_entries if t["is_win"])
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # ── Max drawdown from equity curve ──────────────────────────
        max_drawdown_pct = self._max_drawdown(equity_curve)

        # ── Daily Sharpe ────────────────────────────────────────────
        daily_sharpe, avg_daily_pnl, std_daily_pnl = self._daily_sharpe(
            trade_entries)

        # ── Total return ────────────────────────────────────────────
        total_return_pct = (
            (end_balance - self.start_capital) / self.start_capital * 100
        )

        # ── Target reached ──────────────────────────────────────────
        target_key = str(int(self.target))
        target_reached = milestone_results.get(target_key) is not None

        result = {
            "sim_id": sim_id,
            "start_capital": self.start_capital,
            "end_capital": round(end_balance, 2),
            "target_reached": target_reached,
            "milestones": milestone_results,
            "deaths": deaths,
            "pdt_violations": pdt_violations,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "best_win_streak": best_win_streak,
            "worst_loss_streak": worst_loss_streak,
            "best_day": best_day,
            "worst_day": worst_day,
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "daily_sharpe": round(daily_sharpe, 2),
            "total_return_pct": round(total_return_pct, 2),
        }

        # Save to JSON
        path = os.path.join(RESULTS_DIR, f"growth_{sim_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def _count_pdt_violations(self, trade_entries: list[dict]) -> int:
        """Count rolling 5-business-day windows with >=3 day trades
        while balance < $25k."""
        # A day trade = entry and exit on same calendar day
        day_trades = []
        for t in trade_entries:
            if (t["entry_date"] and t["exit_date"]
                    and t["entry_date"] == t["exit_date"]):
                day_trades.append(t)

        if not day_trades:
            return 0

        # Get all unique trading dates
        all_dates = sorted(set(t["entry_date"] for t in trade_entries
                               if t["entry_date"]))
        if not all_dates:
            return 0

        violations = 0
        for window_end in all_dates:
            # 5 business days back
            biz_count = 0
            window_start = window_end
            d = window_end
            while biz_count < 4 and d > all_dates[0]:
                d -= timedelta(days=1)
                if d.weekday() < 5:
                    biz_count += 1
                    window_start = d

            # Day trades in this window
            dt_in_window = [dt for dt in day_trades
                           if window_start <= dt["entry_date"] <= window_end]

            if len(dt_in_window) >= 3:
                # Check if balance was under $25k
                min_bal = min(dt["balance_after"] for dt in dt_in_window)
                if min_bal < 25000:
                    violations += 1

        return violations

    def _streak_analysis(self, trade_entries: list[dict]) -> tuple[int, int]:
        """Return (best_win_streak, worst_loss_streak)."""
        best_win = 0
        worst_loss = 0
        cur_win = 0
        cur_loss = 0

        for t in trade_entries:
            if t["is_win"]:
                cur_win += 1
                cur_loss = 0
                best_win = max(best_win, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                worst_loss = max(worst_loss, cur_loss)

        return best_win, worst_loss

    def _best_worst_days(self, trade_entries: list[dict]) -> tuple[dict, dict]:
        """Return best and worst single-day PnL."""
        daily_pnl: dict = defaultdict(float)
        for t in trade_entries:
            if t["entry_date"]:
                daily_pnl[t["entry_date"]] += t["pnl"]

        if not daily_pnl:
            return {"date": None, "pnl": 0.0}, {"date": None, "pnl": 0.0}

        best_date = max(daily_pnl, key=daily_pnl.get)
        worst_date = min(daily_pnl, key=daily_pnl.get)

        return (
            {"date": str(best_date), "pnl": round(daily_pnl[best_date], 2)},
            {"date": str(worst_date), "pnl": round(daily_pnl[worst_date], 2)},
        )

    def _max_drawdown(self, equity_curve: list[dict]) -> float:
        """Compute max drawdown percentage from equity curve points."""
        if not equity_curve:
            return 0.0

        peak = 0.0
        max_dd = 0.0

        for pt in equity_curve:
            bal = float(pt.get("balance", 0))
            if bal > peak:
                peak = bal
            if peak > 0:
                dd = (peak - bal) / peak
                max_dd = max(max_dd, dd)

        return max_dd

    def _daily_sharpe(self, trade_entries: list[dict]) -> tuple[float, float, float]:
        """Compute daily Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)."""
        daily_pnl: dict = defaultdict(float)
        for t in trade_entries:
            if t["entry_date"]:
                daily_pnl[t["entry_date"]] += t["pnl"]

        if len(daily_pnl) < 2:
            return 0.0, 0.0, 0.0

        values = list(daily_pnl.values())
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)

        sharpe = (avg / std) * math.sqrt(252) if std > 0 else 0.0

        return sharpe, round(avg, 2), round(std, 2)
