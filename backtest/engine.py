"""
backtest/engine.py
Fully synchronous historical backtesting engine.

Design:
- Iterates over 1-minute stock bars bar-by-bar
- On each bar, checks entry signal for each symbol in the profile's symbols list
- If signal fires: selects an option contract, fetches its bars from Alpaca/cache
- Tracks open position bar-by-bar, applying exit conditions
- Respects death_threshold and target_hit tracking
- Multiple runs: each "run" starts at balance_start (default $3000), ends when blown
  (<= death_threshold) or data is exhausted; then resets and continues from the same bar
- Applies spread-aware slippage based on option DTE and moneyness (via core.slippage)
"""
from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, date, timedelta
from typing import Optional

import pandas as pd
import pytz

from backtest.data_fetcher import fetch_stock_bars, fetch_option_bars, build_occ_symbol
from backtest.signal_adapter import get_signal, compute_features_for_backtest, _prepare_df_with_indicators
from backtest.exit_adapter import check_exit_conditions
from backtest.results import BacktestTrade, BacktestRun, BacktestSummary
from core.slippage import estimate_spread_pct, compute_slippage

MARKET_OPEN = dt_time(9, 31)
ENTRY_CUTOFF = dt_time(15, 45)  # No new entries after this time
EOD_CLOSE = dt_time(15, 58)     # Force close all at this time

DEFAULT_BALANCE_START = 3000.0
DEFAULT_DEATH_THRESHOLD = 150.0
TARGET_BALANCE = 5_000.0

BARS_WARMUP = 30  # Minimum bars needed before signal evaluation

# Adaptive optimization thresholds
ADAPT_MIN_TRADES = 15          # Need this many trades before first adaptation
ADAPT_PHASE2_TRADES = 300      # Enough data to switch to allowlist mode
ADAPT_PHASE3_TRADES = 700      # Enough data for aggressive A-only filtering
ADAPT_MIN_ALLOWED_HOURS = 3    # Never restrict to fewer than this many hours


# ── Adaptive Filter Engine ────────────────────────────────────────────────

class AdaptiveFilters:
    """
    EV-based adaptive optimization for backtesting replay.

    Instead of blocking/allowlisting dimensions (which removes valuable fat-tail
    winners), this system:
    1. Only blocks conditions with NEGATIVE expected value (EV < 0)
    2. Computes sizing multipliers for each condition (1.5x for best, 0.5x for worst)
    3. Progressively tightens as more data accumulates

    Each replay run uses the same data but with improved filters/sizing.
    """

    DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.all_trades: list[dict] = []
        self.run_outcomes: list[dict] = []
        self.generation: int = 0
        self.filter_log: list[dict] = []

        # Only block conditions with clearly NEGATIVE EV
        self.blocked_hours: set[int] = set()
        self.blocked_days: set[int] = set()
        self.blocked_direction: str | None = None
        self.blocked_regimes: set[str] = set()

        # Sizing multipliers: {hour: 1.5, day: 0.7, etc}
        self.hour_multiplier: dict[int, float] = {}
        self.day_multiplier: dict[int, float] = {}

        # Allowlist — only set for very high-confidence situations
        self.allowed_hours: set[int] | None = None
        self.allowed_days: set[int] | None = None
        self.required_direction: str | None = None
        self.max_hold_seconds: int = 0

        self.best_run_wr: float = 0.0
        self.best_run_peak: float = 0.0

    def _get_pnl(self, t: dict) -> float:
        return t.get("realized_pnl_dollars") or t.get("pnl") or 0

    def _is_winner(self, t: dict) -> bool:
        return self._get_pnl(t) > 0

    def update(self, new_trades: list[dict], run_outcome: str = "BLOWN",
               run_wr: float = 0, run_peak: float = 0, run_number: int = 0,
               replay_mode: bool = False) -> dict:
        """Learn from a completed run. Focus: EV-based filtering + sizing."""
        if replay_mode:
            self.all_trades = list(new_trades)
        else:
            self.all_trades.extend(new_trades)
        self.run_outcomes.append({
            "run": run_number, "outcome": run_outcome,
            "wr": run_wr, "peak": run_peak, "trades": len(new_trades),
        })
        total = len(self.all_trades)

        if total < ADAPT_MIN_TRADES:
            return {"gen": self.generation, "action": "gathering_data", "total": total}

        self.generation += 1
        changes = []

        if run_peak > self.best_run_peak:
            self.best_run_peak = run_peak
        if run_wr > self.best_run_wr:
            self.best_run_wr = run_wr

        # ── Compute EV per dimension ──────────────────────────────────────
        hour_ev = self._dim_ev(lambda t: t.get("entry_hour", 0))
        day_ev = self._dim_ev(lambda t: t.get("day_of_week", 0))
        dir_ev = self._dim_ev(lambda t: (t.get("direction") or "").upper())

        overall_ev = sum(self._get_pnl(t) for t in self.all_trades) / total if total else 0

        # ── Block only NEGATIVE EV conditions ─────────────────────────────
        # Only block if: (1) significantly negative EV, (2) enough data, (3) is hurting us
        for h, stats in hour_ev.items():
            if h not in self.blocked_hours and stats["n"] >= 10 and stats["ev"] < -1.0:
                # Negative EV: this hour is actively losing money
                self.blocked_hours.add(h)
                changes.append(f"BLOCK hour {h} (EV=${stats['ev']:.2f}/trade, n={stats['n']})")

        for d, stats in day_ev.items():
            if d not in self.blocked_days and stats["n"] >= 15 and stats["ev"] < -2.0:
                self.blocked_days.add(d)
                changes.append(f"BLOCK {self.DAY_NAMES.get(d, d)} (EV=${stats['ev']:.2f}/trade)")

        for d, stats in dir_ev.items():
            if stats["n"] >= 20 and stats["ev"] < -2.0:
                other_evs = {k: v["ev"] for k, v in dir_ev.items() if k != d and v["n"] >= 10}
                if other_evs and max(other_evs.values()) > stats["ev"] + 3.0:
                    if self.blocked_direction != d:
                        self.blocked_direction = d
                        changes.append(f"BLOCK direction {d} (EV=${stats['ev']:.2f})")

        # ── Compute sizing multipliers ────────────────────────────────────
        # Scale position sizing based on relative EV per hour
        if hour_ev:
            max_hour_ev = max(s["ev"] for s in hour_ev.values())
            min_hour_ev = min(s["ev"] for s in hour_ev.values())
            ev_range = max_hour_ev - min_hour_ev
            new_mult = {}
            for h, stats in hour_ev.items():
                if h in self.blocked_hours:
                    continue
                if ev_range > 0.5:
                    # Scale from 0.5x to 2.0x based on relative EV
                    norm = (stats["ev"] - min_hour_ev) / ev_range  # 0 to 1
                    mult = 0.5 + norm * 1.5  # 0.5 to 2.0
                else:
                    mult = 1.0
                new_mult[h] = round(mult, 2)
            if new_mult != self.hour_multiplier:
                top_hours = sorted(new_mult.items(), key=lambda x: x[1], reverse=True)[:3]
                changes.append(f"SIZE hours: {', '.join(f'h{h}={m:.1f}x' for h, m in top_hours)}")
                self.hour_multiplier = new_mult

        if day_ev:
            max_day_ev = max(s["ev"] for s in day_ev.values())
            min_day_ev = min(s["ev"] for s in day_ev.values())
            ev_range = max_day_ev - min_day_ev
            new_mult = {}
            for d, stats in day_ev.items():
                if d in self.blocked_days:
                    continue
                if ev_range > 0.5:
                    norm = (stats["ev"] - min_day_ev) / ev_range
                    mult = 0.7 + norm * 0.6  # 0.7 to 1.3
                else:
                    mult = 1.0
                new_mult[d] = round(mult, 2)
            if new_mult != self.day_multiplier:
                self.day_multiplier = new_mult

        self.filter_log.append({
            "gen": self.generation, "total": total,
            "changes": changes, "overall_ev": round(overall_ev, 2),
        })

        if self.verbose and changes:
            print(f"  [ADAPT gen#{self.generation}] {'; '.join(changes)}")

        return {"gen": self.generation, "changes": changes}

    def should_skip(self, entry_hour: int, day_of_week: int, direction: str,
                    regime: str, confidence: float = 0) -> str | None:
        """Check if this entry should be blocked (only negative-EV conditions)."""
        if entry_hour in self.blocked_hours:
            return f"adapt_block_hour_{entry_hour}"
        if day_of_week in self.blocked_days:
            return f"adapt_block_day_{self.DAY_NAMES.get(day_of_week, day_of_week)}"
        if self.blocked_direction and direction.upper() == self.blocked_direction:
            return f"adapt_block_dir_{self.blocked_direction}"
        if regime and regime in self.blocked_regimes:
            return f"adapt_block_regime_{regime}"
        if self.allowed_hours is not None and entry_hour not in self.allowed_hours:
            return f"adapt_hour_{entry_hour}_not_in_allowlist"
        if self.allowed_days is not None and day_of_week not in self.allowed_days:
            return f"adapt_day_not_in_allowlist"
        if self.required_direction and direction.upper() != self.required_direction:
            return f"adapt_require_{self.required_direction}_only"
        return None

    def get_sizing_multiplier(self, entry_hour: int, day_of_week: int) -> float:
        """Get the combined sizing multiplier for the current conditions."""
        h_mult = self.hour_multiplier.get(entry_hour, 1.0)
        d_mult = self.day_multiplier.get(day_of_week, 1.0)
        return h_mult * d_mult

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "total_trades_analyzed": len(self.all_trades),
            "blocked_hours": sorted(self.blocked_hours),
            "blocked_days": [self.DAY_NAMES.get(d, str(d)) for d in sorted(self.blocked_days)],
            "blocked_direction": self.blocked_direction,
            "blocked_regimes": sorted(self.blocked_regimes),
            "allowed_hours": sorted(self.allowed_hours) if self.allowed_hours else None,
            "allowed_days": [self.DAY_NAMES.get(d, str(d)) for d in sorted(self.allowed_days)] if self.allowed_days else None,
            "required_direction": self.required_direction,
            "max_hold_seconds": self.max_hold_seconds,
            "hour_multiplier": self.hour_multiplier,
            "day_multiplier": self.day_multiplier,
            "best_run_peak": self.best_run_peak,
            "best_run_wr": self.best_run_wr,
            "run_outcomes": self.run_outcomes,
            "filter_log": self.filter_log,
        }

    def _dim_ev(self, key_fn) -> dict:
        """Compute expected value per dimension group."""
        from collections import defaultdict
        groups = defaultdict(list)
        for t in self.all_trades:
            k = key_fn(t)
            if k is not None:
                groups[k].append(t)
        result = {}
        for k, trades in groups.items():
            n = len(trades)
            if n < 3:
                continue
            total_pnl = sum(self._get_pnl(t) for t in trades)
            wins = sum(1 for t in trades if self._is_winner(t))
            result[k] = {
                "n": n, "ev": total_pnl / n,
                "total_pnl": total_pnl, "wr": wins / n,
            }
        return result


def _get_et_time(ts) -> dt_time:
    """Extract time from a Timestamp or datetime (assumed ET naive or converted)."""
    if isinstance(ts, (pd.Timestamp, datetime)):
        return ts.time()
    return dt_time(0, 0)


def _is_trading_bar(ts) -> bool:
    """Return True if this bar is within trading hours for entries."""
    t = _get_et_time(ts)
    return MARKET_OPEN <= t <= ENTRY_CUTOFF


def _compute_regime(df: pd.DataFrame) -> str:
    """
    Compute a simple regime label from the last N bars.
    TREND if EMA9 > EMA20 or EMA9 < EMA20 and abs spread > 0.05%
    RANGE otherwise
    """
    if df is None or len(df) < 20:
        return "TREND"  # Default to TREND so signals fire
    try:
        last = df.iloc[-1]
        ema9 = float(last.get("ema9", 0) or 0)
        ema20 = float(last.get("ema20", 0) or 0)
        if ema20 == 0:
            return "TREND"
        spread = abs(ema9 - ema20) / ema20
        if spread > 0.0005:
            return "TREND"
        return "RANGE"
    except Exception:
        return "TREND"


def _check_regime_filter(profile: dict, regime: str) -> bool:
    """Return True if regime passes the profile's regime_filter."""
    regime_filter = profile.get("regime_filter")
    if regime_filter is None:
        return True
    if isinstance(regime_filter, list):
        return regime in regime_filter
    if regime_filter == "TREND_ONLY":
        return regime == "TREND"
    if regime_filter == "RANGE_ONLY":
        return regime == "RANGE"
    if regime_filter == "VOLATILE_ONLY":
        return regime == "VOLATILE"
    return True


def _select_option_strike(underlying_price: float, direction: str, otm_pct: float) -> float:
    """Select ATM/OTM strike as a whole-dollar integer."""
    if direction == "BULLISH":
        base = underlying_price * (1 + otm_pct)
    else:
        base = underlying_price * (1 - otm_pct)
    return float(round(base))


def _select_expiry(trade_date: date, dte_min: int, dte_max: int) -> Optional[date]:
    """
    Select the nearest valid expiry date (weekday) within [dte_min, dte_max] trading days.
    Prefers dte_min + 1 to avoid same-day expiry issues.
    """
    candidates = []
    trading_dte = 0
    expiry_date = trade_date
    while trading_dte <= dte_max + 2:
        if expiry_date.weekday() < 5:  # weekday only
            if dte_min <= trading_dte <= dte_max:
                candidates.append((expiry_date, trading_dte))
            trading_dte += 1
        expiry_date += timedelta(days=1)

    if not candidates:
        return None
    # Prefer shortest DTE
    return candidates[0][0]


def _position_size(balance: float, fill_price: float, profile: dict) -> tuple:
    """
    Compute qty using small-account sizing (same as sim_account_mode.compute_small_account_qty).
    Returns (qty, risk_dollars, block_reason).
    """
    if fill_price <= 0:
        return 0, 0.0, "invalid_fill_price"

    death_threshold = float(profile.get("death_threshold", DEFAULT_DEATH_THRESHOLD))
    if balance <= death_threshold:
        return 0, 0.0, "balance_below_death_threshold"

    risk_pct = float(profile.get("risk_per_trade_pct", 0.02))
    max_pos_pct = float(profile.get("max_position_pct", 0.15))

    one_contract_cost = fill_price * 100
    max_notional = balance * max_pos_pct

    if one_contract_cost > max_notional:
        return 0, 0.0, "contract_too_expensive_for_account"

    risk_dollars = balance * risk_pct
    min_risk = max(3.0, balance * 0.01)
    risk_dollars = max(risk_dollars, min_risk)

    qty = max(1, math.floor(risk_dollars / one_contract_cost))
    while qty > 1 and (fill_price * qty * 100) > max_notional:
        qty -= 1

    return qty, risk_dollars, None


@dataclass
class _OpenTrade:
    trade_id: str
    entry_time: str
    entry_price: float
    qty: int
    direction: str
    symbol: str
    contract: str
    expiry: str
    stop_loss_pct: float
    hold_min_seconds: float
    hold_max_seconds: float
    balance_before: float
    peak_price: float = 0.0
    trailing_stop_activated: bool = False
    trailing_stop_high: float = 0.0
    tp2_activated: bool = False
    tp2_target_pct: float = 0.0
    # MFE/MAE tracking (bar-by-bar)
    mfe_pct: float = 0.0   # max favorable excursion (option price pct from entry)
    mae_pct: float = 0.0   # max adverse excursion (option price pct from entry, stored positive)
    mfe_estimated: bool = False  # True if any bar used synthetic price proxy
    # Slippage tracking
    entry_slippage_pct: float = 0.01
    spread_pct_at_entry: float = 0.0
    dte_at_entry: int = 0
    otm_pct_at_entry: float = 0.0
    # Context for optimizer
    regime: str = ""
    confidence: float = 0.0


class BacktestEngine:
    """
    Synchronous backtesting engine for a single sim profile.

    Usage:
        engine = BacktestEngine("SIM03", profile, start_date="2024-01-01", end_date="2024-12-31")
        summary = engine.run()
    """

    def __init__(
        self,
        profile_id: str,
        profile: dict,
        start_date: str,
        end_date: str,
        max_runs: int = 0,
        verbose: bool = True,
        adaptive: bool = False,
    ):
        self.profile_id = profile_id
        self.profile = profile
        self.start_date = start_date
        self.end_date = end_date
        self.max_runs = max_runs
        self.verbose = verbose
        self.adaptive = adaptive
        self.adapt_filters = AdaptiveFilters(verbose=verbose) if adaptive else None

        self.signal_mode = profile.get("signal_mode", "TREND_PULLBACK")
        self.balance_start = float(profile.get("balance_start", DEFAULT_BALANCE_START))
        self.death_threshold = float(profile.get("death_threshold", DEFAULT_DEATH_THRESHOLD))
        self.max_daily_trades = profile.get("max_daily_trades") or profile.get("max_open_trades", 99)

        # Resolve symbol list (use 'symbols' key, or fall back to 'symbol', or SPY)
        symbols_raw = profile.get("symbols")
        if symbols_raw and isinstance(symbols_raw, list):
            self.symbols = [str(s).upper() for s in symbols_raw]
        elif profile.get("symbol"):
            self.symbols = [str(profile["symbol"]).upper()]
        elif profile.get("underlying"):
            self.symbols = [str(profile["underlying"]).upper()]
        elif profile.get("underlying_symbol"):
            self.symbols = [str(profile["underlying_symbol"]).upper()]
        else:
            self.symbols = list(profile.get("symbols", []))

        # Filter to tradeable symbols (have options)
        self._tradeable = [
            s for s in self.symbols
            if s in {"SPY", "QQQ", "IWM", "VXX", "TSLA", "AAPL", "NVDA", "MSFT"}
        ]
        if not self._tradeable:
            self._tradeable = self.symbols[:1] if self.symbols else []

        # Cache: {(contract, date_str): pd.DataFrame}
        self._option_cache: dict = {}

        self.runs: list[BacktestRun] = []
        self._current_run: Optional[BacktestRun] = None

    def run(self) -> BacktestSummary:
        """Execute the full backtest across all tradeable symbols. Returns BacktestSummary."""

        # ── Fetch and prepare bars for ALL tradeable symbols ──────────────
        sym_dfs: dict[str, pd.DataFrame] = {}
        for sym in self._tradeable:
            if self.verbose:
                print(f"  [{self.profile_id}] Fetching {sym} bars {self.start_date} -> {self.end_date}...")
            df = fetch_stock_bars(sym, self.start_date, self.end_date)
            if df is not None and not df.empty:
                df = _prepare_df_with_indicators(df)
                if df is not None and not df.empty:
                    sym_dfs[sym] = df
                    if self.verbose:
                        print(f"    {sym}: {len(df)} bars")

        if not sym_dfs:
            logging.warning("backtest_no_data: %s", self.profile_id)
            return self._build_summary(",".join(self._tradeable))

        # Use the symbol with most bars as the "clock"
        clock_symbol = max(sym_dfs, key=lambda s: len(sym_dfs[s]))
        clock_df = sym_dfs[clock_symbol]
        all_bars = list(clock_df.iterrows())
        total_bars = len(all_bars)

        if self.verbose:
            print(f"  Clock: {clock_symbol} ({total_bars} bars), trading {len(sym_dfs)} symbols: {list(sym_dfs.keys())}")

        # In adaptive replay mode: each run replays ALL data from the start
        if self.adaptive:
            max_adaptive_runs = self.max_runs if self.max_runs > 0 else 10
            no_change_count = 0
            for run_number in range(1, max_adaptive_runs + 1):
                run_result = self._execute_single_run(
                    run_number, sym_dfs, clock_df, all_bars, total_bars
                )
                # Learn from completed run
                if self.adapt_filters is not None and run_result["trades"]:
                    trade_dicts = [
                        {
                            "entry_hour": t.entry_hour,
                            "entry_minute": t.entry_minute,
                            "day_of_week": t.day_of_week,
                            "direction": t.direction,
                            "regime": t.regime,
                            "confidence": t.confidence,
                            "holding_seconds": t.holding_seconds,
                            "pnl": t.pnl,
                            "realized_pnl_dollars": t.pnl,
                            "pnl_pct": t.pnl_pct,
                            "exit_reason": t.exit_reason,
                            "symbol": t.symbol,
                        }
                        for t in run_result["trades"]
                    ]
                    run_wr = sum(1 for t in run_result["trades"] if t.pnl > 0) / len(run_result["trades"])
                    adapt_result = self.adapt_filters.update(
                        trade_dicts,
                        run_outcome=run_result["outcome"],
                        run_wr=run_wr,
                        run_peak=run_result["peak_balance"],
                        run_number=run_number,
                        replay_mode=True,
                    )
                    if not adapt_result.get("changes"):
                        no_change_count += 1
                    else:
                        no_change_count = 0
                    if no_change_count >= 2 and run_number >= 3:
                        if self.verbose:
                            print(f"  [ADAPT] Filters converged after {run_number} runs.")
                        break

            if self.verbose:
                total_trades_all = sum(len(r.trades) for r in self.runs)
                total_wins = sum(r.wins for r in self.runs)
                print(f"  [{self.profile_id}] Backtest complete: {len(self.runs)} runs, {total_trades_all} total trades")
                if total_trades_all > 0:
                    print(f"    Overall win rate: {total_wins/total_trades_all*100:.1f}%")
            return self._build_summary(",".join(sym_dfs.keys()))

        # Non-adaptive: single continuous run
        self._execute_single_run(1, sym_dfs, clock_df, all_bars, total_bars)

        if self.verbose:
            total_trades_all = sum(len(r.trades) for r in self.runs)
            total_wins = sum(r.wins for r in self.runs)
            print(f"  [{self.profile_id}] Backtest complete: {len(self.runs)} runs, {total_trades_all} total trades")
            if total_trades_all > 0:
                print(f"    Overall win rate: {total_wins/total_trades_all*100:.1f}%")
        return self._build_summary(",".join(sym_dfs.keys()))

    def _execute_single_run(self, run_number: int, sym_dfs: dict[str, pd.DataFrame],
                            clock_df: pd.DataFrame, all_bars: list, total_bars: int) -> dict:
        """Execute a single run through the data across all symbols. Returns dict with run metrics."""
        balance = self.balance_start
        peak_balance = self.balance_start
        # Multi-symbol: track open trades per symbol (max 1 per symbol at a time)
        open_trades: dict[str, _OpenTrade] = {}
        max_concurrent = min(3, len(sym_dfs))  # Max simultaneous positions across symbols
        run_trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        hit_target = False
        target_hit_date: Optional[str] = None
        daily_trades = 0
        last_trade_date: Optional[date] = None
        run_start_ts = all_bars[0][0] if all_bars else None
        all_symbols_str = ",".join(sym_dfs.keys())

        # Build per-symbol bar lookup for fast timestamp access
        sym_bar_lookup: dict[str, dict] = {}
        for sym, sdf in sym_dfs.items():
            sym_bar_lookup[sym] = {ts_val: row_val for ts_val, row_val in sdf.iterrows()}

        for bar_idx in range(total_bars):
            ts, row = all_bars[bar_idx]

            if bar_idx % 30 == 0:
                equity_curve.append({
                    "timestamp": str(ts),
                    "balance": round(balance, 2),
                    "run_number": run_number,
                })

            current_date = ts.date() if isinstance(ts, (pd.Timestamp, datetime)) else None
            if current_date and current_date != last_trade_date:
                daily_trades = 0
                last_trade_date = current_date

            close_price = float(row.get("close", 0) or 0)
            if close_price <= 0:
                continue

            bar_time = _get_et_time(ts)

            # ── Check ALL existing open trades for exits ──────────────────
            blown = False
            for trade_sym in list(open_trades.keys()):
                open_trade = open_trades[trade_sym]

                opt_price = self._get_option_price_at(
                    open_trade.contract,
                    str(current_date) if current_date else "",
                    ts,
                )
                _price_is_estimated = False
                if opt_price is None or opt_price <= 0:
                    # Synthetic price proxy from underlying move
                    _price_is_estimated = True
                    sym_row = sym_bar_lookup.get(trade_sym, {}).get(ts)
                    if sym_row is not None:
                        sym_close = float(sym_row.get("close", 0) or 0)
                        sym_open = float(sym_row.get("open", sym_close) or sym_close)
                        if sym_open > 0:
                            opt_price = open_trade.entry_price * max(0.05, (1 + (sym_close / sym_open - 1) * 5))
                        else:
                            opt_price = open_trade.entry_price * 0.95
                    else:
                        opt_price = open_trade.entry_price * max(0.05, (1 + (close_price / float(row.get("open", close_price)) - 1) * 5))

                # ── Bar-by-bar MFE/MAE tracking ──────────────────────
                if open_trade.entry_price > 0:
                    bar_excursion = (opt_price - open_trade.entry_price) / open_trade.entry_price
                    if bar_excursion > open_trade.mfe_pct:
                        open_trade.mfe_pct = bar_excursion
                    if bar_excursion < 0 and abs(bar_excursion) > open_trade.mae_pct:
                        open_trade.mae_pct = abs(bar_excursion)
                    if _price_is_estimated:
                        open_trade.mfe_estimated = True

                trade_dict = {
                    "trade_id": open_trade.trade_id,
                    "entry_price": open_trade.entry_price,
                    "stop_loss_pct": open_trade.stop_loss_pct,
                    "hold_min_seconds": open_trade.hold_min_seconds,
                    "hold_max_seconds": open_trade.hold_max_seconds,
                    "expiry": open_trade.expiry,
                    "peak_price": open_trade.peak_price,
                    "trailing_stop_activated": open_trade.trailing_stop_activated,
                    "trailing_stop_high": open_trade.trailing_stop_high,
                    "tp2_activated": open_trade.tp2_activated,
                    "tp2_target_pct": open_trade.tp2_target_pct,
                }

                entry_dt = datetime.fromisoformat(open_trade.entry_time)
                if entry_dt.tzinfo is None and isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
                    entry_dt = pytz.timezone("America/New_York").localize(entry_dt)
                elapsed_seconds = max(0, ((ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts) - entry_dt).total_seconds())

                if self.adapt_filters and self.adapt_filters.max_hold_seconds > 0:
                    trade_dict["hold_max_seconds"] = min(
                        trade_dict.get("hold_max_seconds", 86400),
                        self.adapt_filters.max_hold_seconds,
                    )

                should_exit, exit_reason, exit_price = check_exit_conditions(
                    trade_dict, self.profile, opt_price, elapsed_seconds, ts,
                )

                open_trade.peak_price = float(trade_dict.get("peak_price") or open_trade.peak_price)
                open_trade.trailing_stop_activated = bool(trade_dict.get("trailing_stop_activated", False))
                open_trade.trailing_stop_high = float(trade_dict.get("trailing_stop_high") or 0)

                if should_exit and exit_reason != "still_open":
                    _exit_slip = compute_slippage("exit", open_trade.spread_pct_at_entry)
                    actual_exit_price = exit_price * (1 - _exit_slip)
                    exit_proceeds = actual_exit_price * open_trade.qty * 100
                    entry_notional = open_trade.entry_price * open_trade.qty * 100
                    pnl = exit_proceeds - entry_notional
                    balance_after = balance + exit_proceeds
                    pnl_pct = (actual_exit_price - open_trade.entry_price) / open_trade.entry_price

                    _entry_dt = datetime.fromisoformat(open_trade.entry_time)
                    bt_trade = BacktestTrade(
                        run_number=run_number,
                        trade_num=len(run_trades) + 1,
                        date=str(current_date),
                        entry_time=open_trade.entry_time,
                        exit_time=str(ts),
                        direction=open_trade.direction,
                        symbol=open_trade.symbol,
                        contract=open_trade.contract,
                        entry_price=open_trade.entry_price,
                        exit_price=round(actual_exit_price, 4),
                        qty=open_trade.qty,
                        pnl=round(pnl, 4),
                        pnl_pct=round(pnl_pct, 4),
                        balance_before=round(open_trade.balance_before, 2),
                        balance_after=round(balance_after, 2),
                        exit_reason=exit_reason,
                        signal_mode=self.signal_mode,
                        regime=open_trade.regime,
                        entry_hour=_entry_dt.hour,
                        entry_minute=_entry_dt.minute,
                        day_of_week=_entry_dt.weekday(),
                        day_of_week_name=_entry_dt.strftime("%A"),
                        confidence=open_trade.confidence,
                        holding_seconds=int(elapsed_seconds),
                        mfe_pct=round(open_trade.mfe_pct, 6),
                        mae_pct=round(open_trade.mae_pct, 6),
                        mfe_estimated=open_trade.mfe_estimated,
                        entry_slippage_pct=round(open_trade.entry_slippage_pct, 4),
                        exit_slippage_pct=round(_exit_slip, 4),
                        spread_pct_at_entry=round(open_trade.spread_pct_at_entry, 4),
                    )
                    run_trades.append(bt_trade)
                    equity_curve.append({
                        "timestamp": str(ts),
                        "balance": round(balance_after, 2),
                        "run_number": run_number,
                        "trade_num": bt_trade.trade_num,
                        "pnl": round(pnl, 4),
                    })

                    balance = balance_after
                    if balance > peak_balance:
                        peak_balance = balance
                    del open_trades[trade_sym]

                    if not hit_target and balance >= TARGET_BALANCE:
                        hit_target = True
                        target_hit_date = str(current_date)
                        if self.verbose:
                            print(f"  [{self.profile_id}] Run #{run_number}: TARGET HIT ${balance:.0f} on {current_date} after {len(run_trades)} trades")

                    if balance <= self.death_threshold:
                        blown = True
                        break

            if blown:
                if self.verbose:
                    print(f"  [{self.profile_id}] Run #{run_number}: BLOWN ${balance:.2f} after {len(run_trades)} trades (peak ${peak_balance:.0f})")
                self._finalize_run(
                    run_number=run_number, symbol=all_symbols_str,
                    balance=balance, peak_balance=peak_balance,
                    outcome="BLOWN", hit_target=hit_target,
                    target_hit_date=target_hit_date, trades=run_trades,
                    equity_curve=equity_curve, start_ts=run_start_ts, end_ts=ts,
                )
                return {
                    "outcome": "BLOWN", "trades": run_trades,
                    "peak_balance": peak_balance, "final_balance": balance,
                    "hit_target": hit_target,
                }

            # ── Check for new entries across all symbols ──────────────────
            if not _is_trading_bar(ts):
                continue
            if bar_idx < BARS_WARMUP:
                continue
            if daily_trades >= self.max_daily_trades:
                continue
            if len(open_trades) >= max_concurrent:
                continue

            for entry_sym in sym_dfs:
                if entry_sym in open_trades:
                    continue
                if len(open_trades) >= max_concurrent:
                    break

                sym_df = sym_dfs[entry_sym]
                sym_row = sym_bar_lookup[entry_sym].get(ts)
                if sym_row is None:
                    continue

                sym_close = float(sym_row.get("close", 0) or 0)
                if sym_close <= 0:
                    continue

                # Window from this symbol's data for signal
                sym_idx = sym_df.index.get_indexer([ts], method="pad")[0]
                if sym_idx < BARS_WARMUP:
                    continue
                lookback = self.profile.get("backtest_lookback", 200)
                window_df = sym_df.iloc[max(0, sym_idx - lookback):sym_idx + 1]

                regime = _compute_regime(window_df)
                if not _check_regime_filter(self.profile, regime):
                    continue

                feature_snapshot = compute_features_for_backtest(window_df, self.profile, self.signal_mode)
                direction, underlying_price, meta = get_signal(
                    window_df, self.signal_mode, self.profile,
                    feature_snapshot=feature_snapshot,
                )

                if direction is None or underlying_price is None:
                    continue

                underlying_price = float(underlying_price)
                if underlying_price <= 0:
                    underlying_price = sym_close

                # Adaptive filter gate
                if self.adapt_filters is not None:
                    signal_confidence = float(meta.get("confidence", 0)) if isinstance(meta, dict) else 0.0
                    if self.adapt_filters.should_skip(
                        entry_hour=bar_time.hour,
                        day_of_week=current_date.weekday() if current_date else 0,
                        direction=direction, regime=regime,
                        confidence=signal_confidence,
                    ):
                        continue

                trade_date = current_date
                dte_min = int(self.profile.get("dte_min", 0))
                dte_max = int(self.profile.get("dte_max", 1))
                expiry = _select_expiry(trade_date, dte_min, dte_max)
                if expiry is None:
                    continue

                otm_pct = float(self.profile.get("otm_pct", 0.005))
                strike = _select_option_strike(underlying_price, direction, otm_pct)
                contract = build_occ_symbol(entry_sym, expiry, direction, strike)

                date_str = str(trade_date)
                opt_df = self._get_or_fetch_option_bars(contract, date_str)
                if opt_df is None or opt_df.empty:
                    continue

                entry_opt_price = self._get_option_price_at(contract, date_str, ts)
                if entry_opt_price is None or entry_opt_price <= 0:
                    continue

                _dte = max(0, (expiry - trade_date).days) if trade_date and expiry else 1
                _entry_spread = estimate_spread_pct(dte=_dte, otm_pct=otm_pct)
                _entry_slip = compute_slippage("entry", _entry_spread)
                fill_price = entry_opt_price * (1 + _entry_slip)

                qty, risk_dollars, block_reason = _position_size(balance, fill_price, self.profile)
                if block_reason or qty <= 0:
                    continue

                if self.adapt_filters is not None:
                    size_mult = self.adapt_filters.get_sizing_multiplier(
                        entry_hour=bar_time.hour,
                        day_of_week=current_date.weekday() if current_date else 0,
                    )
                    qty = max(1, min(qty * 3, round(qty * size_mult)))

                notional = fill_price * qty * 100
                if notional > balance:
                    continue

                balance -= notional
                trade_id = str(uuid.uuid4())[:8]
                signal_confidence = float(meta.get("confidence", 0)) if isinstance(meta, dict) else 0.0

                open_trades[entry_sym] = _OpenTrade(
                    trade_id=trade_id, entry_time=str(ts),
                    entry_price=fill_price, qty=qty,
                    direction=direction, symbol=entry_sym,
                    contract=contract, expiry=str(expiry),
                    stop_loss_pct=float(self.profile.get("stop_loss_pct", 0.30)),
                    hold_min_seconds=float(self.profile.get("hold_min_seconds", 60)),
                    hold_max_seconds=float(self.profile.get("hold_max_seconds", 3600)),
                    balance_before=balance + notional,
                    peak_price=fill_price, regime=regime,
                    confidence=signal_confidence,
                    entry_slippage_pct=_entry_slip,
                    spread_pct_at_entry=_entry_spread,
                    dte_at_entry=_dte,
                    otm_pct_at_entry=otm_pct,
                )
                daily_trades += 1

        # End of data: close all open trades at last price
        if open_trades and all_bars:
            last_ts, last_row = all_bars[-1]
            for trade_sym, open_trade in open_trades.items():
                last_opt_price = self._get_option_price_at(
                    open_trade.contract,
                    str(last_ts.date() if isinstance(last_ts, (pd.Timestamp, datetime)) else date.today()),
                    last_ts,
                )
                if last_opt_price is None or last_opt_price <= 0:
                    last_opt_price = open_trade.entry_price * 0.80
                _eod_exit_slip = compute_slippage("exit", open_trade.spread_pct_at_entry)
                actual_exit = last_opt_price * (1 - _eod_exit_slip)
                exit_proceeds = actual_exit * open_trade.qty * 100
                entry_notional = open_trade.entry_price * open_trade.qty * 100
                pnl = exit_proceeds - entry_notional
                balance_after = balance + exit_proceeds

                bt_trade = BacktestTrade(
                    run_number=run_number,
                    trade_num=len(run_trades) + 1,
                    date=str(last_ts.date()) if isinstance(last_ts, (pd.Timestamp, datetime)) else "",
                    entry_time=open_trade.entry_time,
                    exit_time=str(last_ts),
                    direction=open_trade.direction,
                    symbol=open_trade.symbol,
                    contract=open_trade.contract,
                    entry_price=open_trade.entry_price,
                    exit_price=round(actual_exit, 4),
                    qty=open_trade.qty,
                    pnl=round(pnl, 4),
                    pnl_pct=round((actual_exit - open_trade.entry_price) / open_trade.entry_price, 4),
                    balance_before=round(open_trade.balance_before, 2),
                    balance_after=round(balance_after, 2),
                    exit_reason="data_exhausted_close",
                    signal_mode=self.signal_mode,
                )
                run_trades.append(bt_trade)
                balance = balance_after
                if balance > peak_balance:
                    peak_balance = balance

        # Finalize run (data exhausted)
        last_ts_val = all_bars[-1][0] if all_bars else None
        outcome = "TARGET_HIT" if hit_target else "DATA_EXHAUSTED"
        self._finalize_run(
            run_number=run_number, symbol=all_symbols_str,
            balance=balance, peak_balance=peak_balance,
            outcome=outcome, hit_target=hit_target,
            target_hit_date=target_hit_date, trades=run_trades,
            equity_curve=equity_curve, start_ts=run_start_ts,
            end_ts=last_ts_val,
        )

        if self.verbose:
            print(f"  [{self.profile_id}] Run #{run_number}: {outcome} ${balance:.0f} (peak ${peak_balance:.0f}) after {len(run_trades)} trades")

        return {
            "outcome": outcome, "trades": run_trades,
            "peak_balance": peak_balance, "final_balance": balance,
            "hit_target": hit_target,
        }

    def _get_or_fetch_option_bars(self, contract: str, date_str: str) -> Optional[pd.DataFrame]:
        """Get option bars from cache or fetch from Alpaca."""
        cache_key = (contract, date_str)
        if cache_key in self._option_cache:
            return self._option_cache[cache_key]

        df = fetch_option_bars(contract, date_str)
        self._option_cache[cache_key] = df
        return df

    def _get_option_price_at(
        self,
        contract: str,
        date_str: str,
        ts,
    ) -> Optional[float]:
        """Look up the option close price at or nearest to the given timestamp."""
        df = self._get_or_fetch_option_bars(contract, date_str)
        if df is None or df.empty:
            return None

        # ts may be tz-aware or naive; normalize to naive for comparison
        ts_naive = ts
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is not None:
                ts_naive = ts.tz_localize(None)
        elif isinstance(ts, datetime):
            if ts.tzinfo is not None:
                import pytz as _pytz
                ts_naive = ts.astimezone(_pytz.timezone("America/New_York")).replace(tzinfo=None)
            ts_naive = pd.Timestamp(ts_naive)

        # df.index should be naive ET after fetch_option_bars
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)

        # Find closest bar at or before ts_naive
        mask = idx <= ts_naive
        if not mask.any():
            # Use first bar if ts is before all bars
            if not df.empty:
                return float(df["close"].iloc[0])
            return None
        row = df.loc[idx[mask][-1]]
        price = float(row.get("close", 0) or 0)
        return price if price > 0 else None

    def _finalize_run(
        self,
        run_number: int,
        symbol: str,
        balance: float,
        peak_balance: float,
        outcome: str,
        hit_target: bool,
        target_hit_date: Optional[str],
        trades: list[BacktestTrade],
        equity_curve: list[dict],
        start_ts,
        end_ts,
    ):
        """Build a BacktestRun and append to self.runs."""
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total = len(trades)
        win_rate = len(wins) / total if total else 0.0
        total_pnl = sum(t.pnl for t in trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        # Max drawdown from equity curve
        max_dd_pct = 0.0
        max_dd_dollars = 0.0
        if trades:
            running = self.balance_start
            peak = self.balance_start
            for t in trades:
                running = t.balance_after
                if running > peak:
                    peak = running
                dd_d = peak - running
                dd_pct = dd_d / peak if peak > 0 else 0.0
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
                    max_dd_dollars = dd_d

        # Days active
        days_active = 0
        if start_ts is not None and end_ts is not None:
            try:
                s = start_ts.to_pydatetime() if isinstance(start_ts, pd.Timestamp) else start_ts
                e = end_ts.to_pydatetime() if isinstance(end_ts, pd.Timestamp) else end_ts
                days_active = max(0, (e.date() - s.date()).days) if hasattr(s, "date") else 0
            except Exception:
                pass

        # Serialize trades to dicts for JSON compatibility
        trades_as_dicts = [
            {
                "run_number": t.run_number,
                "trade_num": t.trade_num,
                "date": t.date,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction,
                "symbol": t.symbol,
                "option_symbol": t.contract,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "pnl": t.pnl,
                "realized_pnl_dollars": t.pnl,
                "pnl_pct": t.pnl_pct,
                "balance_before": t.balance_before,
                "balance_after_trade": t.balance_after,
                "exit_reason": t.exit_reason,
                "signal_mode": t.signal_mode,
                "regime": t.regime,
                "entry_hour": t.entry_hour,
                "entry_minute": t.entry_minute,
                "day_of_week": t.day_of_week,
                "day_of_week_name": t.day_of_week_name,
                "confidence": t.confidence,
                "holding_seconds": t.holding_seconds,
                "mfe_pct": t.mfe_pct,
                "mae_pct": t.mae_pct,
                "mfe_estimated": t.mfe_estimated,
                "entry_slippage_pct": t.entry_slippage_pct,
                "exit_slippage_pct": t.exit_slippage_pct,
                "spread_pct_at_entry": t.spread_pct_at_entry,
            }
            for t in trades
        ]

        run = BacktestRun(
            sim_profile=self.profile_id,
            signal_mode=self.signal_mode,
            symbol=symbol,
            run_number=run_number,
            start_date=str(start_ts.date()) if start_ts is not None and hasattr(start_ts, "date") else self.start_date,
            end_date=str(end_ts.date()) if end_ts is not None and hasattr(end_ts, "date") else self.end_date,
            starting_balance=self.balance_start,
            final_balance=round(balance, 4),
            peak_balance=round(peak_balance, 4),
            outcome=outcome,
            hit_target=hit_target,
            target_hit_date=target_hit_date,
            total_trades=total,
            wins=len(wins),
            losses=len(losses),
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            profit_factor=round(profit_factor, 4),
            max_drawdown_pct=round(max_dd_pct, 4),
            max_drawdown_dollars=round(max_dd_dollars, 4),
            days_active=days_active,
            trades=trades_as_dicts,
            equity_curve=equity_curve,
        )
        self.runs.append(run)

    def _build_summary(self, symbol: str) -> BacktestSummary:
        """Build BacktestSummary from all completed runs."""
        if not self.runs:
            return BacktestSummary(
                sim_profile=self.profile_id,
                signal_mode=self.signal_mode,
                symbol=symbol,
                total_runs=0,
                blown_count=0,
                target_hit_count=0,
                best_run_number=0,
                worst_run_number=0,
                avg_trades_per_run=0.0,
                avg_win_rate=0.0,
                avg_max_drawdown=0.0,
                runs=[],
            )

        blown = [r for r in self.runs if r.outcome == "BLOWN"]
        target_hits = [r for r in self.runs if r.hit_target]
        total_trades_list = [r.total_trades for r in self.runs if r.total_trades > 0]
        avg_trades = sum(total_trades_list) / len(total_trades_list) if total_trades_list else 0.0
        avg_wr = sum(r.win_rate for r in self.runs) / len(self.runs)
        avg_dd = sum(r.max_drawdown_pct for r in self.runs) / len(self.runs)

        # Best run: highest final balance
        best_run = max(self.runs, key=lambda r: r.final_balance)
        worst_run = min(self.runs, key=lambda r: r.final_balance)

        # Serialize runs to dicts
        runs_dicts = []
        for r in self.runs:
            runs_dicts.append({
                "sim_profile": r.sim_profile,
                "signal_mode": r.signal_mode,
                "symbol": r.symbol,
                "run_number": r.run_number,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "starting_balance": r.starting_balance,
                "final_balance": r.final_balance,
                "peak_balance": r.peak_balance,
                "outcome": r.outcome,
                "hit_target": r.hit_target,
                "target_hit_date": r.target_hit_date,
                "total_trades": r.total_trades,
                "wins": r.wins,
                "losses": r.losses,
                "win_rate": r.win_rate,
                "total_pnl": r.total_pnl,
                "avg_win": r.avg_win,
                "avg_loss": r.avg_loss,
                "profit_factor": r.profit_factor,
                "max_drawdown_pct": r.max_drawdown_pct,
                "max_drawdown_dollars": r.max_drawdown_dollars,
                "days_active": r.days_active,
                "trades": r.trades,
                "equity_curve": r.equity_curve,
            })

        return BacktestSummary(
            sim_profile=self.profile_id,
            signal_mode=self.signal_mode,
            symbol=symbol,
            total_runs=len(self.runs),
            blown_count=len(blown),
            target_hit_count=len(target_hits),
            best_run_number=best_run.run_number,
            worst_run_number=worst_run.run_number,
            avg_trades_per_run=round(avg_trades, 1),
            avg_win_rate=round(avg_wr, 4),
            avg_max_drawdown=round(avg_dd, 4),
            runs=runs_dicts,
        )
