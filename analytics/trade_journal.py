"""
analytics/trade_journal.py

Auto-generates daily markdown trade journals summarizing all entries/exits
across all sims, with Greeks context and composite scores.
No Discord dependencies — importable standalone.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pytz

from core.paths import DATA_DIR
from simulation.sim_portfolio import SimPortfolio

_ET = pytz.timezone("US/Eastern")
_JOURNALS_DIR = os.path.join(DATA_DIR, "journals")

# Greeks-related exit reasons
_GREEKS_THETA = {"theta_burn", "theta_burn_0dte", "theta_burn_tightened"}
_GREEKS_IV = {"iv_crush_stop", "iv_crush_exit"}
_GREEKS_DELTA = {"delta_erosion"}
_GREEKS_ALL = _GREEKS_THETA | _GREEKS_IV | _GREEKS_DELTA

_MAX_TABLE_ROWS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profiles() -> dict:
    import yaml
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(base, "simulation", "sim_config.yaml")
    try:
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
        return {k: v for k, v in raw.items() if str(k).upper().startswith("SIM") and isinstance(v, dict)}
    except Exception:
        return {}


def _parse_et(ts_str) -> Optional[datetime]:
    """Parse an ISO timestamp string to ET-aware datetime."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_str))
        if dt.tzinfo is None:
            dt = _ET.localize(dt)
        else:
            dt = dt.astimezone(_ET)
        return dt
    except Exception:
        return None


def _fmt_time(dt: Optional[datetime]) -> str:
    return dt.strftime("%H:%M") if dt else "?"


def _fmt_money(val) -> str:
    try:
        v = float(val)
        sign = "+" if v >= 0 else ""
        return f"{sign}${v:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(val) -> str:
    try:
        v = float(val) * 100
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.0f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_duration(seconds) -> str:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "?"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    h, m = divmod(s, 3600)
    return f"{h}h{m // 60:02d}m"


def _hold_duration(entry_time, exit_time) -> str:
    e = _parse_et(entry_time)
    x = _parse_et(exit_time)
    if e and x:
        return _fmt_duration((x - e).total_seconds())
    return "?"


def _greeks_category(exit_reason: str) -> Optional[str]:
    if exit_reason in _GREEKS_THETA:
        return "theta_burn"
    if exit_reason in _GREEKS_IV:
        return "iv_crush"
    if exit_reason in _GREEKS_DELTA:
        return "delta_erosion"
    return None


def _get_regime_from_csv(date_str: str) -> str:
    """Try to get the dominant regime for a date from market data."""
    try:
        from signals.regime import get_regime
        from core.data_service import get_market_dataframe
        df = get_market_dataframe()
        if df is not None and not df.empty:
            regime = get_regime(df)
            return regime or "UNKNOWN"
    except Exception:
        pass
    return "UNKNOWN"


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Core journal generator
# ---------------------------------------------------------------------------

def generate_daily_journal(date_str: str | None = None) -> str:
    """
    Generate a markdown trade journal for the given date (YYYY-MM-DD).
    Returns the markdown string.
    """
    if date_str is None:
        date_str = datetime.now(_ET).date().isoformat()

    profiles = _load_profiles()
    if not profiles:
        return f"# Trade Journal -- {date_str}\n\nNo sim profiles found.\n"

    # Collect all trades for the date
    all_entries = []   # open trades that were entered today
    all_exits = []     # closed trades that exited today
    sim_balances = {}
    sim_open_counts = {}

    for sim_id, profile in sorted(profiles.items()):
        try:
            sim = SimPortfolio(sim_id, profile)
            sim.load()
            sim_balances[sim_id] = sim.balance
            sim_open_counts[sim_id] = len(sim.open_trades)

            # Check open trades for today's entries
            for t in sim.open_trades:
                entry_dt = _parse_et(t.get("entry_time"))
                if entry_dt and entry_dt.date().isoformat() == date_str:
                    t_copy = dict(t)
                    t_copy["sim_id"] = sim_id
                    t_copy["_entry_dt"] = entry_dt
                    all_entries.append(t_copy)

            # Check trade log for today's exits
            trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []
            for t in trade_log:
                exit_dt = _parse_et(t.get("exit_time"))
                if exit_dt and exit_dt.date().isoformat() == date_str:
                    t_copy = dict(t)
                    t_copy["sim_id"] = sim_id
                    t_copy["_exit_dt"] = exit_dt
                    all_exits.append(t_copy)

                # Also check if entry was today (for the entries table)
                entry_dt = _parse_et(t.get("entry_time"))
                if entry_dt and entry_dt.date().isoformat() == date_str:
                    t_copy2 = dict(t)
                    t_copy2["sim_id"] = sim_id
                    t_copy2["_entry_dt"] = entry_dt
                    all_entries.append(t_copy2)
        except Exception as e:
            logging.warning("journal_sim_error: %s %s", sim_id, e)
            continue

    # De-duplicate entries by trade_id
    seen_entry_ids = set()
    unique_entries = []
    for t in all_entries:
        tid = t.get("trade_id")
        if tid and tid not in seen_entry_ids:
            seen_entry_ids.add(tid)
            unique_entries.append(t)
    all_entries = sorted(unique_entries, key=lambda x: x.get("_entry_dt") or datetime.min.replace(tzinfo=_ET))

    all_exits = sorted(all_exits, key=lambda x: x.get("_exit_dt") or datetime.min.replace(tzinfo=_ET))

    # Compute summary stats
    total_entries = len(all_entries)
    total_exits = len(all_exits)
    exit_pnls = [_safe_float(t.get("realized_pnl_dollars")) for t in all_exits]
    total_pnl = sum(exit_pnls)
    wins = sum(1 for p in exit_pnls if p > 0)
    losses = sum(1 for p in exit_pnls if p <= 0)
    win_rate = (wins / total_exits * 100) if total_exits > 0 else 0

    # Best/worst trade
    best_trade = max(all_exits, key=lambda t: _safe_float(t.get("realized_pnl_dollars")), default=None) if all_exits else None
    worst_trade = min(all_exits, key=lambda t: _safe_float(t.get("realized_pnl_dollars")), default=None) if all_exits else None

    # Greeks exits
    greeks_exits = [t for t in all_exits if t.get("exit_reason", "") in _GREEKS_ALL]
    theta_exits = [t for t in greeks_exits if t.get("exit_reason", "") in _GREEKS_THETA]
    iv_exits = [t for t in greeks_exits if t.get("exit_reason", "") in _GREEKS_IV]
    delta_exits = [t for t in greeks_exits if t.get("exit_reason", "") in _GREEKS_DELTA]

    # Get composite scores
    scores = {}
    try:
        from analytics.composite_score import compute_composite_score
        for sim_id, profile in profiles.items():
            scores[sim_id] = compute_composite_score(sim_id, profile)
    except Exception:
        pass

    # Try to parse date for display
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_name = dt.strftime("%A, %B %d, %Y")
    except Exception:
        day_name = date_str

    # Build markdown
    lines = []
    lines.append(f"# Trade Journal -- {day_name}\n")

    # Market context
    regime = _get_regime_from_csv(date_str)
    lines.append("## Market Context")
    lines.append(f"- **Regime**: {regime}")
    lines.append(f"- **Trades today**: {total_entries} entries, {total_exits} exits across {len(profiles)} sims")
    lines.append("")

    # Summary stats
    lines.append("## Summary Stats")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total P&L (all sims) | {_fmt_money(total_pnl)} |")
    lines.append(f"| Win Rate | {win_rate:.1f}% ({wins}/{total_exits} exits) |")
    if best_trade:
        bp = _safe_float(best_trade.get("realized_pnl_pct"))
        lines.append(f"| Best Trade | {best_trade.get('sim_id')} {best_trade.get('direction', best_trade.get('contract_type', '?'))} {_fmt_money(best_trade.get('realized_pnl_dollars'))} ({_fmt_pct(bp)}) |")
    if worst_trade:
        wp = _safe_float(worst_trade.get("realized_pnl_pct"))
        lines.append(f"| Worst Trade | {worst_trade.get('sim_id')} {worst_trade.get('direction', worst_trade.get('contract_type', '?'))} {_fmt_money(worst_trade.get('realized_pnl_dollars'))} ({_fmt_pct(wp)}) |")
    greeks_parts = []
    if theta_exits:
        greeks_parts.append(f"{len(theta_exits)} theta")
    if iv_exits:
        greeks_parts.append(f"{len(iv_exits)} IV crush")
    if delta_exits:
        greeks_parts.append(f"{len(delta_exits)} delta erosion")
    greeks_text = ", ".join(greeks_parts) if greeks_parts else "none"
    lines.append(f"| Greeks Exits Fired | {len(greeks_exits)} ({greeks_text}) |")
    lines.append("")

    # Entries table
    lines.append(f"## Entries ({total_entries})")
    if all_entries:
        lines.append("| Time | SIM | Dir | Strike | DTE | Entry $ | Signal | Regime | Delta | Theta |")
        lines.append("|------|-----|-----|--------|-----|---------|--------|--------|-------|-------|")
        for i, t in enumerate(all_entries[:_MAX_TABLE_ROWS]):
            entry_dt = t.get("_entry_dt")
            direction = t.get("direction", t.get("contract_type", "?"))
            strike = t.get("strike", "?")
            dte = t.get("dte_bucket", "?")
            entry_price = f"${_safe_float(t.get('entry_price')):.2f}"
            signal = t.get("signal_mode", t.get("signal", "?"))
            regime_at = t.get("regime_at_entry", "?")
            delta = f"{_safe_float(t.get('delta_at_entry')):.2f}" if t.get("delta_at_entry") is not None else "-"
            theta = f"{_safe_float(t.get('theta_at_entry')):.2f}" if t.get("theta_at_entry") is not None else "-"
            lines.append(f"| {_fmt_time(entry_dt)} | {t.get('sim_id')} | {direction} | {strike} | {dte} | {entry_price} | {signal} | {regime_at} | {delta} | {theta} |")
        if total_entries > _MAX_TABLE_ROWS:
            lines.append(f"\n*... and {total_entries - _MAX_TABLE_ROWS} more entries*")
    else:
        lines.append("*No entries today.*")
    lines.append("")

    # Exits table
    lines.append(f"## Exits ({total_exits})")
    if all_exits:
        lines.append("| Time | SIM | Dir | Entry>Exit | P&L | Hold | Exit Reason | Greeks Note |")
        lines.append("|------|-----|-----|-----------|-----|------|-------------|-------------|")
        for i, t in enumerate(all_exits[:_MAX_TABLE_ROWS]):
            exit_dt = t.get("_exit_dt")
            direction = t.get("direction", t.get("contract_type", "?"))
            entry_p = _safe_float(t.get("entry_price"))
            exit_p = _safe_float(t.get("exit_price"))
            pnl_d = _safe_float(t.get("realized_pnl_dollars"))
            pnl_p = _safe_float(t.get("realized_pnl_pct"))
            hold = _hold_duration(t.get("entry_time"), t.get("exit_time"))
            exit_reason = t.get("exit_reason", "?")

            greeks_note = "-"
            cat = _greeks_category(exit_reason)
            if cat == "theta_burn":
                dte = t.get("dte_bucket", "?")
                greeks_note = f"DTE={dte}, theta exit"
            elif cat == "iv_crush":
                greeks_note = "IV crush detected"
            elif cat == "delta_erosion":
                d_entry = t.get("delta_at_entry")
                greeks_note = f"delta decay (entry {_safe_float(d_entry):.2f})" if d_entry else "delta decay"

            lines.append(f"| {_fmt_time(exit_dt)} | {t.get('sim_id')} | {direction} | ${entry_p:.2f}>${exit_p:.2f} | {_fmt_money(pnl_d)} ({_fmt_pct(pnl_p)}) | {hold} | {exit_reason} | {greeks_note} |")
        if total_exits > _MAX_TABLE_ROWS:
            lines.append(f"\n*... and {total_exits - _MAX_TABLE_ROWS} more exits*")
    else:
        lines.append("*No exits today.*")
    lines.append("")

    # Greeks exit detail
    if greeks_exits:
        lines.append("## Greeks Exit Detail")
        for category, cat_exits, label in [
            ("theta_burn", theta_exits, "Theta Burn"),
            ("iv_crush", iv_exits, "IV Crush"),
            ("delta_erosion", delta_exits, "Delta Erosion"),
        ]:
            if cat_exits:
                lines.append(f"### {label} Exits ({len(cat_exits)})")
                for t in cat_exits:
                    entry_p = _safe_float(t.get("entry_price"))
                    exit_p = _safe_float(t.get("exit_price"))
                    pnl_p = _safe_float(t.get("realized_pnl_pct"))
                    reason = t.get("exit_reason", "?")
                    dte = t.get("dte_bucket", "?")
                    lines.append(
                        f"- **{t.get('sim_id')}** {t.get('direction', '?')} "
                        f"${entry_p:.2f}>${exit_p:.2f} ({_fmt_pct(pnl_p)}) -- "
                        f"{reason}, DTE={dte}"
                    )
                lines.append("")

    # Leaderboard snapshot
    ranked = [(sid, s) for sid, s in scores.items() if not s.get("unranked")]
    if ranked:
        ranked.sort(key=lambda x: x[1].get("composite_score", 0), reverse=True)
        lines.append("## Sim Leaderboard Snapshot (Top 5 / Bottom 5)")
        lines.append("| Rank | SIM | Grade | Score | Today P&L |")
        lines.append("|------|-----|-------|-------|-----------|")

        # Get today's P&L per sim
        sim_day_pnl = {}
        for t in all_exits:
            sid = t.get("sim_id")
            sim_day_pnl[sid] = sim_day_pnl.get(sid, 0) + _safe_float(t.get("realized_pnl_dollars"))

        show = ranked[:5] + ranked[-5:] if len(ranked) > 10 else ranked
        shown_ids = set()
        for i, (sid, s) in enumerate(show):
            if sid in shown_ids:
                continue
            shown_ids.add(sid)
            rank = ranked.index((sid, s)) + 1
            grade = s.get("grade", "?")
            emoji = s.get("emoji", "")
            score = s.get("composite_score", 0)
            day_pnl = sim_day_pnl.get(sid, 0)
            lines.append(f"| {rank} | {sid} | {grade} {emoji} | {score} | {_fmt_money(day_pnl)} |")
        lines.append("")

    # Notes section
    lines.append("## Notes")
    # Check for dead sims
    for sim_id, profile in profiles.items():
        try:
            sim = SimPortfolio(sim_id, profile)
            sim.load()
            if sim.is_dead:
                lines.append(f"- {sim_id}: DEAD (balance ${sim.balance:.2f})")
        except Exception:
            pass
    if not any("DEAD" in l for l in lines[-10:]):
        lines.append("- No notable events.")
    lines.append("")

    return "\n".join(lines)


def save_daily_journal(date_str: str | None = None) -> str:
    """Generate and save the daily journal to disk. Returns the file path."""
    if date_str is None:
        date_str = datetime.now(_ET).date().isoformat()

    os.makedirs(_JOURNALS_DIR, exist_ok=True)
    journal_md = generate_daily_journal(date_str)
    path = os.path.join(_JOURNALS_DIR, f"journal_{date_str}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(journal_md)
    logging.info("journal_saved: %s", path)
    return path


def generate_weekly_digest(end_date_str: str | None = None) -> str:
    """Aggregate the last 5 trading days into a weekly summary."""
    if end_date_str is None:
        end_date_str = datetime.now(_ET).date().isoformat()

    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Find last 5 weekdays
    dates = []
    d = end_dt
    while len(dates) < 5:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d -= timedelta(days=1)
    dates.reverse()

    profiles = _load_profiles()

    # Collect stats per day
    daily_stats = []
    sim_weekly_pnl = {}
    weekly_exit_reasons = {}
    weekly_regimes = {}

    for date_str in dates:
        day_entries = 0
        day_exits = 0
        day_pnl = 0.0
        day_wins = 0

        for sim_id, profile in profiles.items():
            try:
                sim = SimPortfolio(sim_id, profile)
                sim.load()
                trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []

                for t in trade_log:
                    exit_dt = _parse_et(t.get("exit_time"))
                    if exit_dt and exit_dt.date().isoformat() == date_str:
                        day_exits += 1
                        pnl = _safe_float(t.get("realized_pnl_dollars"))
                        day_pnl += pnl
                        if pnl > 0:
                            day_wins += 1
                        sim_weekly_pnl[sim_id] = sim_weekly_pnl.get(sim_id, 0) + pnl

                        reason = t.get("exit_reason", "unknown")
                        weekly_exit_reasons[reason] = weekly_exit_reasons.get(reason, 0) + 1

                        regime = t.get("regime_at_entry", "UNKNOWN")
                        weekly_regimes[regime] = weekly_regimes.get(regime, 0) + 1

                    entry_dt = _parse_et(t.get("entry_time"))
                    if entry_dt and entry_dt.date().isoformat() == date_str:
                        day_entries += 1

                # Also check open trades for entries
                for t in sim.open_trades:
                    entry_dt = _parse_et(t.get("entry_time"))
                    if entry_dt and entry_dt.date().isoformat() == date_str:
                        day_entries += 1
            except Exception:
                continue

        wr = (day_wins / day_exits * 100) if day_exits > 0 else 0
        daily_stats.append({
            "date": date_str,
            "entries": day_entries,
            "exits": day_exits,
            "pnl": day_pnl,
            "win_rate": wr,
        })

    # Build digest
    lines = []
    total_pnl = sum(d["pnl"] for d in daily_stats)
    total_exits = sum(d["exits"] for d in daily_stats)
    total_wins = sum(1 for d in daily_stats if d["pnl"] > 0)

    lines.append(f"# Weekly Trade Digest -- {dates[0]} to {dates[-1]}\n")

    # Daily breakdown
    lines.append("## Daily Breakdown")
    lines.append("| Date | Entries | Exits | WR | P&L |")
    lines.append("|------|---------|-------|-----|-----|")
    for d in daily_stats:
        lines.append(f"| {d['date']} | {d['entries']} | {d['exits']} | {d['win_rate']:.0f}% | {_fmt_money(d['pnl'])} |")
    lines.append(f"| **TOTAL** | | {total_exits} | | **{_fmt_money(total_pnl)}** |")
    lines.append("")

    # Top/Bottom sims
    sorted_sims = sorted(sim_weekly_pnl.items(), key=lambda x: x[1], reverse=True)
    if sorted_sims:
        lines.append("## Sim Rankings (Week)")
        lines.append("| SIM | Weekly P&L |")
        lines.append("|-----|-----------|")
        top5 = sorted_sims[:5]
        bot5 = sorted_sims[-5:] if len(sorted_sims) > 5 else []
        for sid, pnl in top5:
            lines.append(f"| {sid} | {_fmt_money(pnl)} |")
        if bot5 and bot5 != top5:
            lines.append("| ... | ... |")
            for sid, pnl in bot5:
                lines.append(f"| {sid} | {_fmt_money(pnl)} |")
        lines.append("")

    # Exit reason frequency
    if weekly_exit_reasons:
        lines.append("## Exit Reason Frequency")
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for reason, count in sorted(weekly_exit_reasons.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    # Regime breakdown
    if weekly_regimes:
        lines.append("## Regime Breakdown")
        lines.append("| Regime | Trades |")
        lines.append("|--------|--------|")
        for regime, count in sorted(weekly_regimes.items(), key=lambda x: -x[1]):
            lines.append(f"| {regime} | {count} |")
        lines.append("")

    lines.append(f"\n*Green days: {total_wins}/5 ({total_wins/5*100:.0f}%)*\n")
    return "\n".join(lines)


def build_journal_summary(date_str: str | None = None) -> dict:
    """
    Build a compact summary dict for the Discord embed.
    Returns dict with keys: date, entries, exits, win_rate, net_pnl,
    best_trade, worst_trade, greeks_exits, journal_path.
    """
    if date_str is None:
        date_str = datetime.now(_ET).date().isoformat()

    profiles = _load_profiles()
    total_entries = 0
    total_exits = 0
    total_pnl = 0.0
    wins = 0
    best = None
    worst = None
    greeks_count = 0
    theta_count = 0
    iv_count = 0
    delta_count = 0

    for sim_id, profile in profiles.items():
        try:
            sim = SimPortfolio(sim_id, profile)
            sim.load()
            trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []

            for t in trade_log:
                exit_dt = _parse_et(t.get("exit_time"))
                if exit_dt and exit_dt.date().isoformat() == date_str:
                    total_exits += 1
                    pnl = _safe_float(t.get("realized_pnl_dollars"))
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1

                    if best is None or pnl > _safe_float(best.get("realized_pnl_dollars")):
                        best = dict(t)
                        best["sim_id"] = sim_id
                    if worst is None or pnl < _safe_float(worst.get("realized_pnl_dollars")):
                        worst = dict(t)
                        worst["sim_id"] = sim_id

                    reason = t.get("exit_reason", "")
                    if reason in _GREEKS_ALL:
                        greeks_count += 1
                        if reason in _GREEKS_THETA:
                            theta_count += 1
                        elif reason in _GREEKS_IV:
                            iv_count += 1
                        elif reason in _GREEKS_DELTA:
                            delta_count += 1

                entry_dt = _parse_et(t.get("entry_time"))
                if entry_dt and entry_dt.date().isoformat() == date_str:
                    total_entries += 1

            for t in sim.open_trades:
                entry_dt = _parse_et(t.get("entry_time"))
                if entry_dt and entry_dt.date().isoformat() == date_str:
                    total_entries += 1
        except Exception:
            continue

    wr = (wins / total_exits * 100) if total_exits > 0 else 0

    def _trade_desc(t):
        if not t:
            return "N/A"
        pnl_p = _safe_float(t.get("realized_pnl_pct"))
        return f"{t.get('sim_id')} {t.get('direction', t.get('contract_type', '?'))} {_fmt_pct(pnl_p)}"

    greeks_parts = []
    if theta_count:
        greeks_parts.append(f"{theta_count} theta")
    if iv_count:
        greeks_parts.append(f"{iv_count} IV")
    if delta_count:
        greeks_parts.append(f"{delta_count} delta")

    return {
        "date": date_str,
        "entries": total_entries,
        "exits": total_exits,
        "win_rate": round(wr, 1),
        "net_pnl": round(total_pnl, 2),
        "best_trade": _trade_desc(best),
        "worst_trade": _trade_desc(worst),
        "greeks_exits": f"{greeks_count} ({', '.join(greeks_parts)})" if greeks_parts else "0",
    }
