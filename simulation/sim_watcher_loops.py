# simulation/sim_watcher_loops.py
"""
Long-running async loop bodies extracted from sim_watcher.py.
These loops run as background tasks and post Discord embeds.
"""
import asyncio
import logging
import pytz
import discord
from datetime import datetime, time

from simulation.sim_portfolio import SimPortfolio
from simulation.sim_report_helpers import (
    _format_et,
    _format_option_symbol,
    _format_entry_time,
    _build_circuit_breaker_embed,
    _build_trade_history_ready_embed,
)
from interface.fmt import (
    ab, A, lbl, pnl_col, wr_col, balance_col, drawdown_col,
)


# ── constants used in sim_entry_loop ─────────────────────────────────────────

# Reasons that deduplicate skip notifications (post only on reason change)
SKIP_DEDUP_REASONS: frozenset = frozenset({
    "cutoff_passed",
    "no_candidate_expiry",
    "empty_chain",
    "chain_error",
    "no_snapshot",
    "no_snapshot_all",
    "no_snapshot_most",
    "no_snapshot_all_no_chain_symbols",
    "no_quote",
    "no_quote_all",
    "no_quote_most",
    "no_quote_all_no_chain_symbols",
    "invalid_quote",
    "invalid_quote_all",
    "invalid_quote_most",
    "invalid_quote_all_no_chain_symbols",
    "spread_too_wide",
    "spread_too_wide_all",
    "spread_too_wide_most",
    "spread_too_wide_all_no_chain_symbols",
    "snapshot_error",
    "snapshot_error_all",
    "snapshot_error_most",
    "snapshot_error_all_no_chain_symbols",
    "no_chain_symbols",
    "missing_api_keys",
    "invalid_price",
    "invalid_direction",
    "no_contract",
    "before_entry_window",
    "regime_filter",
})

# Reasons that map to the same dedup bucket — oscillating between these
# (e.g. empty_chain ↔ no_contract ↔ no_snapshot) won't retrigger the notification.
_CHAIN_FAIL_BUCKET: frozenset = frozenset({
    "empty_chain",
    "chain_error",
    "no_chain_symbols",
    "no_contract",
    "no_snapshot",
    "no_snapshot_all",
    "no_snapshot_most",
    "no_snapshot_all_no_chain_symbols",
    "no_quote",
    "no_quote_all",
    "no_quote_most",
    "no_quote_all_no_chain_symbols",
    "snapshot_error",
    "snapshot_error_all",
    "snapshot_error_most",
    "snapshot_error_all_no_chain_symbols",
})


def _skip_dedup_key(reason: str) -> str:
    """Return the dedup bucket key for a skip reason.
    Chain-failure variants all collapse to 'chain_unavailable' so they
    don't retrigger the notification when oscillating between each other.
    """
    if reason in _CHAIN_FAIL_BUCKET:
        return "chain_unavailable"
    return reason


# ── helpers (called inside loops) ────────────────────────────────────────────

def _now_et() -> datetime:
    return datetime.now(pytz.timezone("US/Eastern"))


def _build_eod_open_positions_embed(
    sim_id: str,
    lines: list,
    session_pnl: float,
    now_et: datetime,
    preopen_window: bool,
    last_data_age: str | None,
) -> discord.Embed:
    sim_desc = (
        "Pre-open snapshot of non-daytrade open positions."
        if preopen_window
        else "Open non-daytrade positions at market close."
    )
    sim_embed = discord.Embed(
        title=f"📌 {sim_id} Open Positions",
        description=ab(A(sim_desc, "yellow", bold=True)),
        color=0xF39C12,
    )
    sim_embed.add_field(name="Session PnL", value=ab(pnl_col(session_pnl)), inline=True)
    sim_embed.add_field(
        name="Positions",
        value=ab(*[A(line, "white") for line in lines]),
        inline=False,
    )
    footer_parts = [f"Time: {_format_et(now_et)}"]
    if last_data_age:
        footer_parts.append(last_data_age)
    sim_embed.set_footer(text=" | ".join(footer_parts))
    return sim_embed


def _build_eod_combined_embed(
    lines_by_sim: dict,
    pnl_by_sim: dict,
    now_et: datetime,
    last_data_age: str | None,
) -> discord.Embed:
    if not lines_by_sim:
        embed = discord.Embed(
            title="📌 Open Positions Snapshot",
            description=ab(A("No open non-daytrade SIM positions.", "green", bold=True)),
            color=0x2ECC71,
        )
    else:
        embed = discord.Embed(
            title="📌 Open Positions Snapshot",
            description=ab(A("Non-daytrade SIM positions carrying overnight.", "yellow", bold=True)),
            color=0xF39C12,
        )
        for sim_id, lines in sorted(lines_by_sim.items()):
            session_pnl = pnl_by_sim.get(sim_id, 0.0)
            embed.add_field(
                name=f"{sim_id} | Session PnL {session_pnl:+.2f}",
                value=ab(*[A(line, "white") for line in lines]),
                inline=False,
            )
    footer_parts = [f"Time: {_format_et(now_et)}"]
    if last_data_age:
        footer_parts.append(last_data_age)
    embed.set_footer(text=" | ".join(footer_parts))
    return embed


def _build_graduation_embed(
    grad_sim_id: str,
    grad_source: str,
    src_count: int,
    grad_min: int,
    now_et: datetime,
    last_data_age: str | None,
) -> discord.Embed:
    pct = min(src_count / grad_min * 100, 100)
    filled = int(pct / 10)
    bar = "█" * filled + "░" * (10 - filled)
    embed = discord.Embed(
        title=f"📊 {grad_sim_id} — Trade History Progress",
        description=ab(A("EOD snapshot: trades logged toward live graduation.", "yellow")),
        color=0xF39C12,
    )
    embed.add_field(name="Source Sim", value=ab(A(grad_source, "cyan", bold=True)), inline=True)
    embed.add_field(
        name="Progress",
        value=ab(A(f"{src_count} / {grad_min}  [{bar}]  {pct:.0f}%", "yellow", bold=True)),
        inline=False,
    )
    footer = [f"Time: {_format_et(now_et)}"]
    if last_data_age:
        footer.append(last_data_age)
    embed.set_footer(text=" | ".join(footer))
    return embed


def _build_daily_sim_embed(
    sim_id: str,
    today_trades: list,
    wins: int,
    losses: int,
    day_pnl: float,
    sim_balance: float,
    sim_peak_balance: float,
    mode: str,
    today_str: str,
) -> discord.Embed:
    wr = wins / len(today_trades) if today_trades else 0
    sim_embed = discord.Embed(
        title=f"📊 {sim_id} Daily Summary",
        color=0x2ECC71 if day_pnl >= 0 else 0xE74C3C,
    )
    sim_embed.add_field(name="Trades", value=ab(A(str(len(today_trades)), "white", bold=True)), inline=True)
    sim_embed.add_field(name="W / L", value=ab(A(f"{wins}W", "green"), A(" / ", "gray"), A(f"{losses}L", "red")), inline=True)
    sim_embed.add_field(name="Win Rate", value=ab(wr_col(wr)), inline=True)
    sim_embed.add_field(name="Day PnL", value=ab(pnl_col(day_pnl)), inline=True)
    sim_embed.add_field(name="Balance", value=ab(balance_col(sim_balance)), inline=True)
    dd = sim_peak_balance - sim_balance if sim_peak_balance > sim_balance else 0
    dd_pct = dd / sim_peak_balance if sim_peak_balance > 0 else 0
    sim_embed.add_field(name="Drawdown", value=ab(drawdown_col(dd_pct)), inline=True)
    sim_embed.set_footer(text=f"{today_str} | {mode}")
    return sim_embed


def _build_daily_combined_embed(
    combined_lines: list,
    total_day_trades: int,
    total_day_pnl: float,
    today_str: str,
    now_et: datetime,
) -> discord.Embed:
    color = 0x2ECC71 if total_day_pnl >= 0 else 0xE74C3C
    embed = discord.Embed(title=f"📊 Daily Summary — {today_str}", color=color)
    embed.add_field(
        name="Totals",
        value=ab(
            f"{lbl('Trades')} {A(str(total_day_trades), 'white', bold=True)}  |  "
            f"{lbl('Combined PnL')} {pnl_col(total_day_pnl)}"
        ),
        inline=False,
    )
    combined_lines.sort(key=lambda x: x["day_pnl"], reverse=True)
    for item in combined_lines:
        mode_tag = " 🔴" if item["mode"] == "LIVE" else ""
        embed.add_field(
            name=f"{item['sim_id']}{mode_tag}",
            value=ab(
                f"{lbl('PnL')} {pnl_col(item['day_pnl'])}  |  "
                f"{lbl('W/L')} {A(str(item['wins']), 'green')}/{A(str(item['losses']), 'red')}  |  "
                f"{lbl('WR')} {wr_col(item['wr'])}  |  "
                f"{lbl('Bal')} {balance_col(item['balance'])}"
            ),
            inline=False,
        )
    embed.set_footer(text=f"{_format_et(now_et)}")
    return embed


def _collect_weekly_sim_data(sim_profiles: dict) -> list:
    """Collect PnL, WR, profit-factor data for all sims (weekly leaderboard)."""
    sim_data = []
    for sim_id, profile in sim_profiles.items():
        if str(sim_id).startswith("_"):
            continue
        try:
            sim = SimPortfolio(sim_id, profile)
            sim.load()
            trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []
            pnl_vals = []
            wins = 0
            for t in trade_log:
                pnl = t.get("realized_pnl_dollars")
                if pnl is None:
                    continue
                try:
                    pnl_f = float(pnl)
                except (TypeError, ValueError):
                    continue
                pnl_vals.append(pnl_f)
                if pnl_f > 0:
                    wins += 1
            total = len(pnl_vals)
            total_pnl = sum(pnl_vals) if pnl_vals else 0
            wr = wins / total if total > 0 else 0
            exp = total_pnl / total if total > 0 else 0
            win_sum = sum(p for p in pnl_vals if p > 0)
            loss_sum = abs(sum(p for p in pnl_vals if p < 0))
            pf = win_sum / loss_sum if loss_sum > 0 else 0
            mode = "LIVE" if profile.get("execution_mode") == "live" and profile.get("enabled") else "SIM"
            sim_data.append({
                "sim_id": sim_id,
                "name": profile.get("name", sim_id),
                "mode": mode,
                "trades": total,
                "wr": wr,
                "exp": exp,
                "pf": pf,
                "total_pnl": total_pnl,
                "balance": sim.balance,
                "sufficient": total >= 10,
            })
        except Exception:
            continue
    return sim_data


def _build_weekly_leaderboard_embed(
    sim_data: list,
    now_et: datetime,
) -> discord.Embed:
    try:
        from analytics.composite_score import compute_composite_score, WEIGHTS
        from interface.shared_metrics import _load_sim_profiles
        profiles = _load_sim_profiles()
    except Exception:
        profiles = {}

    eligible_raw = [s for s in sim_data if s["sufficient"]]
    insufficient  = [s for s in sim_data if not s["sufficient"]]

    # Attach composite scores to each eligible item
    scored = []
    for item in eligible_raw:
        sim_id  = item["sim_id"]
        profile = profiles.get(sim_id, {})
        try:
            r = compute_composite_score(sim_id, profile)
        except Exception:
            r = {"unranked": True, "composite_score": 0.0, "grade": "?", "emoji": "?"}
        item["_composite"] = r.get("composite_score") or 0.0
        item["_grade"]     = r.get("grade") or "?"
        item["_emoji"]     = r.get("emoji") or "?"
        scored.append(item)

    scored.sort(key=lambda x: x["_composite"], reverse=True)

    embed = discord.Embed(
        title=f"🏆 Weekly Sim Leaderboard — {now_et.date().isoformat()}",
        color=0xF1C40F,
    )
    rank_lines = []
    for rank, item in enumerate(scored, 1):
        medal      = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        mode_tag   = " 🔴" if item["mode"] == "LIVE" else ""
        comp_str   = f"{item['_composite']:.1f}"
        pf_str     = f"{item['pf']:.2f}"
        rank_lines.append(
            f"{lbl(medal)} {A(item['sim_id'] + mode_tag, 'cyan', bold=True)} "
            f"{A(item['_emoji'], 'white')} {A(item['_grade'], 'yellow', bold=True)} "
            f"{A(comp_str, 'white', bold=True)}  "
            f"{lbl('WR')} {wr_col(item['wr'])}  "
            f"{lbl('PF')} {A(pf_str, 'white')}  "
            f"{lbl('PnL')} {pnl_col(item['total_pnl'])}"
        )

    for item in insufficient:
        insuff_text = f"({item['trades']} trades — need 10)"
        rank_lines.append(
            f"{lbl('—')} {A(item['sim_id'], 'gray')}  "
            f"{A(insuff_text, 'gray')}"
        )

    for i in range(0, len(rank_lines), 5):
        chunk = rank_lines[i:i + 5]
        embed.add_field(name="Rankings" if i == 0 else "​", value=ab(*chunk), inline=False)

    w = WEIGHTS if "WEIGHTS" in dir() else {}
    embed.set_footer(text=(
        f"Composite: 25% P&L + 20% WR + 25% PF + 20% Consistency + 10% DD Control"
        f" | {_format_et(now_et)}"
    ))
    return embed


def _collect_session_sim_rows(sim_profiles: dict, today_str: str) -> tuple[list, float, int]:
    """Collect per-sim session stats for the session leaderboard."""
    sim_rows = []
    total_day_pnl = 0.0
    total_day_trades = 0
    for sim_id, profile in sim_profiles.items():
        if str(sim_id).startswith("_"):
            continue
        try:
            sim = SimPortfolio(sim_id, profile)
            sim.load()
            trade_log = sim.trade_log if isinstance(sim.trade_log, list) else []
            today_trades = []
            for t in trade_log:
                exit_time = t.get("exit_time")
                if not exit_time:
                    continue
                try:
                    exit_date = datetime.fromisoformat(str(exit_time)).date().isoformat()
                except Exception:
                    continue
                if exit_date == today_str:
                    today_trades.append(t)
            wins = 0
            losses = 0
            day_pnl = 0.0
            for t in today_trades:
                try:
                    pnl = float(t.get("realized_pnl_dollars", 0))
                except (TypeError, ValueError):
                    continue
                day_pnl += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
            mode = "LIVE" if profile.get("execution_mode") == "live" else "SIM"
            sim_rows.append({
                "sim_id": sim_id,
                "mode": mode,
                "trades": len(today_trades),
                "wins": wins,
                "losses": losses,
                "wr": wins / len(today_trades) if today_trades else 0.0,
                "day_pnl": day_pnl,
                "balance": sim.balance,
            })
            total_day_pnl += day_pnl
            total_day_trades += len(today_trades)
        except Exception:
            continue
    return sim_rows, total_day_pnl, total_day_trades


def _build_session_leaderboard_page(
    sim_rows: list,
    total_day_pnl: float,
    total_day_trades: int,
    today_str: str,
    now_et: datetime,
    page_idx: int,
    total_pages: int,
    page_sims: list,
    sims_per_page: int,
) -> discord.Embed:
    color = 0x2ECC71 if total_day_pnl >= 0 else 0xE74C3C
    page_num = page_idx + 1
    embed = discord.Embed(
        title=f"🏆 Session Leaderboard — {today_str}  [{page_num}/{total_pages}]",
        color=color,
    )
    if page_idx == 0:
        best = next((s for s in sim_rows if s["trades"] > 0), None)
        summary_lines = [
            f"{lbl('Trades')} {A(str(total_day_trades), 'white', bold=True)}  |  "
            f"{lbl('Total PnL')} {pnl_col(total_day_pnl)}"
        ]
        if best:
            summary_lines.append(
                f"{lbl('Best')} {A(best['sim_id'], 'cyan', bold=True)}  {pnl_col(best['day_pnl'])}"
            )
        embed.add_field(name="Session", value=ab(*summary_lines), inline=False)

    for item in page_sims:
        mode_tag = " 🔴" if item["mode"] == "LIVE" else ""
        sim_label = f"{item['sim_id']}{mode_tag}  ·  {item['mode']}"
        if item["trades"] == 0:
            field_val = ab(A("No trades this session", "gray"))
        else:
            field_val = ab(
                f"{lbl('PnL')} {pnl_col(item['day_pnl'])}  |  "
                f"{lbl('Trades')} {A(str(item['trades']), 'white', bold=True)}  |  "
                f"{lbl('W/L')} {A(str(item['wins']), 'green')}/{A(str(item['losses']), 'red')}  |  "
                f"{lbl('WR')} {wr_col(item['wr'])}"
            )
        embed.add_field(name=sim_label, value=field_val, inline=False)

    embed.set_footer(text=f"{_format_et(now_et)}")
    return embed
