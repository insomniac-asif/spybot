# simulation/sim_report_helpers2.py
"""
Large embed builders extracted from sim_report_helpers.py.
Re-exported by sim_report_helpers for backward compatibility.
"""
import discord
from interface.fmt import (
    ab, A, lbl,
    pnl_col, conf_col, dir_col, regime_col, exit_reason_col,
    balance_col, pct_col,
)


def _helpers():
    """Lazy import of sim_report_helpers to avoid circular imports."""
    from simulation import sim_report_helpers as _rh
    return _rh


def _build_entry_embed(sim_id: str, result: dict) -> "discord.Embed":
    from simulation import sim_watcher as _sw
    status = result.get("status", "opened")
    live_flag = "LIVE" if status == "live_submitted" else "SIM"
    title = f"📥 {sim_id} {live_flag} Entry"
    embed = discord.Embed(title=title, color=0x2ecc71)
    option_symbol = result.get("option_symbol") or "unknown"
    expiry = result.get("expiry")
    direction = result.get("direction") or "N/A"
    raw_strike = result.get("strike")
    strike = None
    if isinstance(raw_strike, (int, float)):
        strike = float(raw_strike)
    elif isinstance(raw_strike, str):
        try:
            strike = float(raw_strike)
        except (TypeError, ValueError):
            strike = None
    if strike is None:
        strike = _helpers()._parse_strike_from_symbol(option_symbol)
    call_put = "CALL" if str(direction).upper() == "BULLISH" else "PUT" if str(direction).upper() == "BEARISH" else None
    expiry_text = expiry[:10] if isinstance(expiry, str) and len(expiry) >= 10 else ""
    contract_label = (result.get("symbol") or "SPY").upper()
    if call_put:
        contract_label = f"{contract_label} {call_put}"
    if expiry_text:
        contract_label = f"{contract_label} {expiry_text}"
    if isinstance(strike, (int, float)):
        contract_label = f"{contract_label} {strike:g}"
    qty = result.get("qty")
    fill_price = result.get("fill_price")
    fill_text = f"${fill_price:.4f}" if isinstance(fill_price, (int, float)) else "N/A"
    direction = result.get("direction") or "N/A"
    mode = result.get("mode") or ("LIVE" if status == "live_submitted" else "SIM")
    contract_lines = [A(contract_label, "magenta", bold=True)]
    if option_symbol:
        contract_lines.append(A(option_symbol, "white"))
    embed.add_field(name="Contract", value=ab(*contract_lines), inline=False)
    embed.add_field(
        name="Order",
        value=ab(
            f"{lbl('Qty')} {A(qty if qty is not None else 'N/A', 'white', bold=True)}  |  "
            f"{lbl('Fill')} {A(fill_text, 'white', bold=True)}  |  "
            f"{lbl('Dir')} {dir_col(direction)}"
        ),
        inline=False,
    )
    embed.add_field(name="Mode", value=ab(A(f"{mode}", "cyan", bold=True)), inline=False)
    entry_price = result.get("entry_price")
    entry_text = f"${entry_price:.4f}" if isinstance(entry_price, (int, float)) else "N/A"
    notional_text = "N/A"
    try:
        if isinstance(entry_price, (int, float)) and isinstance(qty, (int, float)):
            notional_text = f"${(float(entry_price) * float(qty) * 100):,.2f}"
    except Exception:
        notional_text = "N/A"
    risk_dollars = result.get("risk_dollars")
    risk_text = f"${risk_dollars:.0f}" if isinstance(risk_dollars, (int, float)) else "N/A"
    balance_val = result.get("balance")
    embed.add_field(
        name="Risk / Balance",
        value=ab(
            f"{lbl('Entry')} {A(entry_text, 'white', bold=True)}  |  "
            f"{lbl('Notional')} {A(notional_text, 'white', bold=True)}  |  "
            f"{lbl('Risk')} {A(risk_text, 'yellow', bold=True)}  |  "
            f"{lbl('Bal')} {balance_col(balance_val)}"
        ),
        inline=False
    )
    strike = result.get("strike")
    expiry = result.get("expiry")
    dte = result.get("dte")
    spread = result.get("spread_pct")
    regime = result.get("regime") or "N/A"
    time_bucket = result.get("time_bucket") or "N/A"
    details = []
    if strike is not None and expiry is not None:
        details.append(f"{strike} {expiry}")
    if dte is not None:
        details.append(f"DTE {dte}")
    if isinstance(spread, (int, float)):
        details.append(f"spr {spread:.3f}")
    detail_text = " | ".join(details) if details else "details N/A"
    embed.add_field(name="Details", value=ab(A(detail_text, "cyan")), inline=False)
    embed.add_field(name="Context", value=ab(f"{lbl('Regime')} {regime_col(regime)}  |  {lbl('Time')} {A(time_bucket, 'cyan')}"), inline=False)
    feature_snapshot = result.get("feature_snapshot")
    fs_text = _helpers()._format_feature_snapshot(feature_snapshot)
    if fs_text:
        embed.add_field(name="Feature Snapshot", value=ab(fs_text), inline=False)
    entry_context = result.get("entry_context")
    signal_mode = result.get("signal_mode")
    if entry_context or signal_mode:
        ctx_lines = []
        if signal_mode:
            ctx_lines.append(f"{lbl('Signal')} {A(signal_mode, 'magenta', bold=True)}")
        if entry_context:
            parts = _helpers()._format_context_parts(entry_context, drop_keys={"signal: ", "signal_mode"})
            if parts:
                ctx_lines.extend([A(p, "cyan") for p in parts])
        embed.add_field(name="Entry Context", value=ab(*ctx_lines), inline=False)
    pred_dir = result.get("predicted_direction") or "N/A"
    conf_val = result.get("prediction_confidence")
    edge_val = result.get("edge_prob")
    embed.add_field(
        name="ML",
        value=ab(
            f"{lbl('Pred')} {dir_col(pred_dir)}  |  "
            f"{lbl('Conf')} {conf_col(conf_val) if isinstance(conf_val, (int, float)) else A('N/A','gray')}  |  "
            f"{lbl('Edge')} {pct_col(edge_val, good_when_high=True, multiply=True) if isinstance(edge_val, (int, float)) else A('N/A','gray')}"
        ),
        inline=False
    )

    # SIM09-specific: opportunity ranker breakdown
    if sim_id == "SIM09":
        opp_meta = result.get("opportunity_meta")
        if isinstance(opp_meta, dict):
            winning_mode = opp_meta.get("winning_mode") or "N/A"
            composite_score = opp_meta.get("composite_score")
            score_text = f"{composite_score:.1f} / 100" if isinstance(composite_score, (int, float)) else "N/A"
            breakdown = opp_meta.get("breakdown")
            breakdown_text = "N/A"
            if isinstance(breakdown, dict):
                ss = breakdown.get("signal_strength")
                hist = breakdown.get("historical")
                reg = breakdown.get("regime")
                rr = breakdown.get("risk_reward")
                parts = []
                if ss is not None:
                    parts.append(f"Signal: {ss:.0f}")
                if hist is not None:
                    parts.append(f"History: {hist:.0f}")
                if reg is not None:
                    parts.append(f"Regime: {reg:.0f}")
                if rr is not None:
                    parts.append(f"RR: {rr:.0f}")
                if parts:
                    breakdown_text = " | ".join(parts)
            num_candidates = opp_meta.get("num_candidates", 0)
            num_text = f"{num_candidates} fired" if num_candidates else "N/A"
            embed.add_field(
                name="Winning Strategy",
                value=ab(A(winning_mode, "magenta", bold=True)),
                inline=True,
            )
            embed.add_field(
                name="Score",
                value=ab(A(score_text, "green", bold=True)),
                inline=True,
            )
            embed.add_field(
                name="Breakdown",
                value=ab(A(breakdown_text, "cyan")),
                inline=False,
            )
            embed.add_field(
                name="Candidates",
                value=ab(A(num_text, "white")),
                inline=True,
            )

    _und_sym = (result.get("symbol") or "SPY").upper()
    try:
        from core.data_service import get_symbol_dataframe
        _und_df = get_symbol_dataframe(_und_sym)
        _und_close = _und_df.iloc[-1].get("close") if _und_df is not None and len(_und_df) else None
        _und_price = float(_und_close) if _und_close is not None else None
    except Exception:
        _und_price = None
    if isinstance(_und_price, (int, float)):
        embed.add_field(name=f"{_und_sym} Price", value=ab(A(f"${_und_price:.2f}", "white", bold=True)), inline=True)
    _rh = _helpers()
    footer_parts = [f"Time: {_rh._format_et(_rh._now_et())}"]
    if _sw._SIM_LAST_DATA_AGE:
        footer_parts.append(_sw._SIM_LAST_DATA_AGE)
    embed.set_footer(text=" | ".join(footer_parts))
    return embed


def _build_exit_embed(sim_id: str, result: dict) -> "discord.Embed":
    from simulation import sim_watcher as _sw
    pnl_val = result.get("pnl")
    mode = result.get("mode") or "SIM"
    live_flag = " LIVE" if mode == "LIVE" else ""
    badge = "🟡"
    color = 0xF39C12
    if isinstance(pnl_val, (int, float)):
        if pnl_val > 0:
            badge = "✅"
            color = 0x2ECC71
        elif pnl_val < 0:
            badge = "❌"
            color = 0xE74C3C
        else:
            badge = "⚪"
            color = 0x95A5A6
    embed = discord.Embed(title=f"{badge} {sim_id}{live_flag} Exit", color=color)
    option_symbol = result.get("option_symbol") or "unknown"
    expiry = result.get("expiry")
    direction = result.get("direction") or "N/A"
    strike = result.get("strike") or _helpers()._parse_strike_from_symbol(option_symbol)
    call_put = "CALL" if str(direction).upper() == "BULLISH" else "PUT" if str(direction).upper() == "BEARISH" else None
    expiry_text = expiry[:10] if isinstance(expiry, str) and len(expiry) >= 10 else ""
    contract_label = (result.get("symbol") or "SPY").upper()
    if call_put:
        contract_label = f"{contract_label} {call_put}"
    if expiry_text:
        contract_label = f"{contract_label} {expiry_text}"
    if isinstance(strike, (int, float)):
        contract_label = f"{contract_label} {strike:g}"
    qty = result.get("qty")
    exit_price = result.get("exit_price")
    exit_reason = result.get("exit_reason", "unknown")
    entry_price = result.get("entry_price")
    pnl_val = result.get("pnl")
    balance_after = result.get("balance_after")
    hold_seconds = result.get("time_in_trade_seconds")
    exit_text = f"${exit_price:.4f}" if isinstance(exit_price, (int, float)) else "N/A"
    entry_text = f"${entry_price:.4f}" if isinstance(entry_price, (int, float)) else "N/A"
    if isinstance(hold_seconds, (int, float)):
        hold_seconds = int(hold_seconds)
        hours = hold_seconds // 3600
        minutes = (hold_seconds % 3600) // 60
        seconds = hold_seconds % 60
        if hours > 0:
            hold_text = f"{hours}h {minutes}m {seconds}s"
        else:
            hold_text = f"{minutes}m {seconds}s"
    else:
        hold_text = "N/A"
    contract_lines = [A(contract_label, "magenta", bold=True)]
    if option_symbol:
        contract_lines.append(A(option_symbol, "white"))
    embed.add_field(name="Contract", value=ab(*contract_lines), inline=False)
    embed.add_field(
        name="Exit",
        value=ab(
            f"{lbl('Qty')} {A(qty if qty is not None else 'N/A', 'white', bold=True)}  |  "
            f"{lbl('Exit')} {A(exit_text, 'white', bold=True)}  |  "
            f"{lbl('PnL')} {pnl_col(pnl_val)}"
        ),
        inline=False,
    )
    embed.add_field(name="Entry", value=ab(f"{lbl('Entry')} {A(entry_text, 'white')}  |  {lbl('Mode')} {A(mode, 'cyan', bold=True)}"), inline=False)
    embed.add_field(name="Hold / Balance", value=ab(f"{lbl('Hold')} {A(hold_text, 'cyan')}  |  {lbl('Bal')} {balance_col(balance_after)}"), inline=False)
    embed.add_field(name="Reason", value=ab(exit_reason_col(exit_reason)), inline=False)
    exit_context = result.get("exit_context")
    if exit_context:
        parts = _helpers()._format_exit_context(exit_context)
        if parts:
            embed.add_field(name="Exit Context", value=ab(*[A(p, "cyan") for p in parts]), inline=False)
    feature_snapshot = result.get("feature_snapshot")
    fs_text = _helpers()._format_feature_snapshot(feature_snapshot)
    if fs_text:
        embed.add_field(name="Feature Snapshot", value=ab(fs_text), inline=False)
    mae = result.get("mae")
    mfe = result.get("mfe")
    if isinstance(mae, (int, float)) or isinstance(mfe, (int, float)):
        mae_text = f"{mae:.2%}" if isinstance(mae, (int, float)) else "N/A"
        mfe_text = f"{mfe:.2%}" if isinstance(mfe, (int, float)) else "N/A"
        embed.add_field(
            name="Excursion",
            value=ab(f"{lbl('MFE')} {A(mfe_text, 'green')}  |  {lbl('MAE')} {A(mae_text, 'red')}"),
            inline=False,
        )
    pred_dir = result.get("predicted_direction") or "N/A"
    conf_val = result.get("prediction_confidence")
    edge_val = result.get("edge_prob")
    embed.add_field(
        name="ML",
        value=ab(
            f"{lbl('Pred')} {dir_col(pred_dir)}  |  "
            f"{lbl('Conf')} {conf_col(conf_val) if isinstance(conf_val, (int, float)) else A('N/A','gray')}  |  "
            f"{lbl('Edge')} {pct_col(edge_val, good_when_high=True, multiply=True) if isinstance(edge_val, (int, float)) else A('N/A','gray')}"
        ),
        inline=False
    )
    _und_sym2 = (result.get("symbol") or "SPY").upper()
    try:
        from core.data_service import get_symbol_dataframe
        _und_df2 = get_symbol_dataframe(_und_sym2)
        _und_close2 = _und_df2.iloc[-1].get("close") if _und_df2 is not None and len(_und_df2) else None
        _und_price2 = float(_und_close2) if _und_close2 is not None else None
    except Exception:
        _und_price2 = None
    if isinstance(_und_price2, (int, float)):
        embed.add_field(name=f"{_und_sym2} Price", value=ab(A(f"${_und_price2:.2f}", "white", bold=True)), inline=True)
    _rh = _helpers()
    footer_parts = [f"Time: {_rh._format_et(_rh._now_et())}"]
    if _sw._SIM_LAST_DATA_AGE:
        footer_parts.append(_sw._SIM_LAST_DATA_AGE)
    embed.set_footer(text=" | ".join(footer_parts))
    return embed


def _build_skip_embed(sim_id: str, result: dict) -> "discord.Embed":
    from simulation import sim_watcher as _sw
    _rh = _helpers()
    reason = result.get("reason") or "unknown"
    detail = _rh._format_skip_reason(reason)
    title = f"⏸ {sim_id} Live Skipped"
    embed = discord.Embed(title=title, color=0xF1C40F)
    embed.add_field(name="Reason", value=ab(A(reason, "yellow", bold=True)), inline=False)
    if detail:
        embed.add_field(name="Details", value=ab(A(detail, "cyan")), inline=False)
    entry_context = result.get("entry_context")
    signal_mode = result.get("signal_mode")
    if entry_context or signal_mode:
        lines = []
        if signal_mode:
            lines.append(f"{lbl('Signal')} {A(signal_mode, 'magenta', bold=True)}")
        if entry_context:
            lines.append(A(entry_context, "cyan"))
        embed.add_field(name="Context", value=ab(*lines), inline=False)
    if reason == "insufficient_trade_history":
        trade_count = result.get("trade_count")
        min_trades = result.get("min_trades_for_live")
        if isinstance(trade_count, int) and isinstance(min_trades, int):
            embed.add_field(
                name="Trade History",
                value=ab(A(f"{trade_count} / {min_trades} closed trades", "white", bold=True)),
                inline=False,
            )
    footer_parts = [f"Time: {_rh._format_et(_rh._now_et())}"]
    if _sw._SIM_LAST_DATA_AGE:
        footer_parts.append(_sw._SIM_LAST_DATA_AGE)
    embed.set_footer(text=" | ".join(footer_parts))
    return embed
