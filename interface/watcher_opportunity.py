# interface/watcher_opportunity.py
#
# Opportunity watcher embed builder extracted from watchers.py.
# Contains: _build_opportunity_embed

import discord

from signals.regime import get_regime
from signals.volatility import volatility_state
from interface.fmt import ab, A, regime_col, vol_col, tier_col


def _build_opportunity_embed(result: tuple) -> discord.Embed:
    """Build the opportunity alert embed from a result tuple."""
    side = result[0]
    low = result[1]
    high = result[2]
    price = result[3]
    conviction_score = result[4]
    tp_low = result[5] if len(result) > 5 else None
    tp_high = result[6] if len(result) > 6 else None
    stop_loss = result[7] if len(result) > 7 else None

    tier_score = conviction_score

    # caller must pass vol and regime, but we accept the full result for simplicity
    # vol/regime calculation happens outside; just build the embed from provided data
    return side, low, high, price, conviction_score, tp_low, tp_high, stop_loss


def _build_opp_embed_from_parts(
    side: str,
    low: float,
    high: float,
    price: float,
    conviction_score: int,
    tp_low,
    tp_high,
    stop_loss,
    vol: str,
    regime: str,
    _format_et,
    symbol: str = "SPY",
) -> discord.Embed:
    """Build and return the opportunity embed."""
    from datetime import datetime
    import pytz

    tier_score = conviction_score
    if vol == "HIGH":
        tier_score += 1
    if regime == "TREND":
        tier_score += 1

    if tier_score >= 6:
        tier = "HIGH";  emoji = "🔥"
    elif tier_score >= 4:
        tier = "MEDIUM"; emoji = "⚡"
    else:
        tier = "LOW";   emoji = "🟡"

    opp_color = 0x2ECC71 if side == "CALL" else 0xE74C3C if side == "PUT" else 0x3498DB
    if tier == "HIGH":
        opp_color = 0x27AE60 if side == "CALL" else 0xC0392B
    opp_embed = discord.Embed(
        title=f"{emoji} [{symbol}] {tier} Strength {side} Opportunity",
        color=opp_color
    )
    side_color = "green" if side == "CALL" else "red" if side == "PUT" else "blue"
    opp_embed.add_field(name="📍 Symbol", value=ab(A(symbol, "white", bold=True)), inline=True)
    opp_embed.add_field(name="📍 Side", value=ab(A(side, side_color, bold=True)), inline=True)
    opp_embed.add_field(name="🧭 Tier", value=ab(tier_col(tier)), inline=True)
    opp_embed.add_field(name="💰 Current Price", value=ab(A(f"${price:.2f}", "white", bold=True)), inline=True)
    opp_embed.add_field(name="📐 Entry Zone", value=ab(A(f"${low:.2f} – ${high:.2f}", "cyan")), inline=True)
    if tp_low is not None and tp_high is not None:
        opp_embed.add_field(name="🎯 Take-Profit", value=ab(A(f"${tp_low:.2f} – ${tp_high:.2f}", "green", bold=True)), inline=True)
    if stop_loss is not None:
        opp_embed.add_field(name="🛑 Stop-Loss", value=ab(A(f"${stop_loss:.2f}", "red", bold=True)), inline=True)
    opp_embed.add_field(name="🔢 Conviction", value=ab(A(str(conviction_score), "yellow", bold=True)), inline=True)
    opp_embed.add_field(name="🧭 Regime", value=ab(regime_col(regime)), inline=True)
    opp_embed.add_field(name="⚡ Volatility", value=ab(vol_col(vol)), inline=True)
    opp_embed.set_footer(text=f"{_format_et(datetime.now(pytz.timezone('America/New_York')))}")
    return opp_embed
