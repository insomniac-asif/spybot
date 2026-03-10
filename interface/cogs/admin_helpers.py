"""
interface/cogs/admin_helpers.py
Pure helper data and functions for admin_commands.py.
No Discord decorators here — only static data and formatting.
"""

import discord
from interface.shared_state import _append_footer


# ── Static data ────────────────────────────────────────────────────────────────

COMMAND_LEVELS = {
    "spy": "basic",
    "predict": "basic",
    "regime": "basic",
    "conviction": "basic",
    "opportunity": "basic",
    "plan": "basic",
    "trades": "basic",
    "conviction_fix": "advanced",
    "features_reset": "advanced",
    "pred_reset": "advanced",
    "analysis": "advanced",
    "attempts": "advanced",
    "run": "advanced",
    "paperstats": "advanced",
    "career": "advanced",
    "equity": "advanced",
    "risk": "advanced",
    "expectancy": "advanced",
    "regimes": "advanced",
    "accuracy": "advanced",
    "mlstats": "advanced",
    "retrain": "advanced",
    "importance": "advanced",
    "md": "advanced",
    "simstats": "advanced",
    "simcompare": "advanced",
    "simtrades": "advanced",
    "simopen": "advanced",
    "simreset": "advanced",
    "simleaderboard": "advanced",
    "simstreaks": "advanced",
    "simregimes": "advanced",
    "simtimeofday": "advanced",
    "simpf": "advanced",
    "simconsistency": "advanced",
    "simexits": "advanced",
    "simhold": "advanced",
    "simdte": "advanced",
    "simsetups": "advanced",
    "simhealth": "advanced",
    "siminfo": "advanced",
    "simanalyze": "advanced",
    "preopen": "advanced",
    "lastskip": "advanced",
    "system": "advanced",
    "replay": "advanced",
    "helpplan": "advanced",
    "ask": "advanced",
    "askmore": "advanced",
    "backfill": "advanced",
    "query": "advanced",
    "ratelimit": "advanced",
}

COMMAND_GUIDES = {
    "plan": """
`!plan <call/put> <strike> <premium> <contracts> <expiry YYYY-MM-DD>`

Analyzes a proposed options trade using:

• Market Regime
• Volatility State
• Conviction Score
• Structure Alignment
• ATR Context
• Bias Alignment

Example:
`!plan call 435 1.20 2 2026-02-14`

Outputs:
• Market Diagnostics
• Strike Context
• Exposure Size
• AI Grade (A–D)

This does NOT place a trade. It evaluates the idea against your engine.
""",
    "predict": """
`!predict <minutes>`

Forecasts SPY direction.

Allowed values:
30 or 60

Examples:
`!predict 30`
`!predict 60`
""",
    "risk": """
`!risk`

Displays:
• Avg R
• Avg Win R
• Avg Loss R
• Max R
• Drawdown

Requires:
Minimum 10 closed trades.
""",
    "expectancy": """
`!expectancy`

Displays rolling expectancy (R-based).

Requires:
Minimum 10 closed trades.
""",
    "spy": """
`!spy` / `!qqq` / `!iwm` / `!vxx` / `!tsla` / `!aapl` / `!nvda` / `!msft`
Or: `!quote <SYMBOL>` for any symbol

Shows price snapshot for that symbol:
• Price, VWAP, EMA9, EMA20
• Session high/low with timestamps
• Sends a chart image

Note: non-SPY symbols need data — run `!backfill 5 <symbol>` first.
""",
    "regime": """
`!regime`

Displays current market regime label.
""",
    "conviction": """
`!conviction`

Displays:
• Conviction score
• Direction
• Impulse
• Follow-through
""",
    "conviction_fix": """
`!conviction_fix`

Forces a backfill of conviction expectancy:
• Fills fwd_5m / fwd_10m where possible
• Adds price/time metadata and status markers
""",
    "features_reset": """
`!features_reset`

Resets trade_features.csv to a clean header.
Use when the feature file is malformed or legacy.
""",
    "pred_reset": """
`!pred_reset`

Resets predictions.csv to a clean header.
Use when old/stale predictions are present.
""",
    "opportunity": """
`!opportunity`

Returns current opportunity zone if available.
""",
    "run": """
`!run`

Shows runtime stats:
• Trades
• Wins/Losses
• Balance
""",
    "paperstats": """
`!paperstats`

Shows paper account stats:
• Balance
• PnL
• Winrate
""",
    "career": """
`!career`

Shows career stats:
• Total trades
• Winrate
• Best balance
""",
    "equity": """
`!equity`

Sends equity curve chart (requires closed trades).
""",
    "accuracy": """
`!accuracy`

Shows prediction accuracy (requires graded predictions).
""",
    "analysis": """
`!analysis`

Decision analysis summary:
• Trades analyzed
• Corr Delta vs R
• Corr Blended vs R
• Execution no-record exits (if present)
""",
    "attempts": """
`!attempts`

Decision attempt summary (runtime):
• Attempts / Opened / Blocked
• Top block reason
• ML weight
• Avg blended vs threshold
""",
    "trades": """
`!trades <page>`

Shows paginated trade log (5 per page).
Example: `!trades 2`
""",
    "simstats": """
`!simstats` or `!simstats SIM03`

Shows sim performance stats:
• Total trades, win rate, total PnL
• Avg win/loss, expectancy, drawdown
• Best/worst trade
• Regime/time-of-day breakdowns
""",
    "simcompare": """
`!simcompare`

Side-by-side sim comparison table.
""",
    "simleaderboard": """
`!simleaderboard`

Ranks sims by key performance metrics:
• Best win rate
• Best total return / PnL
• Fastest equity growth
• Best expectancy
• Biggest winner
• High-risk / high-reward
""",
    "simanalyze": """
`!simanalyze <SIM_ID>` — Analyze trade quality for a sim.
`!simanalyze all` — Batch summary across all sims.
`!simanalyze <SIM_ID> apply` — Apply recommended entry filters (react ✅ to confirm).
`!simanalyze <SIM_ID> reset` — Remove quality filters from sim config.

Grades every closed trade A–F across 9 dimensions:
• ML confidence, regime alignment, timing quality
• Entry efficiency (MAE), exit efficiency (MFE)
• Risk/reward, hold efficiency, spread quality

Identifies what separates winners from losers and recommends
concrete entry filters (regime whitelist, time bucket blacklist,
ML confidence threshold, direction bias).

Requires ≥15 closed trades per sim for filter recommendations.
""",
    "simstreaks": """
`!simstreaks`

Win/loss streak leaders across sims.
""",
    "simregimes": """
`!simregimes`

Best sim by regime (win rate).
""",
    "simtimeofday": """
`!simtimeofday`

Best sim by time-of-day bucket (win rate).
""",
    "simpf": """
`!simpf`

Profit factor leaderboard.
""",
    "simconsistency": """
`!simconsistency`

Most consistent sims (lowest PnL volatility).
""",
    "simexits": """
`!simexits`

Best exit reason hit rates.
""",
    "simhold": """
`!simhold`

Fastest/slowest average hold time.
""",
    "md": """
`!md status`
`!md enable`
`!md disable`
`!md auto <low|medium|high>`

Toggles Momentum Decay strict mode:
• Enabled = tighter stops during decay
• Status shows last decay + warnings
• Auto mode: OFF at session transitions, ON only when detected decay meets/exceeds level
""",
    "simdte": """
`!simdte`

Best sim by DTE bucket (win rate).
""",
    "simsetups": """
`!simsetups`

Best sim by setup type (win rate).
""",
    "siminfo": """
`!siminfo 0-11`
`!siminfo SIM03`

Shows one sim's detailed strategy/config:
• Strategy intent + signal mode
• DTE/hold/cutoff profile
• Risk, stops, targets
• Optional gates (ORB/vol_z/atr_expansion/regime)
""",
    "preopen": """
`!preopen`

Runs a pre-open readiness check:
• Market open/closed status
• Data age + source
• Latest SPY close
• Option snapshot sanity (call/put + 3 OTM variants)
""",
    "simtrades": """
`!simtrades SIM03 [page]`

Shows paginated sim trade history.
""",
    "simopen": """
`!simopen` or `!simopen SIM03 [page]`

Shows open sim trades:
• Hold time
• SPY CALL/PUT expiry strike
• Entry cost + current PnL
""",
    "simreset": """
`!simreset SIM03`
`!simreset all`
`!simreset live`

Resets a sim to starting balance and clears trade history.
""",
    "lastskip": """
`!lastskip`

Shows the most recent skip reason
for trade attempts.
""",
    "regimes": """
`!regimes`

Regime expectancy stats (R-multiple).
""",
    "system": """
`!system`

Displays:
• Market status
• System health
• Active background systems
""",
    "replay": """
`!replay [symbol]`

Sends recorded session chart and live chart for the given symbol.
Defaults to SPY if no symbol given.

Examples:
  `!replay` → SPY session
  `!replay iwm` → IWM session
  `!replay tsla` → TSLA session
""",
    "helpplan": """
`!helpplan`

Quick reference for `!plan` usage.
""",
    "mlstats": """
`!mlstats`

Displays rolling ML accuracy (last 30 trades).

Requires:
At least 30 ML-evaluated trades.
""",
    "retrain": """
`!retrain`

Retrains:
• Direction model
• Edge model

Requires:
Minimum 50 logged trades in feature file.
""",
    "importance": """
`!importance`

Displays feature importance from Edge ML model.

Model must be trained first.
""",
    "backfill": """
`!backfill [days] [symbol|all]`

Fetches historical 1-min candles from Alpaca and merges into the symbol's CSV.

Examples:
`!backfill`              — SPY, 30 days
`!backfill 60`           — SPY, 60 days
`!backfill 30 QQQ`       — QQQ, 30 days
`!backfill 7 all`        — all registered symbols, 7 days

Registered symbols: SPY, QQQ, IWM, VXX, TSLA, AAPL, NVDA, MSFT
""",
    "ask": """
`!ask <option_contract>`  — Trade chart + AI analysis
`!ask <question>`         — AI reviews your performance

**Trade analysis** (OCC contract format):
`!ask SPY260321C00565000`
`!ask QQQ260321P00480000`

• Searches all sims for trades on that contract
• Generates annotated chart (entry/exit, EMAs, VWAP, RSI panel)
• GPT narrative: entry reasoning, exit quality, grade (A–F), tags
• Posts one embed per matching sim (compare strategies side-by-side)

**Performance review** (free-text question):
`!ask Did I overtrade?`
`!ask Why are my mean reversion trades losing?`

Use `!askmore` for follow-up questions.
""",
    "askmore": """
`!askmore <follow-up question>`

Continues from your previous `!ask` context.

Examples:
`!askmore break down the last 3 trades`
`!askmore include entry context and regime`
"""
}

# ── Help page builder ──────────────────────────────────────────────────────────
HELP_PAGES = [
    {
        "title": "📘 Help — Page 1/3 (Market + Core)",
        "color": 0x3498DB,
        "fields": [
            ("🟢 Market", "`!spy`, `!predict`, `!regime`, `!conviction`, `!opportunity`, `!plan`"),
            ("🟦 Core Performance", "`!trades`, `!analysis`, `!attempts`, `!run`"),
            ("🟣 Risk + Expectancy", "`!risk`, `!expectancy`, `!regimes`, `!accuracy`, `!md`"),
            ("🧭 MD Controls", "`!md status`, `!md enable`, `!md disable`, `!md auto <low|medium|high>`"),
        ],
    },
    {
        "title": "📗 Help — Page 2/3 (ML + Sims)",
        "color": 0x2ECC71,
        "fields": [
            ("🧠 ML", "`!mlstats`, `!retrain`, `!importance`"),
            ("🧪 Sims", "`!simstats`, `!simcompare`, `!simtrades`, `!simopen`, `!simleaderboard`, `!simstreaks`, `!simregimes`, `!simtimeofday`, `!simdte`, `!simsetups`, `!simpf`, `!simconsistency`, `!simexits`, `!simhold`, `!simreset`, `!simhealth`, `!siminfo`, `!simanalyze`"),
            ("⏸ Skip Status", "`!lastskip`, `!preopen`"),
        ],
    },
    {
        "title": "📙 Help — Page 3/3 (System + AI)",
        "color": 0xF39C12,
        "fields": [
            ("🖥 System", "`!system`, `!ratelimit`, `!backfill [days] [sym|all]`, `!query`, `!replay`, `!helpplan`"),
            ("🧭 Momentum Decay", "`!md status`, `!md enable`, `!md disable`, `!md auto <low|medium|high>`"),
            ("🤖 AI Coach", "`!ask <contract>` — chart + narrative  |  `!askmore`"),
            ("🧰 Maintenance", "`!conviction_fix`, `!features_reset`, `!pred_reset`"),
        ],
    },
]


def build_help_page(page_num: int) -> discord.Embed:
    """Build a paginated help embed for the given 1-based page number."""
    pages = HELP_PAGES
    page_index = max(1, min(page_num, len(pages))) - 1
    page = pages[page_index]
    embed = discord.Embed(title=page["title"], color=page["color"])
    embed.description = "Use `!help <command>` for detailed usage. Use `!help 1|2|3` for pages."
    for name, value in page["fields"]:
        embed.add_field(name=name, value=value, inline=False)
    _append_footer(embed, extra=f"Page {page_index + 1}/{len(pages)}")
    return embed
