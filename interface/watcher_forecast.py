# interface/watcher_forecast.py
#
# Forecast watcher cycle helper extracted from watchers.py.
# Contains: _run_forecast_cycle

import logging
from datetime import datetime

import discord
import pytz

from core.data_service import get_symbol_dataframe, _load_symbol_registry
from signals.predictor import make_prediction
from signals.regime import get_regime
from signals.volatility import volatility_state
from analytics.prediction_stats import log_prediction
from simulation.prediction_backfill import is_prediction_locked
from interface.fmt import ab, A, lbl, conf_col


async def _run_forecast_cycle(
    channel,
    df,
    slot_time,
    _format_et,
    _send,
) -> None:
    """
    Execute one forecast watcher cycle: build predictions for all symbols
    and post the unified forecast embed.
    """
    # ── Collect predictions for all registry symbols (including SPY) ──
    _all_preds = []
    try:
        _registry = _load_symbol_registry()
        _registry_syms = list(_registry.items())
    except Exception:
        _registry_syms = []

    import asyncio as _asyncio
    _sym_names = [s.upper() for s, _ in _registry_syms]
    # Determine which symbol the pre-fetched df belongs to (first in registry)
    _primary_sym = _sym_names[0] if _sym_names else None
    # Reuse the pre-fetched df for the primary symbol to avoid a redundant call
    _fetch_tasks = [
        _asyncio.sleep(0, result=df) if (s == _primary_sym and df is not None) else _asyncio.to_thread(get_symbol_dataframe, s)
        for s in _sym_names
    ]
    _fetched_dfs = await _asyncio.gather(*_fetch_tasks, return_exceptions=True)

    for _sym_upper, _sym_df_raw in zip(_sym_names, _fetched_dfs):
        try:
            _sym_df = _sym_df_raw if not isinstance(_sym_df_raw, BaseException) else None
            if _sym_df is None or len(_sym_df) < 30:
                continue
            _sym_pred = make_prediction(10, _sym_df)
            if _sym_pred is None:
                continue
            _sym_regime = get_regime(_sym_df)
            _sym_vola = volatility_state(_sym_df)
            if not is_prediction_locked():
                log_prediction(_sym_pred, _sym_regime, _sym_vola, symbol=_sym_upper)
            _all_preds.append((_sym_upper, _sym_pred, _sym_regime, _sym_vola, _sym_df.iloc[-1]))
        except Exception:
            pass

    if not _all_preds:
        logging.warning("forecast_prediction_none")
        return

    try:
        logging.info(
            "prediction_logged",
            extra={"slot_time": slot_time.isoformat(), "symbol_count": len(_all_preds)},
        )
    except Exception:
        pass

    # Determine embed color from the first available prediction (prefer primary symbol)
    _anchor = next((x for x in _all_preds if x[0] == _primary_sym), _all_preds[0]) if _primary_sym else _all_preds[0]
    direction = _anchor[1].get("direction", "range")
    if direction == "bullish":
        fcast_color = 0x2ECC71
    elif direction == "bearish":
        fcast_color = 0xE74C3C
    else:
        fcast_color = 0x95A5A6

    def _safe_price(val):
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return "N/A"

    # ── Single unified forecast embed — direction + confidence only ──
    _dir_emojis = {"bullish": "🟢", "bearish": "🔴", "range": "⚪"}
    overview_embed = discord.Embed(
        title="📊 10-Min Forecast",
        color=fcast_color
    )
    for _s, _p, _r, _v, _last_row in _all_preds:
        _d = _p.get("direction", "range")
        _c = _p.get("confidence", 0)
        _emoji = _dir_emojis.get(_d, "⚪")
        _cur = f"${_safe_price(_last_row.get('close'))}"
        if _d == "bullish":
            _price_label = "Target"
            _price_val = f"${_safe_price(_p.get('high'))}"
            _price_color = "green"
        elif _d == "bearish":
            _price_label = "Target"
            _price_val = f"${_safe_price(_p.get('low'))}"
            _price_color = "red"
        else:
            _price_label = "Range"
            _price_val = f"${_safe_price(_p.get('low'))}–{_safe_price(_p.get('high'))}"
            _price_color = "white"
        overview_embed.add_field(
            name=f"{_emoji} {_s}",
            value=ab(
                f"{lbl('Dir')}  {A(_d.upper(), _price_color, bold=True)}  {lbl('Conf')} {conf_col(_c)}",
                f"{lbl(_price_label)} {A(_price_val, _price_color)}  {lbl('Now')} {A(_cur, 'white')}",
            ),
            inline=True
        )
    overview_embed.set_footer(text=f"Forecast | {_format_et(datetime.now(pytz.timezone('America/New_York')))}")
    await _send(channel, embed=overview_embed)
