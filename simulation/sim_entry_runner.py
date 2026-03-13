"""
simulation/sim_entry_runner.py
Entry loop for paper and live simulation trades.
Extracted from sim_engine.py to reduce file size.
"""
import asyncio
import logging
import random

from simulation.sim_portfolio import SimPortfolio, get_sim_lock
from simulation.sim_contract import get_iv_series
from execution.option_executor import get_option_price  # noqa: F401 (kept for compat)
from simulation.sim_signals import derive_sim_signal, get_signal_family
from core.market_clock import get_time_bucket
from simulation.sim_ml import predict_sim_trade
from analytics.sim_features import compute_sim_features
from core.data_service import get_symbol_dataframe, _load_symbol_registry

# Private helpers extracted to keep this file under 500 lines
from simulation.sim_entry_helpers import (
    _get_profiles_and_global,
    _trade_grade,
    _count_directional_exposure,
    _count_family_directional_exposure,
    _count_global_open_trades,
    _check_circuit_breaker,
    _build_paper_trade_dict,
    _execute_live_entry,
    _execute_paper_entry,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_sim_entries(
    df,
    regime: "str | None" = None,
    trader_signal: "dict | None" = None,
    structure_data: "dict | None" = None,
    cross_asset_data: "dict | None" = None,
    options_data: "dict | None" = None,
    all_structure_data: "dict | None" = None,
) -> list[dict]:
    _PROFILES, _GLOBAL_CONFIG = _get_profiles_and_global()
    results = []
    if not _PROFILES:
        return [{"sim_id": None, "status": "error", "reason": "no_profiles_loaded"}]
    try:
        from core.runtime_state import RUNTIME
        if not RUNTIME.can_run_paper_sims():
            return [{"sim_id": None, "status": "skipped", "reason": f"runtime_{RUNTIME.state.value.lower()}"}]
    except ImportError:
        pass
    time_of_day_bucket = get_time_bucket()

    # ── Pre-fetch each unique symbol's df ONCE (avoids 24x repeated blocking calls) ──
    _all_syms: set[str] = set()
    for _pid, _prof in _PROFILES.items():
        if str(_pid).startswith("_"):
            continue
        _syms = _prof.get("symbols")
        if _syms and isinstance(_syms, list):
            _all_syms.update(str(s).upper() for s in _syms)
        elif _prof.get("symbol"):
            _all_syms.add(str(_prof["symbol"]).upper())
    _sym_df_cache: dict = {}
    # Pre-seed the primary symbol from the already-fetched df to avoid a redundant call
    if df is not None and len(df) > 30:
        _primary = getattr(df, "attrs", {}).get("symbol") or next(iter(_load_symbol_registry() or {}), None)
        if _primary:
            _sym_df_cache[_primary.upper()] = df
    for _sym in _all_syms:
        if _sym in _sym_df_cache:
            continue
        try:
            _cached = await asyncio.to_thread(get_symbol_dataframe, _sym)
            if _cached is not None and len(_cached) > 30:
                _sym_df_cache[_sym] = _cached
        except Exception:
            pass

    # ── VIX regime overlay: classify VXX level ──────────────────────────────
    _vix_regime = "NEUTRAL"  # LOW (<15), NEUTRAL (15-25), HIGH (>25)
    _vxx_df = _sym_df_cache.get("VXX")
    if _vxx_df is not None and len(_vxx_df) > 0:
        try:
            _vxx_close = float(_vxx_df["close"].iloc[-1])
            if _vxx_close < 15:
                _vix_regime = "LOW"
            elif _vxx_close > 25:
                _vix_regime = "HIGH"
        except Exception:
            pass

    # Families boosted by VIX regime
    _LOW_VIX_FAMILIES = frozenset({"reversal", "fade"})      # MEAN_REVERSION, VWAP_REVERSION, ZSCORE_BOUNCE, etc.
    _HIGH_VIX_FAMILIES = frozenset({"trend", "breakout"})    # TREND_PULLBACK, BREAKOUT, ORB_BREAKOUT, etc.

    # Shuffle paper sims, then sort VIX-favored families to the front for priority.
    # SIM00 (live) always evaluated last so source sims process first.
    _all_sim_ids = [sid for sid in _PROFILES if not str(sid).startswith("_")]
    _paper_ids = [sid for sid in _all_sim_ids if sid != "SIM00"]
    random.shuffle(_paper_ids)

    if _vix_regime != "NEUTRAL":
        _favored = _LOW_VIX_FAMILIES if _vix_regime == "LOW" else _HIGH_VIX_FAMILIES
        _boosted = [s for s in _paper_ids if get_signal_family(str(_PROFILES[s].get("signal_mode", "")).upper()) in _favored]
        _rest = [s for s in _paper_ids if s not in set(_boosted)]
        random.shuffle(_boosted)
        random.shuffle(_rest)
        _paper_ids = _boosted + _rest

    _ordered_ids = _paper_ids + (["SIM00"] if "SIM00" in _all_sim_ids else [])

    # ── Global portfolio exposure limit ──────────────────────────────────
    _max_global = _GLOBAL_CONFIG.get("max_global_open_trades")
    if _max_global is not None:
        try:
            _global_open = _count_global_open_trades()
            if _global_open >= int(_max_global):
                logging.error("global_exposure_cap_hit: %d open trades >= limit %s", _global_open, _max_global)
                return [{"sim_id": None, "status": "skipped", "reason": "global_exposure_cap"}]
        except Exception:
            pass

    for sim_id in _ordered_ids:
        profile = _PROFILES[sim_id]
        _sim_lock = get_sim_lock(sim_id)
        await _sim_lock.acquire()
        try:
            sim = SimPortfolio(sim_id, profile)
            await asyncio.to_thread(sim.load)

            signal_mode = sim.profile.get("signal_mode", "TREND_PULLBACK")

            # ── 0. Blocked-session gate (sim-level, checked once) ─────────
            blocked_sessions = profile.get("blocked_sessions")
            if blocked_sessions and time_of_day_bucket:
                if isinstance(blocked_sessions, list) and time_of_day_bucket in blocked_sessions:
                    results.append({
                        "sim_id": sim_id,
                        "status": "skipped",
                        "reason": "blocked_session",
                        "session": time_of_day_bucket,
                        "signal_mode": signal_mode,
                    })
                    continue

            # ── 0b. Session-filter gate (allow-list, exact bucket match) ──
            session_filter = profile.get("session_filter")
            if session_filter and time_of_day_bucket:
                if time_of_day_bucket != str(session_filter).upper():
                    results.append({
                        "sim_id": sim_id,
                        "status": "skipped",
                        "reason": "session_filter",
                        "required_session": session_filter,
                        "current_session": time_of_day_bucket,
                        "signal_mode": signal_mode,
                    })
                    continue

            # ── Determine symbol list for this sim ───────────────────────
            _sim_symbols = profile.get("symbols")
            if not _sim_symbols or not isinstance(_sim_symbols, list):
                _fallback = profile.get("symbol")
                _sim_symbols = [_fallback] if _fallback else list((_load_symbol_registry() or {}).keys()) or ["SPY"]
            _sim_symbols = list(_sim_symbols)

            # Filter symbols by DTE compatibility
            _dte_max = profile.get("dte_max", 7)
            _DAILY_OPTS = {"SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT"}   # have options every weekday
            _WEEKLY_ONLY = {"VXX"}                  # weekly (Friday) expiry only
            if _dte_max == 0:
                # 0DTE: only symbols with daily options
                _filtered = [s for s in _sim_symbols if str(s).upper() in _DAILY_OPTS]
                if _filtered:
                    _sim_symbols = _filtered
            elif _dte_max < 4:
                # Short DTE (1-3): VXX has weekly-only options — skip on non-Friday days
                from datetime import datetime as _dt
                import pytz as _pytz
                _today_dow = _dt.now(_pytz.timezone("America/New_York")).weekday()
                if _today_dow != 4:  # 4 = Friday
                    _sim_symbols = [s for s in _sim_symbols if str(s).upper() not in _WEEKLY_ONLY]
            elif _dte_max > 7:
                # Long DTE: skip VXX (no monthly options)
                _sim_symbols = [s for s in _sim_symbols if str(s).upper() not in _WEEKLY_ONLY]

            # Liquid symbols (SPY, QQQ, IWM) first — always have options; shuffle within each tier
            _liquid = [s for s in _sim_symbols if str(s).upper() in _DAILY_OPTS]
            _others = [s for s in _sim_symbols if str(s).upper() not in _DAILY_OPTS]
            random.shuffle(_liquid)
            random.shuffle(_others)
            _sim_symbols = _liquid + _others

            # ── Loop over each symbol this sim trades ────────────────────
            _results_before_sym_loop = len(results)
            for _trade_symbol in _sim_symbols:
                _trade_symbol = str(_trade_symbol).upper()

                # Stop trying more symbols once this sim has opened a trade
                # this cycle — avoids spurious empty_chain/no_contract
                # notifications for symbols that have no options (e.g. VXX
                # on non-expiry days) after a trade was already placed.
                if any(
                    r.get("sim_id") == sim_id and r.get("status") == "opened"
                    for r in results[_results_before_sym_loop:]
                ):
                    break

                # Reload sim state so open_trades reflects any entries made
                # earlier in this symbol loop iteration
                await asyncio.to_thread(sim.load)
                trade_count = len(sim.trade_log) if isinstance(sim.trade_log, list) else 0

                # ── Load this symbol's df from pre-fetched cache ─────────
                sim_df = _sym_df_cache.get(_trade_symbol)
                if sim_df is None:
                    try:
                        sim_df = await asyncio.to_thread(get_symbol_dataframe, _trade_symbol)
                        if sim_df is not None and len(sim_df) > 30:
                            _sym_df_cache[_trade_symbol] = sim_df
                        else:
                            sim_df = None
                    except Exception:
                        sim_df = None
                if sim_df is None:
                    results.append({"sim_id": sim_id, "status": "skipped",
                                    "reason": "no_data", "symbol": _trade_symbol})
                    continue

                signal_meta = None
                direction = None
                underlying_price = None
                effective_profile = dict(profile)

                # ── 1. Derive signal FIRST ──────────────────────────────
                # Special path for OPPORTUNITY mode: use the ranker
                if str(signal_mode).upper() == "OPPORTUNITY":
                    try:
                        from simulation.sim_signals import derive_opportunity_signal
                        _opp_dir, _opp_price, _opp_meta = await asyncio.to_thread(
                            derive_opportunity_signal,
                            sim_df,
                            {},  # sim_states not available here; ranker uses trade logs internally
                            regime,
                            trader_signal=trader_signal,
                        )
                        direction = _opp_dir
                        underlying_price = _opp_price
                        signal_meta = _opp_meta
                    except Exception:
                        logging.exception("opportunity_signal_error: sim=%s", sim_id)
                        direction = None
                        underlying_price = None
                        signal_meta = {"reason": "opportunity_signal_error"}

                    # Apply DTE/hold overrides from ranker result
                    if isinstance(signal_meta, dict):
                        dte_min_override = signal_meta.get("recommended_dte_min")
                        dte_max_override = signal_meta.get("recommended_dte_max")
                        hold_max_override = signal_meta.get("recommended_hold_max_minutes")
                        if dte_min_override is not None:
                            effective_profile["dte_min"] = int(dte_min_override)
                        if dte_max_override is not None:
                            effective_profile["dte_max"] = int(dte_max_override)
                        if hold_max_override is not None:
                            effective_profile["hold_max_seconds"] = int(hold_max_override) * 60

                    if direction is None or underlying_price is None:
                        reason = "no_opportunity_above_threshold"
                        if isinstance(signal_meta, dict) and signal_meta.get("reason"):
                            reason = signal_meta.get("reason")
                        results.append({
                            "sim_id": sim_id,
                            "status": "skipped",
                            "reason": reason,
                            "signal_mode": signal_mode,
                        })
                        continue

                    # Skip the standard signal derivation path below (jump to entry context)
                    # We still need: entry_context, cross-sim guards, ml prediction
                    entry_context = (
                        f"signal_mode={signal_mode} | regime={regime or 'N/A'} | bucket={time_of_day_bucket or 'N/A'}"
                    )
                    if isinstance(signal_meta, dict):
                        winning_mode = signal_meta.get("winning_mode")
                        score = signal_meta.get("composite_score")
                        if winning_mode:
                            entry_context += f" | winning_mode={winning_mode}"
                        if score is not None:
                            entry_context += f" | score={score:.1f}"
                        if signal_meta.get("reason"):
                            entry_context += f" | reason={signal_meta.get('reason')}"

                    # Cross-sim directional exposure guard
                    global_config = _GLOBAL_CONFIG
                    if global_config.get("cross_sim_guard_enabled", False):
                        try:
                            max_dir_sims = int(global_config.get("max_directional_sims", 4))
                        except (TypeError, ValueError):
                            max_dir_sims = 4
                        current_dir_count = _count_directional_exposure(direction, _trade_symbol)
                        if current_dir_count >= max_dir_sims:
                            results.append({
                                "sim_id": sim_id,
                                "status": "skipped",
                                "reason": "directional_exposure_limit",
                                "direction": direction,
                                "symbol": _trade_symbol,
                                "current_count": current_dir_count,
                                "max_allowed": max_dir_sims,
                                "entry_context": entry_context,
                                "signal_mode": signal_mode,
                            })
                            continue

                    # Correlation-aware exposure guard (OPPORTUNITY path)
                    if global_config.get("correlation_guard_enabled", False):
                        try:
                            _max_corr = int(global_config.get("max_correlated_group_sims", 6))
                        except (TypeError, ValueError):
                            _max_corr = 6
                        try:
                            from simulation.correlation_guard import check_correlation_limit
                            _corr_block = check_correlation_limit(
                                direction, _trade_symbol, _PROFILES,
                                max_correlated=_max_corr, exclude_sim=sim_id,
                            )
                            if _corr_block is not None:
                                results.append({
                                    "sim_id": sim_id,
                                    "status": "skipped",
                                    "reason": "correlated_exposure_limit",
                                    "correlation_group": _corr_block.get("correlation_group"),
                                    "effective_direction": _corr_block.get("effective_direction"),
                                    "current_count": _corr_block.get("current_count"),
                                    "max_allowed": _corr_block.get("max_allowed"),
                                    "entry_context": entry_context,
                                    "signal_mode": signal_mode,
                                })
                                continue
                        except Exception:
                            pass

                    # ML prediction
                    from datetime import datetime
                    import pytz
                    ml_context = {
                        "direction": direction,
                        "price": underlying_price,
                        "regime": regime,
                        "horizon": effective_profile.get("horizon"),
                        "timestamp": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
                    }
                    ml_prediction = await asyncio.to_thread(
                        predict_sim_trade, sim_df, ml_context, feature_snapshot=feature_snapshot,
                    )

                    # Inject opportunity_meta into signal_meta for embed later
                    if isinstance(signal_meta, dict):
                        signal_meta["opportunity_meta"] = {
                            "winning_mode": signal_meta.get("winning_mode"),
                            "composite_score": signal_meta.get("composite_score"),
                            "breakdown": signal_meta.get("breakdown"),
                            "num_candidates": len(signal_meta.get("competing_candidates", [])),
                        }

                    regime_filter = sim.profile.get("regime_filter")
                    if regime_filter is not None:
                        filtered_out = False
                        if isinstance(regime_filter, list):
                            filtered_out = regime not in regime_filter
                        elif regime_filter == "TREND_ONLY":
                            filtered_out = regime != "TREND"
                        elif regime_filter == "RANGE_ONLY":
                            filtered_out = regime != "RANGE"
                        elif regime_filter == "VOLATILE_ONLY":
                            filtered_out = regime != "VOLATILE"
                        if filtered_out:
                            results.append({
                                "sim_id": sim_id,
                                "status": "skipped",
                                "reason": "regime_filter",
                                "entry_context": entry_context,
                                "signal_mode": signal_mode,
                            })
                            continue

                    _opp_meta_for_embed = signal_meta.get("opportunity_meta") if isinstance(signal_meta, dict) else None
                    _results_len_before = len(results)
                    execution_mode = sim.profile.get("execution_mode")
                    if execution_mode == "live":
                        await _execute_live_entry(
                            sim=sim, sim_id=sim_id, profile=profile,
                            _PROFILES=_PROFILES, direction=direction,
                            underlying_price=underlying_price,
                            ml_prediction=ml_prediction, regime=regime,
                            time_of_day_bucket=time_of_day_bucket,
                            signal_mode=signal_mode, entry_context=entry_context,
                            feature_snapshot=None,
                            _trade_symbol=_trade_symbol,
                            effective_profile=effective_profile, results=results,
                        )
                    else:
                        await _execute_paper_entry(
                            sim=sim, sim_id=sim_id, profile=profile,
                            direction=direction, underlying_price=underlying_price,
                            ml_prediction=ml_prediction, regime=regime,
                            time_of_day_bucket=time_of_day_bucket,
                            signal_mode=signal_mode, entry_context=entry_context,
                            feature_snapshot=None,
                            _trade_symbol=_trade_symbol,
                            effective_profile=effective_profile, df=sim_df, results=results,
                        )
                    # Attach opportunity_meta to the newly-appended result (for embed)
                    if _opp_meta_for_embed and sim_id == "SIM09":
                        for _r in results[_results_len_before:]:
                            if isinstance(_r, dict) and _r.get("status") in ("opened", "live_submitted"):
                                _r["opportunity_meta"] = _opp_meta_for_embed
                    continue  # Done with this symbol for OPPORTUNITY mode

                feature_snapshot = None
                if profile.get("features_enabled"):
                    try:
                        feature_snapshot = compute_sim_features(
                            sim_df,
                            {
                                "direction": None,
                                "price": None,
                                "regime": regime,
                                "signal_mode": signal_mode,
                                "horizon": effective_profile.get("horizon", profile.get("horizon")),
                                "dte_min": effective_profile.get("dte_min"),
                                "dte_max": effective_profile.get("dte_max"),
                                "orb_minutes": effective_profile.get("orb_minutes", profile.get("orb_minutes", 15)),
                                "zscore_window": effective_profile.get("zscore_window", profile.get("zscore_window", 30)),
                                "iv_series": get_iv_series(profile.get("iv_series_window", 200)),
                            },
                        )
                    except Exception:
                        feature_snapshot = None

                # Merge structure + cross-asset data into feature_snapshot
                if isinstance(structure_data, dict) or isinstance(cross_asset_data, dict):
                    if feature_snapshot is None:
                        feature_snapshot = {}
                    if isinstance(structure_data, dict):
                        for _sk, _sv in structure_data.items():
                            if isinstance(_sv, (int, float, bool)):
                                feature_snapshot[f"struct_{_sk}"] = _sv
                    if isinstance(cross_asset_data, dict):
                        for _xk, _xv in cross_asset_data.items():
                            if isinstance(_xv, (int, float, bool)):
                                feature_snapshot[f"xasset_{_xk}"] = _xv
                    if isinstance(options_data, dict):
                        for _ok, _ov in options_data.items():
                            if isinstance(_ov, (int, float, bool)):
                                feature_snapshot[f"opts_{_ok}"] = _ov
                    # FVG features
                    try:
                        from analytics.fair_value_gaps import compute_fvg_features
                        _fvg = compute_fvg_features(sim_df)
                        for _fk, _fv in _fvg.items():
                            if isinstance(_fv, (int, float, bool)):
                                feature_snapshot[_fk] = _fv
                    except Exception:
                        pass
                    # Add key levels from correlated symbols (QQQ, IWM)
                    if isinstance(all_structure_data, dict):
                        for _csym in ("QQQ", "IWM"):
                            _csym_struct = all_structure_data.get(_csym)
                            if isinstance(_csym_struct, dict):
                                for _ckey in ("distance_to_resistance_pct", "distance_to_support_pct", "pivot_zone", "vwap_position"):
                                    _cval = _csym_struct.get(_ckey)
                                    if isinstance(_cval, (int, float, bool)):
                                        feature_snapshot[f"struct_{_csym.lower()}_{_ckey}"] = _cval

                sig = await asyncio.to_thread(
                    derive_sim_signal,
                    sim_df,
                    signal_mode,
                    {
                        "trade_count": trade_count,
                        "atr_expansion_min": profile.get("atr_expansion_min"),
                        "vol_z_min": profile.get("vol_z_min"),
                        "require_trend_bias": profile.get("require_trend_bias"),
                        "iv_rank_max": profile.get("iv_rank_max"),
                        "vwap_z_min": profile.get("vwap_z_min"),
                        "close_z_min": profile.get("close_z_min"),
                    },
                    feature_snapshot=feature_snapshot,
                    profile=profile,
                    signal_params=profile.get("signal_params") or {},
                    structure_data=structure_data,
                    options_data=options_data,
                )
                if isinstance(sig, tuple):
                    if len(sig) >= 2:
                        direction = sig[0]
                        underlying_price = sig[1]
                    if len(sig) >= 3:
                        signal_meta = sig[2]

                # ── 2. Apply signal_meta overrides ──────────────────────
                if isinstance(signal_meta, dict):
                    for k in [
                        "dte_min", "dte_max", "hold_min_seconds", "hold_max_seconds",
                        "horizon", "orb_minutes", "zscore_window",
                    ]:
                        if signal_meta.get(k) is not None:
                            effective_profile[k] = signal_meta.get(k)

                # ── 3. Build entry_context AFTER signal_meta ────────────
                entry_context = f"signal_mode={signal_mode} | regime={regime or 'N/A'} | bucket={time_of_day_bucket or 'N/A'}"
                if isinstance(signal_meta, dict) and signal_meta.get("entry_context"):
                    entry_context = f"{entry_context} | {signal_meta.get('entry_context')}"
                if isinstance(signal_meta, dict) and signal_meta.get("reason"):
                    entry_context = f"{entry_context} | reason={signal_meta.get('reason')}"

                # ── 4. Early exit if no signal ─────────────────────────
                if direction is None or underlying_price is None:
                    if isinstance(signal_meta, dict) and signal_meta.get("reason"):
                        results.append({
                            "sim_id": sim_id,
                            "status": "skipped",
                            "reason": signal_meta.get("reason"),
                            "entry_context": entry_context,
                            "signal_mode": signal_mode,
                        })
                    # Candidate: signal evaluated but did not fire
                    try:
                        from decision.candidate import Candidate
                        from analytics.candidate_logger import log_candidate
                        log_candidate(Candidate(
                            sim_id=sim_id, strategy=signal_mode,
                            symbol=_trade_symbol, direction=None, fired=False,
                            regime=regime or "", time_bucket=time_of_day_bucket or "",
                            signal_params=profile.get("signal_params") or {},
                        ))
                    except Exception:
                        pass
                    continue

                # ── Analytics-driven adjustments ──────────────────────────
                _analytics_adj = None
                try:
                    from analytics.decision_gates import get_analytics_adjustments
                    _analytics_adj = get_analytics_adjustments(sim_id, profile)
                except Exception:
                    pass

                # ── Volatility filter gate (SIM00, SIM09 only) ──────────
                try:
                    from signals.volatility_filter import predict_trending, should_gate_sim
                    if should_gate_sim(sim_id):
                        _vol_result = predict_trending(sim_df)
                        if not _vol_result.get("pass_filter", True):
                            results.append({
                                "sim_id": sim_id,
                                "status": "skipped",
                                "reason": "volatility_filter_range",
                                "direction": direction,
                                "vol_probability": _vol_result.get("probability"),
                                "vol_reason": _vol_result.get("reason"),
                                "entry_context": entry_context,
                                "signal_mode": signal_mode,
                            })
                            continue
                except Exception:
                    pass  # filter failure = allow entry

                # ── Calibrated predictor veto gate ─────────────────────────
                _pred_mode = (_GLOBAL_CONFIG.get("predictor_mode") or "veto_only").lower()
                # Analytics override: disable predictor if accuracy too low
                if isinstance(_analytics_adj, dict) and _analytics_adj.get("predictor_override") == "disabled":
                    _pred_mode = "disabled"
                if _pred_mode == "veto_only" and direction:
                    try:
                        from signals.prediction_calibrator import calibrated_prediction
                        _now_et = None
                        try:
                            from datetime import datetime
                            import pytz
                            _now_et = datetime.now(pytz.timezone("US/Eastern"))
                            _cur_hour = _now_et.hour
                        except Exception:
                            _cur_hour = None
                        _cal_pred = calibrated_prediction(
                            minutes=60, df=sim_df, regime=regime, hour=_cur_hour,
                        )
                        if isinstance(_cal_pred, dict):
                            if not _cal_pred.get("pass_filter", True):
                                results.append({
                                    "sim_id": sim_id,
                                    "status": "skipped",
                                    "reason": "calibrator_filtered",
                                    "direction": direction,
                                    "cal_reason": _cal_pred.get("reason"),
                                    "entry_context": entry_context,
                                    "signal_mode": signal_mode,
                                })
                                continue
                            # Veto: opposing direction with LOW confidence (inverted)
                            _vp_dir = (_cal_pred.get("direction") or "").upper()
                            _vp_conf = _cal_pred.get("confidence", 0) or 0
                            _sig_dir = direction.upper()
                            if (_vp_conf < 0.40
                                    and _vp_dir in ("BULLISH", "BEARISH")
                                    and _vp_dir != _sig_dir):
                                results.append({
                                    "sim_id": sim_id,
                                    "status": "skipped",
                                    "reason": "predictor_veto",
                                    "direction": direction,
                                    "pred_direction": _vp_dir,
                                    "pred_confidence": _vp_conf,
                                    "entry_context": entry_context,
                                    "signal_mode": signal_mode,
                                })
                                continue
                    except Exception:
                        pass  # predictor failure = no veto

                # ── Cross-sim directional exposure guard ─────────────────
                global_config = _GLOBAL_CONFIG
                if global_config.get("cross_sim_guard_enabled", False):
                    try:
                        max_dir_sims = int(global_config.get("max_directional_sims", 4))
                    except (TypeError, ValueError):
                        max_dir_sims = 4
                    current_dir_count = _count_directional_exposure(direction, _trade_symbol)
                    if current_dir_count >= max_dir_sims:
                        results.append({
                            "sim_id": sim_id,
                            "status": "skipped",
                            "reason": "directional_exposure_limit",
                            "direction": direction,
                            "symbol": _trade_symbol,
                            "current_count": current_dir_count,
                            "max_allowed": max_dir_sims,
                            "entry_context": entry_context,
                            "signal_mode": signal_mode,
                        })
                        continue

                # ── Cross-sim family crowding guard ───────────────────────
                if global_config.get("cross_sim_guard_enabled", False):
                    max_family_concurrent = None
                    try:
                        v = global_config.get("max_family_concurrent")
                        if v is not None:
                            max_family_concurrent = int(v)
                    except (TypeError, ValueError):
                        pass
                    if max_family_concurrent is not None:
                        sig_family = get_signal_family(str(signal_mode).upper())
                        if sig_family not in ("unknown", "adaptive"):
                            family_count = _count_family_directional_exposure(sig_family, direction, _trade_symbol)
                            if family_count >= max_family_concurrent:
                                results.append({
                                    "sim_id": sim_id,
                                    "status": "skipped",
                                    "reason": "family_crowding_limit",
                                    "family": sig_family,
                                    "direction": direction,
                                    "family_count": family_count,
                                    "max_allowed": max_family_concurrent,
                                    "entry_context": entry_context,
                                    "signal_mode": signal_mode,
                                })
                                continue

                # ── Correlation-aware exposure guard ──────────────────────
                if global_config.get("correlation_guard_enabled", False):
                    try:
                        _max_corr = int(global_config.get("max_correlated_group_sims", 6))
                    except (TypeError, ValueError):
                        _max_corr = 6
                    try:
                        from simulation.correlation_guard import check_correlation_limit
                        _corr_block = check_correlation_limit(
                            direction, _trade_symbol, _PROFILES,
                            max_correlated=_max_corr, exclude_sim=sim_id,
                        )
                        if _corr_block is not None:
                            results.append({
                                "sim_id": sim_id,
                                "status": "skipped",
                                "reason": "correlated_exposure_limit",
                                "correlation_group": _corr_block.get("correlation_group"),
                                "effective_direction": _corr_block.get("effective_direction"),
                                "current_count": _corr_block.get("current_count"),
                                "max_allowed": _corr_block.get("max_allowed"),
                                "entry_context": entry_context,
                                "signal_mode": signal_mode,
                            })
                            continue
                    except Exception:
                        pass

                # ── Confluence scoring ─────────────────────────────────
                if direction and feature_snapshot is not None:
                    try:
                        from analytics.confluence_scorer import compute_confluence_score
                        conf = compute_confluence_score(sim_df, direction, feature_snapshot)
                        for k, v in conf.items():
                            feature_snapshot[k] = v
                    except Exception:
                        pass

                # ── IV rank entry gate (universal, configurable per sim) ──
                _iv_rank_max = None
                try:
                    _v = effective_profile.get("iv_rank_max")
                    if _v is None:
                        _v = _GLOBAL_CONFIG.get("iv_rank_max")
                    if _v is not None:
                        _iv_rank_max = float(_v)
                except (TypeError, ValueError):
                    pass
                if _iv_rank_max is None:
                    _iv_rank_max = 0.65  # default threshold

                _iv_rank_val = None
                if isinstance(feature_snapshot, dict):
                    _iv_rank_val = feature_snapshot.get("iv_rank_proxy")
                if _iv_rank_val is None:
                    # Compute from IV series if available
                    try:
                        from simulation.sim_contract import get_iv_series as _giv
                        from analytics.iv_features import compute_iv_features as _civ
                        _iv_win = int(effective_profile.get("iv_series_window", 200))
                        _iv_ser = _giv(_iv_win)
                        if _iv_ser and len(_iv_ser) >= 20:
                            _last_iv = _iv_ser[-1] if _iv_ser else None
                            _iv_feats = _civ(list(_iv_ser), _last_iv)
                            _iv_rank_val = _iv_feats.get("iv_rank_proxy")
                    except Exception:
                        pass

                if _iv_rank_val is not None:
                    try:
                        if float(_iv_rank_val) > _iv_rank_max:
                            results.append({
                                "sim_id": sim_id,
                                "status": "skipped",
                                "reason": "iv_too_expensive",
                                "iv_rank": round(float(_iv_rank_val), 3),
                                "iv_rank_max": _iv_rank_max,
                                "entry_context": entry_context,
                                "signal_mode": signal_mode,
                            })
                            continue
                    except (TypeError, ValueError):
                        pass

                # ── 5. ML prediction with real direction/price ─────────
                from datetime import datetime
                import pytz
                ml_context = {
                    "direction": direction,
                    "price": underlying_price,
                    "regime": regime,
                    "horizon": effective_profile.get("horizon"),
                    "timestamp": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
                }
                ml_prediction = await asyncio.to_thread(
                    predict_sim_trade, df, ml_context, feature_snapshot=feature_snapshot,
                )

                # ── 6. Continue with regime_filter, execution_mode, etc. ─
                regime_filter = sim.profile.get("regime_filter")
                if regime_filter is not None:
                    filtered_out = False
                    if isinstance(regime_filter, list):
                        filtered_out = regime not in regime_filter
                    elif regime_filter == "TREND_ONLY":
                        filtered_out = regime != "TREND"
                    elif regime_filter == "RANGE_ONLY":
                        filtered_out = regime != "RANGE"
                    elif regime_filter == "VOLATILE_ONLY":
                        filtered_out = regime != "VOLATILE"
                    if filtered_out:
                        results.append({
                            "sim_id": sim_id,
                            "status": "skipped",
                            "reason": "regime_filter",
                            "entry_context": entry_context,
                            "signal_mode": signal_mode,
                        })
                        continue

                # ── A-tier quality gate (no-op if quality_filters absent or empty) ──────────
                quality_filters = effective_profile.get("quality_filters", {})
                if quality_filters:
                    try:
                        from simulation.trade_analyzer import check_quality_gate
                        _ml_grade_val = _trade_grade({
                            "edge_prob": ml_prediction.get("edge_prob") if isinstance(ml_prediction, dict) else None,
                            "prediction_confidence": ml_prediction.get("confidence") if isinstance(ml_prediction, dict) else None,
                        }) if ml_prediction else None
                        _qf_skip = check_quality_gate(
                            quality_filters=quality_filters,
                            direction=direction,
                            regime=regime,
                            time_bucket=time_of_day_bucket,
                            ml_grade=_ml_grade_val,
                            signal_mode=signal_mode,
                        )
                        if _qf_skip:
                            results.append({
                                "sim_id": sim_id,
                                "status": "skipped",
                                "reason": f"quality_filter:{_qf_skip}",
                                "entry_context": entry_context,
                                "signal_mode": signal_mode,
                            })
                            continue
                    except Exception as _qf_exc:
                        logging.debug("quality_gate_error: %s", _qf_exc)

                # ── Anti-pattern filter (opt-in per sim) ──────────────────
                if profile.get("anti_pattern_filter_enabled"):
                    try:
                        from simulation.anti_pattern_filter import check_anti_patterns
                        _ap_skip = check_anti_patterns(
                            sim_id, df, direction, regime=regime,
                        )
                        if _ap_skip:
                            results.append({
                                "sim_id": sim_id,
                                "status": "skipped",
                                "reason": _ap_skip,
                                "entry_context": entry_context,
                                "signal_mode": signal_mode,
                            })
                            continue
                    except Exception as _ap_exc:
                        logging.debug("anti_pattern_filter_error: %s", _ap_exc)

                # Apply analytics size multiplier (feature drift → half size)
                if isinstance(_analytics_adj, dict) and _analytics_adj.get("size_multiplier", 1.0) < 1.0:
                    _sm = _analytics_adj["size_multiplier"]
                    _orig_risk = float(effective_profile.get("risk_per_trade_pct", 0.02))
                    effective_profile["risk_per_trade_pct"] = _orig_risk * _sm

                execution_mode = sim.profile.get("execution_mode")
                if execution_mode == "live":
                    await _execute_live_entry(
                        sim=sim, sim_id=sim_id, profile=profile,
                        _PROFILES=_PROFILES, direction=direction,
                        underlying_price=underlying_price,
                        ml_prediction=ml_prediction, regime=regime,
                        time_of_day_bucket=time_of_day_bucket,
                        signal_mode=signal_mode, entry_context=entry_context,
                        feature_snapshot=feature_snapshot,
                        _trade_symbol=_trade_symbol,
                        effective_profile=effective_profile, results=results,
                    )
                    continue

                await _execute_paper_entry(
                    sim=sim, sim_id=sim_id, profile=profile,
                    direction=direction, underlying_price=underlying_price,
                    ml_prediction=ml_prediction, regime=regime,
                    time_of_day_bucket=time_of_day_bucket,
                    signal_mode=signal_mode, entry_context=entry_context,
                    feature_snapshot=feature_snapshot,
                    _trade_symbol=_trade_symbol,
                    effective_profile=effective_profile, df=df, results=results,
                    structure_data=structure_data, cross_asset_data=cross_asset_data,
                    options_data=options_data,
                )
        except Exception as e:
            logging.exception("run_sim_entries_error: %s", e)
            results.append({"sim_id": sim_id, "status": "error", "reason": str(e)})
        finally:
            _sim_lock.release()

    # Batch-log candidates for skipped/blocked and live_submitted results.
    # "opened" + "no-fire" are already logged inline above.
    try:
        from decision.candidate import Candidate
        from analytics.candidate_logger import log_candidate
        # Reasons that indicate signal fired but gate blocked it
        _PRE_SIGNAL_REASONS = frozenset({"blocked_session"})
        for _r in results:
            _sid = _r.get("sim_id")
            _status = _r.get("status")
            if not _sid or _status in ("error", "opened"):
                continue
            if _status == "skipped":
                _reason = _r.get("reason", "")
                _fired = _reason not in _PRE_SIGNAL_REASONS
                log_candidate(Candidate(
                    sim_id=_sid,
                    strategy=_r.get("signal_mode", ""),
                    symbol="",
                    direction=None,
                    fired=_fired,
                    blocked=True,
                    block_reason=_reason,
                    regime="",
                    time_bucket=time_of_day_bucket or "",
                ))
            elif _status == "live_submitted":
                log_candidate(Candidate(
                    sim_id=_sid,
                    strategy=_r.get("signal_mode", ""),
                    symbol=_r.get("option_symbol", ""),
                    direction=_r.get("direction"),
                    fired=True,
                    entry_ref=_r.get("entry_price"),
                    regime=_r.get("regime", ""),
                    time_bucket=_r.get("time_bucket", ""),
                    traded=True,
                    trade_id=_r.get("trade_id"),
                ))
    except Exception:
        pass

    # Tag all results with VIX regime for analytics
    for _r in results:
        if isinstance(_r, dict):
            _r["vix_regime"] = _vix_regime

    return results
