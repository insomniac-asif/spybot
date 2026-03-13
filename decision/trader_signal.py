# decision/trader_signal.py
#
# Signal generation and trade creation helpers extracted from decision/trader.py.
# Covers: pre_trade_checks, generate_signal, create_trade_object.
#
# All function signatures are identical to their originals in trader.py.
# ─────────────────────────────────────────────────────────────────────────────

from analytics.edge_decay import edge_decay_status

from core.data_service import get_latest_price, get_market_dataframe
from core.debug import debug_log

from signals.predictor import make_prediction
from signals.regime import get_regime
from signals.volatility import volatility_state
from signals.setup_classifier import classify_trade
from signals.conviction import calculate_conviction

from decision.trader_utils import can_day_trade, _track_confidence_distribution

from datetime import datetime
import pytz
import uuid


def _get_predictor_mode() -> str:
    """Read predictor_mode from sim_config.yaml _global section."""
    try:
        import yaml
        import os
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulation", "sim_config.yaml")
        with open(cfg_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return str((raw.get("_global") or {}).get("predictor_mode", "veto_only")).lower()
    except Exception:
        return "veto_only"


def pre_trade_checks(acc, ctx):

    decay = edge_decay_status()

    if decay["status"] == "DISABLE":
        ctx.set_block("protection_EDGE_DECAY")
        debug_log("trade_gate_exit", gate="pre_trade_checks", reason="EDGE_DECAY")
        return "EDGE_DECAY"

    if acc["balance"] <= acc["starting_balance"] * 0.85:
        ctx.set_block("protection_EQUITY_PROTECTION")
        debug_log("trade_gate_exit", gate="pre_trade_checks", reason="EQUITY_PROTECTION")
        return "EQUITY_PROTECTION"

    if acc["daily_loss"] >= acc["max_daily_loss"]:
        ctx.set_block("protection_DAILY_LIMIT")
        debug_log("trade_gate_exit", gate="pre_trade_checks", reason="DAILY_LIMIT")
        return "DAILY_LIMIT"

    if acc["open_trade"] is not None:
        debug_log("trade_gate_exit", gate="pre_trade_checks", reason="OPEN_TRADE_EXISTS")
        return "OPEN_TRADE_EXISTS"

    if not can_day_trade(acc):
        ctx.set_block("protection_PDT_LIMIT")
        debug_log("trade_gate_exit", gate="pre_trade_checks", reason="PDT_LIMIT")
        return "PDT_LIMIT"

    return None


def generate_signal(acc, ctx):

    df = get_market_dataframe()
    if df is None:
        ctx.set_block("no_market_data")
        debug_log("trade_gate_exit", gate="generate_signal", reason="NO_MARKET_DATA")
        return None

    regime = get_regime(df)
    ctx.regime = regime
    if regime in ["COMPRESSION", "RANGE", "NO_DATA"]:
        ctx.set_block(f"regime_{regime.lower()}")
        debug_log("trade_gate_exit", gate="generate_signal", reason=f"REGIME_{regime}")
        return None

    vol_state = volatility_state(df)
    ctx.volatility = vol_state
    if vol_state in ["DEAD", "LOW"]:
        ctx.set_block(f"volatility_{vol_state.lower()}")
        debug_log("trade_gate_exit", gate="generate_signal", reason=f"VOL_{vol_state}")
        return None

    predictor_mode = _get_predictor_mode()

    bias = make_prediction(60, df)
    trigger = make_prediction(15, df)

    # Conviction-derived direction (used in veto_only/disabled modes)
    score, impulse, follow, conv_direction = calculate_conviction(df)

    if predictor_mode == "disabled":
        # Predictor disabled: skip all prediction gates, use conviction direction
        ctx.direction_60m = bias.get("direction") if bias else None
        ctx.confidence_60m = bias.get("confidence") if bias else None
        ctx.direction_15m = trigger.get("direction") if trigger else None
        ctx.confidence_15m = trigger.get("confidence") if trigger else None
        ctx.dual_alignment = None

        direction = conv_direction.upper() if conv_direction and conv_direction != "neutral" else None
        confidence = score / 6.0  # normalize conviction score (max 6) to 0-1

    elif predictor_mode == "full":
        # Legacy mode: require dual-timeframe alignment at ≥55% confidence
        if bias is None or trigger is None:
            ctx.set_block("prediction_none")
            debug_log("trade_gate_exit", gate="generate_signal", reason="PREDICTION_NONE")
            return None

        _track_confidence_distribution(bias, trigger)
        ctx.direction_60m = bias.get("direction")
        ctx.confidence_60m = bias.get("confidence")
        ctx.direction_15m = trigger.get("direction")
        ctx.confidence_15m = trigger.get("confidence")
        ctx.dual_alignment = bias.get("direction") == trigger.get("direction")

        if bias["direction"] != trigger["direction"]:
            ctx.set_block("direction_mismatch")
            debug_log("trade_gate_exit", gate="generate_signal", reason="DIRECTION_MISMATCH",
                      bias=bias["direction"], trigger=trigger["direction"])
            return None

        if bias["confidence"] < 0.55 or trigger["confidence"] < 0.55:
            ctx.set_block("confidence")
            debug_log("trade_gate_exit", gate="generate_signal", reason="CONFIDENCE_BELOW_THRESHOLD",
                      bias_conf=bias["confidence"], trigger_conf=trigger["confidence"])
            return None

        direction = bias["direction"]
        confidence = (bias["confidence"] + trigger["confidence"]) / 2

    else:
        # veto_only mode (default): predictor can only BLOCK, never originate
        if bias is not None and trigger is not None:
            _track_confidence_distribution(bias, trigger)
        ctx.direction_60m = bias.get("direction") if bias else None
        ctx.confidence_60m = bias.get("confidence") if bias else None
        ctx.direction_15m = trigger.get("direction") if trigger else None
        ctx.confidence_15m = trigger.get("confidence") if trigger else None
        ctx.dual_alignment = None

        # Direction comes from conviction, not predictor
        direction = conv_direction.upper() if conv_direction and conv_direction != "neutral" else None
        confidence = score / 6.0

        # Veto: block if predictor opposes at >70% confidence
        if direction and bias is not None:
            pred_dir = (bias.get("direction") or "").upper()
            pred_conf = bias.get("confidence", 0) or 0
            if pred_conf > 0.70 and pred_dir in ("BULLISH", "BEARISH") and pred_dir != direction:
                ctx.set_block("predictor_veto")
                debug_log("trade_gate_exit", gate="generate_signal", reason="PREDICTOR_VETO",
                          signal_dir=direction, pred_dir=pred_dir, pred_conf=pred_conf)
                return None

    if direction is None:
        ctx.set_block("no_conviction_direction")
        debug_log("trade_gate_exit", gate="generate_signal", reason="NO_CONVICTION_DIRECTION")
        return None

    price = get_latest_price()
    if price is None:
        ctx.set_block("no_latest_price")
        debug_log("trade_gate_exit", gate="generate_signal", reason="NO_LATEST_PRICE")
        return None

    ctx.spy_price = price
    setup_type = classify_trade(price, direction)

    ctx.conviction_score = score
    ctx.impulse = impulse
    ctx.follow = follow
    debug_log(
        "signal_generated",
        direction=direction,
        confidence=round(confidence, 3),
        regime=regime,
        volatility=vol_state,
        conviction=score
    )

    return (
        df,
        regime,
        vol_state,
        direction,
        confidence,
        score,
        impulse,
        follow,
        price,
        setup_type
    )


def create_trade_object(
    direction,
    style,
    price,
    stop,
    target,
    risk,
    trade_size,
    confidence,
    regime,
    vol_state,
    score,
    impulse,
    follow,
    setup_type,
    blended_score,
    ctx
):

    trade_id = str(uuid.uuid4())
    return {
        "trade_id": trade_id,
        "type": direction,
        "style": style,
        "entry_price": price,
        "size": trade_size,
        "risk": risk,
        "confidence": confidence,
        "entry_time": datetime.now(pytz.timezone("US/Eastern")).isoformat(),
        "stop": stop,
        "initial_stop": stop,
        "target": target,
        "regime": regime,
        "volatility": vol_state,
        "conviction_score": score,
        "impulse": impulse,
        "follow_through": follow,
        "ml_probability": round(blended_score, 3) if blended_score else None,
        "setup": setup_type,
        "underlying": None,
        "strike": None,
        "expiry": None,
        "option_symbol": None,
        "quantity": None,
        "decision_snapshot": {
            "timestamp": ctx.timestamp.isoformat(),
            "regime": ctx.regime,
            "volatility": ctx.volatility,
            "direction_60m": ctx.direction_60m,
            "confidence_60m": ctx.confidence_60m,
            "direction_15m": ctx.direction_15m,
            "confidence_15m": ctx.confidence_15m,
            "dual_alignment": ctx.dual_alignment,
            "conviction_score": ctx.conviction_score,
            "impulse": ctx.impulse,
            "follow": ctx.follow,
            "blended_score": ctx.blended_score,
            "threshold": ctx.threshold,
            "threshold_delta": (
                round(ctx.blended_score - ctx.threshold, 6)
                if ctx.blended_score is not None and ctx.threshold is not None
                else None
            ),
            "ml_weight": ctx.ml_weight,
            "regime_samples": ctx.regime_samples,
            "expectancy_samples": ctx.expectancy_samples
        },
        "runner_active": False,
        "partial_taken": False,
        "regime_transition_at_entry": getattr(ctx, "regime_transition", None),
        "regime_transition_severity": getattr(ctx, "regime_transition_severity", None),
    }
