import os
import time
import asyncio
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, PositionIntent, OrderStatus
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.common.exceptions import APIError

from core.debug import debug_log
from core.rate_limiter import rate_limit_sleep, get_cache, get_breaker, get_bucket
from analytics.execution_logger import log_execution
from execution.option_executor_helpers import (
    _extract_snapshot,
    _get_options_feed,
    _build_snapshot_request,
    _cache_quote,
    _get_cached_quote,
    _get_option_quote,
    ALPACA_MIN_CALL_INTERVAL_SEC,
    _LAST_OPTION_QUOTES,
    _QUOTE_TTL_SECONDS,
)

_SNAPSHOT_CACHE_TTL = 10.0  # seconds — for ResponseCache (short-lived mid-fill quotes)


async def _sleep(seconds: float) -> None:
    try:
        asyncio.get_running_loop()
        await asyncio.sleep(seconds)
    except RuntimeError:
        time.sleep(seconds)


def _increment_no_record_exit(acc, reason: str) -> None:
    if not isinstance(acc, dict):
        return
    stats = acc.get("execution_stats", {})
    if not isinstance(stats, dict):
        stats = {}
    no_record = stats.get("no_record_exits", {})
    if not isinstance(no_record, dict):
        no_record = {}
    no_record[reason] = no_record.get(reason, 0) + 1
    stats["no_record_exits"] = no_record
    acc["execution_stats"] = stats


async def _poll_fill_loop(
    client: Any,
    order: Any,
    quantity: int,
    option_symbol: str,
    expected_mid: float,
    bid: float,
    ask: float,
    order_type: str,
    ctx,
    acc,
    timeout: float = 5.0,
):
    """Poll an open order for fills. Returns (result_dict, error_str) or (None, error_str)."""
    start = time.time()
    while (time.time() - start) < timeout:
        current: Any = await asyncio.to_thread(client.get_order_by_id, order.id)
        if current.status == OrderStatus.FILLED and current.filled_avg_price is not None:
            try:
                fill_price = float(current.filled_avg_price)
                fill_ratio = 1.0
                debug_log(
                    "partial_fill_ratio",
                    requested=quantity,
                    filled=quantity,
                    ratio=round(fill_ratio, 3),
                    accepted=True,
                )
                log_execution(
                    option_symbol=option_symbol, side="entry",
                    order_type=order_type,
                    qty_requested=quantity, qty_filled=quantity,
                    fill_ratio=fill_ratio, expected_mid=expected_mid,
                    fill_price=fill_price, bid_at_order=bid, ask_at_order=ask,
                )
                if fill_price > expected_mid * 1.10:
                    try:
                        asyncio.create_task(asyncio.to_thread(client.submit_order,
                            LimitOrderRequest(
                                symbol=option_symbol,
                                qty=quantity,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY,
                                limit_price=float(fill_price),
                                position_intent=PositionIntent.SELL_TO_CLOSE,
                            )
                        ))
                    except Exception:
                        pass
                    _increment_no_record_exit(acc, "slippage_guard_triggered")
                    return None, "slippage_guard_triggered"
                return {
                    "fill_price": fill_price,
                    "filled_qty": quantity,
                    "partial": False,
                    "expected_mid": float(expected_mid),
                    "requested_qty": quantity,
                    "fill_ratio": fill_ratio,
                }, None
            except (TypeError, ValueError):
                return None, "limit_not_filled"

        filled_qty = getattr(current, "filled_qty", None)
        if filled_qty is not None and current.status != OrderStatus.FILLED:
            try:
                qty_val = int(float(filled_qty))
            except (TypeError, ValueError):
                qty_val = 0
            if qty_val > 0 and current.filled_avg_price is not None:
                try:
                    await asyncio.to_thread(client.cancel_order_by_id, order.id)
                except Exception:
                    pass
                try:
                    filled_price = float(current.filled_avg_price)
                except (TypeError, ValueError):
                    return None, "limit_not_filled"
                fill_ratio = min(qty_val, quantity) / float(quantity)
                debug_log(
                    "partial_fill_ratio",
                    requested=quantity,
                    filled=min(qty_val, quantity),
                    ratio=round(fill_ratio, 3),
                    accepted=fill_ratio >= 0.5,
                )
                if fill_ratio < 0.5:
                    try:
                        asyncio.create_task(asyncio.to_thread(close_option_position, option_symbol, min(qty_val, quantity)))
                    except Exception:
                        pass
                    if ctx is not None:
                        ctx.set_block("partial_fill_below_threshold")
                    _increment_no_record_exit(acc, "partial_fill_below_threshold")
                    return None, "partial_fill_below_threshold"
                if filled_price > expected_mid * 1.10:
                    try:
                        asyncio.create_task(asyncio.to_thread(client.submit_order,
                            LimitOrderRequest(
                                symbol=option_symbol,
                                qty=min(qty_val, quantity),
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY,
                                limit_price=float(filled_price),
                                position_intent=PositionIntent.SELL_TO_CLOSE,
                            )
                        ))
                    except Exception:
                        pass
                    _increment_no_record_exit(acc, "slippage_guard_triggered")
                    return None, "slippage_guard_triggered"
                log_execution(
                    option_symbol=option_symbol, side="entry",
                    order_type="limit_partial",
                    qty_requested=quantity, qty_filled=min(qty_val, quantity),
                    fill_ratio=fill_ratio, expected_mid=expected_mid,
                    fill_price=filled_price, bid_at_order=bid, ask_at_order=ask,
                )
                return {
                    "fill_price": filled_price,
                    "filled_qty": min(qty_val, quantity),
                    "partial": True,
                    "expected_mid": float(expected_mid),
                    "requested_qty": quantity,
                    "fill_ratio": fill_ratio,
                }, None

        if current.status in {OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED}:
            break
        await _sleep(1)

    # Timed out or terminal status — return sentinel so caller can cancel and continue
    return None, None  # None error = "not filled yet, try next attempt"


async def execute_option_entry(option_symbol: str, quantity: int, bid: float, ask: float, ctx=None, acc=None):
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        _increment_no_record_exit(acc, "missing_api_keys")
        debug_log("option_snapshot_missing_keys", symbol=option_symbol)
        return None, "missing_api_keys"

    debug_log(
        "option_entry_quote_request",
        symbol=option_symbol,
        qty=quantity,
        bid=bid,
        ask=ask,
    )
    refreshed = await asyncio.to_thread(_get_option_quote, api_key, secret_key, option_symbol)
    if refreshed is None:
        _increment_no_record_exit(acc, "quote_fetch_failed")
        return None, "quote_fetch_failed"
    bid, ask = refreshed

    mid = (bid + ask) / 2
    expected_mid = mid
    spread = ask - bid
    if mid <= 0 or spread < 0:
        _increment_no_record_exit(acc, "invalid_quote_or_spread")
        return None, "invalid_quote_or_spread"
    spread_pct = spread / ask
    if spread_pct > 0.15:
        return None, "spread_too_wide"

    client = TradingClient(api_key, secret_key, paper=True)

    # NOTE: Alpaca does NOT support bracket, OCO, or stop-limit orders for options.
    # Attempting to submit a bracket order returns HTTP 422:
    #   {"code":42210000,"message":"complex orders not supported for options trading"}
    # SL/TP enforcement is therefore handled entirely in software (sim_exit_runner.py /
    # sim_engine.py), not at the broker level. No broker-side stop orders are placed.

    # --- first attempt: limit at mid + 25% of spread ---
    first_limit = round(mid + (spread * 0.25), 2)
    try:
        order: Any = await asyncio.to_thread(client.submit_order,
            LimitOrderRequest(
                symbol=option_symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=float(first_limit),
                position_intent=PositionIntent.BUY_TO_OPEN,
            )
        )
    except APIError as _api_err:
        _increment_no_record_exit(acc, "broker_order_rejected")
        return None, f"broker_order_rejected:{_api_err}"
    result, error = await _poll_fill_loop(
        client, order, quantity, option_symbol, expected_mid, bid, ask,
        order_type="limit_mid_plus", ctx=ctx, acc=acc,
    )
    if result is not None or error is not None:
        return result, error

    try:
        await asyncio.to_thread(client.cancel_order_by_id, order.id)
    except Exception:
        pass

    # --- second attempt: limit at ask ---
    try:
        order: Any = await asyncio.to_thread(client.submit_order,
            LimitOrderRequest(
                symbol=option_symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=float(ask),
                position_intent=PositionIntent.BUY_TO_OPEN,
            )
        )
    except APIError as _api_err:
        _increment_no_record_exit(acc, "broker_order_rejected")
        return None, f"broker_order_rejected:{_api_err}"
    result, error = await _poll_fill_loop(
        client, order, quantity, option_symbol, expected_mid, bid, ask,
        order_type="limit_ask", ctx=ctx, acc=acc,
    )
    if result is not None or error is not None:
        return result, error

    try:
        await asyncio.to_thread(client.cancel_order_by_id, order.id)
    except Exception:
        pass

    _increment_no_record_exit(acc, "limit_not_filled")
    return None, "limit_not_filled"


def close_option_position(option_symbol: str, quantity: int) -> dict:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        return {"ok": False, "filled_avg_price": None, "order_id": None}

    try:
        client = TradingClient(api_key, secret_key, paper=True)
        order = client.submit_order(
            MarketOrderRequest(
                symbol=option_symbol,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                position_intent=PositionIntent.SELL_TO_CLOSE,
            )
        )
    except Exception:
        return {"ok": False, "filled_avg_price": None, "order_id": None}

    order_id = getattr(order, "id", None)
    filled_avg_price = None
    start = time.time()
    while order_id and (time.time() - start) < 10:
        try:
            current: Any = client.get_order_by_id(order_id)
        except Exception:
            break
        if current.status == OrderStatus.FILLED and current.filled_avg_price is not None:
            try:
                filled_avg_price = float(current.filled_avg_price)
            except (TypeError, ValueError):
                filled_avg_price = None
            break
        time.sleep(1)

    if filled_avg_price is not None:
        log_execution(
            option_symbol=option_symbol, side="exit",
            order_type="market",
            qty_requested=quantity, qty_filled=quantity,
            fill_ratio=1.0, expected_mid=None,
            fill_price=filled_avg_price, bid_at_order=None, ask_at_order=None,
        )

    return {"ok": True, "filled_avg_price": filled_avg_price, "order_id": order_id}


def get_option_price(option_symbol: str):
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        debug_log("option_snapshot_missing_keys", symbol=option_symbol)
        return None
    try:
        if not get_breaker().allow_request(is_exit=False):
            cached = _get_cached_quote(option_symbol)
            if cached:
                bid, ask = cached
                return (bid + ask) / 2
            return None
        from alpaca.data.historical import OptionHistoricalDataClient
        client = OptionHistoricalDataClient(api_key, secret_key)
        last_err = None
        for attempt in range(2):
            get_bucket().acquire_wait()
            rate_limit_sleep("alpaca_option_snapshot", ALPACA_MIN_CALL_INTERVAL_SEC)
            req = _build_snapshot_request(option_symbol)
            start_ts = time.time()
            try:
                snapshots = client.get_option_snapshot(req)
                get_breaker().record_success()
            except Exception as exc:
                get_breaker().record_failure()
                raise exc
            elapsed_ms = int((time.time() - start_ts) * 1000)
            snap = _extract_snapshot(snapshots, option_symbol)
            if snap is None:
                debug_log(
                    "option_snapshot_missing",
                    symbol=option_symbol,
                    attempt=attempt + 1,
                    response_type=type(snapshots).__name__,
                    response_time_ms=elapsed_ms,
                )
                last_err = "no_snapshot"
                time.sleep(0.2 * (attempt + 1))
                continue
            quote = getattr(snap, "latest_quote", None)
            if quote and quote.ask_price is not None and quote.bid_price is not None:
                bid = float(quote.bid_price)
                ask = float(quote.ask_price)
                if bid > 0 and ask > 0:
                    _cache_quote(option_symbol, bid, ask)
                    return (bid + ask) / 2
            trade = getattr(snap, "latest_trade", None)
            if trade and trade.price is not None:
                try:
                    return float(trade.price)
                except (TypeError, ValueError):
                    pass
            last_err = "no_quote"
        cached = _get_cached_quote(option_symbol)
        if cached:
            debug_log("option_snapshot_fallback_cached", symbol=option_symbol)
            bid, ask = cached
            return (bid + ask) / 2
        if last_err:
            debug_log("option_snapshot_failed", symbol=option_symbol, reason=last_err)
        return None
    except Exception:
        debug_log("option_snapshot_error", symbol=option_symbol, error="exception")
        return None
    return None
