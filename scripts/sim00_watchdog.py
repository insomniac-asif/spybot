#!/usr/bin/env python3
"""SIM00 Watchdog — Emergency stop-loss monitor.

Runs independently of the Discord bot. If the bot crashes,
this process continues monitoring and will force-close any
SIM00 position that breaches its stop loss.

Usage:
    python scripts/sim00_watchdog.py

Or as a systemd service (see sim00-watchdog.service).
"""

import os
import sys
import json
import time
import signal
import logging
from pathlib import Path
from datetime import datetime, time as dt_time
from logging.handlers import RotatingFileHandler

# ── Configuration (all from environment) ─────────────────────────────────────

ALPACA_API_KEY = (
    os.environ.get("ALPACA_API_KEY")
    or os.environ.get("APCA_API_KEY_ID")
    or ""
)
ALPACA_SECRET_KEY = (
    os.environ.get("ALPACA_SECRET_KEY")
    or os.environ.get("APCA_API_SECRET_KEY")
    or ""
)
WATCHDOG_DISCORD_WEBHOOK = os.environ.get("WATCHDOG_DISCORD_WEBHOOK", "")
WATCHDOG_INTERVAL       = int(os.environ.get("WATCHDOG_INTERVAL",        "30"))
WATCHDOG_SIM_FILE       = os.environ.get("WATCHDOG_SIM_FILE",  "data/sims/SIM00.json")
WATCHDOG_DEFAULT_STOP   = float(os.environ.get("WATCHDOG_DEFAULT_STOP_PCT", "0.30"))
WATCHDOG_CONFIG_FILE    = os.environ.get("WATCHDOG_CONFIG_FILE", "simulation/sim_config.yaml")

# Heartbeat: log OK message no more than once every N seconds
_HEARTBEAT_INTERVAL = 300  # 5 minutes
_last_heartbeat: dict[str, float] = {}

logger = logging.getLogger("sim00_watchdog")


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "watchdog.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=2 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


# ── dotenv loader ─────────────────────────────────────────────────────────────

def load_dotenv() -> None:
    """Load .env from project root — tolerates missing file."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv as _load
        _load(dotenv_path=env_path, override=False)
        # Re-read after load in case they were empty at module import time
        global ALPACA_API_KEY, ALPACA_SECRET_KEY, WATCHDOG_DISCORD_WEBHOOK
        ALPACA_API_KEY = (
            os.environ.get("ALPACA_API_KEY")
            or os.environ.get("APCA_API_KEY_ID")
            or ""
        )
        ALPACA_SECRET_KEY = (
            os.environ.get("ALPACA_SECRET_KEY")
            or os.environ.get("APCA_API_SECRET_KEY")
            or ""
        )
        WATCHDOG_DISCORD_WEBHOOK = os.environ.get("WATCHDOG_DISCORD_WEBHOOK", "")
    except ImportError:
        # Parse manually if python-dotenv is unavailable
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = val


def validate_env() -> None:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.critical(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
            "Set them in .env or environment."
        )
        sys.exit(1)


# ── Banner ────────────────────────────────────────────────────────────────────

def _is_paper() -> bool:
    """Heuristic: Alpaca paper keys start with 'PK'."""
    return ALPACA_API_KEY.startswith("PK")


def print_banner() -> None:
    webhook_status = "configured" if WATCHDOG_DISCORD_WEBHOOK else "not configured"
    mode = "paper" if _is_paper() else "live"
    print(
        f"\nSIM00 Watchdog started\n"
        f"  Interval: {WATCHDOG_INTERVAL}s\n"
        f"  SIM file: {WATCHDOG_SIM_FILE}\n"
        f"  Default stop: {WATCHDOG_DEFAULT_STOP * 100:.1f}%\n"
        f"  Webhook: {webhook_status}\n"
        f"  Alpaca: connected ({mode})\n"
    )
    logger.info(
        "Watchdog started | interval=%ds sim_file=%s default_stop=%.0f%% mode=%s",
        WATCHDOG_INTERVAL, WATCHDOG_SIM_FILE, WATCHDOG_DEFAULT_STOP * 100, mode,
    )


# ── Market hours ──────────────────────────────────────────────────────────────

def is_market_hours() -> bool:
    """True if current ET time is within regular session (9:30–16:00, weekdays)."""
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        now_et = datetime.now(et)
    except ImportError:
        # Fallback: assume UTC-5 (ignores DST, but acceptable for a safety guard)
        from datetime import timezone, timedelta
        now_et = datetime.now(timezone(timedelta(hours=-5)))

    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    open_time  = dt_time(9, 30)
    close_time = dt_time(16, 0)
    return open_time <= now_et.time() < close_time


# ── Config read ───────────────────────────────────────────────────────────────

_config_cache: dict | None = None
_config_mtime: float = 0.0


def _get_sim00_config_stop() -> float | None:
    """Read stop_loss_pct for SIM00 from sim_config.yaml. Returns None on failure."""
    global _config_cache, _config_mtime
    cfg_path = Path(WATCHDOG_CONFIG_FILE)
    if not cfg_path.exists():
        return None
    try:
        mtime = cfg_path.stat().st_mtime
        if _config_cache is None or mtime != _config_mtime:
            import yaml
            with open(cfg_path) as f:
                _config_cache = yaml.safe_load(f)
            _config_mtime = mtime
        sim00 = _config_cache.get("SIM00", {})
        val = sim00.get("stop_loss_pct")
        if val is not None:
            return float(val)
    except Exception as e:
        logger.warning("Config read failed: %s", e)
    return None


# ── SIM file reader ───────────────────────────────────────────────────────────

def read_open_trades() -> list[dict]:
    """Parse open_trades from SIM00.json. Returns empty list on any failure."""
    sim_path = Path(WATCHDOG_SIM_FILE)
    if not sim_path.exists():
        logger.warning("SIM file not found: %s", sim_path)
        return []
    try:
        with open(sim_path) as f:
            data = json.load(f)
        trades = data.get("open_trades", [])
        if not isinstance(trades, list):
            return []
        return trades
    except Exception as e:
        logger.warning("Failed to read SIM file: %s", e)
        return []


# ── Alpaca clients (lazy init) ────────────────────────────────────────────────

_data_client = None
_trading_client = None


def _get_data_client():
    global _data_client
    if _data_client is None:
        from alpaca.data import OptionHistoricalDataClient
        _data_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _data_client


def _get_trading_client():
    global _trading_client
    if _trading_client is None:
        from alpaca.trading.client import TradingClient
        paper = _is_paper()
        _trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=paper)
    return _trading_client


def fetch_mid_price(option_symbol: str) -> float | None:
    """Return mid-price for an option symbol, or None on failure."""
    try:
        from alpaca.data.requests import OptionLatestQuoteRequest
        client = _get_data_client()
        req = OptionLatestQuoteRequest(symbol_or_symbols=[option_symbol])
        quotes = client.get_option_latest_quote(req)
        q = quotes.get(option_symbol)
        if q is None:
            logger.warning("No quote returned for %s", option_symbol)
            return None
        ask = float(q.ask_price or 0)
        bid = float(q.bid_price or 0)
        if ask <= 0 and bid <= 0:
            logger.warning("Zero bid/ask for %s", option_symbol)
            return None
        if ask <= 0:
            return bid
        if bid <= 0:
            return ask
        return (ask + bid) / 2.0
    except Exception as e:
        logger.warning("Quote fetch failed for %s: %s", option_symbol, e)
        return None


def submit_emergency_sell(option_symbol: str, qty: int) -> dict | None:
    """Submit a market sell order. Returns order dict or None on failure."""
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = _get_trading_client()
        order_req = MarketOrderRequest(
            symbol=option_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order_req)
        return {
            "id":          str(result.id),
            "status":      str(result.status),
            "filled_qty":  str(result.filled_qty),
            "filled_avg":  str(result.filled_avg_price),
        }
    except Exception as e:
        logger.warning("Emergency sell failed for %s qty=%s: %s", option_symbol, qty, e)
        return None


# ── Discord webhook ───────────────────────────────────────────────────────────

def send_webhook_alert(message: str) -> None:
    url = WATCHDOG_DISCORD_WEBHOOK
    if not url:
        return
    try:
        import requests as _req
        _req.post(
            url,
            json={"content": f"\U0001f6a8 **WATCHDOG EMERGENCY EXIT**\n{message}"},
            timeout=5,
        )
    except Exception as e:
        logger.warning("Webhook failed: %s", e)


# ── Core check logic ──────────────────────────────────────────────────────────

def check_and_protect() -> None:
    trades = read_open_trades()
    if not trades:
        return

    config_stop = _get_sim00_config_stop()

    for trade in trades:
        option_symbol = trade.get("option_symbol") or trade.get("contract")
        if not option_symbol:
            logger.debug("Trade missing option_symbol — skipping: %s", trade)
            continue

        # Determine qty
        qty_raw = trade.get("qty") or trade.get("quantity") or trade.get("contracts")
        try:
            qty = int(qty_raw)
        except (TypeError, ValueError):
            logger.warning("Invalid qty for %s: %s — skipping", option_symbol, qty_raw)
            continue

        # Determine entry price
        try:
            entry_price = float(trade.get("entry_price") or trade.get("fill_price") or 0)
        except (TypeError, ValueError):
            entry_price = 0.0
        if entry_price <= 0:
            logger.warning("Invalid entry_price for %s — skipping", option_symbol)
            continue

        # Determine stop loss pct: trade dict → sim_config.yaml → default
        trade_stop = trade.get("stop_loss_pct")
        if trade_stop is not None:
            try:
                stop_pct = float(trade_stop)
            except (TypeError, ValueError):
                stop_pct = None
        else:
            stop_pct = None

        if stop_pct is None:
            stop_pct = config_stop if config_stop is not None else WATCHDOG_DEFAULT_STOP

        stop_price = entry_price * (1.0 - stop_pct)

        # Fetch current price
        mid_price = fetch_mid_price(option_symbol)
        if mid_price is None:
            logger.warning(
                "Skipping stop check for %s (quote unavailable)", option_symbol
            )
            continue

        # --- Breach check ---
        if mid_price <= stop_price:
            logger.info(
                "EMERGENCY EXIT: %s mid=%.4f below stop=%.4f (entry=%.4f stop_pct=%.0f%%)",
                option_symbol, mid_price, stop_price, entry_price, stop_pct * 100,
            )
            order = submit_emergency_sell(option_symbol, qty)
            if order:
                logger.info(
                    "ORDER SUBMITTED: %s id=%s status=%s filled_qty=%s filled_avg=%s",
                    option_symbol,
                    order["id"],
                    order["status"],
                    order["filled_qty"],
                    order["filled_avg"],
                )
                alert_msg = (
                    f"Symbol: `{option_symbol}`\n"
                    f"Mid: `${mid_price:.4f}` | Stop: `${stop_price:.4f}` "
                    f"(entry `${entry_price:.4f}` × {100*(1-stop_pct):.0f}%)\n"
                    f"Order ID: `{order['id']}` | Status: `{order['status']}`"
                )
            else:
                alert_msg = (
                    f"Symbol: `{option_symbol}`\n"
                    f"Mid: `${mid_price:.4f}` | Stop: `${stop_price:.4f}`\n"
                    f"\u26a0\ufe0f ORDER FAILED — check Alpaca manually!"
                )
            send_webhook_alert(alert_msg)

        else:
            # Heartbeat: log OK once every 5 minutes per symbol
            now = time.time()
            last = _last_heartbeat.get(option_symbol, 0.0)
            if now - last >= _HEARTBEAT_INTERVAL:
                buffer_pct = ((mid_price / stop_price) - 1.0) * 100
                logger.debug(
                    "WATCHDOG OK: %s mid=%.4f stop=%.4f buffer=%.1f%%",
                    option_symbol, mid_price, stop_price, buffer_pct,
                )
                _last_heartbeat[option_symbol] = now


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    load_dotenv()
    validate_env()
    print_banner()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        while True:
            if not is_market_hours():
                time.sleep(60)
                continue
            check_and_protect()
            time.sleep(WATCHDOG_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Watchdog shutting down (keyboard interrupt)")
    except SystemExit:
        logger.info("Watchdog shutting down (SIGTERM)")


if __name__ == "__main__":
    main()
