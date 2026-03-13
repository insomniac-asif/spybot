# simulation/sim_portfolio.py
import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, date
import pytz


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SIM_DIR = os.path.join(DATA_DIR, "sims")

# ---------------------------------------------------------------------------
# Per-sim async lock registry — prevents concurrent load-mutate-save races
# between entry_runner, exit_runner, and watcher_emergency.
# ---------------------------------------------------------------------------
_sim_locks: dict[str, asyncio.Lock] = {}


def get_sim_lock(sim_id: str) -> asyncio.Lock:
    """Return (or create) an asyncio.Lock for the given sim_id."""
    if sim_id not in _sim_locks:
        _sim_locks[sim_id] = asyncio.Lock()
    return _sim_locks[sim_id]


class SimPortfolio:
    def __init__(self, sim_id: str, profile: dict):
        self.sim_id = sim_id
        self.profile = profile or {}
        self.balance = 0.0
        self.open_trades = []
        self.trade_log = []
        self.daily_loss = 0.0
        self.last_trade_day = self._today_et()
        self.peak_balance = 0.0
        self.schema_version = 1
        self.created_at = self._now_et_iso()
        self.last_updated_at = self._now_et_iso()
        # Anti-overtrading state
        self.last_entry_time_iso: str | None = None
        self.last_stop_exit_time_iso: str | None = None
        # Small-account compounding state
        self.is_dead: bool = False
        self.death_time: str | None = None
        self.death_balance: float | None = None
        self.reset_count: int = 0

    def _now_et_iso(self) -> str:
        eastern = pytz.timezone("US/Eastern")
        return datetime.now(eastern).isoformat()

    def _today_et(self) -> str:
        eastern = pytz.timezone("US/Eastern")
        return datetime.now(eastern).date().isoformat()

    def _init_from_profile(self) -> None:
        starting_balance = self.profile.get("balance_start", 0.0)
        self.balance = float(starting_balance)
        self.open_trades = []
        self.trade_log = []
        self.daily_loss = 0.0
        self.last_trade_day = self._today_et()
        self.peak_balance = float(starting_balance)
        self.schema_version = 1
        self.created_at = self._now_et_iso()
        self.last_updated_at = self.created_at
        self.last_entry_time_iso = None
        self.last_stop_exit_time_iso = None
        # Small-account compounding state
        self.is_dead = False
        self.death_time = None
        self.death_balance = None
        self.reset_count = 0

    def _path(self) -> str:
        os.makedirs(SIM_DIR, exist_ok=True)
        return os.path.join(SIM_DIR, f"{self.sim_id}.json")

    def load(self) -> None:
        path = self._path()
        if not os.path.exists(path):
            self._init_from_profile()
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._init_from_profile()
            return

        schema = data.get("schema_version")
        if schema != 1:
            logging.warning("sim_schema_version_mismatch: %s", schema)

        self.balance = float(data.get("balance", self.profile.get("balance_start", 0.0)))
        self.open_trades = data.get("open_trades", []) if isinstance(data.get("open_trades"), list) else []
        self.trade_log = data.get("trade_log", []) if isinstance(data.get("trade_log"), list) else []
        self.daily_loss = float(data.get("daily_loss", 0.0))
        self.last_trade_day = data.get("last_trade_day", self._today_et())
        self.peak_balance = float(data.get("peak_balance", self.balance))
        self.schema_version = 1
        self.created_at = data.get("created_at", self._now_et_iso())
        self.last_updated_at = data.get("last_updated_at", self.created_at)
        self.profile_snapshot = data.get("profile_snapshot", {})
        self.last_entry_time_iso = data.get("last_entry_time_iso")
        self.last_stop_exit_time_iso = data.get("last_stop_exit_time_iso")
        # Small-account compounding state
        self.is_dead = bool(data.get("is_dead", False))
        self.death_time = data.get("death_time")
        self.death_balance = data.get("death_balance")
        self.reset_count = int(data.get("reset_count", 0))
        self.reset_daily_if_needed()

    def save(self) -> None:
        path = self._path()
        tmp_path = f"{path}.tmp"
        bak_path = f"{path}.bak"
        self.last_updated_at = self._now_et_iso()

        data = {
            "sim_id": self.sim_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "profile_snapshot": self.profile,
            "balance": self.balance,
            "open_trades": self.open_trades,
            "trade_log": self.trade_log,
            "daily_loss": self.daily_loss,
            "last_trade_day": self.last_trade_day,
            "peak_balance": self.peak_balance,
            "last_entry_time_iso": self.last_entry_time_iso,
            "last_stop_exit_time_iso": self.last_stop_exit_time_iso,
            # Small-account compounding state
            "is_dead": self.is_dead,
            "death_time": self.death_time,
            "death_balance": self.death_balance,
            "reset_count": self.reset_count,
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(path):
                shutil.copy2(path, bak_path)

            os.replace(tmp_path, path)
            dir_fd = os.open(os.path.dirname(path), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def reset_daily_if_needed(self) -> None:
        today = self._today_et()
        if self.last_trade_day != today:
            self.daily_loss = 0.0
            self.last_trade_day = today
            self.save()

    def get_daily_loss_including_unrealized(self, current_prices: dict) -> float:
        """Total daily loss = realized closed P&L + unrealized open P&L."""
        realized = self.daily_loss
        unrealized = 0.0
        for trade in self.open_trades:
            if not isinstance(trade, dict):
                continue
            entry_price = trade.get("entry_price")
            qty = trade.get("qty")
            option_symbol = trade.get("option_symbol", "")
            current = current_prices.get(option_symbol)
            if current is not None and entry_price is not None:
                try:
                    loss = (float(entry_price) - float(current)) * float(qty) * 100
                    if loss > 0:
                        unrealized += loss
                except (TypeError, ValueError):
                    pass
        return realized + unrealized

    def can_trade(self) -> tuple[bool, str]:
        # Dead sims may still exit open positions but must not open new ones
        if getattr(self, "is_dead", False):
            return False, "sim_dead"

        daily_loss_limit_pct = self.profile.get("daily_loss_limit_pct")
        if daily_loss_limit_pct is not None:
            daily_loss_limit = self.balance * float(daily_loss_limit_pct)
            if self.daily_loss >= daily_loss_limit:
                return False, "daily_loss_limit"

        max_open = self.profile.get("max_open_trades")
        if max_open is not None and len(self.open_trades) >= int(max_open):
            return False, "max_open_trades"

        exposure_cap_pct = self.profile.get("exposure_cap_pct")
        if exposure_cap_pct is not None:
            exposure_cap = self.balance * float(exposure_cap_pct)
            total_exposure = 0.0
            for t in self.open_trades:
                if isinstance(t, dict):
                    try:
                        entry_price = float(t.get("entry_price", 0.0))
                        qty = float(t.get("qty", 0.0))
                        total_exposure += entry_price * qty * 100
                    except (TypeError, ValueError):
                        pass
            if total_exposure >= exposure_cap:
                return False, "exposure_cap"

        # Minimum seconds between any entries
        min_between = self.profile.get("min_seconds_between_entries")
        if min_between is not None and self.last_entry_time_iso is not None:
            try:
                eastern = pytz.timezone("US/Eastern")
                last_entry_dt = datetime.fromisoformat(self.last_entry_time_iso)
                if last_entry_dt.tzinfo is None:
                    last_entry_dt = eastern.localize(last_entry_dt)
                elapsed = (datetime.now(eastern) - last_entry_dt).total_seconds()
                if elapsed < float(min_between):
                    return False, "min_seconds_between_entries"
            except Exception:
                pass

        # Cooldown after a stop-loss exit
        cooldown = self.profile.get("cooldown_after_stop_seconds")
        if cooldown is not None and self.last_stop_exit_time_iso is not None:
            try:
                eastern = pytz.timezone("US/Eastern")
                last_stop_dt = datetime.fromisoformat(self.last_stop_exit_time_iso)
                if last_stop_dt.tzinfo is None:
                    last_stop_dt = eastern.localize(last_stop_dt)
                elapsed = (datetime.now(eastern) - last_stop_dt).total_seconds()
                if elapsed < float(cooldown):
                    return False, "cooldown_after_stop"
            except Exception:
                pass

        return True, ""

    def record_open(self, trade: dict) -> None:
        if not isinstance(trade, dict):
            return
        required = [
            "trade_id",
            "option_symbol",
            "entry_price",
            "qty",
            "entry_time",
            "sim_id",
            "horizon",
            "dte_bucket",
            "otm_pct",
        ]
        missing = [k for k in required if k not in trade]
        if missing:
            logging.warning("sim_trade_missing_fields: %s", ",".join(missing))
        if any(k not in trade or trade[k] is None for k in ["trade_id", "entry_price", "qty"]):
            logging.warning("sim_record_open_blocked_missing_critical: %s", trade.get("trade_id"))
            return
        # Reserve notional at entry so balance reflects cash in use.
        try:
            entry_price_val = float(trade.get("entry_price", 0))
            qty_val = float(trade.get("qty", 0))
            if entry_price_val > 0 and qty_val > 0 and not trade.get("cash_adjusted"):
                notional = entry_price_val * qty_val * 100
                trade["entry_notional"] = notional
                trade["cash_adjusted"] = True
                self.balance -= notional
        except (TypeError, ValueError):
            pass
        self.open_trades.append(trade)
        self.last_entry_time_iso = self._now_et_iso()

    def record_close(self, trade_id: str, exit_data: dict) -> None:
        trade = None
        remaining = []
        for t in self.open_trades:
            if isinstance(t, dict) and t.get("trade_id") == trade_id:
                trade = t
            else:
                remaining.append(t)
        if trade is None:
            logging.warning("sim_trade_not_found: %s", trade_id)
            return
        self.open_trades = remaining

        if not isinstance(exit_data, dict):
            exit_data = {}

        entry_price = trade.get("entry_price")
        qty = trade.get("qty")
        exit_price = exit_data.get("exit_price")

        try:
            entry_price_val = float(entry_price) if entry_price is not None else None
        except (TypeError, ValueError):
            entry_price_val = None
        try:
            qty_val = float(qty) if qty is not None else None
        except (TypeError, ValueError):
            qty_val = None
        try:
            exit_price_val = float(exit_price) if exit_price is not None else None
        except (TypeError, ValueError):
            exit_price_val = None

        realized_pnl_dollars = None
        if entry_price_val is not None and exit_price_val is not None and qty_val is not None:
            realized_pnl_dollars = (exit_price_val - entry_price_val) * qty_val * 100

        realized_pnl_pct = None
        if entry_price_val is not None and entry_price_val > 0 and exit_price_val is not None:
            realized_pnl_pct = (exit_price_val - entry_price_val) / entry_price_val

        if realized_pnl_dollars is not None:
            if trade.get("cash_adjusted"):
                # Add back exit notional (includes pnl).
                if entry_price_val is not None and exit_price_val is not None and qty_val is not None:
                    exit_value = exit_price_val * qty_val * 100
                    self.balance += exit_value
            else:
                # Legacy behavior (no notional reserved at entry).
                self.balance += realized_pnl_dollars
            if realized_pnl_dollars < 0:
                self.daily_loss += abs(realized_pnl_dollars)
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance

        trade_record = {}
        trade_record.update(trade)
        trade_record.update(exit_data)
        trade_record["trade_id"] = trade_id
        trade_record["sim_id"] = self.sim_id
        trade_record["realized_pnl_dollars"] = realized_pnl_dollars
        trade_record["realized_pnl_pct"] = realized_pnl_pct

        # Stamp balance snapshot onto the trade record for equity-curve analysis
        trade_record["balance_after_trade"]      = round(self.balance, 4)
        trade_record["peak_balance_after_trade"] = round(self.peak_balance, 4)
        trade_record["account_phase"]            = self._get_phase()

        self.trade_log.append(trade_record)

        # Track stop-exit timestamp for cooldown guard
        exit_reason = exit_data.get("exit_reason", "")
        if isinstance(exit_reason, str) and "stop" in exit_reason.lower():
            self.last_stop_exit_time_iso = self._now_et_iso()

        # Check for account death (small-account mode)
        try:
            from simulation.sim_account_mode import check_and_handle_death
            check_and_handle_death(self)
        except Exception:
            pass

        # Persist to SQLite trade journal (best-effort, never raises)
        try:
            from core.trade_db import insert_trade
            insert_trade(trade_record)
        except Exception:
            pass

    def _get_phase(self) -> str:
        try:
            from simulation.sim_account_mode import get_account_phase
            return get_account_phase(self.balance)
        except Exception:
            return "UNKNOWN"

    def update_open_trade_excursion(self, trade_id: str, current_price: float) -> None:
        try:
            trade = None
            for t in self.open_trades:
                if isinstance(t, dict) and t.get("trade_id") == trade_id:
                    trade = t
                    break
            if trade is None:
                return
            entry_price = float(trade.get("entry_price", 0.0))
            if entry_price <= 0:
                return
            if current_price is None:
                return
            excursion = (float(current_price) - entry_price) / entry_price
            old_mae_raw = trade.get("mae_pct")
            old_mfe_raw = trade.get("mfe_pct")
            if old_mae_raw is None or old_mfe_raw is None:
                new_mae = excursion
                new_mfe = excursion
                changed = True
            else:
                try:
                    old_mae = float(old_mae_raw)
                    old_mfe = float(old_mfe_raw)
                except (ValueError, TypeError):
                    new_mae = excursion
                    new_mfe = excursion
                    changed = True
                else:
                    new_mae = min(old_mae, excursion)
                    new_mfe = max(old_mfe, excursion)
                    changed = (new_mae != old_mae or new_mfe != old_mfe)
            if changed:
                trade["mae_pct"] = new_mae
                trade["mfe_pct"] = new_mfe
                self.save()
        except Exception:
            logging.exception("sim_update_excursion_error")
            return
