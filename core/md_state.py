import os
import json
from datetime import datetime, timedelta
import pytz

from core.paths import DATA_DIR
from core.market_clock import market_is_open

FILE = os.path.join(DATA_DIR, "md_state.json")


def _now_et():
    return datetime.now(pytz.timezone("US/Eastern"))


def _load_state() -> dict:
    default = {
        "enabled": False,
        "mode": "manual",
        "auto_level": "high",
        "last_decay": None,
        "last_decay_level": None,
        "last_changed": None,
        "last_auto_eval": None,
        "market_open_prev": None,
    }
    if not os.path.exists(FILE):
        return default
    try:
        with open(FILE, "r") as f:
            data = json.load(f) or {}
        return {**default, **data}
    except Exception:
        return default


def _save_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(FILE), exist_ok=True)
        with open(FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


def is_md_enabled() -> bool:
    state = _load_state()
    if state.get("mode") == "auto":
        now_dt = _now_et()
        open_now = bool(market_is_open())
        prev = state.get("market_open_prev")
        changed = False
        if prev is None or bool(prev) != open_now:
            if state.get("enabled"):
                state["enabled"] = False
                changed = True
        state["market_open_prev"] = open_now
        state["last_auto_eval"] = now_dt.isoformat()
        if changed:
            state["last_changed"] = now_dt.isoformat()
        if changed or prev is None:
            _save_state(state)
    return bool(state.get("enabled"))


def _normalize_level(level) -> str:
    if level is None:
        return "high"
    text = str(level).strip().lower()
    mapping = {
        "1": "low",
        "low": "low",
        "2": "medium",
        "med": "medium",
        "medium": "medium",
        "3": "high",
        "high": "high",
    }
    return mapping.get(text, "high")


def _level_rank(level: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(_normalize_level(level), 2)


def _apply_auto_logic(state: dict, decay_detected: bool, conviction_level: str | None, now_dt: datetime) -> dict:
    open_now = bool(market_is_open())
    prev = state.get("market_open_prev")
    changed = False

    # Session transition rule: always OFF until fresh detection.
    if prev is None or bool(prev) != open_now:
        if state.get("enabled"):
            state["enabled"] = False
            changed = True
    state["market_open_prev"] = open_now
    state["last_auto_eval"] = now_dt.isoformat()

    if not open_now:
        if changed:
            state["last_changed"] = now_dt.isoformat()
        return state

    if not decay_detected:
        if state.get("enabled"):
            state["enabled"] = False
            changed = True
        if changed:
            state["last_changed"] = now_dt.isoformat()
        return state

    required = _normalize_level(state.get("auto_level"))
    observed = _normalize_level(conviction_level)
    should_enable = _level_rank(observed) >= _level_rank(required)
    if bool(state.get("enabled")) != bool(should_enable):
        state["enabled"] = bool(should_enable)
        changed = True
    if changed:
        state["last_changed"] = now_dt.isoformat()
    return state


def set_md_enabled(enabled: bool) -> dict:
    state = _load_state()
    state["mode"] = "manual"
    state["enabled"] = bool(enabled)
    state["last_changed"] = _now_et().isoformat()
    _save_state(state)
    return state


def set_md_auto(level: str | int | None = None) -> dict:
    state = _load_state()
    state["mode"] = "auto"
    state["auto_level"] = _normalize_level(level)
    # Start OFF until fresh detection.
    state["enabled"] = False
    state["last_changed"] = _now_et().isoformat()
    state["market_open_prev"] = bool(market_is_open())
    _save_state(state)
    return state


def evaluate_md_auto(decay_detected: bool, conviction_level: str | None = None, ts: datetime | None = None) -> dict:
    state = _load_state()
    if state.get("mode") != "auto":
        return state
    now_dt = ts if ts is not None else _now_et()
    state = _apply_auto_logic(state, bool(decay_detected), conviction_level, now_dt)
    _save_state(state)
    return state


def record_md_decay(ts: datetime | None = None, level: str | None = None) -> dict:
    state = _load_state()
    dt = ts if ts is not None else _now_et()
    state["last_decay"] = dt.isoformat()
    if level is not None:
        state["last_decay_level"] = _normalize_level(level)
    if state.get("mode") == "auto":
        state = _apply_auto_logic(state, True, state.get("last_decay_level"), dt)
    _save_state(state)
    return state


def get_md_state() -> dict:
    return _load_state()


def md_needs_warning(state: dict | None = None, max_age_minutes: int = 30) -> bool:
    st = state or _load_state()
    if not st.get("enabled"):
        return False
    if st.get("mode") == "auto":
        return False
    last_decay = st.get("last_decay")
    if not last_decay:
        return True
    try:
        dt = datetime.fromisoformat(last_decay)
    except Exception:
        return True
    if dt.tzinfo is None:
        dt = pytz.timezone("US/Eastern").localize(dt)
    age = (_now_et() - dt).total_seconds()
    return age > max_age_minutes * 60
