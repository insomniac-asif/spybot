#!/usr/bin/env python3
"""
scripts/backfill_predictions.py

Walk-forward historical prediction backfill from 1m bar CSVs.

Rules:
  - At each 30-minute mark (10:00, 10:30, 11:00 … 15:30) during every
    trading session, a prediction is generated using ONLY bars that
    existed before that moment — zero lookahead.
  - Each prediction is immediately graded against the bar at T+30.
  - Duplicate (time, symbol) pairs in predictions.csv are skipped by
    default; use --overwrite to regenerate everything.

Usage:
    python scripts/backfill_predictions.py
    python scripts/backfill_predictions.py --symbols SPY QQQ
    python scripts/backfill_predictions.py --from-date 2026-03-01
    python scripts/backfill_predictions.py --overwrite
"""

import argparse
import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.paths import DATA_DIR
from signals.predictor import make_prediction
from signals.session_classifier import classify_session
from analytics.prediction_stats import PRED_FILE, PRED_HEADERS

ALL_SYMBOLS = ["SPY", "QQQ", "IWM", "VXX", "TSLA", "AAPL", "NVDA", "MSFT"]

# Bars required before first prediction (=30 minutes of 1m data)
MIN_HISTORY = 30


# ──────────────────────────────────────────── helpers

def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative intraday VWAP column (resets at each trading day)."""
    df = df.copy()
    df["_date"] = df["timestamp"].dt.date
    df["_tp"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["_tpv"] = df["_tp"] * df["volume"]
    df["_cum_tpv"] = df.groupby("_date")["_tpv"].cumsum()
    df["_cum_v"]   = df.groupby("_date")["volume"].cumsum()
    df["vwap"] = (df["_cum_tpv"] / df["_cum_v"].replace(0.0, float("nan"))).bfill()
    return df.drop(columns=["_date", "_tp", "_tpv", "_cum_tpv", "_cum_v"])


def _regime(df: pd.DataFrame) -> str:
    """Compute market regime from a past-only dataframe."""
    if len(df) < 60:
        return "NO_DATA"
    recent = df.tail(60)
    price_change = recent["close"].iloc[-1] - recent["close"].iloc[0]
    hi = recent["high"].max()
    lo = recent["low"].min()
    total_range = hi - lo
    avg_candle = (recent["high"] - recent["low"]).mean()
    vwap = recent["vwap"].mean() if "vwap" in recent.columns else recent["close"].mean()
    n_above = (recent["close"] > vwap).sum()
    n_below = (recent["close"] < vwap).sum()
    directionality = abs(price_change) / total_range if total_range else 0
    if avg_candle < 0.08:
        return "COMPRESSION"
    if directionality > 0.6 and abs(n_above - n_below) > 30:
        return "TREND"
    if total_range > 1.2:
        return "VOLATILE"
    return "RANGE"


def _volatility(df: pd.DataFrame) -> str:
    if len(df) < 30:
        return "LOW"
    recent = df.tail(30)
    vol = recent["high"].max() - recent["low"].min()
    if vol < 0.35:   return "DEAD"
    if vol < 0.75:   return "LOW"
    if vol < 1.50:   return "NORMAL"
    return "HIGH"


def _conf_band(c: float) -> str:
    if c < 0.60:   return "low"
    if c < 0.75:   return "medium"
    return "high"


def _load_existing_keys(pred_file: str) -> set:
    """Return set of (time_str, symbol) already in predictions.csv."""
    if not os.path.exists(pred_file) or os.path.getsize(pred_file) == 0:
        return set()
    try:
        df = pd.read_csv(pred_file, usecols=["time", "symbol"])
        return set(zip(df["time"].astype(str), df["symbol"].str.upper()))
    except Exception:
        return set()


# ── Iterative learning config ────────────────────────────────────────────────
LEARN_EVERY = 2000  # Update predictor weights every N graded predictions
                     # Increased from 500 to reduce overhead while still learning iteratively


# ──────────────────────────────────────────── core walk-forward logic

def _prepare_symbol_data(symbol: str) -> pd.DataFrame | None:
    """Load and prepare a symbol's CSV for prediction generation."""
    csv_path = os.path.join(DATA_DIR, f"{symbol.lower()}_1m.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(minutes >= 570) & (minutes < 960)].reset_index(drop=True)
    if df.empty:
        return None
    df = _add_vwap(df)
    return df


def _update_weights_from_batch(batch: list[dict], cumulative: list[dict]) -> dict | None:
    """Recompute predictor weights from cumulative graded predictions."""
    from analytics.predictor_optimizer import compute_weights, WEIGHTS_FILE
    graded = [r for r in cumulative if r.get("checked")]
    if len(graded) < 30:
        return None
    graded_df = pd.DataFrame(graded)
    weights = compute_weights(graded_df)
    if weights:
        import json
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
    return weights


def backfill_all_symbols_chronological(
    symbols: list[str],
    from_date=None,
    existing_keys: set | None = None,
) -> list[dict]:
    """
    Walk-forward prediction backfill across ALL symbols chronologically.

    Instead of processing one symbol at a time, this interleaves all symbols
    by date so that predictor weights are updated periodically and later
    predictions benefit from patterns learned from earlier ones.

    Every LEARN_EVERY graded predictions, weights are recomputed and saved,
    so the predictor improves as it processes more data.
    """
    if existing_keys is None:
        existing_keys = set()

    # Load all symbol DataFrames
    sym_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _prepare_symbol_data(sym)
        if df is not None:
            sym_data[sym] = df
            print(f"  Loaded {sym}: {len(df):,} bars")

    if not sym_data:
        return []

    # Build unified date list across all symbols
    all_dates: set = set()
    sym_day_groups: dict[str, dict] = {}
    for sym, df in sym_data.items():
        groups = {}
        for date_val, day_df in df.groupby(df["timestamp"].dt.date):
            groups[date_val] = day_df.sort_values("timestamp").reset_index(drop=True)
            all_dates.add(date_val)
        sym_day_groups[sym] = groups

    from_date_ts = pd.Timestamp(from_date).date() if from_date is not None else None
    sorted_dates = sorted(all_dates)

    rows_out: list[dict] = []
    graded_since_last_learn = 0
    total_graded = 0
    learn_count = 0

    for date in sorted_dates:
        if from_date_ts is not None and date < from_date_ts:
            continue

        # Standard 10-min prediction slots
        session_start = pd.Timestamp(date).replace(hour=9, minute=30)
        session_end = pd.Timestamp(date).replace(hour=15, minute=50)
        pred_timestamps = pd.date_range(
            start=session_start + pd.Timedelta(minutes=10),
            end=session_end,
            freq="10min",
        ).tolist()

        for pred_ts in pred_timestamps:
            for sym in symbols:
                if sym not in sym_day_groups:
                    continue
                day_df = sym_day_groups[sym].get(date)
                if day_df is None:
                    continue

                full_df = sym_data[sym]
                time_str = pred_ts.strftime("%Y-%m-%d %H:%M:%S")

                # Skip duplicates
                if (time_str, sym.upper()) in existing_keys:
                    continue

                # Past data only
                past_df = full_df[full_df["timestamp"] < pred_ts].copy()
                if len(past_df) < MIN_HISTORY:
                    continue

                # Generate prediction
                pred = make_prediction(10, past_df)
                if pred is None:
                    continue

                pred["time"] = pred_ts

                intraday_past = day_df[day_df["timestamp"] < pred_ts]
                regime = _regime(intraday_past if len(intraday_past) >= 60 else past_df)
                vol_state = _volatility(intraday_past if len(intraday_past) >= 30 else past_df)
                session = classify_session(time_str)

                # Grade against 10-min window of future bars
                target_ts = pred_ts + pd.Timedelta(minutes=10)
                future = day_df[(day_df["timestamp"] >= pred_ts) & (day_df["timestamp"] < target_ts)]

                if future.empty:
                    rows_out.append({
                        "time": time_str, "symbol": sym.upper(),
                        "timeframe": 10, "direction": pred["direction"],
                        "confidence": round(pred["confidence"], 3),
                        "high": pred["high"], "low": pred["low"],
                        "regime": regime, "volatility": vol_state,
                        "session": session, "actual": "", "correct": 0,
                        "checked": False, "high_hit": "", "low_hit": "",
                        "price_at_check": "", "close_at_check": "",
                        "confidence_band": _conf_band(pred["confidence"]),
                    })
                    continue

                window_high = float(future["high"].max())
                window_low = float(future["low"].min())
                window_close = float(future.iloc[-1]["close"])

                high_hit = int(window_high >= float(pred["high"]))
                low_hit = int(window_low <= float(pred["low"]))

                if high_hit and not low_hit:       result = "bullish"
                elif low_hit and not high_hit:      result = "bearish"
                elif high_hit and low_hit:          result = "both"
                else:                               result = "range"

                correct = int(result == pred["direction"])

                row = {
                    "time": time_str, "symbol": sym.upper(),
                    "timeframe": 10, "direction": pred["direction"],
                    "confidence": round(pred["confidence"], 3),
                    "high": pred["high"], "low": pred["low"],
                    "regime": regime, "volatility": vol_state,
                    "session": session, "actual": result,
                    "correct": correct, "checked": True,
                    "high_hit": high_hit, "low_hit": low_hit,
                    "price_at_check": round(window_close, 2),
                    "close_at_check": round(window_close, 2),
                    "confidence_band": _conf_band(pred["confidence"]),
                }
                rows_out.append(row)

                graded_since_last_learn += 1
                total_graded += 1

                # ── Periodic learning: update weights every LEARN_EVERY graded predictions
                if graded_since_last_learn >= LEARN_EVERY:
                    graded_rows = [r for r in rows_out if r.get("checked")]
                    weights = _update_weights_from_batch(graded_rows[-LEARN_EVERY:], graded_rows)
                    learn_count += 1
                    if weights:
                        wr = weights.get("meta", {}).get("overall_wr", 0)
                        print(f"  [LEARN #{learn_count}] {total_graded} graded, WR={wr:.1%}, "
                              f"weights updated from {len(graded_rows)} samples")
                    graded_since_last_learn = 0

    print(f"\n  Generated {len(rows_out)} predictions, {total_graded} graded, "
          f"{learn_count} weight updates")
    return rows_out


def backfill_symbol(symbol: str, csv_path: str, from_date=None) -> list[dict]:
    """
    Walk-forward prediction + grading for one symbol (legacy single-symbol mode).
    Returns a list of row dicts (all columns from PRED_HEADERS).
    Caller is responsible for deduplication.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep only regular trading hours: 09:30 – 15:59 ET
    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(minutes >= 570) & (minutes < 960)].reset_index(drop=True)  # 570=9:30, 960=16:00

    if df.empty:
        return []

    df = _add_vwap(df)
    rows_out = []

    # Determine which dates to generate predictions for (from_date only gates OUTPUT, not input)
    from_date_ts = pd.Timestamp(from_date).date() if from_date is not None else None

    for date, day_df in df.groupby(df["timestamp"].dt.date):
        if from_date_ts is not None and date < from_date_ts:
            continue
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)

        # Generate at ALL standard 10-min slots (mirrors live watcher behavior).
        # Slots: 09:40, 09:50, 10:00, … 15:50 ET (first slot after MIN_HISTORY bars)
        session_start = pd.Timestamp(date).replace(hour=9, minute=30)
        session_end   = pd.Timestamp(date).replace(hour=15, minute=50)
        pred_timestamps = pd.date_range(
            start=session_start + pd.Timedelta(minutes=10),
            end=session_end,
            freq="10min",
        ).tolist()

        for pred_ts in pred_timestamps:
            # ── STRICT: past data only (bars BEFORE this timestamp)
            # Use full multi-day history (mirrors how the live predictor works)
            past_df = df[df["timestamp"] < pred_ts].copy()
            if len(past_df) < MIN_HISTORY:
                continue

            # ── Generate prediction (predictor only sees past_df)
            pred = make_prediction(10, past_df)
            if pred is None:
                continue

            # Override the auto-generated "now" timestamp with the real historical one
            pred["time"] = pred_ts

            time_str = pred_ts.strftime("%Y-%m-%d %H:%M:%S")
            # regime/volatility computed from intraday context only
            intraday_past = day_df[day_df["timestamp"] < pred_ts]
            regime    = _regime(intraday_past if len(intraday_past) >= 60 else past_df)
            vol_state = _volatility(intraday_past if len(intraday_past) >= 30 else past_df)
            session   = classify_session(time_str)

            # ── Grade: check bars within the 10-min prediction window
            target_ts = pred_ts + pd.Timedelta(minutes=10)
            future = day_df[(day_df["timestamp"] >= pred_ts) & (day_df["timestamp"] < target_ts)]

            if future.empty:
                # Too close to end of session — write unchecked
                rows_out.append({
                    "time":             time_str,
                    "symbol":           symbol.upper(),
                    "timeframe":        10,
                    "direction":        pred["direction"],
                    "confidence":       round(pred["confidence"], 3),
                    "high":             pred["high"],
                    "low":              pred["low"],
                    "regime":           regime,
                    "volatility":       vol_state,
                    "session":          session,
                    "actual":           "",
                    "correct":          0,
                    "checked":          False,
                    "high_hit":         "",
                    "low_hit":          "",
                    "price_at_check":   "",
                    "close_at_check":   "",
                    "confidence_band":  _conf_band(pred["confidence"]),
                })
                continue

            # Aggregate high/low across the full prediction window
            window_high = float(future["high"].max())
            window_low = float(future["low"].min())
            window_close = float(future.iloc[-1]["close"])

            high_hit = int(window_high >= float(pred["high"]))
            low_hit  = int(window_low  <= float(pred["low"]))

            if   high_hit and not low_hit: result = "bullish"
            elif low_hit  and not high_hit: result = "bearish"
            elif high_hit and low_hit:      result = "both"
            else:                           result = "range"

            correct = int(result == pred["direction"])

            rows_out.append({
                "time":             time_str,
                "symbol":           symbol.upper(),
                "timeframe":        10,
                "direction":        pred["direction"],
                "confidence":       round(pred["confidence"], 3),
                "high":             pred["high"],
                "low":              pred["low"],
                "regime":           regime,
                "volatility":       vol_state,
                "session":          session,
                "actual":           result,
                "correct":          correct,
                "checked":          True,
                "high_hit":         high_hit,
                "low_hit":          low_hit,
                "price_at_check":   round(window_close, 2),
                "close_at_check":   round(window_close, 2),
                "confidence_band":  _conf_band(pred["confidence"]),
            })

    return rows_out


# ──────────────────────────────────────────── entry point

def main():
    parser = argparse.ArgumentParser(description="Walk-forward prediction backfill")
    parser.add_argument("--symbols",   nargs="+", default=ALL_SYMBOLS,
                        help="Symbols to process (default: all 8)")
    parser.add_argument("--from-date", default=None,
                        help="Only process bars on or after YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate and overwrite all existing predictions")
    parser.add_argument("--no-learn", action="store_true",
                        help="Disable iterative learning (process symbols independently)")
    args = parser.parse_args()

    existing_keys = set() if args.overwrite else _load_existing_keys(PRED_FILE)
    print(f"Existing predictions in file: {len(existing_keys)}")

    all_new: list[dict] = []

    if args.no_learn:
        # Legacy mode: process each symbol independently (no iterative learning)
        for sym in args.symbols:
            csv_path = os.path.join(DATA_DIR, f"{sym.lower()}_1m.csv")
            if not os.path.exists(csv_path):
                print(f"[{sym}] No CSV found at {csv_path} — skipping")
                continue

            print(f"[{sym}] Processing...", end=" ", flush=True)
            rows = backfill_symbol(sym, csv_path, from_date=args.from_date)

            if not args.overwrite:
                rows = [r for r in rows if (r["time"], r["symbol"]) not in existing_keys]

            print(f"{len(rows)} new predictions")
            all_new.extend(rows)
    else:
        # Chronological mode: all symbols interleaved, learning every 500 predictions
        print(f"\nIterative learning mode: weights update every {LEARN_EVERY} graded predictions")
        all_new = backfill_all_symbols_chronological(
            symbols=args.symbols,
            from_date=args.from_date,
            existing_keys=existing_keys if not args.overwrite else None,
        )

    if not all_new:
        print("Nothing new to write.")
        return

    # If overwrite mode, rewrite the file from scratch; otherwise append
    if args.overwrite:
        with open(PRED_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PRED_HEADERS)
            writer.writeheader()
            writer.writerows(sorted(all_new, key=lambda r: (r["time"], r["symbol"])))
    else:
        write_header = (not os.path.exists(PRED_FILE) or os.path.getsize(PRED_FILE) == 0)
        with open(PRED_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PRED_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerows(all_new)

    print(f"\nWrote {len(all_new)} predictions to {PRED_FILE}")

    # Update edge_stats.json from the full file
    try:
        from analytics.grader import update_edge_stats
        full_df = pd.read_csv(PRED_FILE)
        graded = full_df[full_df["checked"] == True]
        if not graded.empty:
            update_edge_stats(graded)
            print(f"Edge stats updated ({len(graded)} graded rows).")
    except Exception as e:
        print(f"Edge stats update failed: {e}")

    # Refresh learned predictor bias weights
    try:
        from analytics.predictor_optimizer import update_predictor_weights
        update_predictor_weights()
        print("Predictor weights updated.")
    except Exception as e:
        print(f"Predictor weights update failed: {e}")

    # ── Accuracy summary
    graded_new = [r for r in all_new if r.get("checked")]
    if not graded_new:
        print("No graded predictions in this batch — all near end-of-session.")
        return

    total   = len(graded_new)
    correct = sum(r["correct"] for r in graded_new)
    print(f"\n── Backfill accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    # Per-symbol
    by_sym: dict[str, dict] = {}
    for r in graded_new:
        s = r["symbol"]
        if s not in by_sym:
            by_sym[s] = {"c": 0, "t": 0}
        by_sym[s]["t"] += 1
        by_sym[s]["c"] += r["correct"]
    for s, v in sorted(by_sym.items()):
        print(f"   {s:6s} {v['c']}/{v['t']}  {v['c']/v['t']*100:.1f}%")

    # Per-regime
    by_regime: dict[str, dict] = {}
    for r in graded_new:
        rg = r["regime"]
        if rg not in by_regime:
            by_regime[rg] = {"c": 0, "t": 0}
        by_regime[rg]["t"] += 1
        by_regime[rg]["c"] += r["correct"]
    print("\n── By regime:")
    for rg, v in sorted(by_regime.items()):
        print(f"   {rg:12s} {v['c']}/{v['t']}  {v['c']/v['t']*100:.1f}%")

    # Per-session
    by_sess: dict[str, dict] = {}
    for r in graded_new:
        ss = r["session"]
        if ss not in by_sess:
            by_sess[ss] = {"c": 0, "t": 0}
        by_sess[ss]["t"] += 1
        by_sess[ss]["c"] += r["correct"]
    print("\n── By session:")
    for ss, v in sorted(by_sess.items()):
        print(f"   {ss:10s} {v['c']}/{v['t']}  {v['c']/v['t']*100:.1f}%")

    # Per-confidence band
    by_band: dict[str, dict] = {}
    for r in graded_new:
        b = r["confidence_band"]
        if b not in by_band:
            by_band[b] = {"c": 0, "t": 0}
        by_band[b]["t"] += 1
        by_band[b]["c"] += r["correct"]
    print("\n── By confidence band:")
    for b, v in sorted(by_band.items()):
        print(f"   {b:8s} {v['c']}/{v['t']}  {v['c']/v['t']*100:.1f}%")


if __name__ == "__main__":
    main()
