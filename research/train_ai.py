import os
import pandas as pd
from typing import cast
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import pytz
import time

from core.paths import DATA_DIR

FEATURE_FILE = os.path.join(DATA_DIR, "trade_features.csv")
CANDLE_FILE = os.path.join(DATA_DIR, "qqq_1m.csv")

DIR_MODEL_FILE = os.path.join(DATA_DIR, "direction_model.pkl")
EDGE_MODEL_FILE = os.path.join(DATA_DIR, "edge_model.pkl")
DATA_MAX_AGE_MINUTES = int(os.getenv("ML_TRAIN_DATA_MAX_AGE_MINUTES", "1440"))


def _is_file_recent(path: str, max_age_minutes: int) -> bool:
    try:
        if not os.path.exists(path):
            return False
        age_sec = time.time() - os.path.getmtime(path)
        return age_sec <= (max_age_minutes * 60)
    except Exception:
        return False


def _is_market_data_fresh(max_age_minutes: int) -> bool:
    if not os.path.exists(CANDLE_FILE):
        return False
    try:
        df = pd.read_csv(CANDLE_FILE)
        if df.empty or "timestamp" not in df.columns:
            return False
        last_ts = pd.to_datetime(df["timestamp"].iloc[-1], errors="coerce")
        if pd.isna(last_ts):
            return False
        if last_ts.tzinfo is None:
            last_ts = pytz.timezone("US/Eastern").localize(last_ts)
        else:
            last_ts = last_ts.tz_convert("US/Eastern")
        now = datetime.now(pytz.timezone("US/Eastern"))
        return (now - last_ts) <= timedelta(minutes=max_age_minutes)
    except Exception:
        return False


# =========================================================
# 1️⃣ Direction Model (Market Bias Model)
# =========================================================

def train_direction_model():

    if not os.path.exists(CANDLE_FILE):
        print("Market data file not found.")
        return
    if not _is_market_data_fresh(DATA_MAX_AGE_MINUTES):
        print("Market data is stale. Skipping direction model retrain.")
        return

    try:
        df = pd.read_csv(CANDLE_FILE)
    except Exception:
        print("Market data file unreadable. Skipping direction model retrain.")
        return

    # pandas_ta vwap requires a sorted DatetimeIndex
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    close = cast(pd.Series, df["close"])
    high = cast(pd.Series, df["high"])
    low = cast(pd.Series, df["low"])
    volume = cast(pd.Series, df["volume"])
    df["ema9"] = ta.ema(close, length=9)
    df["ema20"] = ta.ema(close, length=20)
    df["rsi"] = ta.rsi(close, length=14)
    df["vwap"] = ta.vwap(high, low, close, volume)

    df["future_close"] = df["close"].shift(-30)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df = df.dropna()

    features = df[["ema9", "ema20", "rsi", "vwap", "volume"]]
    labels = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Direction Model Accuracy:", round(accuracy, 4))

    joblib.dump(model, DIR_MODEL_FILE)
    print("Direction model saved.")


# =========================================================
# 2️⃣ Edge Model (Trade Quality Filter)
# =========================================================

def train_edge_model():

    if not os.path.exists(FEATURE_FILE):
        print("No trade feature file found.")
        return
    if not _is_file_recent(FEATURE_FILE, DATA_MAX_AGE_MINUTES):
        print("Trade feature data is stale. Skipping edge model retrain.")
        return

    try:
        df = pd.read_csv(FEATURE_FILE)
    except Exception:
        print("Trade feature file unreadable. Skipping edge model retrain.")
        return

    if len(df) < 50:
        print("Not enough trade samples to train.")
        return

    df = df.dropna()

    # -----------------------------------
    # Add Expectancy Intelligence Columns
    # -----------------------------------

    if "setup_raw_avg_R" not in df.columns:
        df["setup_raw_avg_R"] = 0

    if "regime_raw_avg_R" not in df.columns:
        df["regime_raw_avg_R"] = 0

    feature_cols = [
        "regime_encoded",
        "volatility_encoded",
        "conviction_score",
        "impulse",
        "follow_through",
        "setup_encoded",
        "session_encoded",
        "confidence",
        "style_encoded",
        "setup_raw_avg_R",
        "regime_raw_avg_R"
    ]

    X = df[feature_cols]
    y = df["won"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Edge Model Accuracy:", round(accuracy, 4))

    joblib.dump(model, EDGE_MODEL_FILE)
    print("Edge model saved.")

# =========================================================
# Run Both
# =========================================================

if __name__ == "__main__":
    train_direction_model()
    train_edge_model()
