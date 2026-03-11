"""
scripts/backfill_csvs.py
Backfill all symbol 1-minute CSVs from Feb 1, 2024 to present using Alpaca API.
Also populates the backtest/cache/ with parquet files for each symbol.

Usage: python -m scripts.backfill_csvs
"""
import os
import sys
import time
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_fetcher import fetch_stock_bars

SYMBOLS = ["SPY", "QQQ", "IWM", "VXX", "TSLA", "AAPL", "NVDA", "MSFT"]
START_DATE = "2024-02-01"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def backfill_symbol(symbol: str):
    """Fetch historical data and merge with existing CSV."""
    csv_path = os.path.join(DATA_DIR, f"{symbol.lower()}_1m.csv")
    end_date = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"Backfilling {symbol}: {START_DATE} -> {end_date}")
    print(f"{'='*60}")

    # Fetch from Alpaca (uses parquet cache in backtest/cache/)
    hist_df = fetch_stock_bars(symbol, START_DATE, end_date)

    if hist_df is None or hist_df.empty:
        print(f"  WARNING: No historical data returned for {symbol}")
        return

    print(f"  Fetched {len(hist_df)} bars from Alpaca/cache")

    # Load existing CSV if it exists
    existing_df = None
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
            print(f"  Existing CSV: {len(existing_df)} bars ({existing_df.index[0]} -> {existing_df.index[-1]})")
        except Exception as e:
            print(f"  Could not read existing CSV: {e}")

    # Ensure hist_df has the right columns (matching CSV format)
    # hist_df from fetch_stock_bars has: open, high, low, close, volume, vwap
    keep_cols = ["open", "high", "low", "close", "volume"]
    hist_df = hist_df[[c for c in keep_cols if c in hist_df.columns]]

    # Merge: historical data first, then existing (existing takes priority for overlapping timestamps)
    if existing_df is not None:
        existing_df = existing_df[[c for c in keep_cols if c in existing_df.columns]]
        # Combine, preferring existing data for duplicate timestamps
        combined = pd.concat([hist_df, existing_df])
        combined = combined[~combined.index.duplicated(keep="last")]  # keep existing (last)
    else:
        combined = hist_df

    combined = combined.sort_index()

    # Remove any timezone info from index (CSVs use naive ET)
    if combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)

    # Write back
    combined.index.name = "timestamp"
    combined.to_csv(csv_path)
    print(f"  Written: {len(combined)} bars ({combined.index[0]} -> {combined.index[-1]})")
    print(f"  Saved to {csv_path}")


def main():
    print("=" * 60)
    print("CSV Backfill: All symbols from Feb 1, 2024")
    print("=" * 60)

    for symbol in SYMBOLS:
        try:
            backfill_symbol(symbol)
        except Exception as e:
            print(f"  ERROR backfilling {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Backfill complete!")
    print("=" * 60)

    # Summary
    for symbol in SYMBOLS:
        csv_path = os.path.join(DATA_DIR, f"{symbol.lower()}_1m.csv")
        if os.path.exists(csv_path):
            lines = sum(1 for _ in open(csv_path)) - 1
            print(f"  {symbol}: {lines:,} bars")


if __name__ == "__main__":
    main()
