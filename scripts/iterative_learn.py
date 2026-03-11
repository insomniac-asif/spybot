#!/usr/bin/env python3
"""
scripts/iterative_learn.py

Lightweight iterative learning from existing graded predictions.
Walks predictions chronologically in chunks, updating predictor weights
progressively so you can see how accuracy evolves over time.

Does NOT regenerate predictions (no bar data loaded = low RAM usage).
"""

import os
import sys
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.predictor_optimizer import compute_weights, WEIGHTS_FILE, PRED_FILE

CHUNK_SIZE = 5000  # Update weights every N graded predictions


def main():
    if not os.path.exists(PRED_FILE):
        print("No predictions.csv found")
        return

    df = pd.read_csv(PRED_FILE)
    df = df[df["checked"] == True].copy()
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    df = df.sort_values("time").reset_index(drop=True)

    print(f"Total graded predictions: {len(df):,}")
    print(f"Date range: {df['time'].min()} → {df['time'].max()}")
    print(f"Learning every {CHUNK_SIZE} predictions\n")

    n_updates = 0
    best_wr = 0.0
    wr_history = []

    for end_idx in range(CHUNK_SIZE, len(df) + 1, CHUNK_SIZE):
        chunk = df.iloc[:end_idx]
        weights = compute_weights(chunk)
        if not weights:
            continue

        n_updates += 1
        wr = weights["meta"]["overall_wr"]
        samples = weights["meta"]["samples"]
        wr_history.append(wr)

        if wr > best_wr:
            best_wr = wr

        # Show progress every 5 updates
        if n_updates % 5 == 0 or end_idx >= len(df) - CHUNK_SIZE:
            date_at = chunk["time"].iloc[-1].strftime("%Y-%m-%d")
            print(f"  [{n_updates:3d}] {samples:,} samples → WR={wr:.1%}  (date: {date_at})")

    # Final update with ALL data
    weights = compute_weights(df)
    if weights:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)

        wr = weights["meta"]["overall_wr"]
        print(f"\n{'='*50}")
        print(f"Final: {len(df):,} predictions, WR={wr:.1%}")
        print(f"Best WR seen: {best_wr:.1%}")
        print(f"Weights saved → {WEIGHTS_FILE}")

        # Show weight summary
        for section in ["regime", "session", "volatility"]:
            data = weights.get(section, {})
            if data:
                print(f"\n  {section.upper()} biases:")
                for ctx, biases in sorted(data.items()):
                    parts = "  ".join(f"{d}:{b:+.3f}" for d, b in sorted(biases.items()))
                    print(f"    {ctx:<20s} {parts}")


if __name__ == "__main__":
    main()
