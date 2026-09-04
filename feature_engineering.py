

import numpy as np
import pandas as pd

SENSORS = ["vibration_mm_s", "temperature_c", "current_a"]
ROLL_WINDOW = 10
FAILURE_WINDOW = 15  # predict failure within this many cycles ahead


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "cycle"]).copy()
    grouped = df.groupby("machine_id")

    for sensor in SENSORS:
        roll = grouped[sensor].rolling(ROLL_WINDOW, min_periods=3)
        df[f"{sensor}_roll_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{sensor}_roll_std"] = roll.std().reset_index(level=0, drop=True)
        df[f"{sensor}_roll_min"] = roll.min().reset_index(level=0, drop=True)
        df[f"{sensor}_roll_max"] = roll.max().reset_index(level=0, drop=True)

        # Short-term trend: difference vs value N cycles ago
        df[f"{sensor}_trend"] = grouped[sensor].diff(ROLL_WINDOW)

    df[[c for c in df.columns if "_roll_" in c or "_trend" in c]] = (
        df[[c for c in df.columns if "_roll_" in c or "_trend" in c]]
        .bfill()
        .fillna(0)
    )
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_cycles = df.groupby("machine_id")["cycle"].transform("max")
    cycles_to_end = n_cycles - df["cycle"]

    # Positive label only for machines that actually fail, and only within
    # FAILURE_WINDOW cycles of that failure
    df["failure_within_window"] = (
        (df["failed"] == 1) & (cycles_to_end <= FAILURE_WINDOW)
    ).astype(int)
    return df


def build_feature_table(input_path="data/sensor_data.csv",
                         output_path="data/features.csv") -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    feat = add_rolling_features(raw)
    feat = add_labels(feat)
    feat.to_csv(output_path, index=False)
    return feat


if __name__ == "__main__":
    table = build_feature_table()
    print(f"Feature table shape: {table.shape}")
    print(f"Positive (failure-within-window) rate: "
          f"{table['failure_within_window'].mean():.3%}")
    print("Saved to data/features.csv")
