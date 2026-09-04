

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLS = [
    "vibration_mm_s", "temperature_c", "current_a",
    "vibration_mm_s_roll_std", "temperature_c_roll_std", "current_a_roll_std",
    "vibration_mm_s_trend", "temperature_c_trend", "current_a_trend",
]

EARLY_LIFE_CYCLES = 20  # cycles assumed healthy for every machine, used to fit "normal"


def add_health_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    healthy_reference = df[df["cycle"] <= EARLY_LIFE_CYCLES][FEATURE_COLS]

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42,
    )
    iso.fit(healthy_reference)

    # decision_function: higher = more normal, lower/negative = more anomalous
    raw_scores = iso.decision_function(df[FEATURE_COLS])

    # Rescale to an intuitive 0-100 "risk" score (higher = riskier)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    normalized = (raw_scores - min_s) / (max_s - min_s + 1e-9)
    df["health_risk_score"] = ((1 - normalized) * 100).round(1)

    return df


if __name__ == "__main__":
    feat = pd.read_csv("data/features.csv")
    scored = add_health_risk_score(feat)
    scored.to_csv("data/features_scored.csv", index=False)

    avg_risk_failing = scored.loc[scored["failed"] == 1, "health_risk_score"].mean()
    avg_risk_healthy = scored.loc[scored["failed"] == 0, "health_risk_score"].mean()
    print(f"Avg health risk score — failing machines: {avg_risk_failing:.1f}")
    print(f"Avg health risk score — healthy machines: {avg_risk_healthy:.1f}")
    print("Saved to data/features_scored.csv")
