

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay,
)
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLS = [
    "vibration_mm_s", "temperature_c", "current_a",
    "vibration_mm_s_roll_mean", "vibration_mm_s_roll_std",
    "vibration_mm_s_roll_min", "vibration_mm_s_roll_max", "vibration_mm_s_trend",
    "temperature_c_roll_mean", "temperature_c_roll_std",
    "temperature_c_roll_min", "temperature_c_roll_max", "temperature_c_trend",
    "current_a_roll_mean", "current_a_roll_std",
    "current_a_roll_min", "current_a_roll_max", "current_a_trend",
    "health_risk_score",
]
TARGET_COL = "failure_within_window"


def split_by_machine(df, test_size=0.25, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["machine_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def main():
    df = pd.read_csv("data/features_scored.csv")
    train_df, test_df = split_by_machine(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "n_train_machines": train_df["machine_id"].nunique(),
        "n_test_machines": test_df["machine_id"].nunique(),
        "positive_rate_test": float(y_test.mean()),
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Plots ---
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Failure", "Failure Soon"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Failure", "Failure Soon"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve — Failure-within-window prediction")
    fig.tight_layout()
    fig.savefig("results/roc_curve.png", dpi=150)
    plt.close(fig)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(6, 5))
    importances.plot(kind="barh", ax=ax, color="#3b6ea5")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig("results/feature_importance.png", dpi=150)
    plt.close(fig)

    # Example: risk score trajectory for one failing and one healthy machine
    fig, ax = plt.subplots(figsize=(7, 4.5))
    failing_id = df.loc[df["failed"] == 1, "machine_id"].iloc[0]
    healthy_id = df.loc[df["failed"] == 0, "machine_id"].iloc[0]
    for mid, label, color in [(failing_id, "Failing machine", "#d62728"),
                               (healthy_id, "Healthy machine", "#2ca02c")]:
        sub = df[df["machine_id"] == mid]
        ax.plot(sub["cycle"], sub["health_risk_score"], label=label, color=color)
    ax.set_xlabel("Cycle"); ax.set_ylabel("Health Risk Score (0-100)")
    ax.set_title("Health Risk Score Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/risk_score_trajectory.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("Saved metrics.json and plots to results/")


if __name__ == "__main__":
    main()
