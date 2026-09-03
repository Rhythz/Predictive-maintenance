# AI-Based Industrial Predictive Maintenance and Anomaly Detection

Predicts industrial machine failures ahead of time using vibration, temperature,
and current sensor readings, and generates a continuous machine health risk
score for real-time monitoring.

## Problem

Unplanned downtime from equipment failure is expensive. The goal is to flag a
machine as high-risk *before* it fails, using only sensor telemetry, so
maintenance can be scheduled proactively instead of reactively.

## Approach

1. **Data** (`src/generate_data.py`) — A synthetic fleet of 60 machines, each
   producing a multivariate time series (vibration, temperature, current)
   over 120–260 operating cycles. ~65% of machines follow a realistic
   nonlinear degradation curve toward failure; the rest run healthily for the
   full observation window. (Synthetic data was used in place of proprietary
   plant data, but the generator produces the same statistical shape —
   trending sensor drift, increasing noise — as real bearing/motor failure
   signatures.)

2. **Feature engineering** (`src/feature_engineering.py`) — For each sensor,
   computes rolling mean/std/min/max and short-term trend over a 10-cycle
   window, and labels each row as "failure within the next 15 cycles" or not.

3. **Anomaly detection** (`src/anomaly_detection.py`) — An Isolation Forest
   is fit only on each machine's early-life (presumed healthy) readings, then
   scores every subsequent reading for how anomalous it is. This is rescaled
   into an intuitive **0–100 health risk score**.

4. **Predictive model** (`src/train_model.py`) — A Random Forest classifier
   combines the raw sensors, rolling features, and health risk score to
   predict whether a machine will fail within the next 15 cycles. Train/test
   split is done **by machine**, not by row, so the model is evaluated on
   machines it has never seen — this avoids the leakage that comes from
   splitting time series randomly.

## Results

On a held-out set of 15 machines (2,913 readings) never seen during training:

| Metric | Score |
|---|---|
| Accuracy | 98.0% |
| Precision | 84.8% |
| Recall | 88.5% |
| F1 score | 86.6% |
| ROC-AUC | 0.995 |

- Average health risk score for failing machines was **43.8/100** vs.
  **18.4/100** for healthy machines across their full lifetimes — and the
  gap widens sharply in the final ~15% of a failing machine's life
  (see `results/risk_score_trajectory.png`).
- The most predictive signals were raw and rolling-max **vibration**,
  followed by rolling-max **temperature** — consistent with how bearing and
  motor faults typically manifest physically (see
  `results/feature_importance.png`).

Because failures are rare events (~5.4% of rows are labeled "failure
imminent"), precision/recall/F1/ROC-AUC are reported alongside accuracy —
accuracy alone would look artificially high on such an imbalanced dataset.

## Project structure

```
predictive-maintenance/
├── data/                      # generated data + engineered features (gitignored if large)
├── results/                   # metrics.json + plots (generated on run)
├── src/
│   ├── generate_data.py       # synthetic sensor data generator
│   ├── feature_engineering.py # rolling stats + labels
│   ├── anomaly_detection.py   # Isolation Forest health risk score
│   ├── train_model.py         # Random Forest classifier + evaluation + plots
│   └── main.py                # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python src/main.py
```

This regenerates the data, builds features, computes health risk scores,
trains the model, and writes metrics + plots to `results/`.

## Possible extensions

- Swap the synthetic generator for a public dataset (e.g. NASA C-MAPSS or the
  UCI AI4I 2020 Predictive Maintenance dataset) to validate against real data.
- Add remaining-useful-life (RUL) regression alongside the binary
  failure-window classification.
- Serve the trained model behind the FastAPI project in this portfolio for a
  real-time risk-scoring endpoint.
