# EPL Match Outcome Predictor

Predicting Premier League match results (Home Win / Draw / Away Win) using 25 seasons
of historical data (2000-2025), with a focus on avoiding data leakage in time-series
feature engineering.

## Why this project

I wanted a project that combined a genuine interest of mine (football) with a proper
end-to-end machine learning workflow.

## Approach

- Engineered pre-match features only (rolling 5-game form, goal averages, head-to-head
  win rate) — deliberately excluded in-match stats like shots and corners, since these
  aren't known before kickoff and would leak the answer
- Chronological train/test split (not random) to simulate real-world prediction
- Compared Logistic Regression vs Random Forest against a naive baseline

## Results

| Model | Accuracy |
|---|---|
| Baseline (always predict Home Win) | 43.5% |
| Logistic Regression | 51.0% |
| Random Forest | 49.8% |

## Key finding

The model never predicted a single draw across the test set, despite beating baseline
accuracy overall. This revealed a real limitation in using raw accuracy as a metric on
an imbalanced classification problem.

## What I'd explore next

- Class weighting to force the model to take draws seriously
- Predicting goal difference instead of a hard win/draw/loss category
- Additional features: table position, days of rest, squad changes

## Tech used

Python, pandas, scikit-learn, seaborn/matplotlib, Google Colab
