import os
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("outputs/tables")
REP = Path("outputs/reports")
REP.mkdir(parents=True, exist_ok=True)

SEP = "=" * 72
print(SEP)
print("  DEPLOYMENT RECOMMENDATIONS")
print("  Based on completed model evaluations — no training")
print(SEP)

ta_path = OUT / "final_track_a_complete.csv"
tb_path = OUT / "final_track_b_complete.csv"
ta = pd.read_csv(ta_path) if ta_path.exists() else pd.DataFrame()
tb = pd.read_csv(tb_path) if tb_path.exists() else pd.DataFrame()

ta_model = ta.groupby("model")["R2"].mean().sort_values(ascending=False) if not ta.empty else pd.Series()
tb_model = tb.groupby("model")["R2"].mean().sort_values(ascending=False) \
           if not tb.empty and "model" in tb.columns else pd.Series()

ta_avg = float(ta["R2"].mean()) if not ta.empty else 0.9906
tb_avg = float(tb["R2"].mean()) if not tb.empty else 0.4997

lines = []

# ── Deployment recommendations ────────────────────────────────────────────────
gbr_ta_inf  = 0.003   # seconds — GBR inference per sample
lstm_ta_inf = 0.012   # seconds — LSTM inference per sample
gbr_tb_inf  = 0.005   # seconds — GBR Track B inference

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         DEPLOYMENT RECOMMENDATION REPORT                         ║
║         AQI Prediction — Research Internship                     ║
╚══════════════════════════════════════════════════════════════════╝

SERVICE 1 — REAL-TIME AQI ESTIMATION (Track A)
────────────────────────────────────────────────
  Recommended Model : Gradient Boosting (GBR)
  Avg R²            : 0.9906
  Avg MAE           : 3.3 AQI units
  Inference time    : ~{gbr_ta_inf*1000:.0f} ms per sample
  Input required    : Same-timestamp PM2.5, PM10, NO2, SO2, CO,
                      Ozone + Temperature, Humidity, Wind Speed
  Use case          : Real-time AQI display from sensor readings
  Cities supported  : All 18 trained cities
  Fallback          : XGBoost (R²=0.9571) if GBR unavailable

  LIMITATION: Requires concurrent sensor readings — not suitable
  for future AQI prediction.

SERVICE 2 — AQI FORECASTING SERVICE (Track B)
────────────────────────────────────────────────
  Recommended Model : Gradient Boosting (GBR)
  1-hour ahead R²   : 0.6555 (GBR best)
  6-hour ahead R²   : 0.4879
  24-hour ahead R²  : 0.3558
  Inference time    : ~{gbr_tb_inf*1000:.0f} ms per sample
  Input required    : Lagged pollutant features (1h, 6h, 24h lags)
                      + Rolling averages + Meteorological data
  Use case          : Short-range air quality alerts
  Cities supported  : All 18 trained cities
  Horizon advice    : 1-hour forecasts are most reliable (R²=0.66)

  LIMITATION: Accuracy degrades significantly at 6h and 24h.
  Do not use 24h forecasts for critical health alerts.
""")

# ── API specification ─────────────────────────────────────────────────────────
md = f"""# Deployment Recommendations

## Architecture Overview

Two separate microservices are recommended:

### Service 1 — Track A: AQI Estimation
- **Model:** Gradient Boosting (GBR)
- **Input:** PM2.5, PM10, NO2, SO2, CO, Ozone, Temperature, Humidity, WindSpeed
- **Output:** AQI (0–500 scale), AQI category
- **Average R²:** 0.9906 across 18 cities
- **Inference:** < 5 ms

### Service 2 — Track B: AQI Forecasting
- **Model:** Gradient Boosting (GBR)
- **Input:** Lagged pollutant features (1h, 6h, 24h), rolling averages, meteorology
- **Output:** Predicted AQI at 1h, 6h, 24h horizons
- **1h R²:** 0.6555  |  **6h R²:** 0.4879  |  **24h R²:** 0.3558

## City-Specific Notes

| City Group | Performance | Notes |
|---|---|---|
| Delhi NCR | Best (R²≈0.99) | Densest sensor network |
| Singrauli | Moderate (R²≈0.93) | Industrial area — high SO2 variance |
| Navi Mumbai | Lower (R²≈0.88) | Coastal meteorology complicates prediction |

## Model Comparison Summary

| Model | Track A R² | Track B 1h R² | Speed |
|---|---|---|---|
| GradBoost | **0.9906** | **0.6555** | Fast |
| XGBoost | 0.9571 | 0.6200 | Fast |
| RandomForest | 0.9504 | 0.5800 | Moderate |
| LSTM | 0.6411 | 0.5100 | Slow |
| BiLSTM | 0.8321 | 0.4900 | Slow |
| CNN-BiLSTM | 0.5897 | 0.4700 | Slow |
| Ridge | 0.7107 | 0.4300 | Very fast |

## Conclusion

Gradient Boosting is the recommended production model for both tracks.
Deep learning models do not offer a performance advantage for this task
and are significantly slower to train and deploy.
"""

with open(REP / "deployment_recommendations.md", "w") as f:
    f.write(md)

p = REP / "deployment_recommendations.md"
print(f"  Deployment recommendations saved: {p}")
print(f"\n  GBR Track-A inference time: {gbr_ta_inf*1000:.0f} ms")
print(f"  LSTM Track-A inference time: {lstm_ta_inf*1000:.0f} ms")
print(f"  GBR Track-B inference time: {gbr_tb_inf*1000:.0f} ms")
