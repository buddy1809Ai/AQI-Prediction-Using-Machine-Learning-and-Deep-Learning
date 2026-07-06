# Model Cards

## Track A Champion: Gradient Boosting
- Algorithm: sklearn GradientBoostingRegressor (200 estimators, max_depth=5)
- Features: 32 (same-t pollutants + met + cyclical time + interactions)
- Avg R²: 0.9906 | MAE: 3.3 | RMSE: 10.5 | Inference: ~3ms
- Best city: Delhi NCR (R²=0.9993) | Worst: Pune (R²=0.9645)

## Track A Deep Learning: BiLSTM
- Architecture: Bidirectional(LSTM-64) → Dropout(0.2) → Bidirectional(LSTM-32) → Dense(1)
- Avg R²: 0.8321 | MAE: 22.3 | RMSE: 32.0
- Seq length: 24 hours | EarlyStopping patience: 7

## Track B Champion: Gradient Boosting
- Features: lag(1h–48h) + rolling(3h–48h) mean/std + met + cyclical time
- 1h R²: 0.6555 | 6h R²: 0.4879 | 24h R²: 0.3558
- Known issue: Jodhpur anomaly (negative BiLSTM R²) — data quality suspected

## Known Limitations
- Navi Mumbai excluded (insufficient hourly data after cleaning)
- 24h forecasts are indicative only
- Model files not in repo (exceed GitHub 100MB limit)
