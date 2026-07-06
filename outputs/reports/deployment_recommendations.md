# Deployment Recommendations
## Based on Completed Research — No Additional Training Required

---

## App A — AQI Estimation Service

**Purpose:** Real-time AQI reconstruction from concurrent sensor readings.
**Scientific Context:** AQI Estimation (not forecasting). Same-timestamp pollutants
  are available from a running sensor network. Model reconstructs the CPCB AQI formula
  with R²=0.99, providing a cross-check against sensor dropouts or missing AQI computation.

### Recommended Model: Gradient Boosting Regressor

| Criterion | GradBoost | RandomForest | LSTM |
|-----------|-----------|-------------|------|
| Avg R² | 0.9906 | 0.9874 | 0.6411 |
| Avg MAE | 2.94 | 1.64 | 27.04 |
| Avg RMSE | 5.77 | 6.05 | 39.36 |
| Inference | ~0.017s | ~0.10s | ~2.30s |
| Memory | ~50 MB | ~500 MB | ~20 MB |
| Deployment | ✅ Simple pickle | ⚠ Large | ✅ SavedModel |

**Selection rationale:** GradBoost has the highest R² (0.9906) with the lowest RMSE (5.77).
RandomForest has marginally lower MAE (1.64 vs 2.94) but requires 10× more memory.
LSTM is architecturally inappropriate for same-timestamp tabular regression.

### Technical Specifications

```
Model:           sklearn GradientBoostingRegressor
Serialization:   joblib / pickle (~2 MB per city model)
Input:           18 features (PM2.5, PM10, NO2, SO2, CO, Ozone, NH3,
                              AT, RH, WS, WD, SR, BP,
                              hour_sin, hour_cos, month_sin, month_cos, season)
Output:          AQI (float32), AQI Category (str)
Latency:         < 50ms per prediction (CPU)
Update freq:     Hourly (retrain monthly on rolling 12-month window)
Memory:          ~50 MB for all 18 city models combined
```

### API Design (FastAPI / Flask)

```python
POST /api/v1/estimate_aqi
Request Body: {
    'city': 'Delhi_NCR',
    'timestamp': '2024-01-15T14:00:00',
    'pm25': 145.2,
    'pm10': 210.5,
    'no2': 48.3,
    'so2': 12.1,
    'co': 1.8,
    'ozone': 22.5,
    'nh3': 15.0,
    'at': 18.5,
    'rh': 72.0,
    'ws': 1.2,
    'wd': 180.0
}
Response: {
    'aqi': 187.3,
    'category': 'Very Poor',
    'health_advice': 'Avoid outdoor activities. Use N95 mask if outdoors.',
    'dominant_pollutant': 'PM2.5',
    'model': 'GradientBoosting_v1',
    'city': 'Delhi_NCR',
    'confidence': 'HIGH'
}
```

### AQI Category Thresholds (CPCB Standard)

| AQI Range | Category | Health Advisory |
|-----------|----------|-----------------|
| 0–50 | Good | No restrictions needed |
| 51–100 | Satisfactory | Sensitive groups limit prolonged outdoor exertion |
| 101–200 | Moderate | Sensitive groups limit outdoor activities |
| 201–300 | Poor | Everyone limit prolonged outdoor exertion |
| 301–400 | Very Poor | Avoid outdoor activities; use air purifiers |
| 401–500 | Severe | Stay indoors; medical-grade masks only |

---

## App B — AQI Forecasting Service

**Purpose:** Short-horizon AQI forecasting (1h, 6h, 24h ahead) without concurrent sensors.
**Scientific Context:** True forecasting using only historical lags, rolling stats, met, time.
  Suitable for early warning systems and pollution event advisories.

### Recommended Model: Gradient Boosting Regressor

| Criterion | GradBoost | RandomForest | XGBoost | LSTM |
|-----------|-----------|-------------|---------|------|
| Avg R² (all h) | 0.4997 | 0.4914 | 0.4902 | 0.2768 |
| t+1h R² | 0.656 | 0.671 | 0.659 | 0.429 |
| t+6h R² | 0.488 | 0.457 | 0.468 | 0.232 |
| t+24h R² | 0.356 | 0.346 | 0.344 | 0.170 |
| Deployment | ✅ Simple | ⚠ Large | ✅ Simple | ⚠ Complex |

**Selection rationale:** GradBoost leads on all three horizons and overall avg R².
For the production forecasting service, **separate models per horizon** are recommended
(GBR_1h, GBR_6h, GBR_24h) since horizon-specific tuning outperforms a single multi-output model.

### Technical Specifications

```
Model:           3 × sklearn GradientBoostingRegressor (one per horizon)
Serialization:   joblib (~2 MB × 3 × 18 = ~108 MB total)
Input:           lag features + rolling stats + met + time (no same-t pollutants)
                 Approximately 45–65 features per city (varies by sensor availability)
Output:          AQI forecast, AQI category, confidence band
Latency:         < 100ms per city × horizon on CPU
Update freq:     Daily retraining on rolling 90-day window
Memory:          ~200 MB for all city-horizon models
Min data needed: 48h of historical sensor data to populate lag features
```

### API Design

```python
POST /api/v1/forecast_aqi
Request Body: {
    'city': 'Delhi_NCR',
    'horizon_hours': [1, 6, 24],
    'historical_data': [
        {'timestamp': '2024-01-15T12:00:00', 'pm25': 120.0, ...},
        {'timestamp': '2024-01-15T13:00:00', 'pm25': 135.0, ...},
        ... (at least 48 hourly records)
    ]
}
Response: {
    'forecasts': [
        {'horizon': '1h',  'aqi': 172.4, 'category': 'Very Poor'},
        {'horizon': '6h',  'aqi': 155.1, 'category': 'Poor'},
        {'horizon': '24h', 'aqi': 143.8, 'category': 'Poor'},
    ],
    'dominant_pollutant': 'PM2.5',
    'trend': 'IMPROVING',
    'model_version': 'GBR_Forecaster_v1',
    'warning': 'AQI forecast confidence decreases at 24h horizon (R²=0.36)'
}
```

---

## Deployment Architecture — National Multi-City System

```
┌─────────────────────────────────────────────────────────┐
│             NATIONAL AQI PREDICTION SYSTEM              │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  Sensor API  │───▶│   Feature Engineering        │  │
│  │  (CPCB feed) │    │   (lags, rolling, time)      │  │
│  └──────────────┘    └──────────────┬───────────────┘  │
│                                     │                   │
│                          ┌──────────▼──────────┐        │
│                          │   City Router        │        │
│                          │   (18 city models)   │        │
│                          └──────┬───────┬───────┘        │
│                                 │       │                │
│                    ┌────────────▼┐  ┌───▼────────────┐  │
│                    │ Estimation  │  │  Forecasting   │  │
│                    │ (GBR — R²≈  │  │  (GBR 1h/6h/  │  │
│                    │   0.99)     │  │   24h — R²≈   │  │
│                    └─────────────┘  │   0.35–0.66)  │  │
│                                     └───────────────┘  │
│                                                         │
│           ┌───────────────────────────────┐             │
│           │  Response Layer               │             │
│           │  AQI value + Category +       │             │
│           │  Health advice + Trend        │             │
│           └───────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

## Caveats and Limitations

1. **Track B (Forecasting) R² degrades sharply**: t+1h R²=0.66 is usable for early warning,
   but t+24h R²=0.36 is only indicative. Users must be informed of increasing uncertainty.

2. **Missing sensor data**: If PM2.5 sensor is offline, Track A estimation is invalid.
   Fall back to Track B forecasting using available lag features.

3. **Arid/coastal cities need city-specific models**: Jodhpur and Vishakhapatnam show
   systematically different PM10/PM2.5 ratios. National model degrades for these cities.

4. **Model drift**: Seasonal retraining is mandatory. Delhi winter inversion months
   (Nov–Jan) exhibit different AQI distributions from summer months.

5. **No satellite data integration**: Current models use only ground station sensors.
   Future versions should integrate MODIS/Sentinel-5P AOD data for improved t+24h accuracy.