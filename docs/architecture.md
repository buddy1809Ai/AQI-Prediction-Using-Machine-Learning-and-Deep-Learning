# System Architecture

## Pipeline Diagram

```mermaid
flowchart TD
    A[Raw CPCB CSVs 18.7M records] --> B[Data Cleaning hourly aggregate]
    B --> C[EDA 9 figures]
    B --> D[Feature Engineering 115 features]
    D --> E[Preprocessing MinMaxScaler 70/15/15]
    E --> F[Leakage Audit 3 RF experiments]
    F --> G{Two Tracks}
    G --> H[TRACK A Estimation]
    G --> I[TRACK B Forecasting]
    H --> J[GBR R2=0.9906]
    I --> K[GBR 1h R2=0.6555]
    J & K --> L[Outputs CSVs Figures Reports]
    L --> M[Streamlit App 10 pages]
```

## Data Flow
```
Raw CSV → Cleaned Parquet → Engineered Parquet → Recovered Parquet
         (hourly, AQI)       (115 features)        (imputed, AQI-free)
                                                         |
                                MinMaxScaler (train-fit only)
                                         |
                          Track A ───────┴─────── Track B
                       (same-t feats)         (lag feats only)
                             |                       |
                     model_metrics.json ← both tracks
                             |
                       Streamlit App
```
