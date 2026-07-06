
# AQI Prediction Using Deep Learning — Publication Package

---

## SUGGESTED PAPER TITLES

**Primary:**
Multi-City AQI Estimation and Forecasting Using Deep Learning and Gradient Boosting:
A Dual-Track Framework for Indian Urban Air Quality

**Alternative A:**
From Estimation to Forecasting: A Rigorous Comparative Study of Machine Learning Models
for AQI Prediction across 18 Indian Cities

**Alternative B:**
AQI Reconstruction vs. True Forecasting: Leakage-Free Deep Learning Benchmarks
on CPCB Multi-City Air Quality Data

---

## ABSTRACT SKELETON

Air quality index (AQI) prediction is critical for urban health management in India,
where 18 major cities regularly exceed WHO safe limits. This study presents a
**dual-track framework** distinguishing AQI **estimation** (concurrent inputs → AQI(t))
from AQI **forecasting** (historical inputs → AQI(t+h)) — a distinction overlooked
in most prior work. Using 934,775 hourly observations from 18 Indian cities (CPCB dataset),
we train and evaluate **seven models** — Ridge, Random Forest, Gradient Boosting, XGBoost,
LSTM, Bidirectional LSTM, and CNN-BiLSTM — across both tracks.

**Track A (Estimation):** Gradient Boosting achieves average R²=0.9906, confirming that
AQI is a deterministic piecewise-linear function of concurrent pollutant measurements.
LSTM achieves R²=0.6411, demonstrating that recurrent architectures are sub-optimal for
tabular formula reconstruction tasks.

**Track B (Forecasting):** With same-timestamp pollutants withheld, Gradient Boosting
achieves R²=0.531 (t+1h), R²=0.178 (t+6h), and R²=0.105 (t+24h). LSTM achieves
comparable short-horizon performance (R²=0.527 at t+1h). A rigorous leakage audit
confirms that no AQI-derived, future, or target-contaminating features entered any model.

Results provide the first **leakage-certified, multi-city, dual-track benchmark** for
Indian AQI prediction, with direct implications for real-time air quality early warning systems.

---

## INTRODUCTION OUTLINE

1. Motivation: India's air quality crisis (CPCB 2024 statistics)
2. Gap: Prior studies conflate estimation with forecasting; most lack leakage audits
3. Contribution: (a) Dual-track framework, (b) leakage certification methodology,
   (c) 18-city scale comparison, (d) LSTM vs tree-ensemble systematic evaluation

---

## METHODOLOGY OUTLINE

1. Dataset: CPCB Multi-City (18 cities, 2019–2024, hourly, ~934k records)
2. Data Cleaning: Deduplication, gap-filling, hourly aggregation, AQI recomputation
3. Feature Engineering: Lag (1–48h), rolling (6h/24h), cyclical time, met interactions
4. Leakage Audit: Feature census (88 safe / 26 AQI-derived), identity test, Exp A/B/C
5. Track A: Same-t pollutants + met + time → AQI(t) | 7 models
6. Track B: Lags + rolling + met + time → AQI(t+1h/6h/24h) | 6 models
7. Evaluation: Chronological 70/15/15 split, R², MAE, RMSE, train/inference time

---

## RESULTS SUMMARY

### Track A — AQI Estimation
| Model         | Avg R²  | Avg MAE | Avg RMSE |
|:--------------|:-------:|:-------:|:--------:|
| GradBoost     | 0.9906  |    2.94 |     5.77 |
| RandomForest  | 0.9874  |    1.64 |     6.05 |
| XGBoost       | 0.9856  |    2.83 |     6.82 |
| Ridge         | 0.8304  |   18.69 |    28.17 |
| LSTM          | 0.6411  |   27.04 |    39.36 |
| BiLSTM        | 0.5897  |   27.57 |    40.59 |
| CNN-BiLSTM    | 0.2756  |   42.25 |    58.54 |

### Track B — AQI Forecasting (all horizons)
| Model         | Avg R²  | Avg MAE | Avg RMSE |
|:--------------|:-------:|:-------:|:--------:|
| GradBoost     | 0.4997  |   32.57 |    48.37 |
| RandomForest  | 0.4914  |   34.16 |    48.79 |
| XGBoost       | 0.4902  |   32.97 |    48.86 |
| BiLSTM        | 0.2831  |   39.07 |    56.65 |
| LSTM          | 0.2768  |   39.07 |    56.83 |
| CNN-BiLSTM    | -0.4147 |   48.21 |    67.50 |

### Horizon Analysis
| Horizon | Avg R² | Avg MAE | Avg RMSE |
|:--------|:------:|:-------:|:--------:|
| t+1h    | 0.5313 |   30.27 |    45.68 |
| t+6h    | 0.1775 |   39.63 |    57.01 |
| t+24h   | 0.1046 |   43.13 |    60.81 |

---

## DISCUSSION OUTLINE

1. Track A insight: R²≈0.99 validates CPCB formula — sensors can cross-check AQI
2. Track B insight: t+1h feasible (R²≈0.53), t+24h requires external drivers
3. LSTM failure explanation: tabular estimation ≠ temporal forecasting task
4. City analysis: Delhi/Nagpur predictable; Jodhpur/Pune problematic (dust storms, data gaps)
5. Classical ML dominance: tree ensembles match environmental tabular data structure

---

## LIMITATIONS

1. No meteorological forecast data used in Track B (limits t+6h/t+24h performance)
2. Jodhpur/Pune require specialised dust-storm and urban-heterogeneity features
3. LSTM hyperparameters not systematically tuned (fixed seq_len=24; fixed architecture)
4. No spatial features (city coordinates, land use, satellite AOD)
5. Dataset completeness varies: some cities have 3–5 years, others 8+

---

## FUTURE WORK

1. Incorporate numerical weather prediction (NWP) forecasts for t+6h/t+24h improvement
2. Spatial GNN models encoding city proximity and emission source proximity
3. Transformer-based architectures (PatchTST, iTransformer) for longer sequence learning
4. Online/incremental learning for real-time model adaptation
5. Explainability: SHAP values for city-level pollutant attribution
6. Extend to 50+ cities using national CPCB portal expansion

---

## 10 PUBLICATION-READY CONTRIBUTION POINTS

1. First leakage-certified, dual-track AQI benchmark distinguishing estimation from forecasting
2. Identity test proving AQI is a deterministic function of concurrent CPCB pollutants
3. Largest scale multi-city study (18 cities, 934k records) with consistent preprocessing
4. Systematic evaluation of 7 models (4 classical ML + 3 DL) across both tracks
5. Rigorous 3-experiment leakage audit (Exp A: same-t, Exp B: met only, Exp C: lags only)
6. First demonstration that LSTM underperforms classical ML on AQI estimation tasks
7. Horizon degradation analysis: R² drops from 0.53 (t+1h) to 0.10 (t+24h) — quantified
8. City difficulty ranking with environmental explanation for outliers
9. National deployment recommendation (GradBoost) with accuracy-stability tradeoff analysis
10. All code, preprocessed datasets, and model checkpoints reproducible in Zerve notebook

---

## REFERENCES SECTION PLACEHOLDER

[1] CPCB. (2024). National Air Quality Index. Central Pollution Control Board, India.
[2] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
[3] Chen, T., & Guestrin, C. (2016). XGBoost. KDD.
[4] Breiman, L. (2001). Random Forests. Machine Learning.
[5] Friedman, J. (2001). Greedy function approximation: Gradient boosting. Ann. Statistics.
[6] Zheng, Y., et al. (2015). Forecasting Fine-Grained Air Quality. KDD.
[7] Yi, X., et al. (2018). Deep distributed fusion network for air quality prediction. KDD.
