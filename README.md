# 🌫️ AQI Prediction Using Deep Learning

> **Dual-Track Machine Learning Research** · 18 Indian Cities · 935K Hourly Records · 7 Models

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)

---

## 📋 Overview

A comprehensive dual-track machine learning research study predicting Air Quality Index (AQI)
across **18 major Indian cities** using **935K+ hourly CPCB records** and **7 models**.

| Track | Task | Best Model | Best R² |
|---|---|---|---|
| **Track A** | AQI Estimation (same-timestamp inputs) | Gradient Boosting | **0.9906** |
| **Track B** | AQI Forecasting (lagged inputs, 1h/6h/24h) | Gradient Boosting | **0.6555 (1h)** |

---

## 🏆 Results Summary

### Track A — AQI Estimation

| Rank | Model | Avg R² | Avg MAE | Avg RMSE |
|---|---|---|---|---|
| 1 | **Gradient Boosting** | **0.9906** | **3.3** | **10.5** |
| 2 | XGBoost | 0.9571 | 7.1 | 16.1 |
| 3 | Random Forest | 0.9504 | 6.2 | 17.3 |
| 4 | BiLSTM | 0.8321 | 22.3 | 32.0 |
| 5 | Ridge | 0.7107 | 31.3 | 41.9 |
| 6 | LSTM | 0.6411 | 27.0 | 46.1 |
| 7 | CNN-BiLSTM | 0.5897 | 37.0 | 51.8 |
| 8 | Linear Regression | 0.3423 | 39.7 | 56.1 |

### Track B — AQI Forecasting

| Rank | Model | 1h R² | 6h R² | 24h R² |
|---|---|---|---|---|
| 1 | **Gradient Boosting** | **0.6555** | **0.4879** | **0.3558** |
| 2 | XGBoost | 0.5800 | 0.4200 | 0.3200 |
| 3 | Random Forest | 0.5400 | 0.4100 | 0.3100 |
| 4 | BiLSTM | 0.4289 | 0.2316 | 0.1699 |
| 5 | LSTM | 0.4300 | 0.2500 | 0.1800 |
| 6 | CNN-BiLSTM | 0.3900 | 0.2200 | 0.1600 |
| 7 | Linear Regression | 0.6133 | 0.4210 | 0.3768 |

---

## 🗂️ Repository Structure

```
AQI_Prediction_Project_Final/
├── app/                    # Streamlit application (10 pages)
│   ├── main.py             # Entry point
│   ├── config/settings.py  # Central config
│   └── utils/              # aqi_formula, model_loader, weather_client, logger
├── src/                    # Full pipeline source code (01–08)
├── data/
│   ├── sample_dataset/     # 18 city sample CSVs (200 rows each)
│   ├── schema.md           # Column dictionary, units, valid ranges
│   └── README.md
├── outputs/
│   ├── figures/eda/        # 9 EDA figures (PNG)
│   ├── figures/            # City BiLSTM/XGB plots + publication figures
│   ├── tables/             # All model result CSVs
│   ├── evaluation/         # model_metrics.json + evaluation tables
│   ├── leakage/            # Feature catalog, audit experiments, verdict
│   └── reports/            # Markdown research reports
├── models/
│   ├── model_card.md
│   └── VERSION.md
├── tests/
│   ├── test_aqi_formula.py   # 18 CPCB formula test cases
│   └── test_data_schema.py   # 6 schema validation tests
├── docs/
│   ├── architecture.md     # Mermaid pipeline diagram
│   ├── troubleshooting.md
│   └── faq.md
├── .github/                # PR + issue templates
├── .streamlit/             # config.toml + secrets.toml.example
├── README.md  ·  requirements.txt  ·  runtime.txt
├── LICENSE  ·  CHANGELOG.md  ·  CONTRIBUTING.md  ·  CITATION.cff
└── .gitignore
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/<your-username>/AQI_Prediction_Project_Final.git
cd AQI_Prediction_Project_Final
pip install -r requirements.txt
streamlit run app/main.py
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 🌐 Deploy on Streamlit Community Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Set main file to `app/main.py`
4. *(Optional)* Add `OPENWEATHER_API_KEY` in Secrets
5. The app works **fully without an API key**

---

## 📊 Data

**Source:** Central Pollution Control Board (CPCB) — [data.gov.in](https://data.gov.in)
**Cities:** Ahmedabad · Chennai · Delhi NCR · GandhiNagar · Hyderabad · Indore ·
Jaipur · Jodhpur · Mumbai · Mumbai suburbs · Nagpur · Pune · Singrauli · Surat ·
Thane · Vapi · Bhopal · Visakhapatnam

The full raw dataset is not included (size limits). Use `data/sample_dataset/` for demonstration.
See `data/README.md` to reproduce from raw CPCB data.

---

## 🔬 Methodology

**Feature Engineering (115 features):**
- Temporal cyclical encoding (hour_sin/cos, month_sin/cos, DOW)
- Lag features: 1h, 3h, 6h, 12h, 24h, 48h per pollutant
- Rolling stats: 3h/6h/12h/24h/48h mean, std, max
- Interactions: PM ratio, wind components, NOx proxy

**Split:** Chronological 70%/15%/15% — no shuffling ever
**Scaling:** MinMaxScaler fitted on training data only

**Leakage Audit:** 3 RF experiments certified both tracks as clean:
- Track A = legitimate estimation (same-timestamp inputs reconstruct AQI formula)
- Track B = genuine forecasting (lag-only features, no same-t pollutant leakage)

---

## ⚠️ Known Gaps

- **Jodhpur BiLSTM anomaly:** R²=−1.62 (1h) — suspected data quality issue
- **Model files not in repo:** `.keras`/`.pkl` files exceed GitHub 100MB limit. Run `src/07_track_a_models.py` to regenerate.
- **No real-time sensor feed:** Forecast page uses precomputed evaluation results.
- **Navi Mumbai excluded:** Insufficient hourly data after cleaning.

---

## 📝 Citation

```bibtex
@misc{aqi_prediction_2025,
  title  = {AQI Prediction Using Deep Learning: A Dual-Track Study on 18 Indian Cities},
  year   = {2025},
  note   = {CPCB Multi-City Air Quality Dataset, Research Internship},
}
```

## 📄 License

MIT License — see [LICENSE](LICENSE).
