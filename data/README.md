# Data Directory

## Sample Dataset (`data/sample_dataset/`)

Contains one CSV per city with ~200 rows, spanning both high-AQI and low-AQI periods.
Sufficient for running the Streamlit app and tests.

### Derivation Method
From the full cleaned parquet (hourly, ~28K–110K rows per city):
1. Top-50 highest AQI rows (pollution peak coverage)
2. Bottom-50 lowest AQI rows (clean-air baseline)
3. 100 random rows

### Column schema
See `schema.md` for full column dictionary, units, and valid ranges.

---

## Full Dataset (not included)

**Source:** Central Pollution Control Board (CPCB) — [data.gov.in](https://data.gov.in)
**Format:** CSV, 15-minute frequency per monitoring station
**Cities:** 18 Indian cities, ~2010–2025
**Size:** ~18.7 million 15-minute records across 543 files

### How to Reproduce

1. Download CPCB hourly air quality data for each city from data.gov.in
2. Place raw CSVs in `data/raw/<CityName>/`
3. Run the pipeline in order:
   ```bash
   python src/01_data_audit.py
   python src/03_feature_engineering.py
   python src/04_preprocessing.py
   python src/05_leakage_audit.py
   python src/06_city_recovery.py
   python src/07_track_a_models.py
   python src/08_track_b_models.py
   ```
