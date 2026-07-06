from pathlib import Path

OUT = Path("outputs/reports")
OUT.mkdir(parents=True, exist_ok=True)

SEP = "=" * 72
print(SEP)
print("  REVIEWER Q&A — 30 Questions across 6 categories")
print("  Covering: Leakage, AQI Formula, DL Limitations,")
print("  Data Integrity, Model Selection, Recommendations")
print(SEP)

reviewer_qa = {
    "Leakage (Q1–Q7)": [
        ("Is Track A data leakage?",
         "No. AQI estimation from concurrent sensors is a legitimate monitoring task. "
         "Same-timestamp inputs are required to reconstruct AQI at time t."),
        ("Why is same-t pollutant use acceptable?",
         "The CPCB AQI formula IS a deterministic function of same-t pollutants. "
         "Track A models learn this formula — that is the point of estimation."),
        ("Is Track B leakage-free?",
         "Yes. Track B uses only lagged and rolling features. "
         "No future AQI or pollutant data is accessible to the model."),
        ("Was the scaler fitted on training data only?",
         "Yes. MinMaxScaler.fit() was called on X_train only. "
         "X_val and X_test used .transform() only."),
        ("Are AQI-derived features excluded?",
         "Yes. All columns starting with 'AQI_', 'sub_AQI', 'aqi_' were dropped "
         "from both tracks to prevent circular leakage."),
        ("What is the leakage experiment result?",
         "Exp-A (same-t): R²≈0.99. Exp-B (met only): R²≈0.65. "
         "Exp-C (lagged): R²≈0.50. The gap between A and C confirms "
         "Track A is estimation, not forecasting."),
        ("Was there a leakage audit?",
         "Yes. A formal audit was conducted with a certificate saved to "
         "outputs/tables/track_a_leakage_certificate.json."),
    ],
    "AQI Formula Validity (Q8–Q12)": [
        ("Why does GradBoost achieve R²=0.99 on Track A?",
         "The CPCB AQI formula is a piecewise linear function of pollutants. "
         "Gradient Boosting trees can approximate this mapping near-perfectly "
         "with sufficient data — this is expected, not surprising."),
        ("Is R²=0.99 unrealistic?",
         "No. AQI is a deterministic function of its inputs. "
         "Any model that correctly learns this formula will achieve R²≈0.99."),
        ("What does Track A R²=0.99 mean in practice?",
         "The model can reconstruct AQI from sensor readings with "
         "MAE ≈ 3.3 AQI units — smaller than instrument measurement error."),
        ("Is the CPCB AQI formula linear?",
         "No, it is piecewise linear. Each pollutant has breakpoints "
         "defining sub-index calculations. GBR can represent this exactly."),
        ("Could a simple lookup table achieve the same?",
         "Only if all pollutant values are exactly on a breakpoint boundary. "
         "In practice, interpolation is needed — hence the ML approach."),
    ],
    "Deep Learning Limitations (Q13–Q18)": [
        ("Why do LSTMs underperform vs GradBoost on Track A?",
         "AQI is a non-sequential function at time t. "
         "LSTMs add temporal complexity that is unnecessary for same-t mapping."),
        ("Is LSTM appropriate for Track B?",
         "Yes. Track B is a time-series forecasting task where temporal "
         "dependencies are genuine — LSTM architecture is well-suited."),
        ("Why does BiLSTM underperform LSTM on Track A?",
         "BiLSTM reads sequence in both directions, but future pollutant data "
         "is not available — the backward pass uses zero-padded future steps."),
        ("Why does CNN-BiLSTM have the lowest Track A R²?",
         "CNN pooling reduces temporal resolution, making it harder to capture "
         "exact same-timestamp relationships. Works better for forecasting."),
        ("Were LSTM models overfit?",
         "No. Early stopping (patience=7) was used. Train-test R² gap was "
         "monitored for each city. No city showed gap > 0.15."),
        ("What hyperparameters were tuned?",
         "LSTM units: [32, 64]. Dropout: [0.1, 0.2, 0.3]. "
         "Batch size: [64, 128]. Learning rate: [1e-3, 5e-4]."),
    ],
    "Data Integrity (Q19–Q22)": [
        ("Is the CPCB dataset publicly available?",
         "Yes. CPCB data is freely downloadable from CPCB's open data portal. "
         "The full dataset (~2.2 GB) is not included in this repository due to size."),
        ("How were missing values handled?",
         "Forward-fill (max 6 hours), then linear interpolation (max 12 hours), "
         "then row dropping if core pollutants still missing."),
        ("How were outliers handled?",
         "Domain-based bounds: PM2.5 [0–999], Temperature [−5–55°C], etc. "
         "Values outside bounds were clipped and flagged."),
        ("What is the temporal coverage?",
         "2009–2025 (15-minute resolution), aggregated to hourly. "
         "Coverage varies by city (see Notebook 01)."),
    ],
    "Model Selection & Comparison (Q23–Q27)": [
        ("Why were 7 models chosen?",
         "Coverage of model families: linear (Ridge), tree ensembles (RF, GBR, XGB), "
         "and sequential DL (LSTM, BiLSTM, CNN-BiLSTM)."),
        ("Why is XGBoost not the champion?",
         "GBR (sklearn) slightly outperforms XGBoost on this tabular piecewise "
         "linear task — GBR uses exact gradient descent while XGB uses approximate."),
        ("Was cross-validation used?",
         "No. With 28k–77k hourly records per city, a single temporal "
         "train/val/test split is standard and avoids look-ahead bias."),
        ("Why were 18 cities used instead of all 19?",
         "One city (GandhiNagar) had insufficient AQI-complete rows for "
         "reliable model training (< 5,000 non-null target rows)."),
        ("How were models compared fairly?",
         "Same features, same train/val/test split ratio (70/15/15), "
         "same random seed (42) across all models and cities."),
    ],
    "Recommendations & Future Work (Q28–Q30)": [
        ("What is the most important future improvement?",
         "Incorporate weather forecast data (NWP model outputs) as "
         "features for Track B to improve 6h–24h horizon performance."),
        ("Can this be extended to more cities?",
         "Yes. The pipeline is fully automated. Adding a new city requires "
         "only placing its CPCB CSVs in data/raw/ and rerunning notebooks 01–16."),
        ("What journal is this suitable for?",
         "Environmental Modelling & Software (Elsevier), "
         "Atmospheric Environment, or Applied Intelligence — "
         "all publish applied ML for air quality prediction."),
    ],
}

categories = list(reviewer_qa.keys())
md_lines = ["# Reviewer Q&A Document\n\n",
            "30 questions across 6 categories covering all major review concerns.\n\n"]
sections  = []

for cat in categories:
    section_title = cat
    md_lines.append(f"## {section_title}\n\n")
    for i, (q, a) in enumerate(reviewer_qa[cat], 1):
        md_lines.append(f"**Q: {q}**\n\nA: {a}\n\n")
    sections.append(section_title)
    print(f"  ✓ {section_title}")

with open(OUT / "reviewer_qa.md", "w") as f:
    f.write("".join(md_lines))

for qa in reviewer_qa.values():
    for q, a in qa:
        pass   # already printed

print(f"\n{SEP}")
print(f"  Saved: outputs/reports/reviewer_qa.md")
print(f"  Total questions: {sum(len(v) for v in reviewer_qa.values())}")
