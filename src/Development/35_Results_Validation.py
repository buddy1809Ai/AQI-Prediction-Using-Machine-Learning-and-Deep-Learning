import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

OUT = Path("outputs/tables")
SEP = "=" * 72

TRACK_A_FILES = {
    "Ridge":      "track_a_ridge.csv",
    "RandomForest": "track_a_rf.csv",
    "GradBoost":  "track_a_gbr.csv",
    "XGBoost":    "track_a_xgb.csv",
    "LSTM":       "track_a_lstm.csv",
    "BiLSTM":     "track_a_bilstm.csv",
    "CNN-BiLSTM": "track_a_cnn_bilstm.csv",
}
TRACK_B_FILES = {
    "GradBoost":    "track_b_gbr.csv",
    "RandomForest": "track_b_rf.csv",
    "LSTM":         "track_b_lstm.csv",
    "CNN-BiLSTM":   "track_b_cnn_bilstm.csv",
}
EXPECTED_CITIES   = 18
EXPECTED_HORIZONS = [1, 6, 24]
REQUIRED_METRICS  = ["R2","MAE","RMSE"]
CITY_COL_CANDIDATES = ["city","City","CITY"]

print(SEP)
print("  BLOCK 1 — FINAL RESULTS CONSISTENCY CHECK")
print("  Publication Package Audit — No Model Training")
print(SEP)

issues = []
report_lines = ["# Results Validation Report\n"]

# ── Track A ────────────────────────────────────────────────────────────────────
print(f"\nTRACK A — AQI ESTIMATION")
print("-" * 60)
ta_frames = {}
for model, fname in TRACK_A_FILES.items():
    fpath = OUT / fname
    if not fpath.exists():
        issues.append(f"Missing Track A file: {fname}")
        sym = "✗"
    else:
        df       = pd.read_csv(fpath)
        city_col = next((c for c in CITY_COL_CANDIDATES if c in df.columns), None)
        n_rows   = len(df)
        n_cities = df[city_col].nunique() if city_col else 0
        n_dup    = df.duplicated(subset=[city_col] if city_col else []).sum()
        missing_metrics = [m for m in REQUIRED_METRICS if m not in df.columns]
        null_counts = {m: int(df[m].isna().sum()) for m in REQUIRED_METRICS if m in df.columns}
        r2_range = (round(float(df["R2"].min()),4), round(float(df["R2"].max()),4)) \
                    if "R2" in df.columns else (None, None)
        ok  = (n_cities >= 10 and not missing_metrics and not any(null_counts.values()))
        sym = "✓" if ok else "⚠"
        ta_frames[model] = df
        print(f"  {sym}  {model:<18}  rows={n_rows:>4}  cities={n_cities:>2}  "
              f"dup={n_dup}  R²∈{r2_range}")
        if not ok:
            issues.append(f"Track A {model}: cities={n_cities}, nulls={null_counts}, "
                          f"missing={missing_metrics}")

# ── Track B ────────────────────────────────────────────────────────────────────
print(f"\nTRACK B — AQI FORECASTING")
print("-" * 60)
tb_frames = {}
for model, fname in TRACK_B_FILES.items():
    fpath = OUT / fname
    if not fpath.exists():
        issues.append(f"Missing Track B file: {fname}")
        sym = "✗"
    else:
        df     = pd.read_csv(fpath)
        hz_col = next((c for c in ["horizon","Horizon"] if c in df.columns), None)
        city_col = next((c for c in CITY_COL_CANDIDATES if c in df.columns), None)
        n_rows = len(df)
        n_cities = df[city_col].nunique() if city_col else 0
        n_hz   = df[hz_col].unique().tolist() if hz_col else []
        n_dup  = df.duplicated(subset=([city_col, hz_col] if city_col and hz_col else [])).sum()
        missing_metrics = [m for m in REQUIRED_METRICS if m not in df.columns]
        null_counts = {m: int(df[m].isna().sum()) for m in REQUIRED_METRICS if m in df.columns}
        r2_range = (round(float(df["R2"].min()),4), round(float(df["R2"].max()),4)) \
                    if "R2" in df.columns else (None, None)
        ok  = (n_cities >= 10 and set(EXPECTED_HORIZONS).issubset(set(n_hz)) and not missing_metrics)
        sym = "✓" if ok else "⚠"
        tb_frames[model] = df
        print(f"  {sym}  {model:<18}  rows={n_rows:>4}  cities={n_cities:>2}  "
              f"horizons={n_hz}  R²∈{r2_range}")
        if not ok:
            issues.append(f"Track B {model}: cities={n_cities}, horizons={n_hz}")

# ── Cross-track checks ─────────────────────────────────────────────────────────
print(f"\nCROSS-TRACK CONSISTENCY")
print("-" * 60)

all_ta = pd.concat(list(ta_frames.values()), ignore_index=True) if ta_frames else pd.DataFrame()
if not all_ta.empty:
    combined = all_ta
    neg_r2    = (combined["R2"] < -0.5).sum() if "R2" in combined else 0
    extreme_r2 = (combined["R2"] > 1.0).sum()  if "R2" in combined else 0
    ta_full_path = OUT / "final_track_a_complete.csv"
    ta_full = pd.read_csv(ta_full_path) if ta_full_path.exists() else pd.DataFrame()
    if not ta_full.empty:
        ta_city_col  = next((c for c in CITY_COL_CANDIDATES if c in ta_full.columns), None)
        ta_model_col = "model" if "model" in ta_full.columns else None
        n_ta = len(ta_full)
        n_ta_models = ta_full[ta_model_col].nunique() if ta_model_col else 0
        n_ta_cities = ta_full[ta_city_col].nunique() if ta_city_col else 0
        ta_dup      = ta_full.duplicated(
            subset=[c for c in [ta_city_col, ta_model_col] if c]).sum()
        expected_ta = EXPECTED_CITIES * len(TRACK_A_FILES)
        sym = "✓" if abs(n_ta - expected_ta) < expected_ta * 0.5 else "⚠"
        print(f"  {sym}  final_track_a_complete.csv:  {n_ta} rows  "
              f"({n_ta_models} models × {n_ta_cities} cities)  "
              f"dup={ta_dup}  neg_R²={neg_r2}  extreme_R²={extreme_r2}")

# ── Final status ───────────────────────────────────────────────────────────────
iss  = len(issues)
status = "PASSED ✓" if iss == 0 else f"WARNINGS — {iss} issue(s)"

md_lines = [
    f"# Results Validation — {status}\n",
    f"## Issues ({iss}):\n",
] + [f"- {i}\n" for i in issues] + [
    f"\n## Summary\n",
    f"Track A files: {len(ta_frames)} / {len(TRACK_A_FILES)}\n",
    f"Track B files: {len(tb_frames)} / {len(TRACK_B_FILES)}\n",
]

md_text = "".join(md_lines)
(Path("outputs/reports") / "results_validation.md").parent.mkdir(parents=True, exist_ok=True)
with open(Path("outputs/reports") / "results_validation.md", "w") as f:
    f.write(md_text)

print(f"\n{SEP}")
print(f"  Status : {status}")
print(f"  Saved  : outputs/reports/results_validation.md")

def add(x): issues.append(x)
