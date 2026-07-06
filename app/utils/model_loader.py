"""model_loader.py — Load precomputed metrics and evaluation CSVs."""
import json, warnings
import pandas as pd
import streamlit as st
from pathlib import Path
from app.config.settings import METRICS_JSON, TABLES_DIR, EVAL_DIR, METRICS_CACHE_TTL
warnings.filterwarnings("ignore")

@st.cache_data(ttl=METRICS_CACHE_TTL, show_spinner=False)
def load_metrics() -> dict:
    try:
        with open(METRICS_JSON) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=METRICS_CACHE_TTL, show_spinner=False)
def load_track_a_results() -> pd.DataFrame:
    frames = []
    file_map = {"LinearRegression":"track_a_linear_regression","Ridge":"track_a_ridge",
                "RandomForest":"track_a_rf","GradBoost":"track_a_gbr","XGBoost":"track_a_xgb",
                "LSTM":"track_a_lstm","BiLSTM":"track_a_bilstm","CNN-BiLSTM":"track_a_cnn_bilstm"}
    combined = TABLES_DIR / "final_track_a_complete.csv"
    if combined.exists():
        try: return pd.read_csv(combined)
        except: pass
    for model, stem in file_map.items():
        for d in [TABLES_DIR, EVAL_DIR]:
            p = d / f"{stem}.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p); df["model"] = model; frames.append(df); break
                except: continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(ttl=METRICS_CACHE_TTL, show_spinner=False)
def load_track_b_results() -> pd.DataFrame:
    frames = []
    file_map = {"LinearRegression":"trackB_linear_regression_results","RandomForest":"track_b_rf",
                "GradBoost":"track_b_gbr","XGBoost":"track_b_xgb","LSTM":"track_b_lstm",
                "BiLSTM":"track_b_bilstm","CNN-BiLSTM":"track_b_cnn_bilstm"}
    for model, stem in file_map.items():
        for d in [TABLES_DIR, EVAL_DIR]:
            p = d / f"{stem}.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p); df["model"] = model; frames.append(df); break
                except: continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(ttl=METRICS_CACHE_TTL, show_spinner=False)
def load_sample_data() -> dict:
    from app.config.settings import DATA_DIR
    dfs = {}
    for p in sorted(DATA_DIR.glob("*_sample.csv")):
        city = p.stem.replace("_sample","").replace("_"," ")
        try: dfs[city] = pd.read_csv(p)
        except: continue
    return dfs
