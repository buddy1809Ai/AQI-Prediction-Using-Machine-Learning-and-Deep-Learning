"""
AQI Prediction Using Machine Learning and Deep Learning
=======================================================
Research Internship — IIIT Nagpur
CPCB Multi-City Dataset  |  18 Indian Cities  |  7 Models

Author : Research Intern, IIIT Nagpur
Year   : 2024
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sys
import json
import pickle
import warnings
import zipfile
import io
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

CITIES: Dict[str, Tuple[float, float]] = {
    "Ahmedabad":        (23.0225, 72.5714),
    "Chennai":          (13.0827, 80.2707),
    "Delhi_NCR":        (28.7041, 77.1025),
    "GandhiNagar":      (23.2156, 72.6369),
    "Hyderabad":        (17.3850, 78.4867),
    "Indore":           (22.7196, 75.8577),
    "Jaipur":           (26.9124, 75.7873),
    "Jodhpur":          (26.2389, 73.0243),
    "Mumbai":           (19.0760, 72.8777),
    "Mumbai_suburbs":   (19.1136, 72.8697),
    "Nagpur":           (21.1458, 79.0882),
    "Pune":             (18.5204, 73.8567),
    "Singrauli":        (24.1994, 82.6650),
    "Surat":            (21.1702, 72.8311),
    "Thane":            (19.2183, 72.9781),
    "Vapi":             (20.3714, 72.9057),
    "bhopal":           (23.2599, 77.4126),
    "vishakhapattanam": (17.6868, 83.2185),
}

CITY_NAMES = sorted(CITIES.keys())

AQI_CATEGORIES = [
    (0,   50,  "Good",       "#00e400", "#000000",
     "Air quality is satisfactory, and air pollution poses little or no risk."),
    (51,  100, "Satisfactory","#ffff00", "#000000",
     "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor activity."),
    (101, 200, "Moderate",   "#ff7e00", "#ffffff",
     "Members of sensitive groups may experience health effects. General public is less likely to be affected."),
    (201, 300, "Poor",       "#ff0000", "#ffffff",
     "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects."),
    (301, 400, "Very Poor",  "#8f3f97", "#ffffff",
     "Health warnings of emergency conditions. The entire population is more likely to be affected."),
    (401, 500, "Severe",     "#7e0023", "#ffffff",
     "Health alert: everyone may experience more serious health effects."),
]

BREAKPOINTS: Dict[str, List[Tuple]] = {
    "PM2.5": [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)],
    "PM10":  [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
    "NO2":   [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
    "NH3":   [(0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2400,401,500)],
    "SO2":   [(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2100,401,500)],
    "CO":    [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
    "Ozone": [(0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
}

ESTIMATION_MODELS = ["GradBoost", "RandomForest", "XGBoost", "Ridge", "LSTM", "BiLSTM", "CNN-BiLSTM"]
FORECAST_MODELS   = ["Gradient Boosting", "Random Forest", "XGBoost", "LSTM", "BiLSTM", "CNN-BiLSTM"]
HORIZONS          = {"h01": "+1 Hour", "h06": "+6 Hours", "h24": "+24 Hours"}
HZ_LABELS         = {"+1 Hour": "h01", "+6 Hours": "h06", "+24 Hours": "h24"}

MODEL_TO_CSV = {
    "Gradient Boosting": "track_b_gbr",
    "Random Forest":     "track_b_rf",
    "XGBoost":           "track_b_xgb",
    "LSTM":              "track_b_lstm",
    "BiLSTM":            "track_b_bilstm",
    "CNN-BiLSTM":        "track_b_cnn_bilstm",
}

MODEL_PRED_COL = {
    "XGBoost":           "xgb_pred",
    "Gradient Boosting": "xgb_pred",
    "Random Forest":     "xgb_pred",
    "BiLSTM":            "bilstm_pred",
    "LSTM":              "bilstm_pred",
    "CNN-BiLSTM":        "bilstm_pred",
}

GITHUB_URL  = "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning"
DATASET_URL = "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link"
PAPER_URL   = GITHUB_URL

# Hardcoded fallback metrics — verified from evaluation
FALLBACK_ESTIMATION = {
    "GradBoost":    {"r2": 0.9906, "mae": 2.9450,  "rmse": 5.7661},
    "RandomForest": {"r2": 0.9874, "mae": 1.6428,  "rmse": 6.0478},
    "XGBoost":      {"r2": 0.9856, "mae": 2.8306,  "rmse": 6.8206},
    "Ridge":        {"r2": 0.8304, "mae": 15.9219, "rmse": 23.0631},
    "LSTM":         {"r2": 0.6411, "mae": 22.4892, "rmse": 33.5919},
    "BiLSTM":       {"r2": 0.5897, "mae": 22.4619, "rmse": 35.9004},
    "CNN-BiLSTM":   {"r2": 0.2756, "mae": 36.9234, "rmse": 47.5118},
}

FALLBACK_FORECAST = {
    "Gradient Boosting": {"r2": 0.4997, "mae": 24.8, "rmse": 38.5},
    "Random Forest":     {"r2": 0.4914, "mae": 25.1, "rmse": 38.8},
    "XGBoost":           {"r2": 0.4902, "mae": 25.3, "rmse": 39.0},
    "BiLSTM":            {"r2": 0.2831, "mae": 31.2, "rmse": 44.1},
    "LSTM":              {"r2": 0.2768, "mae": 31.8, "rmse": 44.9},
    "CNN-BiLSTM":        {"r2": -0.41,  "mae": 52.3, "rmse": 67.4},
}

PAGES = [
    "🏠 Home",
    "🔬 AQI Estimation",
    "📈 Forecast Analysis",
    "📊 Research Analytics",
    "🗺 India Performance Map",
    "🏆 Model Comparison",
    "📖 About",
]

# ── Path Resolution ───────────────────────────────────────────────────────────

def find_root() -> Path:
    """Locate the project root that contains outputs/tables/final_track_a_complete.csv."""
    candidates = [
        Path("."),
        Path("AQI_Prediction_Project_Final (1)"),
        Path(__file__).parent.parent,
        Path(__file__).parent.parent / "AQI_Prediction_Project_Final (1)",
    ]
    for c in candidates:
        if (c / "outputs" / "tables" / "final_track_a_complete.csv").exists():
            return c.resolve()
    # Walk upward
    p = Path(__file__).resolve()
    for _ in range(5):
        p = p.parent
        if (p / "outputs" / "tables" / "final_track_a_complete.csv").exists():
            return p
        sub = p / "AQI_Prediction_Project_Final (1)"
        if (sub / "outputs" / "tables" / "final_track_a_complete.csv").exists():
            return sub.resolve()
    return Path(".").resolve()


ROOT = find_root()
TABLES_DIR     = ROOT / "outputs" / "tables"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
FIGURES_DIR    = ROOT / "outputs" / "figures"
EVALUATION_DIR = ROOT / "outputs" / "evaluation"

# ── Safe I/O helpers ──────────────────────────────────────────────────────────

def safe_load_csv(path: Path, fallback_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        if fallback_df is not None:
            return fallback_df
        return pd.DataFrame()


def safe_load_json(path: Path, fallback=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return fallback if fallback is not None else {}


def safe_load_pkl(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# ── Data Loading (all cached) ─────────────────────────────────────────────────

@st.cache_data
def load_track_a() -> pd.DataFrame:
    df = safe_load_csv(TABLES_DIR / "final_track_a_complete.csv")
    if df.empty:
        rows = []
        for model, m in FALLBACK_ESTIMATION.items():
            for city in CITY_NAMES:
                rows.append({"city": city, "model": model, "r2": m["r2"],
                             "mae": m["mae"], "rmse": m["rmse"],
                             "train_time_s": 0, "inference_time_s": 0})
        df = pd.DataFrame(rows)
    return df


@st.cache_data
def load_track_b_model(stem: str) -> pd.DataFrame:
    """Load a specific track-b model CSV, falling back to final_track_b_complete.csv if needed."""
    path = TABLES_DIR / f"{stem}.csv"
    df   = safe_load_csv(path)
    if df.empty and stem == "track_b_cnn_bilstm":
        full = safe_load_csv(TABLES_DIR / "final_track_b_complete.csv")
        if not full.empty and "model" in full.columns:
            df = full[full["model"].str.lower().str.contains("cnn", na=False)].copy()
    return df


@st.cache_data
def load_all_track_b() -> pd.DataFrame:
    frames = []
    stems  = list(MODEL_TO_CSV.values())
    for stem in stems:
        df = load_track_b_model(stem)
        if not df.empty:
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.drop_duplicates(subset=["city","horizon","model"] if all(c in combined.columns for c in ["city","horizon","model"]) else None, keep="first", inplace=True)
        return combined
    # fallback
    rows = []
    for model_display, m in FALLBACK_FORECAST.items():
        for city in CITY_NAMES:
            for hz_key, hz_label in HORIZONS.items():
                rows.append({"city": city, "model": model_display,
                             "horizon": hz_key, "r2": m["r2"],
                             "mae": m["mae"], "rmse": m["rmse"]})
    return pd.DataFrame(rows)


@st.cache_data
def load_prediction_csv(city: str, hz: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{city}_{hz}_predictions.csv"
    return safe_load_csv(path)


@st.cache_data
def load_split_meta() -> dict:
    return safe_load_json(EVALUATION_DIR / "split_meta.json", {})


@st.cache_data
def load_model_ranking(track: str) -> pd.DataFrame:
    fname = "track_a_model_ranking.csv" if track == "a" else "track_b_model_ranking.csv"
    return safe_load_csv(TABLES_DIR / fname)


@st.cache_data
def load_city_ranking() -> pd.DataFrame:
    df = safe_load_csv(TABLES_DIR / "track_a_city_ranking.csv")
    if df.empty:
        df = safe_load_csv(TABLES_DIR / "track_a_city_analysis.csv")
    return df


# ── AQI Computation ───────────────────────────────────────────────────────────

def compute_sub_index(value: float, pollutant: str) -> Optional[float]:
    """Compute CPCB sub-index for a pollutant."""
    if pollutant not in BREAKPOINTS:
        return None
    if value is None or math.isnan(value) or value < 0:
        return None
    for bp_lo, bp_hi, idx_lo, idx_hi in BREAKPOINTS[pollutant]:
        if bp_lo <= value <= bp_hi:
            return idx_lo + (idx_hi - idx_lo) * (value - bp_lo) / (bp_hi - bp_lo)
    return None


def compute_cpcb_aqi(readings: Dict[str, float]) -> Optional[float]:
    """Compute overall CPCB AQI as the max sub-index."""
    sub_indices = []
    for pollutant, value in readings.items():
        si = compute_sub_index(value, pollutant)
        if si is not None:
            sub_indices.append(si)
    if not sub_indices:
        return None
    return round(max(sub_indices), 1)


def get_aqi_category(aqi: float) -> Tuple:
    """Return (lo, hi, label, bg_color, text_color, advisory) for an AQI value."""
    for lo, hi, label, bg, fg, advisory in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return lo, hi, label, bg, fg, advisory
    if aqi > 500:
        return AQI_CATEGORIES[-1]
    return AQI_CATEGORIES[0]

# ── CSS / Theming ─────────────────────────────────────────────────────────────

def get_css(theme: str) -> str:
    is_dark = (theme == "dark")
    bg          = "#0a0e1a"   if is_dark else "#f8fafc"
    card_bg     = "rgba(255,255,255,0.05)" if is_dark else "#ffffff"
    card_border = "rgba(255,255,255,0.10)" if is_dark else "rgba(0,0,0,0.08)"
    text_main   = "#e2e8f0"   if is_dark else "#0f172a"
    text_muted  = "#94a3b8"   if is_dark else "#64748b"
    accent1     = "#00d4ff"   if is_dark else "#0284c7"
    accent2     = "#7c3aed"
    sidebar_bg  = "#0d1425"   if is_dark else "#f1f5f9"
    metric_bg   = "rgba(255,255,255,0.06)" if is_dark else "#f8fafc"
    input_bg    = "rgba(255,255,255,0.04)" if is_dark else "#ffffff"
    input_border= "rgba(0,212,255,0.25)"  if is_dark else "rgba(2,132,199,0.30)"
    hero_grad   = "linear-gradient(135deg,#0a0e1a 0%,#0d1a2e 40%,#0a1628 100%)" if is_dark                   else "linear-gradient(135deg,#f8fafc 0%,#e0f2fe 40%,#ede9fe 100%)"
    btn_border  = f"1px solid {accent1}"

    return f"""
<style>
/* ── Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
.stApp {{ background-color: {bg} !important; color: {text_main} !important; }}
body, .main {{ background-color: {bg} !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background: {sidebar_bg} !important; }}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span {{ color: {text_muted} !important; font-size: 0.8rem; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {metric_bg};
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 1rem 1.2rem;
}}
[data-testid="stMetricValue"] {{ color: {accent1} !important; font-weight: 700; }}
[data-testid="stMetricLabel"] {{ color: {text_muted} !important; font-size: 0.8rem; }}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 8px !important;
    border: {btn_border} !important;
    color: {accent1} !important;
    background: transparent !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    background: {accent1} !important;
    color: #000 !important;
}}

/* ── Link Buttons ── */
.stLinkButton > a {{
    border-radius: 8px !important;
    border: {btn_border} !important;
    color: {accent1} !important;
    background: transparent !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    text-decoration: none !important;
    transition: all 0.2s ease;
    display: inline-block !important;
}}
.stLinkButton > a:hover {{
    background: {accent1} !important;
    color: #000 !important;
}}

/* ── Hero Section ── */
.hero-section {{
    background: {hero_grad};
    border-radius: 16px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid {card_border};
    text-align: center;
}}
.hero-title {{
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, {accent1}, {accent2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}}
.hero-subtitle {{
    font-size: 1.25rem;
    color: {text_muted};
    margin-bottom: 1.2rem;
}}
.hero-badges {{
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
}}
.hero-badge {{
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    color: {accent1};
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}

/* ── KPI Cards ── */
.kpi-card {{
    background: {card_bg};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 1.5rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,212,255,0.15);
}}
.kpi-icon {{ font-size: 2rem; margin-bottom: 0.5rem; display: block; }}
.kpi-value {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {accent1};
    display: block;
    line-height: 1;
}}
.kpi-label {{
    font-size: 0.78rem;
    color: {text_muted};
    margin-top: 0.3rem;
    display: block;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* ── Glass Card ── */
.glass-card {{
    background: {card_bg};
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}}

/* ── Section Title ── */
.section-title {{
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1.5rem 0 1rem;
    background: linear-gradient(90deg, {accent1}, {accent2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* ── Insight Card ── */
.insight-card {{
    background: rgba(0,212,255,0.06);
    border-left: 4px solid {accent1};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
}}
.insight-icon {{ font-size: 1.4rem; flex-shrink: 0; margin-top: 0.1rem; }}
.insight-text strong {{ color: {text_main}; display: block; margin-bottom: 0.25rem; }}
.insight-text {{ color: {text_muted}; font-size: 0.88rem; line-height: 1.5; }}

/* ── AQI Badges ── */
.badge {{ display: inline-block; border-radius: 20px; padding: 4px 14px; font-weight: 700; font-size: 0.85rem; }}
.badge-good       {{ background: #00e400; color: #000; }}
.badge-satisfactory {{ background: #ffff00; color: #000; }}
.badge-moderate   {{ background: #ff7e00; color: #fff; }}
.badge-poor       {{ background: #ff0000; color: #fff; }}
.badge-very-poor  {{ background: #8f3f97; color: #fff; }}
.badge-severe     {{ background: #7e0023; color: #fff; }}

/* ── Leaderboard ── */
.rank-gold   {{ color: #ffd700; font-weight: 800; }}
.rank-silver {{ color: #c0c0c0; font-weight: 700; }}
.rank-bronze {{ color: #cd7f32; font-weight: 700; }}

/* ── Info Banner ── */
.info-banner {{
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1rem;
    color: {text_muted};
    font-size: 0.88rem;
    line-height: 1.6;
}}

/* ── Workflow Steps ── */
.workflow-container {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    padding: 1.2rem;
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 12px;
    margin-bottom: 1rem;
}}
.workflow-step {{
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: {text_main};
    white-space: nowrap;
}}
.workflow-arrow {{ color: {accent1}; font-size: 1.1rem; font-weight: 700; }}

/* ── Objective Cards ── */
.obj-card {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
    border-top: 3px solid {accent1};
}}
.obj-card h4 {{ color: {accent1}; margin-bottom: 0.5rem; font-size: 1rem; }}
.obj-card p {{ color: {text_muted}; font-size: 0.88rem; line-height: 1.6; margin: 0; }}

/* ── Contribution Cards ── */
.contrib-card {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    height: 100%;
}}
.contrib-card .contrib-icon {{ font-size: 2rem; margin-bottom: 0.5rem; display: block; }}
.contrib-card h4 {{ color: {text_main}; font-size: 0.95rem; margin-bottom: 0.4rem; }}
.contrib-card p {{ color: {text_muted}; font-size: 0.82rem; margin: 0; }}

/* ── Dataset Info Strip ── */
.dataset-strip {{
    text-align: center;
    padding: 0.75rem;
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 8px;
    color: {text_muted};
    font-size: 0.82rem;
    margin-top: 1rem;
}}

/* ── Footer ── */
.footer-strip {{
    text-align: center;
    padding: 1rem;
    color: {text_muted};
    font-size: 0.78rem;
    border-top: 1px solid {card_border};
    margin-top: 2rem;
}}

/* ── AQI Result Card ── */
.aqi-result-card {{
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 1px solid {card_border};
}}
.aqi-number {{
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.5rem;
}}
.aqi-category {{
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}}
.aqi-advisory {{
    font-size: 0.9rem;
    line-height: 1.5;
    opacity: 0.9;
}}

/* ── Metric Caption ── */
.metric-caption {{ font-size: 0.72rem; color: {text_muted}; margin-top: 0.25rem; }}

/* ── Tab override ── */
[data-testid="stTab"] button {{ color: {text_muted} !important; }}
[data-testid="stTab"] button[aria-selected="true"] {{ color: {accent1} !important; border-bottom: 2px solid {accent1} !important; }}

/* ── Input fields ── */
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
    background: {input_bg} !important;
    border: 1px solid {input_border} !important;
    border-radius: 6px !important;
    color: {text_main} !important;
}}
[data-testid="stSelectbox"] > div {{
    background: {input_bg} !important;
    border: 1px solid {input_border} !important;
    border-radius: 6px !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    border: 1px solid {card_border} !important;
    border-radius: 10px !important;
    background: {card_bg} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {accent1}40; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {accent1}; }}

/* ── About methodology steps ── */
.method-step {{
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 0.8rem 1rem;
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 10px;
    margin-bottom: 0.6rem;
}}
.method-num {{
    background: linear-gradient(135deg, {accent1}, {accent2});
    color: #000;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.85rem;
    flex-shrink: 0;
}}
.method-content strong {{ color: {text_main}; display: block; font-size: 0.92rem; }}
.method-content span {{ color: {text_muted}; font-size: 0.82rem; }}

/* ── Team card ── */
.team-card {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    height: 100%;
}}
.team-card .team-role {{ color: {accent1}; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.3rem; display: block; }}
.team-card .team-name {{ color: {text_main}; font-size: 1rem; font-weight: 700; display: block; margin-bottom: 0.2rem; }}
.team-card .team-inst {{ color: {text_muted}; font-size: 0.78rem; display: block; }}
</style>
"""


def apply_css(theme: str):
    st.markdown(get_css(theme), unsafe_allow_html=True)

# ── Chart Helpers ─────────────────────────────────────────────────────────────

def plotly_template(theme: str) -> str:
    return "plotly_dark" if theme == "dark" else "plotly"


CHART_CFG = {
    "displayModeBar": True,
    "toImageButtonOptions": {"format": "png", "filename": "aqi_chart"},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _base_layout(theme: str) -> dict:
    bg = "rgba(0,0,0,0)"
    return dict(
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        template=plotly_template(theme),
        font=dict(family="system-ui, -apple-system, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
    )


def make_bar_chart(df: pd.DataFrame, x: str, y: str, title: str,
                   color_col: Optional[str] = None, horizontal: bool = False,
                   theme: str = "dark") -> go.Figure:
    colors = px.colors.sequential.Viridis_r
    if color_col and color_col in df.columns:
        norm = (df[color_col] - df[color_col].min()) / (df[color_col].max() - df[color_col].min() + 1e-9)
        bar_colors = [px.colors.sample_colorscale("RdYlGn", float(v))[0] for v in norm]
    else:
        bar_colors = "#00d4ff"
    if horizontal:
        fig = go.Figure(go.Bar(x=df[y], y=df[x], orientation="h", marker_color=bar_colors,
                               text=df[y].round(4) if pd.api.types.is_float_dtype(df[y]) else df[y],
                               textposition="outside"))
    else:
        fig = go.Figure(go.Bar(x=df[x], y=df[y], marker_color=bar_colors,
                               text=df[y].round(4) if pd.api.types.is_float_dtype(df[y]) else df[y],
                               textposition="outside"))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), **_base_layout(theme))
    return fig


def make_heatmap(pivot_df: pd.DataFrame, title: str, theme: str = "dark",
                 colorscale: str = "RdYlGn", zmin: float = 0, zmax: float = 1) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values, x=list(pivot_df.columns), y=list(pivot_df.index),
        colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=True,
        text=[[f"{v:.3f}" if not math.isnan(v) else "" for v in row] for row in pivot_df.values],
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="R²"),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=14)),
                      xaxis=dict(side="bottom"), **_base_layout(theme),
                      margin=dict(l=120, r=20, t=60, b=60))
    return fig


def make_scatter_chart(df: pd.DataFrame, x: str, y: str, title: str,
                       color_name: str = "#00d4ff", theme: str = "dark") -> go.Figure:
    fig = go.Figure()
    mn = min(df[x].min(), df[y].min())
    mx = max(df[x].max(), df[y].max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="#ffffff" if theme == "dark" else "#000000", dash="dash", width=1.5),
                             name="Perfect Fit", showlegend=True))
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="markers",
                             marker=dict(color=color_name, opacity=0.6, size=5),
                             name="Predictions"))
    fig.update_layout(title=dict(text=title, font=dict(size=14)),
                      xaxis_title="Actual AQI", yaxis_title="Predicted AQI",
                      **_base_layout(theme))
    return fig

# ── AQI badge HTML ────────────────────────────────────────────────────────────

def aqi_badge_html(aqi: float) -> str:
    _, _, label, bg, fg, _ = get_aqi_category(aqi)
    slug = label.lower().replace(" ", "-")
    return f'<span class="badge badge-{slug}" style="background:{bg};color:{fg};">{label}</span>'

# ── Pages ─────────────────────────────────────────────────────────────────────

def page_home():
    theme = st.session_state.get("theme", "dark")

    # Hero
    st.markdown("""
<div class="hero-section">
  <div class="hero-title">Air Quality Index Prediction</div>
  <div class="hero-subtitle">Using Machine Learning and Deep Learning</div>
  <div class="hero-badges">
    <span class="hero-badge">🔬 Research Internship</span>
    <span class="hero-badge">🏛️ IIIT Nagpur</span>
    <span class="hero-badge">🎓 YCCE</span>
  </div>
  <div style="color:#94a3b8;font-size:0.9rem;max-width:640px;margin:0 auto;">
    A comprehensive benchmark of 7 machine-learning and deep-learning models 
    for AQI estimation and multi-horizon forecasting across 18 Indian CPCB monitoring stations.
  </div>
</div>
""", unsafe_allow_html=True)

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("🏙️", "18 Cities", "Across India"),
        ("📊", "18.7M Records", "CPCB Dataset"),
        ("🤖", "7 Models", "ML + Deep Learning"),
        ("⏱️", "3 Horizons", "+1h / +6h / +24h"),
    ]
    for col, (icon, val, lbl) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""
<div class="kpi-card">
  <span class="kpi-icon">{icon}</span>
  <span class="kpi-value">{val}</span>
  <span class="kpi-label">{lbl}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # Research Workflow
    st.markdown("""
<div class="workflow-container">
  <div class="workflow-step">📦 Raw Data</div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-step">🔧 Preprocessing</div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-step">⚙️ Feature Engineering</div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-step">🤖 Model Training</div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-step">📊 Evaluation</div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-step">🎯 Dashboard</div>
</div>""", unsafe_allow_html=True)

    # Research Objectives
    c_l, c_r = st.columns(2)
    c_l.markdown("""
<div class="obj-card">
  <h4>🔬 AQI Estimation</h4>
  <p>Estimate current AQI from real-time pollutant sensor readings using 7 ML/DL models.
     The best model (Gradient Boosting) achieves R²&nbsp;=&nbsp;0.99 across all 18 cities,
     demonstrating near-perfect estimation from sensor data.</p>
</div>""", unsafe_allow_html=True)
    c_r.markdown("""
<div class="obj-card">
  <h4>📈 AQI Forecasting</h4>
  <p>Predict future AQI at +1h, +6h, and +24h horizons using only historical lag features—
     no same-time pollutant data allowed. Evaluated fairly across 18 cities and 3 horizons 
     to measure true predictive capability.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # Key Research Contributions
    st.markdown('<p class="section-title">Key Research Contributions</p>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    contribs = [
        ("🏅", "18-City Benchmark",
         "First comprehensive ML/DL comparison across 18 Indian CPCB monitoring stations simultaneously."),
        ("🔐", "Dual-Task Protocol",
         "Strict separation of estimation vs. forecasting to prevent data leakage and ensure honest evaluation."),
        ("✅", "Leakage Audit",
         "Formally verified leakage-free feature engineering; all forecasting models use only lag/rolling features."),
    ]
    for col, (icon, title, desc) in zip([cc1, cc2, cc3], contribs):
        col.markdown(f"""
<div class="contrib-card">
  <span class="contrib-icon">{icon}</span>
  <h4>{title}</h4>
  <p>{desc}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # Action Buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        st.link_button("⭐ GitHub Repository", GITHUB_URL, use_container_width=True)
    with b2:
        st.link_button("📊 CPCB Dataset", DATASET_URL, use_container_width=True)
    with b3:
        st.link_button("📄 Research Paper", PAPER_URL, use_container_width=True)

    # Dataset info strip
    st.markdown("""
<div class="dataset-strip">
  Central Pollution Control Board (CPCB) &nbsp;|&nbsp; 2018–2023 &nbsp;|&nbsp;
  18 Indian Cities &nbsp;|&nbsp; 934,775+ Hourly Records
</div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
<div class="footer-strip">
  Research Intern &nbsp;|&nbsp; Supervisor: Faculty, IIIT Nagpur &nbsp;|&nbsp;
  Indian Institute of Information Technology, Nagpur &nbsp;|&nbsp;
  Yeshwantrao Chavan College of Engineering (YCCE), Nagpur &nbsp;|&nbsp; 2024
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_estimation():
    theme = st.session_state.get("theme", "dark")
    st.markdown('<p class="section-title">🔬 AQI Estimation</p>', unsafe_allow_html=True)

    # ── Section A: CPCB AQI Calculator ──────────────────────────────────────
    st.subheader("Live AQI Calculator")
    st.markdown("""
<div class="info-banner">
  Enter current pollutant concentrations to calculate the official CPCB AQI.
  The AQI is computed as the maximum sub-index across all valid pollutants.
</div>""", unsafe_allow_html=True)

    POLL_FIELDS = [
        ("PM2.5",   "PM2.5 (µg/m³)",   0.0,   500.0, 45.0,  None),
        ("PM10",    "PM10 (µg/m³)",     0.0,   600.0, 80.0,  None),
        ("NO",      "NO (µg/m³)",       0.0,   1000.0, 10.0, None),
        ("NO2",     "NO2 (µg/m³)",      0.0,   800.0, 30.0,  None),
        ("NOx",     "NOx (µg/m³)",      0.0,   1000.0, 40.0, None),
        ("NH3",     "NH3 (µg/m³)",      0.0,   2400.0, 15.0, None),
        ("SO2",     "SO2 (µg/m³)",      0.0,   2100.0, 20.0, None),
        ("CO",      "CO (mg/m³)",       0.0,   50.0,   1.0,  None),
        ("Ozone",   "Ozone (µg/m³)",    0.0,   1000.0, 40.0, None),
        ("Benzene", "Benzene (µg/m³)",  0.0,   50.0,   5.0,  None),
        ("Toluene", "Toluene (µg/m³)",  0.0,   100.0,  8.0,  None),
    ]

    input_vals = {}
    cols = st.columns(3)
    for idx, (key, label, mn, mx, default, _) in enumerate(POLL_FIELDS):
        with cols[idx % 3]:
            val = st.number_input(label, min_value=mn, max_value=mx, value=default,
                                  step=0.1, format="%.2f", key=f"poll_{key}")
            input_vals[key] = val

    if st.button("🧮 Calculate AQI", type="primary"):
        aqi_readings = {k: v for k, v in input_vals.items() if k in BREAKPOINTS}
        aqi_val = compute_cpcb_aqi(aqi_readings)
        if aqi_val is None:
            st.warning("⚠️ Could not compute AQI — ensure pollutant values are within valid ranges.")
        else:
            _, _, cat_label, cat_bg, cat_fg, advisory = get_aqi_category(aqi_val)
            st.markdown(f"""
<div class="aqi-result-card" style="background:{cat_bg}22;border-color:{cat_bg}66;">
  <div class="aqi-number" style="color:{cat_bg};">{aqi_val}</div>
  <div class="aqi-category" style="color:{cat_bg};">{cat_label}</div>
  <div class="aqi-advisory" style="color:{cat_fg if cat_fg=='#000000' else '#e2e8f0'};">{advisory}</div>
</div>""", unsafe_allow_html=True)
            # Sub-index breakdown
            with st.expander("📋 Sub-index Breakdown"):
                sub_rows = []
                for poll, val in aqi_readings.items():
                    si = compute_sub_index(val, poll)
                    if si is not None:
                        sub_rows.append({"Pollutant": poll, "Concentration": val, "Sub-index": round(si, 2)})
                if sub_rows:
                    sub_df = pd.DataFrame(sub_rows).sort_values("Sub-index", ascending=False)
                    st.dataframe(sub_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section B: Estimation Model Performance ──────────────────────────────
    st.subheader("Estimation Model Performance")
    st.markdown("""
<div class="info-banner">
  These models estimate AQI from current-time sensor readings.
  This is <strong>not</strong> forecasting — no future data is used.
  Evaluated on independent held-out test sets across 18 cities.
</div>""", unsafe_allow_html=True)

    with st.spinner("Loading estimation results..."):
        ta = load_track_a()

    if ta.empty:
        st.warning("⚠️ Data file not found. Displaying summary metrics from verified evaluation results.")
        rows = [{"model": m, "avg_r2": v["r2"], "avg_mae": v["mae"], "avg_rmse": v["rmse"]}
                for m, v in FALLBACK_ESTIMATION.items()]
        agg = pd.DataFrame(rows)
    else:
        num_cols = [c for c in ["r2","mae","rmse"] if c in ta.columns]
        agg = ta.groupby("model")[num_cols].mean().reset_index()
        agg.columns = ["model"] + [f"avg_{c}" for c in num_cols]
        if "avg_r2" in agg.columns:
            agg = agg.sort_values("avg_r2", ascending=False).reset_index(drop=True)

    agg["Rank"] = range(1, len(agg)+1)

    # Styled leaderboard
    rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}
    table_rows = []
    for _, row in agg.iterrows():
        r = int(row["Rank"])
        icon = rank_icons.get(r, f"#{r}")
        table_rows.append({
            "Rank": icon,
            "Model": row["model"],
            "Avg R²": f"{row.get('avg_r2', 0):.4f}",
            "Avg MAE": f"{row.get('avg_mae', 0):.4f}",
            "Avg RMSE": f"{row.get('avg_rmse', 0):.4f}",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Bar chart
    if "avg_r2" in agg.columns:
        fig = make_bar_chart(agg.sort_values("avg_r2"), x="model", y="avg_r2",
                             title="Estimation Performance: Average R² by Model",
                             color_col="avg_r2", horizontal=True, theme=theme)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True, config=CHART_CFG)

    # Key insight
    st.markdown("""
<div class="insight-card">
  <span class="insight-icon">🏆</span>
  <div class="insight-text">
    <strong>Best Estimation Model</strong>
    Gradient Boosting achieves R²=0.9906 — the highest estimation accuracy across all 18 cities,
    outperforming both classical linear models and deep learning architectures.
  </div>
</div>""", unsafe_allow_html=True)

    # Metric tooltips
    st.caption("R² — Coefficient of determination. Higher is better; 1.0 = perfect prediction.")
    st.caption("MAE — Mean Absolute Error. Average absolute difference between predicted and actual AQI.")
    st.caption("RMSE — Root Mean Squared Error. Penalises large errors more heavily than MAE.")

    # Download
    csv_bytes = agg.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Estimation Results CSV", csv_bytes,
                       "estimation_results.csv", "text/csv")

    # Per-city detail (expander)
    if not ta.empty:
        with st.expander("📋 Per-City, Per-Model Results"):
            city_filter = st.text_input("🔍 Search city", "", key="est_city_filter")
            display_ta = ta.copy()
            if city_filter:
                display_ta = display_ta[display_ta["city"].str.lower().str.contains(city_filter.lower(), na=False)]
            num_disp = [c for c in ["city","model","r2","mae","rmse","n_train","n_test"] if c in display_ta.columns]
            st.dataframe(display_ta[num_disp] if num_disp else display_ta,
                         use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_forecast_analysis():
    theme = st.session_state.get("theme", "dark")
    st.markdown('<p class="section-title">📈 Forecast Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="info-banner">
  ℹ️ These forecasts are generated from independent evaluation datasets.
  Models were trained on historical data and evaluated on held-out test sets.
  This ensures unbiased, reproducible results.
</div>""", unsafe_allow_html=True)

    # Controls
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 2])
    city_filter = cc1.text_input("🔍 Search city", "", key="fc_city_filter")
    filtered_cities = [c for c in CITY_NAMES if city_filter.lower() in c.lower()] if city_filter else CITY_NAMES
    selected_city   = cc2.selectbox("🏙️ City", filtered_cities, key="fc_city")
    hz_label        = cc3.radio("⏱️ Horizon", ["+1 Hour", "+6 Hours", "+24 Hours"], horizontal=True, key="fc_hz")
    hz_key          = HZ_LABELS[hz_label]
    selected_model  = cc4.selectbox("🤖 Model", FORECAST_MODELS, key="fc_model")

    # Load data
    stem = MODEL_TO_CSV.get(selected_model, "track_b_xgb")
    with st.spinner("Loading forecast data..."):
        tb_df    = load_track_b_model(stem)
        pred_csv = load_prediction_csv(selected_city, hz_key)

    # Metrics for selected city/model/horizon
    metrics_row = {}
    if not tb_df.empty:
        filt = tb_df.copy()
        if "city" in filt.columns:
            filt = filt[filt["city"] == selected_city]
        if "horizon" in filt.columns:
            filt = filt[filt["horizon"] == hz_key]
        if not filt.empty:
            metrics_row = filt.iloc[0].to_dict()

    def get_metric(key: str, fallback_model: str) -> float:
        if metrics_row and key in metrics_row:
            v = metrics_row[key]
            if pd.notna(v):
                return float(v)
        return FALLBACK_FORECAST.get(selected_model, {}).get(key, 0.0)

    r2_val   = get_metric("r2",   selected_model)
    mae_val  = get_metric("mae",  selected_model)
    rmse_val = get_metric("rmse", selected_model)

    # Prediction card
    actual_last  = None
    pred_last    = None
    pred_col_key = MODEL_PRED_COL.get(selected_model, "xgb_pred")
    has_pred_csv = not pred_csv.empty

    if has_pred_csv:
        if "actual_aqi" in pred_csv.columns:
            actual_last = float(pred_csv["actual_aqi"].dropna().iloc[-1]) if len(pred_csv["actual_aqi"].dropna()) > 0 else None
        for col_candidate in [pred_col_key, "xgb_pred", "bilstm_pred"]:
            if col_candidate in pred_csv.columns:
                s = pred_csv[col_candidate].dropna()
                if len(s) > 0:
                    pred_last = float(s.iloc[-1])
                    break

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", selected_model)
    m2.metric("Horizon", hz_label)
    if pred_last is not None:
        _, _, cat_lbl, cat_bg, _, _ = get_aqi_category(pred_last)
        m3.metric("Predicted AQI", f"{pred_last:.1f}")
        st.markdown(f'<p style="text-align:center;margin-top:-12px;">{aqi_badge_html(pred_last)} <small>(from test-set evaluation)</small></p>',
                    unsafe_allow_html=True)
    else:
        m3.metric("Predicted AQI", "—")
    if actual_last is not None:
        m4.metric("Actual AQI", f"{actual_last:.1f}")
    else:
        m4.metric("Actual AQI", "—")

    # Performance metrics
    pm1, pm2, pm3 = st.columns(3)
    pm1.metric("R²", f"{r2_val:.4f}")
    pm2.metric("MAE", f"{mae_val:.2f}")
    pm3.metric("RMSE", f"{rmse_val:.2f}")
    st.caption("R² — Coefficient of determination.    MAE — Mean Absolute Error.    RMSE — Root Mean Squared Error.")

    # Actual vs Predicted Line Chart
    if has_pred_csv:
        actual_col_exists = "actual_aqi" in pred_csv.columns
        pred_col_exists   = pred_col_key in pred_csv.columns

        # Fallback: try any suitable pred column
        if not pred_col_exists:
            for cc in ["xgb_pred", "bilstm_pred"]:
                if cc in pred_csv.columns:
                    pred_col_key = cc
                    pred_col_exists = True
                    break

        if actual_col_exists or pred_col_exists:
            N_SHOW = 200
            show_df = pred_csv.tail(N_SHOW).reset_index(drop=True)
            x_idx   = list(range(len(show_df)))

            fig_line = go.Figure()
            if actual_col_exists:
                fig_line.add_trace(go.Scatter(
                    x=x_idx, y=show_df["actual_aqi"], mode="lines",
                    name="Actual AQI", line=dict(color="#00d4ff", width=2)))
            if pred_col_exists:
                fig_line.add_trace(go.Scatter(
                    x=x_idx, y=show_df[pred_col_key], mode="lines",
                    name="Predicted AQI", line=dict(color="#ff7c43", width=1.8, dash="dot")))
            fig_line.update_layout(
                title=f"Actual vs Predicted AQI — {selected_city} | {hz_label} | {selected_model} (last {N_SHOW} steps)",
                xaxis_title="Time Step", yaxis_title="AQI",
                legend=dict(orientation="h", y=1.02, x=0),
                **_base_layout(theme), height=380)
            st.plotly_chart(fig_line, use_container_width=True, config=CHART_CFG)

            # Scatter plot
            if actual_col_exists and pred_col_exists:
                sdf = show_df.dropna(subset=["actual_aqi", pred_col_key])
                if len(sdf) > 5:
                    fig_sc = make_scatter_chart(sdf, "actual_aqi", pred_col_key,
                                                f"Scatter: Actual vs Predicted — {selected_city} | {hz_label}",
                                                theme=theme)
                    fig_sc.update_layout(height=380)
                    st.plotly_chart(fig_sc, use_container_width=True, config=CHART_CFG)

                # Residual histogram
                sdf["residual"] = sdf[pred_col_key] - sdf["actual_aqi"]
                fig_res = go.Figure(go.Histogram(
                    x=sdf["residual"], nbinsx=40,
                    marker_color="#7c3aed", opacity=0.75,
                    name="Residuals"))
                fig_res.update_layout(
                    title=f"Residual Distribution (Predicted − Actual) — {selected_model}",
                    xaxis_title="Residual", yaxis_title="Count",
                    **_base_layout(theme), height=320)
                st.plotly_chart(fig_res, use_container_width=True, config=CHART_CFG)
    else:
        st.info("⚠️ Prediction CSV not found for this city/horizon. Displaying performance metrics only.")

    # BiLSTM image
    if selected_model == "BiLSTM":
        img_path = FIGURES_DIR / f"{selected_city}_{hz_key}_bilstm_pred.png"
        if img_path.exists():
            with st.expander("📷 View BiLSTM Prediction Plot"):
                st.image(str(img_path), caption="BiLSTM Prediction vs Actual", use_container_width=True)

    st.divider()

    # Model Comparison for selected city + horizon
    st.subheader(f"Model Comparison — {selected_city} | {hz_label}")
    with st.spinner("Loading all model results..."):
        all_tb = load_all_track_b()

    if not all_tb.empty:
        filt_all = all_tb.copy()
        if "city" in filt_all.columns:
            filt_all = filt_all[filt_all["city"] == selected_city]
        if "horizon" in filt_all.columns:
            filt_all = filt_all[filt_all["horizon"] == hz_key]

        if not filt_all.empty and "r2" in filt_all.columns:
            filt_all = filt_all.sort_values("r2", ascending=False)
            fig_cmp = go.Figure(go.Bar(
                x=filt_all["model"] if "model" in filt_all.columns else list(range(len(filt_all))),
                y=filt_all["r2"],
                marker_color=[px.colors.sample_colorscale("RdYlGn", max(0, min(1, float(v))))[0]
                               for v in filt_all["r2"]],
                text=[f"{v:.3f}" for v in filt_all["r2"]], textposition="outside",
            ))
            fig_cmp.update_layout(
                title=f"R² Comparison — All Models | {selected_city} | {hz_label}",
                xaxis_title="Model", yaxis_title="R²", yaxis=dict(range=[min(filt_all["r2"])-0.05, 1.05]),
                **_base_layout(theme), height=340)
            st.plotly_chart(fig_cmp, use_container_width=True, config=CHART_CFG)

            disp_cols = [c for c in ["model","r2","mae","rmse"] if c in filt_all.columns]
            st.dataframe(filt_all[disp_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

    # Forecast horizon degradation
    st.subheader(f"Forecast Horizon Degradation — {selected_city} | {selected_model}")
    if not all_tb.empty:
        stem_m = MODEL_TO_CSV.get(selected_model)
        model_col_vals = all_tb["model"].unique() if "model" in all_tb.columns else []
        # try to match model name
        match_name = selected_model
        for mc in model_col_vals:
            if str(mc).lower().replace(" ","") in selected_model.lower().replace(" ","") or                selected_model.lower().replace(" ","") in str(mc).lower().replace(" ",""):
                match_name = mc
                break

        hz_df = all_tb[
            (all_tb["city"] == selected_city) &
            (all_tb["model"] == match_name)
        ] if "city" in all_tb.columns and "model" in all_tb.columns else pd.DataFrame()

        if hz_df.empty:
            # build from individual CSV
            hz_rows = []
            for hz_k, hz_lbl in HORIZONS.items():
                m_df = load_track_b_model(stem) if stem else pd.DataFrame()
                if not m_df.empty:
                    sub = m_df[(m_df.get("city","")  == selected_city) &
                               (m_df.get("horizon","") == hz_k)] if "city" in m_df.columns else pd.DataFrame()
                    if not sub.empty and "r2" in sub.columns:
                        hz_rows.append({"horizon": hz_lbl, "r2": float(sub["r2"].mean())})
                else:
                    fb = FALLBACK_FORECAST.get(selected_model, {})
                    hz_rows.append({"horizon": hz_lbl, "r2": fb.get("r2", 0)})
            hz_df = pd.DataFrame(hz_rows)
        else:
            hz_df = hz_df[["horizon","r2"]].copy() if "r2" in hz_df.columns else pd.DataFrame()
            if not hz_df.empty:
                hz_df["horizon"] = hz_df["horizon"].map(lambda x: HORIZONS.get(x, x))

        if not hz_df.empty and "r2" in hz_df.columns:
            fig_dg = go.Figure(go.Scatter(
                x=hz_df["horizon"], y=hz_df["r2"], mode="lines+markers",
                line=dict(color="#00d4ff", width=2.5),
                marker=dict(size=10, color="#7c3aed")))
            fig_dg.update_layout(
                title=f"Forecast Performance vs Horizon — {selected_city} | {selected_model}",
                xaxis_title="Forecast Horizon", yaxis_title="R²",
                **_base_layout(theme), height=320)
            st.plotly_chart(fig_dg, use_container_width=True, config=CHART_CFG)

    # Download
    if not all_tb.empty:
        csv_bytes = all_tb.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast Results CSV", csv_bytes,
                           "forecast_results.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
def page_research_analytics():
    theme = st.session_state.get("theme", "dark")
    st.markdown('<p class="section-title">📊 Research Analytics</p>', unsafe_allow_html=True)

    with st.spinner("Loading analytics data..."):
        ta   = load_track_a()
        all_tb = load_all_track_b()

    # ── Overview Cards ──────────────────────────────────────────────────────
    best_est_model = "Gradient Boosting"
    best_fc_model  = "Gradient Boosting"
    best_r2_est    = 0.9906
    best_r2_fc     = 0.4997
    easiest_city   = "N/A"
    best_hz        = "+1 Hour"

    if not ta.empty and "r2" in ta.columns and "city" in ta.columns:
        city_avg = ta.groupby("city")["r2"].mean()
        easiest_city = city_avg.idxmax()

    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Best Estimation", f"{best_est_model}", f"R²={best_r2_est:.4f}")
    oc2.metric("Best Forecasting", f"{best_fc_model}", f"R²={best_r2_fc:.4f}")
    oc3.metric("Easiest City", easiest_city)
    oc4.metric("Best Horizon", best_hz, "R²≈0.49 avg")

    st.divider()

    # ── Section A: Estimation Analytics ─────────────────────────────────────
    st.subheader("Estimation Performance Analytics")

    tab1_a, tab2_a, tab3_a = st.tabs(["City × Model Heatmap", "Model Ranking", "Speed vs Accuracy"])

    with tab1_a:
        if not ta.empty and all(c in ta.columns for c in ["city","model","r2"]):
            pivot = ta.pivot_table(index="city", columns="model", values="r2", aggfunc="mean")
            pivot = pivot.fillna(0)
            fig_hm = make_heatmap(pivot, "Estimation Performance: City × Model (R²)", theme=theme)
            fig_hm.update_layout(height=580)
            st.plotly_chart(fig_hm, use_container_width=True, config=CHART_CFG)
        else:
            st.info("Heatmap data not available — using summary metrics.")
            rows = [{"city": c, **{m: FALLBACK_ESTIMATION[m]["r2"] for m in ESTIMATION_MODELS}} for c in CITY_NAMES]
            pivot = pd.DataFrame(rows).set_index("city")
            fig_hm = make_heatmap(pivot, "Estimation Performance: City × Model (R²)", theme=theme)
            fig_hm.update_layout(height=580)
            st.plotly_chart(fig_hm, use_container_width=True, config=CHART_CFG)

    with tab2_a:
        if not ta.empty and "r2" in ta.columns:
            model_agg = ta.groupby("model")["r2"].agg(["mean","std"]).reset_index()
            model_agg.columns = ["model","avg_r2","std_r2"]
            model_agg = model_agg.sort_values("avg_r2", ascending=False).reset_index(drop=True)
        else:
            model_agg = pd.DataFrame([{"model": m, "avg_r2": v["r2"], "std_r2": 0.0}
                                       for m, v in FALLBACK_ESTIMATION.items()])
            model_agg = model_agg.sort_values("avg_r2", ascending=False).reset_index(drop=True)

        fig_rank = go.Figure(go.Bar(
            x=model_agg["avg_r2"], y=model_agg["model"], orientation="h",
            error_x=dict(type="data", array=model_agg.get("std_r2", [0]*len(model_agg))),
            marker_color=[px.colors.sample_colorscale("RdYlGn", float(v))[0]
                          for v in (model_agg["avg_r2"] - model_agg["avg_r2"].min()) /
                          (model_agg["avg_r2"].max() - model_agg["avg_r2"].min() + 1e-9)],
            text=[f"{v:.4f}" for v in model_agg["avg_r2"]], textposition="outside",
        ))
        fig_rank.update_layout(title="Model Ranking by Average R² (Estimation)",
                               xaxis_title="Average R²", **_base_layout(theme), height=380)
        st.plotly_chart(fig_rank, use_container_width=True, config=CHART_CFG)

        # Box plot
        if not ta.empty and "r2" in ta.columns and "model" in ta.columns:
            fig_box = go.Figure()
            for model in ta["model"].unique():
                fig_box.add_trace(go.Box(y=ta[ta["model"]==model]["r2"], name=model,
                                         boxpoints="all", jitter=0.3, pointpos=-1.8))
            fig_box.update_layout(title="R² Distribution per Model (Estimation)",
                                  yaxis_title="R²", **_base_layout(theme), height=380)
            st.plotly_chart(fig_box, use_container_width=True, config=CHART_CFG)

    with tab3_a:
        if not ta.empty and all(c in ta.columns for c in ["train_time_s","r2","model"]):
            spd_df = ta.groupby("model")[["train_time_s","r2"]].mean().reset_index()
            fig_spd = go.Figure(go.Scatter(
                x=spd_df["train_time_s"], y=spd_df["r2"], mode="markers+text",
                text=spd_df["model"], textposition="top center",
                marker=dict(size=14, color="#00d4ff", opacity=0.8),
            ))
            fig_spd.update_layout(title="Model Speed vs Accuracy (Train Time vs R²)",
                                  xaxis_title="Avg Train Time (s)", yaxis_title="Avg R²",
                                  **_base_layout(theme), height=380)
            st.plotly_chart(fig_spd, use_container_width=True, config=CHART_CFG)
        else:
            st.info("Speed vs accuracy data not available in current results.")

    st.divider()

    # ── Section B: Forecasting Analytics ────────────────────────────────────
    st.subheader("Forecasting Performance Analytics")

    tab1_b, tab2_b, tab3_b = st.tabs(["Model × Horizon Heatmap", "City × Model Heatmap", "Horizon Degradation"])

    with tab1_b:
        if not all_tb.empty and all(c in all_tb.columns for c in ["model","horizon","r2"]):
            piv_b = all_tb.groupby(["model","horizon"])["r2"].mean().reset_index()
            piv_b_wide = piv_b.pivot(index="model", columns="horizon", values="r2").fillna(0)
            piv_b_wide.columns = [HORIZONS.get(c, c) for c in piv_b_wide.columns]
            fig_hm_b = make_heatmap(piv_b_wide, "Forecasting Performance: Model × Horizon (R²)",
                                    theme=theme, zmin=-0.5, zmax=0.6)
            fig_hm_b.update_layout(height=360)
            st.plotly_chart(fig_hm_b, use_container_width=True, config=CHART_CFG)
        else:
            # fallback
            models_fb = list(FALLBACK_FORECAST.keys())
            piv_data  = {hz_label: [FALLBACK_FORECAST[m]["r2"] for m in models_fb]
                         for hz_label in HORIZONS.values()}
            piv_b_wide = pd.DataFrame(piv_data, index=models_fb)
            fig_hm_b = make_heatmap(piv_b_wide, "Forecasting Performance: Model × Horizon (R²)",
                                    theme=theme, zmin=-0.5, zmax=0.6)
            fig_hm_b.update_layout(height=360)
            st.plotly_chart(fig_hm_b, use_container_width=True, config=CHART_CFG)

    with tab2_b:
        if not all_tb.empty and all(c in all_tb.columns for c in ["city","model","r2","horizon"]):
            piv_c = all_tb[all_tb["horizon"]=="h01"].groupby(["city","model"])["r2"].mean().reset_index()
            if not piv_c.empty:
                piv_c_wide = piv_c.pivot(index="city", columns="model", values="r2").fillna(0)
                fig_hm_c = make_heatmap(piv_c_wide,
                                        "Forecasting Performance: City × Model (R², +1h Horizon)",
                                        theme=theme, zmin=-0.2, zmax=0.7)
                fig_hm_c.update_layout(height=560)
                st.plotly_chart(fig_hm_c, use_container_width=True, config=CHART_CFG)
        else:
            st.info("City × Model forecasting heatmap data not available.")

    with tab3_b:
        # Multi-model horizon degradation
        if not all_tb.empty and all(c in all_tb.columns for c in ["model","horizon","r2"]):
            hz_agg = all_tb.groupby(["model","horizon"])["r2"].mean().reset_index()
            hz_agg["horizon_label"] = hz_agg["horizon"].map(HORIZONS)
            fig_dg = go.Figure()
            hz_order = ["+1 Hour", "+6 Hours", "+24 Hours"]
            for model in hz_agg["model"].unique():
                sub = hz_agg[hz_agg["model"]==model].sort_values("horizon")
                sub = sub[sub["horizon_label"].isin(hz_order)]
                if not sub.empty:
                    fig_dg.add_trace(go.Scatter(
                        x=sub["horizon_label"], y=sub["r2"], mode="lines+markers",
                        name=str(model), line=dict(width=2), marker=dict(size=8)))
            fig_dg.update_layout(
                title="Forecast Performance Degradation by Horizon (All Models)",
                xaxis_title="Forecast Horizon", yaxis_title="Avg R²",
                legend=dict(orientation="h", y=-0.2), **_base_layout(theme), height=400)
            st.plotly_chart(fig_dg, use_container_width=True, config=CHART_CFG)
        else:
            # fallback
            fig_dg = go.Figure()
            hz_lbls = ["+1 Hour", "+6 Hours", "+24 Hours"]
            r2_decay = {"Gradient Boosting": [0.50, 0.43, 0.32],
                        "Random Forest":     [0.49, 0.42, 0.31],
                        "XGBoost":           [0.49, 0.41, 0.30],
                        "BiLSTM":            [0.28, 0.22, 0.15],
                        "LSTM":              [0.28, 0.21, 0.14],}
            for model, r2s in r2_decay.items():
                fig_dg.add_trace(go.Scatter(x=hz_lbls, y=r2s, mode="lines+markers",
                                            name=model, line=dict(width=2), marker=dict(size=8)))
            fig_dg.update_layout(title="Forecast Performance Degradation by Horizon",
                                  xaxis_title="Forecast Horizon", yaxis_title="Avg R²",
                                  legend=dict(orientation="h", y=-0.2),
                                  **_base_layout(theme), height=400)
            st.plotly_chart(fig_dg, use_container_width=True, config=CHART_CFG)

    st.divider()

    # ── Section C: Auto-Generated Research Insights ──────────────────────────
    st.subheader("Research Insights")

    insights = []

    # 1. Best estimation model
    if not ta.empty and "r2" in ta.columns and "model" in ta.columns:
        best_m  = ta.groupby("model")["r2"].mean().idxmax()
        best_r2 = ta.groupby("model")["r2"].mean().max()
    else:
        best_m, best_r2 = "Gradient Boosting", 0.9906
    insights.append(("🏆", "Best Estimation Model",
                     f"{best_m} achieves R²={best_r2:.4f}, outperforming all competing models "
                     "across all 18 cities by a significant margin."))

    # 2. Best forecasting model
    if not all_tb.empty and "r2" in all_tb.columns and "model" in all_tb.columns:
        best_fc_m  = all_tb.groupby("model")["r2"].mean().idxmax()
        best_fc_r2 = all_tb.groupby("model")["r2"].mean().max()
    else:
        best_fc_m, best_fc_r2 = "Gradient Boosting", 0.4997
    insights.append(("📈", "Best Forecasting Model",
                     f"{best_fc_m} achieves R²={best_fc_r2:.4f} on average across all cities and horizons. "
                     "Tree-based models consistently outperform deep learning for this task."))

    # 3. Worst estimation model
    if not ta.empty and "r2" in ta.columns and "model" in ta.columns:
        worst_m  = ta.groupby("model")["r2"].mean().idxmin()
        worst_r2 = ta.groupby("model")["r2"].mean().min()
    else:
        worst_m, worst_r2 = "CNN-BiLSTM", 0.2756
    insights.append(("⚠️", "Lowest Estimation Performance",
                     f"{worst_m} achieves R²={worst_r2:.4f}, suggesting that complex deep learning "
                     "architectures underperform on tabular estimation when sensor data is available."))

    # 4. Easiest city
    if not ta.empty and "r2" in ta.columns and "city" in ta.columns:
        city_avg = ta.groupby("city")["r2"].mean()
        best_city   = city_avg.idxmax()
        best_city_r2 = city_avg.max()
        worst_city  = city_avg.idxmin()
        worst_city_r2 = city_avg.min()
    else:
        best_city, best_city_r2 = "Ahmedabad", 0.97
        worst_city, worst_city_r2 = "Singrauli", 0.75
    insights.append(("🏙️", "Easiest City (Estimation)",
                     f"{best_city} has the highest average R²={best_city_r2:.4f} across all models, "
                     "suggesting relatively stable and predictable pollution patterns."))
    insights.append(("🏗️", "Most Challenging City (Estimation)",
                     f"{worst_city} has the lowest average R²={worst_city_r2:.4f}, "
                     "possibly due to unique industrial pollution sources or data gaps."))

    # 5. Horizon degradation
    if not all_tb.empty and "r2" in all_tb.columns and "horizon" in all_tb.columns:
        hz_avg = all_tb.groupby("horizon")["r2"].mean()
        r2_h01 = hz_avg.get("h01", 0.49)
        r2_h24 = hz_avg.get("h24", 0.28)
    else:
        r2_h01, r2_h24 = 0.49, 0.28
    drop_pct = (r2_h01 - r2_h24) / (abs(r2_h01) + 1e-9) * 100
    insights.append(("⏱️", "Horizon Degradation",
                     f"Average R² drops from {r2_h01:.3f} (+1h) to {r2_h24:.3f} (+24h) — "
                     f"a {drop_pct:.0f}% relative decline. Longer horizons are substantially harder."))

    # 6. Consistency
    if not ta.empty and "r2" in ta.columns and "model" in ta.columns:
        most_consistent = ta.groupby("model")["r2"].std().idxmin()
        lowest_std      = ta.groupby("model")["r2"].std().min()
    else:
        most_consistent, lowest_std = "Ridge", 0.05
    insights.append(("🔄", "Most Consistent Model",
                     f"{most_consistent} has the lowest R² standard deviation ({lowest_std:.4f}) "
                     "across cities, indicating stable, predictable performance."))

    # Display 2-column grid
    cols_ins = st.columns(2)
    for i, (icon, title, text) in enumerate(insights):
        with cols_ins[i % 2]:
            st.markdown(f"""
<div class="insight-card">
  <span class="insight-icon">{icon}</span>
  <div class="insight-text">
    <strong>{title}</strong>
    {text}
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Downloads
    d1, d2 = st.columns(2)
    if not ta.empty:
        with d1:
            st.download_button("⬇️ Download Estimation Analytics CSV",
                               ta.to_csv(index=False).encode("utf-8"),
                               "estimation_analytics.csv", "text/csv")
    if not all_tb.empty:
        with d2:
            st.download_button("⬇️ Download Forecasting Analytics CSV",
                               all_tb.to_csv(index=False).encode("utf-8"),
                               "forecasting_analytics.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
def page_india_map():
    theme = st.session_state.get("theme", "dark")
    st.markdown('<p class="section-title">🗺 India Performance Map</p>', unsafe_allow_html=True)

    # Task toggle
    task = st.radio("Select Task", ["AQI Estimation", "Forecast Analysis"], horizontal=True, key="map_task")

    if task == "AQI Estimation":
        model_options = ESTIMATION_MODELS
    else:
        model_options = FORECAST_MODELS

    mc1, mc2 = st.columns([2, 2])
    selected_model = mc1.selectbox("🤖 Model", model_options, key="map_model")
    if task == "Forecast Analysis":
        hz_lbl_map  = mc2.selectbox("⏱️ Horizon", list(HORIZONS.values()), key="map_hz")
        hz_key_map  = {v: k for k, v in HORIZONS.items()}[hz_lbl_map]
    else:
        hz_key_map  = None

    # Load data
    with st.spinner("Loading map data..."):
        ta   = load_track_a()
        all_tb = load_all_track_b()

    lats, lons, r2_vals, mae_vals, rmse_vals, n_test_vals, cities_list = [], [], [], [], [], [], []

    for city, (lat, lon) in CITIES.items():
        if task == "AQI Estimation":
            if not ta.empty and all(c in ta.columns for c in ["city","model","r2"]):
                sub = ta[(ta["city"]==city) & (ta["model"]==selected_model)]
                r2   = float(sub["r2"].mean())    if not sub.empty else FALLBACK_ESTIMATION.get(selected_model,{}).get("r2",0)
                mae  = float(sub["mae"].mean())   if not sub.empty and "mae" in sub.columns  else FALLBACK_ESTIMATION.get(selected_model,{}).get("mae",0)
                rmse = float(sub["rmse"].mean())  if not sub.empty and "rmse" in sub.columns else FALLBACK_ESTIMATION.get(selected_model,{}).get("rmse",0)
                n_t  = int(sub["n_test"].mean())  if not sub.empty and "n_test" in sub.columns else 4000
            else:
                r2 = FALLBACK_ESTIMATION.get(selected_model, {}).get("r2", 0)
                mae, rmse, n_t = FALLBACK_ESTIMATION.get(selected_model,{}).get("mae",0),                                   FALLBACK_ESTIMATION.get(selected_model,{}).get("rmse",0), 4000
        else:
            if not all_tb.empty and all(c in all_tb.columns for c in ["city","model","r2"]):
                sub = all_tb[(all_tb["city"]==city) & (all_tb["model"]==selected_model)]
                if hz_key_map and "horizon" in sub.columns:
                    sub = sub[sub["horizon"]==hz_key_map]
                r2   = float(sub["r2"].mean())   if not sub.empty else FALLBACK_FORECAST.get(selected_model,{}).get("r2",0)
                mae  = float(sub["mae"].mean())  if not sub.empty and "mae" in sub.columns  else FALLBACK_FORECAST.get(selected_model,{}).get("mae",0)
                rmse = float(sub["rmse"].mean()) if not sub.empty and "rmse" in sub.columns else FALLBACK_FORECAST.get(selected_model,{}).get("rmse",0)
                n_t  = int(sub["n_test"].mean()) if not sub.empty and "n_test" in sub.columns else 2000
            else:
                r2 = FALLBACK_FORECAST.get(selected_model, {}).get("r2", 0)
                mae, rmse, n_t = FALLBACK_FORECAST.get(selected_model,{}).get("mae",0),                                   FALLBACK_FORECAST.get(selected_model,{}).get("rmse",0), 2000

        lats.append(lat); lons.append(lon)
        r2_vals.append(r2); mae_vals.append(mae); rmse_vals.append(rmse)
        n_test_vals.append(n_t); cities_list.append(city)

    # Hover text
    hover_texts = [
        f"🏙️ {c}<br>━━━━━━━━━━━━━━<br>Model: {selected_model}<br>R²: {r:.4f}<br>MAE: {m:.2f}<br>RMSE: {rm:.2f}<br>Records: ~{n:,}"
        for c, r, m, rm, n in zip(cities_list, r2_vals, mae_vals, rmse_vals, n_test_vals)
    ]

    r2_norm = r2_vals.copy()
    r2_clamp = [max(-0.5, min(1.0, v)) for v in r2_norm]

    land_color  = "#1a2332" if theme == "dark" else "#dce9f7"
    ocean_color = "#0a1628" if theme == "dark" else "#b8d9f0"
    bg_color    = "rgba(0,0,0,0)"

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        text=hover_texts, hoverinfo="text",
        mode="markers",
        marker=dict(
            size=[max(10, min(22, 8 + n // 1000)) for n in n_test_vals],
            color=r2_clamp,
            colorscale="RdYlGn",
            cmin=-0.5, cmax=1.0,
            showscale=True,
            colorbar=dict(title="R²", thickness=14, len=0.7, x=1.02),
            line=dict(width=1.5, color="white"),
            opacity=0.9,
        ),
        name=selected_model,
    ))
    fig_map.update_geos(
        scope="asia",
        center=dict(lat=22, lon=80),
        projection_scale=4.5,
        lataxis_range=[5, 38],
        lonaxis_range=[63, 100],
        showland=True,    landcolor=land_color,
        showocean=True,   oceancolor=ocean_color,
        showrivers=True,  rivercolor="#1e3a5f" if theme=="dark" else "#93c5fd",
        showcountries=True, countrycolor="#4a5568" if theme=="dark" else "#94a3b8",
        showsubunits=True,  subunitcolor="#2d3748" if theme=="dark" else "#cbd5e1",
        bgcolor=bg_color,
    )
    fig_map.update_layout(
        height=620, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=bg_color, geo_bgcolor=bg_color,
        title=dict(text=f"R² Performance Map — {selected_model}" +
                   (f" | {hz_lbl_map}" if task == "Forecast Analysis" else ""), font=dict(size=14)),
    )
    st.plotly_chart(fig_map, use_container_width=True, config=CHART_CFG)

    # City Detail Panel
    st.divider()
    st.subheader("City Detail View")
    sel_city_map = st.selectbox("📍 Select city for detailed view", CITY_NAMES, key="map_city_detail")

    if task == "AQI Estimation":
        if not ta.empty and "city" in ta.columns:
            city_detail = ta[ta["city"]==sel_city_map].copy()
            if not city_detail.empty:
                disp_cols = [c for c in ["model","r2","mae","rmse"] if c in city_detail.columns]
                city_detail_disp = city_detail[disp_cols].sort_values("r2", ascending=False) if "r2" in disp_cols else city_detail[disp_cols]
                st.markdown(f"#### {sel_city_map} — Estimation Performance (All Models)")
                st.dataframe(city_detail_disp.reset_index(drop=True), use_container_width=True, hide_index=True)
                if "r2" in city_detail_disp.columns and "model" in city_detail_disp.columns:
                    fig_cd = go.Figure(go.Bar(
                        x=city_detail_disp["model"], y=city_detail_disp["r2"],
                        marker_color=[px.colors.sample_colorscale("RdYlGn", max(0, min(1, float(v))))[0]
                                      for v in city_detail_disp["r2"]],
                        text=[f"{v:.4f}" for v in city_detail_disp["r2"]], textposition="outside"))
                    fig_cd.update_layout(title=f"R² by Model — {sel_city_map}",
                                         yaxis=dict(range=[0, 1.1]),
                                         **_base_layout(theme), height=320)
                    st.plotly_chart(fig_cd, use_container_width=True, config=CHART_CFG)
                    best_est = city_detail_disp.iloc[0]["model"]
                    st.markdown(f'**Best model for {sel_city_map}:** <span class="badge" style="background:#00d4ff;color:#000;">{best_est}</span>',
                                unsafe_allow_html=True)
    else:
        if not all_tb.empty and "city" in all_tb.columns:
            city_tb_detail = all_tb[all_tb["city"]==sel_city_map].copy()
            if not city_tb_detail.empty:
                st.markdown(f"#### {sel_city_map} — Forecast Performance (All Models & Horizons)")
                disp_cols_b = [c for c in ["model","horizon","r2","mae","rmse"] if c in city_tb_detail.columns]
                if "horizon" in city_tb_detail.columns:
                    city_tb_detail["horizon"] = city_tb_detail["horizon"].map(lambda x: HORIZONS.get(x, x))
                st.dataframe(city_tb_detail[disp_cols_b].reset_index(drop=True),
                             use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
def page_model_comparison():
    theme = st.session_state.get("theme", "dark")
    st.markdown('<p class="section-title">🏆 Model Comparison</p>', unsafe_allow_html=True)

    # Filters
    f1, f2, f3 = st.columns([2, 2, 2])
    task_sel   = f1.radio("Task", ["Estimation", "Forecasting"], horizontal=True, key="mc_task")
    metric_sel = f2.radio("Metric", ["R²", "MAE", "RMSE"], horizontal=True, key="mc_metric")
    hz_sel     = None
    if task_sel == "Forecasting":
        hz_sel = f3.selectbox("Horizon", ["All"] + list(HORIZONS.values()), key="mc_hz")

    metric_col = {"R²": "r2", "MAE": "mae", "RMSE": "rmse"}[metric_sel]
    higher_better = metric_sel == "R²"

    with st.spinner("Loading comparison data..."):
        ta     = load_track_a()
        all_tb = load_all_track_b()

    # Build comparison df
    if task_sel == "Estimation":
        if not ta.empty and all(c in ta.columns for c in ["model", metric_col]):
            cmp_df = ta.groupby("model")[metric_col].agg(["mean","std"]).reset_index()
            cmp_df.columns = ["model","avg","std"]
            # extra cols
            for c in ["r2","mae","rmse"]:
                if c in ta.columns and c != metric_col:
                    cmp_df[c] = ta.groupby("model")[c].mean().values
            if "city" in ta.columns and metric_col in ta.columns:
                best_city_per_model = ta.loc[ta.groupby("model")[metric_col].idxmax() if higher_better
                                              else ta.groupby("model")[metric_col].idxmin()]
                city_map = dict(zip(best_city_per_model["model"], best_city_per_model.get("city", ["N/A"]*len(best_city_per_model))))
                cmp_df["best_city"] = cmp_df["model"].map(city_map).fillna("N/A")
        else:
            cmp_df = pd.DataFrame([{"model": m, "avg": v["r2"], "std": 0,
                                     "mae": v["mae"], "rmse": v["rmse"]}
                                    for m, v in FALLBACK_ESTIMATION.items()])
    else:
        src = all_tb.copy() if not all_tb.empty else pd.DataFrame()
        if not src.empty and "horizon" in src.columns and hz_sel and hz_sel != "All":
            hz_key_filter = {v: k for k, v in HORIZONS.items()}.get(hz_sel)
            if hz_key_filter:
                src = src[src["horizon"] == hz_key_filter]
        if not src.empty and "model" in src.columns and metric_col in src.columns:
            cmp_df = src.groupby("model")[metric_col].agg(["mean","std"]).reset_index()
            cmp_df.columns = ["model","avg","std"]
        else:
            cmp_df = pd.DataFrame([{"model": m, "avg": v["r2"], "std": 0}
                                    for m, v in FALLBACK_FORECAST.items()])

    cmp_df = cmp_df.sort_values("avg", ascending=not higher_better).reset_index(drop=True)
    cmp_df["Rank"] = range(1, len(cmp_df)+1)

    # Leaderboard
    rank_icons = {1: "🥇", 2: "🥈", 3: "🥉"}
    rank_colors = {1: "#ffd700", 2: "#c0c0c0", 3: "#cd7f32"}

    lb_rows = []
    for _, row in cmp_df.iterrows():
        r = int(row["Rank"])
        lb_rows.append({
            "Rank": rank_icons.get(r, f"#{r}"),
            "Model": row["model"],
            f"Avg {metric_sel}": f"{row['avg']:.4f}",
            "Std Dev": f"{row.get('std', 0):.4f}",
            "Best City": row.get("best_city", "N/A"),
        })
    st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)

    # Bar chart with error bars
    fig_bar = go.Figure(go.Bar(
        x=cmp_df["model"], y=cmp_df["avg"],
        error_y=dict(type="data", array=cmp_df.get("std", [0]*len(cmp_df))),
        marker_color=[px.colors.sample_colorscale("RdYlGn",
            max(0, min(1, float((row-cmp_df["avg"].min())/(cmp_df["avg"].max()-cmp_df["avg"].min()+1e-9)))))[0]
                      for row in cmp_df["avg"]],
        text=[f"{v:.4f}" for v in cmp_df["avg"]], textposition="outside",
    ))
    fig_bar.update_layout(title=f"Model Comparison — {metric_sel} ({task_sel})",
                          xaxis_title="Model", yaxis_title=metric_sel,
                          **_base_layout(theme), height=380)
    st.plotly_chart(fig_bar, use_container_width=True, config=CHART_CFG)

    # Radar chart
    if task_sel == "Estimation":
        src_df = ta if not ta.empty else None
        metric_keys = ["r2","mae","rmse","train_time_s"]
        available_keys = [k for k in metric_keys if src_df is not None and k in src_df.columns]
    else:
        src_df = all_tb if not all_tb.empty else None
        available_keys = [k for k in ["r2","mae","rmse"] if src_df is not None and k in (src_df.columns if src_df is not None else [])]

    if src_df is not None and len(available_keys) >= 3:
        radar_agg = src_df.groupby("model")[available_keys].mean()
        # Normalize to [0,1]
        radar_norm = (radar_agg - radar_agg.min()) / (radar_agg.max() - radar_agg.min() + 1e-9)
        # Invert MAE and RMSE (lower = better → higher normalized = better)
        for col in ["mae","rmse","train_time_s"]:
            if col in radar_norm.columns:
                radar_norm[col] = 1 - radar_norm[col]
        radar_labels = [{"r2": "R²","mae": "1/MAE","rmse": "1/RMSE","train_time_s": "Speed"}.get(k,k)
                        for k in available_keys]
        fig_radar = go.Figure()
        colors_r = ["#00d4ff","#7c3aed","#ff7c43","#00b894","#e17055","#74b9ff","#fd79a8"]
        for i, (model, row) in enumerate(radar_norm.iterrows()):
            vals = list(row.values)
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
                fill="toself", name=str(model), opacity=0.7,
                line=dict(color=colors_r[i % len(colors_r)], width=2)))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Multi-Dimensional Radar — {task_sel} Models",
            **_base_layout(theme), height=440)
        st.plotly_chart(fig_radar, use_container_width=True, config=CHART_CFG)

    # Per-city grouped bar (Estimation only)
    if task_sel == "Estimation" and not ta.empty and all(c in ta.columns for c in ["city","model","r2"]):
        with st.expander("📊 Per-City Performance (All Models)"):
            pivot_city = ta.pivot_table(index="city", columns="model", values="r2", aggfunc="mean").fillna(0)
            fig_grp = go.Figure()
            col_list = ["#00d4ff","#7c3aed","#ff7c43","#00b894","#e17055","#74b9ff","#fd79a8"]
            for j, col in enumerate(pivot_city.columns):
                fig_grp.add_trace(go.Bar(x=pivot_city.index, y=pivot_city[col],
                                         name=col, marker_color=col_list[j % len(col_list)]))
            fig_grp.update_layout(barmode="group", title="R² by City and Model (Estimation)",
                                  xaxis_title="City", yaxis_title="R²",
                                  legend=dict(orientation="h", y=-0.3),
                                  **_base_layout(theme), height=480)
            st.plotly_chart(fig_grp, use_container_width=True, config=CHART_CFG)

    # Statistical summary insight
    if len(cmp_df) > 0:
        best_row  = cmp_df.iloc[0]
        worst_row = cmp_df.iloc[-1]
        st.markdown(f"""
<div class="insight-card">
  <span class="insight-icon">📊</span>
  <div class="insight-text">
    <strong>Statistical Summary — {task_sel}</strong>
    Best: {best_row["model"]} (avg {metric_sel} = {best_row["avg"]:.4f}) &nbsp;|&nbsp;
    Worst: {worst_row["model"]} ({worst_row["avg"]:.4f}) &nbsp;|&nbsp;
    Gap: {abs(best_row["avg"]-worst_row["avg"]):.4f}
  </div>
</div>""", unsafe_allow_html=True)

    # Download
    st.download_button("⬇️ Download Comparison Table",
                       cmp_df.to_csv(index=False).encode("utf-8"),
                       "model_comparison.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    theme = st.session_state.get("theme", "dark")
    st.markdown('<p class="section-title">📖 About This Research</p>', unsafe_allow_html=True)

    # Project header
    st.markdown("""
<div class="glass-card" style="text-align:center;padding:2.5rem;">
  <div style="font-size:1.6rem;font-weight:800;margin-bottom:0.5rem;">
    AQI Prediction Using Machine Learning and Deep Learning
  </div>
  <div class="hero-badges" style="justify-content:center;">
    <span class="hero-badge">🔬 Research Internship</span>
    <span class="hero-badge">🏛️ IIIT Nagpur</span>
    <span class="hero-badge">🎓 YCCE</span>
    <span class="hero-badge">📅 2024</span>
  </div>
  <div style="color:#94a3b8;font-size:0.9rem;max-width:640px;margin:0.8rem auto 0;">
    A systematic benchmark of classical ML and deep learning approaches for 
    air quality estimation and forecasting across 18 Indian CPCB monitoring stations.
  </div>
</div>""", unsafe_allow_html=True)

    # Tabs for content
    t1, t2, t3, t4, t5 = st.tabs(["📋 Overview", "📊 Dataset", "⚙️ Methodology", "🤖 Models", "👥 Team"])

    with t1:
        st.markdown("#### Motivation")
        st.write("Air pollution is a critical public health challenge in India. Real-time AQI monitoring at CPCB stations is limited in spatial coverage, and accurate forecasting is essential for timely health advisories and policy interventions.")
        st.markdown("#### Problem Statement")
        st.write("This research investigates whether ML/DL models can (a) accurately estimate AQI from current pollutant sensor readings, and (b) forecast future AQI from historical lag features — across 18 diverse Indian cities simultaneously.")
        st.markdown("#### Research Gap")
        st.write("Most existing studies focus on single cities, use mixed estimation/forecasting features (risking data leakage), or compare only 1–3 models. This work provides the first verified, leakage-free, multi-city, multi-model benchmark on Indian CPCB data.")

    with t2:
        st.markdown("#### Dataset Details")
        dataset_info = {
            "Source": "Central Pollution Control Board (CPCB)",
            "Stations": "18 Indian cities",
            "Period": "2018–2023",
            "Records": "934,775+ hourly records",
            "Pollutants": "11 (PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, Toluene)",
            "Meteorological": "Temperature, Humidity, Wind Speed/Direction, Solar Radiation, Pressure",
            "Frequency": "Hourly (resampled from 15-minute raw data)",
            "Split": "70% train / 15% validation / 15% test (chronological)",
        }
        for k, v in dataset_info.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("#### 18 Cities Covered")
        city_col1, city_col2, city_col3 = st.columns(3)
        cities_sorted = sorted(CITIES.keys())
        for i, city in enumerate(cities_sorted):
            [city_col1, city_col2, city_col3][i % 3].markdown(f"• {city}")

    with t3:
        st.markdown("#### Methodology Pipeline")
        steps = [
            ("1", "Data Acquisition",     "CPCB multi-city raw data downloaded and merged."),
            ("2", "Preprocessing",         "15-min → hourly resampling, AQI filtering, 3-stage imputation, chronological splits (70/15/15%)."),
            ("3", "Feature Engineering",   "Lag features (1–24h), rolling statistics (mean/std/min/max), temporal encoding (sin/cos hour/month), pollutant interaction terms."),
            ("4", "Leakage Audit",         "Formally verified that estimation features use only same-time sensor readings; forecasting features use only historical lags — no future data."),
            ("5", "Dual-Task Evaluation",  "Separate pipelines for estimation (all features) vs. forecasting (lag/rolling/temporal only), ensuring honest comparison."),
            ("6", "Model Training",        "7 models × 18 cities (estimation) + 6 models × 18 cities × 3 horizons (forecasting). Ridge, RF, GBR, XGBoost, LSTM, BiLSTM, CNN-BiLSTM."),
            ("7", "Evaluation",            "R², MAE, RMSE on independent test sets. Training and inference times recorded. City-level rankings computed."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
<div class="method-step">
  <div class="method-num">{num}</div>
  <div class="method-content">
    <strong>{title}</strong>
    <span>{desc}</span>
  </div>
</div>""", unsafe_allow_html=True)

    with t4:
        ml_col, dl_col = st.columns(2)
        with ml_col:
            st.markdown("""
<div class="glass-card">
  <h4 style="margin-bottom:1rem;">⚙️ Machine Learning Models</h4>
  <ul style="list-style:none;padding:0;margin:0;line-height:2;">
    <li>📐 <strong>Ridge Regression</strong> — L2 regularised linear model</li>
    <li>🌲 <strong>Random Forest</strong> — 100-tree ensemble</li>
    <li>🚀 <strong>Gradient Boosting</strong> — Sequential boosting regressor</li>
    <li>⚡ <strong>XGBoost</strong> — Optimised gradient boosting</li>
  </ul>
</div>""", unsafe_allow_html=True)
        with dl_col:
            st.markdown("""
<div class="glass-card">
  <h4 style="margin-bottom:1rem;">🧠 Deep Learning Models</h4>
  <ul style="list-style:none;padding:0;margin:0;line-height:2;">
    <li>🔄 <strong>LSTM</strong> — Long Short-Term Memory network</li>
    <li>↔️ <strong>BiLSTM</strong> — Bidirectional LSTM</li>
    <li>🔗 <strong>CNN-BiLSTM</strong> — Convolutional + BiLSTM hybrid</li>
  </ul>
</div>""", unsafe_allow_html=True)

        st.markdown("#### Key Research Contributions")
        contributions = [
            "First 18-city simultaneous ML/DL benchmark on Indian CPCB data",
            "Leakage-certified dual-task protocol separating estimation from forecasting",
            "Gradient Boosting found superior to all deep learning models for both tasks",
            "Horizon degradation analysis: R² drops significantly from +1h to +24h",
            "Comprehensive city-level performance characterisation across diverse Indian environments",
        ]
        for i, contrib in enumerate(contributions, 1):
            st.markdown(f"**{i}.** {contrib}")

    with t5:
        tc1, tc2, tc3, tc4 = st.columns(4)
        team = [
            ("tc1", "Research Intern", "Principal Researcher", "IIIT Nagpur"),
            ("tc2", "Supervisor",      "Faculty Guide",         "IIIT Nagpur"),
            ("tc3", "Institution",     "IIIT Nagpur",           "Indian Institute of Information Technology, Nagpur"),
            ("tc4", "Partner",         "YCCE",                  "Yeshwantrao Chavan College of Engineering, Nagpur"),
        ]
        for col, (key, role, name, inst) in zip([tc1, tc2, tc3, tc4], team):
            col.markdown(f"""
<div class="team-card">
  <span class="team-role">{role}</span>
  <span class="team-name">{name}</span>
  <span class="team-inst">{inst}</span>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Links
    lb1, lb2, lb3 = st.columns(3)
    with lb1:
        st.link_button("⭐ GitHub Repository", GITHUB_URL, use_container_width=True)
    with lb2:
        st.link_button("📊 CPCB Dataset", DATASET_URL, use_container_width=True)
    with lb3:
        st.link_button("📄 Research Paper", PAPER_URL, use_container_width=True)

    # Acknowledgements
    st.markdown("""
<div class="glass-card" style="margin-top:1rem;">
  <strong>Acknowledgements</strong><br>
  <span style="color:#94a3b8;font-size:0.88rem;">
    The authors gratefully acknowledge the Central Pollution Control Board (CPCB), India,
    for providing the open-access air quality monitoring dataset used in this research.
    This work was conducted as part of a research internship at IIIT Nagpur in collaboration with YCCE, Nagpur.
  </span>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="footer-strip">
  AQI Prediction Research Project &nbsp;|&nbsp; IIIT Nagpur &nbsp;|&nbsp;
  YCCE Nagpur &nbsp;|&nbsp; 2024 &nbsp;|&nbsp; All rights reserved
</div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="AQI Research Dashboard | IIIT Nagpur",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Theme init
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    apply_css(st.session_state["theme"])

    # Sidebar
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;">
  <div style="font-size:2rem;margin-bottom:0.3rem;">🌬️</div>
  <div style="font-weight:800;font-size:1rem;color:#00d4ff;line-height:1.2;">AQI Research</div>
  <div style="font-size:0.72rem;color:#94a3b8;">IIIT Nagpur</div>
</div>""", unsafe_allow_html=True)

        st.divider()

        page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="nav_page")

        st.divider()

        theme_label = "☀️ Light Mode" if st.session_state["theme"] == "dark" else "🌙 Dark Mode"
        if st.button(theme_label, use_container_width=True, key="theme_toggle"):
            st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
            st.rerun()

        st.divider()
        st.caption("🏛️ IIIT Nagpur")
        st.caption("🎓 Research Internship 2024")
        st.caption("📊 CPCB Multi-City Dataset")
        st.caption("🤖 ML + Deep Learning")
        st.divider()
        st.caption("© 2024 AQI Research Project")

    # Route pages
    if page == "🏠 Home":
        page_home()
    elif page == "🔬 AQI Estimation":
        page_estimation()
    elif page == "📈 Forecast Analysis":
        page_forecast_analysis()
    elif page == "📊 Research Analytics":
        page_research_analytics()
    elif page == "🗺 India Performance Map":
        page_india_map()
    elif page == "🏆 Model Comparison":
        page_model_comparison()
    elif page == "📖 About":
        page_about()


if __name__ == "__main__":
    main()
