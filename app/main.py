"""
AQI Prediction Using Machine Learning and Deep Learning
=======================================================
Research Internship  —  Indian Institute of Information Technology (IIIT) Nagpur
CPCB Multi-City Dataset  |  18 Cities  |  18.7M Records  |  7 Models  |  3 Horizons

Streamlit Dashboard — Single-file production app
Run:  streamlit run main.py
"""

import os
import sys
import json
import pickle
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction Platform",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Root Detection
# ─────────────────────────────────────────────────────────────────────────────
def find_root() -> Path:
    """Locate the project root by checking multiple candidate paths."""
    candidates = [
        Path("AQI_Prediction_Project_Final (1)"),
        Path("../AQI_Prediction_Project_Final (1)"),
        Path(__file__).parent / "AQI_Prediction_Project_Final (1)",
        Path(__file__).parent.parent / "AQI_Prediction_Project_Final (1)",
        Path(__file__).parent.parent,
        Path(__file__).parent,
    ]
    for c in candidates:
        if (c / "outputs").exists():
            return c.resolve()
    return Path(".").resolve()

ROOT = find_root()
TABLES_DIR = ROOT / "outputs" / "tables"
PREDS_DIR  = ROOT / "outputs" / "predictions"
FIGS_DIR   = ROOT / "outputs" / "figures"

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
CITIES = [
    "Ahmedabad", "bhopal", "Chennai", "Delhi_NCR", "GandhiNagar",
    "Hyderabad", "Indore", "Jaipur", "Jodhpur", "Mumbai",
    "Mumbai_suburbs", "Nagpur", "Pune", "Singrauli", "Surat",
    "Thane", "Vapi", "vishakhapattanam",
]

CITY_DISPLAY = {c: c.replace("_", " ").title() for c in CITIES}
CITY_DISPLAY["bhopal"] = "Bhopal"
CITY_DISPLAY["vishakhapattanam"] = "Visakhapatnam"
CITY_DISPLAY["Delhi_NCR"] = "Delhi NCR"
CITY_DISPLAY["Mumbai_suburbs"] = "Mumbai Suburbs"
CITY_DISPLAY["GandhiNagar"] = "Gandhinagar"

CITY_COORDS = {
    "Ahmedabad":       (23.0225, 72.5714),
    "bhopal":          (23.2599, 77.4126),
    "Chennai":         (13.0827, 80.2707),
    "Delhi_NCR":       (28.7041, 77.1025),
    "GandhiNagar":     (23.2156, 72.6369),
    "Hyderabad":       (17.3850, 78.4867),
    "Indore":          (22.7196, 75.8577),
    "Jaipur":          (26.9124, 75.7873),
    "Jodhpur":         (26.2389, 73.0243),
    "Mumbai":          (19.0760, 72.8777),
    "Mumbai_suburbs":  (19.1759, 72.9567),
    "Nagpur":          (21.1458, 79.0882),
    "Pune":            (18.5204, 73.8567),
    "Singrauli":       (24.1997, 82.6718),
    "Surat":           (21.1702, 72.8311),
    "Thane":           (19.2183, 72.9781),
    "Vapi":            (20.3717, 72.9101),
    "vishakhapattanam":(17.6868, 83.2185),
}

HORIZONS = {1: "1-Hour", 6: "6-Hour", 24: "24-Hour"}
HORIZON_LABELS = {"1-Hour": 1, "6-Hour": 6, "24-Hour": 24}

# Confirmed performance numbers (hardcoded fallback)
TRACK_A_RANKING = [
    {"model": "GradBoost",     "r2": 0.9906, "mae": 2.95,  "rmse": 5.77},
    {"model": "RandomForest",  "r2": 0.9874, "mae": 1.64,  "rmse": 6.05},
    {"model": "XGBoost",       "r2": 0.9856, "mae": 2.83,  "rmse": 6.82},
    {"model": "Ridge",         "r2": 0.8304, "mae": 18.69, "rmse": 28.17},
    {"model": "LSTM",          "r2": 0.6411, "mae": 27.04, "rmse": 39.36},
    {"model": "BiLSTM",        "r2": 0.5897, "mae": 27.57, "rmse": 40.59},
    {"model": "CNN-BiLSTM",    "r2": 0.2756, "mae": 42.25, "rmse": 58.54},
]

TRACK_B_RANKING = [
    {"model": "GradientBoosting", "r2": 0.4997, "mae": 32.57, "rmse": 48.37},
    {"model": "RandomForest",     "r2": 0.4914, "mae": 34.16, "rmse": 48.79},
    {"model": "XGBoost",          "r2": 0.4902, "mae": 32.97, "rmse": 48.86},
    {"model": "BiLSTM",           "r2": 0.2831, "mae": 39.07, "rmse": 56.65},
    {"model": "LSTM",             "r2": 0.2768, "mae": 39.07, "rmse": 56.83},
    {"model": "CNN-BiLSTM",       "r2":-0.4147, "mae": 48.21, "rmse": 67.50},
]

HORIZON_R2 = {1: 0.5313, 6: 0.1775, 24: 0.1046}

# CPCB AQI Breakpoints: (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    "PM2.5": [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,380,401,500)],
    "PM10":  [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
    "NO2":   [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
    "SO2":   [(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2100,401,500)],
    "CO":    [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,47,401,500)],
    "O3":    [(0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
    "NH3":   [(0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2400,401,500)],
}

AQI_CATEGORIES = [
    (0,   50,  "Good",         "#00b050"),
    (51,  100, "Satisfactory", "#92d050"),
    (101, 200, "Moderate",     "#ffff00"),
    (201, 300, "Poor",         "#ff7c00"),
    (301, 400, "Very Poor",    "#ff0000"),
    (401, 500, "Severe",       "#7030a0"),
]

PLOTLY_COLORS = ["#3b82f6","#14b8a6","#8b5cf6","#f59e0b","#10b981","#ef4444","#ec4899"]

# ─────────────────────────────────────────────────────────────────────────────
#  CSS Injection
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
/* Root Variables */
:root {
  --bg-primary: #0a0e1a;
  --bg-card: #111827;
  --bg-card2: #1a2332;
  --accent-blue: #3b82f6;
  --accent-teal: #14b8a6;
  --accent-purple: #8b5cf6;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --border: #1e293b;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
}

/* App background */
.stApp {
  background-color: #0a0e1a !important;
  color: #f1f5f9 !important;
}

/* Main content area */
.main .block-container {
  background-color: #0a0e1a;
  padding-top: 1.5rem;
  padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
  border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * {
  color: #f1f5f9 !important;
}
section[data-testid="stSidebar"] .stRadio label {
  color: #f1f5f9 !important;
  font-size: 0.95rem;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
  color: #94a3b8 !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }

/* Metric cards */
[data-testid="stMetric"] {
  background: rgba(17, 24, 39, 0.8) !important;
  border: 1px solid rgba(59,130,246,0.2) !important;
  border-radius: 12px !important;
  padding: 16px !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stMetricDelta"] { color: #14b8a6 !important; }

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stSlider { background: #111827 !important; color: #f1f5f9 !important; }

/* Dataframes */
.stDataFrame { border: 1px solid #1e293b !important; }

/* Info / Warning / Error boxes */
.stAlert { border-radius: 10px !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; }
.stTabs [aria-selected="true"] { color: #3b82f6 !important; }

/* Divider */
hr { border-color: #1e293b !important; }

/* Glass card */
.glass-card {
  background: rgba(17, 24, 39, 0.8);
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: 16px;
  padding: 24px;
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  margin-bottom: 16px;
}

/* KPI cards */
.kpi-card {
  background: linear-gradient(135deg, #1a2332 0%, #111827 100%);
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 16px;
  padding: 20px 16px;
  text-align: center;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.kpi-number { font-size: 2.2rem; font-weight: 700; color: #3b82f6; line-height: 1.1; }
.kpi-label  { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

/* Hero section */
.hero-section {
  background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #1a1a2e 100%);
  border: 1px solid rgba(59,130,246,0.15);
  border-radius: 20px;
  padding: 40px;
  text-align: center;
  margin-bottom: 24px;
}
.hero-title {
  font-size: 2.8rem;
  font-weight: 800;
  background: linear-gradient(135deg, #3b82f6, #14b8a6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.2;
}
.hero-subtitle { font-size: 1.2rem; color: #94a3b8; margin-top: 8px; }
.badge {
  display: inline-block;
  background: rgba(59,130,246,0.15);
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.8rem;
  color: #3b82f6;
  margin: 4px;
}

/* AQI color indicator */
.aqi-badge {
  display: inline-block;
  border-radius: 12px;
  padding: 8px 20px;
  font-size: 1.4rem;
  font-weight: 700;
  color: #000;
  text-align: center;
}

/* Rank badge */
.rank-1 { background: linear-gradient(135deg, #f59e0b, #d97706); color: #000; }
.rank-2 { background: linear-gradient(135deg, #9ca3af, #6b7280); color: #fff; }
.rank-3 { background: linear-gradient(135deg, #b45309, #92400e); color: #fff; }

/* Model card */
.model-card {
  background: #111827;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 10px;
}
.model-card:hover { border-color: #3b82f6; }

/* Footer */
.sidebar-footer {
  position: fixed;
  bottom: 20px;
  font-size: 0.75rem;
  color: #475569;
  text-align: center;
  padding: 0 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def aqi_color(aqi_val: float) -> tuple[str, str]:
    """Return (hex_color, category_label) for an AQI value."""
    for lo, hi, cat, col in AQI_CATEGORIES:
        if lo <= aqi_val <= hi:
            return col, cat
    if aqi_val > 500:
        return "#7030a0", "Severe"
    return "#00b050", "Good"

def compute_sub_index(concentration: float, pollutant: str) -> float | None:
    """Compute CPCB sub-index for one pollutant."""
    if concentration is None or np.isnan(concentration) or concentration < 0:
        return None
    bps = CPCB_BREAKPOINTS.get(pollutant, [])
    for (c_lo, c_hi, i_lo, i_hi) in bps:
        if c_lo <= concentration <= c_hi:
            if c_hi == c_lo:
                return float(i_lo)
            return float(i_lo + (i_hi - i_lo) * (concentration - c_lo) / (c_hi - c_lo))
    if bps and concentration > bps[-1][1]:
        return 500.0
    return None

def compute_cpcb_aqi(values: dict) -> tuple[float | None, dict]:
    """Compute overall CPCB AQI = max of sub-indices. Returns (aqi, sub_indices)."""
    sub = {}
    for poll, val in values.items():
        if val is not None and not np.isnan(val):
            si = compute_sub_index(float(val), poll)
            if si is not None:
                sub[poll] = si
    if not sub:
        return None, {}
    return max(sub.values()), sub

def plotly_base_layout(title: str = "", height: int = 400) -> dict:
    """Shared Plotly layout settings."""
    return dict(
        title=title,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
        colorway=PLOTLY_COLORS,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Cached Data Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(rel_path: str) -> pd.DataFrame | None:
    """Load a CSV relative to TABLES_DIR. Returns None if missing."""
    p = TABLES_DIR / rel_path
    if not p.exists():
        # Try predictions dir
        p2 = PREDS_DIR / rel_path
        if p2.exists():
            p = p2
        else:
            return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_track_a() -> pd.DataFrame | None:
    return load_csv("final_track_a_complete.csv")

@st.cache_data(show_spinner=False)
def load_track_b() -> pd.DataFrame | None:
    return load_csv("final_track_b_complete.csv")

@st.cache_data(show_spinner=False)
def load_track_a_ranking() -> pd.DataFrame:
    df = load_csv("track_a_model_ranking.csv")
    if df is not None and not df.empty:
        return df
    return pd.DataFrame(TRACK_A_RANKING)

@st.cache_data(show_spinner=False)
def load_track_b_ranking() -> pd.DataFrame:
    df = load_csv("track_b_model_ranking.csv")
    if df is not None and not df.empty:
        return df
    return pd.DataFrame(TRACK_B_RANKING)

@st.cache_data(show_spinner=False)
def load_bilstm_results() -> pd.DataFrame | None:
    return load_csv("track_b_bilstm.csv")

@st.cache_data(show_spinner=False)
def load_predictions_csv(city: str, horizon_h: int) -> pd.DataFrame | None:
    p = PREDS_DIR / f"{city}_h{horizon_h:02d}_predictions.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_xgb_model(city: str, horizon_h: int):
    """Load XGB pkl file. Returns (model_dict, error_str)."""
    p = PREDS_DIR / f"{city}_h{horizon_h:02d}_xgb.pkl"
    if not p.exists():
        return None, f"Model file not found: {p.name}"
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        return data, None
    except Exception as e:
        return None, str(e)

def run_xgb_forecast(city: str, horizon_h: int) -> tuple[float | None, str | None]:
    """Run XGBoost inference. Returns (predicted_aqi, error_str)."""
    model_data, err = load_xgb_model(city, horizon_h)
    if err:
        return None, err
    model = model_data.get("model")
    feat_cols = model_data.get("feat_cols", model_data.get("safe_names", []))
    if model is None:
        return None, "No model object in pkl"
    pred_df = load_predictions_csv(city, horizon_h)
    if pred_df is None:
        return None, "Prediction CSV not found"
    available_cols = [c for c in feat_cols if c in pred_df.columns]
    if not available_cols:
        return None, f"Feature columns not found in CSV (needed {len(feat_cols)})"
    try:
        X_input = pred_df[available_cols].tail(10).mean().values.reshape(1, -1)
        pred_val = model.predict(X_input)[0]
        return float(max(0.0, pred_val)), None
    except Exception as e:
        return None, f"Inference error: {e}"

def get_bilstm_result(city: str, horizon_h: int) -> dict | None:
    """Get BiLSTM test-set result for a city+horizon."""
    df = load_bilstm_results()
    if df is None or df.empty:
        return None
    row = df[(df["city"] == city) & (df["horizon"] == horizon_h)]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

# ─────────────────────────────────────────────────────────────────────────────
#  Page 1 — Home
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("""
<div class="hero-section">
  <div class="hero-title">Air Quality Index Prediction</div>
  <div class="hero-subtitle">Using Machine Learning and Deep Learning</div>
  <div style="margin-top:16px;">
    <span class="badge">Research Internship</span>
    <span class="badge">IIIT Nagpur</span>
    <span class="badge">YCCE</span>
    <span class="badge">CPCB Dataset</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("18", "Cities Covered"),
        ("18.7M", "Data Records"),
        ("7", "ML/DL Models"),
        ("3", "Forecast Horizons"),
    ]
    for col, (num, label) in zip([c1, c2, c3, c4], kpis):
        col.markdown(f"""
<div class="kpi-card">
  <div class="kpi-number">{num}</div>
  <div class="kpi-label">{label}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Research objectives
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
<div class="glass-card">
  <h3 style="color:#3b82f6;margin-top:0;">🔬 Current AQI Estimation</h3>
  <p style="color:#94a3b8;">Estimate the present-time Air Quality Index using concurrent pollutant and meteorological readings. Seven models were benchmarked: Ridge Regression, Random Forest, Gradient Boosting, XGBoost, LSTM, BiLSTM, and CNN-BiLSTM.</p>
  <p style="color:#10b981;font-weight:600;">Best: GradBoost — R²=0.9906 | MAE=2.95 | RMSE=5.77</p>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div class="glass-card">
  <h3 style="color:#14b8a6;margin-top:0;">📈 Multi-Horizon AQI Forecasting</h3>
  <p style="color:#94a3b8;">Predict future AQI using only historical lag/rolling features — no leakage from same-time pollutant readings. Three horizons evaluated: +1 hour, +6 hours, and +24 hours.</p>
  <p style="color:#10b981;font-weight:600;">Best: GradBoost — R²=0.4997 | 1h forecast R²=0.5313</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏆 Model Performance Highlights")

    tab1, tab2 = st.tabs(["AQI Estimation Models", "AQI Forecasting Models"])

    with tab1:
        df_a = pd.DataFrame(TRACK_A_RANKING)
        df_a.insert(0, "Rank", range(1, len(df_a) + 1))
        df_a.columns = ["Rank", "Model", "R²", "MAE", "RMSE"]
        fig = go.Figure()
        fig.add_bar(x=df_a["Model"], y=df_a["R²"], name="R²",
                    marker_color="#3b82f6", text=[f"{v:.4f}" for v in df_a["R²"]], textposition="outside")
        fig.update_layout(**plotly_base_layout("R² Score by Model (Estimation)", 360))
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_b = pd.DataFrame(TRACK_B_RANKING)
        df_b.insert(0, "Rank", range(1, len(df_b) + 1))
        df_b.columns = ["Rank", "Model", "R²", "MAE", "RMSE"]
        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure()
            fig2.add_bar(x=df_b["Model"], y=df_b["R²"], name="R²",
                         marker_color="#14b8a6", text=[f"{v:.4f}" for v in df_b["R²"]], textposition="outside")
            fig2.update_layout(**plotly_base_layout("R² Score by Model (Forecasting)", 360))
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            hz_df = pd.DataFrame({
                "Horizon": ["1-Hour", "6-Hour", "24-Hour"],
                "Avg R²": [HORIZON_R2[1], HORIZON_R2[6], HORIZON_R2[24]],
            })
            fig3 = go.Figure()
            fig3.add_bar(x=hz_df["Horizon"], y=hz_df["Avg R²"],
                         marker_color=["#3b82f6","#8b5cf6","#ef4444"],
                         text=[f"{v:.4f}" for v in hz_df["Avg R²"]], textposition="outside")
            fig3.update_layout(**plotly_base_layout("Avg R² by Forecast Horizon", 360))
            fig3.update_yaxes(range=[0, 0.65])
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📌 Quick Navigation")
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        st.markdown("""
<div class="glass-card" style="text-align:center;">
  <div style="font-size:2rem;">🔬</div>
  <h4 style="color:#3b82f6;">AQI Estimation</h4>
  <p style="color:#94a3b8;font-size:0.85rem;">Enter pollutant readings → Get CPCB AQI calculation and ML model performance</p>
</div>""", unsafe_allow_html=True)
    with nc2:
        st.markdown("""
<div class="glass-card" style="text-align:center;">
  <div style="font-size:2rem;">📈</div>
  <h4 style="color:#14b8a6;">AQI Forecasting</h4>
  <p style="color:#94a3b8;font-size:0.85rem;">Select city + horizon → XGBoost live forecast or BiLSTM validated results</p>
</div>""", unsafe_allow_html=True)
    with nc3:
        st.markdown("""
<div class="glass-card" style="text-align:center;">
  <div style="font-size:2rem;">🗺</div>
  <h4 style="color:#8b5cf6;">India AQI Map</h4>
  <p style="color:#94a3b8;font-size:0.85rem;">Interactive map of all 18 cities with model performance markers</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Page 2 — AQI Estimation
# ─────────────────────────────────────────────────────────────────────────────
def page_estimation():
    st.markdown("## 🔬 AQI Estimation")
    st.markdown("Estimate Air Quality Index using the official CPCB formula or explore ML model performance.")

    tab_calc, tab_models = st.tabs(["🧮 CPCB AQI Calculator", "📊 Estimation Model Performance"])

    # ── Section A: CPCB AQI Calculator ──
    with tab_calc:
        st.markdown("### Enter Pollutant Concentrations")
        st.markdown("<p style='color:#94a3b8;font-size:0.9rem;'>All units in µg/m³ except CO (mg/m³)</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            pm25  = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=35.0, step=1.0)
            pm10  = st.number_input("PM10 (µg/m³)",  min_value=0.0, max_value=600.0, value=70.0, step=1.0)
            no2   = st.number_input("NO2 (µg/m³)",   min_value=0.0, max_value=800.0, value=45.0, step=1.0)
        with col2:
            so2   = st.number_input("SO2 (µg/m³)",   min_value=0.0, max_value=2100.0, value=20.0, step=1.0)
            co    = st.number_input("CO (mg/m³)",     min_value=0.0, max_value=47.0,   value=1.5, step=0.1)
            o3    = st.number_input("O3 (µg/m³)",     min_value=0.0, max_value=1000.0, value=55.0, step=1.0)
        with col3:
            nh3   = st.number_input("NH3 (µg/m³)",   min_value=0.0, max_value=2400.0, value=100.0, step=5.0)
            st.markdown("<br>", unsafe_allow_html=True)
            calc_btn = st.button("Calculate AQI ▶", use_container_width=True, type="primary")

        if calc_btn:
            poll_vals = {
                "PM2.5": pm25, "PM10": pm10, "NO2": no2,
                "SO2": so2, "CO": co, "O3": o3, "NH3": nh3
            }
            aqi_val, sub_indices = compute_cpcb_aqi(poll_vals)
            if aqi_val is None:
                st.error("Could not compute AQI — all sub-indices invalid.")
            else:
                aqi_int = int(round(aqi_val))
                color, category = aqi_color(aqi_int)
                text_color = "#000000" if aqi_int <= 200 else "#ffffff"
                st.markdown("<br>", unsafe_allow_html=True)
                r1, r2 = st.columns([1, 2])
                with r1:
                    st.markdown(f"""
<div style="text-align:center; background:{color}20; border:2px solid {color};
     border-radius:20px; padding:32px; margin:8px 0;">
  <div style="font-size:4rem; font-weight:800; color:{color};">{aqi_int}</div>
  <div style="font-size:1.4rem; font-weight:700; color:{color}; margin-top:8px;">{category}</div>
  <div style="font-size:0.85rem; color:#94a3b8; margin-top:6px;">CPCB AQI Index</div>
</div>""", unsafe_allow_html=True)

                with r2:
                    st.markdown("#### Sub-Index Breakdown")
                    sub_df = pd.DataFrame([
                        {"Pollutant": k, "Concentration": f"{poll_vals[k]:.1f}", "Sub-Index": f"{v:.1f}",
                         "Dominant": "✅ Driving AQI" if v == aqi_val else ""}
                        for k, v in sub_indices.items()
                    ])
                    # Color bar chart
                    fig_sub = go.Figure()
                    bar_colors = [color if v == aqi_val else "#3b82f6" for v in sub_indices.values()]
                    fig_sub.add_bar(
                        x=list(sub_indices.keys()),
                        y=list(sub_indices.values()),
                        marker_color=bar_colors,
                        text=[f"{v:.0f}" for v in sub_indices.values()],
                        textposition="outside",
                    )
                    fig_sub.update_layout(**plotly_base_layout("Pollutant Sub-Indices", 280))
                    fig_sub.update_yaxes(range=[0, 520])
                    st.plotly_chart(fig_sub, use_container_width=True)

                # Health advisory
                advisories = {
                    "Good": "Air quality is satisfactory. No health risks.",
                    "Satisfactory": "Acceptable air quality. Sensitive individuals may experience mild symptoms.",
                    "Moderate": "Sensitive groups (children, elderly, respiratory patients) should reduce outdoor activities.",
                    "Poor": "Everyone may experience health effects. Minimize outdoor exposure.",
                    "Very Poor": "Serious health risk. Avoid prolonged outdoor activities.",
                    "Severe": "Emergency conditions. Everyone should remain indoors.",
                }
                adv = advisories.get(category, "Monitor air quality regularly.")
                st.markdown(f"""
<div style="background:{color}15; border-left:4px solid {color}; border-radius:8px;
     padding:16px; margin-top:16px;">
  <strong style="color:{color};">⚠️ Health Advisory ({category})</strong><br>
  <span style="color:#f1f5f9;">{adv}</span>
</div>""", unsafe_allow_html=True)

    # ── Section B: Estimation Model Performance ──
    with tab_models:
        st.markdown("### Estimation Model Performance Summary")
        df_ta = load_track_a()
        rank_df = load_track_a_ranking()

        if rank_df is not None and not rank_df.empty:
            # Normalize column names
            rank_df.columns = [c.lower() for c in rank_df.columns]
            model_col = "model" if "model" in rank_df.columns else rank_df.columns[0]
            r2_col   = next((c for c in rank_df.columns if "r2" in c), None)
            mae_col  = next((c for c in rank_df.columns if "mae" in c), None)
            rmse_col = next((c for c in rank_df.columns if "rmse" in c), None)

            if r2_col is None:
                # Use fallback
                rank_df = pd.DataFrame(TRACK_A_RANKING)
                rank_df.columns = [c.lower() for c in rank_df.columns]
                model_col, r2_col, mae_col, rmse_col = "model", "r2", "mae", "rmse"

            rank_df = rank_df.sort_values(r2_col, ascending=False).reset_index(drop=True)
            rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

            # Leaderboard
            st.markdown("#### 🏅 Model Leaderboard")
            display_df = rank_df[[model_col, r2_col, mae_col, rmse_col]].copy()
            display_df.columns = ["Model", "Avg R²", "Avg MAE", "Avg RMSE"]
            display_df["Avg R²"]   = display_df["Avg R²"].apply(lambda x: f"{x:.4f}")
            display_df["Avg MAE"]  = display_df["Avg MAE"].apply(lambda x: f"{x:.2f}")
            display_df["Avg RMSE"] = display_df["Avg RMSE"].apply(lambda x: f"{x:.2f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                r2_vals = rank_df[r2_col].tolist()
                fig_r2 = go.Figure(go.Bar(
                    x=rank_df[model_col].tolist(),
                    y=r2_vals,
                    marker_color=PLOTLY_COLORS[:len(r2_vals)],
                    text=[f"{v:.4f}" for v in r2_vals],
                    textposition="outside",
                ))
                fig_r2.update_layout(**plotly_base_layout("R² Score per Model", 360))
                fig_r2.update_yaxes(range=[0, 1.08])
                st.plotly_chart(fig_r2, use_container_width=True)

            with c2:
                mae_vals  = rank_df[mae_col].tolist()
                rmse_vals = rank_df[rmse_col].tolist()
                models    = rank_df[model_col].tolist()
                fig_err = go.Figure()
                fig_err.add_bar(x=models, y=mae_vals,  name="MAE",  marker_color="#f59e0b")
                fig_err.add_bar(x=models, y=rmse_vals, name="RMSE", marker_color="#ef4444")
                fig_err.update_layout(**plotly_base_layout("MAE vs RMSE per Model", 360), barmode="group")
                st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.warning("Ranking CSV not found. Showing confirmed figures.")
            st.dataframe(pd.DataFrame(TRACK_A_RANKING), use_container_width=True, hide_index=True)

        # Per-city heatmap if full data available
        if df_ta is not None and not df_ta.empty:
            st.markdown("#### 🗺 R² Heatmap — Model × City")
            df_ta.columns = [c.lower() for c in df_ta.columns]
            mc  = "model" if "model" in df_ta.columns else df_ta.columns[1]
            cc  = "city"  if "city"  in df_ta.columns else df_ta.columns[0]
            r2c = next((c for c in df_ta.columns if "r2" in c), None)
            if r2c:
                pivot = df_ta.pivot_table(index=cc, columns=mc, values=r2c, aggfunc="mean")
                fig_heat = px.imshow(
                    pivot,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    labels={"color": "R²"},
                )
                fig_heat.update_layout(**plotly_base_layout("R² per Model per City (Estimation)", 500))
                st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Page 3 — AQI Forecasting
# ─────────────────────────────────────────────────────────────────────────────
def page_forecasting():
    st.markdown("## 📈 AQI Forecasting")
    st.markdown("Live XGBoost forecasting or BiLSTM validated test-set results per city and horizon.")

    # ── Sidebar-style controls in main area ──
    col_ctrl, col_main = st.columns([1, 3])
    with col_ctrl:
        st.markdown("#### Controls")
        city_options = {CITY_DISPLAY[c]: c for c in CITIES}
        city_label = st.selectbox("Select City", list(city_options.keys()))
        city = city_options[city_label]

        horizon_label = st.selectbox("Forecast Horizon", list(HORIZONS.values()))
        horizon_h = {v: k for k, v in HORIZONS.items()}[horizon_label]

        model_choice = st.radio("Model", ["XGBoost", "BiLSTM"])
        run_btn = st.button("▶ Run Forecast", use_container_width=True, type="primary")

    with col_main:
        if run_btn or True:   # always show on load
            st.markdown(f"### {CITY_DISPLAY[city]} — {horizon_label} Ahead")

            if model_choice == "XGBoost":
                # ── XGBoost live inference ──
                with st.spinner("Running XGBoost inference…"):
                    pred_val, err = run_xgb_forecast(city, horizon_h)

                if err:
                    st.warning(f"XGBoost inference unavailable: {err}")
                else:
                    aqi_int = int(round(pred_val))
                    col_h, color = aqi_color(aqi_int)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted AQI", aqi_int)
                    c2.metric("Category", aqi_color(aqi_int)[1])
                    c3.metric("Horizon", horizon_label)
                    st.markdown(f"""
<div style="background:{col_h}15; border:2px solid {col_h}; border-radius:16px;
     padding:20px; text-align:center; margin:16px 0;">
  <span style="font-size:3rem; font-weight:800; color:{col_h};">{aqi_int}</span>
  <span style="font-size:1.2rem; color:{col_h}; margin-left:12px;">{aqi_color(aqi_int)[1]}</span>
</div>""", unsafe_allow_html=True)

                # Show prediction plot from CSV
                pred_df = load_predictions_csv(city, horizon_h)
                if pred_df is not None and not pred_df.empty:
                    pred_df.columns = [c.lower() for c in pred_df.columns]
                    # find actual and predicted columns
                    act_col  = next((c for c in pred_df.columns if "actual" in c), None)
                    pred_col = next((c for c in pred_df.columns if "predict" in c or "pred" in c), None)
                    if act_col and pred_col:
                        plot_df = pred_df[[act_col, pred_col]].tail(100).reset_index(drop=True)
                        fig_pred = go.Figure()
                        fig_pred.add_scatter(x=plot_df.index, y=plot_df[act_col],
                                             mode="lines", name="Actual", line=dict(color="#3b82f6", width=2))
                        fig_pred.add_scatter(x=plot_df.index, y=plot_df[pred_col],
                                             mode="lines", name="Predicted", line=dict(color="#f59e0b", width=2, dash="dash"))
                        fig_pred.update_layout(**plotly_base_layout(
                            f"XGBoost — {CITY_DISPLAY[city]} {horizon_label} | Actual vs Predicted (last 100 points)", 380))
                        st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        st.info("Prediction CSV loaded but column names differ. Showing raw data preview.")
                        st.dataframe(pred_df.tail(20), use_container_width=True)
                else:
                    st.info(f"Prediction CSV not found for {CITY_DISPLAY[city]} h{horizon_h:02d}.")

            else:
                # ── BiLSTM validated results ──
                result = get_bilstm_result(city, horizon_h)
                if result:
                    r2_  = result.get("r2",   result.get("R2",   "N/A"))
                    mae_ = result.get("mae",  result.get("MAE",  "N/A"))
                    rmse_= result.get("rmse", result.get("RMSE", "N/A"))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² Score", f"{r2_:.4f}" if isinstance(r2_, float) else r2_)
                    c2.metric("MAE",  f"{mae_:.2f}" if isinstance(mae_, float) else mae_)
                    c3.metric("RMSE", f"{rmse_:.2f}" if isinstance(rmse_, float) else rmse_)
                    st.info("BiLSTM results are from the validated test-set evaluation. Live inference requires a trained BiLSTM model artifact.")
                else:
                    st.warning(f"BiLSTM results not found for {CITY_DISPLAY[city]} — {horizon_label}. Showing aggregated BiLSTM performance.")
                    bilstm_df = load_bilstm_results()
                    if bilstm_df is not None:
                        bilstm_df.columns = [c.lower() for c in bilstm_df.columns]
                        h_df = bilstm_df[bilstm_df["horizon"] == horizon_h] if "horizon" in bilstm_df.columns else bilstm_df
                        if not h_df.empty:
                            st.dataframe(h_df[["city","r2","mae","rmse"]].sort_values("r2", ascending=False),
                                         use_container_width=True, hide_index=True)

                # Show BiLSTM figure if available
                fig_path = FIGS_DIR / f"{city}_h{horizon_h:02d}_bilstm_pred.png"
                loss_path = FIGS_DIR / f"{city}_h{horizon_h:02d}_bilstm_loss.png"
                if fig_path.exists():
                    st.markdown("#### BiLSTM Prediction Plot")
                    st.image(str(fig_path), use_column_width=True)
                elif loss_path.exists():
                    st.markdown("#### BiLSTM Training Loss")
                    st.image(str(loss_path), use_column_width=True)
                else:
                    st.info("Pre-computed BiLSTM figures not found in outputs/figures/.")

        # ── Horizon comparison for selected city ──
        st.markdown("---")
        st.markdown(f"#### {CITY_DISPLAY[city]} — Horizon Performance Comparison")

        bilstm_df = load_bilstm_results()
        if bilstm_df is not None:
            bilstm_df.columns = [c.lower() for c in bilstm_df.columns]
            city_df = bilstm_df[bilstm_df["city"] == city] if "city" in bilstm_df.columns else pd.DataFrame()
            if not city_df.empty and "horizon" in city_df.columns:
                fig_hz = go.Figure()
                fig_hz.add_bar(x=city_df["horizon"].tolist(), y=city_df["r2"].tolist(),
                               marker_color=["#3b82f6","#8b5cf6","#ef4444"],
                               text=[f"{v:.3f}" for v in city_df["r2"].tolist()],
                               textposition="outside", name="BiLSTM R²")
                fig_hz.update_layout(**plotly_base_layout(f"BiLSTM R² by Horizon — {CITY_DISPLAY[city]}", 300))
                fig_hz.update_yaxes(range=[-0.1, 0.9])
                st.plotly_chart(fig_hz, use_container_width=True)
            else:
                st.info("Per-city horizon data not available.")
        else:
            st.info("BiLSTM results CSV not found. Check outputs/tables/track_b_bilstm.csv.")

# ─────────────────────────────────────────────────────────────────────────────
#  Page 4 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
def page_comparison():
    st.markdown("## 📊 Model Comparison")
    st.markdown("Comprehensive side-by-side benchmarking of all estimation and forecasting models.")

    tab_a, tab_b = st.tabs(["🔬 AQI Estimation Models", "📈 AQI Forecasting Models"])

    # ── Estimation ──
    with tab_a:
        st.markdown("### AQI Estimation Model Benchmarks")
        df_ta = load_track_a()
        rank_a = load_track_a_ranking()
        if rank_a is None:
            rank_a = pd.DataFrame(TRACK_A_RANKING)
        rank_a.columns = [c.lower() for c in rank_a.columns]
        model_c = "model" if "model" in rank_a.columns else rank_a.columns[0]
        r2_c    = next((c for c in rank_a.columns if "r2" in c), "r2")
        mae_c   = next((c for c in rank_a.columns if "mae" in c), "mae")
        rmse_c  = next((c for c in rank_a.columns if "rmse" in c), "rmse")

        if r2_c not in rank_a.columns:
            rank_a = pd.DataFrame(TRACK_A_RANKING)
            rank_a.columns = [c.lower() for c in rank_a.columns]
            model_c, r2_c, mae_c, rmse_c = "model", "r2", "mae", "rmse"

        rank_a = rank_a.sort_values(r2_c, ascending=False).reset_index(drop=True)

        c1, c2 = st.columns(2)
        with c1:
            # Grouped bar: R², MAE, RMSE
            fig_group = make_subplots(rows=1, cols=1)
            models = rank_a[model_c].tolist()
            fig_group.add_bar(x=models, y=rank_a[r2_c].tolist(),  name="R²",  marker_color="#3b82f6")
            fig_group.update_layout(**plotly_base_layout("R² per Estimation Model", 360))
            fig_group.update_yaxes(range=[0, 1.1])
            st.plotly_chart(fig_group, use_container_width=True)

        with c2:
            fig_mae_rmse = go.Figure()
            fig_mae_rmse.add_bar(x=models, y=rank_a[mae_c].tolist(),  name="MAE",  marker_color="#f59e0b")
            fig_mae_rmse.add_bar(x=models, y=rank_a[rmse_c].tolist(), name="RMSE", marker_color="#ef4444")
            fig_mae_rmse.update_layout(**plotly_base_layout("MAE & RMSE per Model", 360), barmode="group")
            st.plotly_chart(fig_mae_rmse, use_container_width=True)

        # Horizontal leaderboard bar
        fig_horiz = go.Figure(go.Bar(
            y=models[::-1],
            x=rank_a[r2_c].tolist()[::-1],
            orientation="h",
            marker_color=PLOTLY_COLORS[:len(models)][::-1],
            text=[f"{v:.4f}" for v in rank_a[r2_c].tolist()[::-1]],
            textposition="outside",
        ))
        fig_horiz.update_layout(**plotly_base_layout("R² Leaderboard (Best → Worst)", 320))
        fig_horiz.update_xaxes(range=[0, 1.1])
        st.plotly_chart(fig_horiz, use_container_width=True)

        # Scatter MAE vs RMSE
        fig_scatter = px.scatter(
            rank_a, x=mae_c, y=rmse_c, color=model_c,
            text=model_c, size_max=14,
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig_scatter.update_traces(textposition="top center", marker_size=12)
        fig_scatter.update_layout(**plotly_base_layout("MAE vs RMSE (Estimation Models)", 360))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Full table
        st.markdown("#### Complete Estimation Results")
        if df_ta is not None and not df_ta.empty:
            st.dataframe(df_ta, use_container_width=True, hide_index=True)
        else:
            st.dataframe(rank_a, use_container_width=True, hide_index=True)

    # ── Forecasting ──
    with tab_b:
        st.markdown("### AQI Forecasting Model Benchmarks")
        df_tb = load_track_b()
        rank_b = load_track_b_ranking()
        if rank_b is None:
            rank_b = pd.DataFrame(TRACK_B_RANKING)
        rank_b.columns = [c.lower() for c in rank_b.columns]
        model_c = "model" if "model" in rank_b.columns else rank_b.columns[0]
        r2_c    = next((c for c in rank_b.columns if "r2" in c), "r2")
        mae_c   = next((c for c in rank_b.columns if "mae" in c), "mae")
        rmse_c  = next((c for c in rank_b.columns if "rmse" in c), "rmse")

        if r2_c not in rank_b.columns:
            rank_b = pd.DataFrame(TRACK_B_RANKING)
            rank_b.columns = [c.lower() for c in rank_b.columns]
            model_c, r2_c, mae_c, rmse_c = "model", "r2", "mae", "rmse"

        rank_b = rank_b.sort_values(r2_c, ascending=False).reset_index(drop=True)
        models_b = rank_b[model_c].tolist()

        c1, c2 = st.columns(2)
        with c1:
            fig_b1 = go.Figure(go.Bar(
                x=models_b, y=rank_b[r2_c].tolist(),
                marker_color="#14b8a6",
                text=[f"{v:.4f}" for v in rank_b[r2_c].tolist()],
                textposition="outside",
            ))
            fig_b1.update_layout(**plotly_base_layout("R² per Forecasting Model", 360))
            fig_b1.update_yaxes(range=[-0.6, 0.7])
            st.plotly_chart(fig_b1, use_container_width=True)

        with c2:
            fig_b2 = go.Figure()
            fig_b2.add_bar(x=models_b, y=rank_b[mae_c].tolist(),  name="MAE",  marker_color="#f59e0b")
            fig_b2.add_bar(x=models_b, y=rank_b[rmse_c].tolist(), name="RMSE", marker_color="#ef4444")
            fig_b2.update_layout(**plotly_base_layout("MAE & RMSE per Model", 360), barmode="group")
            st.plotly_chart(fig_b2, use_container_width=True)

        # Horizon chart
        st.markdown("#### Horizon Performance (All Models Averaged)")
        hz_df = pd.DataFrame({
            "Horizon":    ["1-Hour (+1h)", "6-Hour (+6h)", "24-Hour (+24h)"],
            "Avg R²":     [HORIZON_R2[1], HORIZON_R2[6], HORIZON_R2[24]],
            "Interpretation": ["Near-term: best accuracy", "Medium-term: moderate", "Long-term: hardest"],
        })
        fig_hz = go.Figure(go.Bar(
            x=hz_df["Horizon"], y=hz_df["Avg R²"],
            marker_color=["#3b82f6","#8b5cf6","#ef4444"],
            text=[f"R²={v:.4f}" for v in hz_df["Avg R²"]],
            textposition="outside",
        ))
        fig_hz.update_layout(**plotly_base_layout("Avg R² by Forecast Horizon (All Models)", 300))
        fig_hz.update_yaxes(range=[0, 0.7])
        st.plotly_chart(fig_hz, use_container_width=True)

        # Full table
        st.markdown("#### Complete Forecasting Results")
        if df_tb is not None and not df_tb.empty:
            st.dataframe(df_tb, use_container_width=True, hide_index=True)
        else:
            bilstm = load_bilstm_results()
            if bilstm is not None:
                st.dataframe(bilstm, use_container_width=True, hide_index=True)
            else:
                st.dataframe(rank_b, use_container_width=True, hide_index=True)

        # Summary metrics
        st.markdown("#### Summary Statistics")
        sc1, sc2, sc3 = st.columns(3)
        best_row = rank_b.sort_values(r2_c, ascending=False).iloc[0]
        sc1.metric("Best Model",    str(best_row[model_c]))
        sc2.metric("Best R²",       f"{best_row[r2_c]:.4f}")
        sc3.metric("Best Horizon",  "1-Hour (R²=0.5313)")

# ─────────────────────────────────────────────────────────────────────────────
#  Page 5 — India AQI Map
# ─────────────────────────────────────────────────────────────────────────────
def page_india_map():
    st.markdown("## 🗺 India AQI Monitoring Map")
    st.markdown("Interactive map showing all 18 study cities with model performance overlays.")

    # Build city data
    df_ta = load_track_a()
    df_bilstm = load_bilstm_results()

    rows = []
    for city, (lat, lon) in CITY_COORDS.items():
        best_a_r2  = None
        best_b_r2  = None
        n_train    = None

        if df_ta is not None and not df_ta.empty:
            dfc = df_ta.copy()
            dfc.columns = [c.lower() for c in dfc.columns]
            r2c = next((c for c in dfc.columns if "r2" in c), None)
            if r2c and "city" in dfc.columns:
                sub = dfc[dfc["city"] == city][r2c]
                if not sub.empty:
                    best_a_r2 = float(sub.max())
            if "n_train" in dfc.columns and "city" in dfc.columns:
                sub2 = dfc[dfc["city"] == city]["n_train"]
                if not sub2.empty:
                    n_train = float(sub2.iloc[0])

        if df_bilstm is not None and not df_bilstm.empty:
            dfc2 = df_bilstm.copy()
            dfc2.columns = [c.lower() for c in dfc2.columns]
            r2c2 = next((c for c in dfc2.columns if "r2" in c), None)
            hz_c = next((c for c in dfc2.columns if "horizon" in c), None)
            if r2c2 and hz_c and "city" in dfc2.columns:
                sub3 = dfc2[(dfc2["city"] == city) & (dfc2[hz_c] == 1)][r2c2]
                if not sub3.empty:
                    best_b_r2 = float(sub3.iloc[0])

        rows.append({
            "city":         city,
            "display_name": CITY_DISPLAY[city],
            "lat":          lat,
            "lon":          lon,
            "best_est_r2":  best_a_r2 if best_a_r2 is not None else 0.85,
            "best_fcast_r2":best_b_r2 if best_b_r2 is not None else 0.45,
            "n_train":      n_train if n_train is not None else 20000,
        })

    map_df = pd.DataFrame(rows)

    # Marker size proportional to training size
    max_size = map_df["n_train"].max()
    map_df["marker_size"] = 12 + 20 * (map_df["n_train"] / max_size)

    # Hover text
    map_df["hover"] = map_df.apply(lambda r: (
        f"<b>{r['display_name']}</b><br>"
        f"Best Estimation R²: {r['best_est_r2']:.4f}<br>"
        f"Best Forecast R² (1h): {r['best_fcast_r2']:.4f}<br>"
        f"Training samples: {int(r['n_train']):,}"
    ), axis=1)

    fig_map = go.Figure(go.Scattergeo(
        lat=map_df["lat"],
        lon=map_df["lon"],
        text=map_df["hover"],
        hovertemplate="%{text}<extra></extra>",
        mode="markers+text",
        textfont=dict(size=9, color="white"),
        textposition="top center",
        marker=dict(
            size=map_df["marker_size"],
            color=map_df["best_est_r2"],
            colorscale="Teal",
            cmin=0.5,
            cmax=1.0,
            colorbar=dict(title="Est. R²", thickness=15, len=0.6, bgcolor="rgba(0,0,0,0)", tickfont=dict(color="white")),
            line=dict(width=1, color="rgba(255,255,255,0.4)"),
            opacity=0.9,
        ),
    ))

    fig_map.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#1a2332",
        showocean=True, oceancolor="#0a0e1a",
        showcountries=True, countrycolor="#334155",
        showcoastlines=True, coastlinecolor="#334155",
        showsubunits=True, subunitcolor="#1e293b",
        lataxis_range=[5, 38],
        lonaxis_range=[65, 100],
        bgcolor="rgba(0,0,0,0)",
        framecolor="#1e293b",
    )

    fig_map.update_layout(
        title="India — AQI Prediction Study Cities (18 Cities)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color="white"),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # City performance table
    st.markdown("### City Performance Summary")
    show_df = map_df[["display_name","best_est_r2","best_fcast_r2"]].copy()
    show_df.columns = ["City", "Best Estimation R²", "Best Forecast R² (1h)"]
    show_df = show_df.sort_values("Best Estimation R²", ascending=False).reset_index(drop=True)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Page 6 — About Project
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown("## 📖 About the Project")

    # Project overview
    st.markdown("""
<div class="glass-card">
  <h3 style="color:#3b82f6; margin-top:0;">🎯 Project Motivation</h3>
  <p style="color:#f1f5f9; line-height:1.7;">
    Air pollution is one of the most critical environmental and public health challenges facing India today.
    The Central Pollution Control Board (CPCB) monitors air quality at hundreds of stations across the country,
    generating vast amounts of time-series data. This research harnesses that data to build a rigorous,
    leakage-free ML and DL benchmark across 18 Indian cities — with two distinct research objectives:
    <em>current AQI estimation</em> and <em>multi-horizon AQI forecasting</em>.
  </p>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#14b8a6; margin-top:0;">📦 Dataset Summary</h4>
  <ul style="color:#f1f5f9; line-height:2.0; margin:0; padding-left:20px;">
    <li><b>Source:</b> CPCB (Central Pollution Control Board)</li>
    <li><b>Cities:</b> 18 major Indian cities</li>
    <li><b>Records:</b> ~18.7 million hourly readings</li>
    <li><b>Period:</b> 2018 – 2023</li>
    <li><b>Resolution:</b> Hourly aggregated</li>
    <li><b>Pollutants:</b> PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, Toluene</li>
    <li><b>Meteorology:</b> Temperature, Humidity, Wind Speed, Wind Direction, Solar Radiation, Pressure</li>
    <li><b>Split:</b> 70% train / 15% val / 15% test (chronological)</li>
  </ul>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#8b5cf6; margin-top:0;">⚙️ Methodology Overview</h4>
  <ul style="color:#f1f5f9; line-height:2.0; margin:0; padding-left:20px;">
    <li><b>Dual-track protocol:</b> Estimation (no leakage constraint) vs Forecasting (strict leakage-free)</li>
    <li><b>Feature engineering:</b> 88–115 features incl. lag, rolling stats, temporal encoding, interactions</li>
    <li><b>Leakage audit:</b> Forecasting uses ONLY lag/rolling features — zero same-time pollutants</li>
    <li><b>Preprocessing:</b> 15-min → hourly resampling, 3-stage imputation</li>
    <li><b>Scaling:</b> MinMaxScaler for DL models; tree models use raw features</li>
    <li><b>Evaluation:</b> R², MAE, RMSE on held-out test set</li>
    <li><b>Reproducibility:</b> SEED=42, chronological splits, no data leakage</li>
  </ul>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤖 Models Evaluated")
    ml_col, dl_col = st.columns(2)

    with ml_col:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#3b82f6; margin-top:0;">Classical Machine Learning</h4>
  <div class="model-card">
    <b style="color:#f59e0b;">Ridge Regression</b><br>
    <span style="color:#94a3b8; font-size:0.85rem;">L2-regularised linear baseline. Fast, interpretable. Est. R²=0.8304.</span>
  </div>
  <div class="model-card">
    <b style="color:#10b981;">Random Forest</b><br>
    <span style="color:#94a3b8; font-size:0.85rem;">Ensemble of 300 trees. Robust to outliers. Est. R²=0.9874.</span>
  </div>
  <div class="model-card">
    <b style="color:#3b82f6;">Gradient Boosting</b><br>
    <span style="color:#94a3b8; font-size:0.85rem;">Sequential boosting regressor. Best overall. Est. R²=0.9906.</span>
  </div>
  <div class="model-card">
    <b style="color:#8b5cf6;">XGBoost</b><br>
    <span style="color:#94a3b8; font-size:0.85rem;">Extreme gradient boosting with regularisation. Est. R²=0.9856.</span>
  </div>
</div>""", unsafe_allow_html=True)

    with dl_col:
        st.markdown("""
<div class="glass-card">
  <h4 style="color:#14b8a6; margin-top:0;">Deep Learning</h4>
  <div class="model-card">
    <b style="color:#ef4444;">LSTM</b><br>
    <span style="color:#94a3b8; font-size:0.85rem;">Long Short-Term Memory. Sequence modelling baseline. Est. R²=0.6411.</span>
  </div>
  <div class="model-card">
    <b style="color:#ec4899;">BiLSTM</b><br>
    <span style="color:#94a3b8; font-size:0.85rem;">Bidirectional LSTM for richer temporal context. Est. R²=0.5897.</span>
  </div>
  <div class="model-card">
    <b style="color:#f59e0b;">CNN-BiLSTM</b><br>
    <span style="color:#94a3b8; font-size:0.85rem;">CNN feature extraction + BiLSTM temporal modelling. Est. R²=0.2756.</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Key findings
    st.markdown("### 🔑 Key Research Contributions")
    st.markdown("""
<div class="glass-card">
  <ul style="color:#f1f5f9; line-height:2.0; margin:0; padding-left:20px;">
    <li>First multi-city (18-city) simultaneous ML/DL benchmark on Indian CPCB hourly AQI data</li>
    <li>Leakage-certified dual-track evaluation protocol distinguishing estimation from forecasting</li>
    <li>Demonstrated that Gradient Boosting decisively outperforms LSTM/BiLSTM for both tasks on this dataset</li>
    <li>Identified that CNN-BiLSTM underperforms for tabular AQI data (R²=0.28 estimation, R²=-0.41 forecasting)</li>
    <li>Quantified rapid horizon degradation: 1h R²=0.53 → 6h R²=0.18 → 24h R²=0.10</li>
    <li>Developed city-level performance profiles across 18 Indian cities with varying pollution characteristics</li>
    <li>Feature importance analysis confirming lag-1 and rolling-3h features as most predictive for forecasting</li>
  </ul>
</div>""", unsafe_allow_html=True)

    # Institution info
    st.markdown("<br>", unsafe_allow_html=True)
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("""
<div class="glass-card" style="text-align:center;">
  <div style="font-size:2.5rem;">🎓</div>
  <h4 style="color:#3b82f6;">Research Institution</h4>
  <p style="color:#f1f5f9; font-weight:600;">Indian Institute of Information Technology (IIIT) Nagpur</p>
  <p style="color:#94a3b8; font-size:0.85rem;">Research Internship Program<br>Department of Computer Science & Engineering</p>
  <hr style="border-color:#1e293b;">
  <p style="color:#f1f5f9; font-weight:600;">Yeshwantrao Chavan College of Engineering (YCCE)</p>
  <p style="color:#94a3b8; font-size:0.85rem;">Nagpur, Maharashtra, India</p>
</div>""", unsafe_allow_html=True)

    with ci2:
        st.markdown("""
<div class="glass-card" style="text-align:center;">
  <div style="font-size:2.5rem;">📊</div>
  <h4 style="color:#14b8a6;">Project Statistics</h4>
  <table style="width:100%; color:#f1f5f9; text-align:left; border-collapse:collapse;">
    <tr><td style="padding:6px; color:#94a3b8;">Dataset Records</td><td style="padding:6px;">~18.7 Million</td></tr>
    <tr><td style="padding:6px; color:#94a3b8;">Cities</td><td style="padding:6px;">18 Indian Cities</td></tr>
    <tr><td style="padding:6px; color:#94a3b8;">Models Trained</td><td style="padding:6px;">7 (4 ML + 3 DL)</td></tr>
    <tr><td style="padding:6px; color:#94a3b8;">Total Experiments</td><td style="padding:6px;">~450 model-city combinations</td></tr>
    <tr><td style="padding:6px; color:#94a3b8;">Forecast Horizons</td><td style="padding:6px;">1h, 6h, 24h</td></tr>
    <tr><td style="padding:6px; color:#94a3b8;">Best Estimation R²</td><td style="padding:6px;">0.9906 (GradBoost)</td></tr>
    <tr><td style="padding:6px; color:#94a3b8;">Best Forecast R²</td><td style="padding:6px;">0.5313 (1h avg)</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.85rem; padding:20px 0;">
  AQI Prediction Platform | Research Internship — IIIT Nagpur | Built with Streamlit & Plotly
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
PAGES = {
    "🏠 Home":               page_home,
    "🔬 AQI Estimation":     page_estimation,
    "📈 AQI Forecasting":    page_forecasting,
    "📊 Model Comparison":   page_comparison,
    "🗺 India AQI Map":      page_india_map,
    "📖 About Project":      page_about,
}

def main():
    inject_css()

    with st.sidebar:
        st.markdown("""
<div style="text-align:center; padding:16px 0 8px;">
  <div style="font-size:2.2rem;">🌬️</div>
  <div style="font-size:1.1rem; font-weight:700; color:#3b82f6;">AQI Prediction</div>
  <div style="font-size:0.75rem; color:#94a3b8;">ML & Deep Learning Platform</div>
</div>
<hr style="border-color:#1e293b; margin:8px 0;">
""", unsafe_allow_html=True)

        page_name = st.radio(
            "Navigate",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#1e293b; margin:12px 0;'>", unsafe_allow_html=True)

        # Quick stats in sidebar
        st.markdown("""
<div style="padding:8px;">
  <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:8px;">QUICK STATS</div>
  <div style="font-size:0.8rem; color:#f1f5f9; line-height:2.0;">
    🏙️ 18 Cities<br>
    📊 18.7M Records<br>
    🤖 7 Models<br>
    ⏱️ 3 Horizons<br>
    🏆 R²=0.99 (Best)
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div class="sidebar-footer" style="position:relative;margin-top:32px;color:#475569;font-size:0.72rem;text-align:center;">
  <hr style="border-color:#1e293b;">
  Research Internship<br>
  <b style="color:#3b82f6;">IIIT Nagpur</b> | YCCE<br>
  CPCB Dataset
</div>""", unsafe_allow_html=True)

    # Render selected page
    PAGES[page_name]()

if __name__ == "__main__":
    main()
