"""
AQI Prediction Platform
=======================
Research Internship · IIIT Nagpur
CPCB National Air Quality Dataset — 18 Cities · 7 Models
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="AQI Prediction Platform",
    page_icon="🌬️",
    initial_sidebar_state="expanded",
)

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
TABLES      = ROOT / "outputs" / "tables"
PRED_DIR    = ROOT / "outputs" / "predictions"
MODELS_B    = ROOT / "models" / "track_b"

# ─── Constants ───────────────────────────────────────────────────────────────
CITIES = [
    "Ahmedabad", "Chennai", "Delhi_NCR", "GandhiNagar", "Hyderabad",
    "Indore", "Jaipur", "Jodhpur", "Mumbai", "Mumbai_suburbs",
    "Nagpur", "Pune", "Singrauli", "Surat", "Thane", "Vapi",
    "bhopal", "vishakhapattanam",
]

CITY_COORDS = {
    "Ahmedabad":         (23.03, 72.58),
    "Chennai":           (13.08, 80.27),
    "Delhi_NCR":         (28.67, 77.22),
    "GandhiNagar":       (23.22, 72.65),
    "Hyderabad":         (17.38, 78.49),
    "Indore":            (22.72, 75.86),
    "Jaipur":            (26.91, 75.79),
    "Jodhpur":           (26.30, 73.02),
    "Mumbai":            (19.08, 72.88),
    "Mumbai_suburbs":    (19.18, 72.98),
    "Nagpur":            (21.15, 79.09),
    "Pune":              (18.52, 73.86),
    "Singrauli":         (24.20, 82.67),
    "Surat":             (21.17, 72.83),
    "Thane":             (19.22, 72.97),
    "Vapi":              (20.37, 72.91),
    "bhopal":            (23.26, 77.41),
    "vishakhapattanam":  (17.69, 83.22),
}

HORIZONS = [1, 6, 24]
HORIZON_LABELS = {1: "+1h", 6: "+6h", 24: "+24h"}

MODEL_NAMES_A = ["Ridge", "RandomForest", "GradBoost", "XGBoost", "LSTM", "BiLSTM", "CNN-BiLSTM"]
MODEL_NAMES_B = ["XGB", "GBR", "RF", "LSTM", "BiLSTM", "CNN-BiLSTM"]

AQI_CATEGORIES = [
    (0,   50,  "Good",         "#00e400"),
    (51,  100, "Satisfactory", "#92d050"),
    (101, 200, "Moderate",     "#ffff00"),
    (201, 300, "Poor",         "#ff7e00"),
    (301, 400, "Very Poor",    "#ff0000"),
    (401, 500, "Severe",       "#7e0023"),
]

# CPCB breakpoints: (pollutant, [(BPlo, BPhi, Ilo, Ihi), ...])
CPCB_BP = {
    "PM2.5": [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
        (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500),
    ],
    "CO": [
        (0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
        (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500),
    ],
    "O3": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
        (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
        (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2400, 401, 500),
    ],
}

# ─── CSS Styling ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1e3a5f 100%);
}
section[data-testid="stSidebar"] * { color: #e0f0ff !important; }

.kpi-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2744 100%);
    border: 1px solid #00d4aa33;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    color: white;
}
.kpi-card .kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00d4aa;
}
.kpi-card .kpi-label {
    font-size: 0.85rem;
    color: #a0c0e0;
    margin-top: 4px;
}

.hero-container {
    background: linear-gradient(135deg, #0d1b2a 0%, #1e3a5f 50%, #0d2744 100%);
    border-radius: 16px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    border: 1px solid #00d4aa44;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: #00d4aa;
    margin-bottom: 16px;
}
.hero-caption {
    font-size: 0.95rem;
    color: #7090a0;
}

.aqi-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    color: #000;
}

.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e3a5f;
    border-left: 4px solid #00d4aa;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

.info-box {
    background: #f0f8ff;
    border: 1px solid #b0d4f0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #1e3a5f;
    margin: 8px 0;
}

.model-card {
    background: white;
    border: 1px solid #e0e8f0;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(30,58,95,0.08);
}

.forecast-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.forecast-table th {
    background: #1e3a5f;
    color: white;
    padding: 8px 12px;
    text-align: center;
}
.forecast-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #e8f0f8;
    text-align: center;
}
.forecast-table tr:hover { background: #f5f9ff; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────────────
def get_aqi_category(aqi: float) -> Tuple[str, str]:
    for lo, hi, cat, col in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return cat, col
    return "Severe", "#7e0023"

def compute_sub_index(pollutant: str, concentration: float) -> Optional[float]:
    if pollutant not in CPCB_BP or concentration < 0:
        return None
    for bp_lo, bp_hi, i_lo, i_hi in CPCB_BP[pollutant]:
        if bp_lo <= concentration <= bp_hi:
            si = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
            return round(si, 1)
    return 500.0

def compute_cpcb_aqi(pm25, pm10, no2, so2, co, o3, nh3) -> Optional[float]:
    inputs = {
        "PM2.5": pm25, "PM10": pm10, "NO2": no2,
        "SO2": so2, "CO": co, "O3": o3, "NH3": nh3,
    }
    sub_indices = []
    for poll, val in inputs.items():
        if val is not None and val >= 0:
            si = compute_sub_index(poll, val)
            if si is not None:
                sub_indices.append(si)
    if not sub_indices:
        return None
    return round(max(sub_indices), 1)

@st.cache_data(show_spinner=False)
def load_track_a_csv() -> Optional[pd.DataFrame]:
    path = TABLES / "final_track_a_complete.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_track_b_csv(model_stem: str) -> Optional[pd.DataFrame]:
    path = TABLES / f"{model_stem}.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def load_xgb_model(city: str, horizon: int) -> Optional[Dict]:
    pkl_path = PRED_DIR / f"{city}_h{horizon:02d}_xgb.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_gbr_model(city: str, horizon: int) -> Optional[Any]:
    model_path = MODELS_B / f"{city}_h{horizon:02d}_gbr.joblib"
    feat_path  = MODELS_B / f"{city}_h{horizon:02d}_gbr_features.json"
    if not model_path.exists():
        return None
    try:
        model = joblib.load(model_path)
        feats = None
        if feat_path.exists():
            with open(feat_path) as f:
                feats = json.load(f)
        return {"model": model, "feat_cols": feats}
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_lstm_model(city: str, horizon: int) -> Optional[Dict]:
    model_path  = MODELS_B / f"{city}_h{horizon:02d}_lstm.keras"
    sx_path     = MODELS_B / f"{city}_h{horizon:02d}_lstm_scaler_x.joblib"
    sy_path     = MODELS_B / f"{city}_h{horizon:02d}_lstm_scaler_y.joblib"
    feat_path   = MODELS_B / f"{city}_h{horizon:02d}_lstm_features.json"
    if not model_path.exists():
        return None
    try:
        from tensorflow import keras
        model = keras.models.load_model(str(model_path))
        sc_x, sc_y, feats = None, None, None
        if sx_path.exists():
            sc_x = joblib.load(sx_path)
        if sy_path.exists():
            sc_y = joblib.load(sy_path)
        if feat_path.exists():
            with open(feat_path) as f:
                feats = json.load(f)
        return {"model": model, "scaler_x": sc_x, "scaler_y": sc_y, "feat_cols": feats}
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_bilstm_csv() -> Optional[pd.DataFrame]:
    return load_track_b_csv("track_b_bilstm")

@st.cache_data(show_spinner=False)
def get_all_track_b_metrics() -> Dict[str, Optional[pd.DataFrame]]:
    stems = {
        "GBR":        "track_b_gbr",
        "RF":         "track_b_rf",
        "XGB":        "track_b_xgb",
        "LSTM":       "track_b_lstm",
        "BiLSTM":     "track_b_bilstm",
        "CNN-BiLSTM": "track_b_cnn_bilstm",
    }
    return {name: load_track_b_csv(stem) for name, stem in stems.items()}

def aqi_badge(aqi: float) -> str:
    cat, col = get_aqi_category(aqi)
    text_col = "#000" if col in ("#00e400", "#92d050", "#ffff00") else "#fff"
    return f'<span class="aqi-badge" style="background:{col};color:{text_col};">AQI {aqi:.0f} - {cat}</span>'

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '''<div style="text-align:center;padding:16px 0 24px;">
            <div style="font-size:2rem;">🌬️</div>
            <div style="font-size:1.1rem;font-weight:700;color:#00d4aa;">AQI Platform</div>
            <div style="font-size:0.75rem;color:#7090a0;margin-top:4px;">IIIT Nagpur</div>
        </div>''',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "🔬 AQI Estimation",
            "📈 AQI Forecasting",
            "📊 Model Comparison",
            "🗺️ India AQI Map",
            "👨‍💻 About Project",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem;color:#506070;text-align:center;">CPCB · 18 Cities · 7 Models</div>',
        unsafe_allow_html=True,
    )

# ─── Pages ───────────────────────────────────────────────────────────────────
def page_home():
    st.markdown(
        '''<div class="hero-container">
            <div class="hero-title">🌬️ AQI Prediction Platform</div>
            <div class="hero-subtitle">Research Internship · IIIT Nagpur</div>
            <div class="hero-caption">CPCB National Air Quality Dataset · Multi-City Air Quality Intelligence</div>
        </div>''',
        unsafe_allow_html=True,
    )

    kpi_data = [
        ("18", "Cities Monitored"),
        ("18.7M", "Raw Records"),
        ("7", "ML/DL Models"),
        ("3", "Forecast Horizons"),
        ("0.99", "Best R² (GBR)"),
    ]
    cols = st.columns(5)
    for col, (val, label) in zip(cols, kpi_data):
        col.markdown(
            f'''<div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>''',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🔬 AQI Estimation (Track A)")
        st.markdown("""
        Enter current pollutant readings and get instant AQI estimates from
        7 machine learning models. Compare performance across Ridge, Random Forest,
        Gradient Boosting, XGBoost, LSTM, BiLSTM, and CNN-BiLSTM.
        """)
    with col_r:
        st.markdown("### 📈 AQI Forecasting (Track B)")
        st.markdown("""
        Select a city and get multi-horizon AQI forecasts (+1h, +6h, +24h) using
        trained XGBoost, GBR, and LSTM models with lag-based features.
        """)

    st.markdown("---")
    st.markdown("### 🏆 Project Highlights")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.info("Dual-Track Protocol\nTrack A: same-time estimation\nTrack B: leakage-free forecasting")
    with h_col2:
        st.success("Best Model: GBR\nAvg R² = 0.99 (Track A)\nAvg R² ≈ 0.90 (Track B)")
    with h_col3:
        st.warning("18 Cities × 7 Models\n934K hourly records\n450+ experiments conducted")

    st.markdown("---")
    st.markdown(
        '''<div class="info-box">
        📌 <b>About this platform:</b> All model performance figures are based on held-out test sets
        (chronological 70/15/15 split). Track A models have access to same-time pollutants (estimation task).
        Track B models use only lag and rolling features (true forecasting — no data leakage).
        </div>''',
        unsafe_allow_html=True,
    )

def page_estimation():
    st.markdown('<div class="section-header">🔬 AQI Estimation</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter current pollutant concentrations to compute the official CPCB AQI "
        "and see how all 7 Track A models performed for your selected city."
    )
    ta_df = load_track_a_csv()
    city = st.selectbox("Select City", CITIES, key="est_city")
    if ta_df is not None:
        city_df = ta_df[ta_df["city"] == city]
        if not city_df.empty:
            best_row = city_df.loc[city_df["r2"].idxmax()]
            st.metric("Best Model for this City", best_row["model"], f"R² = {best_row['r2']:.4f}")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    pm25 = c1.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=999.0, value=60.0, step=1.0)
    pm10 = c2.number_input("PM10 (µg/m³)", min_value=0.0, max_value=999.0, value=100.0, step=1.0)
    no2  = c3.number_input("NO2 (µg/m³)", min_value=0.0, max_value=800.0, value=40.0, step=1.0)
    so2  = c4.number_input("SO2 (µg/m³)", min_value=0.0, max_value=1600.0, value=20.0, step=1.0)
    c5, c6, c7, _ = st.columns(4)
    co   = c5.number_input("CO (mg/m³)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    o3   = c6.number_input("O3 (µg/m³)", min_value=0.0, max_value=800.0, value=30.0, step=1.0)
    nh3  = c7.number_input("NH3 (µg/m³)", min_value=0.0, max_value=2400.0, value=10.0, step=1.0)

    if st.button("Compute AQI", type="primary"):
        cpcb_aqi = compute_cpcb_aqi(pm25, pm10, no2, so2, co, o3, nh3)
        if cpcb_aqi is not None:
            cat, col = get_aqi_category(cpcb_aqi)
            text_col = "#000" if col in ("#00e400", "#92d050", "#ffff00") else "#fff"
            st.markdown("---")
            st.markdown(
                f'''<div style="background:{col};border-radius:12px;padding:20px;text-align:center;margin:8px 0;">
                    <span style="font-size:3rem;font-weight:800;color:{text_col};">{cpcb_aqi:.0f}</span><br>
                    <span style="font-size:1.2rem;font-weight:600;color:{text_col};">{cat}</span>
                </div>''',
                unsafe_allow_html=True,
            )

def page_forecasting():
    st.markdown('<div class="section-header">📈 AQI Forecasting</div>', unsafe_allow_html=True)
    st.markdown("Select a city to view multi-horizon AQI forecasts (+1h, +6h, +24h) using available trained models.")
    city = st.selectbox("Select City", CITIES, key="fore_city")

def page_model_comparison():
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

def page_india_map():
    st.markdown('<div class="section-header">🗺️ India AQI Map</div>', unsafe_allow_html=True)

def page_about():
    st.markdown('<div class="section-header">👨‍💻 About Project</div>', unsafe_allow_html=True)

if page == "🏠 Home":
    page_home()
elif page == "🔬 AQI Estimation":
    page_estimation()
elif page == "📈 AQI Forecasting":
    page_forecasting()
elif page == "📊 Model Comparison":
    page_model_comparison()
elif page == "🗺️ India AQI Map":
    page_india_map()
elif page == "👨‍💻 About Project":
    page_about()
