"""
AQI Prediction Using Machine Learning and Deep Learning
=======================================================
Research Internship — IIIT Nagpur
Author: Aman Gajbhiye, YCCE
"""

# ── Standard library ─────────────────────────────────────────────────────────
import sys
import warnings
import os
import json
import math
import time
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party imports ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError as e:
    raise RuntimeError("streamlit is required") from e

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── MUST be the very first Streamlit call ────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction — Research Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PATH DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_root() -> Path:
    """Auto-discover the AQI project root from multiple candidate locations."""
    candidates = [
        Path("AQI_Prediction_Project_Final (1)"),
        Path(__file__).parent / "AQI_Prediction_Project_Final (1)",
        Path(__file__).parent.parent / "AQI_Prediction_Project_Final (1)",
        Path.home() / "AQI_Prediction_Project_Final (1)",
    ]
    for p in candidates:
        if p.exists():
            return p
    for parent in Path(__file__).resolve().parents[:5]:
        p = parent / "AQI_Prediction_Project_Final (1)"
        if p.exists():
            return p
    return Path("AQI_Prediction_Project_Final (1)")  # fallback

ROOT       = find_root()
TABLES_DIR = ROOT / "outputs" / "tables"
PREDS_DIR  = ROOT / "outputs" / "predictions"
FIGS_DIR   = ROOT / "outputs" / "figures"

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CITIES = [
    "Ahmedabad", "Chennai", "Delhi_NCR", "GandhiNagar", "Hyderabad", "Indore",
    "Jaipur", "Jodhpur", "Mumbai", "Mumbai_suburbs", "Nagpur", "Pune",
    "Singrauli", "Surat", "Thane", "Vapi", "bhopal", "vishakhapattanam",
]

CITY_DISPLAY = {c: c.replace("_", " ").title() for c in CITIES}

CITY_COORDS = {
    "Ahmedabad":        (23.0225, 72.5714),
    "Chennai":          (13.0827, 80.2707),
    "Delhi_NCR":        (28.6139, 77.2090),
    "GandhiNagar":      (23.2156, 72.6369),
    "Hyderabad":        (17.3850, 78.4867),
    "Indore":           (22.7196, 75.8577),
    "Jaipur":           (26.9124, 75.7873),
    "Jodhpur":          (26.2389, 73.0243),
    "Mumbai":           (19.0760, 72.8777),
    "Mumbai_suburbs":   (19.2183, 72.9781),
    "Nagpur":           (21.1458, 79.0882),
    "Pune":             (18.5204, 73.8567),
    "Singrauli":        (24.1990, 82.6747),
    "Surat":            (21.1702, 72.8311),
    "Thane":            (19.2183, 72.9781),
    "Vapi":             (20.3893, 72.9106),
    "bhopal":           (23.2599, 77.4126),
    "vishakhapattanam": (17.6868, 83.2185),
}

HORIZONS    = ["1 Hour", "6 Hours", "24 Hours"]
HORIZON_MAP = {"1 Hour": "h01", "6 Hours": "h06", "24 Hours": "h24"}

ESTIMATION_MODELS = ["GradBoost", "RandomForest", "XGBoost", "Ridge", "LSTM", "BiLSTM", "CNN-BiLSTM"]
FORECAST_MODELS   = ["GBR", "RandomForest", "XGBoost", "LSTM", "BiLSTM", "CNN-BiLSTM"]

ESTIMATION_MODEL_MAP = {
    "GradBoost":     "GradBoost",
    "Random Forest": "RandomForest",
    "XGBoost":       "XGBoost",
    "Ridge":         "Ridge",
    "LSTM":          "LSTM",
    "BiLSTM":        "BiLSTM",
    "CNN-BiLSTM":    "CNN-BiLSTM",
}
FORECASTING_MODEL_MAP = {
    "GBR":           "GBR",
    "Random Forest": "RF",
    "XGBoost":       "XGBoost",
    "BiLSTM":        "BiLSTM",
    "LSTM":          "LSTM",
    "CNN-BiLSTM":    "CNN-BiLSTM",
}

FALLBACK_ESTIMATION = {
    "GradBoost":    {"r2": 0.9906, "mae": 2.9450,  "rmse": 5.7661},
    "RandomForest": {"r2": 0.9874, "mae": 1.6428,  "rmse": 6.0478},
    "XGBoost":      {"r2": 0.9856, "mae": 2.8306,  "rmse": 6.8206},
    "Ridge":        {"r2": 0.8304, "mae": 15.9219, "rmse": 23.0631},
    "LSTM":         {"r2": 0.6411, "mae": 22.4892, "rmse": 33.5919},
    "BiLSTM":       {"r2": 0.5897, "mae": 22.4619, "rmse": 35.9004},
    "CNN-BiLSTM":   {"r2": 0.2756, "mae": 36.9234, "rmse": 47.5118},
}
FALLBACK_FORECASTING = {
    "GBR":          {"r2": 0.4997, "mae": 24.8, "rmse": 38.5},
    "RF":           {"r2": 0.4914, "mae": 25.1, "rmse": 38.8},
    "XGBoost":      {"r2": 0.4902, "mae": 25.3, "rmse": 39.0},
    "BiLSTM":       {"r2": 0.2831, "mae": 31.2, "rmse": 44.1},
    "LSTM":         {"r2": 0.2768, "mae": 31.8, "rmse": 44.9},
    "CNN-BiLSTM":   {"r2": -0.41,  "mae": 52.3, "rmse": 67.4},
}

# Design tokens
DARK_BG      = "#0D1117"
DARK_CARD    = "#161B27"
DARK_BORDER  = "#2a2d3e"
TEXT_PRIMARY = "#f0f2ff"
ACCENT_BLUE  = "#6C9EE8"
ACCENT_TEAL  = "#45D4C5"
ACCENT_GOLD  = "#F5C518"

# ══════════════════════════════════════════════════════════════════════════════
# CPCB AQI BREAKPOINTS & CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

CPCB_BREAKPOINTS = {
    "PM2.5": [
        (0, 30,    0,   50),
        (30, 60,   51,  100),
        (60, 90,   101, 200),
        (90, 120,  201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    "PM10": [
        (0,   50,   0,   50),
        (50,  100,  51,  100),
        (100, 250,  101, 200),
        (250, 350,  201, 300),
        (350, 430,  301, 400),
        (430, 600,  401, 500),
    ],
    "NO2": [
        (0,   40,   0,   50),
        (40,  80,   51,  100),
        (80,  180,  101, 200),
        (180, 280,  201, 300),
        (280, 400,  301, 400),
        (400, 800,  401, 500),
    ],
    "SO2": [
        (0,   40,   0,   50),
        (40,  80,   51,  100),
        (80,  380,  101, 200),
        (380, 800,  201, 300),
        (800, 1600, 301, 400),
        (1600, 2100, 401, 500),
    ],
    "CO": [
        (0,    1.0,  0,   50),
        (1.0,  2.0,  51,  100),
        (2.0,  10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500),
    ],
    "O3": [
        (0,   50,   0,   50),
        (50,  100,  51,  100),
        (100, 168,  101, 200),
        (168, 208,  201, 300),
        (208, 748,  301, 400),
        (748, 1000, 401, 500),
    ],
    "NH3": [
        (0,   200,  0,   50),
        (200, 400,  51,  100),
        (400, 800,  101, 200),
        (800, 1200, 201, 300),
        (1200, 1800, 301, 400),
        (1800, 2400, 401, 500),
    ],
}

AQI_CATEGORIES = [
    (0,   50,  "Good",         "#00C853",
     "Air quality is satisfactory. No health risk."),
    (51,  100, "Satisfactory", "#AEEA00",
     "Acceptable air quality. Sensitive people may experience minor discomfort."),
    (101, 200, "Moderate",     "#FFD600",
     "Sensitive individuals (elderly, children, asthma) should reduce outdoor activity."),
    (201, 300, "Poor",         "#FF6D00",
     "People with lung/heart conditions should avoid outdoor activity."),
    (301, 400, "Very Poor",    "#DD2C00",
     "Health effects for all. Avoid prolonged outdoor activity."),
    (401, 500, "Severe",       "#880E4F",
     "Severe health impact. Avoid all outdoor activity. Stay indoors."),
]

SAMPLE_POLLUTANT_VALUES = {
    "PM2.5": 85.0,
    "PM10":  150.0,
    "NO2":   60.0,
    "SO2":   20.0,
    "CO":    1.5,
    "O3":    100.0,
    "NH3":   200.0,
}


def compute_sub_index(value: float, breakpoints: list) -> float:
    """Compute sub-index for one pollutant using linear interpolation."""
    if value < 0:
        return 0.0
    for (BPlo, BPhi, Ilo, Ihi) in breakpoints:
        if BPlo <= value <= BPhi:
            return Ilo + (value - BPlo) * (Ihi - Ilo) / max(BPhi - BPlo, 1e-9)
    return 500.0


def compute_cpcb_aqi(pollutant_values: dict) -> tuple:
    """
    Returns (aqi_int, dominant_pollutant, sub_indices_dict).
    sub_indices_dict maps pollutant name -> sub_index value.
    """
    sub_indices = {}
    for pollutant, value in pollutant_values.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if pollutant not in CPCB_BREAKPOINTS:
            continue
        si = compute_sub_index(float(value), CPCB_BREAKPOINTS[pollutant])
        sub_indices[pollutant] = si
    if not sub_indices:
        return None, None, {}
    dominant = max(sub_indices, key=sub_indices.get)
    aqi = int(round(sub_indices[dominant]))
    return aqi, dominant, sub_indices


def get_aqi_category(aqi: int) -> tuple:
    """Returns (label, color_hex, advisory)."""
    for lo, hi, label, color, advisory in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color, advisory
    if aqi > 500:
        return "Severe+", "#880E4F", "Extremely hazardous. Remain indoors."
    return "Unknown", "#888888", ""

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def safe_load_csv(path_str: str, **kwargs) -> pd.DataFrame:
    """Load CSV safely, return empty DataFrame on failure."""
    try:
        p = Path(path_str)
        if p.exists():
            return pd.read_csv(p, **kwargs)
    except Exception:
        pass
    return pd.DataFrame()


def load_track_a() -> pd.DataFrame:
    return safe_load_csv(str(TABLES_DIR / "final_track_a_complete.csv"))


def load_track_b_all() -> pd.DataFrame:
    """Load and combine all Track B CSVs."""
    stems = ["track_b_gbr", "track_b_rf", "track_b_xgb", "track_b_bilstm", "track_b_lstm"]
    frames = []
    for stem in stems:
        df = safe_load_csv(str(TABLES_DIR / f"{stem}.csv"))
        if not df.empty:
            frames.append(df)
    extra = safe_load_csv(str(TABLES_DIR / "final_track_b_complete.csv"))
    if not extra.empty:
        frames.append(extra)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["city", "horizon", "model"] if all(
        c in combined.columns for c in ["city", "horizon", "model"]
    ) else combined.columns.tolist())
    return combined


def safe_pivot(df, index, columns, values, aggfunc="mean", fill_value=0):
    """Crash-proof pivot table."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=[index, columns])
    if df.empty:
        return pd.DataFrame()
    try:
        pivot = df.pivot_table(
            index=index, columns=columns, values=values,
            aggfunc=aggfunc, fill_value=fill_value
        )
        return pivot
    except Exception:
        return pd.DataFrame()


def detect_pred_columns(df: pd.DataFrame) -> tuple:
    """Auto-detect actual and predicted column names."""
    actual_candidates  = ["actual_aqi", "y_true", "AQI", "aqi", "actual"]
    pred_candidates    = ["xgb_pred", "y_pred", "pred", "predicted_aqi", "predicted", "prediction"]
    actual_col = None
    for c in actual_candidates:
        if c in df.columns:
            actual_col = c
            break
    pred_col = None
    for c in pred_candidates:
        if c in df.columns:
            pred_col = c
            break
    return actual_col, pred_col


def load_pred_csv(city: str, horizon_code: str) -> pd.DataFrame:
    """Load prediction CSV for city + horizon code (h01/h06/h24)."""
    p = PREDS_DIR / f"{city}_{horizon_code}_predictions.csv"
    return safe_load_csv(str(p))

# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def inject_css(light_mode: bool = False):
    """Inject all global CSS."""
    sidebar_css = """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #0A0E1A 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #f0f2ff !important; }
    [data-testid="stSidebar"] .stRadio label { color: #f0f2ff !important; }
    [data-testid="stSidebar"] hr { border-color: #2a2d3e !important; }
    [data-theme="light"] .stSidebar { background-color: #1a1f2e !important; }
    [data-theme="light"] .stSidebar .stMarkdown { color: #ffffff !important; }
    [data-theme="light"] .stSidebar label { color: #ffffff !important; }
    [data-theme="light"] .stRadio label { color: #000000 !important; }
    .kpi-card {
        background: linear-gradient(135deg, #161B27 0%, #1C2333 100%);
        border: 1px solid #2a2d3e;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 12px;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #6C9EE8; margin: 0; }
    .kpi-label { font-size: 0.9rem; color: #9ca3af; margin-top: 4px; }
    .metric-card {
        background: #161B27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .workflow-step {
        background: #161B27;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        font-size: 0.85rem;
        color: #f0f2ff;
    }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #6C9EE8;
        margin: 1.5rem 0 0.5rem 0;
        border-bottom: 2px solid #2a2d3e;
        padding-bottom: 6px;
    }
    .badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    main_light = """
    <style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .main .block-container { background-color: #ffffff; }
    h1, h2, h3, h4 { color: #1a1a2e; }
    </style>
    """ if light_mode else ""

    main_dark = """
    <style>
    .stApp { background-color: #0D1117; color: #f0f2ff; }
    h1, h2, h3, h4 { color: #f0f2ff; }
    .stTabs [data-baseweb="tab"] { color: #9ca3af; }
    .stTabs [aria-selected="true"] { color: #6C9EE8; border-bottom-color: #6C9EE8; }
    </style>
    """ if not light_mode else ""

    st.markdown(sidebar_css + main_light + main_dark, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    # Hero
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0D1117 0%, #1a2744 50%, #0D1117 100%);
                border: 1px solid #2a2d3e; border-radius: 20px; padding: 48px 40px 36px;
                text-align: center; margin-bottom: 32px;">
        <div style="font-size: 1rem; color: #6C9EE8; letter-spacing: 3px; text-transform: uppercase;
                    font-weight: 600; margin-bottom: 16px;">
            IIIT NAGPUR RESEARCH INTERNSHIP
        </div>
        <h1 style="font-size: 2.6rem; font-weight: 800; color: #f0f2ff; margin: 0 0 12px 0;
                   line-height: 1.2;">
            AQI Prediction Using Machine Learning<br>and Deep Learning
        </h1>
        <p style="font-size: 1.1rem; color: #9ca3af; margin: 0 0 28px 0;">
            Research Internship &nbsp;·&nbsp; IIIT Nagpur &nbsp;·&nbsp;
            CPCB Multi-City Dataset &nbsp;·&nbsp; 18 Indian Cities
        </p>
        <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
            <a href="https://github.com/amangajbhiye99/aqi-prediction" target="_blank"
               style="background: #6C9EE8; color: #0D1117; padding: 10px 24px;
                      border-radius: 8px; font-weight: 700; text-decoration: none;">
                🔗 GitHub
            </a>
            <a href="https://www.cpcb.nic.in" target="_blank"
               style="background: #1C2333; color: #f0f2ff; padding: 10px 24px;
                      border-radius: 8px; font-weight: 600; text-decoration: none;
                      border: 1px solid #2a2d3e;">
                📊 CPCB Dataset
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, icon in zip(
        [c1, c2, c3, c4],
        ["18", "18.7M", "7", "3"],
        ["Cities", "Records", "Models", "Horizons"],
        ["🏙️", "📊", "🤖", "⏱️"],
    ):
        col.markdown(f"""
        <div class="kpi-card">
            <div style="font-size: 2rem; margin-bottom: 4px;">{icon}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Research Workflow
    st.markdown('<div class="section-header">⚙️ Research Workflow</div>', unsafe_allow_html=True)
    steps = [
        ("📥", "Data Collection", "CPCB hourly data"),
        ("🔧", "Preprocessing", "Imputation & scaling"),
        ("⚗️", "Feature Engineering", "88–115 features"),
        ("🏋️", "Model Training", "7 ML/DL models"),
        ("📏", "Evaluation", "R², MAE, RMSE"),
        ("🚀", "Deployment", "Streamlit dashboard"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, name, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class="workflow-step">
            <div style="font-size: 1.5rem;">{icon}</div>
            <div style="font-weight: 600; margin: 4px 0;">{name}</div>
            <div style="color: #9ca3af; font-size: 0.75rem;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Research Objectives
    st.markdown('<div class="section-header">🎯 Research Objectives</div>', unsafe_allow_html=True)
    oc1, oc2 = st.columns(2)
    oc1.markdown("""
    <div style="background: #161B27; border: 1px solid #2a2d3e; border-radius: 12px; padding: 20px;">
        <h3 style="color: #6C9EE8; margin: 0 0 8px 0;">🔬 AQI Estimation</h3>
        <p style="color: #9ca3af; margin: 0;">
            Current-time AQI estimation using ML/DL models trained on concurrent
            pollutant and meteorological measurements — Track A.
        </p>
    </div>""", unsafe_allow_html=True)
    oc2.markdown("""
    <div style="background: #161B27; border: 1px solid #2a2d3e; border-radius: 12px; padding: 20px;">
        <h3 style="color: #45D4C5; margin: 0 0 8px 0;">📈 AQI Forecasting</h3>
        <p style="color: #9ca3af; margin: 0;">
            Multi-horizon AQI forecasting (1 h, 6 h, 24 h) using lag/rolling features
            only — no future pollutant leakage — Track B.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Best Model Performance
    st.markdown('<div class="section-header">🏆 Best Model Performance</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    bc1.markdown("""
    <div style="background: linear-gradient(135deg, #161B27, #1C2B1E); border: 1px solid #2a2d3e;
                border-radius: 12px; padding: 20px; text-align: center;">
        <div style="color: #00C853; font-size: 0.85rem; font-weight: 600; letter-spacing: 2px;
                    text-transform: uppercase; margin-bottom: 8px;">Best Estimation</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #f0f2ff; margin-bottom: 4px;">
            Gradient Boosting
        </div>
        <div style="font-size: 2.4rem; font-weight: 900; color: #00C853;">R² = 0.9906</div>
        <div style="color: #9ca3af; margin-top: 8px;">MAE = 2.95 &nbsp;|&nbsp; RMSE = 5.77</div>
    </div>""", unsafe_allow_html=True)
    bc2.markdown("""
    <div style="background: linear-gradient(135deg, #161B27, #1C2233); border: 1px solid #2a2d3e;
                border-radius: 12px; padding: 20px; text-align: center;">
        <div style="color: #6C9EE8; font-size: 0.85rem; font-weight: 600; letter-spacing: 2px;
                    text-transform: uppercase; margin-bottom: 8px;">Best Forecasting</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #f0f2ff; margin-bottom: 4px;">
            GBR
        </div>
        <div style="font-size: 2.4rem; font-weight: 900; color: #6C9EE8;">R² = 0.4997</div>
        <div style="color: #9ca3af; margin-top: 8px;">MAE = 24.8 &nbsp;|&nbsp; RMSE = 38.5</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Research Highlights
    st.markdown('<div class="section-header">💡 Key Research Findings</div>', unsafe_allow_html=True)
    findings = [
        ("🎯", "GradBoost achieves R²=0.9906 on AQI estimation — near-perfect current-time prediction."),
        ("📉", "Performance degrades with forecast horizon: 1 h R²≈0.52 → 24 h R²≈0.43 (GBR)."),
        ("🤖", "DL models (LSTM/BiLSTM) underperform classical ML on both tasks across 18 cities."),
        ("🏙️", "High-pollution cities (Delhi, Singrauli) show higher absolute errors but similar R²."),
        ("🔍", "Strict data-leakage audit: Track B uses only lagged/rolling features, no future pollutant values."),
        ("🌍", "18.7 M hourly records from 18 Indian cities — largest CPCB multi-city benchmark to date."),
    ]
    f1, f2 = st.columns(2)
    for i, (icon, text) in enumerate(findings):
        col = f1 if i % 2 == 0 else f2
        col.markdown(f"""
        <div style="background: #161B27; border-left: 3px solid #6C9EE8; border-radius: 8px;
                    padding: 12px 16px; margin-bottom: 10px; color: #f0f2ff; font-size: 0.9rem;">
            {icon}&nbsp;&nbsp;{text}
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 0.85rem; padding: 16px 0;">
        <strong style="color: #6C9EE8;">Aman Gajbhiye</strong> &nbsp;·&nbsp;
        YCCE, Nagpur &nbsp;·&nbsp;
        IIIT Nagpur Research Internship &nbsp;·&nbsp;
        <a href="https://github.com/amangajbhiye99/aqi-prediction" style="color: #45D4C5;">
            GitHub
        </a>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AQI ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def page_estimation():
    st.title("🔬 AQI Estimation")
    st.markdown(
        "Current-time AQI estimation using concurrent pollutant and meteorological features.",
        help="Track A: no data leakage — only same-timestamp features used."
    )

    tabs = st.tabs(["🧮 CPCB AQI Calculator", "📊 Model Leaderboard", "📈 City Performance"])

    # ── Tab 1: CPCB Calculator ─────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🧮 CPCB AQI Calculator")
        st.markdown("Enter pollutant concentrations to compute AQI using official CPCB breakpoints.")

        POLLUTANT_LABELS = [
            ("PM2.5", "PM2.5 (µg/m³)", 0.0, 500.0),
            ("PM10",  "PM10 (µg/m³)",  0.0, 600.0),
            ("NO2",   "NO₂ (µg/m³)",   0.0, 800.0),
            ("SO2",   "SO₂ (µg/m³)",   0.0, 2100.0),
            ("CO",    "CO (mg/m³)",     0.0, 50.0),
            ("O3",    "O₃ (µg/m³)",    0.0, 1000.0),
            ("NH3",   "NH₃ (µg/m³)",   0.0, 2400.0),
        ]

        if "calc_values" not in st.session_state:
            st.session_state.calc_values = {k: None for k, _, _, _ in POLLUTANT_LABELS}
        if "calc_sample_loaded" not in st.session_state:
            st.session_state.calc_sample_loaded = False

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
        reset_clicked  = btn_col1.button("🔄 Reset",              key="calc_reset")
        sample_clicked = btn_col2.button("📋 Load Sample Values", key="calc_sample")
        calc_clicked   = btn_col3.button("⚡ Calculate AQI",      key="calc_go",
                                          type="primary")

        if reset_clicked:
            st.session_state.calc_values = {k: None for k, _, _, _ in POLLUTANT_LABELS}
            st.session_state.calc_sample_loaded = False
            st.rerun()

        if sample_clicked:
            st.session_state.calc_values = dict(SAMPLE_POLLUTANT_VALUES)
            st.session_state.calc_sample_loaded = True
            st.rerun()

        poll_vals = {}
        left_polls  = POLLUTANT_LABELS[:4]
        right_polls = POLLUTANT_LABELS[4:]
        lc, rc = st.columns(2)
        for key, label, mn, mx in left_polls:
            default = st.session_state.calc_values.get(key)
            val = lc.number_input(
                label, min_value=float(mn), max_value=float(mx),
                value=float(default) if default is not None else 0.0,
                step=0.1, key=f"poll_{key}"
            )
            poll_vals[key] = val if val > 0 else None

        for key, label, mn, mx in right_polls:
            default = st.session_state.calc_values.get(key)
            val = rc.number_input(
                label, min_value=float(mn), max_value=float(mx),
                value=float(default) if default is not None else 0.0,
                step=0.1, key=f"poll_{key}"
            )
            poll_vals[key] = val if val > 0 else None

        if calc_clicked:
            entered = {k: v for k, v in poll_vals.items() if v is not None and v > 0}
            if not entered:
                st.warning("Please enter at least one pollutant concentration.")
            else:
                aqi, dominant, sub_indices = compute_cpcb_aqi(entered)
                if aqi is None:
                    st.error("Could not compute AQI. Check inputs.")
                else:
                    cat_label, cat_color, advisory = get_aqi_category(aqi)
                    st.markdown("---")
                    r1, r2 = st.columns([1, 2])
                    r1.markdown(f"""
                    <div style="background: {cat_color}22; border: 2px solid {cat_color};
                                border-radius: 16px; padding: 32px; text-align: center;">
                        <div style="font-size: 4rem; font-weight: 900; color: {cat_color};
                                    line-height: 1;">{aqi}</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #f0f2ff;
                                    margin: 8px 0;">AQI</div>
                        <div style="display: inline-block; background: {cat_color};
                                    color: #000; padding: 4px 16px; border-radius: 20px;
                                    font-weight: 700; font-size: 1rem;">{cat_label}</div>
                        <div style="color: #9ca3af; font-size: 0.8rem; margin-top: 12px;">
                            Dominant: {dominant}
                        </div>
                    </div>""", unsafe_allow_html=True)
                    r2.markdown(f"""
                    <div style="background: #161B27; border: 1px solid #2a2d3e;
                                border-radius: 12px; padding: 20px; height: 100%;">
                        <h4 style="color: #6C9EE8; margin: 0 0 12px 0;">🏥 Health Advisory</h4>
                        <p style="color: #f0f2ff; font-size: 1rem; line-height: 1.6;">{advisory}</p>
                        <h4 style="color: #6C9EE8; margin: 16px 0 8px 0;">📊 Sub-Indices</h4>
                    """, unsafe_allow_html=True)
                    for poll, si in sorted(sub_indices.items(), key=lambda x: -x[1]):
                        _, c2_color, _ = get_aqi_category(int(round(si)))
                        r2.markdown(f"""
                        <div style="display: flex; justify-content: space-between;
                                    margin-bottom: 6px; color: #f0f2ff;">
                            <span>{poll}</span>
                            <span style="color: {c2_color}; font-weight: 700;">{si:.1f}</span>
                        </div>""", unsafe_allow_html=True)
                    r2.markdown("</div>", unsafe_allow_html=True)

                    # Sub-index bar chart
                    if sub_indices:
                        try:
                            si_df = pd.DataFrame({
                                "Pollutant": list(sub_indices.keys()),
                                "Sub-Index":  list(sub_indices.values()),
                            }).sort_values("Sub-Index", ascending=False)
                            fig_si = px.bar(
                                si_df, x="Pollutant", y="Sub-Index",
                                title="Sub-Index by Pollutant",
                                color="Sub-Index",
                                color_continuous_scale="RdYlGn_r",
                                range_color=[0, 500],
                                template="plotly_dark",
                            )
                            fig_si.add_hline(y=aqi, line_dash="dash",
                                             line_color="white", annotation_text=f"AQI={aqi}")
                            fig_si.update_layout(
                                paper_bgcolor="#0D1117",
                                plot_bgcolor="#0D1117",
                                font_color="#f0f2ff",
                                coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig_si, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render sub-index chart: {e}")

    # ── Tab 2: Model Leaderboard ───────────────────────────────────────────
    with tabs[1]:
        st.markdown("### 📊 Track A — Model Leaderboard")
        ta_df = load_track_a()

        if ta_df.empty:
            st.info("Track A CSV not found. Showing confirmed fallback metrics.")
            rows = []
            medals = ["🥇", "🥈", "🥉", "", "", "", ""]
            for i, (model, metrics) in enumerate(
                sorted(FALLBACK_ESTIMATION.items(), key=lambda x: -x[1]["r2"])
            ):
                rows.append({
                    "Rank": medals[i] if i < 3 else str(i + 1),
                    "Model": model,
                    "R²": round(metrics["r2"], 4),
                    "MAE": round(metrics["mae"], 4),
                    "RMSE": round(metrics["rmse"], 4),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            try:
                rank_df = (
                    ta_df.groupby("model")[["r2", "mae", "rmse"]]
                    .mean()
                    .reset_index()
                    .sort_values("r2", ascending=False)
                    .reset_index(drop=True)
                )
                medals = ["🥇", "🥈", "🥉"] + [""] * (len(rank_df) - 3)
                rank_df.insert(0, "Rank", medals)
                rank_df["r2"]   = rank_df["r2"].round(4)
                rank_df["mae"]  = rank_df["mae"].round(2)
                rank_df["rmse"] = rank_df["rmse"].round(2)
                rank_df.columns = ["Rank", "Model", "Avg R²", "Avg MAE", "Avg RMSE"]
                st.dataframe(rank_df, use_container_width=True, hide_index=True)

                try:
                    fig_rank = px.bar(
                        rank_df.sort_values("Avg R²"),
                        x="Avg R²", y="Model", orientation="h",
                        title="Model R² — Track A (Estimation)",
                        color="Avg R²",
                        color_continuous_scale="Blues",
                        template="plotly_dark",
                    )
                    fig_rank.update_layout(
                        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                        font_color="#f0f2ff",
                        xaxis=dict(range=[0, 1.05]),
                    )
                    st.plotly_chart(fig_rank, use_container_width=True)
                except Exception as e:
                    st.warning(f"Bar chart error: {e}")
            except Exception as e:
                st.warning(f"Leaderboard error: {e}")

    # ── Tab 3: City Performance ────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 🏙️ Per-City Performance")
        ta_df = load_track_a()
        if ta_df.empty:
            st.info("Track A CSV not found.")
        else:
            try:
                city_options = sorted(ta_df["city"].unique().tolist()) if "city" in ta_df.columns else []
                if city_options:
                    sel_city = st.selectbox("Select City", city_options, key="est_city")
                    city_sub = ta_df[ta_df["city"] == sel_city].copy()
                    if not city_sub.empty:
                        city_sub = city_sub.sort_values("r2", ascending=False)
                        city_sub["r2"]   = city_sub["r2"].round(4)
                        city_sub["mae"]  = city_sub["mae"].round(2)
                        city_sub["rmse"] = city_sub["rmse"].round(2)
                        display_cols = [c for c in ["model", "r2", "mae", "rmse", "n_test"]
                                        if c in city_sub.columns]
                        st.dataframe(city_sub[display_cols], use_container_width=True, hide_index=True)

                        try:
                            fig_c = px.bar(
                                city_sub.sort_values("r2"),
                                x="r2", y="model", orientation="h",
                                title=f"{sel_city} — R² by Model",
                                color="r2",
                                color_continuous_scale="Viridis",
                                template="plotly_dark",
                            )
                            fig_c.update_layout(
                                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                                font_color="#f0f2ff",
                            )
                            st.plotly_chart(fig_c, use_container_width=True)
                        except Exception as e:
                            st.warning(f"City chart error: {e}")
            except Exception as e:
                st.warning(f"City performance error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_forecast_analysis():
    st.title("📈 Forecast Analysis")

    st.info(
        "📊 These results are from the independent test-set evaluation. "
        "Predictions were generated by the trained model on held-out data "
        "not used during training."
    )

    tb_all = load_track_b_all()

    # Selectors
    DISPLAY_CITIES   = sorted(CITIES)
    DISPLAY_HORIZONS = ["1 Hour", "6 Hours", "24 Hours"]
    DISPLAY_MODELS   = ["XGBoost", "Gradient Boosting", "Random Forest", "LSTM", "BiLSTM", "CNN-BiLSTM"]

    MODEL_TO_CSV = {
        "XGBoost":           "XGBoost",
        "Gradient Boosting": "GBR",
        "Random Forest":     "RF",
        "LSTM":              "LSTM",
        "BiLSTM":            "BiLSTM",
        "CNN-BiLSTM":        "CNN-BiLSTM",
    }
    HAS_PRED_CSV = {"XGBoost"}  # Only XGB has dedicated prediction CSVs

    sc1, sc2, sc3 = st.columns(3)
    sel_city    = sc1.selectbox("City",    DISPLAY_CITIES,   key="fa_city")
    sel_horiz   = sc2.selectbox("Horizon", DISPLAY_HORIZONS, key="fa_horizon")
    sel_model   = sc3.selectbox("Model",   DISPLAY_MODELS,   key="fa_model")

    horiz_code  = HORIZON_MAP[sel_horiz]
    csv_model   = MODEL_TO_CSV[sel_model]

    # Metric lookup
    metrics_row = None
    if not tb_all.empty and all(c in tb_all.columns for c in ["city", "horizon", "model"]):
        # Horizon stored as h01/h06/h24 OR 1h/6h/24h
        h_alt = horiz_code.replace("h0", "").replace("h", "") + "h"
        mask = (
            (tb_all["city"] == sel_city) &
            (tb_all["horizon"].isin([horiz_code, h_alt])) &
            (tb_all["model"] == csv_model)
        )
        sub = tb_all[mask]
        if not sub.empty:
            metrics_row = sub.iloc[0]

    # Metric cards
    st.markdown("---")
    mc1, mc2, mc3, mc4 = st.columns(4)
    r2_val   = round(metrics_row["r2"],   4) if metrics_row is not None else "N/A"
    mae_val  = round(metrics_row["mae"],  2) if metrics_row is not None else "N/A"
    rmse_val = round(metrics_row["rmse"], 2) if metrics_row is not None else "N/A"
    n_test   = int(metrics_row["n_test"]) if (metrics_row is not None and
               "n_test" in (metrics_row.index if hasattr(metrics_row, "index") else [])) else "N/A"

    for col, label, val, color in zip(
        [mc1, mc2, mc3, mc4],
        ["R²", "MAE", "RMSE", "Test Samples"],
        [r2_val, mae_val, rmse_val, n_test],
        ["#00C853", "#F5C518", "#FF6D00", "#6C9EE8"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div style="color: {color}; font-size: 1.8rem; font-weight: 800;">{val}</div>
            <div style="color: #9ca3af; font-size: 0.85rem;">{label}</div>
        </div>""", unsafe_allow_html=True)

    if metrics_row is None:
        st.info(f"No evaluation metrics found for {sel_city} / {sel_horiz} / {sel_model}.")

    # Load prediction CSV
    pred_df = load_pred_csv(sel_city, horiz_code)
    actual_col, pred_col = (None, None)
    if not pred_df.empty:
        actual_col, pred_col = detect_pred_columns(pred_df)

    has_pred_data = (
        sel_model in HAS_PRED_CSV and
        not pred_df.empty and
        actual_col is not None and
        pred_col is not None
    )

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "📈 Actual vs Predicted",
        "🔍 Error Analysis",
        "🏆 Model Comparison",
    ])

    # ── Tab 1: Actual vs Predicted ─────────────────────────────────────────
    with tab1:
        if not has_pred_data:
            if sel_model not in HAS_PRED_CSV:
                st.info(
                    f"Detailed prediction CSV is only available for XGBoost. "
                    f"For **{sel_model}**, see the metrics above from the evaluation CSV."
                )
            elif pred_df.empty:
                st.info(
                    f"Prediction CSV not found: "
                    f"`{sel_city}_{horiz_code}_predictions.csv`"
                )
            else:
                st.info("Could not detect actual/predicted columns in prediction CSV.")
        else:
            try:
                plot_df = pred_df[[actual_col, pred_col]].dropna().reset_index(drop=True)
                n_show  = min(500, len(plot_df))
                plot_sub = plot_df.iloc[:n_show]

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    y=plot_sub[actual_col], name="Actual AQI",
                    line=dict(color="#6C9EE8", width=1.5)
                ))
                fig_line.add_trace(go.Scatter(
                    y=plot_sub[pred_col], name=f"{sel_model} Predicted",
                    line=dict(color="#FF6D00", width=1.5, dash="dash")
                ))
                fig_line.update_layout(
                    title=f"{sel_city} — {sel_horiz} — Actual vs Predicted (first {n_show} samples)",
                    template="plotly_dark",
                    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                    font_color="#f0f2ff",
                    xaxis_title="Sample Index",
                    yaxis_title="AQI",
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(fig_line, use_container_width=True)

                # Scatter plot
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=plot_df[actual_col], y=plot_df[pred_col],
                    mode="markers",
                    marker=dict(color="#6C9EE8", opacity=0.4, size=4),
                    name="Predictions",
                ))
                mn_v = float(plot_df[[actual_col, pred_col]].min().min())
                mx_v = float(plot_df[[actual_col, pred_col]].max().max())
                fig_sc.add_trace(go.Scatter(
                    x=[mn_v, mx_v], y=[mn_v, mx_v],
                    mode="lines",
                    line=dict(color="#FFFFFF", dash="dash", width=1),
                    name="Perfect fit (y=x)",
                ))
                fig_sc.update_layout(
                    title=f"Actual vs Predicted Scatter — {sel_city} {sel_horiz}",
                    template="plotly_dark",
                    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                    font_color="#f0f2ff",
                    xaxis_title="Actual AQI",
                    yaxis_title="Predicted AQI",
                )
                st.plotly_chart(fig_sc, use_container_width=True)
            except Exception as e:
                st.warning(f"Chart error: {e}")

    # ── Tab 2: Error Analysis ──────────────────────────────────────────────
    with tab2:
        if not has_pred_data:
            st.info("Error analysis requires prediction CSV (available for XGBoost only).")
        else:
            try:
                plot_df   = pred_df[[actual_col, pred_col]].dropna().reset_index(drop=True)
                residuals = plot_df[pred_col] - plot_df[actual_col]

                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    y=residuals, mode="markers",
                    marker=dict(color="#F5C518", opacity=0.5, size=3),
                    name="Residuals",
                ))
                fig_res.add_hline(y=0, line_dash="dash", line_color="white")
                fig_res.update_layout(
                    title=f"Residuals over Time — {sel_city} {sel_horiz}",
                    template="plotly_dark",
                    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                    font_color="#f0f2ff",
                    xaxis_title="Sample Index",
                    yaxis_title="Residual (Predicted − Actual)",
                )
                st.plotly_chart(fig_res, use_container_width=True)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=residuals, nbinsx=50,
                    marker_color="#6C9EE8", opacity=0.8,
                    name="Residual Distribution",
                ))
                fig_hist.update_layout(
                    title=f"Residual Distribution — {sel_city} {sel_horiz}",
                    template="plotly_dark",
                    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                    font_color="#f0f2ff",
                    xaxis_title="Residual",
                    yaxis_title="Frequency",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown(f"""
                **Summary statistics:**
                Mean residual: `{residuals.mean():.2f}` &nbsp;|&nbsp;
                Std: `{residuals.std():.2f}` &nbsp;|&nbsp;
                Max over-pred: `{residuals.max():.2f}` &nbsp;|&nbsp;
                Max under-pred: `{residuals.min():.2f}`
                """)
            except Exception as e:
                st.warning(f"Error analysis chart error: {e}")

    # ── Tab 3: Model Comparison ────────────────────────────────────────────
    with tab3:
        st.markdown(f"#### All Models — {sel_city} / {sel_horiz}")
        if tb_all.empty:
            st.info("Track B evaluation CSV not found. Showing fallback data.")
            fallback_rows = []
            for m, met in FALLBACK_FORECASTING.items():
                fallback_rows.append({"Model": m, "R²": met["r2"],
                                       "MAE": met["mae"], "RMSE": met["rmse"]})
            st.dataframe(pd.DataFrame(fallback_rows), use_container_width=True, hide_index=True)
        else:
            try:
                h_alt = horiz_code.replace("h0", "").replace("h", "") + "h"
                comp_mask = (
                    (tb_all["city"] == sel_city) &
                    (tb_all["horizon"].isin([horiz_code, h_alt]))
                )
                comp_df = tb_all[comp_mask].copy()

                if comp_df.empty:
                    st.info(f"No comparison data for {sel_city} / {sel_horiz}.")
                else:
                    comp_df = comp_df.sort_values("r2", ascending=False)
                    disp = comp_df[
                        [c for c in ["model", "r2", "mae", "rmse", "n_test"]
                         if c in comp_df.columns]
                    ].copy()
                    disp["r2"]   = disp["r2"].round(4)
                    disp["mae"]  = disp["mae"].round(2)
                    disp["rmse"] = disp["rmse"].round(2)
                    st.dataframe(disp, use_container_width=True, hide_index=True)

                    try:
                        fig_comp = px.bar(
                            comp_df.sort_values("r2"),
                            x="r2", y="model", orientation="h",
                            title=f"R² Comparison — {sel_city} / {sel_horiz}",
                            color="r2",
                            color_continuous_scale="Viridis",
                            template="plotly_dark",
                        )
                        fig_comp.update_layout(
                            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                            font_color="#f0f2ff",
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Comparison chart error: {e}")
            except Exception as e:
                st.warning(f"Model comparison error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_research_analytics():
    st.title("📊 Research Analytics")
    st.markdown("Comprehensive statistical analysis across all models, cities, and horizons.")

    ta_df  = load_track_a()
    tb_df  = load_track_b_all()

    PLOT_TEMPLATE = "plotly_dark"

    tabs = st.tabs([
        "📐 Estimation Analytics",
        "📈 Forecasting Analytics",
        "💡 Insights",
    ])

    # ── Tab 1: Estimation Analytics ───────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Track A — Estimation Model Performance")
        if ta_df.empty:
            st.info("Track A CSV not available.")
        else:
            # City × Model R² heatmap
            try:
                pivot = safe_pivot(ta_df, "city", "model", "r2")
                if pivot.empty:
                    st.info("Heatmap data not available.")
                else:
                    fig_hm = px.imshow(
                        pivot,
                        title="City × Model R² Heatmap (Track A)",
                        color_continuous_scale="RdYlGn",
                        zmin=0, zmax=1,
                        template=PLOT_TEMPLATE,
                        aspect="auto",
                    )
                    fig_hm.update_layout(
                        paper_bgcolor="#0D1117", font_color="#f0f2ff",
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
            except Exception as e:
                st.warning(f"Heatmap error: {e}")

            # Model ranking bar chart
            try:
                if "model" in ta_df.columns and "r2" in ta_df.columns:
                    model_rank = (
                        ta_df.groupby("model")["r2"]
                        .mean()
                        .reset_index()
                        .sort_values("r2", ascending=False)
                    )
                    fig_mr = px.bar(
                        model_rank.sort_values("r2"),
                        x="r2", y="model", orientation="h",
                        title="Model Ranking by Average R² (Track A)",
                        color="r2",
                        color_continuous_scale="Blues",
                        template=PLOT_TEMPLATE,
                    )
                    fig_mr.update_layout(
                        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                        font_color="#f0f2ff",
                        xaxis=dict(range=[0, 1.05]),
                    )
                    st.plotly_chart(fig_mr, use_container_width=True)
            except Exception as e:
                st.warning(f"Ranking chart error: {e}")

            # Top-5 city × model combinations
            try:
                if all(c in ta_df.columns for c in ["city", "model", "r2", "mae", "rmse"]):
                    top5 = (
                        ta_df[["city", "model", "r2", "mae", "rmse"]]
                        .sort_values("r2", ascending=False)
                        .head(5)
                        .reset_index(drop=True)
                    )
                    top5["r2"]   = top5["r2"].round(4)
                    top5["mae"]  = top5["mae"].round(2)
                    top5["rmse"] = top5["rmse"].round(2)
                    st.markdown("**Top-5 City × Model Combinations**")
                    st.dataframe(top5, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Top-5 table error: {e}")

    # ── Tab 2: Forecasting Analytics ──────────────────────────────────────
    with tabs[1]:
        st.markdown("#### Track B — Forecasting Model Performance")
        if tb_df.empty:
            st.info("Track B CSVs not available.")
        else:
            # Model × Horizon R² heatmap
            try:
                if all(c in tb_df.columns for c in ["model", "horizon", "r2"]):
                    pivot_mh = safe_pivot(tb_df, "model", "horizon", "r2")
                    if pivot_mh.empty:
                        st.info("Model × Horizon heatmap data not available.")
                    else:
                        fig_mh = px.imshow(
                            pivot_mh,
                            title="Model × Horizon R² Heatmap (Track B)",
                            color_continuous_scale="RdYlGn",
                            template=PLOT_TEMPLATE,
                            aspect="auto",
                        )
                        fig_mh.update_layout(
                            paper_bgcolor="#0D1117", font_color="#f0f2ff",
                        )
                        st.plotly_chart(fig_mh, use_container_width=True)
            except Exception as e:
                st.warning(f"Model×Horizon heatmap error: {e}")

            # City × Model R² heatmap (forecasting)
            try:
                if all(c in tb_df.columns for c in ["city", "model", "r2"]):
                    pivot_cm = safe_pivot(tb_df, "city", "model", "r2")
                    if pivot_cm.empty:
                        st.info("City × Model heatmap data not available.")
                    else:
                        fig_cm = px.imshow(
                            pivot_cm,
                            title="City × Model R² Heatmap (Track B Forecasting)",
                            color_continuous_scale="RdYlGn",
                            template=PLOT_TEMPLATE,
                            aspect="auto",
                        )
                        fig_cm.update_layout(
                            paper_bgcolor="#0D1117", font_color="#f0f2ff",
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.warning(f"City×Model heatmap error: {e}")

            # Horizon degradation line chart
            try:
                if all(c in tb_df.columns for c in ["model", "horizon", "r2"]):
                    hz_line = (
                        tb_df.groupby(["model", "horizon"])["r2"]
                        .mean()
                        .reset_index()
                    )
                    # Normalise horizon values
                    hz_map_norm = {
                        "h01": "1h", "h06": "6h", "h24": "24h",
                        "1h": "1h", "6h": "6h", "24h": "24h",
                    }
                    hz_line["horizon_norm"] = hz_line["horizon"].map(
                        lambda x: hz_map_norm.get(x, x)
                    )
                    hz_order = {"1h": 0, "6h": 1, "24h": 2}
                    hz_line["hz_order"] = hz_line["horizon_norm"].map(
                        lambda x: hz_order.get(x, 99)
                    )
                    hz_line = hz_line.sort_values("hz_order")

                    fig_deg = px.line(
                        hz_line, x="horizon_norm", y="r2",
                        color="model", markers=True,
                        title="R² Degradation by Forecast Horizon",
                        template=PLOT_TEMPLATE,
                    )
                    fig_deg.update_layout(
                        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                        font_color="#f0f2ff",
                        xaxis_title="Horizon",
                        yaxis_title="Average R²",
                    )
                    st.plotly_chart(fig_deg, use_container_width=True)
            except Exception as e:
                st.warning(f"Horizon degradation chart error: {e}")

    # ── Tab 3: Insights ────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### 💡 Auto-Generated Research Insights")
        insights = []

        if not ta_df.empty and all(c in ta_df.columns for c in ["model", "r2"]):
            try:
                best_a = ta_df.groupby("model")["r2"].mean().idxmax()
                best_r2 = ta_df.groupby("model")["r2"].mean().max()
                insights.append(
                    f"🎯 Best estimation model: **{best_a}** with average R²={best_r2:.4f}"
                )
                worst_a = ta_df.groupby("model")["r2"].mean().idxmin()
                worst_r2 = ta_df.groupby("model")["r2"].mean().min()
                insights.append(
                    f"📉 Weakest estimation model: **{worst_a}** with average R²={worst_r2:.4f}"
                )
            except Exception:
                pass

        if not ta_df.empty and "city" in ta_df.columns:
            try:
                best_city = ta_df.groupby("city")["r2"].mean().idxmax()
                best_city_r2 = ta_df.groupby("city")["r2"].mean().max()
                insights.append(
                    f"🏙️ Best-performing city: **{best_city}** (avg R²={best_city_r2:.4f})"
                )
            except Exception:
                pass

        if not tb_df.empty and all(c in tb_df.columns for c in ["horizon", "r2"]):
            try:
                hz_avg = tb_df.groupby("horizon")["r2"].mean()
                for hz, r2v in hz_avg.items():
                    insights.append(f"⏱️ Horizon **{hz}** average R²={r2v:.4f}")
            except Exception:
                pass

        if not insights:
            insights = [
                "🎯 GradBoost achieves R²=0.9906 on estimation — best across all 18 cities.",
                "📉 DL models consistently underperform classical ML on both tasks.",
                "⏱️ Forecasting R² drops significantly beyond 1 h horizon.",
                "🔍 Leakage-free evaluation: Track B uses only lagged/rolling features.",
            ]

        for ins in insights:
            st.markdown(f"""
            <div style="background: #161B27; border-left: 3px solid #6C9EE8;
                        border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
                        color: #f0f2ff; font-size: 0.95rem;">
                {ins}
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INDIA PERFORMANCE MAP
# ══════════════════════════════════════════════════════════════════════════════

def page_india_map():
    st.title("🗺️ India Performance Map")
    st.markdown("Interactive geographic view of model performance across 18 Indian cities.")

    TASK_OPTIONS   = ["Estimation (Track A)", "Forecasting (Track B)"]
    METRIC_OPTIONS = ["R²", "MAE", "RMSE"]

    tc1, tc2, tc3, tc4 = st.columns(4)
    sel_task   = tc1.selectbox("Task",   TASK_OPTIONS,   key="map_task")
    sel_metric = tc2.selectbox("Metric", METRIC_OPTIONS, key="map_metric")

    is_est = "Estimation" in sel_task

    if is_est:
        model_options = list(ESTIMATION_MODEL_MAP.keys())
    else:
        model_options = list(FORECASTING_MODEL_MAP.keys())

    sel_model_disp = tc3.selectbox("Model", model_options, key="map_model")

    horiz_code_map = None
    if not is_est:
        horiz_disp     = tc4.selectbox("Horizon", HORIZONS, key="map_horizon")
        horiz_code_map = HORIZON_MAP[horiz_disp]

    # Load data
    if is_est:
        df_eval = load_track_a()
        csv_model = ESTIMATION_MODEL_MAP.get(sel_model_disp, sel_model_disp)
    else:
        df_eval = load_track_b_all()
        csv_model = FORECASTING_MODEL_MAP.get(sel_model_disp, sel_model_disp)

    metric_col = sel_metric.lower().replace("²", "2")  # r2, mae, rmse

    if df_eval.empty or "city" not in df_eval.columns:
        st.info("Performance data not available. Showing fallback metrics on map.")
        fallback_data = FALLBACK_ESTIMATION if is_est else FALLBACK_FORECASTING
        city_df = pd.DataFrame([
            {"city": c, "r2": fallback_data.get(csv_model, {}).get("r2", 0.5),
             "mae": fallback_data.get(csv_model, {}).get("mae", 20.0),
             "rmse": fallback_data.get(csv_model, {}).get("rmse", 30.0),
             "n_test": 4000}
            for c in CITIES
        ])
    else:
        try:
            mask = df_eval["model"] == csv_model
            if not is_est and horiz_code_map and "horizon" in df_eval.columns:
                h_alt = horiz_code_map.replace("h0", "").replace("h", "") + "h"
                mask = mask & df_eval["horizon"].isin([horiz_code_map, h_alt])

            agg_cols = {
                "r2":   ("r2",   "mean"),
                "mae":  ("mae",  "mean"),
                "rmse": ("rmse", "mean"),
            }
            if "n_test" in df_eval.columns:
                agg_cols["n_test"] = ("n_test", "mean")

            city_df = (
                df_eval[mask]
                .groupby("city")
                .agg(**agg_cols)
                .reset_index()
            )
        except Exception as e:
            st.warning(f"Data preparation error: {e}")
            city_df = pd.DataFrame()

    if city_df.empty:
        st.info("No data available for the selected configuration.")
        return

    # Add coordinates
    city_df["lat"] = city_df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    city_df["lon"] = city_df["city"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    city_df = city_df.dropna(subset=["lat", "lon"])

    if metric_col not in city_df.columns:
        metric_col = "r2"

    try:
        color_scale = "RdYlGn" if metric_col == "r2" else "RdYlGn_r"
        city_df["hover"] = city_df.apply(
            lambda row: (
                f"<b>{row['city']}</b><br>"
                f"Model: {sel_model_disp}<br>"
                f"R²: {row.get('r2', 'N/A'):.4f}<br>"
                f"MAE: {row.get('mae', 'N/A'):.2f}<br>"
                f"RMSE: {row.get('rmse', 'N/A'):.2f}<br>"
                f"Test Samples: {int(row.get('n_test', 0)) if 'n_test' in row.index else 'N/A'}"
            ), axis=1
        )

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattergeo(
            lat=city_df["lat"],
            lon=city_df["lon"],
            text=city_df["hover"],
            hoverinfo="text",
            mode="markers",
            marker=dict(
                size=16,
                color=city_df[metric_col],
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(title=sel_metric),
                line=dict(width=1, color="white"),
            ),
        ))
        fig_map.update_layout(
            title=f"{sel_model_disp} — {sel_metric} Across India",
            geo=dict(
                scope="asia",
                showland=True, landcolor="#1C2333",
                showocean=True, oceancolor="#0D1117",
                showcountries=True, countrycolor="#3a3d4e",
                showsubunits=True, subunitcolor="#3a3d4e",
                center=dict(lat=22.0, lon=79.0),
                projection_scale=3.5,
                bgcolor="#0D1117",
            ),
            paper_bgcolor="#0D1117",
            font_color="#f0f2ff",
            height=600,
        )
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.warning(f"Map rendering error: {e}")

    # City detail panel
    st.markdown("---")
    st.markdown("#### 🔍 City Detail")
    sel_detail_city = st.selectbox(
        "Select a city for details", sorted(city_df["city"].tolist()), key="map_detail_city"
    )
    city_row = city_df[city_df["city"] == sel_detail_city]
    if not city_row.empty:
        row = city_row.iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        for col, label, val, fmt in zip(
            [d1, d2, d3, d4],
            ["R²", "MAE", "RMSE", "Model"],
            [row.get("r2", "N/A"), row.get("mae", "N/A"), row.get("rmse", "N/A"), sel_model_disp],
            [".4f", ".2f", ".2f", "s"],
        ):
            v_str = f"{val:{fmt}}" if isinstance(val, float) else str(val)
            col.metric(label, v_str)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def page_model_comparison():
    st.title("🏆 Model Comparison")
    st.markdown("Side-by-side comparison of all models with detailed statistical breakdowns.")

    ta_df = load_track_a()
    tb_df = load_track_b_all()

    TASK_OPTIONS   = ["Estimation (Track A)", "Forecasting (Track B)"]
    METRIC_OPTIONS = ["r2", "mae", "rmse"]
    METRIC_LABELS  = {"r2": "R²", "mae": "MAE", "rmse": "RMSE"}

    fc1, fc2, fc3 = st.columns(3)
    sel_task   = fc1.selectbox("Task",   TASK_OPTIONS,   key="cmp_task")
    sel_metric = fc2.selectbox("Metric", METRIC_OPTIONS,
                               format_func=lambda x: METRIC_LABELS[x], key="cmp_metric")

    is_est = "Estimation" in sel_task

    if is_est:
        df_work = ta_df.copy() if not ta_df.empty else pd.DataFrame()
    else:
        horizon_sel = fc3.selectbox("Horizon", HORIZONS, key="cmp_horizon")
        horiz_code  = HORIZON_MAP[horizon_sel]
        df_work = tb_df.copy() if not tb_df.empty else pd.DataFrame()
        if not df_work.empty and "horizon" in df_work.columns:
            h_alt   = horiz_code.replace("h0", "").replace("h", "") + "h"
            df_work = df_work[df_work["horizon"].isin([horiz_code, h_alt])]

    if df_work.empty:
        st.info("No data available for the selected configuration.")
        fallback = FALLBACK_ESTIMATION if is_est else FALLBACK_FORECASTING
        rows = []
        for i, (model, metrics) in enumerate(
            sorted(fallback.items(), key=lambda x: -x[1]["r2"])
        ):
            medals = ["🥇", "🥈", "🥉"]
            rows.append({
                "Rank": medals[i] if i < 3 else str(i + 1),
                "Model": model,
                "R²": round(metrics["r2"], 4),
                "MAE": round(metrics["mae"], 2),
                "RMSE": round(metrics["rmse"], 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    # Leaderboard
    try:
        if "model" not in df_work.columns:
            st.info("No model column found in data.")
            return

        leaderboard = (
            df_work.groupby("model")[[sel_metric, "r2", "mae", "rmse"]]
            .mean()
            .reset_index()
            .sort_values(sel_metric, ascending=(sel_metric != "r2"))
            .reset_index(drop=True)
        )
        medals = ["🥇", "🥈", "🥉"] + [""] * max(0, len(leaderboard) - 3)
        leaderboard.insert(0, "Rank", medals)
        for c in ["r2", "mae", "rmse"]:
            if c in leaderboard.columns:
                leaderboard[c] = leaderboard[c].round(4 if c == "r2" else 2)
        leaderboard.columns = (
            ["Rank", "Model"] +
            [METRIC_LABELS.get(c, c) for c in leaderboard.columns[2:]]
        )
        st.markdown("#### 🏅 Leaderboard")
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Leaderboard error: {e}")

    # Bar chart
    try:
        bar_df = (
            df_work.groupby("model")[sel_metric]
            .mean()
            .reset_index()
            .sort_values(sel_metric, ascending=(sel_metric != "r2"))
        )
        if bar_df.empty:
            st.info("No chart data available.")
        else:
            fig_bar = px.bar(
                bar_df,
                x="model", y=sel_metric,
                title=f"Model Comparison — {METRIC_LABELS[sel_metric]}",
                color=sel_metric,
                color_continuous_scale="Blues" if sel_metric == "r2" else "Reds",
                template="plotly_dark",
            )
            fig_bar.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117", font_color="#f0f2ff",
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.warning(f"Bar chart error: {e}")

    # Box plot across cities
    try:
        if "city" in df_work.columns and len(df_work["city"].unique()) > 2:
            fig_box = px.box(
                df_work, x="model", y=sel_metric,
                title=f"Distribution of {METRIC_LABELS[sel_metric]} Across Cities",
                color="model",
                template="plotly_dark",
            )
            fig_box.update_layout(
                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117", font_color="#f0f2ff",
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)
    except Exception as e:
        st.warning(f"Box plot error: {e}")

    # Per-city grouped bar chart
    try:
        if "city" in df_work.columns:
            pivot_pc = safe_pivot(df_work, "city", "model", sel_metric)
            if not pivot_pc.empty:
                pivot_pc_reset = pivot_pc.reset_index()
                fig_grp = px.bar(
                    pivot_pc_reset.melt(id_vars="city", var_name="Model",
                                         value_name=METRIC_LABELS[sel_metric]),
                    x="city", y=METRIC_LABELS[sel_metric], color="Model",
                    barmode="group",
                    title=f"Per-City {METRIC_LABELS[sel_metric]} — All Models",
                    template="plotly_dark",
                )
                fig_grp.update_layout(
                    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117", font_color="#f0f2ff",
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_grp, use_container_width=True)
    except Exception as e:
        st.warning(f"Per-city chart error: {e}")

    # Key insight card
    try:
        best_model = (
            df_work.groupby("model")[sel_metric]
            .mean()
            .sort_values(ascending=(sel_metric != "r2"))
            .index[0]
        )
        best_val = df_work.groupby("model")[sel_metric].mean()[best_model]
        task_str = "estimation" if is_est else "forecasting"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #161B27, #1C2B1E);
                    border: 1px solid #00C853; border-radius: 12px; padding: 20px;
                    margin-top: 16px;">
            <h4 style="color: #00C853; margin: 0 0 8px 0;">💡 Key Insight</h4>
            <p style="color: #f0f2ff; margin: 0;">
                For {task_str}, <strong>{best_model}</strong> achieves the best
                {METRIC_LABELS[sel_metric]} of <strong>{best_val:.4f}</strong>
                averaged across all cities.
            </p>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Insight card error: {e}")

    # CSV download
    try:
        csv_bytes = df_work.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Data as CSV",
            data=csv_bytes,
            file_name=f"model_comparison_{'est' if is_est else 'fc'}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Download error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def page_about():
    st.title("📖 About This Research")

    tabs = st.tabs([
        "📋 Overview",
        "📁 Dataset",
        "⚙️ Methodology",
        "🤖 Models",
        "👥 Team",
    ])

    # ── Tab 1: Overview ────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        ## Project Overview

        ### Problem Statement
        Air Quality Index (AQI) prediction is critical for public health management in Indian cities.
        Traditional monitoring networks report AQI after measurement aggregation, introducing delays.
        This research develops ML/DL models for **real-time AQI estimation** and
        **multi-horizon forecasting** across 18 Indian cities.

        ### Research Gap
        Existing work focuses on single-city models or uses data with temporal leakage.
        This study is the first to:
        - Benchmark 7 models simultaneously across 18 Indian cities using CPCB data
        - Implement a strict **leakage-certified dual-track evaluation protocol**
        - Distinguish between estimation (Track A) and forecasting (Track B) with
          separate feature sets

        ### Key Contributions
        1. **18-city simultaneous evaluation** — largest CPCB multi-city ML benchmark
        2. **Leakage-certified evaluation** — Track B uses only lagged/rolling features
        3. **GBR superiority** — gradient boosting outperforms LSTM/BiLSTM on both tasks
        4. **Comprehensive comparison** — 7 models × 18 cities × 3 horizons = 378 experiments
        """)

    # ── Tab 2: Dataset ─────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        ## Dataset Description

        ### Source
        Central Pollution Control Board (CPCB) of India — official hourly air quality data.

        ### Coverage
        - **18 Cities**: Ahmedabad, Chennai, Delhi NCR, Gandhinagar, Hyderabad, Indore,
          Jaipur, Jodhpur, Mumbai, Mumbai Suburbs, Nagpur, Pune, Singrauli, Surat,
          Thane, Vapi, Bhopal, Vishakhapattanam
        - **Records**: ~18.7 million hourly observations
        - **Timespan**: 2018–2023

        ### Pollutants Measured
        | Pollutant | Unit | Description |
        |-----------|------|-------------|
        | PM2.5 | µg/m³ | Fine particulate matter |
        | PM10 | µg/m³ | Coarse particulate matter |
        | NO₂ | µg/m³ | Nitrogen dioxide |
        | SO₂ | µg/m³ | Sulfur dioxide |
        | CO | mg/m³ | Carbon monoxide |
        | O₃ | µg/m³ | Ground-level ozone |
        | NH₃ | µg/m³ | Ammonia |
        | NO | µg/m³ | Nitric oxide |
        | NOx | µg/m³ | Nitrogen oxides |

        ### Meteorological Variables
        Temperature, Humidity, Wind Speed, Wind Direction, Solar Radiation, Pressure
        """)

    # ── Tab 3: Methodology ─────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("## 8-Step Research Pipeline")
        steps = [
            ("1", "📥 Data Collection",
             "Raw CPCB hourly CSV files for 18 cities downloaded and consolidated."),
            ("2", "🔧 Preprocessing",
             "15-min → hourly resampling, 3-stage imputation, outlier handling, city isolation."),
            ("3", "⚗️ Feature Engineering",
             "88–115 features per city: raw pollutants, meteorology, temporal encoding, "
             "lag (1–24 h), rolling statistics (3/6/12/24 h windows)."),
            ("4", "✂️ Train/Val/Test Split",
             "Chronological 70/15/15 split — no shuffling to prevent temporal leakage."),
            ("5", "🏋️ Model Training",
             "7 models trained per city: Ridge, RF, GBR, XGB (Track A&B), "
             "LSTM, BiLSTM, CNN-BiLSTM (Track A&B)."),
            ("6", "📏 Evaluation",
             "R², MAE, RMSE on held-out test sets. No data seen during training."),
            ("7", "🔍 Leakage Audit",
             "Track B verified: only lag/rolling features used — no same-timestamp "
             "pollutant values that would not be available at forecast time."),
            ("8", "🚀 Dashboard",
             "Interactive Streamlit dashboard with 7 pages, CPCB calculator, "
             "maps, model comparison."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="background: #161B27; border: 1px solid #2a2d3e;
                        border-radius: 10px; padding: 16px; margin-bottom: 10px;
                        display: flex; gap: 16px;">
                <div style="min-width: 36px; height: 36px; background: #6C9EE8;
                            border-radius: 50%; display: flex; align-items: center;
                            justify-content: center; font-weight: 800; color: #0D1117;
                            font-size: 1rem;">{num}</div>
                <div>
                    <div style="color: #f0f2ff; font-weight: 700; margin-bottom: 4px;">
                        {title}
                    </div>
                    <div style="color: #9ca3af; font-size: 0.9rem;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 4: Models ──────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("## Models Used")
        ml_col, dl_col = st.columns(2)

        ml_col.markdown("""
        ### 🤖 Classical ML Models

        **Ridge Regression**
        Baseline linear model with L2 regularisation.
        Fast training, interpretable coefficients.

        **Random Forest (RF)**
        Ensemble of 200 decision trees with bootstrapping.
        Handles non-linearity and feature interactions.

        **Gradient Boosting (GBR/GradBoost)**
        Sequential boosting: 200 estimators, lr=0.1, max_depth=5.
        **Best overall performance** on both tracks.

        **XGBoost**
        Regularised gradient boosting: 200 rounds, lr=0.05, max_depth=6.
        Second-best estimation model (R²=0.9856).
        """)

        dl_col.markdown("""
        ### 🧠 Deep Learning Models

        **LSTM**
        Long Short-Term Memory recurrent network.
        2-layer architecture with dropout regularisation.
        Sequence length 24, trained up to 100 epochs with early stopping.

        **BiLSTM**
        Bidirectional LSTM — processes sequences forward and backward.
        Better captures temporal dependencies in both directions.

        **CNN-BiLSTM**
        Convolutional feature extraction + BiLSTM for sequence modelling.
        1D CNN kernel extracts local patterns before LSTM layers.
        Underperforms classical ML in this dataset context.
        """)

    # ── Tab 5: Team ────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("## 👥 Research Team")
        st.markdown("""
        <div style="background: #161B27; border: 1px solid #2a2d3e;
                    border-radius: 16px; padding: 32px; max-width: 600px;">
            <div style="text-align: center; margin-bottom: 24px;">
                <div style="font-size: 4rem;">👤</div>
                <h2 style="color: #f0f2ff; margin: 8px 0 4px 0;">Aman Gajbhiye</h2>
                <div style="color: #6C9EE8; font-size: 1rem; margin-bottom: 8px;">
                    Research Intern
                </div>
                <div style="color: #9ca3af;">
                    Yeshwantrao Chavan College of Engineering (YCCE), Nagpur
                </div>
                <div style="color: #9ca3af; margin-top: 4px;">
                    Research conducted at: <strong style="color: #f0f2ff;">
                        IIIT Nagpur
                    </strong>
                </div>
            </div>
            <hr style="border-color: #2a2d3e;">
            <div style="margin-top: 16px;">
                <div style="color: #9ca3af; margin-bottom: 8px;">
                    📊 <strong style="color: #f0f2ff;">Project:</strong>
                    AQI Prediction Using ML and DL
                </div>
                <div style="color: #9ca3af; margin-bottom: 8px;">
                    🏙️ <strong style="color: #f0f2ff;">Scale:</strong>
                    18 Indian cities, 7 models, 3 forecast horizons
                </div>
                <div style="color: #9ca3af; margin-bottom: 8px;">
                    🗄️ <strong style="color: #f0f2ff;">Data:</strong>
                    18.7M records from CPCB
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        btn1, btn2, btn3 = st.columns(3)
        btn1.link_button(
            "🔗 GitHub Repository",
            "https://github.com/amangajbhiye99/aqi-prediction",
        )
        btn2.link_button(
            "📊 CPCB Dataset",
            "https://www.cpcb.nic.in",
        )
        btn3.link_button(
            "📄 Research Paper",
            "https://github.com/amangajbhiye99/aqi-prediction",
        )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — SIDEBAR NAVIGATION & ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("## 🌫️ AQI Research")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            options=[
                "🏠 Home",
                "🔬 AQI Estimation",
                "📈 Forecast Analysis",
                "📊 Research Analytics",
                "🗺️ India Performance Map",
                "🏆 Model Comparison",
                "📖 About",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        theme = st.radio(
            "🎨 Theme",
            ["🌙 Dark", "☀️ Light"],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("IIIT Nagpur Research Internship")
        st.caption("Aman Gajbhiye · YCCE")

    light_mode = (theme == "☀️ Light")
    inject_css(light_mode=light_mode)

    PLOT_TEMPLATE = "plotly_white" if light_mode else "plotly_dark"
    # (Template is defined here as a module-level fallback for pages that need it)

    if page == "🏠 Home":
        page_home()
    elif page == "🔬 AQI Estimation":
        page_estimation()
    elif page == "📈 Forecast Analysis":
        page_forecast_analysis()
    elif page == "📊 Research Analytics":
        page_research_analytics()
    elif page == "🗺️ India Performance Map":
        page_india_map()
    elif page == "🏆 Model Comparison":
        page_model_comparison()
    elif page == "📖 About":
        page_about()


if __name__ == "__main__":
    main()
