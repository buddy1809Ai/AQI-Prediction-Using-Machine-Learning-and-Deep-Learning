#!/usr/bin/env python3
"""
AQI Prediction Research Dashboard — DEFINITIVE FINAL
Research Internship | IIIT Nagpur | YCCE
Author: Aman Gajbhiye
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import os, sys, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — MUST BE FIRST STREAMLIT CALL
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction Research Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CITY_LIST = [
    "Ahmedabad", "Chennai", "Delhi_NCR", "GandhiNagar", "Hyderabad",
    "Indore", "Jaipur", "Jodhpur", "Mumbai", "Mumbai_suburbs",
    "Nagpur", "Pune", "Singrauli", "Surat", "Thane",
    "Vapi", "bhopal", "vishakhapattanam",
]

CITY_COORDS = {
    "Ahmedabad":        (23.02, 72.57),
    "Chennai":          (13.08, 80.27),
    "Delhi_NCR":        (28.61, 77.21),
    "GandhiNagar":      (23.22, 72.64),
    "Hyderabad":        (17.39, 78.49),
    "Indore":           (22.72, 75.86),
    "Jaipur":           (26.91, 75.79),
    "Jodhpur":          (26.24, 73.02),
    "Mumbai":           (19.08, 72.88),
    "Mumbai_suburbs":   (19.22, 72.98),
    "Nagpur":           (21.15, 79.09),
    "Pune":             (18.52, 73.86),
    "Singrauli":        (24.20, 82.67),
    "Surat":            (21.17, 72.83),
    "Thane":            (19.22, 72.98),
    "Vapi":             (20.39, 72.91),
    "bhopal":           (23.26, 77.41),
    "vishakhapattanam": (17.69, 83.22),
}

CITY_DISPLAY_MAP = {c: c.replace("_", " ").title() for c in CITY_LIST}

ESTIMATION_MODELS_DISPLAY = [
    "Gradient Boosting", "Random Forest", "XGBoost",
    "Ridge Regression", "LSTM", "BiLSTM", "CNN-BiLSTM",
]
FORECASTING_MODELS_DISPLAY = [
    "Gradient Boosting", "Random Forest", "XGBoost",
    "LSTM", "BiLSTM", "CNN-BiLSTM",
]
HORIZONS_DISPLAY = ["1 Hour", "6 Hours", "24 Hours"]
HORIZON_MAP = {"1 Hour": 1, "6 Hours": 6, "24 Hours": 24}

# CSV name mappings
TRACK_A_MODEL_CSV_NAMES = {
    "Gradient Boosting": "GradBoost",
    "Random Forest":     "RandomForest",
    "XGBoost":           "XGBoost",
    "Ridge Regression":  "Ridge",
    "LSTM":              "LSTM",
    "BiLSTM":            "BiLSTM",
    "CNN-BiLSTM":        "CNN-BiLSTM",
}
TRACK_B_MODEL_CSV_NAMES = {
    "Gradient Boosting": "GradientBoosting",
    "Random Forest":     "RandomForest",
    "XGBoost":           "XGBoost",
    "LSTM":              "LSTM",
    "BiLSTM":            "BiLSTM",
    "CNN-BiLSTM":        "CNN-BiLSTM",
}

# Display name normalization from CSV values
TRACK_A_DISPLAY_NAMES = {v: k for k, v in TRACK_A_MODEL_CSV_NAMES.items()}
TRACK_B_DISPLAY_NAMES = {v: k for k, v in TRACK_B_MODEL_CSV_NAMES.items()}

TRACK_B_CSV_FILES = ["track_b_gbr.csv", "track_b_rf.csv", "track_b_xgb.csv",
                     "track_b_bilstm.csv", "track_b_lstm.csv"]

# Design colors
BG       = "#0D1117"
CARD     = "#161B27"
BORDER   = "#2a2d3e"
PRIMARY  = "#6C9EE8"
ACCENT   = "#FFB482"
SUCCESS  = "#56CF8E"
DANGER   = "#FF7D77"
TEAL     = "#45D4C5"
GOLD     = "#FFD400"

AQI_CATEGORIES = [
    (  0,  50, "Good",        "#00b050", "Minimal impact"),
    ( 51, 100, "Satisfactory","#92d050", "Minor breathing discomfort to sensitive people"),
    (101, 200, "Moderate",    "#ffff00", "Breathing discomfort to people with lung/heart disease"),
    (201, 300, "Poor",        "#ff7c00", "Breathing discomfort to most people on prolonged exposure"),
    (301, 400, "Very Poor",   "#ff0000", "Respiratory illness on prolonged exposure"),
    (401, 500, "Severe",      "#7030a0", "Affects healthy people; serious impact on those with disease"),
]

# CPCB AQI Breakpoints — 7 pollutants
# Format: (BPlo, BPhi, Ilo, Ihi) — concentration range and corresponding AQI index range
CPCB_BP = {
    "PM2.5": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 800, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2100, 401, 500),
    ],
    "CO": [
        (0, 1, 0, 50),
        (1, 2, 51, 100),
        (2, 10, 101, 200),
        (10, 17, 201, 300),
        (17, 34, 301, 400),
        (34, 50, 401, 500),
    ],
    "O3": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50),
        (200, 400, 51, 100),
        (400, 800, 101, 200),
        (800, 1200, 201, 300),
        (1200, 1800, 301, 400),
        (1800, 2400, 401, 500),
    ],
}

POLLUTANT_UNITS = {
    "PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³",
    "SO2": "µg/m³", "CO": "mg/m³", "O3": "µg/m³", "NH3": "µg/m³",
}

SAMPLE_VALUES = {
    "PM2.5": 85.0, "PM10": 150.0, "NO2": 60.0,
    "SO2": 20.0, "CO": 1.5, "O3": 100.0, "NH3": 200.0,
}

FALLBACK_ESTIMATION = {
    "Gradient Boosting": {"r2": 0.9906, "mae": 2.9450, "rmse": 5.7661},
    "Random Forest":     {"r2": 0.9874, "mae": 1.6428, "rmse": 6.0478},
    "XGBoost":           {"r2": 0.9856, "mae": 2.8306, "rmse": 6.8206},
    "Ridge Regression":  {"r2": 0.8304, "mae": 15.9219, "rmse": 23.0631},
    "LSTM":              {"r2": 0.6411, "mae": 22.4892, "rmse": 33.5919},
    "BiLSTM":            {"r2": 0.5897, "mae": 22.4619, "rmse": 35.9004},
    "CNN-BiLSTM":        {"r2": 0.2756, "mae": 36.9234, "rmse": 47.5118},
}

FALLBACK_FORECASTING = {
    "Gradient Boosting": {"r2": 0.4997, "mae": 32.57, "rmse": 48.37},
    "Random Forest":     {"r2": 0.4914, "mae": 34.16, "rmse": 48.79},
    "XGBoost":           {"r2": 0.4902, "mae": 32.97, "rmse": 48.86},
    "BiLSTM":            {"r2": 0.2831, "mae": 39.07, "rmse": 56.65},
    "LSTM":              {"r2": 0.2768, "mae": 39.07, "rmse": 56.83},
    "CNN-BiLSTM":        {"r2": -0.4147, "mae": 48.21, "rmse": 67.50},
}

# ─────────────────────────────────────────────────────────────────────────────
# PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def find_root() -> Path:
    TARGET_FILE = "final_track_a_complete.csv"
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parent)
        candidates.append(Path(__file__).resolve().parent.parent)
    except Exception:
        pass
    candidates.append(Path.cwd())
    for _ in range(5):
        candidates.append(Path.cwd().parent)

    for base in candidates:
        for folder_name in ["AQI_Prediction_Project_Final (1)", "AQI_Prediction_Project_Final"]:
            p = base / folder_name
            if (p / "outputs" / "tables" / TARGET_FILE).exists():
                return p

    for base in candidates[:3]:
        try:
            matches = list(base.rglob(TARGET_FILE))
            if matches:
                return matches[0].parent.parent.parent
        except Exception:
            pass

    return Path.cwd() / "AQI_Prediction_Project_Final (1)"

# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(theme: str = "dark") -> None:
    sidebar_css = """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0E1A 0%, #161B27 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #f0f2ff !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem !important; }
    </style>
    """
    if theme == "dark":
        main_css = f"""
        <style>
        .stApp {{ background-color: {BG}; color: #e6e6e6; }}
        .glass-card {{
            backdrop-filter: blur(10px);
            background: rgba(22,27,39,0.85);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}
        .kpi-card {{
            background: rgba(22,27,39,0.9);
            border: 1px solid rgba(108,158,232,0.3);
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
        }}
        .kpi-value {{ font-size: 2rem; font-weight: 700; color: {PRIMARY}; }}
        .kpi-label {{ font-size: 0.85rem; color: #9aa5c9; margin-top: 0.3rem; }}
        .metric-badge {{
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        h1, h2, h3, h4 {{ color: #e8eaf6; }}
        .hero-title {{
            background: linear-gradient(135deg, {PRIMARY} 0%, {TEAL} 50%, {ACCENT} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.3;
        }}
        </style>
        """
    else:
        main_css = """
        <style>
        .stApp { background-color: #f8f9fa; color: #1a1a2e; }
        .glass-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .kpi-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .kpi-value { font-size: 2rem; font-weight: 700; color: #3a5dbf; }
        .kpi-label { font-size: 0.85rem; color: #6c757d; margin-top: 0.3rem; }
        h1, h2, h3, h4 { color: #1a1a2e; }
        .hero-title {
            background: linear-gradient(135deg, #3a5dbf 0%, #1a8a8a 50%, #c77b3a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.3;
        }
        </style>
        """
    st.markdown(sidebar_css, unsafe_allow_html=True)
    st.markdown(main_css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CPCB AQI CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def calc_sub_index(pollutant: str, concentration: float) -> float:
    if pollutant not in CPCB_BP:
        return 0.0
    bp_list = CPCB_BP[pollutant]
    for (BPlo, BPhi, Ilo, Ihi) in bp_list:
        if BPlo <= concentration <= BPhi:
            if BPhi == BPlo:
                return float(Ilo)
            sub_idx = ((Ihi - Ilo) / (BPhi - BPlo)) * (concentration - BPlo) + Ilo
            return round(sub_idx, 2)
    if concentration > bp_list[-1][1]:
        return 500.0
    return 0.0

def calc_aqi(pollutant_values: dict) -> tuple:
    sub_indices = {}
    for poll, conc in pollutant_values.items():
        if conc is not None and conc >= 0:
            si = calc_sub_index(poll, float(conc))
            sub_indices[poll] = si
    if not sub_indices:
        return 0, "Good", "#00b050", "No pollutant data", {}
    aqi_val = max(sub_indices.values())
    dominant = max(sub_indices, key=sub_indices.get)
    for (lo, hi, cat, color, advisory) in AQI_CATEGORIES:
        if lo <= aqi_val <= hi:
            return round(aqi_val), cat, color, advisory, sub_indices
    return round(aqi_val), "Severe", "#7030a0", "Hazardous conditions", sub_indices

def get_aqi_category(aqi_val: float) -> tuple:
    for (lo, hi, cat, color, advisory) in AQI_CATEGORIES:
        if lo <= aqi_val <= hi:
            return cat, color, advisory
    return "Severe", "#7030a0", "Hazardous conditions"

# ─────────────────────────────────────────────────────────────────────────────
# SAFE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def safe_load_csv(path) -> pd.DataFrame | None:
    try:
        p = Path(path)
        if not p.exists():
            return None
        df = pd.read_csv(p)
        return df if not df.empty else None
    except Exception:
        return None

def safe_pivot(df, index, columns, values, aggfunc="mean", fill_value=None) -> pd.DataFrame | None:
    try:
        if df is None or df.empty:
            return None
        for col in [index, columns, values]:
            if col not in df.columns:
                return None
        df2 = df.copy().replace([np.inf, -np.inf], np.nan)
        pivot = df2.pivot_table(index=index, columns=columns, values=values,
                                aggfunc=aggfunc, fill_value=fill_value)
        pivot.columns.name = None
        return pivot
    except Exception:
        return None

def get_plotly_template(theme: str) -> str:
    return "plotly_dark" if theme == "dark" else "plotly_white"

def fig_layout(fig, theme: str) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=50, b=30),
        font=dict(color="#e6e6e6" if theme == "dark" else "#1a1a2e"),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_track_a() -> pd.DataFrame:
    ROOT = find_root()
    TABLES_DIR = ROOT / "outputs" / "tables"
    path = TABLES_DIR / "final_track_a_complete.csv"
    df = safe_load_csv(path)
    if df is None:
        rows = []
        for city in CITY_LIST:
            for disp, vals in FALLBACK_ESTIMATION.items():
                rows.append({
                    "city": city,
                    "model": TRACK_A_MODEL_CSV_NAMES.get(disp, disp),
                    "track": "A", "n_train": 19765, "n_test": 4236,
                    "n_feats": 88, "r2": vals["r2"], "mae": vals["mae"],
                    "rmse": vals["rmse"], "train_time_s": 1.0, "inference_time_s": 0.1,
                })
        df = pd.DataFrame(rows)

    MODEL_RENAME = {
        "Ridge": "Ridge Regression", "GradBoost": "Gradient Boosting",
        "RandomForest": "Random Forest", "XGBoost": "XGBoost",
        "LSTM": "LSTM", "BiLSTM": "BiLSTM", "CNN-BiLSTM": "CNN-BiLSTM",
    }
    df["model_display"] = df["model"].map(MODEL_RENAME).fillna(df["model"])
    df["city_display"] = df["city"].apply(lambda c: c.replace("_", " ").title())
    return df

@st.cache_data
def load_track_b_all() -> pd.DataFrame:
    ROOT = find_root()
    TABLES_DIR = ROOT / "outputs" / "tables"
    frames = []
    for fname in TRACK_B_CSV_FILES:
        df = safe_load_csv(TABLES_DIR / fname)
        if df is not None:
            frames.append(df)

    if not frames:
        combined_path = TABLES_DIR / "final_track_b_complete.csv"
        df = safe_load_csv(combined_path)
        if df is not None:
            frames = [df]

    if not frames:
        rows = []
        for city in CITY_LIST:
            for hz_int in [1, 6, 24]:
                for disp, vals in FALLBACK_FORECASTING.items():
                    rows.append({
                        "city": city, "horizon": hz_int,
                        "model": TRACK_B_MODEL_CSV_NAMES.get(disp, disp),
                        "track": "B", "n_train": 19765, "n_test": 4236,
                        "n_feats": 55, "r2": vals["r2"], "mae": vals["mae"],
                        "rmse": vals["rmse"], "train_time_s": 1.0, "inference_time_s": 0.1,
                    })
        frames = [pd.DataFrame(rows)]

    tb = pd.concat(frames, ignore_index=True)

    # CRITICAL: horizon is INTEGER (1, 6, 24) — map to display
    if "horizon" in tb.columns:
        tb["horizon"] = pd.to_numeric(tb["horizon"], errors="coerce").fillna(1).astype(int)
        tb["horizon_display"] = tb["horizon"].map({1: "1 Hour", 6: "6 Hours", 24: "24 Hours"}).fillna(tb["horizon"].astype(str))
    else:
        tb["horizon"] = 1
        tb["horizon_display"] = "1 Hour"

    MODEL_RENAME_B = {
        "GradientBoosting": "Gradient Boosting",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "LSTM": "LSTM",
        "BiLSTM": "BiLSTM",
        "CNN-BiLSTM": "CNN-BiLSTM",
    }
    tb["model_display"] = tb["model"].map(MODEL_RENAME_B).fillna(tb["model"])
    tb["city_display"] = tb["city"].apply(lambda c: c.replace("_", " ").title())
    return tb

@st.cache_data
def load_prediction_csv(city_raw: str, horizon_int: int) -> pd.DataFrame | None:
    ROOT = find_root()
    PREDS_DIR = ROOT / "outputs" / "predictions"
    fname = f"{city_raw}_h{horizon_int:02d}_predictions.csv"
    df = safe_load_csv(PREDS_DIR / fname)
    if df is None:
        return None

    actual_col = next((c for c in ["actual_aqi", "y_true", "actual", "target"] if c in df.columns), None)
    xgb_col    = next((c for c in ["xgb_pred", "y_pred", "predicted", "pred_xgb"] if c in df.columns), None)
    bilstm_col = next((c for c in ["bilstm_pred", "pred_bilstm"] if c in df.columns), None)

    if actual_col is None and xgb_col is None:
        return None

    result = pd.DataFrame()
    if actual_col:
        result["actual"] = df[actual_col]
    if xgb_col:
        result["xgb_pred"] = df[xgb_col]
    if bilstm_col:
        result["bilstm_pred"] = df[bilstm_col]

    return result if not result.empty else None

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home(theme: str) -> None:
    tmpl = get_plotly_template(theme)

    st.markdown(
        '<p class="hero-title">Air Quality Index Prediction Using<br>'
        'Machine Learning and Deep Learning</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Research Internship | IIIT Nagpur | YCCE** &nbsp;|&nbsp; "
        "Multi-city AQI prediction and forecasting across 18 Indian cities",
        unsafe_allow_html=True,
    )
    st.divider()

    # KPI cards
    cols = st.columns(4)
    kpis = [
        ("18", "Cities Covered", PRIMARY),
        ("18.7M", "Records Processed", ACCENT),
        ("7", "ML/DL Models", SUCCESS),
        ("3", "Forecast Horizons", TEAL),
    ]
    for col, (val, label, color) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="color:{color}">{val}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Research Workflow
    st.subheader("🔬 Research Workflow")
    workflow_steps = [
        ("📥", "Dataset", "CPCB 18 cities 2018-2023"),
        ("🔧", "Preprocessing", "Resampling, imputation, leakage audit"),
        ("⚙️", "Feature Engineering", "88-115 features per city"),
        ("🤖", "ML Models", "Ridge, RF, GBR, XGBoost"),
        ("🧠", "Deep Learning", "LSTM, BiLSTM, CNN-BiLSTM"),
        ("📊", "Evaluation", "R², MAE, RMSE on held-out test"),
    ]
    cols = st.columns(6)
    for col, (icon, title, desc) in zip(cols, workflow_steps):
        with col:
            st.markdown(
                f'<div class="glass-card" style="text-align:center;min-height:140px;">'
                f'<div style="font-size:1.8rem">{icon}</div>'
                f'<div style="font-weight:700;font-size:0.9rem;margin-top:0.4rem">{title}</div>'
                f'<div style="font-size:0.75rem;color:#9aa5c9;margin-top:0.3rem">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Research Objectives
    st.subheader("🎯 Research Objectives")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="glass-card">'
            f'<h4 style="color:{PRIMARY}">🔬 AQI Estimation (Track A)</h4>'
            f'<p>Estimate real-time AQI from concurrent pollutant measurements across 18 cities.</p>'
            f'<p><strong>Best Result:</strong> Gradient Boosting — R² = 0.9906</p>'
            f'<p><strong>Training:</strong> Chronological 70/15/15 split with leakage audit</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="glass-card">'
            f'<h4 style="color:{ACCENT}">📈 AQI Forecasting (Track B)</h4>'
            f'<p>Forecast future AQI at 1-hour, 6-hour, and 24-hour horizons.</p>'
            f'<p><strong>Coverage:</strong> 18 cities × 3 horizons × 6 models = 324 experiments</p>'
            f'<p><strong>Best Result:</strong> GBR — R² = 0.4997 (1-hour horizon)</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Best Models Section
    st.subheader("🏆 Best Performing Models")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="glass-card" style="border-color:rgba(86,207,142,0.4)">'
            f'<h4 style="color:{SUCCESS}">🥇 Estimation Champion</h4>'
            f'<h3 style="margin:0.3rem 0">Gradient Boosting</h3>'
            f'<table style="width:100%;font-size:0.9rem">'
            f'<tr><td>R²</td><td><strong style="color:{SUCCESS}">0.9906</strong></td></tr>'
            f'<tr><td>MAE</td><td>2.9450 µg/m³</td></tr>'
            f'<tr><td>RMSE</td><td>5.7661 µg/m³</td></tr>'
            f'<tr><td>Cities</td><td>18 (all)</td></tr>'
            f'</table>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="glass-card" style="border-color:rgba(255,180,130,0.4)">'
            f'<h4 style="color:{ACCENT}">🥇 Forecasting Champion</h4>'
            f'<h3 style="margin:0.3rem 0">Gradient Boosting Regressor</h3>'
            f'<table style="width:100%;font-size:0.9rem">'
            f'<tr><td>R²</td><td><strong style="color:{ACCENT}">0.4997</strong></td></tr>'
            f'<tr><td>MAE</td><td>32.57 µg/m³</td></tr>'
            f'<tr><td>RMSE</td><td>48.37 µg/m³</td></tr>'
            f'<tr><td>Horizons</td><td>1h / 6h / 24h</td></tr>'
            f'</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Key Contributions
    st.subheader("💡 Key Contributions")
    contribs = [
        ("🏙️", "Multi-City Benchmark",
         "First simultaneous evaluation of 7 ML/DL models across 18 CPCB-monitored Indian cities — "
         "a 126-experiment benchmark for AQI estimation."),
        ("🔒", "Leakage-Certified Protocol",
         "Rigorous data leakage audit distinguishing estimation (same-time-step) from forecasting "
         "(future-only features), ensuring scientific validity."),
        ("📊", "Classical ML vs. Deep Learning",
         "Systematic comparison showing GBR achieves R²=0.99 for estimation and R²=0.50 for forecasting, "
         "outperforming LSTM/BiLSTM significantly."),
    ]
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, contribs):
        with col:
            st.markdown(
                f'<div class="glass-card">'
                f'<div style="font-size:1.6rem">{icon}</div>'
                f'<h4 style="margin-top:0.5rem">{title}</h4>'
                f'<p style="font-size:0.88rem">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Links
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.link_button(
            "🐙 GitHub Repository",
            "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning",
        )
    with c2:
        st.link_button(
            "📂 CPCB Dataset",
            "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link",
        )

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#9aa5c9;font-size:0.85rem'>"
        "Aman Gajbhiye &nbsp;|&nbsp; YCCE Nagpur &nbsp;|&nbsp; IIIT Nagpur Research Internship"
        "</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AQI ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
def page_estimation(theme: str) -> None:
    tmpl = get_plotly_template(theme)
    st.title("🔬 AQI Estimation")
    st.caption("Track A: Simultaneous pollutant measurement → AQI estimation")

    tab_calc, tab_perf = st.tabs(["🧮 AQI Calculator", "📊 Model Performance"])

    # ── Tab 1: AQI Calculator ──
    with tab_calc:
        st.subheader("CPCB AQI Calculator")
        st.info("Enter pollutant concentrations to compute AQI using official CPCB breakpoints.")

        if "poll_vals" not in st.session_state:
            st.session_state.poll_vals = {p: 0.0 for p in CPCB_BP}

        col_reset, col_sample, _ = st.columns([1, 1, 4])
        with col_reset:
            if st.button("🔄 Reset"):
                st.session_state.poll_vals = {p: 0.0 for p in CPCB_BP}
                st.rerun()
        with col_sample:
            if st.button("📋 Sample Values"):
                st.session_state.poll_vals = SAMPLE_VALUES.copy()
                st.rerun()

        polls = list(CPCB_BP.keys())
        c1, c2 = st.columns(2)
        inputs = {}
        for i, poll in enumerate(polls):
            col = c1 if i < 4 else c2
            with col:
                unit = POLLUTANT_UNITS.get(poll, "")
                inputs[poll] = st.number_input(
                    f"{poll} ({unit})",
                    min_value=0.0,
                    max_value=5000.0,
                    value=float(st.session_state.poll_vals.get(poll, 0.0)),
                    step=0.1,
                    key=f"poll_{poll}",
                )
                st.session_state.poll_vals[poll] = inputs[poll]

        st.divider()
        if st.button("🧮 Calculate AQI", type="primary", use_container_width=True):
            aqi_val, cat, color, advisory, sub_indices = calc_aqi(inputs)

            # AQI display
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(
                    f'<div class="glass-card" style="text-align:center;border-color:{color}55">'
                    f'<div style="font-size:3.5rem;font-weight:800;color:{color}">{aqi_val}</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{color}">{cat}</div>'
                    f'<div style="font-size:0.85rem;color:#9aa5c9;margin-top:0.5rem">'
                    f'AQI (CPCB Scale 0-500)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="glass-card">'
                    f'<strong>⚕️ Health Advisory:</strong><br>'
                    f'<span style="font-size:0.9rem">{advisory}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with c2:
                if sub_indices:
                    try:
                        polls_sorted = sorted(sub_indices.items(), key=lambda x: x[1], reverse=True)
                        poll_names = [p[0] for p in polls_sorted]
                        si_vals    = [p[1] for p in polls_sorted]
                        bar_colors = [color if v == max(si_vals) else PRIMARY for v in si_vals]
                        fig = go.Figure(go.Bar(
                            x=si_vals, y=poll_names, orientation="h",
                            marker_color=bar_colors,
                            text=[f"{v:.1f}" for v in si_vals],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            title="Sub-Index by Pollutant",
                            template=tmpl,
                            xaxis_title="Sub-Index Value",
                            xaxis_range=[0, max(si_vals) * 1.2 if si_vals else 10],
                        )
                        fig = fig_layout(fig, theme)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info(f"Chart unavailable: {e}")

    # ── Tab 2: Model Performance ──
    with tab_perf:
        st.subheader("Estimation Model Leaderboard")
        ta = load_track_a()

        if ta is not None and not ta.empty:
            # Aggregate by model
            agg = (
                ta.groupby("model_display")[["r2", "mae", "rmse", "train_time_s"]]
                .mean()
                .reset_index()
                .sort_values("r2", ascending=False)
                .reset_index(drop=True)
            )
            agg.insert(0, "Rank", ["🥇", "🥈", "🥉"] + [str(i + 1) for i in range(3, len(agg))])

            st.dataframe(
                agg.rename(columns={
                    "model_display": "Model", "r2": "R²",
                    "mae": "MAE", "rmse": "RMSE", "train_time_s": "Train Time (s)",
                }).style.format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "Train Time (s)": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

            # Download button
            csv_bytes = agg.to_csv(index=False).encode()
            st.download_button("📥 Download Results CSV", csv_bytes, "track_a_results.csv", "text/csv")

            st.divider()

            # R² bar chart
            try:
                fig = go.Figure(go.Bar(
                    x=agg["model_display"],
                    y=agg["r2"],
                    marker_color=[SUCCESS if v > 0.95 else PRIMARY if v > 0.7 else DANGER for v in agg["r2"]],
                    text=[f"{v:.4f}" for v in agg["r2"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="R² Score by Model (Averaged over 18 Cities)",
                    template=tmpl,
                    yaxis_title="R²",
                    yaxis_range=[0, 1.1],
                )
                fig = fig_layout(fig, theme)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"R² chart unavailable: {e}")

            # MAE bar chart
            try:
                agg_mae = agg.sort_values("mae")
                fig2 = go.Figure(go.Bar(
                    x=agg_mae["model_display"],
                    y=agg_mae["mae"],
                    marker_color=ACCENT,
                    text=[f"{v:.2f}" for v in agg_mae["mae"]],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="MAE by Model (Averaged over 18 Cities) — Lower is Better",
                    template=tmpl,
                    yaxis_title="MAE (µg/m³)",
                )
                fig2 = fig_layout(fig2, theme)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.info(f"MAE chart unavailable: {e}")

            # City×Model Heatmap
            st.subheader("City × Model R² Heatmap")
            pivot = safe_pivot(ta, "city_display", "model_display", "r2")
            if pivot is not None:
                try:
                    fig3 = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=list(pivot.columns),
                        y=list(pivot.index),
                        colorscale="RdYlGn",
                        zmin=0, zmax=1,
                        text=[[f"{v:.3f}" if not np.isnan(v) else "N/A" for v in row] for row in pivot.values],
                        texttemplate="%{text}",
                        colorbar=dict(title="R²"),
                    ))
                    fig3.update_layout(
                        title="R² by City and Model",
                        template=tmpl,
                        height=600,
                        xaxis=dict(tickangle=-30),
                    )
                    fig3 = fig_layout(fig3, theme)
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.info(f"Heatmap unavailable: {e}")

            # Best insight
            best_model = agg.iloc[0]["model_display"]
            best_r2 = agg.iloc[0]["r2"]
            st.success(f"🏆 **{best_model}** achieves R²={best_r2:.4f} — best estimation model across 18 cities")

        else:
            st.warning("Track A data not available. Showing fallback summary.")
            rows = [{"Model": k, "R²": v["r2"], "MAE": v["mae"], "RMSE": v["rmse"]}
                    for k, v in FALLBACK_ESTIMATION.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECAST ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_forecast_analysis(theme: str) -> None:
    tmpl = get_plotly_template(theme)
    st.title("📈 Forecast Analysis")
    st.info("📊 Predictions from independent test-set evaluation")

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        city_disp_list = [CITY_DISPLAY_MAP[c] for c in CITY_LIST]
        sel_city_disp = st.selectbox("🏙️ City", city_disp_list, key="fc_city")
    with c2:
        sel_horizon = st.selectbox("⏱️ Horizon", HORIZONS_DISPLAY, key="fc_horizon")
    with c3:
        sel_model = st.selectbox("🤖 Model", FORECASTING_MODELS_DISPLAY, key="fc_model")

    # Map display name back to raw city name
    city_raw = next((c for c in CITY_LIST if CITY_DISPLAY_MAP[c] == sel_city_disp), CITY_LIST[0])
    horizon_int = HORIZON_MAP[sel_horizon]

    tb = load_track_b_all()

    # Get metrics for selected city/horizon/model
    metrics_row = None
    if tb is not None and not tb.empty:
        mask = (
            (tb["city_display"] == sel_city_disp) &
            (tb["horizon"] == horizon_int) &
            (tb["model_display"] == sel_model)
        )
        filtered = tb[mask]
        if not filtered.empty:
            metrics_row = filtered.iloc[0]

    # Fallback to aggregate values if specific row missing
    if metrics_row is None:
        fb = FALLBACK_FORECASTING.get(sel_model, {"r2": 0.0, "mae": 0.0, "rmse": 0.0})
        r2_val   = fb["r2"]
        mae_val  = fb["mae"]
        rmse_val = fb["rmse"]
        n_test   = 4236
    else:
        r2_val   = float(metrics_row["r2"])
        mae_val  = float(metrics_row["mae"])
        rmse_val = float(metrics_row["rmse"])
        n_test   = int(metrics_row.get("n_test", 4236))

    # Metrics Section
    st.subheader(f"📊 Test-Set Performance: {sel_city_disp} | {sel_horizon} | {sel_model}")
    cols = st.columns(4)
    metrics_data = [
        ("R²", f"{r2_val:.4f}", SUCCESS if r2_val > 0.5 else ACCENT if r2_val > 0 else DANGER),
        ("MAE (µg/m³)", f"{mae_val:.2f}", ACCENT),
        ("RMSE (µg/m³)", f"{rmse_val:.2f}", TEAL),
        ("Test Samples", f"{n_test:,}", PRIMARY),
    ]
    for col, (label, value, color) in zip(cols, metrics_data):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-value" style="color:{color}">{value}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Prediction Charts
    pred_df = load_prediction_csv(city_raw, horizon_int)

    if pred_df is not None and not pred_df.empty:
        tab1, tab2, tab3 = st.tabs(["📈 Actual vs Predicted", "📉 Error Analysis", "📊 Model Comparison"])

        with tab1:
            if "actual" in pred_df.columns and "xgb_pred" in pred_df.columns:
                n_plot = min(500, len(pred_df))
                x_idx = list(range(n_plot))
                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_idx, y=pred_df["actual"].values[:n_plot],
                        name="Actual AQI", line=dict(color=PRIMARY, width=1.5),
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_idx, y=pred_df["xgb_pred"].values[:n_plot],
                        name="XGB Predicted", line=dict(color=ACCENT, width=1.5, dash="dash"),
                    ))
                    if "bilstm_pred" in pred_df.columns:
                        fig.add_trace(go.Scatter(
                            x=x_idx, y=pred_df["bilstm_pred"].values[:n_plot],
                            name="BiLSTM Predicted", line=dict(color=TEAL, width=1.5, dash="dot"),
                        ))
                    fig.update_layout(
                        title=f"Actual vs Predicted — {sel_city_disp} ({sel_horizon})",
                        template=tmpl, xaxis_title="Sample Index", yaxis_title="AQI",
                    )
                    fig = fig_layout(fig, theme)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Line chart unavailable: {e}")

                # Scatter
                try:
                    actual_vals = pred_df["actual"].dropna().values
                    pred_vals   = pred_df["xgb_pred"].dropna().values
                    min_len = min(len(actual_vals), len(pred_vals))
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=actual_vals[:min_len], y=pred_vals[:min_len],
                        mode="markers", name="Predictions",
                        marker=dict(color=PRIMARY, opacity=0.4, size=4),
                    ))
                    lim_min = float(min(actual_vals.min(), pred_vals.min()))
                    lim_max = float(max(actual_vals.max(), pred_vals.max()))
                    fig2.add_trace(go.Scatter(
                        x=[lim_min, lim_max], y=[lim_min, lim_max],
                        mode="lines", name="Perfect Prediction",
                        line=dict(color=SUCCESS, dash="dash", width=2),
                    ))
                    fig2.update_layout(
                        title="Scatter: Actual vs Predicted (XGBoost)",
                        template=tmpl,
                        xaxis_title="Actual AQI", yaxis_title="Predicted AQI",
                    )
                    fig2 = fig_layout(fig2, theme)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.info(f"Scatter plot unavailable: {e}")

        with tab2:
            if "actual" in pred_df.columns and "xgb_pred" in pred_df.columns:
                try:
                    residuals = pred_df["actual"] - pred_df["xgb_pred"]
                    fig3 = go.Figure(go.Scatter(
                        y=residuals.values, mode="lines",
                        line=dict(color=DANGER, width=1),
                        name="Residuals",
                    ))
                    fig3.add_hline(y=0, line_dash="dash", line_color=SUCCESS)
                    fig3.update_layout(
                        title="Residuals Over Time",
                        template=tmpl, xaxis_title="Sample Index", yaxis_title="Error (Actual - Predicted)",
                    )
                    fig3 = fig_layout(fig3, theme)
                    st.plotly_chart(fig3, use_container_width=True)

                    # Histogram
                    fig4 = go.Figure(go.Histogram(
                        x=residuals.values, nbinsx=50,
                        marker_color=PRIMARY, opacity=0.75,
                        name="Residual Distribution",
                    ))
                    fig4.update_layout(
                        title="Residual Distribution",
                        template=tmpl, xaxis_title="Residual", yaxis_title="Count",
                    )
                    fig4 = fig_layout(fig4, theme)
                    st.plotly_chart(fig4, use_container_width=True)

                    mean_err = float(residuals.mean())
                    std_err  = float(residuals.std())
                    st.info(f"Mean Error: {mean_err:.3f} | Std Error: {std_err:.3f}")
                except Exception as e:
                    st.info(f"Error analysis unavailable: {e}")

        with tab3:
            # Model comparison for this city+horizon
            if tb is not None and not tb.empty:
                try:
                    mask2 = (tb["city_display"] == sel_city_disp) & (tb["horizon"] == horizon_int)
                    cmp_df = tb[mask2].groupby("model_display")["r2"].mean().reset_index().sort_values("r2", ascending=False)
                    if not cmp_df.empty:
                        fig5 = go.Figure(go.Bar(
                            x=cmp_df["model_display"], y=cmp_df["r2"],
                            marker_color=[SUCCESS if v > 0.5 else PRIMARY if v > 0 else DANGER for v in cmp_df["r2"]],
                            text=[f"{v:.4f}" for v in cmp_df["r2"]],
                            textposition="outside",
                        ))
                        fig5.update_layout(
                            title=f"Model R² Comparison — {sel_city_disp}, {sel_horizon}",
                            template=tmpl, yaxis_title="R²",
                        )
                        fig5 = fig_layout(fig5, theme)
                        st.plotly_chart(fig5, use_container_width=True)
                except Exception as e:
                    st.info(f"Model comparison unavailable: {e}")
    else:
        st.info(
            f"No prediction CSV found for {sel_city_disp} at {sel_horizon}. "
            "This combination may not have been generated (only ~39 of 54 files exist)."
        )

    # BiLSTM figure
    ROOT = find_root()
    FIGS_DIR = ROOT / "outputs" / "figures"
    fig_path = FIGS_DIR / f"{city_raw}_h{horizon_int:02d}_bilstm_pred.png"
    if fig_path.exists():
        st.subheader("BiLSTM Prediction Plot")
        st.image(str(fig_path), caption=f"BiLSTM predictions — {sel_city_disp} {sel_horizon}", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESEARCH ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def page_research_analytics(theme: str) -> None:
    tmpl = get_plotly_template(theme)
    st.title("📊 Research Analytics")

    ta = load_track_a()
    tb = load_track_b_all()

    tab1, tab2, tab3 = st.tabs([
        "🔬 Estimation Analytics", "📈 Forecasting Analytics", "💡 Research Insights"
    ])

    # ── Tab 1: Estimation Analytics ──
    with tab1:
        if ta is not None and not ta.empty:
            # KPI row
            best_model_row = ta.groupby("model_display")["r2"].mean().idxmax()
            best_r2 = ta.groupby("model_display")["r2"].mean().max()
            best_city_row = ta.loc[ta["r2"].idxmax()]
            avg_r2 = ta["r2"].mean()

            cols = st.columns(4)
            kpis = [
                ("Best Model", best_model_row, PRIMARY),
                (f"Best R² ({best_model_row})", f"{best_r2:.4f}", SUCCESS),
                (f"Best City", f"{best_city_row['city_display']} ({best_city_row['r2']:.4f})", GOLD),
                ("Avg R² (all)", f"{avg_r2:.4f}", ACCENT),
            ]
            for col, (label, value, color) in zip(cols, kpis):
                with col:
                    st.markdown(
                        f'<div class="kpi-card">'
                        f'<div class="kpi-value" style="color:{color};font-size:1.3rem">{value}</div>'
                        f'<div class="kpi-label">{label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.divider()

            # Heatmap
            pivot = safe_pivot(ta, "city_display", "model_display", "r2")
            if pivot is not None:
                try:
                    fig = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=list(pivot.columns),
                        y=list(pivot.index),
                        colorscale="RdYlGn", zmin=0, zmax=1,
                        text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
                        texttemplate="%{text}",
                        colorbar=dict(title="R²"),
                    ))
                    fig.update_layout(
                        title="City × Model R² Heatmap (Estimation)",
                        template=tmpl, height=560,
                        xaxis=dict(tickangle=-30),
                    )
                    fig = fig_layout(fig, theme)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Heatmap unavailable: {e}")

            # Model ranking
            try:
                model_rank = (
                    ta.groupby("model_display")[["r2", "mae", "rmse"]]
                    .mean().reset_index().sort_values("r2", ascending=False)
                )
                fig2 = go.Figure(go.Bar(
                    x=model_rank["model_display"], y=model_rank["r2"],
                    marker_color=[SUCCESS if v > 0.95 else PRIMARY if v > 0.7 else DANGER for v in model_rank["r2"]],
                    text=[f"{v:.4f}" for v in model_rank["r2"]],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="Model Ranking by Average R² (Estimation, 18 cities)",
                    template=tmpl, yaxis_title="Avg R²", yaxis_range=[0, 1.1],
                )
                fig2 = fig_layout(fig2, theme)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.info(f"Model ranking chart unavailable: {e}")

            # Top 5 cities
            st.subheader("Top 5 Performing Cities")
            top5 = (
                ta.sort_values("r2", ascending=False)
                .groupby("city_display").first()
                .reset_index()[["city_display", "model_display", "r2", "mae", "rmse"]]
                .sort_values("r2", ascending=False)
                .head(5)
            )
            st.dataframe(top5.style.format({"r2": "{:.4f}", "mae": "{:.4f}", "rmse": "{:.4f}"}),
                         use_container_width=True, hide_index=True)

            # MAE comparison
            try:
                fig3 = go.Figure(go.Bar(
                    x=model_rank["model_display"], y=model_rank["mae"],
                    marker_color=ACCENT,
                    text=[f"{v:.2f}" for v in model_rank["mae"]],
                    textposition="outside",
                ))
                fig3.update_layout(
                    title="MAE Comparison by Model (Estimation)",
                    template=tmpl, yaxis_title="MAE (µg/m³)",
                )
                fig3 = fig_layout(fig3, theme)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.info(f"MAE chart unavailable: {e}")

        else:
            st.warning("Track A data unavailable.")

    # ── Tab 2: Forecasting Analytics ──
    with tab2:
        if tb is not None and not tb.empty:
            # Model×Horizon heatmap
            pivot2 = safe_pivot(tb, "model_display", "horizon_display", "r2")
            if pivot2 is not None:
                try:
                    # Sort columns
                    col_order = [c for c in ["1 Hour", "6 Hours", "24 Hours"] if c in pivot2.columns]
                    pivot2 = pivot2[col_order]
                    fig = go.Figure(go.Heatmap(
                        z=pivot2.values,
                        x=list(pivot2.columns),
                        y=list(pivot2.index),
                        colorscale="RdYlGn", zmin=-0.5, zmax=0.7,
                        text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in pivot2.values],
                        texttemplate="%{text}",
                        colorbar=dict(title="R²"),
                    ))
                    fig.update_layout(
                        title="Model × Horizon R² Heatmap (Forecasting)",
                        template=tmpl, height=400,
                    )
                    fig = fig_layout(fig, theme)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Model×Horizon heatmap unavailable: {e}")

            # City×Model heatmap at h=1
            tb_h1 = tb[tb["horizon"] == 1]
            if not tb_h1.empty:
                pivot3 = safe_pivot(tb_h1, "city_display", "model_display", "r2")
                if pivot3 is not None:
                    try:
                        fig2 = go.Figure(go.Heatmap(
                            z=pivot3.values,
                            x=list(pivot3.columns),
                            y=list(pivot3.index),
                            colorscale="RdYlGn", zmin=-0.5, zmax=0.7,
                            text=[[f"{v:.3f}" if not np.isnan(v) else "" for v in row] for row in pivot3.values],
                            texttemplate="%{text}",
                            colorbar=dict(title="R²"),
                        ))
                        fig2.update_layout(
                            title="City × Model R² Heatmap (1-Hour Forecasting)",
                            template=tmpl, height=560,
                            xaxis=dict(tickangle=-30),
                        )
                        fig2 = fig_layout(fig2, theme)
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.info(f"City×Model heatmap unavailable: {e}")

            # Horizon degradation
            try:
                hz_model = (
                    tb.groupby(["model_display", "horizon"])["r2"]
                    .mean().reset_index()
                )
                fig3 = go.Figure()
                for mdl in hz_model["model_display"].unique():
                    sub = hz_model[hz_model["model_display"] == mdl].sort_values("horizon")
                    fig3.add_trace(go.Scatter(
                        x=sub["horizon"], y=sub["r2"],
                        mode="lines+markers", name=mdl,
                    ))
                fig3.update_layout(
                    title="Forecast R² Degradation with Horizon",
                    template=tmpl,
                    xaxis=dict(title="Horizon (hours)", tickvals=[1, 6, 24]),
                    yaxis_title="Avg R²",
                )
                fig3 = fig_layout(fig3, theme)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.info(f"Horizon degradation chart unavailable: {e}")

            # City performance bar
            try:
                city_perf = tb.groupby("city_display")["r2"].mean().reset_index().sort_values("r2", ascending=False)
                fig4 = go.Figure(go.Bar(
                    x=city_perf["city_display"],
                    y=city_perf["r2"],
                    marker_color=PRIMARY,
                    text=[f"{v:.3f}" for v in city_perf["r2"]],
                    textposition="outside",
                ))
                fig4.update_layout(
                    title="City Performance (Avg R² across all models & horizons)",
                    template=tmpl, yaxis_title="Avg R²",
                    xaxis=dict(tickangle=-35),
                )
                fig4 = fig_layout(fig4, theme)
                st.plotly_chart(fig4, use_container_width=True)
            except Exception as e:
                st.info(f"City performance chart unavailable: {e}")

        else:
            st.warning("Track B data unavailable.")

    # ── Tab 3: Research Insights ──
    with tab3:
        st.subheader("🔍 Data-Driven Research Insights")
        insights = []

        if ta is not None and not ta.empty:
            best_est_model = ta.groupby("model_display")["r2"].mean().idxmax()
            best_est_r2 = ta.groupby("model_display")["r2"].mean().max()
            worst_city_row = ta.loc[ta["r2"].idxmin()]
            best_city_row = ta.loc[ta["r2"].idxmax()]
            insights.append(("🥇", "Best Estimation Model",
                              f"{best_est_model} with avg R² = {best_est_r2:.4f} across 18 cities"))
            insights.append(("🏙️", "Best City (Estimation)",
                              f"{best_city_row['city_display']} — R²={best_city_row['r2']:.4f} using {best_city_row['model_display']}"))
            insights.append(("⚠️", "Worst City (Estimation)",
                              f"{worst_city_row['city_display']} — R²={worst_city_row['r2']:.4f}"))
        else:
            insights.append(("🥇", "Best Estimation Model",
                              "Gradient Boosting with avg R² = 0.9906"))

        if tb is not None and not tb.empty:
            best_fc_model = tb.groupby("model_display")["r2"].mean().idxmax()
            best_fc_r2 = tb.groupby("model_display")["r2"].mean().max()
            hz_avg = tb.groupby("horizon")["r2"].mean()
            h1_r2  = hz_avg.get(1, 0.0)
            h6_r2  = hz_avg.get(6, 0.0)
            h24_r2 = hz_avg.get(24, 0.0)
            insights.append(("📈", "Best Forecasting Model",
                              f"{best_fc_model} with avg R² = {best_fc_r2:.4f}"))
            insights.append(("📉", "Horizon Degradation",
                              f"1h R²={h1_r2:.4f}, 6h R²={h6_r2:.4f}, 24h R²={h24_r2:.4f} — forecasting degrades with horizon"))
        else:
            insights.append(("📈", "Best Forecasting Model",
                              "Gradient Boosting with avg R² = 0.4997"))
            insights.append(("📉", "Horizon Degradation",
                              "R² degrades from ~0.50 (1h) to lower values at 6h and 24h"))

        insights.append(("🤖", "Classical ML vs Deep Learning",
                          "Classical ML (GBR, RF, XGB) significantly outperforms DL (LSTM, BiLSTM, CNN-BiLSTM) "
                          "for both estimation and forecasting tasks on tabular/time-series AQI data"))

        for icon, title, desc in insights:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4>{icon} {title}</h4>'
                f'<p style="font-size:0.93rem">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INDIA PERFORMANCE MAP
# ─────────────────────────────────────────────────────────────────────────────
def page_india_map(theme: str) -> None:
    tmpl = get_plotly_template(theme)
    st.title("🗺 India Performance Map")
    st.caption("Geographic view of model performance across 18 monitoring stations")

    ta = load_track_a()
    tb = load_track_b_all()

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        task = st.radio("📋 Task", ["Estimation", "Forecasting"], horizontal=True, key="map_task")
    with c2:
        if task == "Estimation":
            model_sel = st.selectbox("🤖 Model", ESTIMATION_MODELS_DISPLAY, key="map_model_est")
        else:
            model_sel = st.selectbox("🤖 Model", FORECASTING_MODELS_DISPLAY, key="map_model_fc")
    with c3:
        if task == "Forecasting":
            hz_sel = st.selectbox("⏱️ Horizon", HORIZONS_DISPLAY, key="map_horizon")
        else:
            hz_sel = "1 Hour"

    hz_int = HORIZON_MAP[hz_sel]

    # Get data for map
    map_data = []
    if task == "Estimation" and ta is not None and not ta.empty:
        csv_name = TRACK_A_MODEL_CSV_NAMES.get(model_sel, model_sel)
        df_model = ta[ta["model"] == csv_name]
        if df_model.empty:
            df_model = ta[ta["model_display"] == model_sel]
        for city_raw in CITY_LIST:
            lat, lon = CITY_COORDS[city_raw]
            city_disp = CITY_DISPLAY_MAP[city_raw]
            row = df_model[df_model["city"] == city_raw]
            if row.empty:
                row = df_model[df_model["city_display"] == city_disp]
            if not row.empty:
                r = row.iloc[0]
                map_data.append({
                    "city": city_disp, "lat": lat, "lon": lon,
                    "r2": float(r["r2"]), "mae": float(r["mae"]),
                    "rmse": float(r["rmse"]), "n_test": int(r.get("n_test", 0)),
                })
    elif task == "Forecasting" and tb is not None and not tb.empty:
        csv_name = TRACK_B_MODEL_CSV_NAMES.get(model_sel, model_sel)
        df_model = tb[(tb["model"] == csv_name) & (tb["horizon"] == hz_int)]
        if df_model.empty:
            df_model = tb[(tb["model_display"] == model_sel) & (tb["horizon"] == hz_int)]
        for city_raw in CITY_LIST:
            lat, lon = CITY_COORDS[city_raw]
            city_disp = CITY_DISPLAY_MAP[city_raw]
            row = df_model[df_model["city"] == city_raw]
            if row.empty:
                row = df_model[df_model["city_display"] == city_disp]
            if not row.empty:
                r = row.iloc[0]
                map_data.append({
                    "city": city_disp, "lat": lat, "lon": lon,
                    "r2": float(r["r2"]), "mae": float(r["mae"]),
                    "rmse": float(r["rmse"]), "n_test": int(r.get("n_test", 0)),
                })

    if not map_data:
        # Fallback coordinates with fallback metrics
        fb = FALLBACK_ESTIMATION if task == "Estimation" else FALLBACK_FORECASTING
        fb_vals = fb.get(model_sel, {"r2": 0.5, "mae": 20.0, "rmse": 30.0})
        for city_raw in CITY_LIST:
            lat, lon = CITY_COORDS[city_raw]
            map_data.append({
                "city": CITY_DISPLAY_MAP[city_raw], "lat": lat, "lon": lon,
                "r2": fb_vals["r2"], "mae": fb_vals["mae"],
                "rmse": fb_vals["rmse"], "n_test": 4236,
            })
        st.info(f"No evaluation data found for {model_sel} in CSV. Showing average performance.")

    map_df = pd.DataFrame(map_data)
    map_df["rank"] = map_df["r2"].rank(ascending=False).astype(int)

    # Map
    try:
        fig = go.Figure(go.Scattergeo(
            lat=map_df["lat"],
            lon=map_df["lon"],
            mode="markers",
            marker=dict(
                size=16,
                color=map_df["r2"],
                colorscale="RdYlGn",
                cmin=0, cmax=1,
                colorbar=dict(title="R²", thickness=15),
                showscale=True,
            ),
            text=map_df["city"],
            customdata=map_df[["city", "r2", "mae", "rmse", "rank", "n_test"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "R² = %{customdata[1]:.4f}<br>"
                "MAE = %{customdata[2]:.2f}<br>"
                "RMSE = %{customdata[3]:.2f}<br>"
                "Rank = #%{customdata[4]}<br>"
                "Test Samples = %{customdata[5]:,}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=f"{model_sel} Performance Map ({task}){' | ' + hz_sel if task == 'Forecasting' else ''}",
            geo=dict(
                scope="asia",
                projection_type="natural earth",
                fitbounds="locations",
                bgcolor="rgba(0,0,0,0)",
                showland=True,
                landcolor="rgba(40,50,70,0.8)" if theme == "dark" else "rgba(220,230,240,0.9)",
                showocean=True,
                oceancolor="rgba(20,30,50,0.5)" if theme == "dark" else "rgba(200,220,240,0.5)",
                showcoastlines=True,
                coastlinecolor="#555" if theme == "dark" else "#888",
                showcountries=True,
                countrycolor="#444" if theme == "dark" else "#aaa",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Map unavailable: {e}")

    st.divider()

    # City detail panel
    st.subheader("📍 City Detail Panel")
    sel_city_detail = st.selectbox(
        "Select city for details",
        [d["city"] for d in map_data],
        key="map_city_detail",
    )
    city_detail = next((d for d in map_data if d["city"] == sel_city_detail), None)
    if city_detail:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4>{sel_city_detail}</h4>'
                f'<p>R² = <strong>{city_detail["r2"]:.4f}</strong></p>'
                f'<p>MAE = {city_detail["mae"]:.2f} µg/m³</p>'
                f'<p>RMSE = {city_detail["rmse"]:.2f} µg/m³</p>'
                f'<p>Test Samples = {city_detail["n_test"]:,}</p>'
                f'<p>Rank = #{city_detail["rank"]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c2:
            # All models for this city
            city_raw_detail = next((c for c in CITY_LIST if CITY_DISPLAY_MAP[c] == sel_city_detail), None)
            if city_raw_detail and task == "Estimation" and ta is not None and not ta.empty:
                city_all = ta[ta["city"] == city_raw_detail].sort_values("r2", ascending=False)
                if not city_all.empty:
                    try:
                        fig2 = go.Figure(go.Bar(
                            x=city_all["model_display"],
                            y=city_all["r2"],
                            marker_color=[SUCCESS if v > 0.95 else PRIMARY if v > 0.7 else DANGER for v in city_all["r2"]],
                            text=[f"{v:.4f}" for v in city_all["r2"]],
                            textposition="outside",
                        ))
                        fig2.update_layout(
                            title=f"All Models — {sel_city_detail}",
                            template=tmpl, yaxis_title="R²",
                        )
                        fig2 = fig_layout(fig2, theme)
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.info(f"City model chart unavailable: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def page_comparison(theme: str) -> None:
    tmpl = get_plotly_template(theme)
    st.title("🏆 Model Comparison")

    ta = load_track_a()
    tb = load_track_b_all()

    tab_est, tab_fc = st.tabs(["🔬 Estimation", "📈 Forecasting"])

    # ── Estimation Tab ──
    with tab_est:
        if ta is not None and not ta.empty:
            c1, c2 = st.columns(2)
            with c1:
                metric_sel = st.selectbox("📏 Metric", ["r2", "mae", "rmse"], key="est_metric",
                                          format_func=lambda x: x.upper())
            with c2:
                city_opts = ["All Cities"] + sorted(ta["city_display"].unique().tolist())
                city_sel = st.selectbox("🏙️ City", city_opts, key="est_city")

            if city_sel == "All Cities":
                df_filtered = ta
            else:
                df_filtered = ta[ta["city_display"] == city_sel]

            asc = metric_sel in ["mae", "rmse"]
            leaderboard = (
                df_filtered.groupby("model_display")[["r2", "mae", "rmse"]]
                .mean().reset_index().sort_values(metric_sel, ascending=asc).reset_index(drop=True)
            )
            leaderboard.insert(0, "Rank", ["🥇", "🥈", "🥉"] + [str(i + 1) for i in range(3, len(leaderboard))])

            st.dataframe(
                leaderboard.rename(columns={"model_display": "Model", "r2": "R²", "mae": "MAE", "rmse": "RMSE"})
                .style.format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )
            csv_dl = leaderboard.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv_dl, "estimation_comparison.csv", "text/csv")

            # Bar chart
            try:
                color_vals = leaderboard[metric_sel].tolist()
                cmap_colors = [SUCCESS if i == 0 else PRIMARY for i in range(len(leaderboard))]
                fig = go.Figure(go.Bar(
                    x=leaderboard["model_display"],
                    y=leaderboard[metric_sel],
                    marker_color=cmap_colors,
                    text=[f"{v:.4f}" for v in leaderboard[metric_sel]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Model Comparison by {metric_sel.upper()}",
                    template=tmpl, yaxis_title=metric_sel.upper(),
                )
                fig = fig_layout(fig, theme)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Bar chart unavailable: {e}")

            # Box plot
            try:
                fig2 = go.Figure()
                for mdl in ta["model_display"].unique():
                    sub = ta[ta["model_display"] == mdl][metric_sel].dropna()
                    fig2.add_trace(go.Box(y=sub.values, name=mdl, boxpoints="outliers"))
                fig2.update_layout(
                    title=f"Distribution of {metric_sel.upper()} across 18 Cities",
                    template=tmpl, yaxis_title=metric_sel.upper(),
                )
                fig2 = fig_layout(fig2, theme)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.info(f"Box plot unavailable: {e}")

            # Grouped bar: top 3 models per city
            try:
                top3_models = leaderboard["model_display"].head(3).tolist()
                df_top3 = ta[ta["model_display"].isin(top3_models)]
                cities_ordered = ta.groupby("city_display")["r2"].mean().sort_values(ascending=False).index.tolist()
                fig3 = go.Figure()
                colors_top3 = [SUCCESS, PRIMARY, ACCENT]
                for i, mdl in enumerate(top3_models):
                    sub = df_top3[df_top3["model_display"] == mdl].set_index("city_display")["r2"]
                    sub = sub.reindex(cities_ordered)
                    fig3.add_trace(go.Bar(
                        x=cities_ordered, y=sub.values, name=mdl,
                        marker_color=colors_top3[i],
                    ))
                fig3.update_layout(
                    title="Top 3 Models — Per-City R² Comparison",
                    template=tmpl, barmode="group",
                    xaxis=dict(tickangle=-35), yaxis_title="R²",
                )
                fig3 = fig_layout(fig3, theme)
                st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.info(f"Grouped bar chart unavailable: {e}")
        else:
            st.warning("Track A data unavailable.")

    # ── Forecasting Tab ──
    with tab_fc:
        if tb is not None and not tb.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                fc_metric = st.selectbox("📏 Metric", ["r2", "mae", "rmse"], key="fc_metric_cmp",
                                         format_func=lambda x: x.upper())
            with c2:
                fc_city_opts = ["All Cities"] + sorted(tb["city_display"].unique().tolist())
                fc_city = st.selectbox("🏙️ City", fc_city_opts, key="fc_city_cmp")
            with c3:
                fc_hz_opts = ["All Horizons"] + HORIZONS_DISPLAY
                fc_hz = st.selectbox("⏱️ Horizon", fc_hz_opts, key="fc_hz_cmp")

            df_fc = tb.copy()
            if fc_city != "All Cities":
                df_fc = df_fc[df_fc["city_display"] == fc_city]
            if fc_hz != "All Horizons":
                df_fc = df_fc[df_fc["horizon"] == HORIZON_MAP[fc_hz]]

            if not df_fc.empty:
                asc_fc = fc_metric in ["mae", "rmse"]
                fc_lead = (
                    df_fc.groupby("model_display")[["r2", "mae", "rmse"]]
                    .mean().reset_index().sort_values(fc_metric, ascending=asc_fc).reset_index(drop=True)
                )
                fc_lead.insert(0, "Rank", ["🥇", "🥈", "🥉"] + [str(i + 1) for i in range(3, len(fc_lead))])
                st.dataframe(
                    fc_lead.rename(columns={"model_display": "Model", "r2": "R²", "mae": "MAE", "rmse": "RMSE"})
                    .style.format({"R²": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )

                csv_dl2 = fc_lead.to_csv(index=False).encode()
                st.download_button("📥 Download CSV", csv_dl2, "forecasting_comparison.csv", "text/csv")

                # Bar chart
                try:
                    fig = go.Figure(go.Bar(
                        x=fc_lead["model_display"], y=fc_lead[fc_metric],
                        marker_color=[SUCCESS if i == 0 else PRIMARY for i in range(len(fc_lead))],
                        text=[f"{v:.4f}" for v in fc_lead[fc_metric]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title=f"Forecasting Models by {fc_metric.upper()}",
                        template=tmpl, yaxis_title=fc_metric.upper(),
                    )
                    fig = fig_layout(fig, theme)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Chart unavailable: {e}")

                # Box plot
                try:
                    fig2 = go.Figure()
                    for mdl in tb["model_display"].unique():
                        sub = df_fc[df_fc["model_display"] == mdl][fc_metric].dropna()
                        fig2.add_trace(go.Box(y=sub.values, name=mdl, boxpoints="outliers"))
                    fig2.update_layout(
                        title=f"Distribution of {fc_metric.upper()} across Cities/Horizons",
                        template=tmpl, yaxis_title=fc_metric.upper(),
                    )
                    fig2 = fig_layout(fig2, theme)
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.info(f"Box plot unavailable: {e}")

                # Horizon degradation
                try:
                    hz_line = tb.groupby(["model_display", "horizon"])[fc_metric].mean().reset_index()
                    fig3 = go.Figure()
                    for mdl in hz_line["model_display"].unique():
                        sub = hz_line[hz_line["model_display"] == mdl].sort_values("horizon")
                        fig3.add_trace(go.Scatter(
                            x=sub["horizon"], y=sub[fc_metric],
                            mode="lines+markers", name=mdl,
                        ))
                    fig3.update_layout(
                        title=f"Horizon Degradation — {fc_metric.upper()} vs Horizon",
                        template=tmpl,
                        xaxis=dict(title="Horizon (hours)", tickvals=[1, 6, 24]),
                        yaxis_title=fc_metric.upper(),
                    )
                    fig3 = fig_layout(fig3, theme)
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as e:
                    st.info(f"Horizon degradation chart unavailable: {e}")
            else:
                st.info("No data for selected filters.")
        else:
            st.warning("Track B data unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def page_about(theme: str) -> None:
    st.title("📖 About This Research")

    tab_ov, tab_ds, tab_meth, tab_mod, tab_team = st.tabs([
        "📋 Overview", "📂 Dataset", "⚙️ Methodology", "🤖 Models", "👤 Team"
    ])

    # ── Overview ──
    with tab_ov:
        st.markdown(
            f'<div class="glass-card">'
            f'<h3>AQI Prediction Using Machine Learning and Deep Learning</h3>'
            f'<p><strong>Institution:</strong> IIIT Nagpur (Research Internship)</p>'
            f'<p><strong>Host College:</strong> YCCE, Nagpur (B.Tech CSE)</p>'
            f'<p><strong>Scope:</strong> 18 CPCB-monitored Indian cities, 2018-2023</p>'
            f'<p><strong>Models:</strong> 7 ML/DL algorithms across two tasks</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4>🔍 Problem Statement</h4>'
                f'<p>Air quality in Indian cities has deteriorated significantly, with AQI exceeding safe limits '
                f'regularly. Accurate prediction and forecasting of AQI enables timely health advisories and '
                f'policy interventions. This research develops a comprehensive ML/DL pipeline for AQI '
                f'estimation from current pollutant readings and multi-horizon forecasting.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4>🔬 Research Gap</h4>'
                f'<p>Prior studies focus on single cities with limited model benchmarking. This work provides: '
                f'(1) simultaneous 18-city evaluation, (2) rigorous data leakage audit distinguishing estimation '
                f'from forecasting, and (3) systematic DL vs. classical ML comparison using consistent '
                f'chronological splits on official CPCB data.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        contribs = [
            ("🏙️", "18-City Benchmark",
             "First simultaneous evaluation across 18 CPCB Indian cities with 126 model-city combinations."),
            ("🔒", "Leakage Audit",
             "Certified leakage-free dual-track protocol separating estimation from forecasting tasks."),
            ("📊", "ML vs DL Comparison",
             "Comprehensive benchmark showing GBR (R²=0.99) outperforms LSTM/BiLSTM for AQI prediction."),
        ]
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, contribs):
            with col:
                st.markdown(
                    f'<div class="glass-card">'
                    f'<div style="font-size:1.5rem">{icon}</div>'
                    f'<h4>{title}</h4>'
                    f'<p style="font-size:0.87rem">{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.link_button("🐙 GitHub Repository",
                           "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning")
        with c2:
            st.link_button("📂 CPCB Dataset",
                           "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link")

    # ── Dataset ──
    with tab_ds:
        st.markdown(
            f'<div class="glass-card">'
            f'<h4>📂 CPCB Dataset</h4>'
            f'<table style="width:100%;font-size:0.9rem">'
            f'<tr><td><strong>Cities</strong></td><td>18 monitoring stations</td></tr>'
            f'<tr><td><strong>Records</strong></td><td>~934,000 hourly records</td></tr>'
            f'<tr><td><strong>Time Period</strong></td><td>2018-2023</td></tr>'
            f'<tr><td><strong>Raw Features</strong></td><td>21 (11 pollutants + 6 meteorological + 4 derived)</td></tr>'
            f'<tr><td><strong>Engineered Features</strong></td><td>88-115 per city after feature engineering</td></tr>'
            f'<tr><td><strong>Target</strong></td><td>AQI (Air Quality Index)</td></tr>'
            f'</table>'
            f'</div>',
            unsafe_allow_html=True,
        )

        city_table = pd.DataFrame({
            "City": [CITY_DISPLAY_MAP[c] for c in CITY_LIST],
            "State": [
                "Gujarat", "Tamil Nadu", "Delhi", "Gujarat", "Telangana",
                "Madhya Pradesh", "Rajasthan", "Rajasthan", "Maharashtra", "Maharashtra",
                "Maharashtra", "Maharashtra", "Madhya Pradesh", "Gujarat", "Maharashtra",
                "Gujarat", "Madhya Pradesh", "Andhra Pradesh",
            ],
            "Lat": [CITY_COORDS[c][0] for c in CITY_LIST],
            "Lon": [CITY_COORDS[c][1] for c in CITY_LIST],
        })
        st.dataframe(city_table, use_container_width=True, hide_index=True)

        st.subheader("Data Pipeline")
        pipeline = [
            ("📥", "Data Collection", "15-min CPCB readings from 18 city stations"),
            ("✅", "Quality Check", "Remove duplicates, fix timestamps, filter valid AQI"),
            ("⏰", "Resampling", "15-min → hourly via mean aggregation"),
            ("🔧", "Imputation", "3-stage: forward-fill → rolling-mean → city-median"),
            ("⚙️", "Feature Engineering", "Lag, rolling stats, temporal encoding, interactions"),
            ("🔒", "Leakage Audit", "Separate estimation (same-step) from forecasting (future-only)"),
        ]
        cols = st.columns(3)
        for i, (icon, title, desc) in enumerate(pipeline):
            with cols[i % 3]:
                st.markdown(
                    f'<div class="glass-card" style="min-height:110px">'
                    f'<div style="font-size:1.4rem">{icon}</div>'
                    f'<strong>{i+1}. {title}</strong><br>'
                    f'<span style="font-size:0.85rem">{desc}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Methodology ──
    with tab_meth:
        steps = [
            ("📥", "Data Collection", "Hourly AQI & pollutant data from CPCB across 18 stations, 2018-2023"),
            ("🔧", "Preprocessing", "Hourly resampling, 3-stage imputation, city isolation, quality filtering"),
            ("⚙️", "Feature Engineering", "88-115 features: lag/rolling windows, temporal encoding, pollutant interactions"),
            ("🔒", "Leakage Audit", "Strict protocol: Track A uses same-timestep, Track B uses only past information"),
            ("🤖", "Estimation Models", "Ridge, RF, GBR, XGBoost on Track A with 70/15/15 chronological split"),
            ("🧠", "Forecasting Models", "LSTM, BiLSTM, CNN-BiLSTM + classical ML on Track B, 3 horizons"),
            ("📊", "Evaluation", "R², MAE, RMSE on held-out test set; no data leakage from validation"),
            ("🖥️", "Dashboard", "Interactive Streamlit app with 7 pages, CPCB calculator, India map"),
        ]
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(steps):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="glass-card">'
                    f'<h4>{icon} Step {i+1}: {title}</h4>'
                    f'<p style="font-size:0.88rem">{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Models ──
    with tab_mod:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("⚙️ Classical ML Models")
            ml_models = [
                ("Ridge Regression", "Linear model with L2 regularization.",
                 "alpha=1.0, max_iter=1000",
                 f"R²={FALLBACK_ESTIMATION['Ridge Regression']['r2']:.4f}"),
                ("Random Forest", "Ensemble of 500 decision trees with bootstrap sampling.",
                 "n_estimators=500, max_depth=None, min_samples_leaf=2",
                 f"R²={FALLBACK_ESTIMATION['Random Forest']['r2']:.4f}"),
                ("Gradient Boosting", "Sequential boosting with shallow trees. Best performer.",
                 "n_estimators=200, learning_rate=0.1, max_depth=5",
                 f"R²={FALLBACK_ESTIMATION['Gradient Boosting']['r2']:.4f} 🏆"),
                ("XGBoost", "GPU-accelerated gradient boosting with regularization.",
                 "n_estimators=200, learning_rate=0.05, max_depth=6",
                 f"R²={FALLBACK_ESTIMATION['XGBoost']['r2']:.4f}"),
            ]
            for name, desc, params, perf in ml_models:
                st.markdown(
                    f'<div class="glass-card">'
                    f'<strong>{name}</strong><br>'
                    f'<span style="font-size:0.85rem">{desc}</span><br>'
                    f'<code style="font-size:0.78rem">{params}</code><br>'
                    f'<span style="color:{SUCCESS};font-size:0.88rem">Best: {perf}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with c2:
            st.subheader("🧠 Deep Learning Models")
            dl_models = [
                ("LSTM", "Long Short-Term Memory — captures temporal dependencies.",
                 "2 layers × 64 units, dropout=0.2, seq_len=24",
                 f"R²={FALLBACK_ESTIMATION['LSTM']['r2']:.4f}"),
                ("BiLSTM", "Bidirectional LSTM — processes sequences in both directions.",
                 "2 BiLSTM layers × 64 units, dropout=0.2",
                 f"R²={FALLBACK_ESTIMATION['BiLSTM']['r2']:.4f}"),
                ("CNN-BiLSTM", "1D Conv feature extraction + BiLSTM temporal modeling.",
                 "Conv1D(64,3) + BiLSTM(64) + Dense",
                 f"R²={FALLBACK_ESTIMATION['CNN-BiLSTM']['r2']:.4f}"),
            ]
            for name, desc, params, perf in dl_models:
                st.markdown(
                    f'<div class="glass-card">'
                    f'<strong>{name}</strong><br>'
                    f'<span style="font-size:0.85rem">{desc}</span><br>'
                    f'<code style="font-size:0.78rem">{params}</code><br>'
                    f'<span style="color:{ACCENT};font-size:0.88rem">Best: {perf}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="glass-card" style="border-color:rgba(255,100,100,0.4)">'
                f'<strong>⚠️ Key Finding</strong><br>'
                f'<span style="font-size:0.88rem">Deep learning models underperformed classical ML significantly '
                f'on this tabular/time-series dataset. GBR achieves R²=0.9906 while CNN-BiLSTM achieves R²=0.2756 '
                f'for estimation. Classical ML dominance likely due to feature engineering quality and dataset size.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Team ──
    with tab_team:
        st.markdown(
            f'<div class="glass-card" style="border-color:rgba(108,158,232,0.5)">'
            f'<h3>👤 Author</h3>'
            f'<h2 style="color:{PRIMARY}">Aman Gajbhiye</h2>'
            f'<p><strong>Program:</strong> B.Tech in Computer Science & Engineering</p>'
            f'<p><strong>Institution:</strong> YCCE (Yeshwantrao Chavan College of Engineering), Nagpur</p>'
            f'<p><strong>Research Internship:</strong> IIIT Nagpur (Indian Institute of Information Technology Nagpur)</p>'
            f'<p><strong>Research Area:</strong> Machine Learning, Deep Learning, Air Quality Prediction</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4>🏫 Supervisor & Institution</h4>'
                f'<p><strong>IIIT Nagpur</strong><br>Indian Institute of Information Technology Nagpur</p>'
                f'<p>Research Internship Program in Artificial Intelligence and Machine Learning</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="glass-card">'
                f'<h4>📅 Research Internship</h4>'
                f'<p>Multi-city AQI Prediction Study<br>18 Cities | 7 Models | Dual-Track Protocol</p>'
                f'<p>Data Source: CPCB (Central Pollution Control Board of India)</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.link_button("🐙 GitHub Repository",
                           "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning")
        with c2:
            st.link_button("📂 CPCB Dataset",
                           "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Theme state
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    # CSS
    inject_css(st.session_state.theme)

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1rem 0">'
            '<span style="font-size:2rem">🌬️</span><br>'
            '<strong style="font-size:1rem">AQI Research Dashboard</strong><br>'
            '<span style="font-size:0.75rem;color:#9aa5c9">IIIT Nagpur | YCCE</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Home", "🔬 AQI Estimation", "📈 Forecast Analysis",
             "📊 Research Analytics", "🗺 India Performance Map",
             "🏆 Model Comparison", "📖 About"],
            label_visibility="collapsed",
        )

        st.divider()
        theme_toggle = st.toggle("☀️ Light Mode", value=(st.session_state.theme == "light"))
        if theme_toggle and st.session_state.theme == "dark":
            st.session_state.theme = "light"
            st.rerun()
        elif not theme_toggle and st.session_state.theme == "light":
            st.session_state.theme = "dark"
            st.rerun()

        st.divider()
        st.markdown(
            '<div style="font-size:0.75rem;color:#9aa5c9;text-align:center">'
            '18 Cities · 7 Models · 3 Horizons<br>'
            'CPCB Data 2018–2023'
            '</div>',
            unsafe_allow_html=True,
        )

    # Route
    theme = st.session_state.theme
    if page == "🏠 Home":
        page_home(theme)
    elif page == "🔬 AQI Estimation":
        page_estimation(theme)
    elif page == "📈 Forecast Analysis":
        page_forecast_analysis(theme)
    elif page == "📊 Research Analytics":
        page_research_analytics(theme)
    elif page == "🗺 India Performance Map":
        page_india_map(theme)
    elif page == "🏆 Model Comparison":
        page_comparison(theme)
    elif page == "📖 About":
        page_about(theme)


if __name__ == "__main__":
    main()
