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
/* Global font */
html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1e3a5f 100%);
}
section[data-testid="stSidebar"] * { color: #e0f0ff !important; }

/* Cards */
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

/* Hero */
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

/* AQI Badge */
.aqi-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
    color: #000;
}

/* Section header */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e3a5f;
    border-left: 4px solid #00d4aa;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Info box */
.info-box {
    background: #f0f8ff;
    border: 1px solid #b0d4f0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #1e3a5f;
    margin: 8px 0;
}

/* Model card */
.model-card {
    background: white;
    border: 1px solid #e0e8f0;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(30,58,95,0.08);
}

/* Forecast table */
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
    """Return (category_name, hex_color) for a given AQI value."""
    for lo, hi, cat, col in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return cat, col
    return "Severe", "#7e0023"


def compute_sub_index(pollutant: str, concentration: float) -> Optional[float]:
    """Compute CPCB sub-index for a single pollutant."""
    if pollutant not in CPCB_BP or concentration < 0:
        return None
    for bp_lo, bp_hi, i_lo, i_hi in CPCB_BP[pollutant]:
        if bp_lo <= concentration <= bp_hi:
            si = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
            return round(si, 1)
    return 500.0


def compute_cpcb_aqi(pm25, pm10, no2, so2, co, o3, nh3) -> Optional[float]:
    """Compute official CPCB AQI as max of all available sub-indices."""
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
    """Load Track A evaluation metrics CSV."""
    path = TABLES / "final_track_a_complete.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_track_b_csv(model_stem: str) -> Optional[pd.DataFrame]:
    """Load a Track B evaluation CSV by model stem (e.g. track_b_gbr)."""
    for base in [TABLES]:
        path = base / f"{model_stem}.csv"
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return None
    return None


@st.cache_data(show_spinner=False)
def load_xgb_model(city: str, horizon: int) -> Optional[Dict]:
    """Load XGB model dict from pkl file."""
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
    """Load GBR joblib model if available."""
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
    """Load LSTM keras model + scalers if available."""
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
    """Load BiLSTM evaluation CSV."""
    return load_track_b_csv("track_b_bilstm")


@st.cache_data(show_spinner=False)
def get_all_track_b_metrics() -> Dict[str, Optional[pd.DataFrame]]:
    """Load all Track B evaluation CSVs."""
    stems = {
        "GBR":        "track_b_gbr",
        "RF":         "track_b_rf",
        "XGB":        "track_b_xgb",
        "LSTM":       "track_b_lstm",
        "BiLSTM":     "track_b_bilstm",
        "CNN-BiLSTM": "track_b_cnn_bilstm",
    }
    return {name: load_track_b_csv(stem) for name, stem in stems.items()}


# ─── AQI Badge HTML ──────────────────────────────────────────────────────────
def aqi_badge(aqi: float) -> str:
    cat, col = get_aqi_category(aqi)
    text_col = "#000" if col in ("#00e400", "#92d050", "#ffff00") else "#fff"
    return (
        f'<span class="aqi-badge" style="background:{col};color:{text_col};">'
        f'AQI {aqi:.0f} — {cat}</span>'
    )


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
        '<div style="font-size:0.75rem;color:#506070;text-align:center;">'
        'CPCB · 18 Cities · 7 Models</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown(
        '''<div class="hero-container">
            <div class="hero-title">🌬️ AQI Prediction Platform</div>
            <div class="hero-subtitle">Research Internship · IIIT Nagpur</div>
            <div class="hero-caption">CPCB National Air Quality Dataset · Multi-City Air Quality Intelligence</div>
        </div>''',
        unsafe_allow_html=True,
    )

    # KPI Cards
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

    # Two-column feature overview
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🔬 AQI Estimation (Track A)")
        st.markdown(
            """
            Enter current pollutant readings and get **instant AQI estimates** from
            7 machine learning models. Compare performance across Ridge, Random Forest,
            Gradient Boosting, XGBoost, LSTM, BiLSTM, and CNN-BiLSTM — benchmarked on
            real CPCB data from 18 Indian cities.
            """
        )
        if st.button("→ Go to AQI Estimation", use_container_width=True, key="btn_est"):
            st.session_state["_nav"] = "🔬 AQI Estimation"
            st.rerun()

    with col_r:
        st.markdown("### 📈 AQI Forecasting (Track B)")
        st.markdown(
            """
            Select a city and get **multi-horizon AQI forecasts** (+1h, +6h, +24h) using
            trained XGBoost, GBR, and LSTM models — all trained on leakage-free, lag-based
            feature pipelines. Includes trend analysis and CPCB category badges.
            """
        )
        if st.button("→ Go to AQI Forecasting", use_container_width=True, key="btn_fore"):
            st.session_state["_nav"] = "📈 AQI Forecasting"
            st.rerun()

    st.markdown("---")
    st.markdown("### 🏆 Project Highlights")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.info("**Dual-Track Protocol**\nTrack A: same-time estimation\nTrack B: leakage-free forecasting")
    with h_col2:
        st.success("**Best Model: GBR**\nAvg R² = 0.99 (Track A)\nAvg R² ≈ 0.90 (Track B)")
    with h_col3:
        st.warning("**18 Cities × 7 Models**\n934K hourly records\n450+ experiments conducted")

    st.markdown("---")
    st.markdown(
        '''<div class="info-box">
        📌 <b>About this platform:</b> All model performance figures are based on held-out test sets
        (chronological 70/15/15 split). Track A models have access to same-time pollutants (estimation task).
        Track B models use only lag and rolling features (true forecasting — no data leakage).
        </div>''',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: AQI ESTIMATION
# ════════════════════════════════════════════════════════════════════════════════
def page_estimation():
    st.markdown('<div class="section-header">🔬 AQI Estimation</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter current pollutant concentrations to compute the official **CPCB AQI** "
        "and see how all 7 Track A models performed for your selected city."
    )

    ta_df = load_track_a_csv()

    col_city, col_info = st.columns([1, 2])
    with col_city:
        city = st.selectbox("Select City", CITIES, key="est_city")

    with col_info:
        if ta_df is not None:
            city_df = ta_df[ta_df["city"] == city]
            if not city_df.empty:
                best_row = city_df.loc[city_df["r2"].idxmax()]
                st.metric(
                    "Best Model for this City",
                    best_row["model"],
                    f"R² = {best_row['r2']:.4f}",
                )

    st.markdown("---")
    st.markdown("#### 💨 Enter Pollutant Concentrations (µg/m³)")
    c1, c2, c3, c4 = st.columns(4)
    pm25 = c1.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=999.0, value=60.0, step=1.0)
    pm10 = c2.number_input("PM10 (µg/m³)",  min_value=0.0, max_value=999.0, value=100.0, step=1.0)
    no2  = c3.number_input("NO2 (µg/m³)",   min_value=0.0, max_value=800.0, value=40.0, step=1.0)
    so2  = c4.number_input("SO2 (µg/m³)",   min_value=0.0, max_value=1600.0, value=20.0, step=1.0)

    c5, c6, c7, _ = st.columns(4)
    co   = c5.number_input("CO (mg/m³)",    min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    o3   = c6.number_input("O3 (µg/m³)",   min_value=0.0, max_value=800.0, value=30.0, step=1.0)
    nh3  = c7.number_input("NH3 (µg/m³)",  min_value=0.0, max_value=2400.0, value=10.0, step=1.0)

    if st.button("🔍 Compute AQI", type="primary", use_container_width=False):
        cpcb_aqi = compute_cpcb_aqi(pm25, pm10, no2, so2, co, o3, nh3)
        if cpcb_aqi is not None:
            cat, col = get_aqi_category(cpcb_aqi)
            st.markdown("---")
            st.markdown("#### 🏷️ Official CPCB AQI")
            text_col = "#000" if col in ("#00e400", "#92d050", "#ffff00") else "#fff"
            st.markdown(
                f'''<div style="background:{col};border-radius:12px;padding:20px;text-align:center;margin:8px 0;">
                    <span style="font-size:3rem;font-weight:800;color:{text_col};">{cpcb_aqi:.0f}</span><br>
                    <span style="font-size:1.2rem;font-weight:600;color:{text_col};">{cat}</span>
                </div>''',
                unsafe_allow_html=True,
            )

        # Model performance table for this city
        if ta_df is not None:
            city_df = ta_df[ta_df["city"] == city].copy()
            if not city_df.empty:
                st.markdown("---")
                st.markdown(
                    '<div class="info-box">📊 The table below shows <b>test-set evaluation metrics</b> '
                    'for each model trained on this city. Estimated AQI is interpolated from the test MAE distribution.</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("#### 📋 Model Performance — Track A (Test Set)")

                # Build display table
                display_cols = ["model", "r2", "mae", "rmse"]
                rename_map = {"model": "Model", "r2": "R²", "mae": "MAE", "rmse": "RMSE"}
                disp = city_df[display_cols].rename(columns=rename_map).copy()
                disp["R²"]   = disp["R²"].round(4)
                disp["MAE"]  = disp["MAE"].round(2)
                disp["RMSE"] = disp["RMSE"].round(2)

                # Estimate AQI range
                if cpcb_aqi is not None:
                    disp["Est. AQI (min)"] = (cpcb_aqi - disp["MAE"]).clip(lower=0).round(1)
                    disp["Est. AQI (max)"] = (cpcb_aqi + disp["MAE"]).round(1)

                st.dataframe(
                    disp.set_index("Model").style.background_gradient(subset=["R²"], cmap="Greens"),
                    use_container_width=True,
                )

                # Bar chart
                try:
                    import plotly.express as px
                    fig = px.bar(
                        disp, x="Model", y="R²",
                        color="R²", color_continuous_scale="teal",
                        title=f"Model R² Comparison — {city} (Track A Test Set)",
                        labels={"R²": "R² Score"},
                    )
                    fig.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        font_family="Inter, Arial, sans-serif",
                        title_font_size=15, showlegend=False,
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    fig.update_traces(marker_line_width=0)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(disp.set_index("Model")["R²"])

        elif ta_df is None:
            st.warning("⚠️ Track A evaluation CSV not found. Please ensure `outputs/tables/final_track_a_complete.csv` exists.")

    # Show summary even without button press
    if ta_df is not None and city:
        city_df = ta_df[ta_df["city"] == city]
        if not city_df.empty:
            st.markdown("---")
            st.markdown("#### 📊 Quick Stats — Track A Models for " + city)
            metric_cols = st.columns(len(city_df))
            for i, (_, row) in enumerate(city_df.iterrows()):
                metric_cols[i % len(metric_cols)].metric(
                    label=row["model"], value=f"R²={row['r2']:.3f}", delta=f"MAE={row['mae']:.1f}"
                )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: AQI FORECASTING
# ════════════════════════════════════════════════════════════════════════════════
def page_forecasting():
    st.markdown('<div class="section-header">📈 AQI Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        "Select a city to view **multi-horizon AQI forecasts** (+1h, +6h, +24h) "
        "using available trained models. Results are based on test-set evaluation data."
    )

    city = st.selectbox("Select City", CITIES, key="fore_city")

    # ── Load evaluation metrics for display ──────────────────────────────────
    tb_metrics = get_all_track_b_metrics()

    # ── Discover which XGB pkl files exist for this city ────────────────────
    available_xgb = {}
    for h in HORIZONS:
        obj = load_xgb_model(city, h)
        if obj is not None:
            available_xgb[h] = obj

    # ── Check GBR models ──────────────────────────────────────────────────────
    available_gbr = {}
    for h in HORIZONS:
        obj = load_gbr_model(city, h)
        if obj is not None:
            available_gbr[h] = obj

    # ── Build forecast table from evaluation CSVs ────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔮 Forecast Performance by Horizon")
    st.markdown(
        '<div class="info-box">📌 Showing <b>test-set R² and MAE</b> from evaluation CSVs. '
        'Live XGB inference is available when pkl models are loaded successfully.</div>',
        unsafe_allow_html=True,
    )

    model_stems = {
        "XGB":        "track_b_xgb",
        "GBR":        "track_b_gbr",
        "RF":         "track_b_rf",
        "LSTM":       "track_b_lstm",
        "BiLSTM":     "track_b_bilstm",
        "CNN-BiLSTM": "track_b_cnn_bilstm",
    }

    # Build pivot table: horizons × models
    rows = []
    for h in HORIZONS:
        row = {"Horizon": HORIZON_LABELS[h]}
        for mname, stem in model_stems.items():
            df = tb_metrics.get(mname)
            if df is not None and "city" in df.columns and "horizon" in df.columns:
                sub = df[(df["city"] == city) & (df["horizon"] == h)]
                if not sub.empty:
                    r2  = sub["r2"].values[0]
                    mae = sub["mae"].values[0]
                    row[f"{mname} R²"]  = round(float(r2), 4)
                    row[f"{mname} MAE"] = round(float(mae), 2)
                else:
                    row[f"{mname} R²"]  = "N/A"
                    row[f"{mname} MAE"] = "N/A"
            else:
                row[f"{mname} R²"]  = "N/A"
                row[f"{mname} MAE"] = "N/A"
        rows.append(row)

    fore_df = pd.DataFrame(rows).set_index("Horizon")
    st.dataframe(fore_df, use_container_width=True)

    # ── Plotly line chart of R² across horizons ──────────────────────────────
    try:
        import plotly.graph_objects as go

        r2_plot_data = []
        for mname, stem in model_stems.items():
            df = tb_metrics.get(mname)
            if df is None:
                continue
            if "city" not in df.columns or "horizon" not in df.columns:
                continue
            sub = df[df["city"] == city].copy()
            if sub.empty:
                continue
            sub = sub[sub["horizon"].isin(HORIZONS)].sort_values("horizon")
            r2_plot_data.append({
                "model": mname,
                "horizons": sub["horizon"].tolist(),
                "r2s": sub["r2"].tolist(),
            })

        if r2_plot_data:
            fig = go.Figure()
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            for i, item in enumerate(r2_plot_data):
                fig.add_trace(go.Scatter(
                    x=[HORIZON_LABELS[h] for h in item["horizons"]],
                    y=item["r2s"],
                    mode="lines+markers",
                    name=item["model"],
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=8),
                ))
            fig.update_layout(
                title=f"Model R² by Forecast Horizon — {city}",
                xaxis_title="Forecast Horizon",
                yaxis_title="R² Score",
                plot_bgcolor="white", paper_bgcolor="white",
                font_family="Inter, Arial, sans-serif",
                title_font_size=15,
                legend=dict(x=1.02, y=1, bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=40, r=160, t=50, b=40),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#eee")
            fig.update_yaxes(showgrid=True, gridcolor="#eee", range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    # ── AQI Trend from BiLSTM CSV ─────────────────────────────────────────────
    bilstm_df = tb_metrics.get("BiLSTM")
    if bilstm_df is not None and "city" in bilstm_df.columns:
        city_bilstm = bilstm_df[bilstm_df["city"] == city]
        if not city_bilstm.empty:
            st.markdown("---")
            st.markdown("#### 🎯 BiLSTM Forecast Summary — " + city)
            b_cols = st.columns(len(HORIZONS))
            for i, h in enumerate(HORIZONS):
                row = city_bilstm[city_bilstm["horizon"] == h]
                if not row.empty:
                    r2  = row["r2"].values[0]
                    mae = row["mae"].values[0]
                    b_cols[i].metric(
                        label=f"BiLSTM {HORIZON_LABELS[h]}",
                        value=f"R² = {r2:.4f}",
                        delta=f"MAE = {mae:.2f}",
                    )
                else:
                    b_cols[i].metric(label=f"BiLSTM {HORIZON_LABELS[h]}", value="N/A")

    # ── Trend direction indicator ─────────────────────────────────────────────
    xgb_df = tb_metrics.get("XGB")
    if xgb_df is not None and "city" in xgb_df.columns:
        city_xgb = xgb_df[xgb_df["city"] == city].sort_values("horizon")
        if len(city_xgb) >= 2:
            st.markdown("---")
            r2_vals = city_xgb["r2"].tolist()
            if r2_vals[-1] > r2_vals[0]:
                trend = "📈 Improving accuracy at longer horizons"
                trend_col = "green"
            elif r2_vals[-1] < r2_vals[0] - 0.05:
                trend = "📉 Decreasing accuracy at longer horizons (expected)"
                trend_col = "orange"
            else:
                trend = "➡️ Stable accuracy across horizons"
                trend_col = "blue"
            st.markdown(f"**XGB Trend:** :{trend_col}[{trend}]")

    # ── XGB live model availability ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚙️ Model Availability")
    avail_cols = st.columns(len(HORIZONS))
    for i, h in enumerate(HORIZONS):
        with avail_cols[i]:
            has_xgb = h in available_xgb
            has_gbr = h in available_gbr
            status_xgb = "✅ XGB loaded" if has_xgb else "⚠️ XGB pkl not found"
            status_gbr = "✅ GBR loaded" if has_gbr else "⚠️ GBR not found"
            st.markdown(f"**{HORIZON_LABELS[h]}**")
            st.caption(status_xgb)
            st.caption(status_gbr)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
def page_model_comparison():
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

    tab_a, tab_b = st.tabs(["Track A — Estimation", "Track B — Forecasting"])

    # ── TRACK A ──────────────────────────────────────────────────────────────
    with tab_a:
        ta_df = load_track_a_csv()
        if ta_df is None:
            st.warning("Track A evaluation CSV not found.")
        else:
            st.markdown("#### 🏆 Track A — Overall Model Ranking (avg over all cities)")
            agg = (
                ta_df.groupby("model")
                .agg(avg_r2=("r2", "mean"), avg_mae=("mae", "mean"), avg_rmse=("rmse", "mean"), cities=("city", "nunique"))
                .reset_index()
                .sort_values("avg_r2", ascending=False)
            )
            agg["avg_r2"]   = agg["avg_r2"].round(4)
            agg["avg_mae"]  = agg["avg_mae"].round(2)
            agg["avg_rmse"] = agg["avg_rmse"].round(2)
            agg.columns = ["Model", "Avg R²", "Avg MAE", "Avg RMSE", "Cities"]
            st.dataframe(agg.set_index("Model").style.background_gradient(subset=["Avg R²"], cmap="Greens"), use_container_width=True)

            try:
                import plotly.express as px
                fig_a = px.bar(
                    agg, x="Model", y="Avg R²",
                    color="Avg R²", color_continuous_scale="teal",
                    title="Track A — Average R² by Model (all 18 cities)",
                )
                fig_a.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    font_family="Inter, Arial, sans-serif",
                    title_font_size=15, showlegend=False,
                )
                st.plotly_chart(fig_a, use_container_width=True)
            except ImportError:
                pass

            # Per-city heatmap
            st.markdown("#### 🗂️ Per-City R² Heatmap")
            try:
                import plotly.express as px
                pivot_r2 = ta_df.pivot_table(index="city", columns="model", values="r2")
                fig_hm = px.imshow(
                    pivot_r2, color_continuous_scale="RdYlGn",
                    zmin=0.5, zmax=1.0,
                    title="Track A R² Heatmap — City × Model",
                    aspect="auto",
                )
                fig_hm.update_layout(
                    plot_bgcolor="white", paper_bgcolor="white",
                    font_family="Inter, Arial, sans-serif",
                    title_font_size=15,
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            except ImportError:
                st.dataframe(ta_df.pivot_table(index="city", columns="model", values="r2").round(3))

    # ── TRACK B ──────────────────────────────────────────────────────────────
    with tab_b:
        tb_metrics = get_all_track_b_metrics()
        all_tb_rows = []
        for mname, df in tb_metrics.items():
            if df is not None:
                df2 = df.copy()
                df2["Model"] = mname
                all_tb_rows.append(df2)

        if not all_tb_rows:
            st.warning("No Track B evaluation CSVs found.")
        else:
            tb_all = pd.concat(all_tb_rows, ignore_index=True)

            st.markdown("#### 🏆 Track B — Model Ranking by Horizon")
            for h in HORIZONS:
                sub = tb_all[tb_all["horizon"] == h] if "horizon" in tb_all.columns else pd.DataFrame()
                if sub.empty:
                    continue
                agg_b = (
                    sub.groupby("Model")
                    .agg(avg_r2=("r2", "mean"), avg_mae=("mae", "mean"), cities=("city", "nunique"))
                    .reset_index()
                    .sort_values("avg_r2", ascending=False)
                )
                agg_b["avg_r2"]  = agg_b["avg_r2"].round(4)
                agg_b["avg_mae"] = agg_b["avg_mae"].round(2)
                st.markdown(f"**Horizon: {HORIZON_LABELS[h]}**")
                st.dataframe(
                    agg_b.rename(columns={"avg_r2": "Avg R²", "avg_mae": "Avg MAE", "cities": "Cities"})
                         .set_index("Model")
                         .style.background_gradient(subset=["Avg R²"], cmap="Greens"),
                    use_container_width=True,
                )

            # Combined bar chart
            try:
                import plotly.express as px
                if "horizon" in tb_all.columns:
                    agg_all = (
                        tb_all.groupby(["Model", "horizon"])
                        .agg(avg_r2=("r2", "mean"))
                        .reset_index()
                    )
                    agg_all["horizon"] = agg_all["horizon"].map(HORIZON_LABELS)
                    fig_b = px.bar(
                        agg_all, x="Model", y="avg_r2", color="horizon",
                        barmode="group",
                        title="Track B — Avg R² by Model and Horizon",
                        labels={"avg_r2": "Avg R²", "horizon": "Horizon"},
                        color_discrete_sequence=["#1e3a5f", "#00d4aa", "#ff6b35"],
                    )
                    fig_b.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        font_family="Inter, Arial, sans-serif",
                        title_font_size=15,
                    )
                    st.plotly_chart(fig_b, use_container_width=True)
            except ImportError:
                pass


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: INDIA AQI MAP
# ════════════════════════════════════════════════════════════════════════════════
def page_india_map():
    st.markdown('<div class="section-header">🗺️ India AQI Map</div>', unsafe_allow_html=True)
    st.markdown("Interactive map showing all 18 monitored cities, colored by average model R² for the selected horizon.")

    horizon = st.selectbox("Select Forecast Horizon", HORIZONS, format_func=lambda h: HORIZON_LABELS[h], key="map_horizon")

    tb_metrics = get_all_track_b_metrics()

    # Compute average R² per city for selected horizon
    city_r2 = {}
    for city in CITIES:
        r2_vals = []
        for mname, df in tb_metrics.items():
            if df is None:
                continue
            if "city" not in df.columns or "horizon" not in df.columns:
                continue
            sub = df[(df["city"] == city) & (df["horizon"] == horizon)]
            if not sub.empty:
                r2_vals.append(float(sub["r2"].values[0]))
        city_r2[city] = round(float(np.mean(r2_vals)), 4) if r2_vals else float("nan")

    # Build geo dataframe
    geo_rows = []
    for city in CITIES:
        lat, lon = CITY_COORDS[city]
        r2 = city_r2.get(city, float("nan"))
        geo_rows.append({
            "City": city.replace("_", " "),
            "Latitude": lat,
            "Longitude": lon,
            "Avg R²": r2,
            "R² Display": f"{r2:.4f}" if not np.isnan(r2) else "N/A",
        })
    geo_df = pd.DataFrame(geo_rows)

    try:
        import plotly.express as px
        fig_map = px.scatter_geo(
            geo_df,
            lat="Latitude",
            lon="Longitude",
            color="Avg R²",
            size=[20] * len(geo_df),
            hover_name="City",
            hover_data={"Latitude": False, "Longitude": False, "Avg R²": ":.4f"},
            color_continuous_scale="RdYlGn",
            range_color=[0.6, 1.0],
            title=f"India AQI Monitoring Cities — Avg R² at {HORIZON_LABELS[horizon]}",
            scope="asia",
        )
        fig_map.update_geos(
            center={"lat": 22, "lon": 78},
            projection_scale=4,
            showland=True, landcolor="#e8f4e8",
            showocean=True, oceancolor="#d0e8f8",
            showcountries=True, countrycolor="#999",
            showsubunits=True, subunitcolor="#ccc",
            showrivers=True, rivercolor="#a0c8e8",
        )
        fig_map.update_layout(
            height=600,
            font_family="Inter, Arial, sans-serif",
            paper_bgcolor="white",
            title_font_size=16,
            coloraxis_colorbar=dict(title="Avg R²"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    except ImportError:
        st.warning("Plotly not available. Showing table instead.")
        st.dataframe(geo_df[["City", "Avg R²"]].sort_values("Avg R²", ascending=False), use_container_width=True)

    # City detail table below the map
    st.markdown("---")
    st.markdown("#### 📋 City-Level Performance Details")
    city_sel = st.selectbox("Select City for Details", CITIES, key="map_city_detail")
    detail_rows = []
    for mname, df in tb_metrics.items():
        if df is None:
            continue
        if "city" not in df.columns or "horizon" not in df.columns:
            continue
        sub = df[(df["city"] == city_sel) & (df["horizon"] == horizon)]
        if not sub.empty:
            detail_rows.append({
                "Model": mname,
                "R²": round(float(sub["r2"].values[0]), 4),
                "MAE": round(float(sub["mae"].values[0]), 2),
                "RMSE": round(float(sub["rmse"].values[0]), 2) if "rmse" in sub.columns else "N/A",
            })
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows).set_index("Model").sort_values("R²", ascending=False)
        st.dataframe(detail_df.style.background_gradient(subset=["R²"], cmap="Greens"), use_container_width=True)
    else:
        st.info(f"No evaluation data available for {city_sel} at {HORIZON_LABELS[horizon]}.")

    # Summary metrics
    valid_r2 = [v for v in city_r2.values() if not np.isnan(v)]
    if valid_r2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Best City R²", f"{max(valid_r2):.4f}", delta=max(city_r2, key=lambda c: city_r2.get(c, 0) if not np.isnan(city_r2.get(c, float('nan'))) else 0))
        col2.metric("Avg R² (all cities)", f"{np.mean(valid_r2):.4f}")
        col3.metric("Cities with data", str(len(valid_r2)))


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT PROJECT
# ════════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="section-header">👨‍💻 About Project</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""
        ## AQI Prediction Using Machine Learning and Deep Learning
        **Research Internship** — IIIT Nagpur in collaboration with YCCE Nagpur

        ### 🎯 Project Objectives
        - Develop accurate AQI estimation and forecasting models for 18 major Indian cities
        - Benchmark 7 ML/DL algorithms under a rigorous dual-track evaluation protocol
        - Implement leakage-free forecasting (Track B) with properly lagged features
        - Enable actionable insights for public health and environmental monitoring

        ### 🏛️ Institutional Context
        | | |
        |---|---|
        | **Research Institution** | IIIT Nagpur (Indian Institute of Information Technology) |
        | **Collaborating Institution** | YCCE Nagpur (Yeshwantrao Chavan College of Engineering) |
        | **Data Source** | CPCB — Central Pollution Control Board of India |
        | **Dataset Period** | 2018–2023 (multi-year hourly records) |

        ### 📊 Dataset Overview
        | Metric | Value |
        |---|---|
        | Raw records | ~18.7 million (15-min measurements) |
        | Hourly records (after preprocessing) | 934,775 |
        | Cities covered | 18 major Indian cities |
        | Pollutants measured | 11 (PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, Toluene) |
        | Meteorological features | 6 (Temperature, Humidity, Wind Speed, Direction, Solar Radiation, Pressure) |
        | Train/Val/Test split | 70% / 15% / 15% (chronological) |
        """)

    with col_r:
        st.markdown("### 🌆 Monitored Cities")
        for c in CITIES:
            st.markdown(f"• {c.replace('_', ' ')}")

    st.markdown("---")
    st.markdown("""
    ### 🔬 Dual-Track Methodology

    #### Track A — AQI Estimation (Same-Time Features)
    Models receive same-timestamp pollutant readings as features to estimate AQI.
    This simulates **real-time sensor fusion** where all measurements arrive simultaneously.
    Features: All 11 pollutants + 6 meteorological + temporal encodings + interaction terms.

    #### Track B — AQI Forecasting (Lag-Only Features — No Data Leakage)
    Models use only **past values** (lag and rolling features) to forecast AQI at +1h, +6h, +24h.
    Same-time pollutants are explicitly excluded — this ensures genuine forecast capability.
    Features: lag_1...lag_24, rolling mean/std/min/max at 3/6/12/24h + temporal encodings.

    ### 🤖 Models Evaluated
    """)

    model_info = [
        ("Ridge Regression",    "Baseline linear model with L2 regularization. Fast, interpretable.",    "Track A"),
        ("Random Forest",       "Ensemble of 300 decision trees. Robust to noise and outliers.",          "A + B"),
        ("Gradient Boosting",   "Sequential tree ensemble. **Best overall** — avg R² = 0.99 (Track A).", "A + B"),
        ("XGBoost",             "Optimized gradient boosting. 39 pkl files trained for Track B.",         "A + B"),
        ("LSTM",                "Long Short-Term Memory. Captures sequential AQI patterns.",              "A + B"),
        ("BiLSTM",              "Bidirectional LSTM. Forward + backward temporal context.",               "A + B"),
        ("CNN-BiLSTM",          "Convolutional + BiLSTM hybrid. Local pattern + temporal modeling.",      "A + B"),
    ]
    m_cols = st.columns(3)
    for i, (name, desc, tracks) in enumerate(model_info):
        m_cols[i % 3].markdown(
            f'''<div class="model-card">
                <b>{name}</b><br>
                <span style="font-size:0.82rem;color:#444;">{desc}</span><br>
                <span style="font-size:0.78rem;color:#1e3a5f;font-weight:600;">Tracks: {tracks}</span>
            </div>''',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    ### 🏆 Key Results

    | Model | Track A Avg R² | Track A Avg MAE | Track B Best R² (h=1) |
    |---|---|---|---|
    | **GBR** | **0.9906** | 2.95 | ~0.90 |
    | **RF** | 0.9874 | 1.64 | ~0.88 |
    | **XGBoost** | 0.9856 | 2.83 | Available via pkl |
    | **Ridge** | 0.8304 | — | — |
    | **LSTM** | — | — | ~0.85 |
    | **BiLSTM** | — | — | ~0.87 |
    | **CNN-BiLSTM** | — | — | ~0.86 |

    ### 🔍 Key Scientific Findings
    - **GBR dominates estimation**: Tree-based ensemble methods outperform deep learning for same-time AQI estimation
    - **Forecast accuracy degrades with horizon**: R² drops ~5–15% from +1h to +24h (expected)
    - **Leakage audit**: Same-time pollutants inflate Track A performance; Track B uses only lagged features
    - **City variation**: Singrauli and Nagpur show highest predictability; coastal cities more variable
    - **Feature importance**: PM2.5 lag features are most predictive for forecasting horizons

    ### 📦 Deployment Architecture
    - **Backend**: Python 3.10+, scikit-learn, XGBoost, TensorFlow/Keras, joblib
    - **Frontend**: Streamlit web application (this platform)
    - **Model storage**: 39 XGB pkl files + optional GBR/LSTM joblib/keras artifacts
    - **Inference**: On-demand loading with `@st.cache_data` for performance
    """)

    st.markdown("---")
    st.markdown(
        '<div class="info-box">🔬 This platform was built as part of a research internship project. '
        'All results are based on real CPCB data and rigorous experimental evaluation. '
        'For academic queries, refer to the project documentation in the repository.</div>',
        unsafe_allow_html=True,
    )


# ─── Main Router ──────────────────────────────────────────────────────────────
# Handle button-driven navigation
if "_nav" in st.session_state:
    nav = st.session_state.pop("_nav")
    # Override page by rerunning with sidebar pre-selected
    # (Streamlit doesn't support programmatic radio nav easily — show a note)
    pass

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
