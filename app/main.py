"""
AQI Prediction Using Machine Learning and Deep Learning
=======================================================
Research Internship · Indian Institute of Information Technology Nagpur
CPCB Multi-City Dataset — 18 Indian Cities — 2018–2023

Streamlit Dashboard — 6 Pages
  1. Home
  2. AQI Estimation
  3. AQI Forecasting
  4. Model Comparison
  5. India AQI Map
  6. About
"""

import streamlit as st

# ── must be the first Streamlit call ──────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction Platform",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ROOT / PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
def find_root() -> Path:
    candidates = [
        Path("."),
        Path(__file__).parent.parent,
        Path(__file__).parent,
        Path("/tmp/tmpqfom5tnq/files"),
    ]
    for c in candidates:
        if (c / "AQI_Prediction_Project_Final (1)").exists():
            return c.resolve()
    return Path(".").resolve()


ROOT    = find_root()
PROJECT = ROOT / "AQI_Prediction_Project_Final (1)"
TABLES  = PROJECT / "outputs" / "tables"
PREDS   = PROJECT / "outputs" / "predictions"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS / LOOKUPS
# ─────────────────────────────────────────────────────────────────────────────
CITY_MAP = {
    "Ahmedabad":      "Ahmedabad",
    "Bhopal":         "bhopal",
    "Chennai":        "Chennai",
    "Delhi NCR":      "Delhi_NCR",
    "Gandhinagar":    "GandhiNagar",
    "Hyderabad":      "Hyderabad",
    "Indore":         "Indore",
    "Jaipur":         "Jaipur",
    "Jodhpur":        "Jodhpur",
    "Mumbai":         "Mumbai",
    "Mumbai Suburbs": "Mumbai_suburbs",
    "Nagpur":         "Nagpur",
    "Pune":           "Pune",
    "Singrauli":      "Singrauli",
    "Surat":          "Surat",
    "Thane":          "Thane",
    "Vapi":           "Vapi",
    "Visakhapatnam":  "vishakhapattanam",
}

HORIZON_MAP = {"1 Hour": "01", "6 Hours": "06", "24 Hours": "24"}

CITY_COORDS = {
    "Ahmedabad":      (23.03, 72.58),
    "bhopal":         (23.26, 77.41),
    "Chennai":        (13.08, 80.27),
    "Delhi_NCR":      (28.70, 77.10),
    "GandhiNagar":    (23.22, 72.64),
    "Hyderabad":      (17.39, 78.49),
    "Indore":         (22.72, 75.86),
    "Jaipur":         (26.91, 75.79),
    "Jodhpur":        (26.29, 73.02),
    "Mumbai":         (19.08, 72.88),
    "Mumbai_suburbs": (19.20, 72.98),
    "Nagpur":         (21.15, 79.09),
    "Pune":           (18.52, 73.86),
    "Singrauli":      (24.20, 82.67),
    "Surat":          (21.17, 72.83),
    "Thane":          (19.22, 72.98),
    "Vapi":           (20.37, 72.91),
    "vishakhapattanam":(17.69, 83.22),
}

# CPCB AQI breakpoints: (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    "PM2.5": [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)],
    "PM10":  [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
    "NO2":   [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
    "SO2":   [(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2100,401,500)],
    "CO":    [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
    "Ozone": [(0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
    "NH3":   [(0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2400,401,500)],
}

AQI_CATEGORIES = [
    (0,   50,  "Good",        "#22c55e"),
    (51,  100, "Satisfactory","#84cc16"),
    (101, 200, "Moderate",    "#eab308"),
    (201, 300, "Poor",        "#f97316"),
    (301, 400, "Very Poor",   "#ef4444"),
    (401, 500, "Severe",      "#7c3aed"),
]

HEALTH_ADVISORIES = {
    "Good":         "Air quality is satisfactory. Enjoy outdoor activities.",
    "Satisfactory": "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    "Moderate":     "Members of sensitive groups may experience health effects. General public is less likely to be affected.",
    "Poor":         "Everyone may begin to experience health effects. Members of sensitive groups may experience more serious effects.",
    "Very Poor":    "Health alert: everyone may experience more serious health effects. Avoid prolonged outdoor activity.",
    "Severe":       "Health warnings of emergency conditions. The entire population is likely to be affected. Stay indoors.",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS (dark theme + glassmorphism)
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
  /* Base */
  html, body, [data-testid="stAppViewContainer"] {
      background-color: #0a0e1a !important;
      color: #e2e8f0 !important;
  }
  [data-testid="stSidebar"] {
      background-color: #0d1117 !important;
      border-right: 1px solid rgba(255,255,255,0.08);
  }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Glass card */
  .glass-card {
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1rem;
  }
  .glass-card-accent {
      background: linear-gradient(135deg,rgba(59,130,246,0.15),rgba(6,182,212,0.10));
      backdrop-filter: blur(10px);
      border: 1px solid rgba(59,130,246,0.3);
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1rem;
  }

  /* Hero gradient text */
  .hero-title {
      font-size: 2.4rem;
      font-weight: 800;
      background: linear-gradient(90deg,#3b82f6,#06b6d4);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.2;
      margin-bottom: 0.4rem;
  }
  .hero-sub {
      font-size: 1.05rem;
      color: #94a3b8;
      margin-bottom: 1.4rem;
  }

  /* KPI card */
  .kpi-card {
      text-align: center;
      background: rgba(59,130,246,0.08);
      border: 1px solid rgba(59,130,246,0.25);
      border-radius: 10px;
      padding: 1.1rem 0.5rem;
  }
  .kpi-value {
      font-size: 2rem;
      font-weight: 800;
      color: #06b6d4;
  }
  .kpi-label {
      font-size: 0.82rem;
      color: #94a3b8;
      margin-top: 0.2rem;
  }

  /* Metric badge */
  .metric-pill {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
  }

  /* AQI category colors */
  .aqi-good        { background: #166534; color: #bbf7d0; }
  .aqi-satisfactory{ background: #365314; color: #d9f99d; }
  .aqi-moderate    { background: #713f12; color: #fef08a; }
  .aqi-poor        { background: #7c2d12; color: #fdba74; }
  .aqi-verypoor    { background: #7f1d1d; color: #fca5a5; }
  .aqi-severe      { background: #4c1d95; color: #ddd6fe; }

  /* Section headers */
  .section-header {
      font-size: 1.35rem;
      font-weight: 700;
      color: #e2e8f0;
      margin: 1.4rem 0 0.6rem 0;
      padding-bottom: 0.3rem;
      border-bottom: 1px solid rgba(255,255,255,0.08);
  }

  /* Dataframe overrides */
  [data-testid="stDataFrame"] { border-radius: 8px; }

  /* Predicted AQI display */
  .big-aqi {
      font-size: 4.5rem;
      font-weight: 900;
      line-height: 1;
  }

  /* Scrollable leaderboard */
  .scrollable { max-height: 420px; overflow-y: auto; }

  /* Link buttons spacing */
  .stLinkButton { margin-right: 0.5rem; }

  /* Tab styling */
  [data-testid="stTabs"] button {
      font-size: 0.95rem;
      font-weight: 600;
  }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_track_a_csv() -> pd.DataFrame:
    """Load final_track_a_complete.csv (estimation results)."""
    try:
        p = TABLES / "final_track_a_complete.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_track_b_xgb() -> pd.DataFrame:
    try:
        p = TABLES / "track_b_xgb.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_track_b_bilstm() -> pd.DataFrame:
    try:
        p = TABLES / "track_b_bilstm.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_track_b_complete() -> pd.DataFrame:
    """Load final_track_b_complete.csv; fall back to combining individual csvs."""
    try:
        p = TABLES / "final_track_b_complete.csv"
        if p.exists():
            return pd.read_csv(p)
        # fallback: combine available track_b_*.csv files
        frames = []
        for stem in ["track_b_xgb","track_b_bilstm","track_b_gbr",
                     "track_b_rf","track_b_lstm","track_b_cnn_bilstm"]:
            fp = TABLES / f"{stem}.csv"
            if fp.exists():
                df = pd.read_csv(fp)
                if "model" not in df.columns:
                    df["model"] = stem.replace("track_b_","").upper()
                frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_city_parquet(city_key: str) -> pd.DataFrame | None:
    try:
        pq_path = ROOT / f"outputs_recovered_{city_key}_recovered.parquet"
        if pq_path.exists():
            return pd.read_parquet(pq_path)
    except Exception:
        pass
    return None


@st.cache_resource
def load_xgb_model(city_key: str, hz: str):
    """Load a pickled XGBRegressor dict. Returns None if not found."""
    try:
        pkl_path = PREDS / f"{city_key}_h{hz}_xgb.pkl"
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_prediction_csv(city_key: str, hz: str) -> pd.DataFrame | None:
    """Load {City}_h{hz}_predictions.csv from PREDS folder."""
    try:
        p = PREDS / f"{city_key}_h{hz}_predictions.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def compute_sub_index(value: float, pollutant: str) -> float:
    """Compute CPCB sub-index for a single pollutant reading."""
    if pollutant not in CPCB_BREAKPOINTS or value is None or np.isnan(value):
        return 0.0
    bps = CPCB_BREAKPOINTS[pollutant]
    for (c_lo, c_hi, i_lo, i_hi) in bps:
        if c_lo <= value <= c_hi:
            return i_lo + (value - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    if value > bps[-1][1]:
        return 500.0
    return 0.0


def compute_aqi(readings: dict) -> tuple[float, dict]:
    """Compute overall AQI and sub-index dict from pollutant readings."""
    sub_indices = {}
    for poll, val in readings.items():
        if val is not None and not np.isnan(val):
            sub_indices[poll] = compute_sub_index(val, poll)
    if not sub_indices:
        return 0.0, {}
    aqi = max(sub_indices.values())
    return aqi, sub_indices


def get_aqi_category(aqi: float) -> tuple[str, str]:
    """Return (category_name, hex_color) for an AQI value."""
    for lo, hi, cat, col in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return cat, col
    if aqi > 500:
        return "Severe", "#7c3aed"
    return "Good", "#22c55e"


def aqi_css_class(cat: str) -> str:
    mapping = {
        "Good": "aqi-good",
        "Satisfactory": "aqi-satisfactory",
        "Moderate": "aqi-moderate",
        "Poor": "aqi-poor",
        "Very Poor": "aqi-verypoor",
        "Severe": "aqi-severe",
    }
    return mapping.get(cat, "aqi-good")


def predict_xgb(city_key: str, hz: str) -> tuple[float | None, str | None]:
    """
    Run live XGBoost inference on the last complete row of the city parquet.
    Returns (predicted_aqi, error_message). error_message is None on success.
    """
    obj = load_xgb_model(city_key, hz)
    if obj is None:
        return None, "XGBoost model not available for this city/horizon combination."

    df = load_city_parquet(city_key)
    if df is None:
        return None, "City data file not found."

    feat_cols  = obj["feat_cols"]
    safe_names = obj["safe_names"]
    model      = obj["model"]

    available = [c for c in feat_cols if c in df.columns]
    if len(available) < len(feat_cols) * 0.5:
        return None, "Insufficient feature data available for inference."

    sub = df[[c for c in feat_cols if c in df.columns]].dropna()
    if len(sub) == 0:
        return None, "No complete data rows available for inference."

    last_row = sub.iloc[[-1]]

    row_dict = {}
    for orig, safe in zip(feat_cols, safe_names):
        if orig in last_row.columns:
            row_dict[safe] = float(last_row[orig].values[0])
        else:
            row_dict[safe] = 0.0

    input_df = pd.DataFrame([row_dict])
    pred = model.predict(input_df)[0]
    pred = max(0.0, float(pred))
    return pred, None


def get_bilstm_forecast(city_key: str, hz: str) -> tuple[float | None, dict | None]:
    """
    Return last BiLSTM prediction value from the prediction CSV + eval metrics.
    Returns (pred_value, metrics_dict). metrics_dict keys: r2, mae, rmse.
    """
    pred_val = None
    metrics  = None

    # Try to load prediction CSV
    pdf = load_prediction_csv(city_key, hz)
    if pdf is not None and "bilstm_pred" in pdf.columns:
        vals = pdf["bilstm_pred"].dropna()
        if len(vals) > 0:
            pred_val = float(vals.iloc[-1])

    # Load eval CSV
    bilstm_df = load_track_b_bilstm()
    if not bilstm_df.empty:
        HORIZON_LABEL_MAP = {"01": "1h", "06": "6h", "24": "24h",
                             "01": "h01", "06": "h06", "24": "h24"}
        match = bilstm_df[
            (bilstm_df["city"].str.lower() == city_key.lower()) &
            (bilstm_df["horizon"].astype(str).str.contains(hz, na=False))
        ]
        if match.empty:
            # try partial match
            for _, row in bilstm_df.iterrows():
                if city_key.lower() in str(row.get("city","")).lower():
                    hz_str = str(row.get("horizon",""))
                    if hz.lstrip("0") in hz_str or hz in hz_str:
                        match = bilstm_df.iloc[[row.name]]
                        break
        if not match.empty:
            row = match.iloc[0]
            metrics = {
                "r2":   round(float(row.get("r2",   0)), 4),
                "mae":  round(float(row.get("mae",  0)), 2),
                "rmse": round(float(row.get("rmse", 0)), 2),
            }

    return pred_val, metrics


def xgb_pkl_exists(city_key: str, hz: str) -> bool:
    return (PREDS / f"{city_key}_h{hz}_xgb.pkl").exists()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="padding:0.6rem 0 1.2rem 0;">
        <span style="font-size:1.5rem;font-weight:800;
          background:linear-gradient(90deg,#3b82f6,#06b6d4);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;">🌫️ AQI Platform</span>
        </div>""",
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
            "📖 About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#475569;'>"
        "CPCB Multi-City Dataset<br>18 Cities · 2018–2023<br>7 ML/DL Models"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    # Hero
    st.markdown(
        '''<div class="glass-card-accent">
        <div class="hero-title">Air Quality Index Prediction<br>Using Machine Learning and Deep Learning</div>
        <div class="hero-sub">Research Internship &nbsp;·&nbsp; Indian Institute of Information Technology Nagpur</div>
        </div>''',
        unsafe_allow_html=True,
    )

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, "18",   "Cities"),
        (k2, "18.7M","Records"),
        (k3, "7",    "Models"),
        (k4, "3",    "Horizons"),
    ]:
        col.markdown(
            f'''<div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{lbl}</div>
            </div>''',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Research objective cards
    obj1, obj2 = st.columns(2)
    with obj1:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1.1rem;font-weight:700;color:#06b6d4;margin-bottom:0.5rem;">
                🔬 Current AQI Estimation
            </div>
            <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.6;">
                Estimate present AQI using ML &amp; DL models trained on same-time pollutant 
                and meteorological data. Covers Ridge Regression, Random Forest, Gradient Boosting, 
                XGBoost, LSTM, BiLSTM, and CNN-BiLSTM across 18 Indian cities.
            </div>
            </div>''',
            unsafe_allow_html=True,
        )
    with obj2:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1.1rem;font-weight:700;color:#3b82f6;margin-bottom:0.5rem;">
                📈 Multi-Horizon AQI Forecasting
            </div>
            <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.6;">
                Predict future AQI at +1h, +6h, +24h horizons using lag/rolling 
                feature-based forecasting models. Ensures full data leakage protection 
                — no same-time pollutant features are used in forecasting models.
            </div>
            </div>''',
            unsafe_allow_html=True,
        )

    # Link buttons
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.link_button(
            "🔗 GitHub Repository",
            "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning",
        )
    with c2:
        st.link_button(
            "📊 CPCB Multi-City Dataset",
            "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Research highlights bar chart — top 3 estimation models by R²
    st.markdown('<div class="section-header">📊 Research Highlights — Top Estimation Models</div>', unsafe_allow_html=True)
    ta = load_track_a_csv()
    if not ta.empty and "model" in ta.columns and "r2" in ta.columns:
        avg = ta.groupby("model")["r2"].mean().reset_index()
        avg.columns = ["Model", "Avg R²"]
        avg = avg.sort_values("Avg R²", ascending=False).head(3)
        PALETTE = ["#3b82f6", "#06b6d4", "#6366f1"]
        fig = go.Figure(go.Bar(
            x=avg["Model"],
            y=avg["Avg R²"],
            marker_color=PALETTE[:len(avg)],
            text=[f"{v:.4f}" for v in avg["Avg R²"]],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=13),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 1.05], title="Average R²",
                       gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Model"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Estimation results CSV not found — run training pipeline first.")

    # Dataset info
    st.markdown('<div class="section-header">📂 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass-card">
        <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.8;">
            <b style="color:#06b6d4;">Source:</b> CPCB (Central Pollution Control Board) Multi-City Dataset<br>
            <b style="color:#06b6d4;">Cities:</b> 18 Indian cities — Ahmedabad, Bhopal, Chennai, Delhi NCR, Gandhinagar, 
            Hyderabad, Indore, Jaipur, Jodhpur, Mumbai, Mumbai Suburbs, Nagpur, Pune, Singrauli, Surat, Thane, 
            Vapi, Visakhapatnam<br>
            <b style="color:#06b6d4;">Period:</b> 2018 – 2023 (hourly resolution)<br>
            <b style="color:#06b6d4;">Records:</b> ~18.7 million observations<br>
            <b style="color:#06b6d4;">Features:</b> 11 pollutants (PM2.5, PM10, NO, NO₂, NOₓ, NH₃, SO₂, CO, Ozone, Benzene, Toluene) 
            + 6 meteorological variables + time encodings + lag/rolling features
        </div>
        </div>''',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: AQI ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
def page_estimation():
    st.markdown('<div class="section-header" style="font-size:1.6rem;">🔬 AQI Estimation</div>', unsafe_allow_html=True)

    # ── Part A: CPCB AQI Calculator ──────────────────────────────────────────
    st.markdown('<div class="section-header">🧮 CPCB AQI Calculator</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass-card" style="margin-bottom:0.5rem;">
        <span style="color:#94a3b8;font-size:0.88rem;">
            Enter pollutant concentrations (µg/m³ except CO in mg/m³) to compute the CPCB AQI 
            and health category using official Indian breakpoints.
        </span>
        </div>''',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pm25  = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=45.0, step=1.0)
        no2   = st.number_input("NO₂ (µg/m³)",   min_value=0.0, max_value=800.0, value=30.0, step=1.0)
    with col2:
        pm10  = st.number_input("PM10 (µg/m³)",  min_value=0.0, max_value=600.0, value=80.0, step=1.0)
        so2   = st.number_input("SO₂ (µg/m³)",   min_value=0.0, max_value=2100.0,value=10.0, step=1.0)
    with col3:
        co    = st.number_input("CO (mg/m³)",     min_value=0.0, max_value=50.0,  value=1.5,  step=0.1)
        nh3   = st.number_input("NH₃ (µg/m³)",   min_value=0.0, max_value=2400.0,value=15.0, step=1.0)
    with col4:
        ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, max_value=1000.0,value=40.0, step=1.0)

    readings = {
        "PM2.5": pm25,
        "PM10":  pm10,
        "NO2":   no2,
        "SO2":   so2,
        "CO":    co,
        "Ozone": ozone,
        "NH3":   nh3,
    }

    aqi_val, sub_indices = compute_aqi(readings)
    cat, color = get_aqi_category(aqi_val)
    css_cls = aqi_css_class(cat)
    advisory = HEALTH_ADVISORIES.get(cat, "")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown(
            f'''<div class="glass-card" style="text-align:center;">
            <div style="color:#94a3b8;font-size:0.85rem;margin-bottom:0.5rem;">Calculated AQI</div>
            <div class="big-aqi" style="color:{color};">{aqi_val:.0f}</div>
            <br>
            <span class="metric-pill {css_cls}" style="font-size:1rem;">{cat}</span>
            <br><br>
            <div style="color:#94a3b8;font-size:0.82rem;line-height:1.5;">{advisory}</div>
            </div>''',
            unsafe_allow_html=True,
        )

    with res_col2:
        if sub_indices:
            polls = list(sub_indices.keys())
            vals  = [sub_indices[p] for p in polls]
            bar_colors = []
            for v in vals:
                c2, _ = get_aqi_category(v)
                bar_colors.append(dict(AQI_CATEGORIES)[(0,50,"Good","#22c55e")
                    if c2=="Good" else
                    next((lo,hi,cat2,col2) for lo,hi,cat2,col2 in AQI_CATEGORIES if cat2==c2)][-1])
            # simpler approach: color by value
            def val_to_color(v):
                _, c2 = get_aqi_category(v)
                return c2
            bar_colors = [get_aqi_category(v)[1] for v in vals]

            fig = go.Figure(go.Bar(
                x=polls,
                y=vals,
                marker_color=bar_colors,
                text=[f"{v:.0f}" for v in vals],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=12),
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Sub-Index per Pollutant", font=dict(color="#e2e8f0", size=14)),
                yaxis=dict(range=[0, 550], title="Sub-Index",
                           gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(title="Pollutant"),
                margin=dict(l=10, r=10, t=40, b=20),
                height=260,
                showlegend=False,
            )
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Part B: Model Leaderboard ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 AQI Estimation — Model Performance (Current-Time Prediction)</div>', unsafe_allow_html=True)

    ta = load_track_a_csv()
    if ta.empty:
        st.warning("⚠️ Estimation results file not found. Please ensure final_track_a_complete.csv exists in outputs/tables/.")
        return

    if "model" not in ta.columns:
        st.warning("⚠️ Unexpected CSV format in final_track_a_complete.csv.")
        return

    avg_ta = ta.groupby("model")[["r2","mae","rmse"]].mean().reset_index()
    avg_ta = avg_ta.sort_values("r2", ascending=False).reset_index(drop=True)
    avg_ta.index = avg_ta.index + 1
    avg_ta.index.name = "Rank"

    display_ta = avg_ta.copy()
    display_ta.columns = ["Model", "Avg R²", "Avg MAE", "Avg RMSE"]
    display_ta["Avg R²"]   = display_ta["Avg R²"].round(4)
    display_ta["Avg MAE"]  = display_ta["Avg MAE"].round(2)
    display_ta["Avg RMSE"] = display_ta["Avg RMSE"].round(2)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.dataframe(display_ta, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bar chart: R² per model
    fig2 = go.Figure(go.Bar(
        y=avg_ta["model"],
        x=avg_ta["r2"],
        orientation="h",
        marker=dict(
            color=avg_ta["r2"],
            colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#06b6d4"]],
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in avg_ta["r2"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=12),
    ))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Average R² by Model (Estimation)", font=dict(color="#e2e8f0",size=14)),
        xaxis=dict(range=[0, 1.08], title="R²", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=60, t=40, b=20),
        height=300,
    )
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

    # Per-city R² heatmap
    if "city" in ta.columns:
        st.markdown('<div class="section-header">🗺️ Per-City R² Heatmap (Estimation)</div>', unsafe_allow_html=True)
        try:
            pivot = ta.pivot_table(index="city", columns="model", values="r2", aggfunc="mean")
            pivot = pivot.fillna(0)
            fig3 = go.Figure(go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale="Blues",
                zmin=0, zmax=1,
                text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=9, color="white"),
                hovertemplate="City: %{y}<br>Model: %{x}<br>R²: %{z:.4f}<extra></extra>",
            ))
            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="R² Heatmap — Estimation Models × Cities", font=dict(color="#e2e8f0",size=14)),
                margin=dict(l=10, r=10, t=50, b=10),
                height=max(350, len(pivot.index)*28 + 80),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.info(f"Heatmap could not be rendered: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: AQI FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
def page_forecasting():
    st.markdown('<div class="section-header" style="font-size:1.6rem;">📈 AQI Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass-card" style="margin-bottom:0.8rem;">
        <span style="color:#94a3b8;font-size:0.88rem;">
            Select a city, forecast horizon, and model to generate a future AQI prediction.
            XGBoost uses live inference from the trained model artifact; BiLSTM displays 
            verified test-set evaluation results.
        </span>
        </div>''',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        city_display = st.selectbox("🏙️ City", list(CITY_MAP.keys()), index=0)
    with col2:
        horizon_display = st.selectbox("⏱️ Forecast Horizon", list(HORIZON_MAP.keys()), index=0)
    with col3:
        model_choice = st.selectbox("🤖 Model", ["XGBoost", "BiLSTM"], index=0)

    city_key = CITY_MAP[city_display]
    hz       = HORIZON_MAP[horizon_display]

    run_btn = st.button("🚀 Forecast Now", use_container_width=True, type="primary")

    if not run_btn:
        return

    pred_aqi  = None
    error_msg = None
    data_source = ""
    metrics = None

    # ── XGBoost inference ─────────────────────────────────────────────────────
    if model_choice == "XGBoost":
        pred_aqi, error_msg = predict_xgb(city_key, hz)
        data_source = "Live inference from trained XGBoost model"

        if error_msg:
            st.warning(
                f"⚠️ The selected forecasting model cannot generate a prediction for this city/horizon. "
                f"{error_msg} "
                "Displaying the nearest available evaluation result instead."
            )
            # fallback: show eval row from track_b_xgb.csv
            xgb_df = load_track_b_xgb()
            if not xgb_df.empty:
                match = xgb_df[
                    (xgb_df["city"].str.lower() == city_key.lower()) &
                    (xgb_df["horizon"].astype(str).str.contains(hz, na=False))
                ]
                if not match.empty:
                    row = match.iloc[0]
                    metrics = {
                        "r2":   round(float(row.get("r2",   0)), 4),
                        "mae":  round(float(row.get("mae",  0)), 2),
                        "rmse": round(float(row.get("rmse", 0)), 2),
                    }
                    data_source = "Nearest evaluation result (model artifact unavailable)"
                    pred_aqi = None  # can't show a forecast value without the model
        else:
            # load eval metrics
            xgb_df = load_track_b_xgb()
            if not xgb_df.empty:
                match = xgb_df[
                    (xgb_df["city"].str.lower() == city_key.lower()) &
                    (xgb_df["horizon"].astype(str).str.contains(hz, na=False))
                ]
                if not match.empty:
                    row = match.iloc[0]
                    metrics = {
                        "r2":   round(float(row.get("r2",   0)), 4),
                        "mae":  round(float(row.get("mae",  0)), 2),
                        "rmse": round(float(row.get("rmse", 0)), 2),
                    }

    # ── BiLSTM display ────────────────────────────────────────────────────────
    else:  # BiLSTM
        pred_aqi, metrics = get_bilstm_forecast(city_key, hz)
        data_source = "Verified test-set result (BiLSTM — model artifact not deployed)"

        if pred_aqi is None and metrics is None:
            st.warning(
                "⚠️ The selected forecasting model cannot generate a prediction for this city/horizon. "
                "This may be because the trained model artifact or required feature data is unavailable. "
                "Displaying the nearest available evaluation result instead."
            )

    # ── Forecast Result Card ──────────────────────────────────────────────────
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if pred_aqi is not None:
        cat, color = get_aqi_category(pred_aqi)
        css_cls = aqi_css_class(cat)
        advisory = HEALTH_ADVISORIES.get(cat, "")

        st.markdown(
            f'''<div class="glass-card-accent">
            <div style="display:flex;align-items:flex-start;gap:2rem;flex-wrap:wrap;">
              <div style="min-width:160px;text-align:center;">
                <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:4px;">Predicted AQI</div>
                <div class="big-aqi" style="color:{color};">{pred_aqi:.1f}</div>
                <span class="metric-pill {css_cls}" style="margin-top:0.4rem;display:inline-block;">{cat}</span>
              </div>
              <div style="flex:1;min-width:200px;">
                <table style="width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.88rem;">
                  <tr><td style="padding:3px 8px;color:#94a3b8;">🤖 Forecast Model</td><td style="padding:3px 8px;">{model_choice}</td></tr>
                  <tr><td style="padding:3px 8px;color:#94a3b8;">⏱️ Forecast Horizon</td><td style="padding:3px 8px;">{horizon_display} ahead</td></tr>
                  <tr><td style="padding:3px 8px;color:#94a3b8;">🏙️ City</td><td style="padding:3px 8px;">{city_display}</td></tr>
                  <tr><td style="padding:3px 8px;color:#94a3b8;">💡 Health Advisory</td><td style="padding:3px 8px;">{advisory}</td></tr>
                  <tr><td style="padding:3px 8px;color:#94a3b8;">📡 Data Source</td><td style="padding:3px 8px;font-size:0.82rem;">{data_source}</td></tr>
                  <tr><td style="padding:3px 8px;color:#94a3b8;">🕐 Timestamp</td><td style="padding:3px 8px;font-size:0.82rem;">{now_str}</td></tr>
                </table>
              </div>
            </div>
            </div>''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'''<div class="glass-card">
            <div style="color:#94a3b8;font-size:0.88rem;">
                Forecast value not available.<br>
                <b>Model:</b> {model_choice} &nbsp;|&nbsp;
                <b>Horizon:</b> {horizon_display} &nbsp;|&nbsp;
                <b>City:</b> {city_display}<br>
                <b>Data Source:</b> {data_source}<br>
                <b>Timestamp:</b> {now_str}
            </div>
            </div>''',
            unsafe_allow_html=True,
        )

    # ── Model Performance Card ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Forecasting Performance</div>', unsafe_allow_html=True)
    st.caption("ℹ️ These metrics reflect forecasting performance (future AQI prediction), not estimation performance. R² values are in the 0.0–0.55 range.")

    if metrics:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R² Score",  f"{metrics['r2']:.4f}")
        mc2.metric("MAE",       f"{metrics['mae']:.2f}")
        mc3.metric("RMSE",      f"{metrics['rmse']:.2f}")
    else:
        st.info("Forecasting performance metrics not available for this city/horizon/model combination.")

    # ── Historical context: AQI from prediction CSV ────────────────────────────
    st.markdown('<div class="section-header">📉 Recent AQI — Historical Context</div>', unsafe_allow_html=True)
    pdf = load_prediction_csv(city_key, hz)
    if pdf is not None and "actual_aqi" in pdf.columns:
        recent = pdf.tail(100).reset_index(drop=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=recent["actual_aqi"],
            name="Actual AQI",
            line=dict(color="#06b6d4", width=1.5),
        ))
        if "xgb_pred" in recent.columns:
            fig.add_trace(go.Scatter(
                y=recent["xgb_pred"],
                name="XGBoost Pred",
                line=dict(color="#3b82f6", width=1.5, dash="dot"),
            ))
        if "bilstm_pred" in recent.columns:
            fig.add_trace(go.Scatter(
                y=recent["bilstm_pred"],
                name="BiLSTM Pred",
                line=dict(color="#a78bfa", width=1.5, dash="dash"),
            ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=f"{city_display} — Last 100 Test-Set Observations (h={hz})", font=dict(color="#e2e8f0",size=13)),
            yaxis=dict(title="AQI", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(title="Sample Index"),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
            margin=dict(l=10,r=10,t=50,b=20),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Historical prediction CSV not available for this city/horizon.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def page_comparison():
    st.markdown('<div class="section-header" style="font-size:1.6rem;">📊 Model Comparison</div>', unsafe_allow_html=True)

    tab_est, tab_fcast = st.tabs(["🔬 AQI Estimation", "📈 AQI Forecasting"])

    # ── Estimation tab ─────────────────────────────────────────────────────────
    with tab_est:
        st.markdown('<div class="section-header">Estimation Performance (Current-Time Prediction)</div>', unsafe_allow_html=True)
        st.caption("These are estimation (same-time) R² values — they reflect how well models predict AQI when given current pollutant readings. Expect R² ≈ 0.85–0.99.")

        ta = load_track_a_csv()
        if ta.empty:
            st.warning("⚠️ final_track_a_complete.csv not found.")
        else:
            avg = ta.groupby("model")[["r2","mae","rmse"]].mean().reset_index()
            avg = avg.sort_values("r2", ascending=False).reset_index(drop=True)
            avg.insert(0, "Rank", range(1, len(avg)+1))
            avg.columns = ["Rank","Model","Avg R²","Avg MAE","Avg RMSE"]
            for c in ["Avg R²","Avg MAE","Avg RMSE"]:
                avg[c] = avg[c].round(4 if "R" in c else 2)
            st.dataframe(avg, use_container_width=True, hide_index=True)

            # Grouped bar chart
            fig_grp = go.Figure()
            models = avg["Model"].tolist()
            PALETTE = ["#3b82f6","#06b6d4","#ef4444"]
            for metric, pal in zip(["Avg R²","Avg MAE","Avg RMSE"], PALETTE):
                fig_grp.add_trace(go.Bar(
                    name=metric, x=models, y=avg[metric], marker_color=pal,
                ))
            fig_grp.update_layout(
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Estimation Models — R², MAE, RMSE", font=dict(color="#e2e8f0",size=14)),
                legend=dict(bgcolor="rgba(0,0,0,0.4)"),
                margin=dict(l=10,r=10,t=50,b=20),
                height=320,
            )
            st.plotly_chart(fig_grp, use_container_width=True)

            # Per-city heatmap
            if "city" in ta.columns:
                try:
                    pivot = ta.pivot_table(index="city", columns="model", values="r2", aggfunc="mean").fillna(0)
                    fig_hm = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=list(pivot.columns),
                        y=list(pivot.index),
                        colorscale="Blues", zmin=0, zmax=1,
                        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                        texttemplate="%{text}",
                        textfont=dict(size=9, color="white"),
                        hovertemplate="City: %{y}<br>Model: %{x}<br>R²: %{z:.4f}<extra></extra>",
                    ))
                    fig_hm.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        title=dict(text="Per-City R² — Estimation", font=dict(color="#e2e8f0",size=13)),
                        margin=dict(l=10,r=10,t=50,b=10),
                        height=max(350, len(pivot.index)*28+80),
                        xaxis=dict(side="top"),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
                except Exception as e:
                    st.info(f"Heatmap error: {e}")

    # ── Forecasting tab ─────────────────────────────────────────────────────────
    with tab_fcast:
        st.markdown('<div class="section-header">Forecasting Performance (Future AQI Prediction)</div>', unsafe_allow_html=True)
        st.caption("⚠️ Forecasting R² is inherently lower than estimation R² (typically 0.0–0.60) because these models predict future AQI without access to current pollutant readings.")

        tb = load_track_b_complete()
        if tb.empty:
            st.warning("⚠️ Track B forecasting results not found.")
        else:
            # Ensure numeric
            for col in ["r2","mae","rmse"]:
                if col in tb.columns:
                    tb[col] = pd.to_numeric(tb[col], errors="coerce")

            # Filter suspicious R² (never show ~0.99 here)
            tb = tb[tb["r2"] < 0.90] if "r2" in tb.columns else tb

            avg_b = tb.groupby("model")[["r2","mae","rmse"]].mean().reset_index()
            avg_b = avg_b.sort_values("r2", ascending=False).reset_index(drop=True)
            avg_b.insert(0, "Rank", range(1, len(avg_b)+1))
            avg_b.columns = ["Rank","Model","Avg R²","Avg MAE","Avg RMSE"]
            for c in ["Avg R²","Avg MAE","Avg RMSE"]:
                avg_b[c] = avg_b[c].round(4 if "R" in c else 2)
            st.dataframe(avg_b, use_container_width=True, hide_index=True)

            # Bar chart
            fig_b = go.Figure(go.Bar(
                y=avg_b["Model"], x=avg_b["Avg R²"], orientation="h",
                marker=dict(color=avg_b["Avg R²"],
                            colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#06b6d4"]],
                            showscale=False),
                text=[f"{v:.4f}" for v in avg_b["Avg R²"]],
                textposition="outside", textfont=dict(color="#e2e8f0", size=12),
            ))
            fig_b.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Average R² by Model (Forecasting)", font=dict(color="#e2e8f0",size=14)),
                xaxis=dict(range=[0, max(0.65, avg_b["Avg R²"].max()*1.15)],
                           title="R²", gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=10,r=70,t=50,b=20),
                height=300,
            )
            st.plotly_chart(fig_b, use_container_width=True)

            # Horizon degradation chart (R² vs horizon)
            if "horizon" in tb.columns:
                st.markdown('<div class="section-header">📉 R² Degradation by Forecast Horizon</div>', unsafe_allow_html=True)
                try:
                    hz_avg = tb.groupby(["model","horizon"])["r2"].mean().reset_index()
                    # normalize horizon labels
                    def norm_hz(h):
                        s = str(h).strip()
                        if s in ["1","01","1h","h01","h1"]:   return "1h"
                        if s in ["6","06","6h","h06","h6"]:   return "6h"
                        if s in ["24","24h","h24"]:           return "24h"
                        return s
                    hz_avg["horizon_label"] = hz_avg["horizon"].apply(norm_hz)
                    hz_order = ["1h","6h","24h"]
                    hz_avg["horizon_label"] = pd.Categorical(
                        hz_avg["horizon_label"], categories=hz_order, ordered=True
                    )
                    hz_avg = hz_avg.sort_values(["model","horizon_label"])

                    PALETTE_HZ = ["#3b82f6","#06b6d4","#a78bfa","#f59e0b","#ef4444","#22c55e"]
                    fig_hz = go.Figure()
                    models_hz = hz_avg["model"].unique()
                    for i, mdl in enumerate(models_hz):
                        sub = hz_avg[hz_avg["model"]==mdl]
                        fig_hz.add_trace(go.Scatter(
                            x=sub["horizon_label"].astype(str),
                            y=sub["r2"],
                            name=mdl,
                            mode="lines+markers",
                            line=dict(color=PALETTE_HZ[i % len(PALETTE_HZ)], width=2),
                            marker=dict(size=8),
                        ))
                    fig_hz.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        title=dict(text="R² vs Forecast Horizon per Model", font=dict(color="#e2e8f0",size=14)),
                        xaxis=dict(title="Horizon", categoryorder="array", categoryarray=hz_order),
                        yaxis=dict(title="R²", gridcolor="rgba(255,255,255,0.05)"),
                        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
                        margin=dict(l=10,r=10,t=50,b=20),
                        height=340,
                    )
                    st.plotly_chart(fig_hz, use_container_width=True)
                except Exception as e:
                    st.info(f"Horizon chart error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: INDIA AQI MAP
# ─────────────────────────────────────────────────────────────────────────────
def page_india_map():
    st.markdown('<div class="section-header" style="font-size:1.6rem;">🗺️ India AQI Map</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass-card" style="margin-bottom:0.6rem;">
        <span style="color:#94a3b8;font-size:0.88rem;">
            Interactive map of 18 study cities. Marker color reflects the best estimation R²
            achieved at that city. Hover to see city details, best model, and forecast availability.
        </span>
        </div>''',
        unsafe_allow_html=True,
    )

    ta = load_track_a_csv()

    # Build per-city stats
    cities_data = []
    for disp_name, city_key in CITY_MAP.items():
        lat, lon = CITY_COORDS.get(city_key, (20.0, 78.0))
        best_r2    = None
        best_model = "N/A"
        approx_rec = "—"

        if not ta.empty and "city" in ta.columns:
            city_rows = ta[ta["city"].str.lower() == city_key.lower()]
            if city_rows.empty:
                # fuzzy match
                for _, row in ta.iterrows():
                    if city_key.lower() in str(row.get("city","")).lower():
                        city_rows = ta.iloc[[row.name]]
                        break
            if not city_rows.empty:
                idx_max = city_rows["r2"].idxmax()
                best_r2    = round(float(city_rows.loc[idx_max, "r2"]), 4)
                best_model = str(city_rows.loc[idx_max, "model"])
                if "n_rows" in city_rows.columns:
                    nr = city_rows["n_rows"].dropna()
                    if len(nr) > 0:
                        approx_rec = f"{int(nr.iloc[0]):,}"

        # check pkl availability
        xgb_avail_any = any(
            xgb_pkl_exists(city_key, hz) for hz in ["01","06","24"]
        )
        xgb_str = "Yes" if xgb_avail_any else "No"

        cities_data.append(dict(
            display=disp_name,
            key=city_key,
            lat=lat, lon=lon,
            best_r2=best_r2 if best_r2 is not None else 0.0,
            best_model=best_model,
            xgb_available=xgb_str,
            approx_records=approx_rec,
        ))

    df_map = pd.DataFrame(cities_data)
    r2_vals = df_map["best_r2"].tolist()

    # Hover text
    hover_texts = []
    for _, row in df_map.iterrows():
        ht = (
            f"<b>{row['display']}</b><br>"
            f"Best Estimation R²: {row['best_r2']:.4f}<br>"
            f"Best Model: {row['best_model']}<br>"
            f"XGB Forecast Available: {row['xgb_available']}<br>"
            f"Approx. Records: {row['approx_records']}"
        )
        hover_texts.append(ht)

    fig = go.Figure(go.Scattergeo(
        lat=df_map["lat"].tolist(),
        lon=df_map["lon"].tolist(),
        text=hover_texts,
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=16,
            color=r2_vals,
            colorscale=[[0,"#1e3a5f"],[0.4,"#3b82f6"],[0.7,"#06b6d4"],[1,"#22d3ee"]],
            cmin=0, cmax=1,
            colorbar=dict(
                title="Best Est. R²",
                thickness=14,
                len=0.6,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#94a3b8"),
                titlefont=dict(color="#94a3b8"),
            ),
            line=dict(color="rgba(255,255,255,0.4)", width=1.5),
        ),
    ))

    fig.update_layout(
        geo=dict(
            scope="asia",
            showland=True,    landcolor="#1a2332",
            showocean=True,   oceancolor="#0a0e1a",
            showlakes=True,   lakecolor="#0a0e1a",
            showcountries=True, countrycolor="#334155",
            showrivers=False,
            center=dict(lat=22, lon=78),
            projection_scale=4.5,
            lataxis_range=[6, 38],
            lonaxis_range=[65, 100],
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
        font=dict(color="#94a3b8"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # City stats table
    st.markdown('<div class="section-header">📋 City Coverage Summary</div>', unsafe_allow_html=True)
    disp_df = df_map[["display","best_r2","best_model","xgb_available","approx_records"]].copy()
    disp_df.columns = ["City","Best Est. R²","Best Model","XGB Forecast","Approx. Records"]
    disp_df = disp_df.sort_values("Best Est. R²", ascending=False).reset_index(drop=True)
    st.dataframe(disp_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown('<div class="section-header" style="font-size:1.6rem;">📖 About This Project</div>', unsafe_allow_html=True)

    st.markdown(
        '''<div class="glass-card-accent">
        <div class="hero-title">Air Quality Index Prediction<br>Using Machine Learning and Deep Learning</div>
        <div class="hero-sub">Research Internship &nbsp;·&nbsp; Indian Institute of Information Technology Nagpur</div>
        </div>''',
        unsafe_allow_html=True,
    )

    # Link buttons
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.link_button(
            "🔗 GitHub Repository",
            "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning",
        )
    with c2:
        st.link_button(
            "📊 CPCB Multi-City Dataset",
            "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Research motivation
    st.markdown('<div class="section-header">🎯 Research Motivation</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass-card">
        <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.8;">
            Air pollution is one of the most critical environmental challenges in India.
            The Air Quality Index (AQI) is the standard metric used by the Central Pollution Control 
            Board (CPCB) to communicate air quality levels to the public. Accurate AQI prediction 
            enables early warnings, better urban planning, and informed public health decisions.<br><br>
            This research benchmarks seven ML/DL models across 18 Indian cities on two complementary tasks:
            <b style="color:#06b6d4;">current AQI estimation</b> (using same-time pollutant readings) 
            and <b style="color:#3b82f6;">multi-horizon AQI forecasting</b> (predicting future AQI 
            using only historical lag/rolling features — no data leakage).
        </div>
        </div>''',
        unsafe_allow_html=True,
    )

    # Dataset summary
    st.markdown('<div class="section-header">📊 Dataset Summary</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1rem;font-weight:700;color:#06b6d4;margin-bottom:0.5rem;">📂 CPCB Multi-City Dataset</div>
            <table style="width:100%;color:#cbd5e1;font-size:0.88rem;border-collapse:collapse;">
              <tr><td style="padding:3px 0;color:#94a3b8;">Source</td><td>Central Pollution Control Board (CPCB)</td></tr>
              <tr><td style="padding:3px 0;color:#94a3b8;">Cities</td><td>18 Indian cities</td></tr>
              <tr><td style="padding:3px 0;color:#94a3b8;">Period</td><td>2018 – 2023</td></tr>
              <tr><td style="padding:3px 0;color:#94a3b8;">Resolution</td><td>Hourly</td></tr>
              <tr><td style="padding:3px 0;color:#94a3b8;">Records</td><td>~18.7 million observations</td></tr>
              <tr><td style="padding:3px 0;color:#94a3b8;">Pollutants</td><td>PM2.5, PM10, NO, NO₂, NOₓ, NH₃, SO₂, CO, Ozone, Benzene, Toluene</td></tr>
              <tr><td style="padding:3px 0;color:#94a3b8;">Meteorology</td><td>Temperature, Humidity, Wind Speed/Direction, Solar Radiation, Pressure</td></tr>
            </table>
            </div>''',
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1rem;font-weight:700;color:#3b82f6;margin-bottom:0.5rem;">🏙️ Cities Studied</div>
            <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.8;">
              Ahmedabad · Bhopal · Chennai · Delhi NCR · Gandhinagar<br>
              Hyderabad · Indore · Jaipur · Jodhpur · Mumbai<br>
              Mumbai Suburbs · Nagpur · Pune · Singrauli · Surat<br>
              Thane · Vapi · Visakhapatnam
            </div>
            </div>''',
            unsafe_allow_html=True,
        )

    # Methodology
    st.markdown('<div class="section-header">🔬 Methodology Overview</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1rem;font-weight:700;color:#06b6d4;margin-bottom:0.5rem;">🎯 AQI Estimation</div>
            <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.7;">
              Predict <i>current</i> AQI from simultaneous pollutant and meteorological readings.<br><br>
              <b>Features:</b> Same-time pollutants + meteorological variables + time encodings<br>
              <b>Split:</b> Chronological 70/15/15 (train/val/test)<br>
              <b>Target:</b> Current-hour AQI<br>
              <b>Typical R²:</b> 0.85 – 0.99 (high, because input features are co-measured)
            </div>
            </div>''',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1rem;font-weight:700;color:#3b82f6;margin-bottom:0.5rem;">📈 AQI Forecasting</div>
            <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.7;">
              Predict <i>future</i> AQI at +1h, +6h, +24h using only historical features.<br><br>
              <b>Features:</b> Lag (1–24h) + rolling (3/6/12/24h) + met + time — NO same-time pollutants<br>
              <b>Split:</b> Chronological 70/15/15<br>
              <b>Target:</b> AQI shifted by horizon h<br>
              <b>Typical R²:</b> 0.0 – 0.55 (lower, inherent future uncertainty)
            </div>
            </div>''',
            unsafe_allow_html=True,
        )

    # Models
    st.markdown('<div class="section-header">🤖 Models Used</div>', unsafe_allow_html=True)
    ml1, ml2 = st.columns(2)
    with ml1:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1rem;font-weight:700;color:#06b6d4;margin-bottom:0.5rem;">📐 Classical ML Models</div>
            <ul style="color:#cbd5e1;font-size:0.88rem;line-height:1.9;margin:0;padding-left:1.2rem;">
              <li><b>Ridge Regression</b> — L2-regularized linear baseline</li>
              <li><b>Random Forest</b> — Ensemble of decision trees</li>
              <li><b>Gradient Boosting</b> — Sequential boosting (GBR)</li>
              <li><b>XGBoost</b> — Extreme Gradient Boosting</li>
            </ul>
            </div>''',
            unsafe_allow_html=True,
        )
    with ml2:
        st.markdown(
            '''<div class="glass-card">
            <div style="font-size:1rem;font-weight:700;color:#a78bfa;margin-bottom:0.5rem;">🧠 Deep Learning Models</div>
            <ul style="color:#cbd5e1;font-size:0.88rem;line-height:1.9;margin:0;padding-left:1.2rem;">
              <li><b>LSTM</b> — Long Short-Term Memory networks</li>
              <li><b>BiLSTM</b> — Bidirectional LSTM</li>
              <li><b>CNN-BiLSTM</b> — Convolutional + Bidirectional LSTM</li>
            </ul>
            </div>''',
            unsafe_allow_html=True,
        )

    # Key contributions
    st.markdown('<div class="section-header">🏆 Key Research Contributions</div>', unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass-card">
        <ul style="color:#cbd5e1;font-size:0.9rem;line-height:2;margin:0;padding-left:1.2rem;">
          <li>First systematic benchmarking of 7 ML/DL models across <b>18 Indian cities simultaneously</b> using CPCB data</li>
          <li>Strict <b>leakage-free dual-task protocol</b>: estimation (same-time) vs. forecasting (future-only features)</li>
          <li>Comprehensive <b>feature engineering</b>: 88–115 features per city including lag, rolling, interaction, and temporal encodings</li>
          <li>Identification of <b>Gradient Boosting as top estimator</b> (avg R² ≈ 0.99) and XGBoost as best forecaster</li>
          <li>Demonstrates that <b>deep learning models do not outperform tree ensembles</b> on tabular air quality data</li>
          <li>Full <b>reproducibility</b>: chronological splits, no data leakage, city-level evaluation</li>
        </ul>
        </div>''',
        unsafe_allow_html=True,
    )

    # Institutions
    st.markdown('<div class="section-header">🏛️ Research Internship</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    for col, name, desc in [
        (i1, "IIIT Nagpur",
         "Indian Institute of Information Technology Nagpur — Host institution for the research internship"),
        (i2, "YCCE",
         "Yeshwantrao Chavan College of Engineering — Affiliated institute"),
        (i3, "CPCB",
         "Central Pollution Control Board — Data provider for the multi-city air quality dataset"),
    ]:
        col.markdown(
            f'''<div class="glass-card" style="text-align:center;min-height:120px;">
            <div style="font-size:1.1rem;font-weight:700;color:#06b6d4;margin-bottom:0.4rem;">{name}</div>
            <div style="color:#94a3b8;font-size:0.83rem;line-height:1.5;">{desc}</div>
            </div>''',
            unsafe_allow_html=True,
        )

    # Author card
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '''<div class="glass-card" style="text-align:center;max-width:600px;margin:auto;">
        <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.3rem;">👤 Research Author</div>
        <div style="color:#94a3b8;font-size:0.88rem;line-height:1.7;">
            Research Internship in Machine Learning &amp; Deep Learning for Environmental Data Science<br>
            <span style="color:#06b6d4;">Indian Institute of Information Technology (IIIT) Nagpur</span><br>
            CPCB Multi-City AQI Prediction Study · 2018–2023
        </div>
        </div>''',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    page_home()
elif page == "🔬 AQI Estimation":
    page_estimation()
elif page == "📈 AQI Forecasting":
    page_forecasting()
elif page == "📊 Model Comparison":
    page_comparison()
elif page == "🗺️ India AQI Map":
    page_india_map()
elif page == "📖 About":
    page_about()
