"""
AQI Prediction Using Machine Learning and Deep Learning
=======================================================
Research Internship — IIIT Nagpur
CPCB Multi-City Dataset | 18 Cities | 7 Models | 3 Horizons
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be FIRST Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction Research Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%) !important;
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stRadio label { color: #e6edf3 !important; }
.kpi-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(100,180,255,0.2);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.kpi-number { font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
.kpi-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(120deg, #58a6ff, #3fb950);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { font-size: 1.1rem; color: #8b949e; margin-top: 8px; }
.research-card {
    background: rgba(22,27,34,0.8);
    border: 1px solid rgba(48,54,61,1);
    border-radius: 10px;
    padding: 16px;
}
.workflow-step {
    background: linear-gradient(135deg, rgba(88,166,255,0.1), rgba(63,185,80,0.1));
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.insight-card {
    background: rgba(63,185,80,0.1);
    border-left: 4px solid #3fb950;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}
@media (prefers-color-scheme: light) {
    .kpi-card { background: rgba(0,0,0,0.04); border-color: rgba(0,100,255,0.2); }
    .kpi-number { color: #0050b3; }
    .research-card { background: #f6f8fa; border-color: #d0d7de; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
CITIES = {
    "Ahmedabad": (23.03, 72.58), "bhopal": (23.25, 77.41),
    "Chennai": (13.08, 80.27), "Delhi_NCR": (28.70, 77.10),
    "GandhiNagar": (23.22, 72.68), "Hyderabad": (17.39, 78.49),
    "Indore": (22.72, 75.86), "Jaipur": (26.91, 75.79),
    "Jodhpur": (26.30, 73.02), "Mumbai": (19.08, 72.88),
    "Mumbai_suburbs": (19.20, 72.97), "Nagpur": (21.15, 79.09),
    "Pune": (18.52, 73.86), "Singrauli": (24.20, 82.67),
    "Surat": (21.17, 72.83), "Thane": (19.22, 72.98),
    "Vapi": (20.37, 72.91), "vishakhapattanam": (17.69, 83.22),
}

HORIZON_LABELS = {1: "1 Hour", 6: "6 Hours", 24: "24 Hours"}
HORIZON_INT    = {"1 Hour": 1, "6 Hours": 6, "24 Hours": 24}
HORIZON_ORDER  = ["1 Hour", "6 Hours", "24 Hours"]

MODEL_NORM_B = {
    "GradientBoosting": "Gradient Boosting",
    "RandomForest":     "Random Forest",
    "XGBoost":          "XGBoost",
    "LSTM":             "LSTM",
    "BiLSTM":           "BiLSTM",
    "CNN-BiLSTM":       "CNN-BiLSTM",
}
MODEL_NORM_A = {
    "GradBoost":   "Gradient Boosting",
    "RandomForest":"Random Forest",
    "Ridge":       "Ridge Regression",
    "XGBoost":     "XGBoost",
    "LSTM":        "LSTM",
    "BiLSTM":      "BiLSTM",
    "CNN-BiLSTM":  "CNN-BiLSTM",
}
ESTIM_MODELS   = ["Gradient Boosting","Random Forest","XGBoost","Ridge Regression","LSTM","BiLSTM","CNN-BiLSTM"]
FORECAST_MODELS= ["Gradient Boosting","Random Forest","XGBoost","LSTM","BiLSTM","CNN-BiLSTM"]
COLOR_SEQ = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657","#79c0ff","#56d364"]

CPCB_BP = {
    "PM2.5": [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)],
    "PM10":  [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
    "NO2":   [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
    "SO2":   [(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2100,401,500)],
    "CO":    [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
    "O3":    [(0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
    "NH3":   [(0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2400,401,500)],
}

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def find_csv(name: str) -> Path | None:
    candidates = [
        Path(f"./{name}"),
        Path(f"AQI_Prediction_Project_Final (1)/outputs/predictions/{name}"),
        Path(f"AQI_Prediction_Project_Final (1)/outputs/tables/{name}"),
    ]
    for base in [Path("."), Path(".."), Path("../.."), Path("../../.."), Path("../../../..")]:
        candidates.append(base / "outputs" / "tables" / name)
        candidates.append(base / "outputs" / "predictions" / name)
    for c in candidates:
        if c.exists() and c.stat().st_size > 50:
            return c
    return None


def safe_pivot(df, index, columns, values, aggfunc="mean"):
    try:
        if df is None or df.empty:
            return None
        if not all(c in df.columns for c in [index, columns, values]):
            return None
        df_clean = df.dropna(subset=[index, columns, values]).copy()
        if df_clean.empty:
            return None
        pivot = df_clean.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
        pivot = pivot.replace([np.inf, -np.inf], np.nan)
        return pivot
    except Exception:
        return None


def get_template():
    dark = st.session_state.get("dark_mode", True)
    return "plotly_dark" if dark else "plotly_white"


def chart_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLOR_SEQ,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def aqi_category(aqi: float):
    if aqi <= 50:   return "Good",       "#00b050", "Air quality is satisfactory. Outdoor activities are safe."
    if aqi <= 100:  return "Satisfactory","#92d050", "Acceptable air quality. Sensitive persons may experience minor discomfort."
    if aqi <= 200:  return "Moderate",   "#ffff00", "Moderate health concern. Sensitive groups should reduce outdoor activity."
    if aqi <= 300:  return "Poor",       "#ff7e00", "Health effects for everyone. Avoid prolonged outdoor exertion."
    if aqi <= 400:  return "Very Poor",  "#ff0000", "Health alert. Everyone may experience serious effects."
    return "Severe", "#7e0023", "Emergency conditions. Avoid all outdoor activity."


def calc_sub_index(Cp: float, breakpoints) -> float:
    if Cp <= 0:
        return 0.0
    for (BPlo, BPhi, Ilo, Ihi) in breakpoints:
        if BPlo <= Cp <= BPhi:
            return round(((Ihi - Ilo) / (BPhi - BPlo)) * (Cp - BPlo) + Ilo, 2)
    return 500.0


def compute_cpcb_aqi(pollutants: dict):
    sub_indices = {}
    for poll, val in pollutants.items():
        if val is not None and val > 0 and poll in CPCB_BP:
            si = calc_sub_index(float(val), CPCB_BP[poll])
            if si > 0:
                sub_indices[poll] = si
    if not sub_indices:
        return 0.0, "Good", "#00b050", "Air quality is satisfactory.", "N/A"
    aqi = max(sub_indices.values())
    dominant = max(sub_indices, key=sub_indices.get)
    cat, color, rec = aqi_category(aqi)
    return round(aqi, 1), cat, color, rec, dominant


def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_track_a():
    try:
        p = find_csv("final_track_a_complete.csv")
        if p is None:
            return pd.DataFrame()
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        # Normalize uppercase cols from LSTM file
        for old, new in [("r2","r2"),("mae","mae"),("rmse","rmse")]:
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        # Map model display names
        df["model_display"] = df["model"].map(MODEL_NORM_A).fillna(df["model"])
        # Ensure numeric
        for col in ["r2","mae","rmse","n_test","n_train","n_feats"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["r2"])
    except Exception as e:
        st.warning(f"Could not load Track A data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_track_b():
    try:
        dfs = []
        # Standard schema files
        standard_files = {
            "track_b_gbr.csv":      "GradientBoosting",
            "track_b_rf.csv":       "RandomForest",
            "track_b_xgb.csv":      "XGBoost",
            "track_b_lstm.csv":     "LSTM",
            "track_b_cnn_bilstm.csv":"CNN-BiLSTM",
        }
        for fname, expected_model in standard_files.items():
            p = find_csv(fname)
            if p is None:
                continue
            df = pd.read_csv(p)
            df.columns = [c.lower() for c in df.columns]
            for col in ["r2","mae","rmse","n_test","n_train"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "horizon" in df.columns:
                df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
            dfs.append(df)

        # BiLSTM from training CSV
        p_bilstm = find_csv("track_b_bilstm_training.csv")
        if p_bilstm is not None:
            df_b = pd.read_csv(p_bilstm)
            df_b.columns = [c.lower() for c in df_b.columns]
            # Map columns
            rename_map = {}
            if "r2_test" in df_b.columns:  rename_map["r2_test"]  = "r2"
            if "mae_test" in df_b.columns: rename_map["mae_test"] = "mae"
            if "rmse_test" in df_b.columns:rename_map["rmse_test"]= "rmse"
            df_b = df_b.rename(columns=rename_map)
            for col in ["r2","mae","rmse","n_test","n_train"]:
                if col in df_b.columns:
                    df_b[col] = pd.to_numeric(df_b[col], errors="coerce")
            if "horizon" in df_b.columns:
                df_b["horizon"] = pd.to_numeric(df_b["horizon"], errors="coerce").astype("Int64")
            dfs.append(df_b)

        if not dfs:
            return pd.DataFrame()

        tb = pd.concat(dfs, ignore_index=True)
        # Normalize model names
        if "model" in tb.columns:
            tb["model_display"] = tb["model"].map(MODEL_NORM_B).fillna(tb["model"])
        else:
            tb["model_display"] = "Unknown"
        # Horizon display label
        if "horizon" in tb.columns:
            tb["horizon_label"] = tb["horizon"].map(HORIZON_LABELS).fillna(tb["horizon"].astype(str))
        return tb.dropna(subset=["r2"])
    except Exception as e:
        st.warning(f"Could not load Track B data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_training_detail():
    try:
        dfs = []
        p_bilstm = find_csv("track_b_bilstm_training.csv")
        if p_bilstm is not None:
            df = pd.read_csv(p_bilstm)
            df.columns = [c.lower() for c in df.columns]
            rename_map = {}
            for old, new in [("r2_train","r2_train"),("r2_val","r2_val"),("r2_test","r2_test"),
                             ("mae_test","mae_test"),("rmse_test","rmse_test")]:
                pass  # already lowercase
            df["model_display"] = "BiLSTM"
            dfs.append(df)
        p_xgb = find_csv("track_b_xgb_training.csv")
        if p_xgb is not None:
            df2 = pd.read_csv(p_xgb)
            df2.columns = [c.lower() for c in df2.columns]
            df2["model_display"] = "XGBoost"
            dfs.append(df2)
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌫️ AQI Dashboard")
    st.markdown("Research Internship | IIIT Nagpur")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "🔬 AQI Estimation",
        "📈 Forecast Analysis",
        "📊 Research Analytics",
        "🗺 India Performance Map",
        "🏆 Model Comparison",
        "📖 About",
    ], key="nav")
    st.divider()
    dark_mode = st.toggle("🌙 Dark Mode", value=True, key="dark_mode")
    st.divider()
    st.markdown("**IIIT Nagpur**")
    st.markdown("CPCB Multi-City Dataset")
    st.markdown("18 Cities | 7 Models | 3 Horizons")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ──────────────────────────────────────────────────────────────────────────────
def page_home():
    try:
        st.markdown("""
        <div style='text-align:center; padding: 40px 20px 20px;'>
          <div class='hero-title'>AQI Prediction Using Machine Learning and Deep Learning</div>
          <div class='hero-subtitle'>Research Internship — IIIT Nagpur &nbsp;|&nbsp; CPCB Multi-City Dataset &nbsp;|&nbsp; 18 Indian Cities</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        for col, num, label in [
            (c1, "18", "Cities Studied"),
            (c2, "18.7M", "Records Processed"),
            (c3, "7", "ML/DL Models"),
            (c4, "3", "Forecast Horizons"),
        ]:
            col.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-number'>{num}</div>
              <div class='kpi-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Research Workflow
        st.markdown("### 🔄 Research Workflow")
        steps = ["📥 Dataset", "🔧 Preprocessing", "⚙️ Feature Engineering", "🤖 ML/DL Models", "📊 Evaluation"]
        cols = st.columns(len(steps))
        for col, step in zip(cols, steps):
            col.markdown(f"""
            <div class='workflow-step'>
              <div style='font-size:1.1rem; font-weight:600;'>{step}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Research Objectives
        st.markdown("### 🎯 Research Objectives")
        oc1, oc2 = st.columns(2)
        oc1.markdown("""
        <div class='research-card'>
          <div style='font-size:1.2rem; font-weight:700; color:#58a6ff;'>🔬 Current AQI Estimation</div>
          <div style='margin-top:8px; color:#c9d1d9;'>Estimate present AQI using 7 ML/DL models simultaneously. 
          All models trained on CPCB pollutant and meteorological features with strict leakage controls.</div>
          <div style='margin-top:12px;'>
            <span style='background:#3fb950;color:#000;padding:3px 8px;border-radius:12px;font-weight:600;font-size:0.9rem;'>
              Best: Gradient Boosting R²=0.9906
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        oc2.markdown("""
        <div class='research-card'>
          <div style='font-size:1.2rem; font-weight:700; color:#3fb950;'>📈 Multi-Horizon AQI Forecasting</div>
          <div style='margin-top:8px; color:#c9d1d9;'>Forecast AQI at 1-hour, 6-hour, and 24-hour horizons 
          using lag and rolling features. Evaluated on held-out chronological test sets.</div>
          <div style='margin-top:12px;'>
            <span style='background:#58a6ff;color:#000;padding:3px 8px;border-radius:12px;font-weight:600;font-size:0.9rem;'>
              Best: Gradient Boosting R²=0.4997
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Links
        st.markdown("### 🔗 Project Resources")
        lc1, lc2, lc3 = st.columns(3)
        lc1.link_button(
            "📂 GitHub Repository",
            "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning",
            use_container_width=True,
        )
        lc2.link_button(
            "🗃️ CPCB Dataset",
            "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link",
            use_container_width=True,
        )
        lc3.markdown(
            "<div style='text-align:center; padding:10px; color:#8b949e;'>📄 Internship Report — IIIT Nagpur 2024</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#8b949e; font-size:0.85rem;'>"
            "IIIT Nagpur &nbsp;|&nbsp; YCCE Nagpur &nbsp;|&nbsp; CPCB Dataset &nbsp;|&nbsp; "
            "18 Cities &nbsp;|&nbsp; 2018–2023 Hourly Data"
            "</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Home page error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: AQI ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────
def page_estimation():
    try:
        st.markdown("## 🔬 AQI Estimation")
        tab_calc, tab_perf = st.tabs(["🧮 CPCB AQI Calculator", "📊 Estimation Model Performance"])

        # ── Tab 1: CPCB Calculator ──────────────────────────────────────────
        with tab_calc:
            st.markdown("### 🧮 CPCB Air Quality Index Calculator")
            st.info("Enter pollutant concentrations to compute AQI using the official CPCB sub-index formula.")

            if "calc_result" not in st.session_state:
                st.session_state["calc_result"] = None

            col_a, col_b = st.columns(2)
            with col_a:
                pm25 = st.number_input("PM2.5 (µg/m³)",  min_value=0.0, max_value=500.0, value=0.0, step=1.0)
                pm10 = st.number_input("PM10 (µg/m³)",   min_value=0.0, max_value=600.0, value=0.0, step=1.0)
                no2  = st.number_input("NO2 (µg/m³)",    min_value=0.0, max_value=800.0, value=0.0, step=1.0)
                so2  = st.number_input("SO2 (µg/m³)",    min_value=0.0, max_value=2100.0, value=0.0, step=1.0)
            with col_b:
                co   = st.number_input("CO (mg/m³)",     min_value=0.0, max_value=50.0, value=0.0, step=0.1)
                o3   = st.number_input("O3 (µg/m³)",     min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
                nh3  = st.number_input("NH3 (µg/m³)",    min_value=0.0, max_value=2400.0, value=0.0, step=1.0)

            btn_c, btn_s, btn_r = st.columns(3)
            calc_clicked   = btn_c.button("🔍 Calculate AQI", type="primary", use_container_width=True)
            sample_clicked = btn_s.button("📋 Load Sample",   use_container_width=True)
            reset_clicked  = btn_r.button("🔄 Reset",         use_container_width=True)

            if sample_clicked:
                st.info("Sample loaded: PM2.5=85, PM10=150, NO2=60, SO2=20, CO=1.5, O3=100, NH3=200")
                st.rerun()

            if reset_clicked:
                st.session_state["calc_result"] = None
                st.rerun()

            pollutants = {
                "PM2.5": pm25, "PM10": pm10, "NO2": no2,
                "SO2": so2, "CO": co, "O3": o3, "NH3": nh3,
            }

            if calc_clicked or sample_clicked:
                if sample_clicked:
                    pollutants = {"PM2.5": 85.0, "PM10": 150.0, "NO2": 60.0,
                                  "SO2": 20.0, "CO": 1.5, "O3": 100.0, "NH3": 200.0}
                aqi_val, cat, color, rec, dom = compute_cpcb_aqi(pollutants)
                st.session_state["calc_result"] = {
                    "aqi": aqi_val, "cat": cat, "color": color, "rec": rec,
                    "dom": dom, "pollutants": pollutants,
                }

            if st.session_state.get("calc_result"):
                res = st.session_state["calc_result"]
                st.markdown("---")
                rc1, rc2, rc3 = st.columns(3)
                rc1.markdown(f"""
                <div class='kpi-card' style='border-color:{res["color"]};'>
                  <div class='kpi-number' style='color:{res["color"]};'>{res["aqi"]}</div>
                  <div class='kpi-label'>AQI Value</div>
                </div>
                """, unsafe_allow_html=True)
                rc2.markdown(f"""
                <div class='kpi-card'>
                  <div style='font-size:1.5rem; font-weight:700; color:{res["color"]};'>{res["cat"]}</div>
                  <div class='kpi-label'>Category</div>
                </div>
                """, unsafe_allow_html=True)
                rc3.markdown(f"""
                <div class='kpi-card'>
                  <div style='font-size:1.4rem; font-weight:700; color:#58a6ff;'>{res["dom"]}</div>
                  <div class='kpi-label'>Dominant Pollutant</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class='insight-card'>
                  <strong>Health Recommendation:</strong> {res["rec"]}
                </div>
                """, unsafe_allow_html=True)

                # Sub-index bar chart
                sub_data = {}
                for poll, val in res["pollutants"].items():
                    if val > 0 and poll in CPCB_BP:
                        si = calc_sub_index(float(val), CPCB_BP[poll])
                        if si > 0:
                            sub_data[poll] = si
                if sub_data:
                    polls = list(sub_data.keys())
                    svals = [sub_data[p] for p in polls]
                    bar_colors = []
                    for v in svals:
                        c_tuple = aqi_category(v)
                        bar_colors.append(c_tuple[1])
                    fig = go.Figure(go.Bar(
                        x=svals, y=polls, orientation="h",
                        marker_color=bar_colors,
                        text=[f"{v:.1f}" for v in svals],
                        textposition="auto",
                    ))
                    fig.update_layout(
                        template=get_template(),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        title="Sub-Index Values by Pollutant",
                        xaxis_title="Sub-Index", height=300,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ── Tab 2: Model Performance ────────────────────────────────────────
        with tab_perf:
            st.markdown("### 📊 Estimation Model Performance")
            ta = load_track_a()
            if ta.empty:
                st.warning("Estimation data not available.")
                return

            # Leaderboard
            rank_df = (ta.groupby("model_display")[["r2","mae","rmse"]]
                       .mean().reset_index()
                       .sort_values("r2", ascending=False)
                       .reset_index(drop=True))
            rank_df.insert(0, "Rank", [medal(i+1) for i in range(len(rank_df))])
            rank_df.columns = ["Rank","Model","Avg R²","Avg MAE","Avg RMSE"]
            rank_df["Avg R²"]  = rank_df["Avg R²"].round(4)
            rank_df["Avg MAE"] = rank_df["Avg MAE"].round(2)
            rank_df["Avg RMSE"]= rank_df["Avg RMSE"].round(2)
            st.markdown("#### 🏆 Model Leaderboard (sorted by R²)")
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

            # R² bar chart
            st.markdown("#### 📈 R² by Model")
            models_sorted = rank_df["Model"].tolist()
            r2_vals = rank_df["Avg R²"].tolist()
            fig_r2 = go.Figure(go.Bar(
                x=models_sorted, y=r2_vals,
                marker_color=COLOR_SEQ[:len(models_sorted)],
                text=[f"{v:.4f}" for v in r2_vals], textposition="auto",
            ))
            fig_r2.update_layout(
                template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", title="Average R² — All Cities",
                yaxis_title="R²", height=350, margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_r2, use_container_width=True)

            # City × Model heatmap
            st.markdown("#### 🗺 City × Model R² Heatmap")
            pivot = safe_pivot(ta, "city", "model_display", "r2", "mean")
            if pivot is not None:
                fig_hm = px.imshow(
                    pivot, text_auto=".3f", color_continuous_scale="RdYlGn",
                    aspect="auto", title="R² — City × Model",
                )
                fig_hm.update_layout(
                    template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                    height=550, margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_hm, use_container_width=True)

            # Download
            csv_bytes = rank_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Leaderboard CSV", csv_bytes, "estimation_leaderboard.csv", "text/csv")
    except Exception as e:
        st.error(f"Estimation page error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: FORECAST ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def page_forecast_analysis():
    try:
        st.markdown("## 📈 Forecast Analysis")
        st.info("Showing **verified test-set evaluation results** from held-out chronological test datasets.")

        tb = load_track_b()
        td = load_training_detail()

        if tb.empty:
            st.warning("Forecast data not available.")
            return

        available_cities  = sorted(tb["city"].dropna().unique().tolist())
        available_horizons= [h for h in HORIZON_ORDER if HORIZON_INT[h] in tb["horizon"].dropna().unique()]
        available_models  = [m for m in FORECAST_MODELS if m in tb["model_display"].dropna().unique()]

        fc1, fc2, fc3 = st.columns(3)
        sel_city    = fc1.selectbox("City",    available_cities or ["No data"],    key="fc_city")
        sel_horizon = fc2.selectbox("Horizon", available_horizons or ["No data"],  key="fc_horizon")
        sel_model   = fc3.selectbox("Model",   available_models or ["No data"],    key="fc_model")

        if not available_cities or not available_horizons or not available_models:
            st.warning("No data available.")
            return

        hz_int = HORIZON_INT.get(sel_horizon, 1)

        # Filter for selected combo
        mask = (
            (tb["city"] == sel_city) &
            (tb["horizon"] == hz_int) &
            (tb["model_display"] == sel_model)
        )
        df_sel = tb[mask]

        # KPI Cards
        st.markdown("---")
        st.markdown("#### 📋 Performance Metrics — Selected Combination")
        if df_sel.empty:
            st.info(f"No data for {sel_city} | {sel_horizon} | {sel_model}")
        else:
            row = df_sel.iloc[0]
            kc1, kc2, kc3, kc4 = st.columns(4)
            r2_val   = row.get("r2", float("nan"))
            mae_val  = row.get("mae", float("nan"))
            rmse_val = row.get("rmse", float("nan"))
            n_test   = row.get("n_test", float("nan"))
            kc1.markdown(f"<div class='kpi-card'><div class='kpi-number'>{r2_val:.4f}</div><div class='kpi-label'>R² (Test)</div></div>", unsafe_allow_html=True)
            kc2.markdown(f"<div class='kpi-card'><div class='kpi-number'>{mae_val:.2f}</div><div class='kpi-label'>MAE</div></div>", unsafe_allow_html=True)
            kc3.markdown(f"<div class='kpi-card'><div class='kpi-number'>{rmse_val:.2f}</div><div class='kpi-label'>RMSE</div></div>", unsafe_allow_html=True)
            kc4.markdown(f"<div class='kpi-card'><div class='kpi-number'>{int(n_test) if not pd.isna(n_test) else 'N/A'}</div><div class='kpi-label'>Test Samples</div></div>", unsafe_allow_html=True)

        # Train/Val/Test Progression (BiLSTM & XGBoost only)
        if sel_model in ("BiLSTM", "XGBoost") and not td.empty:
            td_mask = (
                (td["city"] == sel_city) &
                (td["model_display"] == sel_model)
            )
            if "horizon" in td.columns:
                td_mask = td_mask & (td["horizon"] == hz_int)
            td_sel = td[td_mask]
            if not td_sel.empty:
                row_td = td_sel.iloc[0]
                stages, vals = [], []
                for stage_col, label in [
                    ("r2_train", "Train R²"), ("r2_val", "Val R²"), ("r2_test", "Test R²")
                ]:
                    if stage_col in row_td.index and not pd.isna(row_td[stage_col]):
                        stages.append(label)
                        vals.append(float(row_td[stage_col]))
                if stages:
                    st.markdown("#### 📉 Train / Val / Test R² Progression")
                    fig_prog = go.Figure(go.Bar(
                        x=stages, y=vals, marker_color=["#58a6ff","#ffa657","#3fb950"],
                        text=[f"{v:.4f}" for v in vals], textposition="auto",
                    ))
                    fig_prog.update_layout(
                        template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        title=f"R² Progression — {sel_model} | {sel_city} | {sel_horizon}",
                        yaxis_title="R²", height=300,
                    )
                    st.plotly_chart(fig_prog, use_container_width=True)

        # Model Comparison at selected horizon & city
        st.markdown(f"#### 🤝 All Models — {sel_city} | {sel_horizon}")
        mask_hz = (tb["city"] == sel_city) & (tb["horizon"] == hz_int)
        df_hz = tb[mask_hz].copy()
        if not df_hz.empty:
            df_hz_grp = df_hz.groupby("model_display")[["r2","mae","rmse"]].mean().reset_index()
            df_hz_grp = df_hz_grp.sort_values("r2", ascending=False)
            fig_cmp = go.Figure(go.Bar(
                x=df_hz_grp["model_display"], y=df_hz_grp["r2"],
                marker_color=COLOR_SEQ[:len(df_hz_grp)],
                text=df_hz_grp["r2"].round(4), textposition="auto",
            ))
            fig_cmp.update_layout(
                template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=f"R² by Model — {sel_city} at {sel_horizon}",
                yaxis_title="R²", height=350,
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Cross-Horizon Performance
        st.markdown(f"#### 📉 Horizon Degradation — {sel_model} | {sel_city}")
        mask_mdl = (tb["city"] == sel_city) & (tb["model_display"] == sel_model)
        df_mdl = tb[mask_mdl].copy()
        if not df_mdl.empty:
            df_mdl_grp = df_mdl.groupby("horizon")[["r2","mae","rmse"]].mean().reset_index()
            df_mdl_grp["horizon_label"] = df_mdl_grp["horizon"].map(HORIZON_LABELS)
            df_mdl_grp = df_mdl_grp.sort_values("horizon")
            fig_deg = go.Figure()
            fig_deg.add_trace(go.Scatter(
                x=df_mdl_grp["horizon_label"], y=df_mdl_grp["r2"],
                mode="lines+markers+text",
                text=df_mdl_grp["r2"].round(4), textposition="top center",
                line=dict(color="#58a6ff", width=2),
                marker=dict(size=10),
            ))
            fig_deg.update_layout(
                template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title=f"R² Across Horizons — {sel_model} | {sel_city}",
                yaxis_title="R²", height=300,
            )
            st.plotly_chart(fig_deg, use_container_width=True)

        # Metrics table for selected model
        st.markdown(f"#### 📋 All Cities × Horizons — {sel_model}")
        mask_full = tb["model_display"] == sel_model
        df_tbl = tb[mask_full][["city","horizon","r2","mae","rmse","n_test"]].copy()
        df_tbl["horizon_label"] = df_tbl["horizon"].map(HORIZON_LABELS)
        df_tbl = df_tbl.drop(columns=["horizon"]).sort_values("r2", ascending=False).reset_index(drop=True)
        df_tbl.columns = ["City","R²","MAE","RMSE","N Test","Horizon"]
        for col in ["R²","MAE","RMSE"]:
            df_tbl[col] = pd.to_numeric(df_tbl[col], errors="coerce").round(4)
        st.dataframe(df_tbl, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = df_tbl.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes, f"forecast_{sel_model}.csv", "text/csv")
    except Exception as e:
        st.error(f"Forecast analysis error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4: RESEARCH ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
def page_research_analytics():
    try:
        st.markdown("## 📊 Research Analytics")
        ta = load_track_a()
        tb = load_track_b()

        tab1, tab2, tab3 = st.tabs(["🔬 Estimation Analytics", "📈 Forecasting Analytics", "💡 Insights"])

        # ── Tab 1 ───────────────────────────────────────────────────────────
        with tab1:
            if ta.empty:
                st.warning("No estimation data.")
            else:
                best_model  = ta.loc[ta["r2"].idxmax(), "model_display"]
                best_city   = ta.loc[ta["r2"].idxmax(), "city"]
                avg_r2      = ta["r2"].mean()
                avg_mae     = ta["mae"].mean()
                st.markdown("#### Overview")
                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.markdown(f"<div class='kpi-card'><div class='kpi-number' style='font-size:1.3rem;'>{best_model}</div><div class='kpi-label'>Best Model</div></div>", unsafe_allow_html=True)
                oc2.markdown(f"<div class='kpi-card'><div class='kpi-number' style='font-size:1.3rem;'>{best_city}</div><div class='kpi-label'>Best City</div></div>", unsafe_allow_html=True)
                oc3.markdown(f"<div class='kpi-card'><div class='kpi-number'>{avg_r2:.4f}</div><div class='kpi-label'>Avg R²</div></div>", unsafe_allow_html=True)
                oc4.markdown(f"<div class='kpi-card'><div class='kpi-number'>{avg_mae:.2f}</div><div class='kpi-label'>Avg MAE</div></div>", unsafe_allow_html=True)

                # Heatmap
                st.markdown("#### City × Model R² Heatmap")
                piv = safe_pivot(ta, "city", "model_display", "r2", "mean")
                if piv is not None:
                    fig_h = px.imshow(piv, text_auto=".3f", color_continuous_scale="RdYlGn", aspect="auto")
                    fig_h.update_layout(template=get_template(), paper_bgcolor="rgba(0,0,0,0)", height=550)
                    st.plotly_chart(fig_h, use_container_width=True)

                # Model ranking bar with error bars
                st.markdown("#### Model Ranking")
                grp = ta.groupby("model_display")["r2"].agg(["mean","std"]).reset_index()
                grp.columns = ["Model","Mean R²","Std R²"]
                grp = grp.sort_values("Mean R²", ascending=False)
                fig_rank = go.Figure(go.Bar(
                    x=grp["Model"], y=grp["Mean R²"],
                    error_y=dict(type="data", array=grp["Std R²"].fillna(0).tolist()),
                    marker_color=COLOR_SEQ[:len(grp)],
                    text=grp["Mean R²"].round(4), textposition="auto",
                ))
                fig_rank.update_layout(
                    template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", title="Mean R² ± Std by Model",
                    yaxis_title="R²", height=350,
                )
                st.plotly_chart(fig_rank, use_container_width=True)

                # Box plot
                st.markdown("#### Performance Distribution")
                fig_box = px.box(
                    ta, x="model_display", y="r2", color="model_display",
                    color_discrete_sequence=COLOR_SEQ, title="R² Distribution by Model",
                    labels={"model_display": "Model", "r2": "R²"},
                )
                fig_box.update_layout(
                    template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350,
                )
                st.plotly_chart(fig_box, use_container_width=True)

                # Top 5 city-model
                st.markdown("#### Top 5 City-Model Combinations")
                top5 = ta.nlargest(5, "r2")[["city","model_display","r2","mae","rmse"]].reset_index(drop=True)
                top5.columns = ["City","Model","R²","MAE","RMSE"]
                top5["R²"]  = top5["R²"].round(4)
                top5["MAE"] = top5["MAE"].round(2)
                top5["RMSE"]= top5["RMSE"].round(2)
                st.dataframe(top5, use_container_width=True, hide_index=True)

        # ── Tab 2 ───────────────────────────────────────────────────────────
        with tab2:
            if tb.empty:
                st.warning("No forecasting data.")
            else:
                best_fmdl  = tb.loc[tb["r2"].idxmax(), "model_display"]
                best_fcity = tb.loc[tb["r2"].idxmax(), "city"]
                avg_fr2    = tb["r2"].mean()
                best_hz_grp= tb.groupby("horizon_label")["r2"].mean()
                best_hz    = best_hz_grp.idxmax() if not best_hz_grp.empty else "N/A"

                st.markdown("#### Overview")
                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.markdown(f"<div class='kpi-card'><div class='kpi-number' style='font-size:1.3rem;'>{best_fmdl}</div><div class='kpi-label'>Best Forecasting Model</div></div>", unsafe_allow_html=True)
                oc2.markdown(f"<div class='kpi-card'><div class='kpi-number' style='font-size:1.3rem;'>{best_fcity}</div><div class='kpi-label'>Best City</div></div>", unsafe_allow_html=True)
                oc3.markdown(f"<div class='kpi-card'><div class='kpi-number'>{avg_fr2:.4f}</div><div class='kpi-label'>Avg R²</div></div>", unsafe_allow_html=True)
                oc4.markdown(f"<div class='kpi-card'><div class='kpi-number'>{best_hz}</div><div class='kpi-label'>Best Horizon</div></div>", unsafe_allow_html=True)

                # Model × Horizon heatmap
                st.markdown("#### Model × Horizon R² Heatmap")
                piv_mh = safe_pivot(tb, "model_display", "horizon_label", "r2", "mean")
                if piv_mh is not None:
                    # Reorder columns if possible
                    col_order = [h for h in HORIZON_ORDER if h in piv_mh.columns]
                    piv_mh = piv_mh[col_order] if col_order else piv_mh
                    fig_mh = px.imshow(piv_mh, text_auto=".3f", color_continuous_scale="RdYlGn", aspect="auto")
                    fig_mh.update_layout(template=get_template(), paper_bgcolor="rgba(0,0,0,0)", height=400)
                    st.plotly_chart(fig_mh, use_container_width=True)

                # City × Model heatmap for forecasting
                st.markdown("#### City × Model R² Heatmap")
                piv_cm = safe_pivot(tb, "city", "model_display", "r2", "mean")
                if piv_cm is not None:
                    fig_cm = px.imshow(piv_cm, text_auto=".3f", color_continuous_scale="RdYlGn", aspect="auto")
                    fig_cm.update_layout(template=get_template(), paper_bgcolor="rgba(0,0,0,0)", height=550)
                    st.plotly_chart(fig_cm, use_container_width=True)

                # Horizon degradation line chart
                st.markdown("#### Horizon Degradation by Model")
                hz_deg = tb.groupby(["model_display","horizon","horizon_label"])["r2"].mean().reset_index()
                hz_deg = hz_deg.sort_values("horizon")
                fig_deg = px.line(
                    hz_deg, x="horizon_label", y="r2", color="model_display",
                    color_discrete_sequence=COLOR_SEQ, markers=True,
                    title="Avg R² Across Horizons per Model",
                    labels={"horizon_label":"Horizon","r2":"Avg R²","model_display":"Model"},
                )
                fig_deg.update_layout(
                    template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", height=400,
                )
                st.plotly_chart(fig_deg, use_container_width=True)

                # Box plot
                st.markdown("#### Distribution by Model")
                fig_box2 = px.box(
                    tb, x="model_display", y="r2", color="model_display",
                    color_discrete_sequence=COLOR_SEQ, title="R² Distribution — Forecasting Models",
                    labels={"model_display":"Model","r2":"R²"},
                )
                fig_box2.update_layout(
                    template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350,
                )
                st.plotly_chart(fig_box2, use_container_width=True)

        # ── Tab 3 ───────────────────────────────────────────────────────────
        with tab3:
            st.markdown("#### 💡 Research Insights")
            insights = []

            if not ta.empty:
                ta_rank = ta.groupby("model_display")["r2"].mean().sort_values(ascending=False)
                best_a  = ta_rank.index[0]
                best_a_r2 = ta_rank.iloc[0]
                best_city_a = ta.groupby("city")["r2"].mean().idxmax()
                insights.append(f"🏆 Best Estimation Model: **{best_a}** (avg R²={best_a_r2:.4f})")
                insights.append(f"🌆 Top Performing City (Estimation): **{best_city_a}**")

            if not tb.empty:
                tb_rank = tb.groupby("model_display")["r2"].mean().sort_values(ascending=False)
                best_b  = tb_rank.index[0]
                best_b_r2 = tb_rank.iloc[0]
                best_city_b = tb.groupby("city")["r2"].mean().idxmax()
                hz_avg = tb.groupby("horizon")["r2"].mean().sort_values(ascending=False)
                insights.append(f"📈 Best Forecasting Model: **{best_b}** (avg R²={best_b_r2:.4f})")
                insights.append(f"🌆 Top Performing City (Forecasting): **{best_city_b}**")
                if len(hz_avg) >= 2:
                    hz_sorted = sorted(hz_avg.index.tolist())
                    if len(hz_sorted) >= 2:
                        degradation = hz_avg[hz_sorted[0]] - hz_avg[hz_sorted[-1]]
                        insights.append(f"📉 R² degrades by **{degradation:.4f}** from {HORIZON_LABELS.get(hz_sorted[0],'1h')} → {HORIZON_LABELS.get(hz_sorted[-1],'24h')} horizon")

            insights.append("✅ Strict chronological 70/15/15 split prevents data leakage")
            insights.append("📊 Leakage audit confirms estimation vs. forecasting feature separation")
            insights.append("🏙️ Study covers 18 Indian cities from CPCB hourly monitoring data")

            for ins in insights:
                st.markdown(f"""
                <div class='insight-card'>
                  {ins}
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Research analytics error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 5: INDIA PERFORMANCE MAP
# ──────────────────────────────────────────────────────────────────────────────
def page_india_map():
    try:
        st.markdown("## 🗺 India Performance Map")
        ta = load_track_a()
        tb = load_track_b()

        task = st.radio("Task", ["🔬 Estimation", "📈 Forecasting"], horizontal=True, key="map_task")

        if "Estimation" in task:
            if ta.empty:
                st.warning("No estimation data.")
                return
            models_avail = sorted(ta["model_display"].dropna().unique().tolist())
        else:
            if tb.empty:
                st.warning("No forecasting data.")
                return
            models_avail = sorted(tb["model_display"].dropna().unique().tolist())

        sel_model = st.selectbox("Select Model", models_avail, key="map_model")

        if "Estimation" in task:
            data = ta[ta["model_display"] == sel_model].copy()
            grp  = data.groupby("city").agg(r2=("r2","mean"), mae=("mae","mean"),
                                             rmse=("rmse","mean"), n_test=("n_test","sum")).reset_index()
        else:
            data = tb[tb["model_display"] == sel_model].copy()
            grp  = data.groupby("city").agg(r2=("r2","mean"), mae=("mae","mean"),
                                             rmse=("rmse","mean"), n_test=("n_test","sum")).reset_index()

        if grp.empty:
            st.info(f"No data available for {sel_model}.")
            return

        # Merge with city coordinates
        city_rows = []
        for _, row in grp.iterrows():
            city_name = row["city"]
            # Try case-insensitive match
            coords = CITIES.get(city_name)
            if coords is None:
                for k, v in CITIES.items():
                    if k.lower() == city_name.lower():
                        coords = v
                        break
            if coords:
                city_rows.append({
                    "city": city_name, "lat": coords[0], "lon": coords[1],
                    "r2": row["r2"], "mae": row["mae"],
                    "rmse": row["rmse"], "n_test": row["n_test"],
                })

        if not city_rows:
            st.warning("Could not match city names to coordinates.")
            return

        map_df = pd.DataFrame(city_rows)
        map_df["rank"] = map_df["r2"].rank(ascending=False).astype(int)
        map_df["size"] = np.clip(map_df["n_test"] / map_df["n_test"].max() * 25 + 8, 8, 33)

        fig_map = go.Figure(go.Scattergeo(
            lat=map_df["lat"], lon=map_df["lon"],
            mode="markers",
            marker=dict(
                size=map_df["size"],
                color=map_df["r2"],
                colorscale="RdYlGn",
                colorbar=dict(title="R²"),
                showscale=True,
                cmin=map_df["r2"].min(),
                cmax=map_df["r2"].max(),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Model: " + sel_model + "<br>"
                "R²: %{customdata[1]:.4f}<br>"
                "MAE: %{customdata[2]:.2f}<br>"
                "RMSE: %{customdata[3]:.2f}<br>"
                "Test Samples: %{customdata[4]:,}<br>"
                "Rank: %{customdata[5]}<extra></extra>"
            ),
            customdata=map_df[["city","r2","mae","rmse","n_test","rank"]].values,
        ))
        fig_map.update_layout(
            geo=dict(
                scope="asia",
                center=dict(lat=21, lon=80),
                projection_scale=4,
                showland=True, landcolor="rgb(30,30,40)",
                showocean=True, oceancolor="rgb(10,10,30)",
                showcoastlines=True, coastlinecolor="gray",
                showframe=False,
            ),
            template=get_template(),
            paper_bgcolor="rgba(0,0,0,0)",
            title=f"R² Performance Map — {sel_model} ({task.split()[-1]})",
            height=580,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Table below map
        st.markdown(f"#### 📋 City Rankings — {sel_model}")
        disp_df = map_df.sort_values("r2", ascending=False).reset_index(drop=True)
        disp_df.insert(0, "Rank", [medal(i+1) for i in range(len(disp_df))])
        disp_df = disp_df[["Rank","city","r2","mae","rmse","n_test"]].copy()
        disp_df.columns = ["Rank","City","R²","MAE","RMSE","Test Samples"]
        for col in ["R²","MAE","RMSE"]:
            disp_df[col] = disp_df[col].round(4)
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"India map error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 6: MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def page_model_comparison():
    try:
        st.markdown("## 🏆 Model Comparison")
        ta = load_track_a()
        tb = load_track_b()

        tab_e, tab_f = st.tabs(["🔬 Estimation Models", "📈 Forecasting Models"])

        # ── Estimation Tab ──────────────────────────────────────────────────
        with tab_e:
            if ta.empty:
                st.warning("No estimation data.")
            else:
                metric_e = st.selectbox("Metric", ["r2","mae","rmse"], key="cmp_e_metric",
                                        format_func=lambda x: x.upper())
                ascending_e = metric_e != "r2"

                grp_e = ta.groupby("model_display")[["r2","mae","rmse"]].mean().reset_index()
                grp_e = grp_e.sort_values(metric_e, ascending=ascending_e).reset_index(drop=True)
                grp_e.insert(0, "Rank", [medal(i+1) for i in range(len(grp_e))])
                grp_e.columns = ["Rank","Model","Avg R²","Avg MAE","Avg RMSE"]
                for col in ["Avg R²","Avg MAE","Avg RMSE"]:
                    grp_e[col] = grp_e[col].round(4)
                st.markdown("#### 🏆 Leaderboard")
                st.dataframe(grp_e, use_container_width=True, hide_index=True)

                # Bar chart
                fig_be = go.Figure(go.Bar(
                    x=grp_e["Model"], y=grp_e[f"Avg {metric_e.upper()}"],
                    marker_color=COLOR_SEQ[:len(grp_e)],
                    text=grp_e[f"Avg {metric_e.upper()}"], textposition="auto",
                ))
                fig_be.update_layout(
                    template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title=f"Avg {metric_e.upper()} by Model (Estimation)", height=350,
                )
                st.plotly_chart(fig_be, use_container_width=True)

                # Box plot
                fig_bxe = px.box(
                    ta, x="model_display", y=metric_e, color="model_display",
                    color_discrete_sequence=COLOR_SEQ,
                    title=f"{metric_e.upper()} Distribution by Model",
                    labels={"model_display":"Model", metric_e: metric_e.upper()},
                )
                fig_bxe.update_layout(
                    template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350,
                )
                st.plotly_chart(fig_bxe, use_container_width=True)

                # Per-city grouped bar (top 6 cities)
                top_cities_e = ta.groupby("city")["r2"].mean().nlargest(6).index.tolist()
                ta_top = ta[ta["city"].isin(top_cities_e)]
                if not ta_top.empty:
                    fig_grp = px.bar(
                        ta_top, x="city", y=metric_e, color="model_display",
                        color_discrete_sequence=COLOR_SEQ, barmode="group",
                        title=f"{metric_e.upper()} — Top 6 Cities (All Estimation Models)",
                        labels={"city":"City", metric_e: metric_e.upper(), "model_display":"Model"},
                    )
                    fig_grp.update_layout(
                        template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", height=420,
                    )
                    st.plotly_chart(fig_grp, use_container_width=True)

                # Insight card
                best_name = grp_e.iloc[0]["Model"]
                best_val  = grp_e.iloc[0][f"Avg {metric_e.upper()}"]
                worst_val = grp_e.iloc[-1][f"Avg {metric_e.upper()}"]
                gap       = abs(best_val - worst_val)
                st.markdown(f"""
                <div class='insight-card'>
                  🏆 Best Estimation Model by {metric_e.upper()}: <strong>{best_name}</strong>
                  &nbsp;|&nbsp; Performance gap vs worst model: <strong>{gap:.4f}</strong>
                </div>
                """, unsafe_allow_html=True)

                csv_e = ta.to_csv(index=False).encode()
                st.download_button("⬇️ Download Estimation Data", csv_e, "estimation_all.csv", "text/csv")

        # ── Forecasting Tab ─────────────────────────────────────────────────
        with tab_f:
            if tb.empty:
                st.warning("No forecasting data.")
            else:
                fc1, fc2 = st.columns(2)
                metric_f = fc1.selectbox("Metric", ["r2","mae","rmse"], key="cmp_f_metric",
                                         format_func=lambda x: x.upper())
                hz_filter = fc2.selectbox("Horizon", ["All"] + HORIZON_ORDER, key="cmp_f_hz")

                ascending_f = metric_f != "r2"
                tb_f = tb.copy()
                if hz_filter != "All":
                    hz_int_f = HORIZON_INT[hz_filter]
                    tb_f = tb_f[tb_f["horizon"] == hz_int_f]

                if tb_f.empty:
                    st.info("No data for selected horizon.")
                else:
                    grp_f = tb_f.groupby("model_display")[["r2","mae","rmse"]].mean().reset_index()
                    grp_f = grp_f.sort_values(metric_f, ascending=ascending_f).reset_index(drop=True)
                    grp_f.insert(0, "Rank", [medal(i+1) for i in range(len(grp_f))])
                    grp_f.columns = ["Rank","Model","Avg R²","Avg MAE","Avg RMSE"]
                    for col in ["Avg R²","Avg MAE","Avg RMSE"]:
                        grp_f[col] = grp_f[col].round(4)
                    st.markdown("#### 🏆 Leaderboard")
                    st.dataframe(grp_f, use_container_width=True, hide_index=True)

                    # Bar chart
                    fig_bf = go.Figure(go.Bar(
                        x=grp_f["Model"], y=grp_f[f"Avg {metric_f.upper()}"],
                        marker_color=COLOR_SEQ[:len(grp_f)],
                        text=grp_f[f"Avg {metric_f.upper()}"], textposition="auto",
                    ))
                    fig_bf.update_layout(
                        template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        title=f"Avg {metric_f.upper()} by Model (Forecasting — {hz_filter})", height=350,
                    )
                    st.plotly_chart(fig_bf, use_container_width=True)

                    # Box plot
                    fig_bxf = px.box(
                        tb_f, x="model_display", y=metric_f, color="model_display",
                        color_discrete_sequence=COLOR_SEQ,
                        title=f"{metric_f.upper()} Distribution by Model",
                        labels={"model_display":"Model", metric_f: metric_f.upper()},
                    )
                    fig_bxf.update_layout(
                        template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350,
                    )
                    st.plotly_chart(fig_bxf, use_container_width=True)

                    # Per-city grouped bar
                    top_cities_f = tb_f.groupby("city")["r2"].mean().nlargest(6).index.tolist()
                    tb_top = tb_f[tb_f["city"].isin(top_cities_f)]
                    if not tb_top.empty:
                        fig_grpf = px.bar(
                            tb_top, x="city", y=metric_f, color="model_display",
                            color_discrete_sequence=COLOR_SEQ, barmode="group",
                            title=f"{metric_f.upper()} — Top 6 Cities (All Forecasting Models — {hz_filter})",
                            labels={"city":"City", metric_f: metric_f.upper(), "model_display":"Model"},
                        )
                        fig_grpf.update_layout(
                            template=get_template(), paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", height=420,
                        )
                        st.plotly_chart(fig_grpf, use_container_width=True)

                    best_fn = grp_f.iloc[0]["Model"]
                    best_fv = grp_f.iloc[0][f"Avg {metric_f.upper()}"]
                    st.markdown(f"""
                    <div class='insight-card'>
                      🏆 Best Forecasting Model by {metric_f.upper()} ({hz_filter}): <strong>{best_fn}</strong>
                      — avg {metric_f.upper()} = <strong>{best_fv:.4f}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    csv_f = tb.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Forecasting Data", csv_f, "forecasting_all.csv", "text/csv")
    except Exception as e:
        st.error(f"Model comparison error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 7: ABOUT
# ──────────────────────────────────────────────────────────────────────────────
def page_about():
    try:
        st.markdown("## 📖 About This Research")
        tab_ov, tab_ds, tab_meth, tab_mdl, tab_team = st.tabs([
            "📋 Overview", "🗃️ Dataset", "⚙️ Methodology", "🤖 Models", "👥 Team"
        ])

        with tab_ov:
            st.markdown("""
### Project Overview

Air pollution is one of the most critical public health challenges in India. The **Central Pollution Control Board (CPCB)** 
operates a nationwide network of monitoring stations, collecting hourly pollutant concentrations across major cities.

#### Problem Statement
AQI prediction is both a **regression problem** (predict a continuous AQI value) and a task that bridges 
*estimation* (what is the current AQI given sensor readings?) and *forecasting* (what will the AQI be in 1/6/24 hours?).

#### Research Gap
No prior comprehensive multi-city, multi-model, leakage-audited study exists on CPCB hourly data at this scale.

#### Objectives
1. **Current AQI Estimation** — Evaluate 7 ML/DL models for simultaneous estimation across 18 cities
2. **Multi-Horizon Forecasting** — Forecast AQI at 1h, 6h, 24h horizons with lag/rolling features

#### Key Contributions
- 18-city simultaneous ML/DL evaluation (largest CPCB study to date)
- Dual-track leakage-audited protocol: estimation vs. forecasting feature sets
- Gradient Boosting achieves R²=0.9906 for estimation
- Documented horizon degradation: R² drops from 1h → 24h across all models
- Open-source: GitHub & CPCB dataset links provided
""")

        with tab_ds:
            st.markdown("""
### Dataset Information

| Property | Value |
|---|---|
| **Source** | Central Pollution Control Board (CPCB), India |
| **Cities** | 18 major Indian cities |
| **Records** | ~18.7 million hourly observations |
| **Period** | 2018–2023 |
| **Resolution** | Hourly |
| **Pollutant Features** | PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, Toluene |
| **Meteorological Features** | Temperature, Humidity, Wind Speed, Wind Direction, Solar Radiation, Pressure |
| **Split** | 70% Train / 15% Val / 15% Test (chronological) |

#### Feature Engineering
- Lag features (1h, 2h, 3h, 6h, 12h, 24h)
- Rolling statistics (mean, std, min, max)
- Temporal encodings (hour_sin/cos, month_sin/cos)
- Total: 88–115 features per model

#### Leakage Audit
Strict separation between estimation features (same-time pollutants) and forecasting features 
(only past-time lags and rolling statistics).
""")
            c1, c2 = st.columns(2)
            c1.link_button("📂 GitHub Repository",
                "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning",
                use_container_width=True)
            c2.link_button("🗃️ CPCB Dataset",
                "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link",
                use_container_width=True)

        with tab_meth:
            st.markdown("### 8-Step Research Pipeline")
            steps = [
                ("1. Data Collection",        "CPCB raw data download, 18 city isolation, format standardization"),
                ("2. Preprocessing",          "Resampling to hourly, 3-stage imputation, outlier detection and removal"),
                ("3. Feature Engineering",    "Lag features, rolling windows (mean/std/min/max), temporal sin/cos encodings — 88–115 features"),
                ("4. Leakage Audit",          "Strict separation of estimation vs. forecasting features; chronological splits enforced"),
                ("5. Model Training",         "Chronological train/val/test splits, cross-validated hyperparameter tuning per city"),
                ("6. Evaluation",             "R², MAE, RMSE, MAPE computed on held-out test sets for all 18 cities"),
                ("7. Statistical Testing",    "Effect size analysis, Wilcoxon signed-rank tests for model comparisons"),
                ("8. Visualization & Report", "Streamlit research dashboard, internship technical report, GitHub publication"),
            ]
            for title, desc in steps:
                st.markdown(f"""
                <div class='research-card' style='margin-bottom:10px;'>
                  <strong style='color:#58a6ff;'>{title}</strong>
                  <div style='margin-top:4px; color:#c9d1d9;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab_mdl:
            st.markdown("### Model Architecture")
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("#### 📐 Classical ML Models")
                ml_models = [
                    ("Ridge Regression", "alpha=1.0", "Baseline linear model; fast, interpretable, good for sparse features"),
                    ("Random Forest",    "500 trees, max_depth=None, min_samples_leaf=2", "Ensemble bagging; robust to noise, no scaling needed"),
                    ("Gradient Boosting","200 estimators, lr=0.1, max_depth=5", "Sequential boosting; best estimation R²=0.9906"),
                    ("XGBoost",          "200 rounds, lr=0.05, max_depth=6", "Regularized boosting with column subsampling"),
                ]
                for name, params, desc in ml_models:
                    st.markdown(f"""
                    <div class='research-card' style='margin-bottom:8px;'>
                      <strong style='color:#3fb950;'>{name}</strong><br>
                      <span style='color:#8b949e; font-size:0.85rem;'>Params: {params}</span><br>
                      <span style='color:#c9d1d9; font-size:0.9rem;'>{desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
            with mc2:
                st.markdown("#### 🧠 Deep Learning Models")
                dl_models = [
                    ("LSTM",        "2 layers × 128 units, dropout=0.2, Adam", "Long Short-Term Memory; captures temporal dependencies"),
                    ("BiLSTM",      "2 layers × 64 units (bi-directional), dropout=0.2", "Bidirectional LSTM; leverages both past and future context"),
                    ("CNN-BiLSTM",  "Conv1D(64, k=3) + BiLSTM(64) + Dense", "Hybrid CNN for local patterns + BiLSTM for sequence modeling"),
                ]
                for name, params, desc in dl_models:
                    st.markdown(f"""
                    <div class='research-card' style='margin-bottom:8px;'>
                      <strong style='color:#d2a8ff;'>{name}</strong><br>
                      <span style='color:#8b949e; font-size:0.85rem;'>Params: {params}</span><br>
                      <span style='color:#c9d1d9; font-size:0.9rem;'>{desc}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_team:
            st.markdown("""
### Team & Acknowledgements

<div class='research-card' style='margin-bottom:12px;'>
  <strong style='color:#58a6ff; font-size:1.1rem;'>👤 Research Intern</strong><br>
  <div style='margin-top:8px; color:#c9d1d9;'>B.Tech CSE — Yeshwantrao Chavan College of Engineering (YCCE), Nagpur</div>
</div>

<div class='research-card' style='margin-bottom:12px;'>
  <strong style='color:#3fb950; font-size:1.1rem;'>🏛️ Institution</strong><br>
  <div style='margin-top:8px; color:#c9d1d9;'>
    <strong>IIIT Nagpur</strong> — Indian Institute of Information Technology, Nagpur<br>
    Affiliate College: Yeshwantrao Chavan College of Engineering (YCCE), Nagpur
  </div>
</div>

<div class='research-card' style='margin-bottom:12px;'>
  <strong style='color:#ffa657; font-size:1.1rem;'>🗃️ Data Source</strong><br>
  <div style='margin-top:8px; color:#c9d1d9;'>
    Central Pollution Control Board (CPCB), India — National Ambient Air Quality Monitoring Programme
  </div>
</div>
""", unsafe_allow_html=True)
            lc1, lc2 = st.columns(2)
            lc1.link_button("📂 GitHub Repository",
                "https://github.com/buddy1809Ai/AQI-Prediction-Using-Machine-Learning-and-Deep-Learning",
                use_container_width=True)
            lc2.link_button("🗃️ CPCB Dataset",
                "https://drive.google.com/drive/folders/1b9-j1RqWOviFtg6pS5b_JWSswNb7RuGQ?usp=drive_link",
                use_container_width=True)
            st.markdown("""
<div style='text-align:center; margin-top:24px; color:#8b949e;'>
  IIIT Nagpur &nbsp;|&nbsp; YCCE Nagpur &nbsp;|&nbsp; CPCB India<br>
  AQI Prediction Research — 18 Cities | 7 Models | 3 Horizons | 2018–2023
</div>
""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"About page error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────────────────────
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
