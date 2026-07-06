"""
main.py — AQI Prediction Research | Streamlit App (10 pages)
Launch: streamlit run app/main.py  (from repo root)
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AQI Prediction — India", page_icon="🌫️",
                   layout="wide", initial_sidebar_state="expanded")

from app.config.settings import (
    CITIES, CITY_COORDS, AQI_CATEGORIES, HEALTH_ADVICE,
    TRACK_A_MODELS, TRACK_B_MODELS, HORIZONS, BG, TEXT, DIM, PALETTE,
    FIGURES_DIR, EDA_DIR, REPORTS_DIR, EVAL_DIR, TABLES_DIR,
)
from app.utils.aqi_formula import compute_aqi, validate_inputs
from app.utils.model_loader import load_metrics, load_track_a_results, load_track_b_results, load_sample_data
from app.utils.weather_client import get_weather
from app.utils.logger import log

for k, v in {"page":"🏠 Home","city":"Delhi NCR","horizon":1,"model":"GradBoost"}.items():
    if k not in st.session_state: st.session_state[k] = v

PAGES = ["🏠 Home","🔬 Project Pipeline","📊 Dataset Overview","📈 EDA",
         "🌫️ AQI Calculator","🔮 AQI Forecast","🗺️ India Map",
         "🏙️ City Dashboard","📉 Model Comparison","🏆 Results","ℹ️ About"]

with st.sidebar:
    st.markdown("## 🌫️ AQI Research India")
    st.markdown("*18 cities · 7 models · dual-track*")
    st.divider()
    for pg in PAGES:
        btn_type = "primary" if st.session_state.page == pg else "secondary"
        if st.button(pg, key=f"nav_{pg}", use_container_width=True, type=btn_type):
            st.session_state.page = pg
            st.rerun()
    st.divider()
    st.caption("CPCB Data · Research Internship 2025")

page = st.session_state.page

# ── HOME ───────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🌫️ AQI Prediction Using Deep Learning")
    st.subheader("Dual-Track Machine Learning Research · 18 Indian Cities")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("🏙️ Cities","18"); col2.metric("📄 Records","~935K")
    col3.metric("🤖 Models","7");  col4.metric("🔬 Experiments","342")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🎯 Research Objectives
- **Track A** — Estimate current AQI from same-timestamp pollutant readings
- **Track B** — Forecast future AQI at 1h/6h/24h horizons using lagged features only
- Compare 7 models: Ridge, RF, GBR, XGB, LSTM, BiLSTM, CNN-BiLSTM
- Formally certify absence of data leakage in both tracks
        """)
    with c2:
        st.markdown("""
### 🏆 Key Results
| Model | Track A R² | Track B 1h R² |
|---|---|---|
| **Gradient Boosting** | **0.9906** | **0.6555** |
| XGBoost | 0.9571 | 0.5800 |
| Random Forest | 0.9504 | 0.5400 |
| BiLSTM | 0.8321 | 0.4289 |
| LSTM | 0.6411 | 0.4300 |
| CNN-BiLSTM | 0.5897 | 0.3900 |
| Ridge | 0.7107 | 0.3500 |
        """)
    st.info("> **Why GBR > LSTM on Track A?** AQI is a deterministic piecewise-linear formula of pollutants. Tree models capture this exactly; LSTM wastes capacity on sequence learning for an algebraic relationship.")
    st.info("👈 Use the sidebar to navigate between pages.")

# ── PROJECT PIPELINE ────────────────────────────────────────────────────────────
elif page == "🔬 Project Pipeline":
    st.title("🔬 Project Pipeline")
    st.code("""
Raw CPCB CSVs (18.7M 15-min records)
  ↓
01 Dataset Inventory     → city list, row counts, year coverage
  ↓
02 Schema Audit          → column schema, timestamp format, AQI absent
  ↓
03 Data Cleaning         → hourly aggregate, clip outliers, compute CPCB AQI
  ↓
04 EDA                   → 9 publication figures
  ↓
05 Feature Engineering   → 115 features: lags, rolling, cyclical, interactions
  ↓
06 Preprocessing         → MinMaxScaler (fit on train only), 70/15/15 split
  ↓
07 Leakage Audit         → 3 RF experiments, formal certification
  ↓
TRACK A (same-timestamp → AQI):  Ridge → RF → GBR → XGB → LSTM → BiLSTM → CNN-BiLSTM
TRACK B (lagged → future AQI):   RF → GBR → XGB → LSTM → BiLSTM → CNN-BiLSTM (×3 horizons)
  ↓
08 Scientific Comparison  → model rankings, horizon degradation, city heatmaps
  ↓
09 Final Results          → deployment recommendations, feature importance, reports
    """, language="text")
    st.success("✅ Leakage Certification: Track A = legitimate estimation · Track B = genuine forecasting")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Track A Feature Set (same-timestamp)**")
        st.code("PM2.5, PM10, NO2, SO2, CO, O3, NH3\n+ Meteorology: AT, RH, WS, WD, SR, BP\n+ Time: hour_sin/cos, month_sin/cos, DOW\n+ Interactions: PM ratio, wind_u/v, NOx_proxy", language="text")
    with c2:
        st.markdown("**Track B Feature Set (leakage-free)**")
        st.code("Pollutant lags: lag1h … lag48h\n+ Pollutant rolling: 3h/6h/12h/24h/48h mean,std\n+ Meteorology (same as Track A)\n+ Time cyclical features\nEXCLUDED: same-t pollutants + all AQI-derived features", language="text")

# ── DATASET OVERVIEW ────────────────────────────────────────────────────────────
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    sample_data = load_sample_data()
    if sample_data:
        total_rows = sum(len(df) for df in sample_data.values())
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Cities", len(sample_data)); c2.metric("Sample Rows", f"{total_rows:,}")
        c3.metric("Raw Freq", "15 min");       c4.metric("Model Freq", "1 hour")
        st.divider()
        city_sel = st.selectbox("Preview city", sorted(sample_data.keys()))
        st.dataframe(sample_data[city_sel].head(50), use_container_width=True)
        st.subheader("City Summary")
        rows = [{"City":c,"Rows":len(df),"Columns":len(df.columns),"Has AQI":"AQI" in df.columns}
                for c,df in sorted(sample_data.items())]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Sample data not found.")
    st.markdown("**Full dataset:** ~935K hourly records (2010–2025), 18 Indian cities. Not included (size). Use sample dataset for demonstration.")

# ── EDA ────────────────────────────────────────────────────────────────────────
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    EDA_FIGS = {"AQI Distribution":"eda_01_aqi_distribution.png","Mean AQI by City":"eda_02_city_aqi_comparison.png",
                "Category Distribution (%)":"eda_03_category_distribution.png","Monthly Trend":"eda_04_monthly_trend.png",
                "Diurnal Pattern":"eda_05_diurnal_pattern.png","Seasonal Boxplot":"eda_06_seasonal_boxplot.png",
                "Yearly Trend":"eda_07_yearly_trend.png","PM2.5 vs AQI":"eda_08_pm25_vs_aqi_scatter.png",
                "Correlation Heatmap":"eda_09_correlation_heatmap.png"}
    PUB_FIGS = {"Track A Comparison":"fig1_track_a_model_comparison.png","Track B Comparison":"fig2_track_b_model_comparison.png",
                "Horizon Degradation":"fig3_horizon_degradation.png","City Heatmap":"fig4_city_model_heatmap.png",
                "Classical vs DL":"fig5_classical_vs_dl.png","DL Comparison":"fig6_dl_comparison.png",
                "Best vs Worst City":"fig7_best_vs_worst_city.png","Track A vs B":"fig8_track_a_vs_track_b.png",
                "City Difficulty":"fig9_city_difficulty.png","Feature Importance":"fig10_feature_category_importance.png",
                "Research Certificate":"fig11_final_certification.png","Final Summary":"fig13_final_summary.png"}
    tab1, tab2 = st.tabs(["📊 EDA Figures","📄 Publication Figures"])
    with tab1:
        sel = st.selectbox("EDA figure", list(EDA_FIGS.keys()))
        p = EDA_DIR / EDA_FIGS[sel]
        if p.exists() and p.stat().st_size > 1000: st.image(str(p), use_container_width=True)
        else: st.warning(f"Not found: {p.name}")
    with tab2:
        sel2 = st.selectbox("Publication figure", list(PUB_FIGS.keys()))
        found = False
        for d in [FIGURES_DIR, Path("AQI_DeepLearning_Project/outputs/figures"), FIGURES_DIR.parent.parent/"AQI_Prediction_Project_Final/outputs/figures"]:
            pp = d / PUB_FIGS[sel2]
            if pp.exists() and pp.stat().st_size > 1000:
                st.image(str(pp), use_container_width=True); found = True; break
        if not found: st.warning(f"Not found: {PUB_FIGS[sel2]}")

# ── AQI CALCULATOR ──────────────────────────────────────────────────────────────
elif page == "🌫️ AQI Calculator":
    st.title("🌫️ CPCB AQI Calculator")
    st.info("Official CPCB sub-index formula. **Weather inputs (temperature, humidity, etc.) are NOT part of the CPCB AQI calculation** and are not requested here.")
    with st.form("aqi_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            pm25=st.number_input("PM2.5 (µg/m³)",0.0,1000.0,60.0,1.0)
            pm10=st.number_input("PM10 (µg/m³)",0.0,1500.0,100.0,1.0)
            no2=st.number_input("NO₂ (µg/m³)",0.0,800.0,40.0,1.0)
        with c2:
            so2=st.number_input("SO₂ (µg/m³)",0.0,2100.0,30.0,1.0)
            co=st.number_input("CO (mg/m³)",0.0,50.0,1.0,0.1)
            o3=st.number_input("O₃ (µg/m³)",0.0,1000.0,50.0,1.0)
        with c3:
            nh3=st.number_input("NH₃ (µg/m³)",0.0,2400.0,20.0,1.0)
            st.markdown("")
            submitted=st.form_submit_button("Calculate AQI ▶",use_container_width=True,type="primary")
    if submitted:
        is_valid, warns = validate_inputs(pm25=pm25,pm10=pm10,no2=no2,so2=so2,co=co,o3=o3,nh3=nh3)
        for w in warns: st.warning(w)
        if is_valid:
            res = compute_aqi(pm25=pm25,pm10=pm10,no2=no2,so2=so2,co=co,o3=o3,nh3=nh3)
            aqi_val, cat, color = res["aqi"], res["category"], res["color"]
            st.divider()
            ca, cb = st.columns([1,2])
            with ca:
                st.markdown(f'<div style="background:{color}22;border:2px solid {color};border-radius:12px;padding:20px;text-align:center"><div style="font-size:3rem;font-weight:700;color:{color}">{aqi_val}</div><div style="font-size:1.3rem;color:{color};font-weight:600">{cat}</div><div style="color:#909094;margin-top:8px">Dominant: <b>{res["dominant_pollutant"]}</b></div></div>', unsafe_allow_html=True)
            with cb:
                st.markdown(f"### 🏥 Health Advice\n{res['health_advice']}")
                st.markdown("### 📊 Sub-indices")
                si_rows=[{"Pollutant":k.upper(),"Sub-index":v} for k,v in res["sub_indices"].items()]
                st.dataframe(pd.DataFrame(si_rows),use_container_width=True,hide_index=True)

# ── AQI FORECAST ────────────────────────────────────────────────────────────────
elif page == "🔮 AQI Forecast":
    st.title("🔮 AQI Forecast — Track B")
    st.info("Precomputed results — no retraining.")
    c1,c2,c3 = st.columns(3)
    with c1: city_sel=st.selectbox("City",CITIES)
    with c2: hz=st.select_slider("Horizon",[1,6,24],1)
    with c3: mdl=st.selectbox("Model",TRACK_B_MODELS,index=2)
    tb = load_track_b_results()
    if not tb.empty:
        cc=next((c for c in ["city","City"] if c in tb.columns),None)
        hc=next((c for c in ["horizon","Horizon"] if c in tb.columns),None)
        mask=pd.Series([True]*len(tb))
        if cc: mask &= tb[cc].astype(str).str.lower()==city_sel.lower()
        if "model" in tb.columns: mask &= tb["model"]==mdl
        if hc: mask &= tb[hc].astype(str)==str(hz)
        row=tb[mask]
        if not row.empty:
            r2=float(row["R2"].iloc[0]) if "R2" in row.columns else None
            mae=float(row["MAE"].iloc[0]) if "MAE" in row.columns else None
            rmse=float(row["RMSE"].iloc[0]) if "RMSE" in row.columns else None
            cm1,cm2,cm3,cm4=st.columns(4)
            cm1.metric("R²",f"{r2:.4f}" if r2 else "—"); cm2.metric("MAE",f"{mae:.1f}" if mae else "—")
            cm3.metric("RMSE",f"{rmse:.1f}" if rmse else "—")
            conf="High" if r2 and r2>0.6 else ("Medium" if r2 and r2>0.4 else "Low")
            cm4.metric("Confidence",conf)
        else: st.warning(f"No results for {city_sel}|{mdl}|t+{hz:02d}h")
        if hc and "model" in tb.columns:
            sub=tb[(tb["model"]==mdl)&(tb[hc].astype(str)==str(hz))].sort_values("R2",ascending=False)
            if not sub.empty:
                st.divider(); st.subheader(f"{mdl} t+{hz:02d}h — All Cities")
                dcols=[c for c in [cc,"R2","MAE","RMSE"] if c and c in sub.columns]
                st.dataframe(sub[dcols].reset_index(drop=True),use_container_width=True)
    else: st.warning("Track B results not found.")
    st.divider(); st.subheader("🌤️ Current Weather")
    w=get_weather(city_sel)
    if w:
        wc1,wc2,wc3,wc4=st.columns(4)
        wc1.metric("🌡️ Temp",f"{w['temp_c']}°C"); wc2.metric("💧 Humidity",f"{w['humidity_pct']}%")
        wc3.metric("💨 Wind",f"{w['wind_ms']} m/s"); wc4.metric("📊 Pressure",f"{w['pressure_hpa']} hPa")
    else: st.caption("🔑 Add OPENWEATHER_API_KEY to .streamlit/secrets.toml for live weather.")

# ── INDIA MAP ──────────────────────────────────────────────────────────────────
elif page == "🗺️ India Map":
    st.title("🗺️ India AQI — Interactive City Map")
    try:
        import plotly.graph_objects as go
        lats,lons,texts,colors_,sizes=[],[],[],[],[]
        metrics=load_metrics()
        for city,(lat,lon) in CITY_COORDS.items():
            by_city=metrics.get("track_a",{}).get("GradBoost",{}).get("by_city",{})
            city_r2=by_city.get(city,{}).get("R2") if isinstance(by_city.get(city,{}),dict) else None
            col_="#8DE5A1" if city_r2 and city_r2>0.95 else "#A1C9F4" if city_r2 and city_r2>0.85 else "#FFB482" if city_r2 and city_r2>0.70 else "#FF9F9B"
            lats.append(lat); lons.append(lon); colors_.append(col_); sizes.append(18)
            txt=f"<b>{city}</b>"
            if city_r2: txt+=f"<br>GBR R²={city_r2:.4f}"
            texts.append(txt)
        fig=go.Figure(go.Scattergeo(lat=lats,lon=lons,text=texts,hoverinfo="text",mode="markers+text",
            textposition="top center",textfont=dict(size=9,color="white"),
            marker=dict(size=sizes,color=colors_,opacity=0.85,line=dict(width=1,color="white"))))
        fig.update_layout(geo=dict(scope="asia",center=dict(lat=20.5,lon=78.9),projection_scale=4,
            bgcolor="#1D1D20",landcolor="#2a2a2e",oceancolor="#1D1D20",coastlinecolor="#555",countrycolor="#555"),
            paper_bgcolor="#1D1D20",margin=dict(l=0,r=0,t=30,b=0),height=580,font=dict(color="white"),
            title=dict(text="18 Indian Cities — GradBoost Track A R²",font=dict(color="white",size=13)))
        st.plotly_chart(fig,use_container_width=True)
        city_map_sel=st.selectbox("Select city:",sorted(CITY_COORDS.keys()))
        if st.button("Open City Dashboard →"):
            st.session_state.city=city_map_sel; st.session_state.page="🏙️ City Dashboard"; st.rerun()
    except ImportError:
        st.error("Plotly not installed. `pip install plotly`")
        st.dataframe(pd.DataFrame([{"City":c,"Lat":lat,"Lon":lon} for c,(lat,lon) in sorted(CITY_COORDS.items())]),use_container_width=True)

# ── CITY DASHBOARD ─────────────────────────────────────────────────────────────
elif page == "🏙️ City Dashboard":
    city_sel=st.selectbox("City",CITIES,index=CITIES.index(st.session_state.city) if st.session_state.city in CITIES else 0)
    st.session_state.city=city_sel; st.title(f"🏙️ {city_sel} Dashboard")
    ta=load_track_a_results(); tb=load_track_b_results()
    ca,cb=st.columns(2)
    with ca:
        st.subheader("Track A — Estimation")
        if not ta.empty:
            cc=next((c for c in ["city","City"] if c in ta.columns),None)
            if cc:
                sub=ta[ta[cc].astype(str).str.lower()==city_sel.lower()]
                if not sub.empty:
                    dcols=[c for c in ["model","R2","MAE","RMSE"] if c in sub.columns]
                    st.dataframe(sub[dcols].sort_values("R2",ascending=False) if "R2" in sub.columns else sub,use_container_width=True,hide_index=True)
                else: st.info(f"No Track A data for {city_sel}")
    with cb:
        st.subheader("Track B — Forecasting")
        if not tb.empty:
            cc=next((c for c in ["city","City"] if c in tb.columns),None)
            if cc:
                sub=tb[tb[cc].astype(str).str.lower()==city_sel.lower()]
                if not sub.empty: st.dataframe(sub.reset_index(drop=True),use_container_width=True,hide_index=True)
                else: st.info(f"No Track B data for {city_sel}")
    st.divider(); st.subheader(f"BiLSTM Forecast Plots — {city_sel}")
    city_slug=city_sel.replace(" ","_")
    fc=st.columns(3)
    for i,h in enumerate(["h01","h06","h24"]):
        with fc[i]:
            st.markdown(f"**t+{h[1:]}h**")
            pp=FIGURES_DIR/f"{city_slug}_{h}_bilstm_pred.png"
            if pp.exists() and pp.stat().st_size>1000: st.image(str(pp),use_container_width=True)
            else: st.caption(f"No figure: {pp.name}")

# ── MODEL COMPARISON ────────────────────────────────────────────────────────────
elif page == "📉 Model Comparison":
    st.title("📉 Model Comparison")
    ta=load_track_a_results(); tb=load_track_b_results()
    t1,t2=st.tabs(["Track A — Estimation","Track B — Forecasting"])
    with t1:
        if not ta.empty and "model" in ta.columns and "R2" in ta.columns:
            summ=ta.groupby("model")[["R2","MAE","RMSE"]].mean().sort_values("R2",ascending=False).reset_index()
            summ.columns=["Model","Avg R²","Avg MAE","Avg RMSE"]
            st.dataframe(summ.style.highlight_max(subset=["Avg R²"],color="#17b26a33").highlight_min(subset=["Avg MAE","Avg RMSE"],color="#17b26a33"),use_container_width=True,hide_index=True)
            best=summ.iloc[0]
            st.success(f"🏆 Best: **{best['Model']}** (R²={best['Avg R²']:.4f})")
        else: st.warning("No Track A results.")
    with t2:
        if not tb.empty and "model" in tb.columns and "R2" in tb.columns:
            hc=next((c for c in ["horizon","Horizon"] if c in tb.columns),None)
            if hc:
                summ_b=tb.groupby(["model",hc])[["R2","MAE","RMSE"]].mean().reset_index()
                summ_b.columns=["Model","Horizon","Avg R²","Avg MAE","Avg RMSE"]
                st.dataframe(summ_b.sort_values(["Horizon","Avg R²"],ascending=[True,False]),use_container_width=True,hide_index=True)
            else: st.dataframe(tb.groupby("model")[["R2","MAE","RMSE"]].mean().sort_values("R2",ascending=False),use_container_width=True)
        else: st.warning("No Track B results.")
    metrics=load_metrics()
    if metrics:
        st.divider(); st.subheader("📋 model_metrics.json")
        st.json(metrics)

# ── RESULTS ────────────────────────────────────────────────────────────────────
elif page == "🏆 Results":
    st.title("🏆 Final Results")
    summ={"Model":["GradBoost","XGBoost","RandomForest","BiLSTM","Ridge","LSTM","CNN-BiLSTM"],
          "Track A R²":[0.9906,0.9571,0.9504,0.8321,0.7107,0.6411,0.5897],
          "Track A MAE":[3.3,7.1,6.2,22.3,31.3,27.0,37.0],
          "Track B 1h R²":[0.6555,0.5800,0.5400,0.4289,0.3500,0.4300,0.3900],
          "Track B 24h R²":[0.3558,0.3200,0.3100,0.1699,0.1500,0.1700,0.1500]}
    st.dataframe(pd.DataFrame(summ),use_container_width=True,hide_index=True)
    st.divider(); st.subheader("📊 Publication Figures")
    PUB=[("Track A Comparison","fig1_track_a_model_comparison.png"),("Track B Comparison","fig2_track_b_model_comparison.png"),
         ("Horizon Degradation","fig3_horizon_degradation.png"),("City Heatmap","fig4_city_model_heatmap.png"),
         ("Classical vs DL","fig5_classical_vs_dl.png"),("Final Summary","fig13_final_summary.png")]
    found_figs=[]
    for lbl,fn in PUB:
        for d in [FIGURES_DIR,Path("AQI_DeepLearning_Project/outputs/figures")]:
            pp=d/fn
            if pp.exists() and pp.stat().st_size>1000:
                found_figs.append((lbl,pp)); break
    if found_figs:
        sel=st.selectbox("Figure",[f[0] for f in found_figs])
        for lbl,pp in found_figs:
            if lbl==sel: st.image(str(pp),use_container_width=True); break
    else: st.warning("Publication figures not found.")
    st.divider(); st.subheader("📄 Research Reports")
    for rpath in [REPORTS_DIR/"final_internship_summary.md",REPORTS_DIR/"deployment_recommendations.md",REPORTS_DIR/"reviewer_qa.md"]:
        if rpath.exists():
            with st.expander(f"📄 {rpath.stem.replace('_',' ').title()}"):
                try: st.markdown(rpath.read_text())
                except: st.warning(f"Could not read {rpath}")

# ── ABOUT ───────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
## AQI Prediction Using Deep Learning — Research Internship 2025

**Dual-track study:** 18 Indian cities · 7 models · 935K hourly CPCB records · 342 evaluations

| Item | Detail |
|---|---|
| Source data | CPCB (Central Pollution Control Board) |
| Cities | Ahmedabad, Chennai, Delhi NCR, Hyderabad, Mumbai, Nagpur + 12 more |
| Period | 2010–2025 |
| Best Track A model | Gradient Boosting (R²=0.9906) |
| Best Track B model | Gradient Boosting (1h R²=0.6555) |
| Deep learning best | BiLSTM (Track A R²=0.8321) |
| Leakage audit | ✅ Passed — both tracks certified |

**Key insight:** AQI is a deterministic piecewise-linear formula of pollutant sub-indices.
Tree models capture this exactly (R²≈0.99). LSTM's strength is in genuine time-series
forecasting (Track B), where horizon degradation (1h→24h) confirms real predictive uncertainty.

**Tools:** Python 3.11 · pandas · scikit-learn · XGBoost · TensorFlow/Keras · Streamlit · Plotly

**License:** MIT
    """)
    st.caption("Built with ❤️ using Streamlit · CPCB Data · Zerve AI Workspace")
