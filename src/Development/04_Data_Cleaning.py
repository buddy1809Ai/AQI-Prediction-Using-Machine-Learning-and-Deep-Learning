
import os, glob, re, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_ROOT   = "CPCB_Data-20260602T165735Z-3-001/CPCB_Data"
OUT_DIR     = "outputs/cleaned"
CHUNK_SIZE  = 100_000
MAX_FF_HRS  = 2      # max consecutive hours to forward-fill
MAX_INTERP_HRS = 6   # max gap for linear interpolation (after forward-fill)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── Column definitions ───────────────────────────────────────────────────────
TS_COL   = "Timestamp"
DROP_COLS = ["Xylene (µg/m³)", "O Xylene (µg/m³)", "Eth-Benzene (µg/m³)",
             "MP-Xylene (µg/m³)", "VWS (m/s)", "TOT-RF (mm)", "RF (mm)"]

CORE_POLLUTANTS = [
    "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)", "NO2 (µg/m³)",
    "NOx (ppb)", "NH3 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)",
    "Ozone (µg/m³)", "Benzene (µg/m³)", "Toluene (µg/m³)",
]
MET_COLS = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "SR (W/mt2)", "BP (mmHg)"]

# Domain-knowledge outlier bounds (physical maxima)
BOUNDS = {
    "PM2.5 (µg/m³)": (0, 1200), "PM10 (µg/m³)": (0, 1500),
    "NO (µg/m³)": (0, 2000),    "NO2 (µg/m³)": (0, 500),
    "NOx (ppb)": (0, 3000),     "NH3 (µg/m³)": (0, 600),
    "SO2 (µg/m³)": (0, 800),    "CO (mg/m³)": (0, 50),
    "Ozone (µg/m³)": (0, 400),  "Benzene (µg/m³)": (0, 50),
    "Toluene (µg/m³)": (0, 500),
    "AT (°C)": (-10, 55),       "RH (%)": (0, 100),
    "WS (m/s)": (0, 50),        "WD (deg)": (0, 360),
    "SR (W/mt2)": (0, 1500),    "BP (mmHg)": (600, 850),
}

# ─── CPCB AQI computation ────────────────────────────────────────────────────
AQI_BREAKPOINTS = {
    "PM2.5 (µg/m³)": [
        (0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)],
    "PM10 (µg/m³)": [
        (0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
    "NO2 (µg/m³)": [
        (0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
    "SO2 (µg/m³)": [
        (0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2100,401,500)],
    "CO (mg/m³)": [
        (0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
    "Ozone (µg/m³)": [
        (0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
    "NH3 (µg/m³)": [
        (0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2400,401,500)],
}

def _sub_index(val, bps):
    if pd.isna(val) or val < 0:
        return np.nan
    for (cl, ch, al, ah) in bps:
        if cl <= val <= ch:
            return al + (val - cl) * (ah - al) / max(ch - cl, 1e-9)
    return 500.0

def compute_aqi(row):
    sub_indices = []
    for poll, bps in AQI_BREAKPOINTS.items():
        if poll in row.index:
            si = _sub_index(row[poll], bps)
            if not np.isnan(si):
                sub_indices.append(si)
    return round(max(sub_indices), 1) if sub_indices else np.nan

def aqi_category(aqi):
    if pd.isna(aqi): return "Unknown"
    if aqi <= 50:    return "Good"
    if aqi <= 100:   return "Satisfactory"
    if aqi <= 200:   return "Moderate"
    if aqi <= 300:   return "Poor"
    if aqi <= 400:   return "Very Poor"
    return "Severe"

def is_stub(fpath):
    try:
        df = pd.read_csv(fpath, nrows=3, low_memory=False)
        return len(df) == 0
    except Exception:
        return True

# ─── Per-city cleaning pipeline ──────────────────────────────────────────────
city_folders = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
)

cleaning_log = {}
all_city_dfs  = {}

print("=" * 72)
print("02 — DATA CLEANING PIPELINE")
print("=" * 72)

for city in city_folders:
    cpath    = os.path.join(DATA_ROOT, city)
    all_csv  = sorted(glob.glob(os.path.join(cpath, "*.csv")))
    real_csv = [f for f in all_csv if not is_stub(f)]
    
    if not real_csv:
        print(f"\n  {city}: no real files, skipping")
        continue

    print(f"\n[{city}] Processing {len(real_csv)} files…")
    
    chunks = []
    for fpath in real_csv:
        try:
            for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
                chunk[TS_COL] = pd.to_datetime(chunk[TS_COL], errors='coerce')
                chunk = chunk.dropna(subset=[TS_COL])
                chunks.append(chunk)
        except Exception as e:
            print(f"  ⚠ Read error {os.path.basename(fpath)}: {e}")
    
    if not chunks:
        print(f"  {city}: no readable chunks, skipping")
        continue

    df = pd.concat(chunks, ignore_index=True)
    raw_rows = len(df)
    
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing, errors='ignore')
    
    df = df.drop_duplicates(subset=[TS_COL])
    dup_removed = raw_rows - len(df)
    
    df = df.sort_values(TS_COL).reset_index(drop=True)
    
    full_idx = pd.date_range(df[TS_COL].min(), df[TS_COL].max(), freq="15T")
    df = df.set_index(TS_COL).reindex(full_idx)
    df.index.name = TS_COL
    gaps_15min = df.isnull().all(axis=1).sum()
    
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    
    num_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32]]
    df[num_cols] = df[num_cols].fillna(method='ffill', limit=MAX_FF_HRS * 4)
    df[num_cols] = df[num_cols].interpolate(method='linear', limit=MAX_INTERP_HRS * 4)
    
    df_hr = df.resample("1H").mean(numeric_only=True)
    
    df_hr["AQI"] = df_hr.apply(compute_aqi, axis=1)
    df_hr["AQI_Category"] = df_hr["AQI"].apply(aqi_category)
    df_hr["City"] = city
    
    core_present = [c for c in CORE_POLLUTANTS if c in df_hr.columns]
    df_hr = df_hr.dropna(subset=core_present, how='all')
    df_hr = df_hr.dropna(subset=["AQI"])
    
    for col in df_hr.select_dtypes('float64').columns:
        df_hr[col] = df_hr[col].astype(np.float32)
    
    out_path = os.path.join(OUT_DIR, f"{city.replace(' ','_')}_cleaned.parquet")
    df_hr.to_parquet(out_path)
    
    miss_after = df_hr[core_present].isnull().mean().mean() * 100
    cleaning_log[city] = dict(
        raw_rows=raw_rows, dup_removed=dup_removed,
        gaps_15min=int(gaps_15min), hourly_rows=len(df_hr),
        aqi_valid=int(df_hr["AQI"].notna().sum()),
        missing_pct_after=round(miss_after, 2),
        year_min=df_hr.index.year.min(), year_max=df_hr.index.year.max(),
        aqi_mean=round(float(df_hr["AQI"].mean()), 1),
        aqi_std=round(float(df_hr["AQI"].std()), 1),
        file=out_path,
    )
    all_city_dfs[city] = df_hr
    
    print(f"  Raw rows: {raw_rows:,} → Hourly rows: {len(df_hr):,} | "
          f"AQI valid: {cleaning_log[city]['aqi_valid']:,} | "
          f"Miss%: {miss_after:.1f}% | AQI μ={cleaning_log[city]['aqi_mean']}")

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("CLEANING SUMMARY")
print("=" * 72)
print(f"\n{'City':<20} {'Raw':>10} {'Hourly':>10} {'AQI Valid':>10} {'Miss%':>7} {'AQI_Mean':>9} {'Yrs'}")
print("-" * 72)
total_hrly = 0
for city, lg in cleaning_log.items():
    print(f"{city:<20} {lg['raw_rows']:>10,} {lg['hourly_rows']:>10,} "
          f"{lg['aqi_valid']:>10,} {lg['missing_pct_after']:>7.1f}% "
          f"{lg['aqi_mean']:>9.1f} "
          f"{lg['year_min']}–{lg['year_max']}")
    total_hrly += lg['hourly_rows']
print("-" * 72)
print(f"{'TOTAL':<20} {'':>10} {total_hrly:>10,}")
print(f"\nCleaned parquet files saved to: {OUT_DIR}/")
print(f"\ncleaning_log exported ✓")
print(f"all_city_dfs exported ({len(all_city_dfs)} cities)")
