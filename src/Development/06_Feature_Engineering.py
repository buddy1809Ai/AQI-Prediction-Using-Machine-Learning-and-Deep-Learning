
import os, glob
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

IN_DIR  = "outputs/cleaned"
OUT_DIR = "outputs/engineered"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

TARGET    = "AQI"
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48]
ROLL_WINS = [3, 6, 12, 24, 48]
KEY_POLL  = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)",
             "SO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)", "NH3 (µg/m³)"]
ALL_POLL  = KEY_POLL + ["NO (µg/m³)", "NOx (ppb)", "Benzene (µg/m³)", "Toluene (µg/m³)"]
MET_COLS  = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "SR (W/mt2)", "BP (mmHg)"]

parquets    = sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
city_names  = [os.path.basename(p).replace("_cleaned.parquet","").replace("_"," ")
               for p in parquets]
city2id     = {c: i for i, c in enumerate(sorted(set(city_names)))}
print(f"Cities ({len(city2id)}): {city2id}")

feat_logs  = {}
all_feat_dfs = {}

print("\n" + "=" * 72)
print("04 — FEATURE ENGINEERING")
print("=" * 72)

for p in parquets:
    city = os.path.basename(p).replace("_cleaned.parquet","").replace("_"," ")
    df = pd.read_parquet(p)
    if len(df) == 0:
        print(f"  {city}: empty, skipping")
        continue

    df = df.sort_index()
    n0 = len(df)

    df["hour"]        = df.index.hour.astype(np.int8)
    df["day_of_week"] = df.index.dayofweek.astype(np.int8)
    df["month"]       = df.index.month.astype(np.int8)
    df["day_of_year"] = df.index.dayofyear.astype(np.int16)
    df["is_weekend"]  = (df.index.dayofweek >= 5).astype(np.int8)
    df["season"]      = df["month"].map({
        12:0, 1:0, 2:0,
        3:1, 4:1, 5:1,
        6:2, 7:2, 8:2, 9:2,
        10:3, 11:3
    }).astype(np.int8)

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7).astype(np.float32)

    lag_sources = [TARGET] + [c for c in KEY_POLL if c in df.columns]
    for col in lag_sources:
        for lag in LAG_HOURS:
            tag = col.split(" ")[0].replace(".","").replace("(","")
            df[f"{tag}_lag{lag}h"] = df[col].shift(lag).astype(np.float32)

    for win in ROLL_WINS:
        df[f"AQI_roll_mean_{win}h"] = (df[TARGET].shift(1).rolling(win, min_periods=1)
                                        .mean().astype(np.float32))
        df[f"AQI_roll_std_{win}h"]  = (df[TARGET].shift(1).rolling(win, min_periods=2)
                                        .std().fillna(0).astype(np.float32))
        df[f"AQI_roll_max_{win}h"]  = (df[TARGET].shift(1).rolling(win, min_periods=1)
                                        .max().astype(np.float32))

    for col in ["PM2.5 (µg/m³)", "PM10 (µg/m³)"]:
        if col in df.columns:
            tag = col.split(" ")[0].replace(".","")
            for win in [6, 24]:
                df[f"{tag}_roll_mean_{win}h"] = (
                    df[col].shift(1).rolling(win, min_periods=1).mean().astype(np.float32))

    df["AQI_diff1h"]  = df[TARGET].diff(1).astype(np.float32)
    df["AQI_diff6h"]  = df[TARGET].diff(6).astype(np.float32)
    df["AQI_diff24h"] = df[TARGET].diff(24).astype(np.float32)
    df["AQI_trend"]   = np.sign(df["AQI_diff1h"]).fillna(0).astype(np.int8)

    eps = 1e-6
    if "PM2.5 (µg/m³)" in df.columns and "PM10 (µg/m³)" in df.columns:
        df["PM25_PM10_ratio"] = (df["PM2.5 (µg/m³)"] /
                                  (df["PM10 (µg/m³)"] + eps)).clip(0,1).astype(np.float32)
    if "NO (µg/m³)" in df.columns and "NO2 (µg/m³)" in df.columns:
        df["NOx_proxy"] = (df["NO (µg/m³)"] + df["NO2 (µg/m³)"]).astype(np.float32)
    if "CO (mg/m³)" in df.columns and "PM2.5 (µg/m³)" in df.columns:
        df["CO_PM25_product"] = (df["CO (mg/m³)"] * df["PM2.5 (µg/m³)"]).astype(np.float32)
    if "SO2 (µg/m³)" in df.columns and "NO2 (µg/m³)" in df.columns:
        df["SO2_NO2_sum"] = (df["SO2 (µg/m³)"] + df["NO2 (µg/m³)"]).astype(np.float32)

    if "WS (m/s)" in df.columns and "WD (deg)" in df.columns:
        _wd_rad = np.deg2rad(df["WD (deg)"])
        df["wind_u"] = (-df["WS (m/s)"] * np.sin(_wd_rad)).astype(np.float32)
        df["wind_v"] = (-df["WS (m/s)"] * np.cos(_wd_rad)).astype(np.float32)

    df["city_id"] = city2id.get(city, 0)

    df = df.dropna(subset=[TARGET])

    feat_cols = [c for c in df.columns if c not in [TARGET, "AQI_Category", "City"]]
    row_miss   = df[feat_cols].isnull().mean(axis=1)
    df = df[row_miss < 0.5].copy()

    n1 = len(df)
    n_feat = len(feat_cols)

    out_path = os.path.join(OUT_DIR, f"{city.replace(' ','_')}_engineered.parquet")
    df.to_parquet(out_path, compression='snappy')
    all_feat_dfs[city] = df

    feat_logs[city] = dict(
        rows_in=n0, rows_out=n1, n_features=n_feat,
        file=out_path, city_id=city2id.get(city, 0)
    )
    print(f"  {city:<22}: {n0:>7,} → {n1:>7,} rows | {n_feat} features")

print("\n" + "=" * 72)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 72)
total_out = sum(v['rows_out'] for v in feat_logs.values())
n_feat_ex = next(iter(feat_logs.values()))['n_features'] if feat_logs else 0
print(f"\n  Total engineered rows : {total_out:,}")
print(f"  Features per row      : {n_feat_ex}")
print(f"  Engineered parquets saved to: {OUT_DIR}/")

sample_city = next(iter(all_feat_dfs.values()))
feat_cats = {
    "Time features":           [c for c in sample_city.columns if any(x in c for x in ['hour','day','month','season','weekend','sin','cos'])],
    "Lag features":            [c for c in sample_city.columns if '_lag' in c],
    "Rolling statistics":      [c for c in sample_city.columns if '_roll_' in c],
    "Trend indicators":        [c for c in sample_city.columns if 'diff' in c or 'trend' in c],
    "Interaction features":    [c for c in sample_city.columns if any(x in c for x in ['ratio','proxy','product','sum','wind_u','wind_v'])],
    "Core pollutants":         [c for c in sample_city.columns if any(x in c for x in ['PM2.5','PM10','NO','SO2','CO','Ozone','NH3','Benzene','Toluene','NOx'])
                                  and 'lag' not in c and 'roll' not in c and 'diff' not in c],
    "Meteorological":          [c for c in sample_city.columns if any(x in c for x in ['AT','RH','WS','WD','SR','BP']) and 'lag' not in c],
}
print("\n  Feature Categories:")
for cat, cols in feat_cats.items():
    print(f"    {cat:<28}: {len(cols):>3} features")
print(f"\n  City ID mapping: {city2id}")
print(f"\nfeat_logs and all_feat_dfs exported ✓")
print(f"city2id exported ✓")
