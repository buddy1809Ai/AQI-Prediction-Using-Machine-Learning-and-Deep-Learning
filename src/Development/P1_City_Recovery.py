import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ENG_DIR  = Path("data/processed/engineered")
OUT_DIR  = Path("data/processed/recovered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "AQI"
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]

SEP = "=" * 72

print(SEP)
print("  PHASE 1 — CITY RECOVERY")
print("  Strategy: drop >95%-missing features, then median/ffill/bfill imputation")
print("  AQI-derived features: always excluded")
print(SEP)

eng_parquets = sorted(ENG_DIR.glob("*.parquet")) if ENG_DIR.exists() else []

MISS_THRESH        = 0.95
MIN_ROWS           = 5_000
AQI_DERIVED_PREFIXES = ["AQI_", "sub_AQI", "aqi_", "AQI_cat", "AQI_label"]

recovery_log = []

for pq in eng_parquets:
    city = pq.stem
    df   = pd.read_parquet(pq)
    n0   = len(df)

    aqi_derived = [c for c in df.columns
                   if any(c.startswith(pfx) for pfx in AQI_DERIVED_PREFIXES)
                   and c != TARGET]
    df = df.drop(columns=aqi_derived, errors="ignore")

    miss_rate      = df.isnull().mean()
    drop_high_miss = miss_rate[miss_rate > MISS_THRESH].index.tolist()
    n_dropped_feats = len(drop_high_miss)
    df = df.drop(columns=drop_high_miss, errors="ignore")

    num_cols           = df.select_dtypes(include="number").columns.tolist()
    num_cols_no_target = [c for c in num_cols if c != TARGET]

    n_train  = int(len(df) * 0.7)
    medians  = df.iloc[:n_train][num_cols_no_target].median()
    df[num_cols_no_target] = df[num_cols_no_target].fillna(medians)
    df[num_cols_no_target] = df[num_cols_no_target].ffill().bfill()

    n_before_aqi_drop = len(df)
    n1 = df[TARGET].notna().sum() if TARGET in df.columns else 0

    remaining_cols  = list(df.columns)
    final_same_t    = [c for c in SAME_T_POLLS if c in remaining_cols]
    final_met       = [c for c in MET_COLS     if c in remaining_cols]
    final_time      = [c for c in TIME_COLS    if c in remaining_cols]
    final_inter     = [c for c in INTER_COLS   if c in remaining_cols]
    track_a_feats   = final_same_t + final_met + final_time + final_inter

    lag_feats  = [c for c in remaining_cols if c.startswith("lag_")]
    roll_feats = [c for c in remaining_cols if c.startswith("roll_")]
    track_b_feats = lag_feats + roll_feats + final_met + final_time

    status = "READY" if n1 >= MIN_ROWS and len(track_a_feats) >= 5 else "SKIP"

    recovery_log.append({
        "city": city, "n_raw": n0, "n_recovered": n1,
        "track_a_feats": len(track_a_feats),
        "track_b_feats": len(track_b_feats),
        "dropped_miss": n_dropped_feats, "status": status,
    })

    if status == "READY":
        out_path = OUT_DIR / f"{city}.parquet"
        df.to_parquet(out_path, index=False)

    sym = "✓" if status == "READY" else "✗"
    print(f"  {sym}  {city:<22}  rows={n1:>7,}  "
          f"track_a_feats={len(track_a_feats):>3}  "
          f"track_b_feats={len(track_b_feats):>3}  [{status}]")

recovery_df = pd.DataFrame(recovery_log)
n_ok = (recovery_df["status"] == "READY").sum()

recovery_df.to_csv(Path("outputs/evaluation/city_feature_recovery.csv"), index=False)

print(f"\n{SEP}")
print(f"  Cities recovered and ready: {n_ok} / {len(recovery_log)}")
print(f"  Saved: outputs/evaluation/city_feature_recovery.csv")
