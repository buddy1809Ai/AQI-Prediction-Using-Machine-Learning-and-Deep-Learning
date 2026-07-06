import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

CLEANED_DIR = Path("outputs/cleaned")
ENG_DIR     = Path("outputs/engineered")
OUT_DIR     = Path("outputs/leakage")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "AQI"
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene","Xylene"]
MET_COLS    = ["WS","WD","RH","Temp","RF","SR","BP","VWS"]
TIME_COLS   = ["hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos","is_weekend","season"]
INTER_COLS  = ["PM_ratio","PM_total","NO_ratio","wind_u","wind_v","PM2.5_x_RH","PM10_x_Temp"]
LAG_PREFIXES  = ("AQI_lag","PM2.5_lag","PM10_lag","NO2_lag","SO2_lag","CO_lag","Ozone_lag")
ROLL_PREFIXES = ("AQI_roll","PM2.5_roll","PM10_roll","NO2_roll")
AQI_PREFIXES  = ("AQI_lag","AQI_roll","AQI_trend")
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

SEP = "="*72

print(SEP)
print("  PHASE 0 — CITY FORENSICS")
eng_parquets = sorted(ENG_DIR.glob("*.parquet"))
print(f"  Engineered cities found: {len(eng_parquets)}")
print(SEP)

forensics_rows = []
miss_rows      = []

cln_parquets = {p.stem: p for p in CLEANED_DIR.glob("*.parquet")}

for pq in eng_parquets:
    city = pq.stem
    df   = pd.read_parquet(pq)
    n_eng = len(df)

    cln_key = city
    df_cln  = pd.read_parquet(cln_parquets[cln_key]) if cln_key in cln_parquets else None
    n_cln   = len(df_cln) if df_cln is not None else 0

    has_aqi       = TARGET in df.columns
    n_aqi_nonnull = int(df[TARGET].notna().sum()) if has_aqi else 0

    miss_same_t = {c: int(df[c].isna().sum()) for c in SAME_T_POLLS if c in df.columns}
    miss_met    = {c: int(df[c].isna().sum()) for c in MET_COLS    if c in df.columns}
    miss_time   = {c: int(df[c].isna().sum()) for c in TIME_COLS   if c in df.columns}
    miss_inter  = {c: int(df[c].isna().sum()) for c in INTER_COLS  if c in df.columns}

    ta_present = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    needed     = set(TRACK_A_CANDIDATES)
    n_complete_strict = int((df[ta_present].notna().all(axis=1)).sum()) if ta_present else 0

    miss_per_feat = df[ta_present].isna().mean()
    worst = miss_per_feat[miss_per_feat > 0].sort_values(ascending=False).head(3)

    core_poll = [c for c in SAME_T_POLLS[:6] if c in df.columns]
    if core_poll:
        mask = df[core_poll].notna().all(axis=1)
        n_recoverable = int(mask.sum())
        recoverable = "YES" if n_recoverable > 0.5 * n_eng else "PARTIAL"
    else:
        n_recoverable = 0
        recoverable = "NO"

    reason = "OK" if n_complete_strict > 0.7 * n_eng else (
             "recoverable" if recoverable != "NO" else "insufficient_data")

    print(f"\n  [{city}]")
    print(f"    Cleaned rows : {n_cln:,}")
    print(f"    Engineered   : {n_eng:,}")
    print(f"    AQI non-null : {n_aqi_nonnull:,}")
    print(f"    Complete rows: {n_complete_strict:,}  ({100*n_complete_strict/max(n_eng,1):.1f}%)")
    print(f"    Recoverable  : {recoverable}  ({n_recoverable:,} rows with core pollutants)")
    if len(worst):
        print(f"    Worst missing: {dict(worst.round(3))}")

    all_groups = {
        "same_t_polls": miss_same_t,
        "met":          miss_met,
        "time":         miss_time,
        "interactions": miss_inter,
    }
    for grp, feats in all_groups.items():
        high = {f: v for f, v in feats.items() if v > 0.1 * n_eng}
        if high:
            print(f"    ⚠ {grp} high-miss: {high}")

    forensics_rows.append({
        "city": city, "n_cleaned": n_cln, "n_engineered": n_eng,
        "n_aqi_nonnull": n_aqi_nonnull, "n_complete_strict": n_complete_strict,
        "pct_complete": round(100*n_complete_strict/max(n_eng,1), 2),
        "n_recoverable": n_recoverable, "recoverable": recoverable, "reason": reason,
    })

    for feat, cnt in {**miss_same_t, **miss_met, **miss_time, **miss_inter}.items():
        if cnt > 0:
            miss_rows.append({"city": city, "feature": feat, "n_missing": cnt,
                               "pct_missing": round(100*cnt/max(n_eng,1), 2)})

city_forensics_df = pd.DataFrame(forensics_rows)
feat_miss_df      = pd.DataFrame(miss_rows)

print(f"\n{SEP}")
print(f"  SUMMARY")
print(f"  Cities processed : {len(forensics_rows)}")
r = city_forensics_df
print(f"  Fully complete   : {(r['pct_complete'] >= 70).sum()}")
print(f"  Recoverable      : {(r['recoverable'] == 'YES').sum()}")
print(f"  Partial          : {(r['recoverable'] == 'PARTIAL').sum()}")
for _, row in r.sort_values('pct_complete').head(3).iterrows():
    cnt = row['n_missing'] if 'n_missing' in row else '—'
    print(f"  ⚠ {row['city']:<22} complete={row['pct_complete']:.1f}%  recoverable={row['recoverable']}")
print(SEP)
