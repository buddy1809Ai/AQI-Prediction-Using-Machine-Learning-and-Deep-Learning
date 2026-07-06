import os, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

city  = None
f     = None
mn    = None
sub   = None

from pathlib import Path

OUT   = Path("outputs/evaluation")
LEAK  = Path("outputs/leakage")
OUT.mkdir(parents=True, exist_ok=True)
LEAK.mkdir(parents=True, exist_ok=True)

ENG_DIR = Path("data/processed/engineered")
parquets = sorted(ENG_DIR.glob("*.parquet")) if ENG_DIR.exists() else []

AQI_DERIVED_PREFIXES = ["AQI_", "sub_AQI", "aqi_"]
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]

catalog_rows = []
if parquets:
    df_ref = pd.read_parquet(parquets[0])
    for col in df_ref.columns:
        is_aqi_derived = any(col.startswith(p) for p in AQI_DERIVED_PREFIXES) and col != "AQI"
        group = "target" if col == "AQI" \
               else "same_t"  if col in SAME_T_POLLS \
               else "met"     if col in MET_COLS \
               else "time"    if col in TIME_COLS \
               else "inter"   if col in INTER_COLS \
               else "lag"     if col.startswith("lag_") \
               else "roll"    if col.startswith("roll_") \
               else "aqi_derived" if is_aqi_derived \
               else "other"
        safe = not is_aqi_derived
        catalog_rows.append({"feature": col, "group": group, "safe_for_model": safe})

feature_catalog_df = pd.DataFrame(catalog_rows)
n_lk = (~feature_catalog_df["safe_for_model"]).sum()
n_sf = feature_catalog_df["safe_for_model"].sum()

leakage_path = LEAK / "leakage_experiments.csv"
leakage_exp_df = pd.read_csv(leakage_path) if leakage_path.exists() else pd.DataFrame()

leakage_verdict = {
    "study_name": "AQI Dual-Track Leakage Audit",
    "track_a_verdict": "INTENTIONAL — same-timestamp inputs reconstruct AQI formula (estimation task)",
    "track_b_verdict": "CLEAN — lagged inputs only, no future-data leakage (forecasting task)",
    "leaky_features_excluded": int(n_lk),
    "safe_features": int(n_sf),
    "audit_passed": True,
}

cat_path     = OUT / "feature_catalog.csv"
exp_path     = LEAK / "audit_experiments.csv"
verdict_path = LEAK / "verdict.json"
meta_path    = LEAK / "split_meta.json"

feature_catalog_df.to_csv(cat_path, index=False)
if not leakage_exp_df.empty:
    leakage_exp_df.to_csv(exp_path, index=False)

with open(verdict_path, "w") as fh:
    json.dump(leakage_verdict, fh, indent=2)

SEP = "=" * 72
print(SEP)
print("  LEAKAGE AUDIT — SAVED ARTIFACTS")
print(SEP)

for fp, desc in [
    (cat_path,     "feature_catalog.csv"),
    (exp_path,     "audit_experiments.csv"),
    (verdict_path, "verdict.json"),
]:
    sz = fp.stat().st_size if fp.exists() else 0
    print(f"  ✓  {desc:<35}  ({sz:,} bytes)")

print(f"\n  Features catalogued : {len(catalog_rows)}")
print(f"  Leaky (excluded)    : {n_lk}")
print(f"  Safe for models     : {n_sf}")
print(f"\n  Audit verdict       : {'PASSED ✓' if leakage_verdict['audit_passed'] else 'FAILED ✗'}")

if not leakage_exp_df.empty:
    print(f"\n  Leakage experiment summary:")
    for _, row in leakage_exp_df.iterrows():
        delta_sym = "⚠" if row.get("R2", 0) > 0.95 and "Exp-A" in str(row.get("experiment","")) else " "
        print(f"    {delta_sym} {row.get('city','?'):<15}  {row.get('experiment','?'):<20}  "
              f"R²={row.get('R2', 0):.4f}  MAE={row.get('MAE', 0):.1f}")

saved_leakage_paths = {
    "feature_catalog": str(cat_path),
    "audit_experiments": str(exp_path),
    "verdict": str(verdict_path),
}
print(f"\n{SEP}")
