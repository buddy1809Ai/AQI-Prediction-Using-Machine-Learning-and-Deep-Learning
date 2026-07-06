import os, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

OUT_DIR  = Path("outputs/leakage")
REC_DIR  = Path("data/processed/recovered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "AQI"
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
RANDOM_STATE = 42
AQI_PFXS     = ["AQI_","sub_AQI","aqi_"]

SEP = "=" * 72
print(SEP)
print("  PHASE 2 — LEAKAGE EXPERIMENTS (RandomForest, 100 trees)")
print("  Exp-A: same-t pollutants + met + time")
print("  Exp-B: met + time ONLY")
print("  Exp-C: pollutant lags + rolling + met + time")
print(SEP)

parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []
results  = []

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def split_and_train(df, feat_cols, target=TARGET, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    df_m = df[[target] + feat_cols].dropna()
    n  = len(df_m)
    nt = int(n * train_frac)
    nv = int(n * val_frac)
    X_tr = df_m.iloc[:nt][feat_cols].values
    y_tr = df_m.iloc[:nt][target].values
    X_te = df_m.iloc[nt + nv:][feat_cols].values
    y_te = df_m.iloc[nt + nv:][target].values
    mdl  = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    mdl.fit(X_tr, y_tr)
    yp   = mdl.predict(X_te)
    from sklearn.metrics import r2_score, mean_absolute_error
    return r2_score(y_te, yp), mean_absolute_error(y_te, yp), rmse(y_te, yp), len(feat_cols)

for pq in parquets[:2]:
    city = pq.stem
    df   = pd.read_parquet(pq)

    lag_feats  = [c for c in df.columns if c.startswith("lag_")]
    roll_feats = [c for c in df.columns if c.startswith("roll_")]

    exp_a_feats = [c for c in SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS
                   if c in df.columns]
    r2_a, mae_a, rmse_a, fa = split_and_train(df, exp_a_feats)

    exp_b_feats = [c for c in MET_COLS + TIME_COLS if c in df.columns]
    r2_b, mae_b, rmse_b, fb = split_and_train(df, exp_b_feats)

    exp_c_feats = [c for c in lag_feats + roll_feats + MET_COLS + TIME_COLS
                   if c in df.columns]
    r2_c, mae_c, rmse_c, fc = split_and_train(df, exp_c_feats)

    delta_ab = r2_a - r2_b
    delta_ac = r2_a - r2_c

    for exp_id, r2v, maev, rmsev, nf in [
        ("Exp-A_same_t", r2_a, mae_a, rmse_a, fa),
        ("Exp-B_met_only", r2_b, mae_b, rmse_b, fb),
        ("Exp-C_lagged", r2_c, mae_c, rmse_c, fc),
    ]:
        results.append({"city": city, "experiment": exp_id,
                        "n_features": nf, "R2": round(r2v, 4),
                        "MAE": round(maev, 2), "RMSE": round(rmsev, 2)})

    print(f"\n  [{city}]")
    print(f"    Exp-A (same-t pollutants) : R²={r2_a:.4f}  MAE={mae_a:.1f}")
    print(f"    Exp-B (met + time only)   : R²={r2_b:.4f}  MAE={mae_b:.1f}")
    print(f"    Exp-C (lagged features)   : R²={r2_c:.4f}  MAE={mae_c:.1f}")
    print(f"    ΔR² A→B: {delta_ab:+.4f}  |  ΔR² A→C: {delta_ac:+.4f}")
    if delta_ab > 0.3:
        print(f"    ⚠ HIGH LEAKAGE RISK: same-t pollutants dominate R² by {delta_ab:.2f}")
    else:
        print(f"    ✓ Leakage risk acceptable (Δ={delta_ab:.4f})")

leakage_exp_df = pd.DataFrame(results)
leakage_exp_df.to_csv(OUT_DIR / "leakage_experiments.csv", index=False)

print(f"\n{SEP}")
print(f"  Leakage experiments saved: outputs/leakage/leakage_experiments.csv")
print(f"  NOTE: Track A uses same-t inputs by design (AQI estimation task).")
print(f"        Track B uses lagged inputs only (true forecasting task).")
