import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths — parquets live in outputs/recovered (not data/processed/recovered) ─
REC_DIR = Path("outputs/recovered")
OUT_DIR = Path("outputs/evaluation")
CKPT    = OUT_DIR / "trackB_linear_regression_results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "AQI"
TRAIN    = 0.70
VAL      = 0.15
HORIZONS = [1, 6, 24]

AQI_PFX      = ("AQI_", "sub_AQI", "aqi_")
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek",
                "day_of_week","season","is_weekend","dow_sin","dow_cos"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp","wind_u","wind_v"]

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []

SEP = "=" * 72
print(SEP)
print("  TRACK B — LINEAR REGRESSION  (18 cities × 3 horizons)")
print("  Features: lags + rolling + met + time")
print("  EXCLUDED: same-t pollutants, all AQI-derived features")
print(SEP)

results = []

for pq in parquets:
    # strip the _recovered suffix from the filename stem
    city = pq.stem.replace("_recovered", "")
    df   = pd.read_parquet(pq)

    # Build leakage-free Track B feature set (matches 22_Track_B_RF and 23_Track_B_GBR)
    drop_cols  = [col for col in df.columns
                  if any(col.startswith(p) for p in AQI_PFX) or col in SAME_T_POLLS]
    lag_feats  = [col for col in df.columns if "_lag"       in col and col not in drop_cols]
    roll_feats = [col for col in df.columns if "_roll_mean" in col and col not in drop_cols]
    met_avail  = [col for col in MET_COLS   if col in df.columns]
    time_avail = [col for col in TIME_COLS  if col in df.columns]
    feat_cols  = lag_feats + roll_feats + met_avail + time_avail

    needed    = feat_cols + [TARGET]
    df_m      = df[[col for col in needed if col in df.columns]].copy()
    feat_cols = [col for col in feat_cols if col in df_m.columns]
    n_feats   = len(feat_cols)
    n_rows    = len(df_m)

    print(f"\n  [{city}]  n={n_rows:,}  feats={n_feats}")

    if n_feats == 0:
        print(f"    WARN: no features — skipping")
        continue

    for h in HORIZONS:
        y_shifted  = df_m[TARGET].shift(-h)
        valid_mask = df_m[feat_cols].notna().all(axis=1) & y_shifted.notna()
        Xv = df_m.loc[valid_mask, feat_cols].values
        yv = y_shifted[valid_mask].values

        ntr = int(len(Xv) * TRAIN)
        nvl = int(len(Xv) * VAL)

        if len(Xv) < 500:
            print(f"    t+{h:02d}h  LR  (skipped — insufficient data)")
            continue

        X_tr = Xv[:ntr]
        y_tr = yv[:ntr]
        X_va = Xv[ntr:ntr + nvl]
        y_va = yv[ntr:ntr + nvl]
        X_te = Xv[ntr + nvl:]
        y_te = yv[ntr + nvl:]

        if len(X_te) < 100:
            print(f"    t+{h:02d}h  LR  (skipped — test set too small)")
            continue

        t0  = time.time()
        mdl = LinearRegression()
        mdl.fit(X_tr, y_tr)
        tr_t = time.time() - t0

        yp   = mdl.predict(X_te)
        r2v  = r2_score(y_te, yp)
        maev = mean_absolute_error(y_te, yp)
        rv   = _rmse(y_te, yp)

        results.append({
            "City":               city,
            "Model":              "LinearRegression",
            "R2":                 round(r2v,  4),
            "MAE":                round(maev, 2),
            "RMSE":               round(rv,   2),
            "Number_of_Features": n_feats,
            "Train_Samples":      ntr,
            "Validation_Samples": nvl,
            "Test_Samples":       len(X_te),
            "Horizon":            h,
        })

        print(f"    t+{h:02d}h  LR  R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rv:.1f}  "
              f"train={ntr:,}  val={nvl:,}  test={len(X_te):,}  feats={n_feats}  t={tr_t:.1f}s")

# ── Save evaluation CSV ───────────────────────────────────────────────────────
track_b_lr_df = pd.DataFrame(results,
    columns=["City","Model","R2","MAE","RMSE",
             "Number_of_Features","Train_Samples","Validation_Samples","Test_Samples","Horizon"])
track_b_lr_df.to_csv(str(CKPT), index=False)

# ── Summary banner (LSTM-style per horizon) ───────────────────────────────────
print(f"\n{SEP}")
print("  TRACK B — LINEAR REGRESSION SUMMARY")
print(SEP)

for h in HORIZONS:
    sub_h = track_b_lr_df[track_b_lr_df["Horizon"] == h]
    if sub_h.empty:
        continue
    avg_r2   = sub_h["R2"].mean()
    avg_mae  = sub_h["MAE"].mean()
    avg_rmse = sub_h["RMSE"].mean()
    best_city  = sub_h.loc[sub_h["R2"].idxmax(), "City"]
    worst_city = sub_h.loc[sub_h["R2"].idxmin(), "City"]
    best_r2    = sub_h["R2"].max()
    worst_r2   = sub_h["R2"].min()
    n_cities   = len(sub_h)
    print(f"  Horizon t+{h:02d}h  |  cities={n_cities}")
    print(f"    Avg R²={avg_r2:.4f}  Avg MAE={avg_mae:.2f}  Avg RMSE={avg_rmse:.2f}")
    print(f"    Best : {best_city}  R²={best_r2:.4f}")
    print(f"    Worst: {worst_city}  R²={worst_r2:.4f}")

print(f"  Saved → {CKPT}")
print(SEP)
