import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

REC_DIR = Path("data/processed/recovered")
OUT_DIR = Path("outputs/tables")
CKPT    = OUT_DIR / "track_b_rf.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET     = "AQI"
TRAIN      = 0.70
VAL        = 0.15
SEED       = 42
HORIZONS   = [1, 6, 24]

AQI_PFX      = ["AQI_","sub_AQI","aqi_"]
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

done_keys = set()
if CKPT.exists():
    done_df = pd.read_csv(CKPT)
    done_keys = set(zip(done_df["city"], done_df["horizon"]))

results  = []
parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []

SEP = "=" * 72
print(SEP)
print("  TRACK B — RANDOM FOREST  (18 cities × 3 horizons)")
print("  Features: lags + rolling + met + time")
print("  EXCLUDED: same-t pollutants, all AQI-derived features")
print(SEP)

for pq in parquets:
    city = pq.stem
    df   = pd.read_parquet(pq)

    # Track B feature set: lagged + rolling ONLY (no same-t pollutants)
    drop_cols  = ([TARGET] +
                  SAME_T_POLLS + INTER_COLS +
                  [c for c in df.columns if any(c.startswith(p) for p in AQI_PFX)])
    lag_feats  = [c for c in df.columns if c.startswith("lag_")]
    roll_feats = [c for c in df.columns if c.startswith("roll_")]
    met_avail  = [c for c in MET_COLS  if c in df.columns]
    time_avail = [c for c in TIME_COLS if c in df.columns]
    feat_cols  = lag_feats + roll_feats + met_avail + time_avail

    needed = feat_cols + [TARGET]
    df_m   = df[[c for c in needed if c in df.columns]].copy()
    feat_cols = [c for c in feat_cols if c in df_m.columns]
    c = len(feat_cols)
    n = len(df_m.dropna(subset=[TARGET]))

    nt_base = int(n * TRAIN)
    nv_base = int(n * VAL)

    for h in HORIZONS:
        key = (city, h)
        if key in done_keys:
            continue
        y_shifted = df_m[TARGET].shift(-h)
        valid     = df_m[feat_cols].notna().all(axis=1) & y_shifted.notna()
        Xv = df_m.loc[valid, feat_cols].values
        yv = y_shifted[valid].values
        ntr = int(len(Xv) * TRAIN); nvl = int(len(Xv) * VAL)
        X_tr = Xv[:ntr];     y_tr = yv[:ntr]
        X_te = Xv[ntr+nvl:]; y_te = yv[ntr+nvl:]
        if len(X_te) < 200:
            continue
        t0  = time.time()
        mdl = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)
        mdl.fit(X_tr, y_tr)
        tr_t = time.time() - t0
        yp   = mdl.predict(X_te)
        inf_t = time.time() - t0 - tr_t
        r2v  = r2_score(y_te, yp)
        maev = mean_absolute_error(y_te, yp)
        rv   = rmse(y_te, yp)
        results.append({"city": city, "model": "RandomForest", "horizon": h,
                        "R2": round(r2v,4), "MAE": round(maev,2), "RMSE": round(rv,2),
                        "train_s": round(tr_t,1)})

    h_r2 = {h: next((r["R2"] for r in results if r["city"]==city and r["horizon"]==h), None)
             for h in HORIZONS}
    print(f"  [{city}]  feats={c}  "
          + "  ".join(f"h{h}h: R²={h_r2[h]:.3f}" for h in HORIZONS if h_r2[h] is not None))

track_b_rf_df = pd.DataFrame(results)
if CKPT.exists():
    old = pd.read_csv(CKPT)
    track_b_rf_df = pd.concat([old, track_b_rf_df], ignore_index=True)

sub  = track_b_rf_df.groupby("horizon")["R2"].mean()
track_b_rf_df.to_csv(CKPT, index=False)

print(f"\n  Saved: {CKPT}  ({len(track_b_rf_df)} rows)")
print(f"  Track B RF average R²: " + " | ".join(f"h{h}h={sub.get(h,0):.3f}" for h in HORIZONS))
