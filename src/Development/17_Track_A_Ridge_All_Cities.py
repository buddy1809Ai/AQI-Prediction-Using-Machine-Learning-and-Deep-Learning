import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

REC_DIR = Path("data/processed/recovered")
OUT_DIR = Path("outputs/tables")
CKPT    = OUT_DIR / "track_a_ridge.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "AQI"
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
SEED        = 42

SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

AQI_PFX = ["AQI_","sub_AQI","aqi_"]

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

done_cities = set()
if CKPT.exists():
    done_cities = set(pd.read_csv(CKPT)["city"].tolist())
    print(f"Checkpoint found — {len(done_cities)} cities already done")

results  = []
parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []

SEP = "=" * 72
print(SEP)
print("  TRACK A — RIDGE REGRESSION — All Cities")
print(SEP)

for pq in parquets:
    city = pq.stem
    if city in done_cities:
        continue
    df   = pd.read_parquet(pq)
    drop_aqi = [c for c in df.columns if any(c.startswith(p) for p in AQI_PFX) and c != TARGET]
    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    needed    = feat_cols + [TARGET]
    df_m      = df[needed].dropna()
    c         = len(df_m.columns)
    n         = len(df_m)
    if n < 2000:
        continue

    nt = int(n * TRAIN_FRAC)
    nv = int(n * VAL_FRAC)
    X  = df_m[feat_cols].values
    y  = df_m[TARGET].values

    X_tr = X[:nt];       y_tr = y[:nt]
    X_va = X[nt:nt+nv];  y_va = y[nt:nt+nv]
    X_te = X[nt+nv:];    y_te = y[nt+nv:]

    sc     = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)

    t0  = time.time()
    mdl = Ridge(alpha=10.0)
    mdl.fit(X_tr_s, y_tr)
    tr_t = time.time() - t0
    yp   = mdl.predict(X_te_s)
    inf_t = time.time() - t0 - tr_t

    r2v   = r2_score(y_te, yp)
    maev  = mean_absolute_error(y_te, yp)
    rmsev = rmse(y_te, yp)
    results.append({"city": city, "model": "Ridge",
                    "n_rows": n, "n_features": len(feat_cols),
                    "R2": round(r2v, 4), "MAE": round(maev, 2),
                    "RMSE": round(rmsev, 2), "train_s": round(tr_t, 2)})
    print(f"  [{city}]  n={n:,}  feats={len(feat_cols)}  "
          f"R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rmsev:.1f}  t={tr_t:.1f}s")

track_a_ridge_df = pd.DataFrame(results)
if CKPT.exists():
    old = pd.read_csv(CKPT)
    track_a_ridge_df = pd.concat([old, track_a_ridge_df], ignore_index=True)

track_a_ridge_df.to_csv(CKPT, index=False)
print(f"\n  Saved: {CKPT}  ({len(track_a_ridge_df)} rows)")
