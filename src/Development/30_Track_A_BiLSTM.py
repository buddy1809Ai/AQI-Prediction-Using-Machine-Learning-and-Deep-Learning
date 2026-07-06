import sys, os, io, subprocess
import platform as _plat
import time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

REC_DIR = Path("data/processed/recovered")
OUT_DIR = Path("outputs/tables")
CKPT    = OUT_DIR / "track_a_bilstm.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "AQI"
SEQ          = 24
TRAIN        = 0.70
VAL          = 0.15
AQI_PFX      = ["AQI_","sub_AQI","aqi_"]
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

def make_sequences(X, y, seq_len=24):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

def build_bilstm(n_features, seq_len=24):
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True),
            input_shape=(seq_len, n_features)),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

done_cities = set()
if CKPT.exists():
    done_cities = set(pd.read_csv(CKPT)["city"].tolist())

results  = []
parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []

SEP = "=" * 72
print(SEP)
print("  TRACK A — BiLSTM — All Cities (18 cities)")
print(SEP)

for pq in parquets:
    city = pq.stem
    if city in done_cities:
        continue
    df   = pd.read_parquet(pq)
    drop_aqi  = [c for c in df.columns if any(c.startswith(p) for p in AQI_PFX) and c != TARGET]
    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    df_m      = df[[TARGET] + feat_cols].dropna()
    col       = len(feat_cols)
    c         = col
    n         = len(df_m)
    if n < 5000:
        continue

    nt = int(n * TRAIN); nv = int(n * VAL)
    X_all = df_m[feat_cols].values; y_all = df_m[TARGET].values
    X_tr  = X_all[:nt];     y_tr = y_all[:nt]
    X_va  = X_all[nt:nt+nv]; y_va = y_all[nt:nt+nv]
    X_te  = X_all[nt+nv:];  y_te = y_all[nt+nv:]

    sc  = MinMaxScaler()
    sc_y = MinMaxScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xva_s = sc.transform(X_va)
    Xte_s = sc.transform(X_te)
    ytr_s = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    yva_s = sc_y.transform(y_va.reshape(-1,1)).ravel()
    yte_s = sc_y.transform(y_te.reshape(-1,1)).ravel()

    Xs_tr, ys_tr = make_sequences(Xtr_s, ytr_s, SEQ)
    Xs_va, ys_va = make_sequences(Xva_s, yva_s, SEQ)
    Xs_te, ys_te = make_sequences(Xte_s, yte_s, SEQ)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    cbs = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    model = build_bilstm(c, SEQ)
    t0   = time.time()
    model.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
              epochs=50, batch_size=128, callbacks=cbs, verbose=0)
    tr_t = time.time() - t0

    t1     = time.time()
    yp_s   = model.predict(Xs_te, verbose=0)
    inf_t  = time.time() - t1
    yp_inv = sc_y.inverse_transform(yp_s).ravel()
    yt_inv = sc_y.inverse_transform(ys_te.reshape(-1,1)).ravel()

    r2v   = r2_score(yt_inv, yp_inv)
    maev  = mean_absolute_error(yt_inv, yp_inv)
    rmsev = rmse(yt_inv, yp_inv)
    vr    = round(r2v, 4)

    results.append({"city": city, "model": "BiLSTM",
                    "n_rows": n, "n_features": c,
                    "R2": vr, "MAE": round(maev,2), "RMSE": round(rmsev,2),
                    "train_s": round(tr_t,1)})
    print(f"  [{city}]  n={n:,}  feats={c}  R²={r2v:.4f}  MAE={maev:.1f}  "
          f"RMSE={rmsev:.1f}  t={tr_t:.0f}s")

track_a_bilstm_df = pd.DataFrame(results)
if CKPT.exists():
    old = pd.read_csv(CKPT)
    track_a_bilstm_df = pd.concat([old, track_a_bilstm_df], ignore_index=True)
track_a_bilstm_df.to_csv(CKPT, index=False)

print(f"\n{SEP}")
print(f"  Saved: {CKPT}  ({len(track_a_bilstm_df)} rows)")
