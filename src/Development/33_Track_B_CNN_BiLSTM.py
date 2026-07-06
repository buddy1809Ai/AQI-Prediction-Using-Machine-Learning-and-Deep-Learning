import os, time, warnings
import _io as _c_io
import io as _io_mod
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

SEP  = "=" * 72
SEQ  = 24
TRAIN = 0.70
VAL   = 0.15
SEED  = 42

REC_DIR  = Path("data/processed/recovered")
OUT_DIR  = Path("outputs/tables")
CKPT     = OUT_DIR / "track_b_cnn_bilstm.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET   = "AQI"
HORIZONS = [1, 6, 24]

AQI_PFX      = ["AQI_","sub_AQI","aqi_"]
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]

def make_sequences(X, y, seq_len=24):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

def build_cnn_bilstm(n_features, seq_len=24):
    from tensorflow import keras
    inputs = keras.Input(shape=(seq_len, n_features))
    x      = keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x      = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x      = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(x)
    x      = keras.layers.Dropout(0.2)(x)
    x      = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)
    x      = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1)(x)
    model  = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

done_keys = set()
if CKPT.exists():
    done_df   = pd.read_csv(CKPT)
    done_keys = set(zip(done_df["city"], done_df["horizon"]))

results  = []
parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []

print(SEP)
print("  TRACK B — CNN-BiLSTM  (18 cities × 3 horizons)")
print("  seq_len=24  |  Features: lags + rolling + met + time")
print("  EXCLUDED: same-t pollutants, all AQI-derived features")
print(SEP)

for pq in parquets:
    city = pq.stem
    df   = pd.read_parquet(pq)

    lag_feats  = [c for c in df.columns if c.startswith("lag_")]
    roll_feats = [c for c in df.columns if c.startswith("roll_")]
    met_avail  = [c for c in MET_COLS  if c in df.columns]
    time_avail = [c for c in TIME_COLS if c in df.columns]
    feat_cols  = lag_feats + roll_feats + met_avail + time_avail
    df_m       = df[[c for c in feat_cols + [TARGET] if c in df.columns]].copy()
    feat_cols  = [c for c in feat_cols if c in df_m.columns]
    col = len(feat_cols); c = col
    n   = len(df_m)

    X_base = df_m[feat_cols].values
    y_base = df_m[TARGET].values

    for h in HORIZONS:
        key = (city, h)
        if key in done_keys:
            continue
        y_shifted  = df_m[TARGET].shift(-h)
        valid_mask = df_m[feat_cols].notna().all(axis=1) & y_shifted.notna()
        Xv = df_m.loc[valid_mask, feat_cols].values
        yv = y_shifted[valid_mask].values
        ntr = int(len(Xv) * TRAIN); nvl = int(len(Xv) * VAL)
        if len(Xv) < SEQ + 500:
            continue

        sc   = MinMaxScaler()
        sc_y = MinMaxScaler()
        Xtr_s = sc.fit_transform(Xv[:ntr])
        Xva_s = sc.transform(Xv[ntr:ntr+nvl])
        Xte_s = sc.transform(Xv[ntr+nvl:])
        ytr_s = sc_y.fit_transform(yv[:ntr].reshape(-1,1)).ravel()
        yva_s = sc_y.transform(yv[ntr:ntr+nvl].reshape(-1,1)).ravel()
        yte   = yv[ntr+nvl:]

        Xs_tr, ys_tr = make_sequences(Xtr_s, ytr_s, SEQ)
        Xs_va, ys_va = make_sequences(Xva_s, yva_s, SEQ)
        Xs_te, yte_aligned = make_sequences(Xte_s, yte, SEQ)

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        cbs = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
        model = build_cnn_bilstm(c, SEQ)
        t0  = time.time()
        model.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
                  epochs=50, batch_size=128, callbacks=cbs, verbose=0)
        tr_t = time.time() - t0
        t1   = time.time()
        yp_s = model.predict(Xs_te, verbose=0)
        inf_t = time.time() - t1
        yp   = sc_y.inverse_transform(yp_s).ravel()
        r2v  = r2_score(yte_aligned, yp)
        maev = mean_absolute_error(yte_aligned, yp)
        rv   = rmse(yte_aligned, yp)
        results.append({"city": city, "model": "CNN-BiLSTM", "horizon": h,
                        "R2": round(r2v,4), "MAE": round(maev,2), "RMSE": round(rv,2),
                        "train_s": round(tr_t,1)})

    h_r2 = {h: next((r["R2"] for r in results if r["city"]==city and r["horizon"]==h), None)
             for h in HORIZONS}
    print(f"  [{city}]  feats={c}  "
          + "  ".join(f"h{h}h: R²={h_r2[h]:.3f}" for h in HORIZONS if h_r2[h] is not None))

track_b_cnn_bilstm_df = pd.DataFrame(results)
if CKPT.exists():
    old = pd.read_csv(CKPT)
    track_b_cnn_bilstm_df = pd.concat([old, track_b_cnn_bilstm_df], ignore_index=True)

sub  = track_b_cnn_bilstm_df.groupby("horizon")["R2"].mean()
best = track_b_cnn_bilstm_df.loc[track_b_cnn_bilstm_df["R2"].idxmax()] if not track_b_cnn_bilstm_df.empty else None
worst = track_b_cnn_bilstm_df.loc[track_b_cnn_bilstm_df["R2"].idxmin()] if not track_b_cnn_bilstm_df.empty else None
track_b_cnn_bilstm_df.to_csv(CKPT, index=False)

print(f"\n  Saved: {CKPT}  ({len(track_b_cnn_bilstm_df)} rows)")
print(f"  Track B CNN-BiLSTM: " + " | ".join(f"h{h}h R²={sub.get(h,0):.4f}" for h in HORIZONS))
