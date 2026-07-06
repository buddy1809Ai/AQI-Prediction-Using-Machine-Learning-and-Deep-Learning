import os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ENG_DIR       = Path("data/processed/recovered")
OUT_DIR       = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET        = "AQI"
SEQ_LEN       = 24
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
RANDOM_STATE  = 42

SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

def rmse_fn(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def make_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

def build_lstm(n_features, seq_len=24):
    try:
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True,
                              input_shape=(seq_len, n_features)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model
    except ImportError:
        return None

parquets = sorted(ENG_DIR.glob("*.parquet")) if ENG_DIR.exists() else []
track_a_results = []

SEP = "=" * 72
print(SEP)
print("  TRACK A — AQI ESTIMATION  (same-timestamp features → AQI(t))")
print(f"  Cities: {len(parquets)} | Target: {TARGET} | Seq_len: {SEQ_LEN}")
print(SEP)

for pq in parquets:
    city = pq.stem
    df   = pd.read_parquet(pq)

    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    all_needed = feat_cols + [TARGET]
    df_model = df[all_needed].dropna()

    if len(df_model) < 5000:
        print(f"  [SKIP] {city}: only {len(df_model)} complete rows")
        continue

    n   = len(df_model)
    nt  = int(n * TRAIN_FRAC)
    nv  = int(n * VAL_FRAC)
    nte = n - nt - nv

    X_train = df_model.iloc[:nt][feat_cols].values
    y_train = df_model.iloc[:nt][TARGET].values
    X_val   = df_model.iloc[nt:nt+nv][feat_cols].values
    y_val   = df_model.iloc[nt:nt+nv][TARGET].values
    X_test  = df_model.iloc[nt+nv:][feat_cols].values
    y_test  = df_model.iloc[nt+nv:][TARGET].values

    sc = MinMaxScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_val_sc   = sc.transform(X_val)
    X_test_sc  = sc.transform(X_test)

    classical = [
        ("Ridge",        Ridge(alpha=10.0)),
        ("RandomForest", RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
        ("GradBoost",    GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=RANDOM_STATE)),
        ("XGBoost",      XGBRegressor(n_estimators=300, max_depth=5, verbosity=0, random_state=RANDOM_STATE)),
    ]

    for mname, mdl in classical:
        Xtr = X_train_sc if mname == "Ridge" else X_train
        Xte = X_test_sc  if mname == "Ridge" else X_test
        t0  = time.time()
        mdl.fit(Xtr, y_train)
        tr_t = time.time() - t0
        yp   = mdl.predict(Xte)
        inf_t = time.time() - t0 - tr_t
        r2v   = r2_score(y_test, yp)
        maev  = mean_absolute_error(y_test, yp)
        rmsev = rmse_fn(y_test, yp)
        track_a_results.append({"city": city, "model": mname,
                                 "R2": round(r2v, 4), "MAE": round(maev, 2),
                                 "RMSE": round(rmsev, 2), "train_s": round(tr_t, 1)})

    lstm_model = build_lstm(len(feat_cols), SEQ_LEN)
    if lstm_model is not None:
        from tensorflow.keras.callbacks import EarlyStopping
        Xtr_s = sc.fit_transform(X_train)
        Xva_s = sc.transform(X_val)
        Xte_s = sc.transform(X_test)
        sc_y  = MinMaxScaler()
        ytr_s = sc_y.fit_transform(y_train.reshape(-1,1)).ravel()
        yva_s = sc_y.transform(y_val.reshape(-1,1)).ravel()
        Xs_tr, ys_tr = make_sequences(Xtr_s, ytr_s, SEQ_LEN)
        Xs_va, ys_va = make_sequences(Xva_s, sc_y.transform(y_val.reshape(-1,1)).ravel(), SEQ_LEN)
        Xs_te, ys_te = make_sequences(Xte_s, sc_y.transform(y_test.reshape(-1,1)).ravel(), SEQ_LEN)
        cb_list = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        t0 = time.time()
        lstm_model.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
                       epochs=50, batch_size=128, callbacks=cb_list, verbose=0)
        tr_t  = time.time() - t0
        yp_lstm = sc_y.inverse_transform(lstm_model.predict(Xs_te, verbose=0)).ravel()
        y_te_inv = sc_y.inverse_transform(ys_te.reshape(-1,1)).ravel()
        r2v  = r2_score(y_te_inv, yp_lstm)
        maev = mean_absolute_error(y_te_inv, yp_lstm)
        rmsev = rmse_fn(y_te_inv, yp_lstm)
        track_a_results.append({"city": city, "model": "LSTM",
                                 "R2": round(r2v, 4), "MAE": round(maev, 2),
                                 "RMSE": round(rmsev, 2), "train_s": round(tr_t, 1)})

    city_rows = [r for r in track_a_results if r["city"] == city]
    best_r2   = max(r["R2"] for r in city_rows)
    print(f"  [{city:<22}]  n={n:>7,}  feats={len(feat_cols):>3}  best_R²={best_r2:.4f}")

track_a_df = pd.DataFrame(track_a_results)

for mf, mname in [
    ("track_a_ridge.csv", "Ridge"),
    ("track_a_rf.csv", "RandomForest"),
    ("track_a_gbr.csv", "GradBoost"),
    ("track_a_xgb.csv", "XGBoost"),
    ("track_a_lstm.csv", "LSTM"),
]:
    fpath = OUT_DIR / mf
    if fpath.exists():
        existing = pd.read_csv(fpath)
        existing["model"] = mname
        track_a_df = pd.concat([track_a_df, existing], ignore_index=True).drop_duplicates(
            subset=["city","model"], keep="last")

track_a_df.to_csv(OUT_DIR / "final_track_a_complete.csv", index=False)
ranked = (track_a_df.sort_values("R2", ascending=False)
          .drop_duplicates("model")[["model","R2","MAE","RMSE"]])
ranked.to_csv(OUT_DIR / "track_a_model_ranking.csv", index=False)

print(f"\n{SEP}")
print(f"  Track A complete — {len(parquets)} cities × {track_a_df['model'].nunique()} models")
print(f"  Best overall: {ranked.iloc[0]['model']}  R²={ranked.iloc[0]['R2']:.4f}")
print(f"  Saved: outputs/tables/final_track_a_complete.csv")
