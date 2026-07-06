import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

SEP = "=" * 72
SEQ = 24
TRAIN = 0.70
VAL   = 0.15
SEED  = 42

REC_DIR = Path("data/processed/recovered")
OUT_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "AQI"
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

def build_lstm(n_features, seq_len=24):
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

parquets     = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []
lstm_results = []

print(SEP)
print("  TRACK A — LSTM — All Cities (18 cities)")
print(SEP)

for pq in parquets:
    city = pq.stem
    df   = pd.read_parquet(pq)

    drop_aqi  = [c for c in df.columns if any(c.startswith(p) for p in AQI_PFX) and c != TARGET]
    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]
    needed    = feat_cols + [TARGET]
    df_m      = df[needed].dropna()
    c         = len(feat_cols)
    n         = len(df_m)
    if n < 5000:
        print(f"  [SKIP] {city}: only {n:,} rows")
        continue

    nt = int(n * TRAIN);  nv = int(n * VAL)
    X_tr = df_m.iloc[:nt][feat_cols].values;       y_tr = df_m.iloc[:nt][TARGET].values
    X_va = df_m.iloc[nt:nt+nv][feat_cols].values;  y_va = df_m.iloc[nt:nt+nv][TARGET].values
    X_te = df_m.iloc[nt+nv:][feat_cols].values;    y_te = df_m.iloc[nt+nv:][TARGET].values

    sc = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)

    sc_y   = MinMaxScaler()
    y_tr_s = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_s = sc_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_s = sc_y.transform(y_te.reshape(-1,1)).ravel()

    Xs_tr, ys_tr = make_sequences(X_tr_s, y_tr_s, SEQ)
    Xs_va, ys_va = make_sequences(X_va_s, y_va_s, SEQ)
    Xs_te, ys_te = make_sequences(X_te_s, y_te_s, SEQ)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    cb_list = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    model = build_lstm(c, SEQ)
    t0    = time.time()
    hist  = model.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
                      epochs=50, batch_size=128, callbacks=cb_list, verbose=0)
    tr_t  = time.time() - t0
    ep    = len(hist.history["loss"])

    t1     = time.time()
    yp_s   = model.predict(Xs_te, verbose=0)
    inf_t  = time.time() - t1
    yp_inv = sc_y.inverse_transform(yp_s).ravel()
    yt_inv = sc_y.inverse_transform(ys_te.reshape(-1,1)).ravel()

    r2v   = r2_score(yt_inv, yp_inv)
    maev  = mean_absolute_error(yt_inv, yp_inv)
    rmsev = rmse(yt_inv, yp_inv)

    lstm_results.append({"city": city, "model": "LSTM",
                          "n_rows": n, "n_features": c, "epochs": ep,
                          "R2": round(r2v,4), "MAE": round(maev,2),
                          "RMSE": round(rmsev,2), "train_s": round(tr_t,1)})
    print(f"  [{city}]  n={n:,}  feats={c}  ep={ep}  "
          f"R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rmsev:.1f}  t={tr_t:.0f}s")

track_a_lstm_df = pd.DataFrame(lstm_results)
track_a_lstm_df.to_csv(OUT_DIR / "track_a_lstm.csv", index=False)

classical_path = OUT_DIR / "final_track_a_classical.csv"
if classical_path.exists():
    classical_df  = pd.read_csv(classical_path)
    shared        = [c for c in ["city","model","R2","MAE","RMSE"] if c in classical_df.columns]
    track_a_complete_df = pd.concat([classical_df[shared], track_a_lstm_df[shared]],
                                     ignore_index=True)
    track_a_complete_df.to_csv(OUT_DIR / "final_track_a_complete.csv", index=False)
    valid        = track_a_complete_df.dropna(subset=["R2"])
    best_c       = valid.loc[valid["R2"].idxmax()]
    worst_c      = valid.loc[valid["R2"].idxmin()]
    valid_sorted = valid.sort_values("R2", ascending=True)

    BG, TEXT, DIM = "#1D1D20", "#fbfbff", "#909094"
    PAL = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4"]
    fig_track_a_lstm, ax = plt.subplots(figsize=(10, max(5, len(valid_sorted) * 0.4)), facecolor=BG)
    ax.set_facecolor(BG)
    colors = [PAL[0] if m == "LSTM" else PAL[2]
              for m in valid_sorted["model"]]
    bars = ax.barh(range(len(valid_sorted)), valid_sorted["R2"], color=colors)
    ax.set_yticks(range(len(valid_sorted)))
    ax.set_yticklabels(valid_sorted["city"], color=TEXT, fontsize=8)
    ax.set_xlabel("R²", color=TEXT)
    ax.set_title("Track A — LSTM vs Best Classical Model (R² by City)", color=TEXT, fontsize=12)
    ax.tick_params(colors=TEXT)
    for sp in ax.spines.values(): sp.set_color(DIM)
    for bar, val in zip(bars, valid_sorted["R2"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color=TEXT, fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PAL[0], label="LSTM"),
                       Patch(color=PAL[2], label="Classical")],
              facecolor=BG, labelcolor=TEXT)
    fig_track_a_lstm.tight_layout()
    fig_track_a_lstm.savefig(FIG_DIR / "track_a_lstm_comparison.png", dpi=120,
                              bbox_inches="tight", facecolor=BG)
    plt.close(fig_track_a_lstm)

print(f"\n{SEP}")
print(f"  Track A LSTM complete — {len(lstm_results)} cities")
print(f"  Saved: outputs/tables/track_a_lstm.csv")
