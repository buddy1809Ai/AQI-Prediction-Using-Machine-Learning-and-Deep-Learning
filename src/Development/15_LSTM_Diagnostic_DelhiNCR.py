import os, time, warnings
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
SEQ       = 24
TRAIN     = 0.70
VAL       = 0.15
SEED      = 42
REC_DIR   = Path("data/processed/recovered")
OUT_DIR   = Path("outputs/tables")
FIG_DIR   = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "AQI"
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]
AQI_PFX      = ["AQI_","sub_AQI","aqi_"]

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

parquets  = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []
pq        = next((p for p in parquets if "Delhi" in p.stem), parquets[0] if parquets else None)

if pq is None:
    print("No data found — skipping LSTM diagnostic")
    lstm_diagnostic = {"verdict": "SKIP", "proceed": False}
else:
    city = pq.stem
    df   = pd.read_parquet(pq)

    print(SEP)
    print(f"  LSTM DIAGNOSTIC — {city}")
    print(SEP)
    print(f"  Loaded : {len(df):,} rows  |  {df.shape[1]} columns")

    AQI_PFX_LIST = ["AQI_","sub_AQI","aqi_"]
    candidates   = [c for c in SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS
                    if c in df.columns]
    feat_cols    = candidates
    drop_aqi     = [c for c in df.columns if any(c.startswith(p) for p in AQI_PFX_LIST) and c != TARGET]
    df_m         = df[[TARGET] + feat_cols].dropna()
    c            = len(feat_cols)
    n            = len(df_m)

    print(f"  Features available : {c}")
    print(f"  Target             : {TARGET}  (non-null: {df_m[TARGET].notna().sum():,})")

    nt  = int(n * TRAIN)
    nv  = int(n * VAL)
    nte = n - nt - nv

    X_all = df_m[feat_cols].values
    y_all = df_m[TARGET].values

    X_tr = X_all[:nt];        y_tr = y_all[:nt]
    X_va = X_all[nt:nt+nv];   y_va = y_all[nt:nt+nv]
    X_te = X_all[nt+nv:];     y_te = y_all[nt+nv:]

    sc     = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)

    sc_y   = MinMaxScaler()
    y_tr_s = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_s = sc_y.transform(y_va.reshape(-1,1)).ravel()

    Xs_tr, ys_tr = make_sequences(X_tr_s, y_tr_s, SEQ)
    Xs_va, ys_va = make_sequences(X_va_s, y_va_s, SEQ)
    Xs_te, ys_te = make_sequences(X_te_s,
                                   sc_y.transform(y_te.reshape(-1,1)).ravel(), SEQ)

    model = build_lstm(c, SEQ)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    cb_list = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    t0         = time.time()
    hist       = model.fit(Xs_tr, ys_tr, validation_data=(Xs_va, ys_va),
                           epochs=50, batch_size=128, callbacks=cb_list, verbose=0)
    train_time = time.time() - t0
    epochs_ran = len(hist.history["loss"])

    yp_s    = model.predict(Xs_te, verbose=0)
    yp_inv  = sc_y.inverse_transform(yp_s).ravel()
    yt_inv  = sc_y.inverse_transform(ys_te.reshape(-1,1)).ravel()

    r2_test  = r2_score(yt_inv, yp_inv)
    mae_test = mean_absolute_error(yt_inv, yp_inv)
    rmse_test = rmse(yt_inv, yp_inv)

    yp_tr_s  = model.predict(Xs_tr, verbose=0)
    yp_tr_inv = sc_y.inverse_transform(yp_tr_s).ravel()
    yt_tr_inv = sc_y.inverse_transform(ys_tr.reshape(-1,1)).ravel()
    r2_train = r2_score(yt_tr_inv, yp_tr_inv)

    pass_threshold = 0.70
    diag_verdict   = "PASS" if r2_test >= pass_threshold else "BORDERLINE"
    proceed_lstm   = r2_test >= 0.55

    print(f"\n  Epochs  : {epochs_ran}  |  Train time: {train_time:.0f}s")
    print(f"  R² test : {r2_test:.4f}  |  MAE: {mae_test:.1f}  |  RMSE: {rmse_test:.1f}")
    print(f"  R² train: {r2_train:.4f}  (gap: {r2_train - r2_test:.4f})")
    print(f"  Verdict : {diag_verdict}  |  Proceed to full 18-city run: {proceed_lstm}")

    BG, TEXT, DIM = "#1D1D20", "#fbfbff", "#909094"
    fig_lstm_loss, ax = plt.subplots(figsize=(9, 4), facecolor=BG)
    ax.set_facecolor(BG)
    epochs_x = list(range(1, epochs_ran + 1))
    ax.plot(epochs_x, hist.history["loss"],     label="Train loss", color="#A1C9F4", lw=2)
    ax.plot(epochs_x, hist.history["val_loss"], label="Val loss",   color="#FFB482", lw=2, ls="--")
    ax.set_xlabel("Epoch", color=TEXT); ax.set_ylabel("MSE Loss", color=TEXT)
    ax.set_title(f"LSTM Diagnostic — {city} — Loss Curve", color=TEXT, fontsize=12)
    ax.legend(facecolor=BG, labelcolor=TEXT)
    ax.tick_params(colors=TEXT)
    for sp in ax.spines.values(): sp.set_color(DIM)
    fig_lstm_loss.tight_layout()
    fig_lstm_loss.savefig(FIG_DIR / "lstm_diagnostic_loss.png", dpi=120,
                          bbox_inches="tight", facecolor=BG)
    plt.close(fig_lstm_loss)

    n_plot = min(500, len(yp_inv))
    fig_lstm_pred, ax2 = plt.subplots(figsize=(11, 4), facecolor=BG)
    ax2.set_facecolor(BG)
    ax2.plot(yt_inv[:n_plot],  label="Actual AQI",    color="#A1C9F4", lw=1.5)
    ax2.plot(yp_inv[:n_plot],  label="Predicted AQI", color="#FFB482", lw=1.5, ls="--")
    ax2.set_title(f"LSTM Diagnostic — {city} — Actual vs Predicted (R²={r2_test:.4f})",
                  color=TEXT, fontsize=11)
    ax2.set_xlabel("Hour", color=TEXT); ax2.set_ylabel("AQI", color=TEXT)
    ax2.legend(facecolor=BG, labelcolor=TEXT)
    ax2.tick_params(colors=TEXT)
    for sp in ax2.spines.values(): sp.set_color(DIM)
    fig_lstm_pred.tight_layout()
    fig_lstm_pred.savefig(FIG_DIR / "lstm_diagnostic_pred.png", dpi=120,
                          bbox_inches="tight", facecolor=BG)
    plt.close(fig_lstm_pred)

    lstm_diagnostic = {
        "city": city, "r2_test": round(r2_test, 4), "mae_test": round(mae_test, 2),
        "r2_train": round(r2_train, 4), "epochs": epochs_ran,
        "verdict": diag_verdict, "proceed": proceed_lstm,
    }

    print(f"\n  Loss curve saved: outputs/figures/lstm_diagnostic_loss.png")
    print(f"  Prediction plot saved: outputs/figures/lstm_diagnostic_pred.png")

print(f"\n{SEP}")
