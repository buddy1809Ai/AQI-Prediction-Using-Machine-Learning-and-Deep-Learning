import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# FIX: Patch platform & subprocess BEFORE importing TensorFlow.
# TF → h5py → platform.processor → subprocess.check_output → Popen →
#   io.open(int_fd) → Zerve interceptor crash.
# We short-circuit by (1) stubbing platform._Processor.get() to '' and
# (2) wrapping subprocess.check_output to catch the processor probe.
# ═══════════════════════════════════════════════════════════════════════════
import platform as _platform
import subprocess as _subprocess

# Stub platform.processor so h5py never spawns a subprocess for it
class _NullProcessor:
    @staticmethod
    def get():
        return ''
_platform._Processor = _NullProcessor   # type: ignore

# Belt-and-suspenders: also wrap subprocess.check_output to return b'' for
# any call that tries to query the CPU processor string
_real_check_output = _subprocess.check_output
def _safe_check_output(args, **kwargs):
    try:
        return _real_check_output(args, **kwargs)
    except (TypeError, OSError):
        return b''
_subprocess.check_output = _safe_check_output

# NOW it is safe to import TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Restore subprocess after TF is fully loaded
_subprocess.check_output = _real_check_output

# ── Paths ────────────────────────────────────────────────────────────────────
REC_DIR = Path("outputs/recovered")
OUT_DIR = Path("outputs/tables")
MDL_DIR = Path("outputs/models/track_b_bilstm")
FIG_DIR = Path("outputs/figures")
CKPT    = OUT_DIR / "track_b_bilstm.csv"

for _d in [OUT_DIR, MDL_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Remove stale empty checkpoint
if CKPT.exists() and CKPT.stat().st_size < 50:
    CKPT.unlink()
    print(f"  Removed empty checkpoint: {CKPT}")

TARGET   = "AQI"
SEQ      = 24
TRAIN    = 0.70
VAL      = 0.15
SEED     = 42
HORIZONS = [1, 6, 24]
SEP      = "=" * 72

AQI_PFX      = ("AQI_", "sub_AQI", "aqi_")
SAME_T_POLLS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS     = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS    = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek",
                "day_of_week","season","is_weekend","dow_sin","dow_cos"]
INTER_COLS   = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp","wind_u","wind_v"]

# ── Helpers ──────────────────────────────────────────────────────────────────
def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def _mape(y_true, y_pred, eps=1e-6):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.abs(yt) > eps
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)

def make_sequences(X, y, seq_len=24):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_bilstm(n_features, seq_len=24):
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

# ── Resume from checkpoint ───────────────────────────────────────────────────
done_keys = set()
if CKPT.exists():
    _ck       = pd.read_csv(CKPT)
    done_keys = set(zip(_ck["city"], _ck["horizon"]))
    print(f"  Checkpoint: {len(done_keys)} keys already done")

results  = []
parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []
print(f"  Parquets found: {len(parquets)}  in  {REC_DIR}")

print(SEP)
print("  TRACK B — BiLSTM  (18 cities × 3 horizons)")
print(f"  seq_len={SEQ}  |  Features: lags + rolling + met + time")
print("  EXCLUDED: same-t pollutants, all AQI-derived features")
print(SEP)

for pq in parquets:
    city = pq.stem.replace("_recovered", "")
    df   = pd.read_parquet(pq)

    drop_cols   = [col for col in df.columns
                   if any(col.startswith(p) for p in AQI_PFX) or col in SAME_T_POLLS]
    lag_feats   = [col for col in df.columns if "_lag"       in col and col not in drop_cols]
    roll_feats  = [col for col in df.columns if "_roll_mean" in col and col not in drop_cols]
    met_avail   = [col for col in MET_COLS   if col in df.columns]
    time_avail  = [col for col in TIME_COLS  if col in df.columns]
    inter_avail = [col for col in INTER_COLS if col in df.columns]
    feat_cols   = lag_feats + roll_feats + met_avail + time_avail + inter_avail

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
        key = (city, h)
        if key in done_keys:
            print(f"    t+{h:02d}h  BiLSTM  (checkpoint — skipped)")
            continue

        y_shifted  = df_m[TARGET].shift(-h)
        valid_mask = df_m[feat_cols].notna().all(axis=1) & y_shifted.notna()
        Xv = df_m.loc[valid_mask, feat_cols].values
        yv = y_shifted[valid_mask].values

        ntr = int(len(Xv) * TRAIN)
        nvl = int(len(Xv) * VAL)

        if len(Xv) < SEQ + 500:
            print(f"    t+{h:02d}h  BiLSTM  (skipped — insufficient data)")
            continue

        sc   = MinMaxScaler()
        sc_y = MinMaxScaler()
        Xtr_s = sc.fit_transform(Xv[:ntr])
        Xva_s = sc.transform(Xv[ntr:ntr + nvl])
        Xte_s = sc.transform(Xv[ntr + nvl:])
        ytr_s = sc_y.fit_transform(yv[:ntr].reshape(-1, 1)).ravel()
        yva_s = sc_y.transform(yv[ntr:ntr + nvl].reshape(-1, 1)).ravel()
        yte   = yv[ntr + nvl:]

        Xs_tr, ys_tr       = make_sequences(Xtr_s, ytr_s, SEQ)
        Xs_va, ys_va       = make_sequences(Xva_s, yva_s, SEQ)
        Xs_te, yte_aligned = make_sequences(Xte_s, yte,   SEQ)

        if len(Xs_tr) < 50 or len(Xs_va) < 10:
            print(f"    t+{h:02d}h  BiLSTM  (skipped — too few sequences)")
            continue

        ckpt_path = str(MDL_DIR / f"{city}_h{h:02d}_bilstm.keras")
        cbs = [
            EarlyStopping(monitor="val_loss", patience=7,
                          restore_best_weights=True, verbose=0),
            ModelCheckpoint(filepath=ckpt_path, monitor="val_loss",
                            save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-6, verbose=0),
        ]

        model = build_bilstm(n_feats, SEQ)
        t0    = time.time()
        hist  = model.fit(
            Xs_tr, ys_tr,
            validation_data=(Xs_va, ys_va),
            epochs=60, batch_size=128,
            callbacks=cbs, verbose=0,
        )
        tr_t = time.time() - t0
        ep   = len(hist.history["loss"])

        t1    = time.time()
        yp_sc = model.predict(Xs_te, verbose=0)
        inf_t = time.time() - t1
        yp    = sc_y.inverse_transform(yp_sc).ravel()

        r2v   = r2_score(yte_aligned, yp)
        maev  = mean_absolute_error(yte_aligned, yp)
        rv    = _rmse(yte_aligned, yp)
        mapev = _mape(yte_aligned, yp)

        # Save history
        with open(str(MDL_DIR / f"{city}_h{h:02d}_history.json"), "w") as _f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in hist.history.items()}, _f)

        # Save predictions
        pd.DataFrame({"y_true": yte_aligned, "y_pred": yp}).to_csv(
            str(MDL_DIR / f"{city}_h{h:02d}_predictions.csv"), index=False)

        # Loss curve
        _ep_x = list(range(1, ep + 1))
        _fig_l, _ax_l = plt.subplots(figsize=(8, 4))
        _fig_l.patch.set_facecolor("#1D1D20"); _ax_l.set_facecolor("#1D1D20")
        _ax_l.plot(_ep_x, hist.history["loss"],     color="#A1C9F4", label="Train loss")
        _ax_l.plot(_ep_x, hist.history["val_loss"], color="#FFB482", label="Val loss")
        _ax_l.set_xlabel("Epoch", color="#fbfbff")
        _ax_l.set_ylabel("MSE Loss", color="#fbfbff")
        _ax_l.set_title(f"Track B BiLSTM — Loss\n{city}  t+{h:02d}h",
                        color="#fbfbff", fontweight="bold")
        _ax_l.legend(facecolor="#1D1D20", labelcolor="#fbfbff")
        _ax_l.tick_params(colors="#909094")
        for _sp in _ax_l.spines.values(): _sp.set_edgecolor("#909094")
        _fig_l.tight_layout()
        _fig_l.savefig(str(FIG_DIR / f"{city}_h{h:02d}_bilstm_loss.png"),
                       dpi=120, facecolor="#1D1D20")
        plt.close(_fig_l)

        # Prediction plot
        _np_plot = min(500, len(yte_aligned))
        _fig_p, _ax_p = plt.subplots(figsize=(12, 4))
        _fig_p.patch.set_facecolor("#1D1D20"); _ax_p.set_facecolor("#1D1D20")
        _ax_p.plot(range(_np_plot), yte_aligned[:_np_plot],
                   color="#8DE5A1", label="Actual",    lw=1)
        _ax_p.plot(range(_np_plot), yp[:_np_plot],
                   color="#FFB482", label="Predicted", lw=1, alpha=0.8)
        _ax_p.set_xlabel("Test Sample", color="#fbfbff")
        _ax_p.set_ylabel("AQI",         color="#fbfbff")
        _ax_p.set_title(f"Track B BiLSTM — Pred vs Actual\n{city}  t+{h:02d}h  "
                        f"R²={r2v:.4f}  MAE={maev:.1f}",
                        color="#fbfbff", fontweight="bold")
        _ax_p.legend(facecolor="#1D1D20", labelcolor="#fbfbff")
        _ax_p.tick_params(colors="#909094")
        for _sp in _ax_p.spines.values(): _sp.set_edgecolor("#909094")
        _fig_p.tight_layout()
        _fig_p.savefig(str(FIG_DIR / f"{city}_h{h:02d}_bilstm_pred.png"),
                       dpi=120, facecolor="#1D1D20")
        plt.close(_fig_p)

        results.append({
            "city":    city,
            "model":   "BiLSTM",
            "horizon": h,
            "R2":      round(r2v,   4),
            "MAE":     round(maev,  2),
            "RMSE":    round(rv,    2),
            "MAPE":    round(mapev, 2),
            "epochs":  ep,
            "train_s": round(tr_t,  1),
        })
        print(f"    t+{h:02d}h  BiLSTM  ep={ep}  R²={r2v:.4f}  "
              f"MAE={maev:.1f}  RMSE={rv:.1f}  MAPE={mapev:.1f}%  t={tr_t:.1f}s")

        # Flush to checkpoint after every city×horizon
        _cur = pd.DataFrame(results)
        if CKPT.exists():
            _cur = pd.concat([pd.read_csv(CKPT), _cur], ignore_index=True)
        _cur.to_csv(str(CKPT), index=False)
        results = []

# ── Final result ──────────────────────────────────────────────────────────────
track_b_bilstm_df = pd.read_csv(CKPT) if CKPT.exists() else pd.DataFrame()

print(f"\n{SEP}")
print("  TRACK B — BiLSTM SUMMARY")
print(SEP)

if not track_b_bilstm_df.empty:
    for h in HORIZONS:
        sub_h = track_b_bilstm_df[track_b_bilstm_df["horizon"] == h]
        if sub_h.empty:
            continue
        avg_r2   = sub_h["R2"].mean()
        avg_mae  = sub_h["MAE"].mean()
        avg_rmse = sub_h["RMSE"].mean()
        best_city  = sub_h.loc[sub_h["R2"].idxmax(), "city"]
        worst_city = sub_h.loc[sub_h["R2"].idxmin(), "city"]
        best_r2    = sub_h["R2"].max()
        worst_r2   = sub_h["R2"].min()
        print(f"  Horizon t+{h:02d}h  |  cities={len(sub_h)}")
        print(f"    Avg R²={avg_r2:.4f}  Avg MAE={avg_mae:.2f}  Avg RMSE={avg_rmse:.2f}")
        print(f"    Best : {best_city}  R²={best_r2:.4f}")
        print(f"    Worst: {worst_city}  R²={worst_r2:.4f}")
    print(f"  Saved → {CKPT}")
else:
    print("  WARNING: no results produced")
print(SEP)
