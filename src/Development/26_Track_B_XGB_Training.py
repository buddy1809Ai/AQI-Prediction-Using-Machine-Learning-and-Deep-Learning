import os, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths & constants ───────────────────────────────────────────────────────
REC_DIR  = Path("outputs/recovered")       # actual location of *_recovered.parquet
OUT_DIR  = Path("outputs/tables")
MDL_DIR  = Path("outputs/models/track_b_xgb")
FIG_DIR  = Path("outputs/figures")
CKPT     = OUT_DIR / "track_b_xgb_training.csv"

for _d in [OUT_DIR, MDL_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

TARGET   = "AQI"
TRAIN    = 0.70
VAL      = 0.15
SEED     = 42
HORIZONS = [1, 6, 24]
SEP      = "=" * 72

# ── Actual column names in the recovered parquets ────────────────────────────
# Lag cols:  PM25_lag1h, PM10_lag6h, NO2_lag12h … (contain '_lag')
# Roll cols: PM25_roll_mean_6h, PM10_roll_mean_24h … (contain '_roll_mean')
# Met cols:  actual raw names from parquet
MET_COLS  = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "SR (W/mt2)", "BP (mmHg)"]
TIME_COLS = ["hour", "month", "day_of_week", "season", "is_weekend",
             "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
INTER_COLS = ["wind_u", "wind_v"]

# AQI-derived prefixes to exclude
AQI_PFX = ("AQI_", "sub_AQI", "aqi_")

# ── Helpers ──────────────────────────────────────────────────────────────────
def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def _mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# ── Resume from checkpoint ───────────────────────────────────────────────────
done_keys = set()
if CKPT.exists():
    _ck       = pd.read_csv(CKPT)
    done_keys = set(zip(_ck["city"], _ck["horizon"]))

results  = []
parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []

print(SEP)
print("  TRACK B — XGBoost Training  (18 cities × 3 horizons)")
print("  Features: lags + rolling + met + time")
print("  EXCLUDED: same-t pollutants, all AQI-derived features")
print("  Saves: models / predictions / evaluation CSV / feature importance")
print(SEP)
print(f"  Parquets found: {len(parquets)}  |  REC_DIR: {REC_DIR}")

for pq in parquets:
    # Files are named Ahmedabad_recovered.parquet → city = "Ahmedabad"
    city = pq.stem.replace("_recovered", "")
    df   = pd.read_parquet(pq)

    # Build leakage-free Track B feature set using correct substring patterns
    lag_feats  = [c for c in df.columns if "_lag" in c]
    roll_feats = [c for c in df.columns if "_roll_mean" in c]
    met_avail  = [c for c in MET_COLS   if c in df.columns]
    time_avail = [c for c in TIME_COLS  if c in df.columns]
    inter_avail = [c for c in INTER_COLS if c in df.columns]
    feat_cols  = lag_feats + roll_feats + met_avail + time_avail + inter_avail

    df_m      = df[[c for c in feat_cols + [TARGET] if c in df.columns]].copy()
    feat_cols = [c for c in feat_cols if c in df_m.columns]
    n_feats   = len(feat_cols)
    n         = len(df_m)
    print(f"\n  [{city}]  n={n:,}  feats={n_feats}")

    if n_feats == 0:
        print(f"    WARN: no features found — skipping")
        continue

    # Accumulate importances across horizons for summary plot
    _imp_sum = np.zeros(n_feats)

    for h in HORIZONS:
        key = (city, h)
        if key in done_keys:
            print(f"    t+{h:02d}h  XGB  (checkpoint — skipped)")
            continue

        y_shifted = df_m[TARGET].shift(-h)
        valid     = df_m[feat_cols].notna().all(axis=1) & y_shifted.notna()
        Xv = df_m.loc[valid, feat_cols].values
        yv = y_shifted[valid].values
        ntr = int(len(Xv) * TRAIN)
        nvl = int(len(Xv) * VAL)

        if len(Xv) - ntr - nvl < 200:
            print(f"    t+{h:02d}h  XGB  (skipped — insufficient test rows)")
            continue

        X_tr = Xv[:ntr];              y_tr = yv[:ntr]
        X_va = Xv[ntr:ntr+nvl];       y_va = yv[ntr:ntr+nvl]
        X_te = Xv[ntr+nvl:];          y_te = yv[ntr+nvl:]

        # XGBoost requires safe feature names (no special chars like °, %, /)
        safe_names = [f"f{i}" for i in range(n_feats)]
        X_tr_df = pd.DataFrame(X_tr, columns=safe_names)
        X_va_df = pd.DataFrame(X_va, columns=safe_names)
        X_te_df = pd.DataFrame(X_te, columns=safe_names)

        t0  = time.time()
        mdl = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, eval_metric="rmse",
            verbosity=0, random_state=SEED,
        )
        mdl.fit(X_tr_df, y_tr,
                eval_set=[(X_va_df, y_va)],
                verbose=False)
        tr_t = time.time() - t0

        # ── Evaluate on all three splits ─────────────────────────────────
        t1    = time.time()
        yp_te = mdl.predict(X_te_df)
        inf_t = time.time() - t1

        yp_tr = mdl.predict(X_tr_df)
        yp_va = mdl.predict(X_va_df)

        r2_tr = r2_score(y_tr, yp_tr);      mae_tr = mean_absolute_error(y_tr, yp_tr)
        r2_va = r2_score(y_va, yp_va);      mae_va = mean_absolute_error(y_va, yp_va)
        r2_te = r2_score(y_te, yp_te);      mae_te = mean_absolute_error(y_te, yp_te)
        rmse_te = _rmse(y_te, yp_te)
        mape_te = _mape(y_te, yp_te)

        # ── Save model ───────────────────────────────────────────────────
        mdl_path = MDL_DIR / f"{city}_h{h:02d}_xgb.pkl"
        with open(mdl_path, "wb") as _f:
            pickle.dump({"model": mdl, "feat_cols": feat_cols,
                         "safe_names": safe_names}, _f)

        # ── Save predictions ─────────────────────────────────────────────
        pred_path = MDL_DIR / f"{city}_h{h:02d}_predictions.csv"
        pd.DataFrame({"y_true": y_te, "y_pred": yp_te}).to_csv(pred_path, index=False)

        # ── Accumulate feature importances ───────────────────────────────
        _imp_sum += mdl.feature_importances_

        results.append({
            "city": city, "model": "XGBoost", "horizon": h,
            "R2_train": round(r2_tr, 4), "MAE_train": round(mae_tr, 2),
            "R2_val":   round(r2_va, 4), "MAE_val":   round(mae_va, 2),
            "R2_test":  round(r2_te, 4), "MAE_test":  round(mae_te, 2),
            "RMSE_test": round(rmse_te, 2), "MAPE_test": round(mape_te, 2),
            "train_s":  round(tr_t, 1),
        })
        print(f"    t+{h:02d}h  XGB  "
              f"R²(tr={r2_tr:.3f} va={r2_va:.3f} te={r2_te:.4f})  "
              f"MAE={mae_te:.1f}  RMSE={rmse_te:.1f}  MAPE={mape_te:.1f}%  "
              f"t={tr_t:.1f}s")

    # ── Feature importance plot (averaged across horizons) ───────────────
    if _imp_sum.sum() > 0 and n_feats > 0:
        top_n    = min(20, n_feats)
        top_idx  = np.argsort(_imp_sum)[-top_n:][::-1]
        top_imp  = _imp_sum[top_idx] / (_imp_sum[top_idx].sum() + 1e-12)
        top_nms  = [feat_cols[i] for i in top_idx]

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        fig_imp.patch.set_facecolor("#1D1D20")
        ax_imp.set_facecolor("#1D1D20")
        ax_imp.barh(range(top_n), top_imp[::-1], color="#A1C9F4")
        ax_imp.set_yticks(range(top_n))
        ax_imp.set_yticklabels(top_nms[::-1], color="#fbfbff", fontsize=8)
        ax_imp.set_xlabel("Normalised Importance", color="#fbfbff")
        ax_imp.set_title(f"Track B XGBoost — Feature Importance\n{city}",
                         color="#fbfbff", fontweight="bold")
        ax_imp.tick_params(colors="#909094")
        for sp in ax_imp.spines.values():
            sp.set_edgecolor("#909094")
        fig_imp.tight_layout()
        fig_imp.savefig(FIG_DIR / f"{city}_track_b_xgb_feat_imp.png",
                        dpi=120, facecolor="#1D1D20")
        plt.close(fig_imp)

# ── Merge with checkpoint and persist ────────────────────────────────────────
track_b_xgb_train_df = pd.DataFrame(results)
if CKPT.exists():
    _old = pd.read_csv(CKPT)
    track_b_xgb_train_df = pd.concat([_old, track_b_xgb_train_df], ignore_index=True)
track_b_xgb_train_df.to_csv(CKPT, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
if not track_b_xgb_train_df.empty:
    _sub = track_b_xgb_train_df.groupby("horizon")[["R2_test", "MAE_test", "RMSE_test"]].mean()
    print(f"  TRACK B — XGBoost Training Complete")
    print(f"  Rows saved : {len(track_b_xgb_train_df)} (cities × horizons)")
    print(f"  Eval CSV   : {CKPT}")
    print(f"  Models     : {MDL_DIR}")
    print(f"  Figures    : {FIG_DIR}")
    print(f"\n  {'Horizon':>8}  {'R²(test)':>10}  {'MAE':>8}  {'RMSE':>8}")
    print(f"  {'-'*40}")
    for hz in HORIZONS:
        if hz in _sub.index:
            _r = _sub.loc[hz]
            print(f"  t+{hz:02d}h     {_r['R2_test']:>10.4f}  {_r['MAE_test']:>8.2f}  {_r['RMSE_test']:>8.2f}")
    _best  = track_b_xgb_train_df.loc[track_b_xgb_train_df["R2_test"].idxmax()]
    _worst = track_b_xgb_train_df.loc[track_b_xgb_train_df["R2_test"].idxmin()]
    print(f"\n  Best  : {_best['city']}  t+{int(_best['horizon'])}h  R²={_best['R2_test']:.4f}")
    print(f"  Worst : {_worst['city']}  t+{int(_worst['horizon'])}h  R²={_worst['R2_test']:.4f}")
print(SEP)
