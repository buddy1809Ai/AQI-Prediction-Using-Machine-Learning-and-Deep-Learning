import os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Recovered parquets live in outputs/recovered/*.parquet (not data/processed/recovered)
REC_DIR = Path("outputs/recovered")
OUT_DIR = Path("outputs/tables")
CKPT    = OUT_DIR / "track_a_linear_regression.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET     = "AQI"
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
SEED       = 42

# ── Track A feature candidates — exact column names from recovered parquets ───
# Pollutants: exact names with unit suffixes as present in the parquet columns
SAME_T_POLLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2",
    "CO", "Ozone", "Benzene", "Toluene",
]
# Meteorological: exact names with special characters as they appear in the parquet
MET_COLS = [
    "AT (°C)", "BP (mmHg)", "RH (%)", "WS (m/s)", "WD (deg)", "SR (W/mt2)",
]
# Temporal/cyclical: exact column names confirmed from parquet inspection
TIME_COLS = [
    "hour", "day_of_week", "month", "day_of_year", "is_weekend", "season",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
]
# Interaction / derived features
INTER_COLS = [
    "PM25_PM10_ratio", "NOx_proxy", "CO_PM25_product", "SO2_NO2_sum",
    "wind_u", "wind_v",
]
TRACK_A_CANDIDATES = SAME_T_POLLS + MET_COLS + TIME_COLS + INTER_COLS

# Exclude any AQI-derived feature to avoid leakage
AQI_PFX = ("AQI_", "sub_AQI", "aqi_", "AQI_cat", "AQI_label")

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

# ── Checkpoint: skip already-done cities ─────────────────────────────────────
done_cities: set = set()
if CKPT.exists():
    _prev = pd.read_csv(CKPT)
    if "city" in _prev.columns:
        done_cities = set(_prev["city"].tolist())
        print(f"Checkpoint: {len(done_cities)} cities already done — skipping them.")

# ── Discover parquets ─────────────────────────────────────────────────────────
parquets = sorted(REC_DIR.glob("*.parquet")) if REC_DIR.exists() else []

SEP = "=" * 72
print(SEP)
print("  TRACK A — LINEAR REGRESSION  (18 cities)")
print("  AQI Estimation: same-timestamp features → AQI(t)")
print("  Features: same-t pollutants + met + time/cyclical + interactions")
print("  EXCLUDED: all AQI-derived features (AQI_*, sub_AQI*, aqi_*)")
print("  Split: 70% Train / 15% Val / 15% Test  |  Scaled: MinMaxScaler")
print(SEP)

results = []

for pq in parquets:
    # Strip the _recovered suffix that P1_City_Recovery adds
    city = pq.stem.replace("_recovered", "")

    if city in done_cities:
        print(f"  [SKIP] {city}  (already in checkpoint)")
        continue

    df = pd.read_parquet(pq)

    # Drop any AQI-derived columns (but keep the TARGET itself)
    _aqi_derived = [
        c for c in df.columns
        if any(c.startswith(p) for p in AQI_PFX) and c != TARGET
    ]
    df = df.drop(columns=_aqi_derived, errors="ignore")

    # Build Track A feature set: only what actually exists in this city's parquet
    feat_cols = [c for c in TRACK_A_CANDIDATES if c in df.columns]

    if not feat_cols:
        print(f"  [WARN] {city}: no Track A features found — skipping")
        continue

    needed = feat_cols + [TARGET]
    df_m   = df[needed].dropna()

    n = len(df_m)
    if n < 2000:
        print(f"  [SKIP] {city}: only {n} complete rows after dropna")
        continue

    nt  = int(n * TRAIN_FRAC)
    nv  = int(n * VAL_FRAC)
    nte = n - nt - nv

    X = df_m[feat_cols].values
    y = df_m[TARGET].values

    X_tr = X[:nt];        y_tr = y[:nt]
    X_va = X[nt:nt+nv];   y_va = y[nt:nt+nv]
    X_te = X[nt+nv:];     y_te = y[nt+nv:]

    # Fit scaler on train only — no leakage
    sc      = MinMaxScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_va_s  = sc.transform(X_va)
    X_te_s  = sc.transform(X_te)

    # Train
    t0  = time.time()
    mdl = LinearRegression()
    mdl.fit(X_tr_s, y_tr)
    tr_t = time.time() - t0

    # Evaluate on test set
    t1    = time.time()
    yp_te = mdl.predict(X_te_s)
    inf_t = time.time() - t1

    r2v   = r2_score(y_te, yp_te)
    maev  = mean_absolute_error(y_te, yp_te)
    rmsev = _rmse(y_te, yp_te)

    results.append({
        "city":             city,
        "model":            "LinearRegression",
        "track":            "A",
        "n_rows":           n,
        "n_train":          nt,
        "n_val":            nv,
        "n_test":           nte,
        "n_features":       len(feat_cols),
        "R2":               round(r2v,   4),
        "MAE":              round(maev,  2),
        "RMSE":             round(rmsev, 2),
        "train_time_s":     round(tr_t,  4),
        "inference_time_s": round(inf_t, 6),
    })

    print(f"  [{city}]  n={n:,}  feats={len(feat_cols)}"
          f"  R²={r2v:.4f}  MAE={maev:.1f}  RMSE={rmsev:.1f}  t={tr_t:.2f}s")

# ── Save CSV ──────────────────────────────────────────────────────────────────
track_a_lr_df = pd.DataFrame(results)

if CKPT.exists() and done_cities:
    _old = pd.read_csv(CKPT)
    track_a_lr_df = pd.concat([_old, track_a_lr_df], ignore_index=True)

track_a_lr_df.to_csv(CKPT, index=False)

# ── Summary banner ────────────────────────────────────────────────────────────
valid = track_a_lr_df.dropna(subset=["R2"])
avg_r2   = valid["R2"].mean()
avg_mae  = valid["MAE"].mean()
avg_rmse = valid["RMSE"].mean()
best_city  = valid.loc[valid["R2"].idxmax(), "city"]
worst_city = valid.loc[valid["R2"].idxmin(), "city"]

print(f"\n{SEP}")
print("  TRACK A — LINEAR REGRESSION SUMMARY")
print(SEP)
print(f"  Cities done : {len(valid)}")
print(f"  Avg R²      : {avg_r2:.4f}")
print(f"  Avg MAE     : {avg_mae:.2f}")
print(f"  Avg RMSE    : {avg_rmse:.2f}")
print(f"  Best city   : {best_city}  (R²={valid['R2'].max():.4f})")
print(f"  Worst city  : {worst_city}  (R²={valid['R2'].min():.4f})")
print(f"  Saved → {CKPT}")
print(SEP)
