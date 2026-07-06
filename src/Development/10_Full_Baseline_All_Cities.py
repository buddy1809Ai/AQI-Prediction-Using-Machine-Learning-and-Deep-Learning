import sys, types, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("sklearn loaded ✓")

PALETTE = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4"]
BG, TEXT, DIM = "#1D1D20", "#fbfbff", "#909094"

TARGET     = "AQI"
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
ENG_DIR    = Path("data/processed/recovered")
OUT        = Path("outputs/tables")
FIG_DIR    = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_EXCLUDE    = ["Timestamp","City","city_id","AQI_cat","AQI_label","AQI_category"]
AQI_PFX         = ["AQI_","sub_AQI","aqi_"]
SAME_T_POLLS    = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene"]
MET_COLS        = ["Temperature_C","Humidity_pct","WindSpeed_ms","WindDirection_deg",
                   "SolarRadiation_Wm2","Pressure_hPa"]
TIME_COLS       = ["hour_sin","hour_cos","month_sin","month_cos","hour","month","dayofweek"]
INTER_COLS      = ["PM25_temp","PM25_humid","PM25_wind","NO2_temp"]

parquets    = sorted(ENG_DIR.glob("*.parquet")) if ENG_DIR.exists() else []
all_cities  = [p.stem for p in parquets]
print(f"Cities found: {len(parquets)}")

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

all_results = []

SEP = "=" * 72
print(f"\n{SEP}")
print("  Experiment B — Full Baseline — All 18 Cities")
print(SEP)

for pq in parquets:
    city      = pq.stem
    df        = pd.read_parquet(pq)
    city_slug = city

    drop_cols = (BASE_EXCLUDE +
                 [c for c in df.columns if any(c.startswith(p) for p in AQI_PFX) and c != TARGET])
    fc = [c for c in df.columns
          if c not in drop_cols and c != TARGET
          and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    y  = df[TARGET]
    n  = len(df)
    nt = int(n * TRAIN_FRAC)
    nv = int(n * VAL_FRAC)

    X_tr = df.iloc[:nt][fc].values;       y_tr = y.iloc[:nt].values
    X_va = df.iloc[nt:nt+nv][fc].values;  y_va = y.iloc[nt:nt+nv].values
    X_te = df.iloc[nt+nv:][fc].values;    y_te = y.iloc[nt+nv:].values

    sc  = MinMaxScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    for mname, mdl in [
        ("Ridge",          Ridge(alpha=10.0)),
        ("RandomForest",   RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
        ("GradBoost",      GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)),
    ]:
        Xtr_use = X_tr_s if mname == "Ridge" else X_tr
        Xte_use = X_te_s if mname == "Ridge" else X_te
        mdl.fit(Xtr_use, y_tr)
        yp = mdl.predict(Xte_use)
        r2  = r2_score(y_te, yp)
        mae = mean_absolute_error(y_te, yp)
        rv  = rmse(y_te, yp)
        all_results.append({"city": city, "model": mname,
                             "R2": round(r2,4), "MAE": round(mae,2), "RMSE": round(rv,2),
                             "n_features": len(fc)})

    best_r2 = max(r["R2"] for r in all_results if r["city"] == city)
    print(f"  [{city:<22}]  feats={len(fc):>3}  best_R²={best_r2:.4f}")

baseline_all_df = pd.DataFrame(all_results)
best_per_city   = (baseline_all_df.sort_values("R2", ascending=False)
                   .drop_duplicates("city")[["city","model","R2","MAE","RMSE"]]
                   .reset_index(drop=True))
rank = best_per_city.sort_values("R2", ascending=False).reset_index(drop=True)
rank.index += 1

baseline_weak_cities = best_per_city[best_per_city["R2"] < 0.70]["city"].tolist()

out_all  = OUT / "baseline_all_cities.csv"
out_best = OUT / "best_per_city.csv"
baseline_all_df.to_csv(out_all,  index=False)
best_per_city.to_csv(out_best, index=False)

# ── R² heatmap ────────────────────────────────────────────────────────────────
MODEL_COLS = ["Ridge","RandomForest","GradBoost"]
pivot_r2   = baseline_all_df.pivot_table(index="city", columns="model", values="R2")
pivot_r2   = pivot_r2.reindex(columns=MODEL_COLS)

fig_baseline_r2, ax = plt.subplots(figsize=(10, max(6, len(pivot_r2) * 0.38)),
                                    facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(pivot_r2.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(MODEL_COLS)))
ax.set_xticklabels(MODEL_COLS, color=TEXT, fontsize=10)
ax.set_yticks(range(len(pivot_r2)))
ax.set_yticklabels(pivot_r2.index, color=TEXT, fontsize=9)
for i in range(len(pivot_r2)):
    for j, mc in enumerate(MODEL_COLS):
        v = pivot_r2.iloc[i, j] if mc in pivot_r2.columns else float("nan")
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if v > 0.5 else "white", fontsize=8)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
ax.set_title("Baseline Model R² by City", color=TEXT, fontsize=13, pad=12)
sp = ax.spines
for s in sp.values():
    s.set_color(DIM)
fig_baseline_r2.tight_layout()
fig_baseline_r2.savefig(FIG_DIR / "baseline_r2_heatmap.png", dpi=120,
                         bbox_inches="tight", facecolor=BG)
plt.close(fig_baseline_r2)

# ── Best model bar ─────────────────────────────────────────────────────────────
fig_best_bar, ax2 = plt.subplots(figsize=(10, max(5, len(rank) * 0.35)), facecolor=BG)
ax2.set_facecolor(BG)
bars2 = ax2.barh(rank["city"], rank["R2"], color=PALETTE[0])
for bar in bars2:
    rv = bar.get_width()
    ax2.text(rv + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{rv:.3f}", va="center", color=TEXT, fontsize=8)
ax2.set_xlim(0, 1.1)
ax2.set_xlabel("R² (test set)", color=TEXT)
ax2.set_title("Best Model R² per City (Baseline)", color=TEXT, fontsize=12, pad=10)
ax2.tick_params(colors=TEXT)
for sp_ in ax2.spines.values():
    sp_.set_color(DIM)
fig_best_bar.tight_layout()
fig_best_bar.savefig(FIG_DIR / "baseline_best_per_city.png", dpi=120,
                      bbox_inches="tight", facecolor=BG)
plt.close(fig_best_bar)

baseline_summary = {
    "n_cities": len(parquets),
    "avg_best_r2": round(best_per_city["R2"].mean(), 4),
    "weak_cities": baseline_weak_cities,
}

# Load leakage experiment data from expB
expB_feats = None
pre_loaded = {}

VALID_DT = ["float64","float32","int64","int32","int8","int16"]
flag = None
ntv  = None
X    = None
sc   = None
Xs   = None
Xtv  = None
ytv  = None
Xte  = None
yte  = None
yp   = None
r2   = None
mae  = None
rms  = None
leakage_exp_df = None

print(f"\n{SEP}")
print(f"  Baseline complete — {len(parquets)} cities × 3 models")
print(f"  Average best R²: {baseline_summary['avg_best_r2']:.4f}")
print(f"  Weak cities (<0.70): {baseline_weak_cities}")
print(f"  Saved: baseline_all_cities.csv  |  best_per_city.csv")
