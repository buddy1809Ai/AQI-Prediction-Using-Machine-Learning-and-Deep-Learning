import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

OUT = Path("outputs/tables")
REP = Path("outputs/reports")
REP.mkdir(parents=True, exist_ok=True)

SEP = "=" * 72
print(SEP)
print("  EFFECT SIZE ANALYSIS")
print("  Quantifying model improvement gaps — no training")
print(SEP)

ta_path = OUT / "final_track_a_complete.csv"
tb_path = OUT / "final_track_b_complete.csv"
ta = pd.read_csv(ta_path) if ta_path.exists() else pd.DataFrame()
tb = pd.read_csv(tb_path) if tb_path.exists() else pd.DataFrame()

city_col = next((c for c in ["city","City"] if c in ta.columns), "city") if not ta.empty else "city"
ta_city  = ta.groupby(city_col)["R2"].mean() if not ta.empty else pd.Series(dtype=float)
ta_model = ta.groupby("model")[["R2","MAE","RMSE"]].mean() if not ta.empty else pd.DataFrame()
tb_city  = tb.groupby("city")["R2"].mean() if (not tb.empty and "city" in tb.columns) else pd.Series(dtype=float)
tb_model = tb.groupby("model")[["R2","MAE","RMSE"]].mean() if (not tb.empty and "model" in tb.columns) else pd.DataFrame()
tb_hz    = tb.groupby("horizon")["R2"].mean() if (not tb.empty and "horizon" in tb.columns) else pd.Series(dtype=float)

ta_avg = float(ta["R2"].mean()) if not ta.empty else 0.0
tb_avg = float(tb["R2"].mean()) if not tb.empty else 0.0

lines = []

# ── Track A effect sizes ───────────────────────────────────────────────────────
ta_effect_rows = []
if not ta_model.empty and "GradBoost" in ta_model.index:
    gbr_r2   = float(ta_model.loc["GradBoost","R2"])
    gbr_mae  = float(ta_model.loc["GradBoost","MAE"])
    gbr_rmse = float(ta_model.loc["GradBoost","RMSE"])

    TA_ORDER = ["GradBoost","XGBoost","RandomForest","BiLSTM","LSTM","Ridge","CNN-BiLSTM"]
    print(f"\nTRACK A — PAIRWISE EFFECT SIZES (vs GradBoost champion)")
    print("-" * 60)
    for rank, mdl in enumerate([m for m in TA_ORDER if m in ta_model.index], 1):
        r2_m   = float(ta_model.loc[mdl,"R2"])
        mae_m  = float(ta_model.loc[mdl,"MAE"])
        rmse_m = float(ta_model.loc[mdl,"RMSE"])
        delta_r2  = r2_m - gbr_r2
        pct_r2    = (delta_r2 / abs(gbr_r2) * 100) if gbr_r2 != 0 else 0
        delta_mae = mae_m - gbr_mae
        tag = "champion" if mdl == "GradBoost" else (
              "strong" if r2_m > 0.90 else ("moderate" if r2_m > 0.75 else "weak"))
        ta_effect_rows.append({"model": mdl, "avg_R2": round(r2_m,4),
                                "delta_R2_vs_GBR": round(delta_r2,4),
                                "pct_R2_vs_GBR": round(pct_r2,1),
                                "avg_MAE": round(mae_m,2),
                                "delta_MAE_vs_GBR": round(delta_mae,2),
                                "category": tag})
        sym = "★" if mdl == "GradBoost" else " "
        print(f"  {sym} {rank}. {mdl:<18}  R²={r2_m:.4f}  ΔR²={delta_r2:+.4f}  "
              f"({pct_r2:+.1f}%)  [{tag}]")

# ── Track B horizon degradation ────────────────────────────────────────────────
hz_rows = []
if not tb_hz.empty and len(tb_hz) >= 2:
    gbr_tb_r2  = float(tb.loc[tb["model"]=="GradBoost","R2"].mean()) \
                  if "model" in tb.columns else 0.0
    gbr_tb_mae = float(tb.loc[tb["model"]=="GradBoost","MAE"].mean()) \
                  if "model" in tb.columns else 0.0

    TB_ORDER = ["GradBoost","RandomForest","LSTM","CNN-BiLSTM"]
    print(f"\nTRACK B — MODEL × HORIZON DEGRADATION")
    print("-" * 60)

    DL_SET = {"LSTM","BiLSTM","CNN-BiLSTM"}
    CL_SET = {"Ridge","RandomForest","GradBoost","XGBoost"}

    ta_cl = float(ta[ta["model"].isin(CL_SET)]["R2"].mean()) if not ta.empty else 0.0
    ta_dl = float(ta[ta["model"].isin(DL_SET)]["R2"].mean()) if not ta.empty else 0.0
    tb_cl = float(tb[tb["model"].isin(CL_SET)]["R2"].mean()) if not tb.empty and "model" in tb.columns else 0.0
    tb_dl = float(tb[tb["model"].isin(DL_SET)]["R2"].mean()) if not tb.empty and "model" in tb.columns else 0.0

    h1  = float(tb_hz.get(1, 0))
    h6  = float(tb_hz.get(6, 0))
    h24 = float(tb_hz.get(24, 0))
    d16  = h6  - h1
    d124 = h24 - h1
    stability = "GRADUAL" if abs(d16) < 0.15 else "STEEP"

    hz_rows = [
        {"horizon": "1h",  "avg_R2": round(h1,4),  "delta_from_1h": 0.0},
        {"horizon": "6h",  "avg_R2": round(h6,4),  "delta_from_1h": round(d16,4)},
        {"horizon": "24h", "avg_R2": round(h24,4), "delta_from_1h": round(d124,4)},
    ]
    for row in hz_rows:
        print(f"  {row['horizon']:<5}  avg R²={row['avg_R2']:.4f}  "
              f"Δ from 1h={row['delta_from_1h']:+.4f}")
    print(f"  Degradation pattern: {stability}")
    print(f"\n  Track A — Classical R²={ta_cl:.4f}  DL R²={ta_dl:.4f}")
    print(f"  Track B — Classical R²={tb_cl:.4f}  DL R²={tb_dl:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
effect_df = pd.DataFrame(ta_effect_rows)
effect_df.to_csv(OUT / "effect_size_analysis.csv", index=False)

md = [
    "# Effect Size Analysis\n\n",
    "## Track A — Model Comparison vs GradBoost Champion\n\n",
]
if not effect_df.empty:
    for _, row in effect_df.iterrows():
        md.append(f"- **{row['model']}**: R²={row['avg_R2']:.4f}  "
                  f"ΔR²={row['delta_R2_vs_GBR']:+.4f}  [{row['category']}]\n")

md.append(f"\n## Track B — Horizon Degradation\n\n")
for row in hz_rows:
    md.append(f"- {row['horizon']}: avg R²={row['avg_R2']:.4f}  Δ={row['delta_from_1h']:+.4f}\n")

with open(REP / "effect_size_analysis.md", "w") as f:
    f.write("".join(md))

p = REP / "effect_size_analysis.md"
print(f"\n{SEP}")
print(f"  Saved: outputs/tables/effect_size_analysis.csv")
print(f"  Saved: {p}")
