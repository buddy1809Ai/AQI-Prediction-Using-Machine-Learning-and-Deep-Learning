import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

warnings.filterwarnings("ignore")

BG, TEXT, DIM = "#1D1D20", "#fbfbff", "#909094"
PAL  = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4","#9467BD"]
GOLD, GREEN, WARN = "#ffd400", "#17b26a", "#f04438"

OUT = Path("outputs/tables")
FIG = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEP = "=" * 72

def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT)
    for sp in ax.spines.values():
        sp.set_color(DIM)

# ── Load Track A ───────────────────────────────────────────────────────────────
print(SEP); print("  LOADING TRACK A"); print(SEP)
ta_map = {
    "Ridge":      OUT / "track_a_ridge.csv",
    "RandomForest": OUT / "track_a_rf.csv",
    "GradBoost":  OUT / "track_a_gbr.csv",
    "XGBoost":    OUT / "track_a_xgb.csv",
    "LSTM":       OUT / "track_a_lstm.csv",
    "BiLSTM":     OUT / "track_a_bilstm.csv",
    "CNN-BiLSTM": OUT / "track_a_cnn_bilstm.csv",
}
ta_dfs = {}
for mdl, fp in ta_map.items():
    d = load_csv_safe(fp)
    if not d.empty:
        d["model"] = mdl
        ta_dfs[mdl] = d
        print(f"  ✓ {mdl:<15} {len(d)} rows")
    else:
        print(f"  ✗ {mdl:<15} (not found)")

track_a_all = pd.concat(list(ta_dfs.values()), ignore_index=True) if ta_dfs else pd.DataFrame()

# ── Load Track B ───────────────────────────────────────────────────────────────
print(f"\n{SEP}"); print("  LOADING TRACK B"); print(SEP)
tb_map = {
    "GradBoost":  OUT / "track_b_gbr.csv",
    "RandomForest": OUT / "track_b_rf.csv",
    "LSTM":       OUT / "track_b_lstm.csv",
    "CNN-BiLSTM": OUT / "track_b_cnn_bilstm.csv",
}
tb_dfs = {}
for mdl, fp in tb_map.items():
    d = load_csv_safe(fp)
    if not d.empty:
        d["model"] = mdl
        tb_dfs[mdl] = d
        print(f"  ✓ {mdl:<15} {len(d)} rows")

track_b_all = pd.concat(list(tb_dfs.values()), ignore_index=True) if tb_dfs else pd.DataFrame()

# ── Rankings ───────────────────────────────────────────────────────────────────
if not track_a_all.empty:
    ta_rank = (track_a_all.groupby("model")[["R2","MAE","RMSE"]].mean()
               .sort_values("R2", ascending=False).reset_index())
    ta_rank.index += 1
    print(f"\n  TRACK A Rankings:")
    for i, r in ta_rank.iterrows():
        print(f"    {i}. {r['model']:<15}  R²={r['R2']:.4f}  MAE={r['MAE']:.1f}")
    ta_rank.to_csv(OUT / "track_a_model_ranking.csv", index=False)

if not track_b_all.empty:
    h_col = next((c for c in ["horizon","Horizon"] if c in track_b_all.columns), None)
    if h_col:
        tb_rank = (track_b_all.groupby(["model", h_col])[["R2","MAE","RMSE"]].mean()
                   .reset_index().sort_values([h_col,"R2"], ascending=[True,False]))
        tb_rank.to_csv(OUT / "track_b_model_ranking.csv", index=False)
        horiz_rank = (track_b_all[track_b_all[h_col]==1].groupby("model")["R2"].mean()
                      .sort_values(ascending=False))
        print(f"\n  TRACK B Rankings (1-hour horizon):")
        for rank_, (mdl, r2) in enumerate(horiz_rank.items(), 1):
            print(f"    {rank_}. {mdl:<15}  R²={r2:.4f}")
        horiz_rank.to_csv(OUT / "track_b_horizon_ranking.csv")

# ── Comparison figures ─────────────────────────────────────────────────────────
models_ord = list(ta_dfs.keys())

if not track_a_all.empty and len(models_ord) >= 2:
    fig_track_a_comparison, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    for ax, col_, ttl in zip(axes, ["R2","MAE","RMSE"],
                              ["R²","MAE","RMSE"]):
        ax.set_facecolor(BG)
        ci = [track_a_all[track_a_all["model"]==m][col_].mean() for m in models_ord]
        ci = [v if not np.isnan(v) else 0 for v in ci]
        vals  = ci
        cb    = PAL[:len(models_ord)]
        bars  = ax.bar(models_ord, vals, color=cb)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.3f}", ha="center", color=TEXT, fontsize=7)
        ax.set_title(f"Track A — {ttl}", color=TEXT, fontsize=11)
        ax.set_xticklabels(models_ord, rotation=30, ha="right", color=TEXT, fontsize=8)
        style_ax(ax)
    fig_track_a_comparison.tight_layout()
    fig_track_a_comparison.savefig(FIG / "fig1_track_a_model_comparison.png", dpi=120,
                                    bbox_inches="tight", facecolor=BG)
    plt.close(fig_track_a_comparison)
    print(f"\n  Saved: fig1_track_a_model_comparison.png")

if not track_b_all.empty:
    models_b = list(tb_dfs.keys())
    h_col    = next((c for c in ["horizon","Horizon"] if c in track_b_all.columns), None)
    if h_col:
        fig_track_b_comparison, axes2 = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
        for ax, col_, ttl in zip(axes2, ["R2","MAE","RMSE"], ["R²","MAE","RMSE"]):
            ax.set_facecolor(BG)
            for mi, mdl in enumerate(models_b):
                sub   = track_b_all[(track_b_all["model"]==mdl)].groupby(h_col)[col_].mean()
                y_vals = [sub.get(h, np.nan) for h in [1,6,24]]
                ax.plot([1,6,24], y_vals, marker="o", label=mdl, color=PAL[mi % len(PAL)])
            ax.set_title(f"Track B — {ttl} by Horizon", color=TEXT, fontsize=11)
            ax.set_xlabel("Horizon (h)", color=TEXT)
            ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
            style_ax(ax)
        fig_track_b_comparison.tight_layout()
        fig_track_b_comparison.savefig(FIG / "fig2_track_b_model_comparison.png", dpi=120,
                                        bbox_inches="tight", facecolor=BG)
        plt.close(fig_track_b_comparison)
        print(f"  Saved: fig2_track_b_model_comparison.png")

        fig_horizon_degradation, ax3 = plt.subplots(figsize=(9, 5), facecolor=BG)
        ax3.set_facecolor(BG)
        for mi, mdl in enumerate(models_b):
            sub = track_b_all[track_b_all["model"]==mdl].groupby(h_col)["R2"].mean()
            h_vals = [sub.get(h, np.nan) for h in [1,6,24]]
            h_lbls = ["1h","6h","24h"]
            ax3.plot([1,6,24], h_vals, marker="s", label=mdl, color=PAL[mi % len(PAL)], lw=2)
        ax3.set_xlabel("Forecast Horizon", color=TEXT)
        ax3.set_ylabel("R²", color=TEXT)
        ax3.set_title("Horizon Degradation — Track B Models", color=TEXT, fontsize=12)
        ax3.set_xticks([1,6,24])
        ax3.set_xticklabels(["1h","6h","24h"], color=TEXT)
        ax3.legend(facecolor=BG, labelcolor=TEXT)
        style_ax(ax3)
        fig_horizon_degradation.tight_layout()
        fig_horizon_degradation.savefig(FIG / "fig3_horizon_degradation.png", dpi=120,
                                         bbox_inches="tight", facecolor=BG)
        plt.close(fig_horizon_degradation)
        print(f"  Saved: fig3_horizon_degradation.png")

# ── Classical vs DL comparison ─────────────────────────────────────────────────
DL_SET  = {"LSTM","BiLSTM","CNN-BiLSTM"}
CL_SET  = {"Ridge","RandomForest","GradBoost","XGBoost"}
if not track_a_all.empty:
    ta_cl = track_a_all[track_a_all["model"].isin(CL_SET)]["R2"].mean()
    ta_dl = track_a_all[track_a_all["model"].isin(DL_SET)]["R2"].mean()
    dl_detail = {m: track_a_all[track_a_all["model"]==m]["R2"].mean() for m in DL_SET}
    print(f"\n  Track A — Classical avg R²={ta_cl:.4f}  DL avg R²={ta_dl:.4f}")

track_a_all.to_csv(OUT / "final_comparison.csv", index=False)

comparison_summary = {
    "track_a_n_models": int(track_a_all["model"].nunique()) if not track_a_all.empty else 0,
    "track_b_n_models": int(track_b_all["model"].nunique()) if not track_b_all.empty else 0,
    "best_track_a_model": str(ta_rank.iloc[0]["model"]) if not track_a_all.empty else "—",
    "best_track_a_r2": float(ta_rank.iloc[0]["R2"]) if not track_a_all.empty else 0.0,
}
with open(OUT / "comparison_summary.json", "w") as fh:
    json.dump(comparison_summary, fh, indent=2)

print(f"\n{SEP}")
print(f"  Model comparison complete.")
print(f"  Saved: final_comparison.csv  |  comparison_summary.json")
