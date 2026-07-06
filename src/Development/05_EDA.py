
import os, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "outputs/cleaned"
PALETTE = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF',
           '#1F77B4','#9467BD','#8C564B','#C49C94','#E377C2',
           '#F7B6D2','#ffd400','#17b26a','#f04438','#A1C9F4',
           '#FFB482','#8DE5A1','#FF9F9B','#D0BBFF']
BG, TEXT, DIM = '#1D1D20', '#fbfbff', '#909094'
AQI_COLORS = {'Good':'#17b26a','Satisfactory':'#A1C9F4','Moderate':'#ffd400',
              'Poor':'#FFB482','Very Poor':'#f04438','Severe':'#9467BD','Unknown':'#909094'}
AQI_ORDER  = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']

# ─── Load all cleaned parquets ────────────────────────────────────────────────
parquets = sorted(glob.glob(os.path.join(OUT_DIR, "*.parquet")))
city_dfs = {}
for p in parquets:
    city = os.path.basename(p).replace("_cleaned.parquet","").replace("_"," ")
    df   = pd.read_parquet(p)
    if len(df) > 0:
        city_dfs[city] = df

combined_df = pd.concat(
    [df.assign(City=city) for city, df in city_dfs.items()],
    ignore_index=False
)
combined_df.index.name = "Timestamp"
combined_df = combined_df.reset_index()
combined_df["month"]  = combined_df["Timestamp"].dt.month
combined_df["hour"]   = combined_df["Timestamp"].dt.hour
combined_df["year"]   = combined_df["Timestamp"].dt.year
combined_df["season"] = combined_df["month"].map({
    12:"Winter",1:"Winter",2:"Winter",
    3:"Spring",4:"Spring",5:"Spring",
    6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
    10:"Post-Monsoon",11:"Post-Monsoon"
})
cities = sorted(city_dfs.keys())
N = len(cities)
print(f"Loaded {N} cities | {len(combined_df):,} hourly records total")

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelsize=9)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, color=TEXT, fontsize=12, pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=10)
    ax.title.set_color(TEXT)

# ═══════════════════════════════════════════════════════════════════════
# CHART 1: AQI Distribution (histogram) — all cities combined
# ═══════════════════════════════════════════════════════════════════════
fig_aqi_dist, ax = plt.subplots(figsize=(12, 5))
fig_aqi_dist.patch.set_facecolor(BG)
ax.set_facecolor(BG)
vals = combined_df["AQI"].dropna()
ax.hist(vals, bins=80, color='#A1C9F4', edgecolor='none', alpha=0.85)
for lo, hi, col, lbl in [(0,50,'#17b26a','Good'),(50,100,'#A1C9F4','Satisf.'),
    (100,200,'#ffd400','Moderate'),(200,300,'#FFB482','Poor'),
    (300,400,'#f04438','V.Poor'),(400,500,'#9467BD','Severe')]:
    ax.axvspan(lo, hi, alpha=0.08, color=col)
    ax.axvline(lo, color=col, linewidth=0.6, alpha=0.5)
ax.axvline(vals.mean(), color='#ffd400', linewidth=1.5, linestyle='--', label=f'Mean={vals.mean():.0f}')
ax.axvline(vals.median(), color='#FFB482', linewidth=1.5, linestyle=':', label=f'Median={vals.median():.0f}')
ax.legend(facecolor='#2a2a2e', edgecolor='none', labelcolor=TEXT, fontsize=9)
style_ax(ax,"AQI Distribution — All Cities Combined (Hourly)","AQI Value","Frequency")
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 2: Mean AQI per City (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════
city_aqi = combined_df.groupby("City")["AQI"].mean().sort_values(ascending=True)
fig_city_aqi, ax = plt.subplots(figsize=(11, 7))
fig_city_aqi.patch.set_facecolor(BG)
ax.set_facecolor(BG)
colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(city_aqi))]
ax.barh(city_aqi.index, city_aqi.values, color=colors_bar, edgecolor='none')
for i, (city, val) in enumerate(city_aqi.items()):
    ax.text(val + 1.5, i, f"{val:.0f}", va='center', color=TEXT, fontsize=9)
ax.axvline(50,  color='#17b26a', linestyle='--', linewidth=0.9, alpha=0.7, label='Good ≤50')
ax.axvline(100, color='#ffd400', linestyle='--', linewidth=0.9, alpha=0.7, label='Mod ≤100')
ax.axvline(200, color='#FFB482', linestyle='--', linewidth=0.9, alpha=0.7, label='Poor ≤200')
ax.legend(facecolor='#2a2a2e', edgecolor='none', labelcolor=TEXT, fontsize=8, loc='lower right')
style_ax(ax,"Mean AQI by City","Mean AQI (CPCB Sub-Index)","")
ax.tick_params(colors=TEXT, labelsize=9)
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 3: AQI Category stacked bar per city
# ═══════════════════════════════════════════════════════════════════════
cat_counts = combined_df.groupby(["City","AQI_Category"]).size().unstack(fill_value=0)
cat_pct    = cat_counts.div(cat_counts.sum(axis=1), axis=0) * 100
for cat in AQI_ORDER:
    if cat not in cat_pct.columns:
        cat_pct[cat] = 0.0
cat_pct = cat_pct[AQI_ORDER]
cat_pct_sorted = cat_pct.sort_values("Good", ascending=False)

fig_cat_bar, ax = plt.subplots(figsize=(14, 7))
fig_cat_bar.patch.set_facecolor(BG)
ax.set_facecolor(BG)
bottom = np.zeros(len(cat_pct_sorted))
for cat in AQI_ORDER:
    ax.bar(cat_pct_sorted.index, cat_pct_sorted[cat], bottom=bottom,
           color=AQI_COLORS[cat], label=cat, edgecolor='none')
    bottom += cat_pct_sorted[cat].values
ax.set_ylim(0, 100)
ax.legend(facecolor='#2a2a2e', edgecolor='none', labelcolor=TEXT,
          fontsize=9, bbox_to_anchor=(1.01,1), loc='upper left')
ax.set_xticklabels(cat_pct_sorted.index, rotation=35, ha='right', color=TEXT, fontsize=9)
style_ax(ax,"AQI Category Distribution per City (%)","","% of Hours")
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 4: Monthly AQI trend
# ═══════════════════════════════════════════════════════════════════════
monthly = combined_df.groupby("month")["AQI"].agg(["mean","std"]).reset_index()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig_monthly, ax = plt.subplots(figsize=(11, 5))
fig_monthly.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.plot(monthly["month"], monthly["mean"], color='#A1C9F4', linewidth=2, marker='o', ms=5)
ax.fill_between(monthly["month"],
                monthly["mean"] - monthly["std"],
                monthly["mean"] + monthly["std"],
                alpha=0.2, color='#A1C9F4')
ax.set_xticks(range(1,13))
ax.set_xticklabels(month_names, color=DIM, fontsize=10)
style_ax(ax,"Monthly AQI Trend — All Cities Combined","Month","Mean AQI ± 1 SD")
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 5: Hourly AQI diurnal pattern
# ═══════════════════════════════════════════════════════════════════════
hourly = combined_df.groupby("hour")["AQI"].agg(["mean","std"]).reset_index()
fig_hourly, ax = plt.subplots(figsize=(11, 5))
fig_hourly.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.plot(hourly["hour"], hourly["mean"], color='#FFB482', linewidth=2, marker='s', ms=4)
ax.fill_between(hourly["hour"],
                hourly["mean"] - hourly["std"],
                hourly["mean"] + hourly["std"],
                alpha=0.2, color='#FFB482')
ax.set_xticks(range(0,24,2))
ax.axvspan(6,9, alpha=0.07, color='#ffd400', label='Morning rush')
ax.axvspan(17,21,alpha=0.07, color='#FFB482', label='Evening rush')
ax.legend(facecolor='#2a2a2e', edgecolor='none', labelcolor=TEXT, fontsize=9)
style_ax(ax,"Diurnal AQI Pattern — All Cities Combined","Hour of Day","Mean AQI ± 1 SD")
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 6: Seasonal AQI box plot
# ═══════════════════════════════════════════════════════════════════════
season_order = ["Winter","Spring","Monsoon","Post-Monsoon"]
season_data  = [combined_df[combined_df["season"]==s]["AQI"].dropna().values
                for s in season_order]
fig_season, ax = plt.subplots(figsize=(9, 5))
fig_season.patch.set_facecolor(BG)
ax.set_facecolor(BG)
bp = ax.boxplot(season_data, labels=season_order,
                patch_artist=True, medianprops=dict(color='#ffd400', linewidth=2),
                whiskerprops=dict(color=DIM), capprops=dict(color=DIM),
                flierprops=dict(marker='.', color=DIM, markersize=2))
for patch, color in zip(bp['boxes'], ['#A1C9F4','#8DE5A1','#FFB482','#D0BBFF']):
    patch.set_facecolor(color); patch.set_alpha(0.8); patch.set_edgecolor('none')
ax.tick_params(colors=TEXT, labelsize=10)
style_ax(ax,"Seasonal AQI Distribution — All Cities","Season","AQI")
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 7: Yearly AQI trend per city
# ═══════════════════════════════════════════════════════════════════════
yearly = combined_df.groupby(["year","City"])["AQI"].mean().reset_index()
fig_yearly, ax = plt.subplots(figsize=(13, 6))
fig_yearly.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for i, city in enumerate(cities):
    sub = yearly[yearly["City"]==city]
    if len(sub) >= 2:
        ax.plot(sub["year"], sub["AQI"], color=PALETTE[i % len(PALETTE)],
                linewidth=1.4, marker='o', ms=3, label=city, alpha=0.9)
ax.legend(facecolor='#2a2a2e', edgecolor='none', labelcolor=TEXT,
          fontsize=7, bbox_to_anchor=(1.01,1), loc='upper left', ncol=1)
style_ax(ax,"Yearly Mean AQI Trend per City (2010–2025)","Year","Mean AQI")
ax.tick_params(colors=DIM)
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 8: PM2.5 vs AQI scatter
# ═══════════════════════════════════════════════════════════════════════
_samp = combined_df[["PM2.5 (µg/m³)","AQI","City"]].dropna().sample(
    min(30000, len(combined_df)), random_state=42)
fig_scatter, ax = plt.subplots(figsize=(10, 6))
fig_scatter.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for i, city in enumerate(cities):
    sub = _samp[_samp["City"]==city]
    ax.scatter(sub["PM2.5 (µg/m³)"], sub["AQI"],
               color=PALETTE[i % len(PALETTE)], alpha=0.25, s=4, label=city)
ax.set_xlim(0, 500); ax.set_ylim(0, 500)
ax.legend(facecolor='#2a2a2e', edgecolor='none', labelcolor=TEXT,
          fontsize=7, bbox_to_anchor=(1.01,1), loc='upper left', markerscale=3)
style_ax(ax,"PM2.5 vs AQI — Sample (30K pts)","PM2.5 (µg/m³)","AQI")
plt.tight_layout()
plt.close('all')

# ═══════════════════════════════════════════════════════════════════════
# CHART 9: Correlation heatmap
# ═══════════════════════════════════════════════════════════════════════
CORR_COLS = ["PM2.5 (µg/m³)","PM10 (µg/m³)","NO2 (µg/m³)","NH3 (µg/m³)",
             "SO2 (µg/m³)","CO (mg/m³)","Ozone (µg/m³)","AT (°C)","RH (%)","WS (m/s)","AQI"]
_corr_data = combined_df[[c for c in CORR_COLS if c in combined_df.columns]].dropna()
corr_mat   = _corr_data.corr()
cols_short = [c.split(" ")[0] for c in corr_mat.columns]
fig_corr, ax = plt.subplots(figsize=(10, 8))
fig_corr.patch.set_facecolor(BG)
ax.set_facecolor(BG)
im = ax.imshow(corr_mat.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(cols_short))); ax.set_xticklabels(cols_short, rotation=45, ha='right', color=DIM, fontsize=9)
ax.set_yticks(range(len(cols_short))); ax.set_yticklabels(cols_short, color=DIM, fontsize=9)
for i in range(len(cols_short)):
    for j in range(len(cols_short)):
        v = corr_mat.values[i,j]
        ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                color=TEXT if abs(v) < 0.6 else BG, fontsize=7)
cbar = fig_corr.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(colors=DIM, labelsize=8)
style_ax(ax,"Pollutant + Met Correlation Heatmap (Hourly Data)")
for sp in ax.spines.values(): sp.set_visible(False)
plt.tight_layout()
plt.close('all')

print("EDA charts generated:")
for name in ["fig_aqi_dist","fig_city_aqi","fig_cat_bar","fig_monthly",
             "fig_hourly","fig_season","fig_yearly","fig_scatter","fig_corr"]:
    print(f"  {name}")

print(f"\nGlobal AQI: mean={combined_df['AQI'].mean():.1f} std={combined_df['AQI'].std():.1f}")

# Export summary
eda_summary = dict(
    n_cities=N, total_records=len(combined_df),
    aqi_mean=round(float(combined_df["AQI"].mean()), 1),
    aqi_std=round(float(combined_df["AQI"].std()), 1),
    aqi_median=round(float(combined_df["AQI"].median()), 1),
    year_range=(int(combined_df["year"].min()), int(combined_df["year"].max())),
)
print("eda_summary exported ✓")
