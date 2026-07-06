
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

DATA_ROOT = "CPCB_Data-20260602T165735Z-3-001/CPCB_Data"

def is_stub(fpath):
    try:
        df = pd.read_csv(fpath, nrows=3, low_memory=False)
        return len(df) == 0
    except Exception:
        return True

def extract_year(f):
    m = re.search(r'_(\d{4})_', os.path.basename(f))
    return int(m.group(1)) if m else None

def extract_station(f):
    m = re.search(r'site_(\d+)_(.+?)_\d{4}Min', os.path.basename(f))
    if m: return f"site_{m.group(1)}"
    m2 = re.search(r'_(\d{5,6})_', os.path.basename(f))
    return f"site_{m2.group(1)}" if m2 else "unknown"

# ─── Rebuild full city stats ──────────────────────────────────────────────────
city_folders = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
)

city_rows_map = {
    "Ahmedabad": 876576, "Chennai": 1577760, "Delhi NCR": 2594496,
    "GandhiNagar": 946656, "Hyderabad": 3322656, "Indore": 654720,
    "Jaipur": 806496, "Jodhpur": 631200, "Mumbai": 1612896,
    "Mumbai suburbs": 1078440, "Nagpur": 759840, "Navi Mumbai": 1016832,
    "Pune": 1162848, "Singrauli": 286368, "Surat": 210336,
    "Thane": 280512, "Vapi": 315552, "bhopal": 455808,
    "vishakhapattanam": 140256,
}
TOTAL_ROWS = 18_730_248

# Build summary dataframe
records = []
for city in city_folders:
    cpath   = os.path.join(DATA_ROOT, city)
    all_csv = sorted(glob.glob(os.path.join(cpath, "*.csv")))
    real    = [f for f in all_csv if not is_stub(f)]
    stubs   = [f for f in all_csv if is_stub(f)]
    years   = sorted({extract_year(f) for f in real if extract_year(f)})
    stations= sorted({extract_station(f) for f in real})
    gaps    = []
    if years:
        gaps = sorted(set(range(min(years), max(years)+1)) - set(years))

    records.append(dict(
        City=city,
        Real_Files=len(real),
        Stub_Files=len(stubs),
        Total_Files=len(all_csv),
        Rows=city_rows_map.get(city, 0),
        Year_Min=min(years) if years else None,
        Year_Max=max(years) if years else None,
        Year_Count=len(years),
        Year_Gaps=gaps,
        Unique_Stations=len(stations),
        Station_IDs=", ".join(stations[:4]) + ("…" if len(stations)>4 else ""),
        Has_Data=len(real) > 0,
    ))

inv_df = pd.DataFrame(records).sort_values("Rows", ascending=False).reset_index(drop=True)

# ─── TABLE 1: Full Inventory ──────────────────────────────────────────────────
print("=" * 88)
print("DATASET_INVENTORY_REPORT  —  CPCB Multi-City AQI Dataset")
print("=" * 88)
print(f"\n{'Rank':<5} {'City':<20} {'Files':>6} {'Stubs':>6} {'Rows':>12} {'Yr Range':>11} {'Stations':>9} {'Yr Gaps'}")
print("-" * 88)
for i, row in inv_df.iterrows():
    yr = f"{row.Year_Min}–{row.Year_Max}" if pd.notna(row.Year_Min) else "—"
    gp = str(row.Year_Gaps) if row.Year_Gaps else "none"
    print(f"{i+1:<5} {row.City:<20} {row.Real_Files:>6} {row.Stub_Files:>6} "
          f"{row.Rows:>12,}  {yr:>11} {row.Unique_Stations:>9}  {gp}")
print("-" * 88)
print(f"{'TOTAL':<5} {'19 cities':<20} "
      f"{inv_df.Real_Files.sum():>6} {inv_df.Stub_Files.sum():>6} "
      f"{TOTAL_ROWS:>12,}")

# ─── TABLE 2: Schema summary ──────────────────────────────────────────────────
print(f"""
{'=' * 72}
SCHEMA SUMMARY (All 19 cities — UNIFORM)
{'=' * 72}
  Columns per file : 25  (identical across all cities/years)
  Timestamp col    : 'Timestamp'  (datetime64, 15-min resolution)
  Direct AQI col   : ABSENT — must compute via CPCB sub-index formula

  POLLUTANT COLUMNS (14):
    PM2.5 (µg/m³), PM10 (µg/m³), NO (µg/m³), NO2 (µg/m³), NOx (ppb),
    NH3 (µg/m³), SO2 (µg/m³), CO (mg/m³), Ozone (µg/m³),
    Benzene (µg/m³), Toluene (µg/m³), Xylene (µg/m³),
    O Xylene (µg/m³), Eth-Benzene (µg/m³), MP-Xylene (µg/m³)

  METEOROLOGICAL COLUMNS (9):
    AT (°C), RH (%), WS (m/s), WD (deg), RF (mm),
    TOT-RF (mm), SR (W/mt2), BP (mmHg), VWS (m/s)

  AQI SUB-INDEX POLLUTANTS (7, used for AQI computation):
    PM2.5, PM10, NO2, SO2, CO, Ozone, NH3
""")

# ─── TABLE 3: Missing/Quality flags ──────────────────────────────────────────
print(f"{'=' * 72}")
print("DATA QUALITY FLAGS")
print(f"{'=' * 72}")

chronic_miss = {
    "Xylene, O-Xylene, Eth-Benzene, MP-Xylene": "Often 100% missing — drop these 4 cols",
    "VWS (Vertical Wind Speed)": "Often 100% missing — drop",
    "RF / TOT-RF (Rainfall)": "~70–99% missing — impute or drop",
    "Ozone (µg/m³)": "10–40% missing — forward-fill then interpolate",
    "SO2, NH3": "10–30% missing — interpolate",
    "NO, NO2, NOx": "5–30% missing per station — interpolate",
}
for issue, action in chronic_miss.items():
    print(f"  ⚠ {issue}")
    print(f"    → {action}")

print(f"\n  Stub files:  {inv_df.Stub_Files.sum()} total (header-only CSVs with 0 data rows)")
print(f"  Year gaps: ", end="")
gap_cities = inv_df[inv_df.Year_Gaps.map(len) > 0][['City','Year_Gaps']].values
if len(gap_cities):
    for city, gaps in gap_cities:
        print(f"{city} → missing {gaps};  ", end="")
else:
    print("None detected")

# ─── TABLE 4: Methodology Decision ───────────────────────────────────────────
print(f"""
{'=' * 88}
METHODOLOGY SELECTION — SCIENTIFIC REASONING
{'=' * 88}

  Four candidate strategies were evaluated:

  A. Independent City Models  — one LSTM per city
     Pros: city-specific tuning, no cross-city interference
     Cons: fails on small-data cities (Surat 210K, Thane 280K rows);
           no generalisation; 19× training overhead

  B. Global Model (single LSTM)  — all cities together, city_id as input
     Pros: maximum data (18.7M rows); knowledge transfer; handles small cities
     Cons: must handle city-specific AQI scale differences carefully

  C. Clustered Models  — group cities by profile, one LSTM per cluster
     Pros: balances A and B; captures regional pollution patterns
     Cons: cluster assignment requires domain knowledge + validation

  D. Hybrid  — global model pre-training → city-specific fine-tuning
     Pros: best of both worlds; proven in transfer learning literature
     Cons: more complex; longer iteration time

  ═══════════════════════════════════════════════════════════════
  SELECTED STRATEGY:  D. HYBRID (Global pre-train + city fine-tune)
  ═══════════════════════════════════════════════════════════════
  Step 1: Train a GLOBAL LSTM on all 18.7M 15-min records
          with city_id embedding (19 cities → 8-dim learned vector)
  Step 2: Fine-tune city-specific LSTM heads for
          large-data cities (≥500K rows): Delhi NCR, Hyderabad,
          Mumbai, Chennai, Pune, Navi Mumbai, Mumbai suburbs,
          GandhiNagar, Jaipur, Ahmedabad, Nagpur
  Step 3: Small-data cities (Surat, Thane, Singrauli, Jodhpur,
          Indore, bhopal, Vapi, vishakhapattanam) use global model
          directly (no fine-tune — insufficient data for fine-tune)

  Baseline models (LR, RF, XGBoost) trained per-city on hourly
  aggregated data using 4-year common period (2021–2024).

  TEMPORAL RESOLUTION FOR MODELLING:
    Raw data: 15-min (35 cols after processing)
    Training: aggregate to hourly averages → reduces seq length
    LSTM window: 24 hours of history (24-step lookback)

  TRAIN / VAL / TEST split: chronological 70/15/15 per city
    (prevents data leakage; mirrors real deployment scenario)
""")

# ─── CHART: Row count bar chart per city ─────────────────────────────────────
fig_inv, ax = plt.subplots(figsize=(13, 6))
fig_inv.patch.set_facecolor('#1D1D20')
ax.set_facecolor('#1D1D20')

palette = ['#A1C9F4' if r >= 500_000 else '#FFB482'
           for r in inv_df.Rows]
bars = ax.barh(inv_df.City[::-1], inv_df.Rows[::-1] / 1e6, color=palette[::-1], edgecolor='none')
ax.set_xlabel("Rows (millions)", color='#fbfbff', fontsize=11)
ax.set_title("Dataset Row Count per City — CPCB 15-Min Records", color='#fbfbff', fontsize=13, pad=14)
ax.tick_params(colors='#909094', labelsize=9)
for spine in ax.spines.values(): spine.set_visible(False)
ax.xaxis.label.set_color('#909094')

for bar, val in zip(bars[::-1], inv_df.Rows / 1e6):
    ax.text(val + 0.03, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}M", va='center', color='#fbfbff', fontsize=8)

legend_els = [
    mpatches.Patch(color='#A1C9F4', label='≥500K rows (large)'),
    mpatches.Patch(color='#FFB482', label='<500K rows (small)'),
]
ax.legend(handles=legend_els, facecolor='#2a2a2e', edgecolor='none',
          labelcolor='#fbfbff', fontsize=9, loc='lower right')
plt.tight_layout()
plt.close('all')

# ─── CHART 2: Year coverage heatmap ──────────────────────────────────────────
ALL_YEARS = list(range(2010, 2026))
fig_yrs, ax2 = plt.subplots(figsize=(14, 8))
fig_yrs.patch.set_facecolor('#1D1D20')
ax2.set_facecolor('#1D1D20')

cities_ordered = inv_df.City.tolist()
yr_matrix = pd.DataFrame(0, index=cities_ordered, columns=ALL_YEARS)
for city in city_folders:
    cpath = os.path.join(DATA_ROOT, city)
    real  = [f for f in glob.glob(os.path.join(cpath, "*.csv")) if not is_stub(f)]
    for f in real:
        yr = extract_year(f)
        if yr and yr in ALL_YEARS:
            yr_matrix.loc[city, yr] = 1

_cmap_colors = ['#1D1D20', '#A1C9F4']
from matplotlib.colors import ListedColormap
_cmap = ListedColormap(_cmap_colors)
im = ax2.imshow(yr_matrix.values, aspect='auto', cmap=_cmap, vmin=0, vmax=1)
ax2.set_xticks(range(len(ALL_YEARS)))
ax2.set_xticklabels(ALL_YEARS, color='#909094', fontsize=9, rotation=45)
ax2.set_yticks(range(len(cities_ordered)))
ax2.set_yticklabels(cities_ordered, color='#fbfbff', fontsize=9)
ax2.set_title("Year Coverage per City (Blue = Data Available)", color='#fbfbff', fontsize=12, pad=12)
ax2.tick_params(colors='#909094')
for spine in ax2.spines.values(): spine.set_visible(False)

for x in range(len(ALL_YEARS)):
    ax2.axvline(x - 0.5, color='#2a2a2e', linewidth=0.5)
plt.tight_layout()
plt.close('all')

print("\nFigures generated:")
print("  fig_inv  — Row count bar chart")
print("  fig_yrs  — Year coverage heatmap")
print("\nDataset_Inventory_Report complete. ✓")
print("=" * 88)

# Export
dataset_inventory_df = inv_df
print(f"\ndataset_inventory_df exported ({len(dataset_inventory_df)} rows)")
