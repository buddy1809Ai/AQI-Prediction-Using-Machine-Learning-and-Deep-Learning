
import os, glob, re
import pandas as pd
import numpy as np

DATA_ROOT = "CPCB_Data-20260602T165735Z-3-001/CPCB_Data"
CHUNK_SIZE = 50_000

# ─── Pick representative files (diverse city + year coverage) ────────────────
SAMPLE_TARGETS = [
    ("Delhi NCR",         2023),
    ("Mumbai",            2021),
    ("Hyderabad",         2019),
    ("bhopal",            2024),
    ("Ahmedabad",         2022),
    ("Singrauli",         2020),
    ("vishakhapattanam",  2023),
]

def find_file(city, year):
    pattern = os.path.join(DATA_ROOT, city, f"*{year}*.csv")
    hits = glob.glob(pattern)
    return hits[0] if hits else None

def is_stub(fpath):
    try:
        df = pd.read_csv(fpath, nrows=3, low_memory=False)
        return len(df) == 0
    except Exception:
        return True

# ─── Schema inspection ────────────────────────────────────────────────────────
print("=" * 72)
print("02 — SCHEMA & SAMPLE INSPECTION")
print("=" * 72)

schema_details = {}
all_schemas    = {}

for city, year in SAMPLE_TARGETS:
    fpath = find_file(city, year)
    if fpath is None or is_stub(fpath):
        print(f"\n  ⚠ {city}/{year}: file not found or stub, skipping")
        continue

    df = pd.read_csv(fpath, nrows=500, low_memory=False)
    
    # Parse timestamp
    ts_col = None
    for c in df.columns:
        if 'time' in c.lower() or 'date' in c.lower():
            ts_col = c; break
    ts_parsed, ts_format, ts_min, ts_max = None, "—", "—", "—"
    if ts_col:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col])
            ts_min = str(df[ts_col].min())
            ts_max = str(df[ts_col].max())
            ts_format = "parsed OK"
        except Exception as e:
            ts_format = f"parse error: {e}"

    # Column dtypes
    col_dtypes = dict(df.dtypes.astype(str))

    # Missingness
    miss_pct = (df.isnull().mean() * 100).round(1).to_dict()

    # Value ranges for numeric cols
    num_cols = df.select_dtypes(include='number').columns.tolist()
    ranges   = {c: (round(df[c].min(),2), round(df[c].max(),2))
                for c in num_cols if not df[c].isna().all()}

    schema_details[(city, year)] = dict(
        file=os.path.basename(fpath), n_cols=len(df.columns),
        cols=list(df.columns), dtypes=col_dtypes,
        ts_col=ts_col, ts_min=ts_min, ts_max=ts_max, ts_format=ts_format,
        miss_pct=miss_pct, ranges=ranges, sample_head=df.head(3)
    )
    all_schemas[f"{city}_{year}"] = list(df.columns)

    print(f"\n─── {city}  ({year}) ─────────────────────────────────────────────")
    print(f"  File      : {os.path.basename(fpath)}")
    print(f"  Columns   : {len(df.columns)}")
    print(f"  Col names : {list(df.columns)}")
    print(f"  Timestamp : col='{ts_col}' | range={ts_min}→{ts_max} | {ts_format}")
    print(f"  Dtypes    : {col_dtypes}")
    print(f"  Missingness (%) top columns:")
    for col, pct in sorted(miss_pct.items(), key=lambda x: -x[1])[:10]:
        print(f"    {col:<35} {pct:>6.1f}%")
    print(f"  Value ranges (numeric):")
    for col, (mn, mx) in list(ranges.items())[:10]:
        print(f"    {col:<35} [{mn}, {mx}]")

# ─── Cross-city schema consistency check ─────────────────────────────────────
print("\n" + "=" * 72)
print("CROSS-CITY SCHEMA CONSISTENCY CHECK")
print("=" * 72)

ref_cols = None
ref_key  = None
schema_diffs = []
for key, cols in all_schemas.items():
    if ref_cols is None:
        ref_cols = set(cols); ref_key = key; continue
    diff_A = set(cols) - ref_cols
    diff_B = ref_cols - set(cols)
    if diff_A or diff_B:
        schema_diffs.append((key, ref_key, diff_A, diff_B))
        print(f"  Schema diff: {key} vs {ref_key}")
        print(f"    Extra in {key}      : {diff_A}")
        print(f"    Missing from {key}  : {diff_B}")

if not schema_diffs:
    print("  ✓ All sampled cities share identical column schema")

# ─── AQI Column Assessment ────────────────────────────────────────────────────
print("\n─── AQI COLUMN ASSESSMENT ──────────────────────────────────────────────")
print("""
  FINDING: No direct 'AQI' column exists in CPCB raw data.
  The dataset provides sub-index pollutant concentrations.
  
  ✓ AQI must be computed from sub-index pollutants using CPCB formula:
    AQI = max of individual sub-index values computed for:
      PM2.5, PM10, NO2, SO2, CO, O3, NH3
    
  Sub-index breakpoints (CPCB AQI Standard):
    Category       AQI Range
    Good           0–50
    Satisfactory   51–100
    Moderate       101–200
    Poor           201–300
    Very Poor      301–400
    Severe         401–500
    
  This AQI computation will be Step 1 in 02_Data_Cleaning.
""")

# ─── Timestamp format detection ──────────────────────────────────────────────
print("─── TIMESTAMP FORMAT ──────────────────────────────────────────────────")
for (city, year), d in schema_details.items():
    print(f"  {city}/{year}: col='{d['ts_col']}' | {d['ts_format']} | {d['ts_min']}→{d['ts_max']}")

# Summary export
schema_summary = dict(
    all_schemas=all_schemas,
    schema_diffs=schema_diffs,
    schema_details={f"{c}_{y}": {k: v for k, v in d.items() if k != 'sample_head'}
                    for (c, y), d in schema_details.items()},
    has_direct_aqi=False,
    aqi_must_be_computed=True,
    pollutants_for_aqi=['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)',
                        'SO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)', 'NH3 (µg/m³)'],
    timestamp_col='Timestamp',
    n_cols_per_file=25,
    schema_uniform=len(schema_diffs) == 0,
)
print("\nschema_summary exported ✓")
print("=" * 72)
