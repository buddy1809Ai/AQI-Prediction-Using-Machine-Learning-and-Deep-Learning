import os
import glob
import re
import pandas as pd
import numpy as np

DATA_ROOT = "CPCB_Data-20260602T165735Z-3-001/CPCB_Data"
CHUNK_SIZE = 50_000
EXPECTED_CITIES = [
    "Ahmedabad","Chennai","Delhi NCR","GandhiNagar","Hyderabad",
    "Indore","Jaipur","Jodhpur","Mumbai","Mumbai suburbs",
    "Nagpur","Navi Mumbai","Pune","Singrauli","Surat","Thane",
    "Vapi","Vishakhapattanam","Bhopal"
]

def is_stub(fpath, threshold=500):
    return os.path.getsize(fpath) < threshold

def count_rows(fpath):
    try:
        total = 0
        for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
            total += len(chunk)
        return total
    except Exception:
        return 0

city_folders = sorted([
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
])

city_stats = {}
schema_samples = {}

for city in city_folders:
    cpath = os.path.join(DATA_ROOT, city)
    all_csv = glob.glob(os.path.join(cpath, "*.csv"))
    real = [f for f in all_csv if not is_stub(f)]
    stubs = [f for f in all_csv if is_stub(f)]

    years = []
    freqs = []
    for f in real:
        m = re.search(r"_(\d{4})_", os.path.basename(f))
        if m:
            years.append(int(m.group(1)))
        mf = re.search(r"(\d+Min)", os.path.basename(f))
        if mf:
            freqs.append(mf.group(1))

    sample_df = None
    sample_cols = None
    err = None
    if real:
        try:
            sample_df = pd.read_csv(real[0], nrows=5, low_memory=False)
            sample_cols = list(sample_df.columns)
        except Exception as e:
            err = str(e)

    total_rows = sum(count_rows(f) for f in real)

    city_stats[city] = {
        "real_files": len(real), "stubs": len(stubs),
        "years": sorted(set(years)), "freqs": sorted(set(freqs)),
        "total_rows": total_rows, "sample_cols": sample_cols, "err": err
    }
    if sample_cols:
        schema_samples[city] = sample_cols

POLL_KEYS = ["PM2.5","PM10","NO","NO2","NOx","NH3","SO2","CO","Ozone","Benzene","Toluene","Xylene"]
MET_KEYS  = ["WS","WD","RH","Temp","RF","SR","BP","VWS"]
AQI_KEYS  = ["AQI","AQI_Bucket"]

print("="*72)
print("PHASE 0 — DATASET INVENTORY REPORT")
print("="*72)
print(f"\nCity folders found ({len(city_folders)}): {city_folders}\n")

rc = lambda c: city_stats[c]["total_rows"]
large_c = sorted([c for c in city_folders if rc(c) > 500_000], key=rc, reverse=True)
small_c = sorted([c for c in city_folders if rc(c) <= 500_000], key=rc, reverse=True)

total_rows = sum(rc(c) for c in city_folders)
est_gb = total_rows * 92 * 15 / 1e9

city_flags = {}
for city, s in city_stats.items():
    sdf = schema_samples.get(city, [])
    cu = [c.upper() for c in sdf]
    cu_up = [c.strip().upper() for c in cu]
    has_aqi   = any(k.upper() in cu_up for k in AQI_KEYS)
    pollutants = [k for k in POLL_KEYS if k.upper() in cu_up]
    met_cols   = [k for k in MET_KEYS  if k.upper() in cu_up]
    all_cols   = len(sdf)
    found_low  = [c for c in sdf if c.lower() in [k.lower() for k in POLL_KEYS + MET_KEYS]]
    expected_low = [k.lower() for k in POLL_KEYS[:6]]
    missing_exp = [k for k in expected_low if k not in [c.lower() for c in sdf]]
    unexpected = [c for c in sdf if c not in POLL_KEYS + MET_KEYS + AQI_KEYS + ["From Date","To Date","Stations"]]

    yrs   = s["years"]
    gaps  = [yrs[i+1]-yrs[i] for i in range(len(yrs)-1) if yrs[i+1]-yrs[i] > 1] if yrs else []

    city_flags[city] = {
        "files": s["real_files"], "rows": s["total_rows"],
        "years": yrs, "year_gaps": gaps, "has_aqi": has_aqi,
        "pollutants": pollutants, "met_cols": met_cols,
        "missing_core": missing_exp, "stubs": s["stubs"],
    }

for city, fl in city_flags.items():
    tag = "⚠ " if fl["year_gaps"] or fl["missing_core"] else "✓ "
    yrs_str = f"{min(fl['years'])}–{max(fl['years'])}" if fl["years"] else "—"
    print(f"{tag}{city:<20} files={fl['files']:2d}  rows={fl['rows']:>9,}  years={yrs_str}"
          f"  polls={len(fl['pollutants'])}  met={len(fl['met_cols'])}"
          f"  AQI={'Y' if fl['has_aqi'] else 'N'}")
    if fl["year_gaps"]:
        print(f"      ⚠ year gaps: {fl['year_gaps']}")
    if fl["missing_core"]:
        print(f"      ⚠ missing core pollutants: {fl['missing_core']}")

print(f"\n  Total rows  : {total_rows:,}")
print(f"  Est. size   : ~{est_gb:.1f} GB")
print(f"  Cities OK   : {sum(1 for f in city_flags.values() if not f['year_gaps'] and not f['missing_core'])}")
print(f"  Cities ⚠    : {sum(1 for f in city_flags.values() if f['year_gaps'] or f['missing_core'])}")

inventory_report = city_flags
