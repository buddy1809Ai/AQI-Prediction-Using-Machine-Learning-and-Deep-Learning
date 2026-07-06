import numpy as _np
import pandas as _pd
import sklearn as _sk
import xgboost as _xgb
import h5py as _h5

pkgs = [
    ("numpy",       _np),
    ("pandas",      _pd),
    ("scikit-learn",_sk),
    ("xgboost",     _xgb),
    ("h5py",        _h5),
]

extra = [
    "tensorflow", "keras", "matplotlib",
    "seaborn", "scipy", "pathlib",
]

SEP = "="*66

results = []
print(SEP)
print("  ENVIRONMENT VERIFICATION — PACKAGE REPORT")
print(SEP)
print(f"  {'Package':<20} {'Version / Error':<40} Status")
print("-"*66)

for name, mod in pkgs:
    ver = getattr(mod, "__version__", "?")
    results.append((name, ver, "✓"))
    print(f"  {name:<20} {ver:<40} ✓")

for name in extra:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        results.append((name, ver, "✓"))
        print(f"  {name:<20} {ver:<40} ✓")
    except ImportError as e:
        results.append((name, str(e), "✗"))
        print(f"  {name:<20} {'NOT FOUND':<40} ✗")

print(SEP)
ok  = sum(1 for _, _, s in results if s == "✓")
err = sum(1 for _, _, s in results if s == "✗")
print(f"  Packages OK : {ok}  |  Missing : {err}")
if err == 0:
    print("  ✓ Environment ready for all AQI modelling tasks.")
else:
    print("  ⚠ Some packages missing — check above.")
print(SEP)

env_status = "ready" if err == 0 else "incomplete"
