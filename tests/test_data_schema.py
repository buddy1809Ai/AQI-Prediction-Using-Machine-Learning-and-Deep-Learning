"""Validate sample datasets match documented schema — 6 test cases."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import pytest

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample_dataset"
AQI_VALID_CATEGORIES = {"Good","Satisfactory","Moderate","Poor","Very Poor","Severe"}

def get_files(): return list(SAMPLE_DIR.glob("*_sample.csv"))

@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="No sample data")
def test_sample_dir_not_empty(): assert len(get_files()) > 0

@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="No sample data")
def test_aqi_column_present():
    for f in get_files():
        df = pd.read_csv(f, nrows=5)
        assert "AQI" in df.columns, f"{f.name}: missing AQI column"

@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="No sample data")
def test_aqi_numeric():
    for f in get_files():
        df = pd.read_csv(f)
        if "AQI" in df.columns:
            assert pd.to_numeric(df["AQI"],errors="coerce").notna().any()

@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="No sample data")
def test_aqi_non_negative():
    for f in get_files():
        df = pd.read_csv(f)
        if "AQI" in df.columns:
            assert (pd.to_numeric(df["AQI"],errors="coerce").dropna() >= 0).all()

@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="No sample data")
def test_no_all_null_columns():
    for f in get_files():
        df = pd.read_csv(f)
        for col in df.columns:
            assert not df[col].isna().all(), f"{f.name}: '{col}' is entirely null"

@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="No sample data")
def test_minimum_row_count():
    for f in get_files():
        df = pd.read_csv(f)
        assert len(df) >= 10, f"{f.name}: only {len(df)} rows"
