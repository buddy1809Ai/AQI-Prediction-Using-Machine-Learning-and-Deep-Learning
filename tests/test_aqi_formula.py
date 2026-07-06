"""Unit tests for CPCB AQI formula — 18 test cases."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.utils.aqi_formula import compute_aqi, compute_sub_index, validate_inputs

def test_pm25_good(): assert 0 <= compute_sub_index("pm25", 15.0) <= 50
def test_pm25_severe(): assert compute_sub_index("pm25", 300.0) > 400
def test_pm25_zero(): assert compute_sub_index("pm25", 0.0) == 0.0
def test_co_boundary(): assert 0 <= compute_sub_index("co", 1.0) <= 51
def test_unknown_pollutant(): assert compute_sub_index("benzene", 10.0) is None
def test_negative_value(): assert compute_sub_index("pm25", -5.0) is None
def test_out_of_range(): assert compute_sub_index("pm25", 2000.0) is None
def test_good_air(): r=compute_aqi(pm25=15.0,pm10=30.0,no2=20.0,so2=10.0,co=0.5,o3=30.0); assert r["aqi"]<=100
def test_very_poor(): r=compute_aqi(pm25=200.0,pm10=400.0,no2=250.0,so2=300.0); assert r["aqi"]>200
def test_dominant_pm25(): assert compute_aqi(pm25=300.0,pm10=50.0,no2=40.0)["dominant_pollutant"]=="PM25"
def test_single_pollutant(): r=compute_aqi(pm25=60.0); assert r["aqi"] is not None
def test_no_inputs(): assert compute_aqi()["aqi"] is None
def test_health_advice(): assert len(compute_aqi(pm25=150.0)["health_advice"]) > 10
def test_sub_indices(): r=compute_aqi(pm25=60.0,pm10=100.0); assert "pm25" in r["sub_indices"]
def test_aqi_max(): r=compute_aqi(pm25=60.0,pm10=100.0,no2=40.0); assert math.isclose(r["aqi"],max(r["sub_indices"].values()),rel_tol=1e-3)
def test_validate_negative_rejected(): ok,_=validate_inputs(pm25=-10.0); assert not ok
def test_validate_no_inputs(): ok,_=validate_inputs(); assert not ok
def test_validate_good(): ok,_=validate_inputs(pm25=60.0,pm10=80.0); assert ok
