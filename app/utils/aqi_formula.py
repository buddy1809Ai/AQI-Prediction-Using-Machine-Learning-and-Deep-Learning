"""
aqi_formula.py — Official CPCB AQI sub-index calculation.
Pure function: no Streamlit/IO dependency. Unit-tested in tests/test_aqi_formula.py.
"""
import math
from typing import Dict, Optional, Tuple

_BREAKPOINTS = {
    "pm25": [(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,500,401,500)],
    "pm10": [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,600,401,500)],
    "no2":  [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,800,401,500)],
    "so2":  [(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2100,401,500)],
    "co":   [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
    "o3":   [(0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
    "nh3":  [(0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2400,401,500)],
}
_CATEGORIES = [(0,50,"Good"),(51,100,"Satisfactory"),(101,200,"Moderate"),
               (201,300,"Poor"),(301,400,"Very Poor"),(401,999,"Severe")]
_COLORS = {"Good":"#17b26a","Satisfactory":"#A1C9F4","Moderate":"#ffd400",
           "Poor":"#FFB482","Very Poor":"#f04438","Severe":"#9467BD"}
_VALID_RANGES = {"pm25":(0,1000),"pm10":(0,1500),"no2":(0,800),"so2":(0,2100),
                 "co":(0,50),"o3":(0,1000),"nh3":(0,2400)}

def compute_sub_index(pollutant: str, value: float) -> Optional[float]:
    """Compute CPCB sub-index for one pollutant. Returns None for invalid inputs."""
    key = pollutant.lower().replace(".","").replace(" ","")
    bps = _BREAKPOINTS.get(key)
    if bps is None or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    lo_v, hi_v = _VALID_RANGES.get(key, (0, 1e9))
    if value < lo_v or value > hi_v:
        return None
    for (cl, ch, al, ah) in bps:
        if cl <= value <= ch:
            return al + (value - cl) * (ah - al) / max(ch - cl, 1e-9)
    return 500.0

def validate_inputs(pm25=None, pm10=None, no2=None, so2=None,
                    co=None, o3=None, nh3=None) -> Tuple[bool, list]:
    """Validate pollutant inputs. Returns (is_valid, warnings_list)."""
    inputs = {k: v for k, v in dict(pm25=pm25,pm10=pm10,no2=no2,so2=so2,co=co,o3=o3,nh3=nh3).items()
              if v is not None and not (isinstance(v,float) and math.isnan(v))}
    if not inputs:
        return False, ["At least one pollutant value is required."]
    warnings_out = []
    for key, val in inputs.items():
        if val < 0:
            return False, [f"{key.upper()} cannot be negative (got {val})."]
        _, hi = _VALID_RANGES.get(key, (0, 1e9))
        if val > hi:
            warnings_out.append(f"{key.upper()}={val} exceeds maximum ({hi}). Value capped.")
    return True, warnings_out

def compute_aqi(pm25=None, pm10=None, no2=None, so2=None,
                co=None, o3=None, nh3=None) -> Dict:
    """Compute CPCB AQI. Units: pm25/pm10/no2/so2/o3/nh3 in µg/m³; co in mg/m³."""
    inputs = dict(pm25=pm25,pm10=pm10,no2=no2,so2=so2,co=co,o3=o3,nh3=nh3)
    sub_indices = {}
    for key, val in inputs.items():
        if val is not None and not (isinstance(val,float) and math.isnan(val)):
            si = compute_sub_index(key, float(val))
            if si is not None:
                sub_indices[key] = round(si, 1)
    if not sub_indices:
        return {"aqi":None,"category":"Unknown","color":"#909094",
                "dominant_pollutant":None,"sub_indices":{},"health_advice":"No valid inputs."}
    aqi_val  = max(sub_indices.values())
    dominant = max(sub_indices, key=sub_indices.get)
    category, color = "Severe", "#9467BD"
    for lo, hi, cat in _CATEGORIES:
        if lo <= aqi_val <= hi:
            category, color = cat, _COLORS[cat]; break
    advice = {"Good":"Air quality is satisfactory. Enjoy outdoor activities freely.",
              "Satisfactory":"Acceptable. Sensitive individuals may feel mild discomfort.",
              "Moderate":"Sensitive groups may experience effects. Limit outdoor exertion.",
              "Poor":"Everyone may begin to experience effects. Sensitive groups stay indoors.",
              "Very Poor":"Health warning. Everyone should avoid outdoor exertion.",
              "Severe":"Emergency. Everyone must stay indoors."}
    return {"aqi":round(aqi_val,1),"category":category,"color":color,
            "dominant_pollutant":dominant.upper(),"sub_indices":sub_indices,
            "health_advice":advice.get(category,"")}
