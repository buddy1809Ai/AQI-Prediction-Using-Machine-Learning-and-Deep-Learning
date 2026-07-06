"""settings.py — Central config: all paths, constants, thresholds, cache TTLs."""
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = REPO_ROOT / "data" / "sample_dataset"
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EDA_DIR     = FIGURES_DIR / "eda"
TABLES_DIR  = OUTPUTS_DIR / "tables"
EVAL_DIR    = OUTPUTS_DIR / "evaluation"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_JSON = EVAL_DIR / "model_metrics.json"

CITIES = [
    "Ahmedabad","Chennai","Delhi NCR","GandhiNagar","Hyderabad","Indore",
    "Jaipur","Jodhpur","Mumbai","Mumbai suburbs","Nagpur","Pune",
    "Singrauli","Surat","Thane","Vapi","bhopal","vishakhapattanam",
]
CITY_COORDS = {
    "Ahmedabad":(23.0225,72.5714),"Chennai":(13.0827,80.2707),
    "Delhi NCR":(28.7041,77.1025),"GandhiNagar":(23.2156,72.6369),
    "Hyderabad":(17.3850,78.4867),"Indore":(22.7196,75.8577),
    "Jaipur":(26.9124,75.7873),"Jodhpur":(26.2389,73.0243),
    "Mumbai":(19.0760,72.8777),"Mumbai suburbs":(19.1663,72.8526),
    "Nagpur":(21.1458,79.0882),"Pune":(18.5204,73.8567),
    "Singrauli":(24.1997,82.6875),"Surat":(21.1702,72.8311),
    "Thane":(19.2183,72.9781),"Vapi":(20.3714,72.9052),
    "bhopal":(23.2599,77.4126),"vishakhapattanam":(17.6868,83.2185),
}
TRACK_A_MODELS = ["LinearRegression","Ridge","RandomForest","GradBoost","XGBoost","LSTM","BiLSTM","CNN-BiLSTM"]
TRACK_B_MODELS = ["LinearRegression","RandomForest","GradBoost","XGBoost","LSTM","BiLSTM","CNN-BiLSTM"]
HORIZONS = [1, 6, 24]
AQI_CATEGORIES = {
    "Good":(0,50,"#17b26a"),"Satisfactory":(51,100,"#A1C9F4"),
    "Moderate":(101,200,"#ffd400"),"Poor":(201,300,"#FFB482"),
    "Very Poor":(301,400,"#f04438"),"Severe":(401,999,"#9467BD"),
}
HEALTH_ADVICE = {
    "Good":"Air quality is satisfactory. Enjoy outdoor activities.",
    "Satisfactory":"Air quality is acceptable. Sensitive individuals may feel mild discomfort.",
    "Moderate":"Sensitive groups may experience health effects. Limit prolonged outdoor exertion.",
    "Poor":"Everyone may begin to experience adverse effects. Sensitive groups should stay indoors.",
    "Very Poor":"Health warning. Everyone should avoid outdoor exertion.",
    "Severe":"Emergency conditions. Everyone must stay indoors.",
}
WEATHER_CACHE_TTL = 900
METRICS_CACHE_TTL = 3600
DATA_CACHE_TTL    = 3600
BG, TEXT, DIM     = "#1D1D20", "#fbfbff", "#909094"
PALETTE = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4","#9467BD"]
