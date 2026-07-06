"""weather_client.py — OpenWeatherMap API wrapper with graceful fallback."""
import streamlit as st
from typing import Optional, Dict
try:
    import requests; _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
from app.config.settings import WEATHER_CACHE_TTL
from app.utils.logger import log

@st.cache_data(ttl=WEATHER_CACHE_TTL, show_spinner=False)
def get_weather(city_name: str) -> Optional[Dict]:
    """Fetch current weather. Returns None if no API key or request fails."""
    api_key = None
    try: api_key = st.secrets.get("OPENWEATHER_API_KEY") or st.secrets.get("openweather_api_key")
    except: pass
    if not api_key or not _HAS_REQUESTS: return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name},IN&appid={api_key}&units=metric"
        resp = requests.get(url, timeout=8); resp.raise_for_status(); data = resp.json()
        return {"temp_c":round(data["main"]["temp"],1),"humidity_pct":data["main"]["humidity"],
                "wind_ms":data["wind"]["speed"],"pressure_hpa":data["main"]["pressure"],
                "description":data["weather"][0]["description"].title()}
    except Exception as e:
        log.warning(f"Weather API failed for {city_name}: {e}"); return None
