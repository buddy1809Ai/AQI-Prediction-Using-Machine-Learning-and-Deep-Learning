# Data Schema

## Columns

| Column | Type | Unit | Description |
|---|---|---|---|
| `Timestamp` | datetime |  | Hourly timestamp (UTC+5:30) |
| `PM2.5 (µg/m³)` | float | µg/m³ | Particulate matter <2.5µm — primary AQI driver |
| `PM10 (µg/m³)` | float | µg/m³ | Particulate matter <10µm |
| `NO (µg/m³)` | float64 |  |  |
| `NO2 (µg/m³)` | float | µg/m³ | Nitrogen dioxide |
| `NOx (ppb)` | float64 |  |  |
| `NH3 (µg/m³)` | float | µg/m³ | Ammonia |
| `SO2 (µg/m³)` | float | µg/m³ | Sulphur dioxide |
| `CO (mg/m³)` | float | mg/m³ | Carbon monoxide |
| `Ozone (µg/m³)` | float | µg/m³ | Ozone |
| `Benzene (µg/m³)` | float64 |  |  |
| `Toluene (µg/m³)` | float64 |  |  |
| `AT (°C)` | float | °C | Ambient temperature |
| `RH (%)` | float | % | Relative humidity |
| `WS (m/s)` | float | m/s | Wind speed |
| `WD (deg)` | float64 |  |  |
| `SR (W/mt2)` | float64 |  |  |
| `BP (mmHg)` | float | mmHg | Barometric pressure |
| `AQI` | float | 0–500 | Computed CPCB AQI (max sub-index) |
| `AQI_Category` | str |  | CPCB category: Good/Satisfactory/Moderate/Poor/Very Poor/Severe |
| `City` | object |  |  |

## Valid Ranges (CPCB Standard)

| Pollutant | Min | Max | Unit |
|---|---|---|---|
| PM2.5 | 0 | 1000 | µg/m³ |
| PM10  | 0 | 1500 | µg/m³ |
| NO2   | 0 | 800  | µg/m³ |
| SO2   | 0 | 2100 | µg/m³ |
| CO    | 0 | 50   | mg/m³ |
| O3    | 0 | 1000 | µg/m³ |
| NH3   | 0 | 2400 | µg/m³ |
| AQI   | 0 | 500  | —     |

## Sampling Method

Each city sample (200 rows) was derived from the cleaned hourly parquet by:
- Taking the 50 highest-AQI rows (pollution peak coverage)
- Taking the 50 lowest-AQI rows (clean-air baseline coverage)  
- Random-sampling 100 remaining rows
- Deduplicating and resetting index
