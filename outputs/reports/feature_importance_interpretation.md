# Feature Importance Interpretation

## Top 20 Features

See `outputs/track_a_feature_importance.csv` for full ranked list.


WHY PM2.5 DOMINATES FEATURE IMPORTANCE
─────────────────────────────────────────────────────────────────────
PM2.5 (fine particulate matter ≤ 2.5 µm) is the PRIMARY driver of
India's CPCB AQI for the following reasons:

1. FORMULA PRIMACY: The CPCB AQI = max(SI_PM2.5, SI_PM10, ..., SI_Ozone).
   In Indian cities, PM2.5 sub-index (SI_PM2.5) exceeds all other
   sub-indices in the majority of hourly records. When PM2.5 determines
   the AQI, its feature importance score reflects pure formula identity.

2. EPIDEMIOLOGICAL DOMINANCE: PM2.5 penetrates deep into lung tissue.
   CPCB assigns the steepest breakpoint slopes to PM2.5 compared to
   PM10 or gaseous pollutants, amplifying its contribution per µg/m³.

3. URBAN EMISSION PROFILE: Indian cities have high vehicular, industrial,
   and construction-dust PM2.5 emissions. On most hourly records, PM2.5
   concentrations in the "Very Poor" and "Severe" AQI range (>150 µg/m³)
   far exceed WHO guidelines, making PM2.5 the dominant AQI setter.

WHY PM10 MATTERS IN SPECIFIC CITIES
─────────────────────────────────────────────────────────────────────
In arid/semi-arid cities (Jodhpur, Jaipur, Ahmedabad, Gandhinagar),
wind-driven coarse dust elevates PM10 sub-index above PM2.5 in many
records. This is why tree models assign PM10 higher importance in
desert-climate cities compared to industrial cities like Delhi or Nagpur.

WHY METEOROLOGICAL FEATURES HAVE MODERATE IMPORTANCE
─────────────────────────────────────────────────────────────────────
Met variables (AT, RH, WS, WD) influence DISPERSION and FORMATION
rather than directly entering the AQI formula:
• High RH (>70%): hygroscopic growth of PM2.5 particles (visual haziness)
• Low WS (<2 m/s): reduced horizontal dispersion → pollutant accumulation
• High AT + Low RH: promotes photochemical Ozone formation (urban smog)
• WD: determines source region (industrial vs. clean upwind)

Met features appear moderately important because they explain residual
variance that same-t pollutants alone cannot: AQI spikes during calm,
humid conditions even at similar emission levels.

WHY TIME FEATURES CONTRIBUTE MINIMALLY IN TRACK A
─────────────────────────────────────────────────────────────────────
Once same-t pollutants are present, time features add minimal signal
because AQI(t) is directly computable. However, time features become
CRITICAL in Track B (forecasting) where same-t pollutants are absent:
• Hourly patterns: traffic rush hours (8am, 6pm peak emissions)
• Monthly/seasonal: winter inversions (Nov–Jan highest AQI nationally)
• Weekend effects: reduced heavy vehicle traffic

CITY-SPECIFIC DIFFERENCES
─────────────────────────────────────────────────────────────────────
• Delhi NCR: PM2.5 overwhelmingly dominant (agricultural burning +
  vehicular + industrial = chronic severe AQI). LSTM R²=0.915 — highest
  DL performance among all cities due to strong seasonal regularity.

• Jodhpur: PM10 often > PM2.5 (Thar Desert dust). High AQI volatility
  from episodic dust storms → LSTM R²=-0.10 (cannot forecast dust spikes).

• Hyderabad: Mixed industrial-vehicular profile. NO2 and SO2 contribute
  more than in northern cities. Harder for all models (GBR R²=0.969).

• Vishakhapatnam: Coastal sea-salt contribution elevates PM10 baseline.
  High humidity dampens PM2.5 measurement precision.

• Singrauli: Coal power plants → elevated SO2 and NO2. PM2.5 from fly
  ash makes it one of the most predictable cities (GBR R²=0.998).


## Environmental Summary

The feature importance analysis confirms that the CPCB AQI Estimation task (Track A) is dominated by instantaneous pollutant measurements, primarily PM2.5. This is scientifically expected and does NOT indicate target leakage. The tree-based models are learning the CPCB piecewise-linear formula — a valid and reproducible mapping that can be described as "AQI reconstruction from concurrent sensor measurements."

For Track B (True Forecasting), the same pollutants appear as **lag features** rather than concurrent measurements, reducing their predictive power as the horizon extends. At t+24h, meteorological and time features become relatively more important because pollutant autocorrelation decays substantially over 24 hours.
