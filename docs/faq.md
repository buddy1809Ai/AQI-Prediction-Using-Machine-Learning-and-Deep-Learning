# FAQ

**Q: Why does GBR outperform LSTM on Track A?**
A: AQI is a deterministic piecewise-linear formula. Tree models capture this exactly.

**Q: Is Track A data leakage?**
A: No. It's a legitimate estimation task: given sensor readings at time t, compute AQI(t).

**Q: Why does Track B degrade from 1h→24h?**
A: Atmospheric chaos. 24h forecasts reflect genuine predictive uncertainty, not a bug.

**Q: Why is Jodhpur BiLSTM R² negative?**
A: Data quality anomaly (suspected station relocation). Classical models are unaffected.

**Q: Can I add a new city?**
A: Yes — add CSVs to `data/raw/<CityName>/`, re-run `src/01` through `src/08`, add coords to settings.py.
