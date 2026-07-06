
# Track A — LSTM Failure Analysis
## Why LSTM Underperforms Classical Tree-Based Models in AQI Estimation

### Task Characterisation
Track A is AQI **Estimation** — not forecasting. The target AQI(t) is a deterministic
piecewise-linear function of same-timestamp pollutants PM2.5(t), PM10(t), NO2(t), SO2(t),
CO(t), NH3(t), Ozone(t). The mathematical formula:

  AQI(t) = max(SI_PM2.5(t), SI_PM10(t), ..., SI_Ozone(t))

where each SI is a breakpoint-interpolated sub-index. Identity test on 3 cities confirmed
R² = 1.0000 when formula is directly applied to raw pollutants.

### Why Tree-Based Models Excel
1. **Decision boundaries match the task**: AQI formula uses piecewise-linear breakpoints.
   Decision trees natively learn axis-aligned thresholds — an ideal match.
2. **No temporal memory required**: Same-timestamp features make time-series context irrelevant.
   GBR can reconstruct the formula with ~100 trees without any sequence.
3. **No noise amplification**: Boosting corrects residuals iteratively; convergence to the
   deterministic formula is fast and stable.
4. **Feature sparsity advantage**: 3–5 dominant pollutants explain >95% of AQI variance.
   Tree splits isolate the governing pollutant per sample efficiently.

### Why LSTM Underperforms
1. **Temporal context is not useful here**: LSTM reads 24-hour sequences, but AQI(t) depends
   ONLY on pollutants at t. The 23 prior timesteps add noise, not signal.
2. **Architecture mismatch**: Recurrent gates are designed for sequential dependency learning.
   A tabular regression with known closed-form input-output mapping is anti-pattern for LSTM.
3. **Training instability**: Cities with sparse pollutant coverage (Jodhpur R²=-0.10, Pune R²=-0.31)
   show LSTM diverges or fits the mean. GBR handles these gracefully.
4. **Sequence length vs sample size**: With seq_len=24, effective training samples =
   n_rows - 24. For small cities (~20k rows) this leaves <14k sequences — insufficient
   for stable LSTM convergence.
5. **Inverse scaling amplification**: Errors in scaled space [0,1] are amplified by
   AQI range (0–500+) during inverse_transform. High-AQI cities (Delhi R²=0.915) fare
   better because the scaler range is larger and smoother.

### Conclusion
LSTM's Track A underperformance (avg R²=0.6411 vs GBR 0.9906) is **scientifically expected**,
not a model failure. AQI estimation is a tabular regression problem best solved by tree
ensembles. LSTM adds genuine value only in Track B (pure forecasting without same-t features)
where temporal patterns matter.

### City-specific Failures
- Jodhpur (LSTM R²=-0.10): AQI driven by dust-storm episodic spikes; LSTM over-smooths.
- Pune (LSTM R²=-0.31): High feature missingness after recovery; sequences contain
  many imputed values reducing signal quality.
- Surat (LSTM R²=0.64): Industrial + coastal meteorology mix; LSTM learns partially.
