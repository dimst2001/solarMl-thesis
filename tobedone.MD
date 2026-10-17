 What's Next — In Order
 (finished steps are in alreadydone.MD)

 Thesis: Μοντέλα Πρόβλεψης Παραγόμενης Ισχύος και Ανίχνευσης Ανωμαλιών σε Φωτοβολταϊκά Συστήματα
 Title and description are fixed; the methodology in the description is "ενδεικτική", so deviations are
 allowed when the thesis text justifies them (see "Thesis-text notes" at the end).

 Order: system 4 notebooks 1 → 2 → 3 → 4  →  system 1283  →  anomaly detection  →  summary



A. Notebook 1 — Cleaning changes (system 4)  ✅ DONE (see alreadydone.MD 6b)
   daylight filter poa_irradiance > 50 W/m²; faults flagged (is_fault, 6.5% of rows); 15-minute grid saved



B. Notebook 2 — Rebuild features (system 4)

B1. ✅ DONE: sections 1-8B re-run on the normal rows (is_fault == False)

B2. ✅ DONE: pipeline 8C — forecasting datasets 1 h / 3 h ahead from the 15-minute grid (alreadydone.MD 12)



C. Notebook 3 — Training (system 4)

C1. ✅ DONE: 3A and 3B re-run; endogenous RMSE 69 -> 28 W (R2 0.99), exogenous R2 ~0.45 (alreadydone.MD 9-10)

C2. ✅ DONE: notebooks 3C (1 h) and 3D (3 h) with persistence / same-time-yesterday baselines (alreadydone.MD 13)



D. Notebook 4 — Evaluation & Comparison (system 4, reads saved files only)

D1. Metrics table: MAE, RMSE, R², CV mean ± std — estimation (endogenous / exogenous) and forecasting (1 h / 3 h);
    add a diurnally adjusted (clear-sky index) persistence as a stricter forecasting reference than plain persistence

D2. Ties: CV spread + paired test / bootstrap of RMSE differences; call ties ties

D3. Random vs chronological split: train one model with a random 80/20 split as in the indicative methodology
    and show how much it overestimates performance (adjacent minutes leak) — justifies the chronological split with numbers

D4. Efficiency: fit time, predict time, model size against accuracy

D5. Predicted vs actual: scatter per model + time-series overlay for sample days (clear, cloudy, winter);
    forecasts vs actual for the same days

D6. Residuals by irradiance level, month / season and test year
    (test period is mostly 2016 + 2018 + early 2023; 2019-2022 missing — describe it accurately)

D7. Feature importance: Random Forest + XGBoost built-in, permutation importance on a test sample

D8. Best model per task (accuracy + efficiency); the best estimation model is the anomaly-detection baseline



E. System 1283 — Generalization (after system 4 notebooks 1-4 are finished)

E1. Same pipeline as notebooks 1-4 with SYSTEM_ID = 1283 (copies of the notebooks with its column names):
    target dc_power__1136; POA_irradiance__1055, module_temp__1056, ambient_temp__1053 (+ wind_speed__1051)
E2. Differences to handle: 15-second data (aggregate to 1 min for estimation, 15 min for forecasting);
    only one year (2015) — a chronological 80/20 split tests on Oct-Dec only, so state that or use
    a split that keeps all seasons in the test set (e.g. last week of every month)
E3. Compare with system 4: do the model ranking, endogenous vs exogenous gap and forecast skill hold?



F. Notebook 5 — Anomaly Detection (system 4, then 1283)

Avoid circular evaluation: the is_fault rule cannot be both the method and the judge.
If faults are labelled with "power < ratio × irradiance" and a detector built on the same idea is scored against
those labels, it looks perfect without proving anything (like grading your own exam with your own answer key).

F1. Residuals of the best estimation model on all rows (normal + flagged)
F2. Rule baseline: the is_fault rule itself is the simplest detector; every other method must beat it
F3. Residual detector: threshold learned on the TRAINING period only (e.g. 99th percentile of normal residuals)
F4. Unsupervised detector: Isolation Forest on power, irradiance, temperature and residual; never sees the labels
F5. Evaluation by anomaly type, not one overall score:
    - full outages (sun strong, power ≈ 0): the rule catches these
    - partial losses (shading, soiling, snow, a failed string: 10-40% below normal for hours/days): the rule misses most
    - sensor faults (irradiance sensor dirty / misaligned: power normal, irradiance wrong): the rule cannot see these
    precision / recall against the rule labels only for full outages
F6. Agreement analysis: where the methods agree (confident anomalies) and where they disagree (inspect and explain)
F7. Example events over time (single minutes vs hours/days of outage)



G. Notebook 6 — Results Summary

Final tables: estimation, forecasting (1 h / 3 h), anomaly detection; system 4 vs 1283
Conclusions for both parts of the title; charts ready for the thesis document



Thesis-text notes (deviations from the indicative methodology, with the reason)

- Voltage / current excluded as features: P = V × I defines the power, so using them is data leakage, not prediction
  (high correlation with the power is supporting evidence, not the main reason)
- Chronological split + time-series CV instead of a random split: adjacent minutes are almost identical, so a random
  split leaks the answers; D3 shows the overestimate in numbers
- "Πρόβλεψη" covers both: estimation of the current power from same-time measurements (monitoring / telemetry,
  basis of the anomaly detection) and forecasting 1 h / 3 h ahead (grid integration, load planning, storage)
- Multiple systems: 13 collected and cleaned; system 4 analysed in depth, system 1283 for generalization
- Sampling: «Οι μετρήσεις 2007–2009 είναι ανά 15 λεπτά και από το 2010 ανά 1 λεπτό» (no overlap; the forecasting
  data use a common 15-minute grid)
- XGBoost and Isolation Forest are additions beyond the indicative list



Optional

Light hyperparameter tuning (small time-series grid search on the CV subsample)
