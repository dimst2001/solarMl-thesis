# To Be Done

All analysis steps are finished (notebooks 1-6 and the app; details and results in alreadydone.MD).
Thesis: Μοντέλα Πρόβλεψης Παραγόμενης Ισχύος και Ανίχνευσης Ανωμαλιών σε Φωτοβολταϊκά Συστήματα. The methodology in
the description is "ενδεικτική", so deviations are allowed when the text justifies them (notes below).


## Points to state in the text

- Model ties are very strict with 500-800 test days (0.6 W differences count); mention a practical-equivalence
  margin (e.g. within 2% RMSE) when comparing models
- System 1283: the forecasting test set is small (403 slots, 17-19 days) and only 10 clean days were available for
  the synthetic faults, so its forecast ranking and anomaly numbers are indicative
- Anomaly detection: losses below ~20% are not detected (model noise); the 5σ threshold is a trade-off between
  sensitivity and false alarms (sensitivity table in notebook 5, section 5.5.3)

## Thesis-text notes (deviations from the indicative methodology, with the reason)

- Voltage / current excluded as features: P = V × I defines the power, so using them is data leakage, not prediction
- Chronological split + time-series CV instead of a random split: adjacent minutes are almost identical, so a random
  split leaks the answers (notebook 4, section 4.3: a random split looks 9-13% better)
- "Πρόβλεψη" covers both: estimation of the current power from same-time measurements (monitoring, basis of the
  anomaly detection) and forecasting 1 h / 3 h ahead (grid integration, load planning, storage)
- Multiple systems: 13 collected and cleaned; system 4 analysed in depth, system 1283 for generalization
- System 1283 split: one year only, so the test set is the last 7 days of every month instead of a chronological
  80/20 that would train on Jan-May only
- Sampling: «Οι μετρήσεις 2007–2009 είναι ανά 15 λεπτά και από το 2010 ανά 1 λεπτό» (the forecasting data use a
  common 15-minute grid)
- XGBoost and Isolation Forest are additions beyond the indicative list

- Anomaly detection is evaluated on synthetic events (partial losses, sensor faults) because the data have no labels
  for them; the rule labels judge only full outages (otherwise the evaluation would be circular)

## Optional

- Light hyperparameter tuning (small time-series grid search on the CV subsample)
- App: CSV upload for batch checks; an hourly temperature profile for the day curve
- Code simplification without changing results (verify by re-running and comparing the CSVs byte-for-byte, as in
  step 8): keep one block per model but move the repeated code into a
  shared helper module (paths, metrics, CV, model factories) for all notebooks; 3A-3D: one loop over a
  MODELS dict instead of 5 copied blocks; optionally one training notebook with CONFIG as parameter; system 1283 as
  a SYSTEM_ID switch instead of copied notebooks; move metrics_summary.csv / sensors.txt out of artifacts/metrics
- Optional: k-Means distance as a fourth unsupervised anomaly detector next to Isolation Forest
- Optional: the example notebook example_simple_training.ipynb (untracked) could become an appendix or a teaching demo; it is not part
  of the pipeline
