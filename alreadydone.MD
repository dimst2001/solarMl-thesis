 
 What i have already done
1. Dataset Selection
Chose NREL PVDAQ (National Renewable Energy Laboratory — Photovoltaic Data Acquisition) from data.openei.org. Free, CC-BY licensed, used in hundreds of IEEE papers. From 1,862 available systems we selected 13 usable systems based on data quality (qa_status=pass), sensor coverage, and geographic diversity.
-----------------------------------------------------------------------
2. Data Download
Downloaded raw data from AWS S3 (oedi-data-lake) using aws s3 cp --no-sign-request. Each system stored as daily Apache Parquet files (compressed binary format).
-----------------------------------------------------------------------
3. Data Transformation
Converted each daily parquet file from long format (one row per metric per timestamp) to wide format (one row per timestamp, one column per sensor) using transform_data.py. Output saved to pvdata_processed/.
-----------------------------------------------------------------------
4. Merging
Merged all daily CSVs per system into one file per system using merge_systems.py. Output saved to merged/.

---------------------------------------------------------------------------
5. Environment Setup
Python 3.12, virtual environment myenv
Libraries: pandas, pyarrow, scikit-learn, matplotlib, seaborn, xgboost, jupyterlab
VS Code + Jupyter kernel registered
thesis/ folder created as clean GitHub repo

--------------------------------------------------------------------------
6. Data Cleaning (clean_systems.py)
System by system:
Replaced sentinel values (-99999, -7999) with NaN
Converted Fahrenheit → Celsius (systems 34, 35, 1200, 1276, 1277)
Fixed power_factor_1000 ÷ 1000 (systems 34, 35)
Fixed dc_power_hW ÷ 2 (systems 34, 35, 1276, 1277)
Fixed wind_speed_mph × 0.447
Dropped broken/constant sensors
Clipped physically impossible values
Resampled system_1283 from sub-minute → 1-minute


--------------------------------------------------------------------------

6b. Preprocessing (notebooks/1_Data_preproccesing.ipynb, system 4) — revised after the review (step 11)
Loaded data/merged/system_4_merged.csv: 4,206,520 rows x 12 columns (no duplicates, no invalid timestamps)
Sampling: Aug 2007 - Jun 2009 every 15 min (59,713 rows), no data Jul - Dec 2009, from Jan 2010 every 1 min
Sentinel codes (-99999, -7999) -> NaN; no unit conversions needed for system 4
Daylight filter on irradiance (not on the target): keep poa_irradiance__313 > 50 W/m² -> 1,632,969 rows
  (removes nights, the inverter start-up zone and ~7,000 night-time irradiance sensor offsets of ~25 W/m²;
  the old dc_power ≥ 20 W floor had silently removed most daylight outages)
Removed high outliers above 5 x median daylight power (median 319.80 W, cutoff 1,598.98 W): 1 row
Kept dc_power__314, poa_irradiance__313, module_temp_1__321, ambient_temp__320, das_temp__325
Dropped rows with any remaining missing value (complete-case, no imputation): 216,338 rows
Fault band inspected: power / irradiance ratio is bimodal (outage spike near 0, normal peak at 1);
  k = 0.8771 W per W/m² (median ratio above 200 W/m²) -> artifacts/metrics/system_4_fault_band.png
Faults FLAGGED, not deleted: is_fault = irradiance > 50 W/m² and dc_power < 0.5 × k × irradiance
  92,307 rows (6.52%), 2,643 events (median 7 min, 90th percentile 120 min, longest 735 min)
Saved data/cleaned/system_4_cleaned.csv: 1,416,630 daylight rows (1,324,323 normal + 92,307 faults), no missing values
Saved data/cleaned/system_4_cleaned_15min.csv: regular 15-minute grid 2007-08-26 -> 2023-01-31, nights included,
  541,248 slots, 336,967 valid (≥ 50% of the expected readings), 25,428 slots with a fault (for forecasting)

------------------------------------------------------------------------------------------
7. Feature Selection (notebooks/2_Feature_Selection.ipynb, system 4)
Uses the normal rows only (is_fault == False): 1,324,323 rows
Target: dc_power__314. Candidates: poa_irradiance__313, module_temp_1__321, ambient_temp__320, das_temp__325
Basic statistics per column (no near-constant or highly skewed columns) -> artifacts/metrics/system_4_feature_statistics.csv
Distributions (histogram + box plot per column)
Pearson correlation with dc_power (normal rows): 313 r=0.995, 321 r=0.64, 325 r=0.24, 320 r=0.09
Correlation heatmap: ambient_temp__320 and das_temp__325 are collinear (r=0.93) -> artifacts/metrics/system_4_correlation_heatmap.png
Scatter plots per feature (50k-row sample) -> artifacts/metrics/system_4_scatter_dc_power__314.png
Voltage / current not used as features: P = V × I defines the power (data leakage)
Two feature configurations defined (FEATURE_CONFIGS):
  endogenous = 313 + 321 (drop 320, 325) -> pipeline 8A (8A.1-8A.7)
  exogenous = 320 + hour_sin, hour_cos, doy_sin, doy_cos (cyclical time of day / day of year) -> pipeline 8B (8B.1-8B.7)
Both: chronological 80/20 split with identical rows: train 1,059,458 rows (2007-08 -> 2016-01-21),
test 264,865 rows (2016-01-21 -> 2023-01; mostly 2016 and 2018, 2019-2022 missing)
MinMaxScaler fitted on train only; scaled features written with 5 decimals
-> data/features/group_4_<config>/ + artifacts/models/scaler_group_4_<config>.pkl
Config summary -> artifacts/metrics/system_4_feature_configs.csv

8. Code cleanup
Both notebooks simplified to system 4 only (same outputs, verified byte-for-byte on the cleaned CSV, train/test CSVs, scaler and plots)
All logic lives in the notebooks; the unused src/ package and tests/ folder were removed

9. Training Regressor Models — Endogenous (notebooks/3A_Training_Regressor_Models_Endogenous.ipynb)
One block per model: 3A.x.1 cross-validation, 3A.x.2 training, 3A.x.3 evaluation, 3A.x.4 metrics; 3A.6 comparison summary
CV: TimeSeriesSplit (5 expanding folds) x seeds 0, 1, 2 on every 10th training row; final model on the full train set
Fixed hyperparameters (no tuning yet); XGBoost uses configs/train_config.yaml except colsample_bytree = 1.0
(with 2 features, 0.8 gave each tree a single feature)
Test results on normal rows (264,865 rows), after the cleaning revision:
  Model              test MAE   test RMSE  test R2   CV RMSE (mean ± std)  fit time   model size
  MLP                18.9 W     28.3 W     0.991     24.4 ± 1.5 W          9 s        62 KB
  XGBoost            19.1 W     28.9 W     0.991     25.0 ± 1.1 W          4 s        2.2 MB
  Random Forest      19.0 W     28.9 W     0.991     24.5 ± 1.4 W          15 s       39.4 MB
  Decision Tree      19.2 W     29.1 W     0.991     24.9 ± 1.3 W          2 s        434 KB
  Linear Regression  22.0 W     32.0 W     0.989     28.1 ± 1.3 W          0.03 s     1 KB
  (before the revision: RMSE 69-74 W, R2 0.95, CV std 10-17 W — the error was dominated by unflagged faults)
Outputs: artifacts/metrics/system_4_endogenous_model_metrics.csv + _model_rmse.png,
artifacts/experiments/system_4_endogenous_cv_folds.csv + _test_predictions.csv, artifacts/models/system_4_endogenous_<model>.pkl

10. Training Regressor Models — Exogenous (notebooks/3B_Training_Regressor_Models_Exogenous.ipynb)
Same structure, CV and hyperparameters as 3A, on ambient_temp__320 + 4 cyclical calendar features
Test results (same 264,865 normal test rows), after the cleaning revision:
  Model              test MAE   test RMSE  test R2   CV RMSE (mean ± std)  fit time   model size
  MLP                166.5 W    217.2 W    0.478     218.0 ± 8.7 W         84 s       69 KB
  XGBoost            167.0 W    222.2 W    0.454     231.2 ± 6.8 W         4 s        2.4 MB
  Random Forest      167.5 W    222.8 W    0.451     222.0 ± 7.3 W         24 s       29.3 MB
  Linear Regression  181.0 W    223.1 W    0.450     225.7 ± 6.5 W         0.1 s      1 KB
  Decision Tree      171.6 W    232.0 W    0.405     232.7 ± 6.5 W         3 s        353 KB
  (before the revision: RMSE 211-226 W, R2 0.49-0.55)
Finding: without on-site irradiance the error is ~7-8x higher (R2 ~0.45 vs 0.99): time of day and season explain
less than half of the variance; the rest is clouds, which only the irradiance sensor sees.
Outputs: same files as 3A with _exogenous_ in the name

12. Forecasting datasets (notebook 2, pipeline 8C)
Source: the regular 15-minute grid from notebook 1 (nights included, empty slots for gaps)
Target: DC power of a 15-minute slot 1 h (4 slots) or 3 h (12 slots) after the latest observed slot (the origin)
13 features known at forecast time: power at the origin and 15 min / 1 h / 24 h before, irradiance at the origin
and 1 h / 24 h before, ambient temperature at the origin, power of the target slot 24 h earlier,
calendar of the target time (hour / day of year sin/cos)
Daylight target slots from a calendar profile built on the training period only (95th percentile irradiance per
week of year and slot > 50 W/m²: 9.0 h/day in winter to 13.8 h/day in summer); fault slots excluded
Same split time as 8A/8B (2016-01-21): forecast_1h 93,909 train / 21,011 test rows, forecast_3h 93,812 / 20,884
-> data/features/group_4_forecast_1h/, group_4_forecast_3h/ (+ baselines_test CSV with persistence and same time yesterday)

13. Forecasting Regressor Models (notebooks/3C_Forecasting_Regressor_Models_1h.ipynb, 3D_..._3h.ipynb)
Same block structure as 3A/3B plus section .0 Baselines; CV on every training row (5 folds x 3 seeds)
Skill = 1 − RMSE / RMSE_persistence
1 h ahead (21,011 test slots):
  Model                 test RMSE  test R2  skill vs persistence  CV RMSE (mean ± std)
  Random Forest         140.0 W    0.810    0.34                  149.2 ± 5.6 W
  XGBoost               140.9 W    0.808    0.34                  151.6 ± 5.9 W
  MLP                   141.6 W    0.806    0.33                  150.2 ± 6.1 W
  Decision Tree         146.0 W    0.793    0.31                  156.3 ± 5.7 W
  Linear Regression     159.8 W    0.752    0.25                  164.8 ± 6.1 W
  Persistence           211.8 W    0.565    0                     —
  Same time yesterday   279.2 W    0.244   -0.32                  —
3 h ahead (20,884 test slots):
  MLP                   179.9 W    0.686    0.55                  189.5 ± 7.9 W
  Random Forest         180.4 W    0.684    0.55                  190.2 ± 7.5 W
  XGBoost               182.0 W    0.678    0.54                  195.5 ± 8.7 W
  Decision Tree         187.5 W    0.659    0.53                  201.3 ± 8.8 W
  Linear Regression     204.5 W    0.594    0.49                  210.0 ± 7.1 W
  Same time yesterday   278.9 W    0.245    0.30                  —
  Persistence           397.5 W   -0.533    0                     —
Findings: every model beats both baselines; error grows with the horizon (R2 0.99 estimation -> 0.81 at 1 h -> 0.69 at 3 h);
RF / XGBoost / MLP are close (within ~2 W at 1 h), Linear Regression clearly behind.
Note: plain persistence ignores the daily sun cycle, so at 3 h it is weaker than same time yesterday and the 3 h skill
looks large; a diurnally adjusted (clear-sky) persistence is a stricter reference for notebook 4.
Outputs: artifacts/metrics/system_4_forecast_<h>_model_metrics.csv + _model_rmse.png,
artifacts/experiments/system_4_forecast_<h>_cv_folds.csv + _test_predictions.csv

14. Evaluation & Comparison (notebooks/4_Evaluation_Comparison.ipynb)
Reads the saved metrics, predictions and models of 3A-3D (only 4.3 re-trains two quick models on purpose)
4.1 Metrics for all four tasks -> artifacts/metrics/system_4_eval_metrics_all_tasks.csv
  Clear-sky persistence added as a stricter forecasting reference (clear-sky index kept constant; P_cs = 95th
  percentile power per week of year and 15-min slot, training period only):
  RMSE 1 h: persistence 211.8, clear-sky persistence 165.2, best model 140.0 W -> skill vs clear-sky 0.15
  RMSE 3 h: persistence 397.5, clear-sky persistence 237.0, best model 179.9 W -> skill vs clear-sky 0.24
  nRMSE (% of mean power): endogenous 5.8%, exogenous 44.4%, forecast 1 h 34.5%, forecast 3 h 44.4%
4.2 Ties: paired block bootstrap over test days (1,000 resamples). With 477-833 test days even 0.6 W differences
  are statistically significant: endogenous MLP best (28.3 W; XGB / RF / DT +0.6-0.8 W), exogenous MLP best,
  forecast 1 h RF best (XGB +0.8 W), forecast 3 h MLP and RF tied
4.3 Random vs chronological split (LR and DT re-trained with a random 80/20 split):
  a random split looks 9-13% better for the endogenous task and 13% better for the exogenous Decision Tree
  (neighbouring minutes leak into the test set); no gain for Linear Regression on exogenous data or for the
  15-minute forecasting data -> the optimism comes from minute-level neighbours and flexible models
4.4 Efficiency plot (training time vs RMSE) + table of fit / predict time and model size
4.5 Predicted vs actual (density per task) and three sample test days: clear 2016-10-04, cloudy 2016-04-27,
  winter 2018-01-23. Endogenous follows the 1-minute cloud swings; exogenous underestimates clear days
  (it predicts an average day); 1 h forecasts follow the trend, 3 h forecasts lag
4.6 Residuals: exogenous error is largest at low irradiance; endogenous nRMSE ~4-7% every month except
  December (13%, likely snow / partial shading below the fault threshold — candidate for the anomaly part);
  2017 (4k rows) has very high forecast errors (sparse data)
4.7 Feature importance: endogenous = irradiance (0.998); exogenous = ambient temperature + calendar;
  forecast 1 h = irradiance at the origin, then time of day; forecast 3 h = time of day, then irradiance
4.8 Best model per task (among models tied with the most accurate, the fastest to train):
  endogenous MLP 28.3 W, exogenous MLP 217.2 W, forecast 1 h Random Forest 140.0 W,
  forecast 3 h Random Forest 180.4 W (tied with MLP, faster) -> artifacts/metrics/system_4_eval_best_models.csv
  The endogenous MLP is the normal-behaviour reference for the anomaly detection

15. System 1283 — Generalization (notebooks *_1283.ipynb: 1, 2, 3A-3D, 4; moved on 2026-10-20 to
    ~/Desktop/solar-data-all/system_1283_notebooks/, only their results stay in artifacts/metrics/system_1283_*)
Copies of the system-4 notebooks made by a script (column renames + the changes below); same models, hyperparameters
and rules, so the two systems are comparable. System 1283: one year (2015), 15-second readings, ~250 kW inverter.
Data findings:
  dc_power__1136 = inv1 + inv2 DC power, but inverter 1 logged no DC data in 2015 -> the target is inverter 2 only
  (= V x I; AC meters show both inverters producing); the irradiance sensor is on the same plane, so the rule still holds
  Extra sentinel codes 63,984,001 = 7999² and 127,968,002 (V x I of two -7999 codes): 470 rows
  Jun-Nov only short bursts of a few minutes every few hours are logged (dense data: Jan-May and Dec)
Notebook 1: 15 s -> 1-min means (>= 2 of 4 readings): 204,186 minutes; daylight (> 50 W/m²) 84,346 rows
  (5 x median rule removed 7 rows at ~250 kW); k = 202.35 W per W/m²
  Faults: 18,419 rows (21.8%), 107 events (median 28 min, longest 640 min), mostly Nov-Mar (snow);
  Feb 22-28 is 100% flagged. The low side of the ratio histogram is a long tail (partial snow), not only a spike;
  FAULT_RATIO 0.5 kept (shallow valley 0.5-0.6) -> artifacts/metrics/system_1283_fault_band.png
  15-min grid: 35,024 slots, 13,521 valid (38.6%), 1,307 with a fault
Notebook 2: 65,927 normal rows; Pearson r: irradiance 0.977, module temp 0.841, ambient 0.506, wind -0.169 (weak);
  no collinear pair. Same configs as system 4 (endogenous = irradiance + module temp; exogenous = ambient + calendar)
  Split: test = last 7 days of every month (a chronological 80/20 would train on Jan-May only):
  estimation 54,947 train / 10,980 test (66 test days; Feb has no test rows, all faults);
  forecasting 3,234 / 403 test slots (1 h) and 3,205 / 403 (3 h), only 17-19 test days
  Daylight / clear-sky calendar profiles use (month, slot) instead of (week, slot): with the last week of every month
  held out, test weeks would have no training weeks of their own
Notebooks 3A-3D: CV on every training row (CV_SUBSAMPLE_STEP = 1); test results (W, nRMSE % of mean power):
  Task             best          RMSE       nRMSE   R2     others
  Endogenous       MLP           10,194 W   11.4%   0.966  RF 10,616 (tied), DT 10,918, XGB 11,119, LR 11,265
  Exogenous        MLP           33,651 W   37.5%   0.634  RF 34,955, LR 35,017, XGB 36,710 (all tied), DT 39,571
  Forecast 1 h     MLP           20,178 W   24.3%   0.868  LR 21,231 (tied), RF 23,273, XGB 26,474, DT 26,797;
                                                           persistence 32,672, clear-sky persistence 21,435
  Forecast 3 h     MLP           26,088 W   31.4%   0.780  LR 27,295, XGB 34,776 (tied), RF 32,014, DT 38,197;
                                                           persistence 66,108, clear-sky persistence 25,961
  Exogenous MLP CV is unstable (113k ± 96k W: some folds extrapolate badly); its test score is fine
Notebook 4 (+ new 4.9 comparison with system 4 -> artifacts/metrics/system_1283_eval_vs_system_4.csv/.png):
  Best (fastest among tied): endogenous RF, exogenous / 1 h / 3 h Linear Regression
  Endogenous error highest in winter (Nov 26%, Dec 19%, Jan 18% vs 7-12% Mar-Sep): partial snow below the fault
  threshold, as in system 4's December -> anomaly notebook
  13 forecast test slots in February with power 0 but no fault flag: the irradiance sensor was also covered
  (irradiance < 50 W/m²), so the rule cannot see it -> sensor-fault example for notebook 5
  Random vs month-block split: random looks better only for the exogenous Decision Tree (+40%); with the small
  forecasting set the random split is even worse (-35% LR), so the leakage effect is not stable on this system
E3 answers (system 4 -> system 1283):
  Ranking: MLP best in 3 of 4 tasks on both systems (and in all 4 on 1283); LR last on system 4 but tied with the best
  for 1283 forecasting (3,200 training rows favour the simple model); DT worst or near-worst in both
  Endogenous vs exogenous gap holds: RMSE ratio 7.7 (system 4) vs 3.3 (system 1283); R2 0.99/0.48 vs 0.97/0.63
  (the clearer sky / stronger seasonal cycle of 1283 helps the calendar model, the snow hurts the endogenous one)
  Forecast skill vs persistence holds (0.34 / 0.55 vs 0.38 / 0.61 at 1 h / 3 h), but vs clear-sky persistence it
  shrinks to 0.06 at 1 h and 0.00 at 3 h for 1283 (0.15 / 0.24 for system 4): with ~3k training slots the models
  learn little beyond the daily clear-sky shape

16. Anomaly Detection (notebooks/5_Anomaly_Detection.ipynb; copy notebooks/1283/5_Anomaly_Detection_1283.ipynb)
Reference = endogenous model chosen in notebook 4 (system 4 MLP, system 1283 Random Forest) with its scaler; all daylight rows
Detectors: rule (is_fault, reproduced exactly from k), residual, Isolation Forest
  Residual: relative residual = (actual - expected) / max(expected, 0.1 x P_REF), trailing 15-min median; thresholds
  median ± 5 robust σ (1.4826 x MAD) of the training-period normal rows -> system 4: -0.210 / +0.226 (σ 0.044);
  1283: -0.342 / +0.375 (σ 0.072). First version used the 0.5% / 99.5% quantiles: low threshold -0.43 because the
  "normal" rows contain unflagged partial losses down to the rule boundary -> 0% partial losses detected; replaced
  (longer windows 30/60 min did not help: the error is per-day model bias, not minute noise)
  Isolation Forest: power, irradiance, module temp, point residual; fitted on 200k training-period rows (labels unused),
  contamination 5%
Evaluation without circularity:
  Real test period: rule labels judge only full outages (power < 0.1 x k x irradiance)
  Synthetic events on clean test days (no rule flag 09:30-14:30), 120 min from 10:00-12:00: full outage (power 0),
  partial loss (power x 0.6-0.9), sensor fault (irradiance x 0.6-0.8); 25% of clean days reserved for false alarms;
  event detected if >= 50% of its minutes flagged. System 4: 100 events/type on 322 days; 1283: 33/type on 10 days
  (10 rounds reuse the days)
Results (event recall %, false alarms % on reserved clean days):
                     full outage  partial loss  sensor fault  false alarms  real full-outage recall
  System 4 rule          100           0             0            0.0            100
           residual      100          58            97            4.9             99.5
           iforest        99          18            19            0.8             78.3
  System 1283 rule       100           0             0            0.0            100
           residual      100          30            45            2.2            100
           iforest        67          36            73            3.5             10.2
  System 4 partial losses by size (residual): 10-20% 0, 20-30% 82, 30-40% 100
  Trade-off (system 4, residual): 3σ 86% partial losses at 9.7% false alarms; 5σ 59% at 4.9%; 8σ 18% at 1.8%
Agreement (system 4 test period): all three 3.8% of rows (outages, ratio 0.03); residual only 4.6% (actual/expected 0.72,
  low light, partial-loss candidates, highest Dec / Apr-Jul); iforest only 0.6% (very high irradiance, normal power)
Example days (system 4): longest outage 2016-04-17 (all day ~0 W); short events 2016-02-15; 2016-12-08 power capped at
  ~700 W vs ~1,050 W expected (snow / partial loss: rule flags only the morning, residual the whole day);
  2016-04-29 power above expected for hours on a dull day (irradiance sensor reading low?)
Outputs: artifacts/metrics/system_<id>_anomaly_summary.csv, _agreement.csv, _residual_sensitivity.csv/.png,
  _recall_by_type.png, _examples.png

17. Results Summary (notebooks/6_Results_Summary.ipynb)
Reads the saved metrics of notebooks 4 and 5 for both systems; nothing is trained
Tables -> artifacts/metrics/thesis_table_1_estimation.csv, _2_forecasting.csv, _3_best_models.csv, _4_anomaly_detection.csv
Figures -> thesis_figure_A_nrmse_per_task.png, _B_forecast_skill.png, _C_anomaly_detection.png
Conclusions (6.6): irradiance is the key input (R2 0.99 / 0.97 vs 0.48 / 0.63 without it); model choice matters less
  than inputs (MLP most accurate in 7 of 8 task/system cases); forecasts beat persistence but only modestly beat
  clear-sky persistence (system 4 0.15 / 0.24, 1283 0.06 / 0.00); the rule only sees full outages; the model residual
  finds partial losses and sensor faults; detection quality follows prediction quality (1283 weaker)

18. Local prediction portal (app/app.py, Streamlit; run from thesis/: streamlit run app/app.py -> http://localhost:8501)
System 4 only; portable: app/export_bundle.py copies the chosen models + scalers (.pkl), the notebook-5 detector
  settings and the daylight (month, 15-min slot) list into app/bundle/ (145 KB, bundle.json); the app reads only
  bundle/, so app/ can be copied to another PC (pip install -r app/requirements.txt; scikit-learn pinned to 1.9.0)
  and served to the local network with --server.address 0.0.0.0; verified by running a copy of app/ outside the project
Tab "Predict power": (a) any date (also future) + time + ambient temperature -> exogenous model, plus the whole-day
  curve and daily energy at that temperature (night slots = 0, daylight from the (month, 15-min slot) irradiance
  profile of the 15-min grid); (b) irradiance + module temperature (or NOCT estimate from ambient) -> endogenous model
Tab "Fault check": irradiance, module temperature (or NOCT estimate) and measured DC power -> expected power
  (endogenous model), deviation in W and %, notebook-1 rule, and a severity colour from the distance to the median in
  robust σ of the point relative residual (notebook 5, section 5.9, saved in system_<id>_anomaly_detector.json):
  green < 3σ normal, yellow 3-5σ watch, orange 5-8σ fault (5σ = residual-detector threshold), red > 8σ severe
  (levels of the sensitivity analysis 5.5.3); σ = 4.7% (system 4) -> ±15% / ±24% / ±38%; σ = 8.1% (1283) ->
  ±26% / ±42% / ±67%; under-production = partial loss or full outage (< 10% of expected), over-production =
  irradiance sensor reading low; a table gives the power range of every colour for the entered conditions
Example: system 4, 12:00 26/10/2026, 20 °C -> 782 W (typical day, ~5.6 kWh for the day); at 800 W/m² and 40 °C module
  (expected 698 W): 610 W green, 560 W yellow, 480 W orange (partial loss 31%), 300 W / 20 W red
Smoke-tested with streamlit AppTest; streamlit added to myenv; configs/requirments.txt updated to the installed versions
  (the old pins, e.g. scikit-learn 1.4, could not load the saved models)

19. Simplification (2026-10-20): README and structure shortened, tobedone trimmed to open points; train_config.yaml
  removed (only mentioned in comments; the XGBoost values are in the notebooks); data/features/ no longer tracked by
  Git (recreated by notebook 2); all charts / result tables incl. system 1283 and thesis_* now tracked; folder
  'functions ' renamed to 'functions'
  Earlier: README.md rewritten: environment, data download (functions/ scripts), notebook order (incl. 1283 and notebook 6),
  the app (local, portable copy, local-network access)

11. Review of the results (findings that led to the cleaning revision in 6b)
Fault band: 0.69% of test rows (strong sun, almost no power) caused ~69% of the squared error (RMSE 69 W -> 38 W without them)
The dc_power ≥ 20 W floor selected rows using the target and hid most daylight outages
Before the revision the top four endogenous models were within 1 W while the CV std was 10-15 W (tied);
after it the CV std is 1.1-1.5 W and Linear Regression is clearly behind the other four (32.0 vs 28.3-29.1 W)
Framing decided: the title's "Πρόβλεψη" = estimation of the power from same-time inputs (+ forecasting 1 h / 3 h, planned);
the power models are the normal-behaviour baseline for the anomaly detection (residual = actual − expected)

--------------------------------------------------------------------
What You Have Now
SystemResolutionRowsKey Sensorssystem_41-min4,206,520dc_power, poa_irradiance, ambient_temp, module_temp, dc_voltage, dc_currentsystem_1015-min35,130same + inverter_tempsystem_331-min524,159same + power_factorsystem_3415-min34,773same + wind_speed, humidity, error_codesystem_3515-min34,816same as 34system_361-min508,797dc_power, irradiance×2, module_temp×8, photodiodes×6system_5015-min29,989dc_power, poa_irradiance, ambient_temp, dc_pos/neg voltage & currentsystem_5115-min21,594same as 50system_120015-min37,152dc_power, poa_irradiance, per-inverter data×6, wind_speedsystem_127615-min34,976dc_power, poa_irradiance, ambient_temp, wind_speed, humidity, error_codesystem_127715-min33,645same as 1276system_12831-min817,000dc_power, poa_irradiance, ambient_temp, module_tempsystem_12891-min525,470dc_power, poa_irradiance, ambient_temp, module_temp
Total: ~6.8 million rows, 638 MB cleaned data




