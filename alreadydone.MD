 
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

6b. Preprocessing (notebooks/1_Data_preproccesing.ipynb, system 4)
Loaded data/merged/system_4_merged.csv: 4,206,520 rows x 12 columns (no duplicates, no invalid timestamps)
Sentinel codes (-99999, -7999) -> NaN; no unit conversions needed for system 4
Removed rows below the 20 W DC-power floor (night / no output): 2,606,809 rows
Removed high outliers above 5 x median (median 390.49 W, cutoff 1,952.45 W): 82 rows
Kept dc_power__314, poa_irradiance__313, module_temp_1__321, ambient_temp__320, das_temp__325
Dropped rows with any remaining missing value (complete-case, no imputation)
Saved data/cleaned/system_4_cleaned.csv: 1,463,640 rows, no missing values

------------------------------------------------------------------------------------------
7. Feature Selection (notebooks/2_Feature_Selection.ipynb, system 4)
Target: dc_power__314. Candidates: poa_irradiance__313, module_temp_1__321, ambient_temp__320, das_temp__325
Basic statistics per column (no near-constant or highly skewed columns) -> artifacts/metrics/system_4_feature_statistics.csv
Distributions (histogram + box plot per column)
Pearson / Spearman correlation with dc_power: 313 r=0.97, 321 r=0.66, 325 r=0.27, 320 r=0.13
Correlation heatmap: ambient_temp__320 and das_temp__325 are collinear (r=0.93) -> artifacts/metrics/system_4_correlation_heatmap.png
Scatter plots per feature (50k-row sample): 313 is linear with dc_power; there is a small low-power band at high irradiance (possible outages/curtailment)
Two feature configurations defined (FEATURE_CONFIGS):
  endogenous = 313 + 321 (drop 320, 325) -> built
  exogenous = 320 + datetime features -> placeholder, disabled
Endogenous: chronological 80/20 train/test split (train 2007-08 -> 2016-01, test 2016-01 -> 2023-01),
MinMaxScaler fitted on train only, saved to data/features/group_4_endogenous/ + artifacts/models/scaler_group_4_endogenous.pkl
Config summary -> artifacts/metrics/system_4_feature_configs.csv

8. Code cleanup
Both notebooks simplified to system 4 only (same outputs, verified byte-for-byte on the cleaned CSV, train/test CSVs, scaler and plots)
All logic lives in the notebooks; the unused src/ package and tests/ folder were removed

--------------------------------------------------------------------
What You Have Now
SystemResolutionRowsKey Sensorssystem_41-min4,206,520dc_power, poa_irradiance, ambient_temp, module_temp, dc_voltage, dc_currentsystem_1015-min35,130same + inverter_tempsystem_331-min524,159same + power_factorsystem_3415-min34,773same + wind_speed, humidity, error_codesystem_3515-min34,816same as 34system_361-min508,797dc_power, irradiance×2, module_temp×8, photodiodes×6system_5015-min29,989dc_power, poa_irradiance, ambient_temp, dc_pos/neg voltage & currentsystem_5115-min21,594same as 50system_120015-min37,152dc_power, poa_irradiance, per-inverter data×6, wind_speedsystem_127615-min34,976dc_power, poa_irradiance, ambient_temp, wind_speed, humidity, error_codesystem_127715-min33,645same as 1276system_12831-min817,000dc_power, poa_irradiance, ambient_temp, module_tempsystem_12891-min525,470dc_power, poa_irradiance, ambient_temp, module_temp
Total: ~6.8 million rows, 638 MB cleaned data




