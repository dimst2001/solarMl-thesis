 
 What i have already done
1. Dataset Selection
Chose NREL PVDAQ (National Renewable Energy Laboratory — Photovoltaic Data Acquisition) from data.openei.org. Free, CC-BY licensed, used in hundreds of IEEE papers. From 1,862 available systems we selected 13 usable systems based on data quality (qa_status=pass), sensor coverage, and geographic diversity.
2. Data Download
Downloaded raw data from AWS S3 (oedi-data-lake) using aws s3 cp --no-sign-request. Each system stored as daily Apache Parquet files (compressed binary format).
3. Data Transformation
Converted each daily parquet file from long format (one row per metric per timestamp) to wide format (one row per timestamp, one column per sensor) using transform_data.py. Output saved to pvdata_processed/.
4. Merging
Merged all daily CSVs per system into one file per system using merge_systems.py. Output saved to merged/.

5. Data Cleaning (clean_systems.py)
System by system:
Replaced sentinel values (-99999, -7999) with NaN
Converted Fahrenheit → Celsius (systems 34, 35, 1200, 1276, 1277)
Fixed power_factor_1000 ÷ 1000 (systems 34, 35)
Fixed dc_power_hW ÷ 2 (systems 34, 35, 1276, 1277)
Fixed wind_speed_mph × 0.447
Dropped broken/constant sensors
Clipped physically impossible values
Resampled system_1283 from sub-minute → 1-minute
Output saved to cleaned/

6. Environment Setup
Python 3.12, virtual environment myenv
Libraries: pandas, pyarrow, scikit-learn, matplotlib, seaborn, xgboost, jupyterlab
VS Code + Jupyter kernel registered
thesis/ folder created as clean GitHub repo



7. Feature Selection & Artifact Generation (Notebook 1)
- Initialized core data package (`src/data/feature_selection.py` and `__init__.py`) for modular execution[cite: 1].
- Handled data cleaning edge cases: temporal datetime coercion, spatial averaging of duplicated duplicate sensors, and target variable aggregation (`dc_power` sum).
- Clustered all 13 systems into 8 distinct hardware-specific groups to stabilize covariance matrices and feature registries.
- Executed experimental feature selection tracking correlation metrics (Pearson/Spearman), multicollinearity pruning ($\rho_{max}$), and physical irradiance masking ($POA \ge 20 \ W/m^2$).
- Generated baseline and experimental feature registries (`.txt`), correlation heatmaps, and scatter plots saved to `artifacts/` and `artifacts/experiments/`[cite: 1].
- Locked optimal linear feature space parameters: Pearson $r$, Target $\ge 0.3$, Collinearity $\le 0.8$, Irradiance $\ge 20.0 \ W/m^2$.
Notebook 1 — Feature Selection ← YOU ARE HERE

Load system_4 from cleaned/

Rename columns (dc_power__314 → dc_power)

Filter daytime rows (dc_power > 0)

Basic statistics and distributions

Pearson correlation with dc_power

Correlation heatmap

Scatter plots per feature

Select final features (|r| ≥ 0.3)

Validate on systems 10, 33, 50

Save selected_features.txt

What You Have Now
SystemResolutionRowsKey Sensorssystem_41-min4,206,520dc_power, poa_irradiance, ambient_temp, module_temp, dc_voltage, dc_currentsystem_1015-min35,130same + inverter_tempsystem_331-min524,159same + power_factorsystem_3415-min34,773same + wind_speed, humidity, error_codesystem_3515-min34,816same as 34system_361-min508,797dc_power, irradiance×2, module_temp×8, photodiodes×6system_5015-min29,989dc_power, poa_irradiance, ambient_temp, dc_pos/neg voltage & currentsystem_5115-min21,594same as 50system_120015-min37,152dc_power, poa_irradiance, per-inverter data×6, wind_speedsystem_127615-min34,976dc_power, poa_irradiance, ambient_temp, wind_speed, humidity, error_codesystem_127715-min33,645same as 1276system_12831-min817,000dc_power, poa_irradiance, ambient_temp, module_tempsystem_12891-min525,470dc_power, poa_irradiance, ambient_temp, module_temp
Total: ~6.8 million rows, 638 MB cleaned data


