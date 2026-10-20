# Project Structure

```text
thesis/
├── README.md              # setup, data, notebook order, app
├── structure.MD           # this map
├── alreadydone.MD         # every completed step with its results
├── tobedone.MD            # open tasks and thesis-text notes
├── app/                   # local Streamlit app for system 4 (runs on its own)
│   ├── app.py             #   predict power, fault check
│   ├── export_bundle.py   #   copies the chosen models and settings into bundle/
│   ├── requirements.txt
│   └── bundle/            #   2 models + 2 scalers (.pkl), bundle.json
├── configs/
│   ├── requirments.txt    # full environment
│   └── sensors.txt        # sensor list of every PVDAQ system
├── notebooks/             # 1 → 2 → 3A-3D → 4 → 5 → 6 (system 4; notebook 6 also reads the 1283 results)
│                          #   + example_simple_training.ipynb (untracked, simple training demo, saves nothing)
├── data/                  # not in Git: merged/, cleaned/, features/ (+ raw pvdata/, pvdata_processed/, metrics/)
└── artifacts/
    ├── metrics/           # all charts and result tables: system_4_*, system_1283_*, thesis_table_*, thesis_figure_*
    ├── experiments/       # CV folds and test predictions per task (system 4)
    ├── models/            # trained models (.pkl, not in Git), scalers, anomaly-detector settings
    └── original_features_scatterplots-heatmaps/   # early correlation plots of the 13 collected systems
```



Outside `thesis/` (in `~/Desktop/solar-data-all/`): `functions/` (download and merge scripts), `myenv/` (Python
environment) and `system_1283_notebooks/` (the notebooks that produced the system-1283 results; to re-run them,
copy them into `thesis/notebooks/`).

## Who writes and reads each artifact (`<id>` = 4 or 1283, `<task>` = endogenous / exogenous / forecast_1h / forecast_3h)

| Written by | Files | Read by |
|---|---|---|
| 1 | metrics/system_<id>_raw_dc_power_vs_date.png, _fault_band.png, _cleaned_<Y_FEATURE>_vs_date.png (one per value of Y_FEATURE) | — (thesis figures) |
| 2 | metrics/system_<id>_feature_statistics.csv, _feature_distributions.png, _pearson_<target>.csv/.png, _correlation_matrix.csv, _correlation_heatmap.png, _scatter_<target>.png | — |
| 2 | metrics/system_<id>_feature_configs.csv (where data/features/<group> is, split, row counts) | 3A-3D, 4, 5, 6 |
| 2 | models/scaler_group_<id>_<task>.pkl | 4, 5, app/export_bundle.py |
| 3A / 3B / 3C / 3D | metrics/system_<id>_<task>_model_metrics.csv, _model_rmse.png | 4, export_bundle |
| 3A / 3B / 3C / 3D | experiments/system_<id>_<task>_cv_folds.csv, _test_predictions.csv | 4 (predictions) |
| 3A / 3B / 3C / 3D | models/system_<id>_<task>_<model>.pkl (5 models per task) | 4, 5, export_bundle |
| 4 | metrics/system_<id>_eval_metrics_all_tasks.csv | 6 |
| 4 | metrics/system_<id>_eval_best_models.csv | 5, 6, export_bundle |
| 4 | metrics/system_<id>_eval_efficiency / _predicted_vs_actual / _sample_days / _residuals / _permutation_importance .png, system_4_eval_random_vs_chronological.csv, system_1283_eval_random_vs_month_block.csv, system_1283_eval_vs_system_4.csv/.png | — |
| 5 | metrics/system_<id>_anomaly_summary.csv | 6 |
| 5 | metrics/system_<id>_anomaly_agreement.csv, _residual_sensitivity.csv/.png, _recall_by_type.png, _examples.png | — |
| 5 | models/system_<id>_anomaly_detector.json (thresholds, σ) | export_bundle -> app |
| 6 | metrics/thesis_table_1-4_*.csv, thesis_figure_A-C_*.png | — (final thesis tables / figures) |
| functions/ scripts (not notebooks) | metrics/metrics_summary.csv, metrics/sensors.txt (copy of configs/sensors.txt) | — |
| early exploration (before the refactor, 2026-09) | original_features_scatterplots-heatmaps/ (13 systems in sensor groups), metrics/system_4_poa_irradiance_over_time.png | — |
