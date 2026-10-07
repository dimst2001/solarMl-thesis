# Project Structure

The workflow is notebook-only and covers one PV system (system 4). Notebook 1 loads, audits, cleans and saves the merged data. Notebook 2 analyses the cleaned features, defines the feature configurations, and prepares the chronological train/test split, the MinMax scaling and the exported training data. There is no `src/` package: every step lives in the notebooks.

```text
thesis/
├── README.md                         # Project overview and setup notes
├── alreadydone.MD                    # Work completed and current dataset summary
├── tobedone.MD                       # Remaining thesis tasks, in order
├── structure.MD                      # This project map
├── .gitignore                        # Excludes environments, datasets, and generated files
├── configs/
│   ├── requirments.txt               # Existing dependency list (filename spelling retained)
│   ├── sensors.txt                   # Sensor metadata notes
│   └── train_config.yaml             # Existing training configuration
├── data/                             # Local datasets; ignored by Git
│   ├── pvdata/                       # Downloaded source data
│   ├── pvdata_processed/             # Transformed daily files
│   ├── merged/                       # Per-system merged CSV inputs
│   ├── cleaned/                      # Cleaned CSV written by notebook 1
│   ├── metrics/                      # Local sensor metadata
│   └── features/                     # Scaled train/test CSVs per feature configuration (notebook 2)
├── notebooks/
│   ├── 1_Data_preproccesing.ipynb    # Load, audit, clean and save system 4
│   └── 2_Feature_Selection.ipynb     # Feature analysis, configs, train/test split, scaling, export
└── artifacts/                        # Generated outputs
    ├── metrics/                      # Statistics, correlation tables and plots
    ├── experiments/
    ├── models/                       # Fitted scalers (and later, trained models)
    └── original_features_scatterplots-heatmaps/
```
