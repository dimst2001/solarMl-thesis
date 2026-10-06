# Project Structure

The active workflow is notebook-first: the preprocessing notebook contains the single-system loading, audit, raw visualization, and cleaning code. `src/data/` is reserved for helpers when the workflow is generalized to multiple systems. Model utilities are retained for a later stage but are not used by the current preprocessing notebook.

```text
thesis/
├── README.md                         # Project overview and setup notes
├── alreadydone.MD                    # Work completed and current dataset summary
├── tobedone.MD                       # Remaining thesis tasks
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
│   ├── cleaned/                      # Notebook-generated cleaned Parquet outputs
│   ├── metrics/                      # Local sensor metadata
│   └── features/                     # Reserved for later feature-selection/model data
├── notebooks/
│   └── 1_Data_preproccesing.ipynb    # Active single-system exploration and preprocessing
├── src/
│   ├── __init__.py                   # Package marker
│   ├── data/
│   │   ├── __init__.py               # No automatic imports or filesystem side effects
│   │   ├── initial_data.py           # Note: active loader/audit currently lives in notebook
│   │   ├── cleaning.py               # Note: active single-system cleaner currently lives in notebook
│   │   ├── feature_selection.py      # Deferred until cleaning decisions are stable
│   │   ├── preparation.py            # Former grouped-system pipeline, retired for now
│   │   └── multi_system.py           # TODO stubs for NaN handling, imputation, and feature dropping
│   ├── models/
│   │   ├── __init__.py               # Package marker
│   │   └── evaluator.py              # Dormant temporal split utility for future modeling
│   ├── utils/
│   │   ├── __init__.py               # Package marker
│   │   └── io.py                     # Dormant scaling/export utility for future modeling
│   └── visualization/
│       └── __init__.py               # Empty package reserved for future plots
├── tests/
│   └── test_model_utilities.py       # Tests for retained temporal split and scaling utilities
├── artifacts/                       # Generated outputs
│   ├── metrics/
│   ├── experiments/
│   ├── models/
│   └── original_features_scatterplots-heatmaps/
└── notebook2_test.csv               # Existing standalone sample/output file
```

The earlier structure draft listed files and directories that are not in this checkout, including `setup.py`, a root `requirements.txt`, multiple future notebooks, and feature-engineering/model modules. Add those only when the project needs them; they are not prerequisites for the current single-system workflow.
