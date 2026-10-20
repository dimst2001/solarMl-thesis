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
