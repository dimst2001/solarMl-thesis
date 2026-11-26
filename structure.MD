# Project Structure

```text
thesis/
├── README.md              # how to replicate: A = run the app, B = data + notebooks + app bundle
├── structure.MD           # this map
├── alreadydone.MD         # every completed step with its results
├── tobedone.MD            # open tasks and thesis-text notes
├── docs/
│   ├── PIPELINE.md        # pipeline explained in words and figures (links artifacts/metrics/*.png)
│   └── figures/           #   system_4_preview_days.png
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
├── data/                  # only data/sample/ in Git (3 days of system 4, raw + cleaned, README); not in Git: merged/, cleaned/, features/ (+ raw pvdata/, pvdata_processed/, metrics/)
└── artifacts/
    ├── metrics/           # all charts and result tables: system_4_*, system_1283_*, thesis_table_*, thesis_figure_*
    ├── experiments/       # CV folds and test predictions per task (system 4)
    ├── models/            # trained models (.pkl, not in Git), scalers, anomaly-detector settings
    └── original_features_scatterplots-heatmaps/   # early correlation plots of the 13 collected systems
```