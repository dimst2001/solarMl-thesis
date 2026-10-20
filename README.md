# PV Power Prediction and Anomaly Detection

Thesis: *Μοντέλα Πρόβλεψης Παραγόμενης Ισχύος και Ανίχνευσης Ανωμαλιών σε Φωτοβολταϊκά Συστήματα*
· Author: Dimitrios Stampoulidis · Data: [NREL PVDAQ](https://data.openei.org/submissions/4568) (CC-BY)

Predicts the DC power of PV system 4 from its sensors and from the weather, forecasts it 1 h and 3 h ahead, and
detects faults by comparing the measured power with the expected power. System 1283 is used only as a
generalization check; its results are in `artifacts/metrics/system_1283_*`. Folder map: `structure.MD`.

## Setup

```bash
cd ~/Desktop/solar-data-all
python3.12 -m venv myenv && source myenv/bin/activate
pip install -r thesis/configs/requirments.txt
```

Keep these versions: the models are saved with pickle and need scikit-learn 1.9.0.

## Data

`data/` is not in Git. The scripts in `~/Desktop/solar-data-all/functions/` download the public data and merge it
into one CSV per system (no AWS account needed, about 20 GB):

```bash
cd ~/Desktop/solar-data-all/functions
python download_list_then_cp.py && python transform_data.py && python merge_systems.py
mkdir -p ../thesis/data && mv ../merged ../thesis/data/
```

The notebooks only need `thesis/data/merged/system_4_merged.csv`.

## Notebooks

Run them in order, each from top to bottom (JupyterLab or VS Code, kernel `myenv`):

| # | Notebook | Result |
|---|---|---|
| 1 | `1_Data_preproccesing` | cleaned data, fault flags, 15-minute grid |
| 2 | `2_Feature_Selection` | feature sets, train/test split, forecasting datasets |
| 3 | `3A`, `3B` (estimation), `3C`, `3D` (forecast 1 h / 3 h) | 5 trained models per task |
| 4 | `4_Evaluation_Comparison` | comparison and best model per task |
| 5 | `5_Anomaly_Detection` | rule vs model residual vs Isolation Forest |
| 6 | `6_Results_Summary` | thesis tables, figures A-C and conclusions (systems 4 and 1283) |

All charts and tables are saved in `artifacts/metrics/` (`thesis_table_*` and `thesis_figure_*` are the final ones).

## Local app

A Streamlit page for system 4: predict the power from a date, time and temperature (or from irradiance and module
temperature), and check a measured power for faults (green / yellow / orange / red).

```bash
cd ~/Desktop/solar-data-all/thesis && source ../myenv/bin/activate
python app/export_bundle.py   # only after re-running notebooks 3-5
streamlit run app/app.py      # open http://localhost:8501, keep the terminal open
```

The app only reads `app/bundle/` (about 150 KB), so the `app/` folder runs on any computer after
`pip install -r app/requirements.txt`. To open it from another device on the same network, start it with
`--server.address 0.0.0.0` and browse to `http://<this computer's IP>:8501`.
