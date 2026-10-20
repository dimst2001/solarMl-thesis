"""Copy everything the local app needs for system 4 into app/bundle/ (about 150 KB).

Run from the thesis folder after notebooks 1-5:
    python app/export_bundle.py
Afterwards the app/ folder is self-contained: it no longer needs data/, artifacts/ or the notebooks.
"""
import json
import pickle
import shutil
from pathlib import Path

import pandas as pd

SYSTEM_ID = 4
TASKS = {"endogenous": "Estimation, endogenous", "exogenous": "Estimation, exogenous"}
MODEL_SLUGS = {"Linear Regression": "linear_regression", "Decision Tree": "decision_tree",
               "Random Forest": "random_forest", "MLP": "mlp", "XGBoost": "xgboost"}
DAYLIGHT_IRRADIANCE_W_M2 = 50.0

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
BUNDLE_DIR = APP_DIR / "bundle"


def daylight_slots():
    """(month, 15-minute slot) pairs where the 95th percentile of the measured irradiance is above 50 W/m²."""
    grid = pd.read_csv(PROJECT_ROOT / "data" / "cleaned" / f"system_{SYSTEM_ID}_cleaned_15min.csv",
                       index_col="datetime", parse_dates=True)
    irradiance = grid[[c for c in grid.columns if "irradiance" in c.lower()][0]].dropna()
    key = [irradiance.index.month, irradiance.index.hour * 4 + irradiance.index.minute // 15]
    daylight = irradiance.groupby(key).quantile(0.95).gt(DAYLIGHT_IRRADIANCE_W_M2)
    return [[int(month), int(slot)] for month, slot in daylight[daylight].index]


def main():
    BUNDLE_DIR.mkdir(exist_ok=True)
    best = pd.read_csv(METRICS_DIR / f"system_{SYSTEM_ID}_eval_best_models.csv", index_col="task")
    info = {"system": SYSTEM_ID, "models": {}}
    for task, label in TASKS.items():
        name = best.loc[label, "chosen"]
        shutil.copy(MODELS_DIR / f"system_{SYSTEM_ID}_{task}_{MODEL_SLUGS[name]}.pkl", BUNDLE_DIR / f"{task}_model.pkl")
        shutil.copy(MODELS_DIR / f"scaler_group_{SYSTEM_ID}_{task}.pkl", BUNDLE_DIR / f"{task}_scaler.pkl")
        metrics = pd.read_csv(METRICS_DIR / f"system_{SYSTEM_ID}_{task}_model_metrics.csv", index_col="model").loc[name]
        info["models"][task] = {"name": name, "test_RMSE": float(metrics["test_RMSE"]),
                                "test_R2": float(metrics["test_R2"])}
    info["detector"] = json.loads((MODELS_DIR / f"system_{SYSTEM_ID}_anomaly_detector.json").read_text())
    info["daylight_slots"] = daylight_slots()
    (BUNDLE_DIR / "bundle.json").write_text(json.dumps(info, indent=1))

    # Check that the copied files load and say which libraries the app needs to unpickle them.
    for task in TASKS:
        with (BUNDLE_DIR / f"{task}_model.pkl").open("rb") as file:
            pickle.load(file)
    needs_xgboost = any(model["name"] == "XGBoost" for model in info["models"].values())
    size_kb = sum(path.stat().st_size for path in BUNDLE_DIR.iterdir()) / 1024
    print(f"Bundle written to {BUNDLE_DIR} ({size_kb:.0f} KB): "
          + ", ".join(f"{task} = {model['name']}" for task, model in info["models"].items()))
    if needs_xgboost:
        print("An XGBoost model was chosen: add xgboost to app/requirements.txt.")


if __name__ == "__main__":
    main()
