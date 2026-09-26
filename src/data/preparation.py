"""Orchestration for preparing one selected solar-system feature group."""

from pathlib import Path

import pandas as pd

from src.data.cleaning import target_protected_impute
from src.models.evaluator import temporal_train_test_split
from src.utils.io import scale_and_export_features


TARGET_NAMES = ("dc_power", "poa_irradiance")
TIMESTAMP_NAMES = ("datetime", "timestamp", "time")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_input_file(cleaned_directory: Path, system_id: int) -> Path:
    candidates = [
        cleaned_directory / f"system_{system_id}.csv",
        cleaned_directory / f"system_{system_id}_clean.csv",
        cleaned_directory / f"system_{system_id}.parquet",
        cleaned_directory / f"system_{system_id}_clean.parquet",
    ]
    input_path = next((path for path in candidates if path.is_file()), None)
    if input_path is None:
        raise FileNotFoundError(
            f"No cleaned data found for system {system_id}; checked {candidates}"
        )
    return input_path


def _resolve_registry(artifacts_directory: Path, system_id: int) -> Path:
    registry_directory = artifacts_directory / "feature_selected_scatterplots-heatmaps"
    candidates = [
        registry_directory / f"selected_features_group_{system_id}_1283_1289.txt",
        registry_directory / f"selected_features_group_base_{system_id}_1283_1289.txt",
        registry_directory / f"selected_features_group_{system_id}.txt",
    ]
    registry_path = next((path for path in candidates if path.is_file()), None)
    if registry_path is None:
        raise FileNotFoundError(
            f"No feature registry found for system {system_id}; checked {candidates}"
        )
    return registry_path


def _resolve_column(name: str, columns: pd.Index) -> str:
    if name in columns:
        return name
    matches = [column for column in columns if column.startswith(f"{name}__")]
    if len(matches) != 1:
        raise ValueError(f"Expected one column for {name!r}, found {matches}")
    return matches[0]


def prepare_group_features(
    group_id: int,
    system_id: int = 4,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Load, prepare, scale, and export a selected feature group."""
    root = _project_root() if project_root is None else Path(project_root)
    cleaned_directory = root / "data" / "cleaned"
    artifacts_directory = root / "artifacts"
    data_path = _resolve_input_file(cleaned_directory, system_id)
    registry_path = _resolve_registry(artifacts_directory, system_id)

    if data_path.suffix == ".parquet":
        raw_data = pd.read_parquet(data_path)
    else:
        raw_data = pd.read_csv(data_path, low_memory=False)

    timestamp_column = next(
        (name for name in TIMESTAMP_NAMES if name in raw_data.columns), None
    )
    if timestamp_column is None:
        raise KeyError(f"No timestamp column found in {list(raw_data.columns)}")

    selected_names = [
        line.strip()
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    independent_names = [
        name for name in selected_names if name not in TARGET_NAMES
    ]
    if not independent_names:
        raise ValueError("The registry contains no independent features.")

    timestamp = pd.to_datetime(raw_data[timestamp_column], errors="coerce")
    target_columns = [_resolve_column(name, raw_data.columns) for name in TARGET_NAMES]
    feature_columns = [
        _resolve_column(name, raw_data.columns) for name in independent_names
    ]
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError(f"Registry resolves to duplicate feature columns: {feature_columns}")
    if set(target_columns).intersection(feature_columns):
        raise ValueError("A target column was included as an independent feature.")

    model_data = raw_data[[*feature_columns, *target_columns]].copy()
    model_data.index = pd.DatetimeIndex(timestamp, name=timestamp_column)
    model_data = model_data.loc[~model_data.index.isna()].sort_index(kind="stable")
    model_data = model_data.apply(pd.to_numeric, errors="raise")
    model_data = model_data.dropna(subset=target_columns)

    train_raw, test_raw = temporal_train_test_split(model_data)
    ordered_valid_data = pd.concat([train_raw, test_raw])
    X, y, imputation_means = target_protected_impute(
        ordered_valid_data,
        target_columns=target_columns,
        feature_columns=feature_columns,
        fit_data=train_raw,
    )
    train_size = len(train_raw)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    y_train.columns = list(TARGET_NAMES)
    y_test.columns = list(TARGET_NAMES)

    outputs = scale_and_export_features(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        group_id=group_id,
        project_root=root,
    )
    outputs.update(
        {
            "system_id": system_id,
            "data_path": data_path,
            "registry_path": registry_path,
            "imputation_means": imputation_means,
        }
    )
    return outputs