"""Orchestration for preparing one selected solar-system feature group."""

from pathlib import Path

import pandas as pd

from src.data.cleaning import target_protected_impute
from src.models.evaluator import temporal_train_test_split
from src.utils.io import scale_and_export_features


TARGET_NAMES = ("dc_power", "poa_irradiance")
TIMESTAMP_NAMES = ("datetime", "timestamp", "time")
GROUP_CONFIGS = {
    1: {
        "name": "group_base_4_1283_1289",
        "systems": (4, 1283, 1289),
        "registry_names": ("group_4_1283_1289", "group_base_4_1283_1289"),
    },
    2: {"name": "group_34_35", "systems": (34, 35), "registry_names": ("group_34_35",)},
    3: {"name": "group_50_51", "systems": (50, 51), "registry_names": ("group_50_51",)},
    4: {
        "name": "group_1276_1277",
        "systems": (1276, 1277),
        "registry_names": ("group_1276_1277",),
    },
    5: {"name": "group_10", "systems": (10,), "registry_names": ("group_10",)},
    6: {"name": "group_33", "systems": (33,), "registry_names": ("group_33",)},
    7: {"name": "group_36", "systems": (36,), "registry_names": ("group_36",)},
    8: {"name": "group_1200", "systems": (1200,), "registry_names": ("group_1200",)},
}


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


def _resolve_registry(artifacts_directory: Path, registry_names: tuple[str, ...]) -> Path:
    registry_directory = artifacts_directory / "feature_selected_scatterplots-heatmaps"
    candidates = [
        registry_directory / f"selected_features_{name}.txt"
        for name in registry_names
    ]
    registry_path = next((path for path in candidates if path.is_file()), None)
    if registry_path is None:
        raise FileNotFoundError(
            f"No feature registry found for names {registry_names}; checked {candidates}"
        )
    return registry_path


def _matching_columns(name: str, columns: pd.Index) -> list[str]:
    if name in columns:
        return [name]
    matches = [column for column in columns if column.startswith(f"{name}__")]
    if matches:
        return matches
    return [
        column
        for column in columns
        if column.startswith(f"{name}_") and "__" in column
    ]


def _is_target_variant(name: str) -> bool:
    return any(
        name == target or name.startswith(f"{target}_")
        for target in TARGET_NAMES
    )


def _target_sensor_columns(name: str, columns: pd.Index) -> list[str]:
    return [
        column
        for column in columns
        if column == name
        or column.startswith(f"{name}__")
        or (column.startswith(f"{name}_") and "__" in column)
    ]


def prepare_group_features(
    group_id: int,
    system_id: int | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Load, prepare, scale, and export the group selected by its numeric ID."""
    if group_id not in GROUP_CONFIGS:
        raise ValueError(
            f"Unknown group_id={group_id}; choose one of {sorted(GROUP_CONFIGS)}."
        )
    group_config = GROUP_CONFIGS[group_id]
    if system_id is None:
        system_id = group_config["systems"][0]
    elif system_id not in group_config["systems"]:
        raise ValueError(
            f"system_id={system_id} is not in {group_config['name']}; "
            f"valid systems are {group_config['systems']}."
        )

    root = _project_root() if project_root is None else Path(project_root)
    cleaned_directory = root / "data" / "cleaned"
    artifacts_directory = root / "artifacts"
    registry_path = _resolve_registry(
        artifacts_directory, group_config["registry_names"]
    )

    selected_names = [
        line.strip()
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    independent_names = [name for name in selected_names if not _is_target_variant(name)]
    if not independent_names:
        raise ValueError("The registry contains no independent features.")

    feature_columns = independent_names
    target_columns = list(TARGET_NAMES)
    if set(target_columns).intersection(feature_columns):
        raise ValueError("A target column was included as an independent feature.")

    logical_columns = feature_columns
    member_frames = []
    systems_used = []
    data_paths = []
    for member_system_id in group_config["systems"]:
        try:
            member_path = _resolve_input_file(cleaned_directory, member_system_id)
        except FileNotFoundError:
            continue

        if member_path.suffix == ".parquet":
            raw_data = pd.read_parquet(member_path)
        else:
            raw_data = pd.read_csv(member_path, low_memory=False)
        timestamp_column = next(
            (name for name in TIMESTAMP_NAMES if name in raw_data.columns), None
        )
        if timestamp_column is None:
            raise KeyError(
                f"No timestamp column found in {member_path.name}: "
                f"{list(raw_data.columns)}"
            )

        timestamp = pd.to_datetime(raw_data[timestamp_column], errors="coerce")
        member_data = pd.DataFrame(index=pd.DatetimeIndex(timestamp, name=timestamp_column))
        for logical_name in logical_columns:
            matches = _matching_columns(logical_name, raw_data.columns)
            if len(matches) > 1:
                raise ValueError(
                    f"System {member_system_id} has ambiguous columns for "
                    f"{logical_name!r}: {matches}"
                )
            if matches:
                member_data[logical_name] = raw_data[matches[0]].to_numpy()
            else:
                member_data[logical_name] = float("nan")

        for target_name in target_columns:
            target_matches = _target_sensor_columns(target_name, raw_data.columns)
            target_values = raw_data[target_matches].apply(
                pd.to_numeric, errors="coerce"
            )
            if target_values.shape[1] == 0:
                member_data[target_name] = float("nan")
            elif target_name == "dc_power":
                member_data[target_name] = target_values.sum(axis=1, min_count=1).to_numpy()
            else:
                member_data[target_name] = target_values.mean(axis=1).to_numpy()

        member_data = member_data.loc[~member_data.index.isna()]
        member_frames.append(member_data)
        systems_used.append(member_system_id)
        data_paths.append(member_path)

    if not member_frames:
        raise FileNotFoundError(
            f"No cleaned data found for any system in {group_config['name']}: "
            f"{group_config['systems']}"
        )

    model_data = pd.concat(member_frames).sort_index(kind="stable")
    model_data = model_data.apply(pd.to_numeric, errors="coerce")
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
    group_name = group_config["name"]

    outputs = scale_and_export_features(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        group_name=group_name,
        project_root=root,
    )
    outputs.update(
        {
            "group_id": group_id,
            "group_name": group_name,
            "system_id": system_id,
            "systems_used": systems_used,
            "data_paths": data_paths,
            "registry_path": registry_path,
            "imputation_means": imputation_means,
        }
    )
    return outputs