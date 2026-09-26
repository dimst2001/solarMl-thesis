"""Serialization helpers for model features and fitted preprocessing objects."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_and_export_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    group_name: str,
    project_root: Path,
) -> dict[str, object]:
    """Fit MinMax scaling on training features and export aligned partitions."""
    if not group_name.startswith("group_") or not group_name.replace("_", "").isalnum():
        raise ValueError("group_name must be a valid group name.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns must match in order.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training feature and target indices must align.")
    if not X_test.index.equals(y_test.index):
        raise ValueError("Test feature and target indices must align.")
    for name, frame in (
        ("X_train", X_train),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_test", y_test),
    ):
        if frame.empty:
            raise ValueError(f"{name} cannot be empty.")
        if frame.isna().any().any():
            raise ValueError(f"{name} contains missing values.")

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    root = Path(project_root)
    feature_directory = root / "data" / "features" / group_name
    model_directory = root / "artifacts" / "models"
    feature_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    partitions = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
    }
    output_paths = {}
    for name, frame in partitions.items():
        output_path = feature_directory / f"{name}_{group_name}.csv"
        frame.to_csv(output_path, index=True)
        output_paths[name] = output_path

    scaler_path = model_directory / f"scaler_{group_name}.pkl"
    with scaler_path.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    return {
        **partitions,
        "scaler": scaler,
        "feature_directory": feature_directory,
        "output_paths": output_paths,
        "scaler_path": scaler_path,
    }