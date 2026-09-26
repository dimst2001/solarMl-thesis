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
    group_id: int,
    project_root: Path,
) -> dict[str, object]:
    """Fit MinMax scaling on training features and export aligned partitions."""
    if not isinstance(group_id, int) or isinstance(group_id, bool) or group_id < 1:
        raise ValueError("group_id must be a positive integer.")
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
    feature_directory = root / "data" / "features" / f"g{group_id}"
    model_directory = root / "artifacts" / "models"
    feature_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    partitions = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
    }
    for name, frame in partitions.items():
        frame.to_parquet(feature_directory / f"{name}.parquet", index=True)

    scaler_path = model_directory / f"scaler_g{group_id}.pkl"
    with scaler_path.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    return {
        **partitions,
        "scaler": scaler,
        "feature_directory": feature_directory,
        "scaler_path": scaler_path,
    }