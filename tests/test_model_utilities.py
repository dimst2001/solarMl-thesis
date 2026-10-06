import pickle

import pandas as pd
import pytest

from src.models.evaluator import temporal_train_test_split
from src.utils.io import scale_and_export_features


def test_temporal_split_sorts_and_keeps_duplicate_timestamps_together():
    timestamps = pd.to_datetime(
        [
            "2024-01-01 05:00", "2024-01-01 00:00", "2024-01-01 01:00",
            "2024-01-01 02:00", "2024-01-01 03:00", "2024-01-01 04:00",
            "2024-01-01 06:00", "2024-01-01 07:00", "2024-01-01 07:00",
            "2024-01-01 08:00",
        ]
    )
    data = pd.DataFrame({"value": range(10)}, index=timestamps)

    train, test = temporal_train_test_split(data)

    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing
    assert train.index.max() < test.index.min()
    assert len(train) == 7
    assert len(test) == 3


def test_scale_and_export_fits_training_range_and_writes_group_files(tmp_path):
    train_index = pd.date_range("2024-01-01", periods=2, freq="h")
    test_index = pd.date_range("2024-01-01 02:00", periods=1, freq="h")
    X_train = pd.DataFrame({"feature": [1.0, 3.0]}, index=train_index)
    X_test = pd.DataFrame({"feature": [5.0]}, index=test_index)
    y_train = pd.DataFrame({"dc_power": [10.0, 20.0]}, index=train_index)
    y_test = pd.DataFrame({"dc_power": [30.0]}, index=test_index)

    outputs = scale_and_export_features(
        X_train,
        X_test,
        y_train,
        y_test,
        group_name="group_4_1283_1289",
        project_root=tmp_path,
    )

    assert outputs["X_train"]["feature"].tolist() == pytest.approx([0.0, 1.0])
    assert outputs["X_test"]["feature"].tolist() == pytest.approx([2.0])
    feature_directory = tmp_path / "data" / "features" / "group_4_1283_1289"
    for filename in ("X_train", "X_test", "y_train", "y_test"):
        assert (feature_directory / f"{filename}_group_4_1283_1289.csv").is_file()
    scaler_path = tmp_path / "artifacts" / "models" / "scaler_group_4_1283_1289.pkl"
    assert scaler_path.is_file()
    with scaler_path.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    assert scaler.data_min_.tolist() == pytest.approx([1.0])
    saved_y_test = pd.read_csv(
        feature_directory / "y_test_group_4_1283_1289.csv",
        index_col=0,
        parse_dates=True,
    )
    assert saved_y_test.index.equals(test_index)