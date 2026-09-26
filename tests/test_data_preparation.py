import pickle

import pandas as pd
import pytest

from src.data.cleaning import target_protected_impute
from src.data.preparation import prepare_group_features
from src.models.evaluator import temporal_train_test_split
from src.utils.io import scale_and_export_features


def test_target_protected_impute_drops_missing_targets_and_uses_fit_data_means():
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    data = pd.DataFrame(
        {
            "dc_power": [1, 2, 3, 4, None, 6],
            "poa_irradiance": [5, 6, 7, 8, 9, 10],
            "feature_a": [0.0, None, 4.0, 100.0, 50.0, 60.0],
            "feature_b": [10.0, 12.0, 14.0, 16.0, 18.0, None],
        },
        index=index,
    )

    features, targets, means = target_protected_impute(
        data,
        target_columns=["dc_power", "poa_irradiance"],
        feature_columns=["feature_a", "feature_b"],
        fit_data=data.iloc[:3],
    )

    assert len(features) == 5
    assert features.index.equals(targets.index)
    assert index[4] not in targets.index
    assert means["feature_a"] == pytest.approx(2.0)
    assert means["feature_b"] == pytest.approx(12.0)
    assert features.loc[index[1], "feature_a"] == pytest.approx(2.0)
    assert features.loc[index[5], "feature_b"] == pytest.approx(12.0)
    assert not targets.isna().any().any()


def test_target_protected_impute_rejects_features_missing_from_training_fit():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    data = pd.DataFrame(
        {
            "target": [1.0, 2.0],
            "feature": [None, 3.0],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="no finite values"):
        target_protected_impute(
            data,
            target_columns=["target"],
            feature_columns=["feature"],
            fit_data=data.iloc[:1],
        )


def test_temporal_split_sorts_and_keeps_duplicate_timestamps_together():
    timestamps = pd.to_datetime(
        ["2024-01-01 05:00", "2024-01-01 00:00", "2024-01-01 01:00",
         "2024-01-01 02:00", "2024-01-01 03:00", "2024-01-01 04:00",
         "2024-01-01 06:00", "2024-01-01 07:00", "2024-01-01 07:00",
         "2024-01-01 08:00"]
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
        X_train, X_test, y_train, y_test, group_id=1, project_root=tmp_path
    )

    assert outputs["X_train"]["feature"].tolist() == pytest.approx([0.0, 1.0])
    assert outputs["X_test"]["feature"].tolist() == pytest.approx([2.0])
    feature_directory = tmp_path / "data" / "features" / "g1"
    for filename in ("X_train", "X_test", "y_train", "y_test"):
        assert (feature_directory / f"{filename}.parquet").is_file()
    scaler_path = tmp_path / "artifacts" / "models" / "scaler_g1.pkl"
    assert scaler_path.is_file()
    with scaler_path.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    assert scaler.data_min_.tolist() == pytest.approx([1.0])
    assert pd.read_parquet(feature_directory / "y_test.parquet").index.equals(test_index)


def test_prepare_group_features_resolves_registry_and_sensor_suffixes(tmp_path):
    cleaned_directory = tmp_path / "data" / "cleaned"
    registry_directory = (
        tmp_path / "artifacts" / "feature_selected_scatterplots-heatmaps"
    )
    cleaned_directory.mkdir(parents=True)
    registry_directory.mkdir(parents=True)
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
    data = pd.DataFrame(
        {
            "datetime": timestamps,
            "dc_power__1": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "poa_irradiance__2": [10.0] * 10,
            "sensor_a__3": [0.0, None, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    data.to_csv(cleaned_directory / "system_4_clean.csv", index=False)
    (
        registry_directory / "selected_features_group_4_1283_1289.txt"
    ).write_text("poa_irradiance\nsensor_a\n", encoding="utf-8")

    outputs = prepare_group_features(group_id=1, system_id=4, project_root=tmp_path)

    assert outputs["X_train"].shape == (7, 1)
    assert outputs["X_test"].shape == (2, 1)
    assert list(outputs["y_train"].columns) == ["dc_power", "poa_irradiance"]
    assert outputs["registry_path"].name == "selected_features_group_4_1283_1289.txt"