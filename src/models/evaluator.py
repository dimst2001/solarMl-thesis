"""Temporal data-partitioning utilities for model evaluation."""

import pandas as pd


def temporal_train_test_split(
    data: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split observations chronologically without separating equal timestamps."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    if data.empty:
        raise ValueError("Cannot split an empty dataset.")

    ordered_data = data.sort_index(kind="stable")
    split_index = int(len(ordered_data) * train_fraction)
    if split_index == 0 or split_index >= len(ordered_data):
        raise ValueError(
            f"Split leaves an empty partition for {len(ordered_data)} observations."
        )

    boundary_timestamp = ordered_data.index[split_index]
    split_index = int(ordered_data.index.searchsorted(boundary_timestamp, side="left"))
    if split_index == 0 or split_index >= len(ordered_data):
        raise ValueError("A timestamp-safe split would leave an empty partition.")

    train_data = ordered_data.iloc[:split_index].copy()
    test_data = ordered_data.iloc[split_index:].copy()
    if train_data.index.max() >= test_data.index.min():
        raise ValueError("Train and test partitions must have strictly ordered timestamps.")
    return train_data, test_data