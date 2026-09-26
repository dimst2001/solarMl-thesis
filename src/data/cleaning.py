"""Cleaning helpers for model-ready solar-system data."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def target_protected_impute(
    data: pd.DataFrame,
    target_columns: Sequence[str],
    feature_columns: Sequence[str],
    fit_data: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Drop missing-target rows and impute features from independent-column means.

    When ``fit_data`` is provided, means are fitted from it only. This lets callers
    pass the chronological training partition and avoid learning imputation
    statistics from future observations.
    """
    target_columns = list(target_columns)
    feature_columns = list(feature_columns)
    if not target_columns:
        raise ValueError("At least one target column is required.")
    if not feature_columns:
        raise ValueError("At least one independent feature column is required.")
    if set(target_columns).intersection(feature_columns):
        raise ValueError("Target columns cannot also be independent features.")

    required_columns = [*target_columns, *feature_columns]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Input data is missing required columns: {missing_columns}")

    valid_data = data.dropna(subset=target_columns).copy()
    if valid_data.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    reference_data = valid_data if fit_data is None else fit_data
    missing_fit_columns = [
        column for column in feature_columns if column not in reference_data.columns
    ]
    if missing_fit_columns:
        raise KeyError(f"Imputation fit data is missing features: {missing_fit_columns}")

    features = valid_data[feature_columns].apply(pd.to_numeric, errors="raise")
    targets = valid_data[target_columns].apply(pd.to_numeric, errors="raise")
    fit_features = reference_data[feature_columns].apply(pd.to_numeric, errors="raise")
    feature_means = fit_features.mean()
    invalid_means = feature_means[~np.isfinite(feature_means)].index.tolist()
    if invalid_means:
        raise ValueError(
            f"Features have no finite values in the imputation fit data: {invalid_means}"
        )

    imputed_features = features.fillna(feature_means)
    if imputed_features.isna().any().any():
        raise ValueError("Feature imputation did not resolve every missing value.")

    return imputed_features, targets, feature_means