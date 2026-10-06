"""TODO scaffolding for a future multi-system preprocessing pipeline."""

from collections.abc import Sequence

import pandas as pd


def nan_handling(data: pd.DataFrame, sentinel_values: Sequence[float]) -> pd.DataFrame:
    """TODO: define shared sentinel and missing-value rules across systems."""
    raise NotImplementedError("Multi-system NaN handling has not been designed yet.")


def imputation(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    strategy: str,
) -> pd.DataFrame:
    """TODO: define leakage-safe imputation after a temporal split."""
    raise NotImplementedError("Multi-system imputation has not been designed yet.")


def drop_features(data: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """TODO: define auditable feature dropping after cross-system review."""
    raise NotImplementedError("Multi-system feature dropping has not been designed yet.")