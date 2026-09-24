"""Feature selection logic for daytime filtering, correlation analysis, and multi-system validation."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def resolve_system_file_path(cleaned_dir: Path, system_name: str) -> Path:
    """Resolves system file path matching system_x_clean or system_x in Parquet or CSV format."""
    sys_num = system_name.replace("system_", "")
    candidate_names = [
        f"system_{sys_num}_clean.parquet",
        f"system_{sys_num}_clean.csv",
        f"system_{sys_num}.parquet",
        f"system_{sys_num}.csv",
    ]

    for candidate in candidate_names:
        file_path = cleaned_dir / candidate
        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        f"Could not locate dataset for '{system_name}' in {cleaned_dir.resolve()}. "
        f"Checked candidates: {candidate_names}"
    )


def load_and_preprocess_system(
    file_path: Path, target_prefix: str = "dc_power"
) -> pd.DataFrame:
    """Loads dataset, standardizes columns by stripping sensor IDs, and filters daytime records.

    Parameters
    ----------
    file_path : Path
        Path to input CSV or Parquet file.
    target_prefix : str, default="dc_power"
        Prefix identifying the target power column.

    Returns
    -------
    pd.DataFrame
        Filtered daytime DataFrame where $P_{\\text{DC}} > 0$.
    """
    file_path = Path(file_path)

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, low_memory=False)

        if df.columns[0] in ["Unnamed: 0", "index", "time"]:
            df = df.rename(columns={df.columns[0]: "timestamp"})

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
    else:
        raise ValueError(f"Unsupported file extension: {file_path.suffix}")

    # 1. Isolate and aggregate the target column FIRST
    target_cols = [col for col in df.columns if col.startswith(target_prefix)]
    
    if not target_cols:
        raise KeyError(
            f"Target column matching '{target_prefix}' not found in {file_path.name}"
        )
        
    if len(target_cols) > 1:
        # Sum multiple inverters/arrays for total generation
        df["dc_power"] = df[target_cols].sum(axis=1)
        df = df.drop(columns=target_cols)
    else:
        df = df.rename(columns={target_cols[0]: "dc_power"})

    # 2. Strip '__ID' suffixes from all remaining columns (e.g. poa_irradiance__313 -> poa_irradiance)
    df.columns = [
        col.split("__")[0] if isinstance(col, str) and "__" in col else col 
        for col in df.columns
    ]

    # 3. Handle duplicated columns (e.g. multiple module_temp sensors) by taking the spatial mean
    if df.columns.duplicated().any():
        df = df.groupby(df.columns, axis=1).mean()

    # 4. Filter operational daytime records (P_DC > 0)
    return df[df["dc_power"] > 0].copy()


def calculate_pearson_correlations(
    df: pd.DataFrame, target_col: str = "dc_power"
) -> pd.Series:
    """Computes Pearson correlation coefficients against the target variable."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr(method="pearson")
    return (
        corr_matrix[target_col].drop(labels=[target_col]).sort_values(ascending=False)
    )


def validate_features_across_systems(
    cleaned_dir: Path,
    system_names: List[str],
    selected_features: List[str],
) -> pd.DataFrame:
    """Validates candidate feature correlations across secondary evaluation systems."""
    results: Dict[str, pd.Series] = {}

    for sys_name in system_names:
        try:
            target_file = resolve_system_file_path(cleaned_dir, sys_name)
            df_val = load_and_preprocess_system(target_file)
            corrs = calculate_pearson_correlations(df_val)
            results[sys_name] = corrs
        except FileNotFoundError:
            results[sys_name] = pd.Series(dtype=float)

    val_df = pd.DataFrame(results)
    return val_df.reindex(selected_features)


def run_feature_selection_pipeline(
    cleaned_dir: Path,
    artifacts_dir: Path,
    primary_system: str = "system_4",
    validation_systems: List[str] = ["system_10", "system_33", "system_50"],
    threshold: float = 0.3,
) -> Tuple[List[str], pd.Series, pd.DataFrame]:
    """Executes the full feature selection pipeline and writes selected_features.txt."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sys_file = resolve_system_file_path(cleaned_dir, primary_system)
    df_primary = load_and_preprocess_system(sys_file)
    primary_corr = calculate_pearson_correlations(df_primary)

    # Filter features by correlation threshold: |r| >= threshold[cite: 3]
    selected_features = primary_corr[
        primary_corr.abs() >= threshold
    ].index.tolist()

    val_df = validate_features_across_systems(
        cleaned_dir, validation_systems, selected_features
    )

    txt_path = artifacts_dir / "selected_features.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    return selected_features, primary_corr, val_df