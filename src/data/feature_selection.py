"""Feature selection logic for daytime filtering, correlation analysis, and multi-system validation."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    """Loads dataset, standardizes columns by stripping sensor IDs, and filters daytime records."""
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

    # 2. Strip '__ID' suffixes from all remaining columns
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


def run_group_feature_selection(
    cleaned_dir: Path,
    artifacts_dir: Path,
    system_groups: Dict[str, List[str]],
    threshold: float = 0.3,
) -> Dict[str, List[str]]:
    """Executes feature selection on grouped systems, generating heatmaps and scatter plots per group."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    group_results = {}

    for group_name, systems in system_groups.items():
        df_list = []
        for sys_name in systems:
            try:
                sys_file = resolve_system_file_path(cleaned_dir, sys_name)
                df = load_and_preprocess_system(sys_file)
                df_list.append(df)
            except FileNotFoundError:
                print(f"Warning: {sys_name} not found for {group_name}.")
                continue
        
        if not df_list:
            continue

        # Concatenate matched systems to compute global group correlation
        df_group = pd.concat(df_list, ignore_index=True)
        
        # Compute Pearson correlation
        corrs = calculate_pearson_correlations(df_group)
        selected = corrs[corrs.abs() >= threshold].index.tolist()
        group_results[group_name] = selected

        # Export selected features
        with open(artifacts_dir / f"selected_features_{group_name}.txt", "w", encoding="utf-8") as f:
            for feat in selected:
                f.write(f"{feat}\n")

        # --- Generate Visualizations ---
        numeric_cols = df_group.select_dtypes(include=[np.number]).columns
        corr_matrix = df_group[numeric_cols].corr(method="pearson")
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"{group_name} - Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(artifacts_dir / f"{group_name}_correlation_heatmap.png", dpi=300)
        plt.close()

        # Scatter Plots
        features_to_plot = [c for c in numeric_cols if c != "dc_power"]
        n_cols = 2
        n_rows = max(1, (len(features_to_plot) + 1) // n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, col in enumerate(features_to_plot):
            # Subsample scatter plot to max 50,000 points to prevent memory/render overload
            df_sample = df_group.sample(min(50000, len(df_group)), random_state=42)
            axes[idx].scatter(df_sample[col], df_sample["dc_power"], alpha=0.1, s=1)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("dc_power")
            axes[idx].set_title(f"{col} vs dc_power (r = {corrs.get(col, 0):.2f})")

        for idx in range(len(features_to_plot), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(artifacts_dir / f"{group_name}_scatter_plots.png", dpi=300)
        plt.close()

    return group_results


def run_experimental_feature_selection(
    cleaned_dir: Path,
    artifacts_dir: Path,
    system_groups: Dict[str, List[str]],
    target_corr_threshold: float = 0.3,
    collinearity_threshold: float = 0.85,
    corr_method: str = "pearson",
    irradiance_threshold: float = 0.0,
) -> Dict[str, List[str]]:
    """
    Experimental pipeline allowing tuning of correlation metrics, multicollinearity pruning, 
    and physical irradiance masking.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    group_results = {}

    for group_name, systems in system_groups.items():
        df_list = []
        for sys_name in systems:
            try:
                sys_file = resolve_system_file_path(cleaned_dir, sys_name)
                df = load_and_preprocess_system(sys_file)
                df_list.append(df)
            except FileNotFoundError:
                continue
        
        if not df_list:
            continue

        df_group = pd.concat(df_list, ignore_index=True)
        
        # EXPERIMENT 1: Physical Irradiance Masking (Dawn/Dusk filtering)
        if irradiance_threshold > 0 and "poa_irradiance" in df_group.columns:
            df_group = df_group[df_group["poa_irradiance"] >= irradiance_threshold]
            
        numeric_cols = df_group.select_dtypes(include=[np.number]).columns
        
        # EXPERIMENT 2: Non-Linear Metric Switching
        corrs = df_group[numeric_cols].corr(method=corr_method)
        target_corr = corrs["dc_power"].drop(labels=["dc_power"]).sort_values(ascending=False)
        
        # EXPERIMENT 3: Target Correlation Filter
        candidate_features = target_corr[target_corr.abs() >= target_corr_threshold].index.tolist()
        
        # EXPERIMENT 4: Multicollinearity Pruning (Drop redundant sensors)
        final_features = []
        for feat in candidate_features:
            is_collinear = False
            for accepted_feat in final_features:
                if abs(corrs.loc[feat, accepted_feat]) >= collinearity_threshold:
                    is_collinear = True
                    break
            if not is_collinear:
                final_features.append(feat)

        group_results[group_name] = final_features
        
        # Save experimental registry
        exp_name = f"exp_{corr_method}_t{target_corr_threshold}_c{collinearity_threshold}"
        with open(artifacts_dir / f"selected_features_{group_name}_{exp_name}.txt", "w", encoding="utf-8") as f:
            for feat in final_features:
                f.write(f"{feat}\n")

    return group_results