"""Data processing package initialization exposing paths and pipeline functions."""

from pathlib import Path

# Base paths
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent.parent

# Check internal data/cleaned first, fallback to parent directory if unpopulated
_INTERNAL_CLEANED = BASE_DIR / "data" / "cleaned"
_EXTERNAL_CLEANED = BASE_DIR.parent / "cleaned"

if _INTERNAL_CLEANED.exists() and any(_INTERNAL_CLEANED.iterdir()):
    CLEANED_DIR = _INTERNAL_CLEANED
elif _EXTERNAL_CLEANED.exists() and any(_EXTERNAL_CLEANED.iterdir()):
    CLEANED_DIR = _EXTERNAL_CLEANED
else:
    CLEANED_DIR = _INTERNAL_CLEANED

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Explicitly import and expose subpackage functions
from src.data.feature_selection import (
    calculate_pearson_correlations,
    load_and_preprocess_system,
    resolve_system_file_path,
    run_feature_selection_pipeline,
    validate_features_across_systems,
    run_group_feature_selection,
    run_experimental_feature_selection,
)

__all__ = [
    "CLEANED_DIR",
    "ARTIFACTS_DIR",
    "resolve_system_file_path",
    "load_and_preprocess_system",
    "calculate_pearson_correlations",
    "validate_features_across_systems",
    "run_feature_selection_pipeline",
    "run_group_feature_selection",
    "run_experimental_feature_selection",
]