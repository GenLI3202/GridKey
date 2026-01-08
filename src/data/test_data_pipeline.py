"""Test script to verify both data pipeline paths produce identical results.

Tests:
1. Submission path: Excel → optimizer.load_and_preprocess_data() → extract_country_data()
2. Validation path: Preprocessed parquet → load_preprocessed_country_data()
3. Verify both paths produce bit-identical DataFrames

Usage:
    python test_data_pipeline.py
"""

from pathlib import Path
import logging
import sys
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.core.optimizer import BESSOptimizerModelIII
from src.data.load_process_market_data import load_preprocessed_country_data


def test_submission_path(excel_path: Path, test_country: str = 'DE_LU'):
    """Test the submission flow: Excel -> optimizer -> country data."""
    logger.info("=" * 70)
    logger.info("Testing SUBMISSION PATH (Excel -> optimizer)")
    logger.info("=" * 70)

    try:
        # Initialize optimizer
        optimizer = BESSOptimizerModelIII()
        logger.info("[1/3] Optimizer initialized")

        # Load and preprocess data from Excel
        logger.info(f"[2/3] Loading data from Excel: {excel_path}")
        full_data = optimizer.load_and_preprocess_data(str(excel_path))
        logger.info(f"      Loaded: {full_data.shape} (rows x columns)")
        logger.info(f"      Time range: {full_data.index.min()} to {full_data.index.max()}")

        # Extract country-specific data
        logger.info(f"[3/3] Extracting country data: {test_country}")
        country_data = optimizer.extract_country_data(full_data, test_country)
        logger.info(f"      Extracted: {country_data.shape}")
        logger.info(f"      Columns: {list(country_data.columns)}")

        logger.info("[OK] Submission path completed successfully\n")
        return country_data

    except Exception as e:
        logger.error(f"[FAIL] Submission path failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_validation_path(test_country: str = 'DE_LU'):
    """Test the validation flow: Preprocessed parquet -> direct load."""
    logger.info("=" * 70)
    logger.info("Testing VALIDATION PATH (Preprocessed parquet -> direct load)")
    logger.info("=" * 70)

    try:
        # Load preprocessed country data directly
        logger.info(f"[1/1] Loading preprocessed data for {test_country}")
        country_data = load_preprocessed_country_data(test_country)
        logger.info(f"      Loaded: {country_data.shape}")
        logger.info(f"      Columns: {list(country_data.columns)}")

        logger.info("[OK] Validation path completed successfully\n")
        return country_data

    except Exception as e:
        logger.error(f"[FAIL] Validation path failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> bool:
    """Compare two DataFrames for equality."""
    logger.info("=" * 70)
    logger.info("COMPARING RESULTS")
    logger.info("=" * 70)

    if df1 is None or df2 is None:
        logger.error("[FAIL] One or both DataFrames are None")
        return False

    # Check shape
    if df1.shape != df2.shape:
        logger.error(f"[FAIL] Shape mismatch: {name1}={df1.shape}, {name2}={df2.shape}")
        return False
    logger.info(f"[OK] Shape match: {df1.shape}")

    # Check columns
    if not df1.columns.equals(df2.columns):
        logger.error(f"[FAIL] Column mismatch")
        logger.error(f"   {name1} cols: {list(df1.columns)}")
        logger.error(f"   {name2} cols: {list(df2.columns)}")
        return False
    logger.info(f"[OK] Columns match: {len(df1.columns)} columns")

    # Check row counts
    if len(df1) != len(df2):
        logger.error(f"[FAIL] Row count mismatch: {name1}={len(df1)}, {name2}={len(df2)}")
        return False
    logger.info(f"[OK] Row count match: {len(df1)} rows")

    # Compare values column by column
    all_match = True
    for col in df1.columns:
        # Handle NaN values
        df1_col = df1[col]
        df2_col = df2[col]

        # Check if both have same NaN pattern
        df1_nan = df1_col.isna()
        df2_nan = df2_col.isna()

        if not df1_nan.equals(df2_nan):
            logger.warning(f"[WARN] NaN pattern mismatch in column '{col}'")
            nan_diff = (df1_nan != df2_nan).sum()
            logger.warning(f"        {nan_diff} positions differ")
            all_match = False
            continue

        # Compare non-NaN values
        mask = ~df1_nan  # Non-NaN positions
        if mask.any():
            # Use np.allclose for floating point comparison
            if np.issubdtype(df1_col.dtype, np.number) and np.issubdtype(df2_col.dtype, np.number):
                if not np.allclose(df1_col[mask], df2_col[mask], rtol=1e-9, atol=1e-12, equal_nan=True):
                    logger.warning(f"[WARN] Values differ in column '{col}'")
                    max_diff = np.max(np.abs(df1_col[mask] - df2_col[mask]))
                    logger.warning(f"        Max difference: {max_diff}")
                    all_match = False
            else:
                # For non-numeric columns, use direct equality
                if not df1_col[mask].equals(df2_col[mask]):
                    logger.warning(f"[WARN] Values differ in column '{col}'")
                    all_match = False

    if all_match:
        logger.info("\n" + "=" * 70)
        logger.info("SUCCESS! Both paths produce IDENTICAL results")
        logger.info("=" * 70)
        return True
    else:
        logger.error("\n" + "=" * 70)
        logger.error("FAIL: Results differ between paths")
        logger.error("=" * 70)
        return False


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("#" * 70)
    logger.info("# Data Pipeline Validation Test")
    logger.info("#" * 70)
    logger.info("\n")

    # Define paths
    excel_path = project_root / "data" / "TechArena2025_Phase2_data.xlsx"
    test_country = 'DE_LU'

    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        return 1

    # Test both paths
    submission_data = test_submission_path(excel_path, test_country)
    validation_data = test_validation_path(test_country)

    # Compare results
    if submission_data is not None and validation_data is not None:
        match = compare_dataframes(
            submission_data,
            validation_data,
            "Submission Path",
            "Validation Path"
        )
        return 0 if match else 1
    else:
        logger.error("\n" + "=" * 70)
        logger.error("FAIL: One or both paths failed to complete")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
