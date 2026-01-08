"""Test script to verify notebook data loading pipeline works correctly.

This mimics the notebook's data loading logic to ensure both paths work.
"""

from pathlib import Path
import logging
import sys
import pandas as pd

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

from src.core.optimizer import BESSOptimizerModelI
from src.data.load_process_market_data import load_preprocessed_country_data


def test_fast_path(test_country: str = 'HU'):
    """Test the fast path: preprocessed parquet loading."""
    logger.info("=" * 70)
    logger.info("TEST 1: FAST PATH (Preprocessed Parquet)")
    logger.info("=" * 70)

    preprocessed_path = project_root / "data" / "parquet" / "preprocessed" / f"{test_country.lower()}.parquet"

    try:
        if not preprocessed_path.exists():
            logger.warning(f"[SKIP] Preprocessed file not found: {preprocessed_path}")
            return None

        logger.info(f"Loading preprocessed data: {preprocessed_path.name}")
        country_data = load_preprocessed_country_data(test_country)
        logger.info(f"[OK] Loaded {len(country_data)} time steps for {test_country}")

        # Validate data
        logger.info("Validating data structure...")
        required_cols = [
            'price_day_ahead', 'price_fcr', 'price_afrr_pos', 'price_afrr_neg',
            'price_afrr_energy_pos', 'price_afrr_energy_neg',
            'w_afrr_pos', 'w_afrr_neg',
            'hour', 'day_of_year', 'month', 'year', 'block_of_day', 'block_id', 'day_id', 'timestamp'
        ]

        missing_cols = [col for col in required_cols if col not in country_data.columns]
        if missing_cols:
            logger.error(f"[FAIL] Missing columns: {missing_cols}")
            return None

        logger.info(f"[OK] All required columns present ({len(required_cols)} columns)")
        logger.info(f"[OK] Data shape: {country_data.shape}")

        return country_data

    except Exception as e:
        logger.error(f"[FAIL] Fast path failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_submission_path(test_country: str = 'HU'):
    """Test the submission path: Excel loading."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: SUBMISSION PATH (Excel Loading)")
    logger.info("=" * 70)

    excel_path = project_root / "data" / "TechArena2025_Phase2_data.xlsx"

    try:
        if not excel_path.exists():
            logger.warning(f"[SKIP] Excel file not found: {excel_path}")
            return None

        logger.info(f"Loading from Excel: {excel_path.name}")
        logger.info("This matches Huawei submission requirements...")

        # Create temporary optimizer for data loading
        temp_opt = BESSOptimizerModelI()

        # Load using new Phase 2 Excel loader
        logger.info("Loading Phase 2 market tables from Excel...")
        full_data = temp_opt.load_and_preprocess_data(str(excel_path))

        # Extract country-specific data
        logger.info(f"Extracting country data for {test_country}...")
        country_data = temp_opt.extract_country_data(full_data, test_country)
        logger.info(f"[OK] Loaded {len(country_data)} time steps for {test_country}")

        # Validate data
        logger.info("Validating data structure...")
        required_cols = [
            'price_day_ahead', 'price_fcr', 'price_afrr_pos', 'price_afrr_neg',
            'price_afrr_energy_pos', 'price_afrr_energy_neg',
            'w_afrr_pos', 'w_afrr_neg',
            'hour', 'day_of_year', 'month', 'year', 'block_of_day', 'block_id', 'day_id', 'timestamp'
        ]

        missing_cols = [col for col in required_cols if col not in country_data.columns]
        if missing_cols:
            logger.error(f"[FAIL] Missing columns: {missing_cols}")
            return None

        logger.info(f"[OK] All required columns present ({len(required_cols)} columns)")
        logger.info(f"[OK] Data shape: {country_data.shape}")

        return country_data

    except Exception as e:
        logger.error(f"[FAIL] Submission path failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_optimization_workflow(country_data: pd.DataFrame, test_country: str = 'HU'):
    """Test a small optimization to verify end-to-end workflow."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: OPTIMIZATION WORKFLOW")
    logger.info("=" * 70)

    try:
        # Take a small slice (36 hours = 144 steps)
        data_slice = country_data.iloc[0:144].copy()
        data_slice.reset_index(drop=True, inplace=True)

        logger.info(f"Testing with {len(data_slice)} time steps (36 hours)")

        # Initialize optimizer
        logger.info("Initializing optimizer...")
        optimizer = BESSOptimizerModelI()

        # Build model (Model I requires daily_cycle_limit parameter)
        logger.info("Building optimization model...")
        model = optimizer.build_optimization_model(data_slice, c_rate=0.5, daily_cycle_limit=None)
        logger.info(f"[OK] Model built: {model.nvariables()} vars, {model.nconstraints()} constraints")

        # Solve model (with short timeout for testing)
        logger.info("Solving model...")
        import time
        start = time.time()
        solved_model, solver_results = optimizer.solve_model(model)
        solve_time = time.time() - start

        logger.info(f"[OK] Model solved in {solve_time:.2f}s")
        logger.info(f"   Status: {solver_results.solver.status}")
        logger.info(f"   Termination: {solver_results.solver.termination_condition}")

        # Extract solution
        logger.info("Extracting solution...")
        solution_dict = optimizer.extract_solution(solved_model, solver_results)
        logger.info(f"[OK] Objective value: {solution_dict['objective_value']:.2f} EUR")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Optimization workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("#" * 70)
    logger.info("# Notebook Data Pipeline Test")
    logger.info("#" * 70)
    logger.info("\n")

    test_country = 'HU'

    # Test fast path
    fast_data = test_fast_path(test_country)

    # Test submission path
    submission_data = test_submission_path(test_country)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    fast_ok = fast_data is not None
    submission_ok = submission_data is not None

    logger.info(f"Fast Path (Preprocessed):  {'[OK]' if fast_ok else '[SKIP/FAIL]'}")
    logger.info(f"Submission Path (Excel):   {'[OK]' if submission_ok else '[SKIP/FAIL]'}")

    # Test optimization with whichever path worked
    if fast_ok:
        logger.info("\nTesting optimization workflow with FAST PATH data...")
        opt_ok = test_optimization_workflow(fast_data, test_country)
    elif submission_ok:
        logger.info("\nTesting optimization workflow with SUBMISSION PATH data...")
        opt_ok = test_optimization_workflow(submission_data, test_country)
    else:
        logger.error("\n[FAIL] Neither data path available, cannot test optimization")
        opt_ok = False

    # Final result
    logger.info("\n" + "=" * 70)
    if (fast_ok or submission_ok) and opt_ok:
        logger.info("SUCCESS! Notebook pipeline is ready to use")
        logger.info("=" * 70)
        if fast_ok:
            logger.info("\nRecommendation: Use FAST PATH for validation testing")
        else:
            logger.info("\nNote: Generate preprocessed files for faster testing")
            logger.info("Run: python src/data/generate_preprocessed_country_data.py")
        return 0
    else:
        logger.error("FAIL: Some tests failed")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
