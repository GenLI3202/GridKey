"""
Huawei TechArena 2025 - Phase 2 BESS Optimization Solution
Main Execution Script for Competition Submission

This script demonstrates our Model III BESS optimization approach with:
- 4 electricity markets (DA, FCR, aFRR capacity, aFRR energy)
- Full battery degradation modeling (cyclic + calendar aging)
- MPC control with rolling horizon optimization

Data Loading:
    PREPROCESSED_DATA_READY = False (default for submission):
        → Loads from Input/TechArena2025_Phase2_data.xlsx (original Excel)
        → Automatically converts aFRR energy 0 prices to NaN
        → Suitable for competition submission/evaluation

    PREPROCESSED_DATA_READY = True (development mode):
        → Loads from preprocessed parquet files (10-100x faster)
        → Requires prior data generation

Usage:
    # Quick test (3-day sample, ~5-10 minutes)
    Set TEST_MODE = True, then run: python main.py

    # Full year (365 days, ~3-6 hours per scenario)
    Set TEST_MODE = False, then run: python main.py

Output:
    Results are saved in output/ directory:
    - TechArena_Phase2_Operation.xlsx (15 sheets: 5 countries × 3 C-rates)
    - TechArena_Phase2_Configuration.xlsx (5 sheets: one per country)
    - TechArena_Phase2_Investment.xlsx (5 sheets: one per country)

Author: SoloGen Team
Date: November 2025
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Core optimizer
from py_script.core.optimizer import BESSOptimizerModelIII

# MPC simulation
from py_script.mpc.mpc_simulator import MPCSimulator
from py_script.mpc.transform_mpc_results import (
    transform_mpc_results_for_viz,
    extract_iteration_summary
)

# Data loading
from py_script.data.load_process_market_data import load_preprocessed_country_data

# Submission format conversion
from py_script.submission.convert_results import (
    convert_operation_file,
    calculate_10year_roi,
    WACC, INFLATION, BATTERY_CAPACITY_KWH
)

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

# *** TEST MODE - Set this to control execution time ***
TEST_MODE = True  # True = 3-day test (~5-10 min), False = full 365-day (~3-6 hours)

# *** DATA LOADING MODE ***
# Set to False for submission (load from Excel)
# Set to True for development (use preprocessed parquet - 10-100x faster)
PREPROCESSED_DATA_READY = False

# Test duration
TEST_DAYS = 3 if TEST_MODE else 365

# Scenarios to run (can modify to run subset)
COUNTRIES = ['DE_LU']  # Full: ['DE_LU', 'AT', 'CH', 'HU', 'CZ']
C_RATES = [0.25, 0.33, 0.5]      # Full: [0.25, 0.33, 0.5]

# Model parameters
ALPHA = 1.0  # Degradation cost weight (0.5-1.5 typical)
INITIAL_SOC_FRACTION = 0.5  # Start at 50% SOC
USE_AFRR_EV_WEIGHTING = False  # aFRR energy activation probability weighting

# MPC parameters (from config file)
config_dir = project_root / "data" / "p2_config"
with open(config_dir / "mpc_config.json", 'r') as f:
    mpc_config = json.load(f)

HORIZON_HOURS = mpc_config['mpc_parameters']['horizon_hours']  # 36h
EXECUTION_HOURS = mpc_config['mpc_parameters']['execution_hours']  # 24h

# Output directory
# For submission: save directly to output/ (no timestamp subdirectory)
# For development: can use timestamped subdirectory to keep multiple runs
USE_TIMESTAMPED_OUTPUT = False  # Set to True to create timestamped subdirectories

if USE_TIMESTAMPED_OUTPUT:
    OUTPUT_DIR = project_root / "output" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
else:
    OUTPUT_DIR = project_root / "output"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def run_single_scenario(country: str, c_rate: float, logger, full_market_data=None) -> dict:
    """
    Run MPC optimization for a single country-C-rate scenario.

    Args:
        country: Country code (e.g., 'DE_LU', 'CH')
        c_rate: C-rate value (0.25, 0.33, or 0.5)
        logger: Logger instance
        full_market_data: Pre-loaded full market data (MultiIndex DataFrame) or None

    Returns:
        Dictionary containing results and solution data
    """
    scenario_start = time.time()

    try:
        logger.info("=" * 80)
        logger.info(f"SCENARIO: {country} | C-rate: {c_rate}")
        logger.info("=" * 80)

        # 1. Load market data
        logger.info(f"[1/4] Loading market data for {country}...")

        if full_market_data is not None:
            # Use pre-loaded full market data (Excel mode - efficient)
            logger.info("  → Extracting from pre-loaded market data...")
            temp_optimizer = BESSOptimizerModelIII(alpha=ALPHA, use_afrr_ev_weighting=USE_AFRR_EV_WEIGHTING)
            country_data = temp_optimizer.extract_country_data(full_market_data, country)
            logger.info(f"  → Extracted {country} data with {len(country_data.columns)} price columns")
        elif PREPROCESSED_DATA_READY:
            # Load from preprocessed parquet (development fast path)
            logger.info("  → Loading from preprocessed parquet (fast path)...")
            preprocessed_dir = project_root / "data" / "parquet" / "preprocessed"
            country_data = load_preprocessed_country_data(country, data_dir=preprocessed_dir)
            logger.info(f"  → Loaded preprocessed data")
        else:
            raise ValueError("No market data available. Please check data loading configuration.")

        # Slice to test duration
        duration_timesteps = TEST_DAYS * 96  # 96 timesteps per day (15-min intervals)
        country_data_slice = country_data.iloc[:duration_timesteps].copy()
        logger.info(f"  → Using {len(country_data_slice)} timesteps ({TEST_DAYS} days)")

        # 2. Initialize optimizer with Model III
        logger.info(f"[2/4] Initializing Model III optimizer (alpha={ALPHA})...")
        optimizer = BESSOptimizerModelIII(alpha=ALPHA, use_afrr_ev_weighting=USE_AFRR_EV_WEIGHTING)
        logger.info(f"  → Battery: {optimizer.battery_params['capacity_kwh']} kWh")
        logger.info(f"  → Degradation segments: {len(optimizer.degradation_params.get('marginal_costs', []))}")
        logger.info(f"  → aFRR EV weighting: {USE_AFRR_EV_WEIGHTING}")

        # 3. Setup MPC simulator
        logger.info(f"[3/4] Setting up MPC simulator...")
        logger.info(f"  → Horizon: {HORIZON_HOURS}h | Execution: {EXECUTION_HOURS}h")

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=country_data_slice,
            horizon_hours=HORIZON_HOURS,
            execution_hours=EXECUTION_HOURS,
            c_rate=c_rate,
            validate_constraints=False  # Disable for speed
        )

        # 4. Run MPC simulation
        logger.info(f"[4/4] Running MPC simulation...")
        expected_iterations = len(country_data_slice) // (EXECUTION_HOURS * 4)
        logger.info(f"  → Expected iterations: ~{expected_iterations}")

        mpc_results = simulator.run_full_simulation(
            initial_soc_fraction=INITIAL_SOC_FRACTION
        )

        runtime = time.time() - scenario_start

        logger.info(f"  ✓ Simulation complete in {runtime/60:.1f} minutes!")
        logger.info(f"  → Profit: €{mpc_results['net_profit']:,.0f}")
        logger.info(f"  → Revenue: €{mpc_results['total_revenue']:,.0f}")
        logger.info(f"  → Degradation: €{mpc_results['total_degradation_cost']:,.0f}")

        # Transform results for output
        total_bids_df = mpc_results['total_bids_df']
        viz_df = transform_mpc_results_for_viz(
            total_bids_df,
            country_data_slice,
            battery_capacity_kwh=BATTERY_CAPACITY_KWH
        )

        return {
            'status': 'success',
            'country': country,
            'c_rate': c_rate,
            'runtime_minutes': runtime / 60,
            'performance': {
                'total_profit_eur': mpc_results['net_profit'],
                'total_revenue_eur': mpc_results['total_revenue'],
                'total_cost_eur': mpc_results['total_degradation_cost'],
                'revenue_da_eur': mpc_results.get('da_revenue', 0),
                'revenue_afrr_energy_eur': mpc_results.get('afrr_e_revenue', 0),
                'revenue_as_eur': mpc_results.get('as_revenue', 0),
                'degradation_cyclic_eur': mpc_results.get('cyclic_cost', 0),
                'degradation_calendar_eur': mpc_results.get('calendar_cost', 0),
            },
            'solution_df': viz_df,
            'mpc_results': mpc_results
        }

    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'country': country,
            'c_rate': c_rate,
            'error': str(e)
        }


def save_results_to_excel(results_list: list, output_dir: Path, logger):
    """
    Save results to Excel files in submission format.

    Args:
        results_list: List of result dictionaries
        output_dir: Directory to save Excel files
        logger: Logger instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUBMISSION OUTPUT FILES")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter successful results
    successful_results = [r for r in results_list if r['status'] == 'success']
    if not successful_results:
        logger.error("No successful results to export!")
        return

    logger.info(f"Exporting {len(successful_results)} successful scenarios...")

    # 1. Generate Operation Excel (15 sheets)
    logger.info("\n[1/3] Generating Operation file...")
    operation_path = output_dir / 'TechArena_Phase2_Operation.xlsx'

    with pd.ExcelWriter(operation_path, engine='openpyxl') as writer:
        for result in successful_results:
            country = result['country']
            c_rate = result['c_rate']
            sheet_name = f"{country}_{c_rate}"

            # Convert to submission format
            operation_df = convert_operation_file(result['solution_df'])
            operation_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"  ✓ Sheet: {sheet_name} ({len(operation_df)} rows)")

    logger.info(f"  → Saved: {operation_path}")

    # 2. Generate Configuration Excel (5 sheets)
    logger.info("\n[2/3] Generating Configuration file...")
    config_path = output_dir / 'TechArena_Phase2_Configuration.xlsx'

    # Group by country
    country_results = {}
    for result in successful_results:
        country = result['country']
        if country not in country_results:
            country_results[country] = []
        country_results[country].append(result)

    with pd.ExcelWriter(config_path, engine='openpyxl') as writer:
        for country, country_res in sorted(country_results.items()):
            config_data = []

            for result in sorted(country_res, key=lambda x: x['c_rate']):
                c_rate = result['c_rate']
                perf = result['performance']

                # Calculate metrics
                battery_power_mw = BATTERY_CAPACITY_KWH * c_rate / 1000
                annual_profit = perf['total_profit_eur']

                # Scale to full year if in test mode
                if TEST_MODE:
                    annual_profit = annual_profit * (365 / TEST_DAYS)

                yearly_profit_per_mw = (annual_profit / battery_power_mw) / 1000  # kEUR/MW

                # Calculate 10-year ROI
                roi_percent, _ = calculate_10year_roi(annual_profit, country)

                # Estimate cycles from degradation
                cyclic_cost = perf['degradation_cyclic_eur']
                num_cycles = cyclic_cost / 50 if cyclic_cost > 0 else 0  # Rough estimate

                config_data.append({
                    'C-rate': c_rate,
                    'number of cycles': round(num_cycles, 2),
                    'yearly profits [kEUR/MW]': round(yearly_profit_per_mw, 2),
                    'levelized ROI [%]': round(roi_percent, 2)
                })

            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name=country, index=False)
            logger.info(f"  ✓ Sheet: {country} ({len(config_df)} configurations)")

    logger.info(f"  → Saved: {config_path}")

    # 3. Generate Investment Excel (5 sheets)
    logger.info("\n[3/3] Generating Investment file...")
    investment_path = output_dir / 'TechArena_Phase2_Investment.xlsx'

    with pd.ExcelWriter(investment_path, engine='openpyxl') as writer:
        for country, country_res in sorted(country_results.items()):
            # Use best C-rate for this country
            best_result = max(country_res, key=lambda x: x['performance']['total_profit_eur'])
            annual_profit = best_result['performance']['total_profit_eur']

            # Scale to full year if in test mode
            if TEST_MODE:
                annual_profit = annual_profit * (365 / TEST_DAYS)

            c_rate = best_result['c_rate']

            # Calculate 10-year ROI
            roi_percent, years_df = calculate_10year_roi(annual_profit, country)

            # Create investment sheet
            investment_data = []
            investment_data.append(['WACC', WACC[country]])
            investment_data.append(['Inflation Rate', INFLATION[country]])
            investment_data.append(['Discount Rate', WACC[country]])
            investment_data.append(['Yearly Profits (2024) [kEUR]', annual_profit / 1000])
            investment_data.append(['Best C-rate', c_rate])
            investment_data.append([])

            # Year-by-year table
            investment_data.append(['Year', 'Capacity [%]', 'Yearly Profit [kEUR]',
                                   'Inflated Profit [kEUR]', 'PV Contribution [kEUR]'])
            for _, row in years_df.iterrows():
                investment_data.append([
                    row['Year'], row['Capacity [%]'],
                    row['Yearly Profit [kEUR]'],
                    row['Inflated Profit [kEUR]'],
                    row['PV Contribution [kEUR]']
                ])

            investment_data.append([])
            investment_data.append(['Levelized ROI [%]', round(roi_percent, 2)])
            investment_data.append(['Total Investment [kEUR]',
                                   BATTERY_CAPACITY_KWH * 200 / 1000])

            inv_df = pd.DataFrame(investment_data)
            inv_df.to_excel(writer, sheet_name=country, index=False, header=False)
            logger.info(f"  ✓ Sheet: {country} (ROI: {roi_percent:.2f}%)")

    logger.info(f"  → Saved: {investment_path}")

    logger.info("\n" + "=" * 80)
    logger.info("OUTPUT FILES GENERATION COMPLETE")
    logger.info("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print("HUAWEI TECHARENA 2025 - PHASE 2 BESS OPTIMIZATION")
    print("=" * 80)
    print(f"Mode: {'TEST (3-day sample)' if TEST_MODE else 'FULL (365-day)'}")
    print(f"Data: {'Preprocessed parquet' if PREPROCESSED_DATA_READY else 'Excel (TechArena2025_Phase2_data.xlsx)'}")
    print(f"Duration: {TEST_DAYS} days")
    print(f"Scenarios: {len(COUNTRIES)} countries × {len(C_RATES)} C-rates = {len(COUNTRIES) * len(C_RATES)} total")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Setup logging
    logger = setup_logging()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full market data once if using Excel mode
    full_market_data = None
    if not PREPROCESSED_DATA_READY:
        logger.info("\n" + "=" * 80)
        logger.info("LOADING MARKET DATA FROM EXCEL")
        logger.info("=" * 80)
        excel_path = project_root / "Input" / "TechArena2025_Phase2_data.xlsx"
        logger.info(f"Loading from: {excel_path}")

        # Use optimizer's data loading method
        temp_optimizer = BESSOptimizerModelIII(alpha=ALPHA, use_afrr_ev_weighting=USE_AFRR_EV_WEIGHTING)
        full_market_data = temp_optimizer.load_and_preprocess_data(str(excel_path))
        logger.info(f"[OK] Loaded {len(full_market_data)} timesteps for all countries")
        logger.info(f"[OK] Countries available: {list(set([col[0] for col in full_market_data.columns]))}")
        logger.info(f"[OK] aFRR EV weighting: {USE_AFRR_EV_WEIGHTING}")
        logger.info("=" * 80)

    # Run all scenarios
    start_time = time.time()
    results_list = []

    for i, country in enumerate(COUNTRIES, 1):
        for j, c_rate in enumerate(C_RATES, 1):
            scenario_num = (i-1) * len(C_RATES) + j
            total_scenarios = len(COUNTRIES) * len(C_RATES)

            logger.info(f"\n\n{'='*80}")
            logger.info(f"SCENARIO {scenario_num}/{total_scenarios}")
            logger.info(f"{'='*80}")

            result = run_single_scenario(country, c_rate, logger, full_market_data)
            results_list.append(result)

    total_time = time.time() - start_time

    # Generate submission output files
    save_results_to_excel(results_list, OUTPUT_DIR, logger)

    # Final summary
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Scenarios completed: {len([r for r in results_list if r['status'] == 'success'])}/{len(results_list)}")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - TechArena_Phase2_Operation.xlsx")
    print("  - TechArena_Phase2_Configuration.xlsx")
    print("  - TechArena_Phase2_Investment.xlsx")
    print("=" * 80)


if __name__ == "__main__":
    main()
