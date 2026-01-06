# -*- coding: utf-8 -*-
"""
Batch Execution Script for Competition Submission

Generates MPC optimization results for all country-C-rate combinations:
- 5 countries: DE_LU, AT, CH, HU, CZ
- 3 C-rates: 0.25, 0.33, 0.5

Features:
- Flexible data loading: Excel (competition mode) or Parquet (development mode)
- Checkpointing: Auto-saves progress every 3 minutes (recoverable)
- Dual output:
  1. Detailed results in submission_results/ (for p2d_results_ana.py)
  2. Competition CSV files in output/ (for submission)

Data Loading Modes:
  PREPROCESSED_DATA_READY = False (default):
    → Loads from Input/TechArena2025_Phase2_data.xlsx (competition mode)
    → Suitable for final submission and evaluation

  PREPROCESSED_DATA_READY = True:
    → Loads from preprocessed parquet files (10-100x faster)
    → Suitable for development and testing

Output Files:
  submission_results/TIMESTAMP_country_crateX.X/
    - performance_summary.json (financial metrics)
    - solution_timeseries.csv (15-min decision variables)
    - iteration_summary.csv (daily MPC iteration stats)

  output/
    - TechArena_Phase2_Operation.csv (combined bidding schedule)
    - TechArena_Phase2_Configuration.csv (per-country C-rate comparison)
    - TechArena_Phase2_Investment.csv (per-country ROI analysis)

Usage:
  # Competition mode (load from Excel, generate all outputs)
  python run_submission_batch.py

  # Development mode (fast parquet loading)
  Set PREPROCESSED_DATA_READY = True, then:
  python run_submission_batch.py

  # Analyze & plot results (run after batch completes)
  python notebook/py_version/p2d_results_ana.py

Author: SoloGen Team
Date: November 2025
"""
# %%
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
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

# Results export
from py_script.validation.results_exporter import save_optimization_results

# Configure module logging levels
logging.getLogger('py_script.core.optimizer').setLevel(logging.WARNING)  # Suppress optimizer verbose output
logging.getLogger('py_script.mpc.mpc_simulator').setLevel(logging.INFO)  # Show MPC iteration progress

# CRITICAL: Suppress Pyomo's verbose DEBUG output
logging.getLogger('pyomo').setLevel(logging.ERROR)  # Only show Pyomo errors
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
logging.getLogger('pyomo.opt').setLevel(logging.ERROR)

# ================================================================================
# CONFIGURATION
# ================================================================================

# *** DATA LOADING MODE ***
# Set to False for competition submission (load from Excel)
# Set to True for development (use preprocessed parquet - 10-100x faster)
PREPROCESSED_DATA_READY = False

# Input data path (used only if PREPROCESSED_DATA_READY = False)
INPUT_EXCEL_PATH = "Input/TechArena2025-Phase2_onsite-challenge2025.xlsx"

# Test parameters
TEST_DURATION_DAYS = 3  # Full year for final submission
ALPHA = 1.0  # Full degradation cost
INITIAL_SOC_FRACTION = 0.5  # 50% initial SOC
USE_AFRR_EV_WEIGHTING = False  # aFRR energy activation probability weighting

# SOC limits (0-100% to avoid constraint bug)
SOC_MIN = 0.1
SOC_MAX = 0.9

# Degradation settings
REQUIRE_SEQUENTIAL = False  # Faster solving
LIFO_EPSILON_KWH = 0

# Optimizer settings
MAX_AS_RATIO = 0.9 # If set to 0.8, when C-rate = 0.25, max AS capacity = 4472 * 0.25 * 0.8 = 894.4 kW < 1MW miniumm bid requirement
ENABLE_CROSS_MARKET_EXCLUSIVITY = True

# Output settings
GENERATE_COMPETITION_CSV = True  # Generate 3 CSV files for competition submission
SKIP_CSV_EXPORT = False  # Set to True to skip CSV generation (faster testing)

# MPC settings (load from config)
config_dir = project_root / "data" / "p2_config"
with open(config_dir / "mpc_config.json", 'r') as f:
    mpc_config = json.load(f)

HORIZON_HOURS = mpc_config['mpc_parameters']['horizon_hours']
EXECUTION_HOURS = mpc_config['mpc_parameters']['execution_hours']
VALIDATE_CONSTRAINTS = False  # Disable for speed

# Solver settings
with open(config_dir / "solver_config.json", 'r') as f:
    solver_config = json.load(f)

DEFAULT_SOLVER = solver_config.get('default_solver', 'gurobi')
DEFAULT_SOLVER_TIME_LIMIT = solver_config.get('solver_time_limit_sec', 900)

# Checkpointing
ENABLE_CHECKPOINTING = True
CHECKPOINT_INTERVAL_MINUTES = 3

# Output settings
SAVE_RESULTS = True
BASE_OUTPUT_DIR = "submission_results"


# %% 
# country = 'AT'

# preprocessed_dir = project_root / "data" / "parquet" / "preprocessed"
# country_data = load_preprocessed_country_data(country, data_dir=preprocessed_dir)
# country_data.tail(23)

# %%

# ================================================================================
# SCENARIO DEFINITIONS (Priority-ordered)
# ================================================================================

SCENARIOS = [
    # Round 1: C-rate 0.25
    # ('CH', 0.25), # DONE
    # ('DE_LU', 0.25), #DONE
    # ('AT', 0.25), #DONE
    # ('HU', 0.25),  #DONE
    # ('CZ', 0.25), #DONE


    # # Round 2: C-rate 0.33
    # # ('CH', 0.33), #DONE
    # ('DE_LU', 0.33),  #DONE
    # ('AT', 0.33), #DONE
    ('HU', 0.33),  #DONE
    # ('CZ', 0.33), #DONE

    # Round 3: C-rate 0.5 (highest priority)
    # ('CH', 0.5), # DONE
    # ('DE_LU', 0.5),
    # ('AT', 0.5),
    # ('HU', 0.5),
    # ('CZ', 0.5),

]

# ================================================================================
# LOGGING SETUP
# ================================================================================
TIMENOW_STR = datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logging():
    """Setup logging: detailed file logs + MPC progress to console"""
    log_file = project_root / BASE_OUTPUT_DIR / f"batch_execution_{TIMENOW_STR}.log"

    # Configure root logger to capture all module logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO to reduce console noise

    # File handler - detailed logs (INFO and above - no DEBUG spam)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)  # Changed from DEBUG - we don't need Pyomo's DEBUG spam
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)

    # Console handler - show MPC iteration progress (INFO) + errors
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')  # Clean format for console
    ch.setFormatter(console_formatter)

    # Create filter to only show MPC iteration messages on console
    class MPCProgressFilter(logging.Filter):
        def filter(self, record):
            # Show messages from MPC simulator
            if record.name == 'py_script.mpc.mpc_simulator':
                # Show iteration headers, solving messages, and success/failure
                return any(keyword in record.getMessage() for keyword in [
                    '>>> MPC Iteration',
                    'Solving iteration',
                    '✓ Iteration',
                    '✗ Iteration',
                    '=======',
                    'MPC SIMULATOR INITIALIZED',
                    'STARTING FULL-YEAR MPC'
                ])
            # Show batch executor INFO messages
            if record.name == 'batch_executor':
                return record.levelno >= logging.INFO
            # Show all ERROR/CRITICAL from any module
            return record.levelno >= logging.ERROR

    ch.addFilter(MPCProgressFilter())
    root_logger.addHandler(ch)

    # Create batch executor logger
    logger = logging.getLogger('batch_executor')
    logger.setLevel(logging.INFO)

    return logger
# %%
# ================================================================================
# MAIN EXECUTION FUNCTION
# ================================================================================

def run_scenario(country, c_rate, logger, full_market_data=None):
    """
    Run MPC simulation for a single scenario.

    Parameters
    ----------
    country : str
        Country code (e.g., 'CH', 'DE_LU')
    c_rate : float
        C-rate (0.25, 0.33, or 0.5)
    logger : logging.Logger
        Logger instance
    full_market_data : pd.DataFrame, optional
        Pre-loaded full market data (MultiIndex DataFrame) or None

    Returns
    -------
    dict
        Results dictionary with keys: success, profit, revenue, degradation, etc.
    """
    scenario_start = time.time()

    try:
        logger.info("=" * 80)
        logger.info(f"STARTING: {country} | C-rate: {c_rate}")
        logger.info("=" * 80)

        # 1. Load market data
        logger.info(f"[1/5] Loading market data for {country}...")

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
            raise ValueError("No market data available. full_market_data must be provided when PREPROCESSED_DATA_READY=False")

        # Slice to test duration
        duration_timesteps = TEST_DURATION_DAYS * 96
        if duration_timesteps > len(country_data):
            logger.warning(f"Requested {duration_timesteps} timesteps but only {len(country_data)} available")
            country_data_slice = country_data.copy()
        else:
            country_data_slice = country_data.iloc[:duration_timesteps].copy()

        logger.info(f"  → Using {len(country_data_slice)} timesteps ({TEST_DURATION_DAYS} days)")

        # 2. Initialize optimizer
        logger.info(f"[2/5] Initializing optimizer (Alpha={ALPHA})...")
        optimizer = BESSOptimizerModelIII(alpha=ALPHA, use_afrr_ev_weighting=USE_AFRR_EV_WEIGHTING)

        # Configure optimizer
        optimizer.max_as_ratio = MAX_AS_RATIO
        optimizer.market_params['enable_cross_market_exclusivity'] = ENABLE_CROSS_MARKET_EXCLUSIVITY
        optimizer.battery_params['soc_min'] = SOC_MIN
        optimizer.battery_params['soc_max'] = SOC_MAX
        optimizer.degradation_params['lifo_epsilon_kwh'] = LIFO_EPSILON_KWH
        optimizer.degradation_params['require_sequential_segment_activation'] = REQUIRE_SEQUENTIAL

        logger.info(f"  Battery: {optimizer.battery_params['capacity_kwh']} kWh")
        logger.info(f"  SOC Limits: {SOC_MIN*100:.0f}% - {SOC_MAX*100:.0f}%")
        logger.info(f"  Degradation segments: {len(optimizer.degradation_params.get('marginal_costs', []))}")

        # 3. Setup MPC simulator
        logger.info(f"[3/5] Setting up MPC simulator...")
        logger.info(f"  Horizon: {HORIZON_HOURS}h | Execution: {EXECUTION_HOURS}h")

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=country_data_slice,
            horizon_hours=HORIZON_HOURS,
            execution_hours=EXECUTION_HOURS,
            c_rate=c_rate,
            validate_constraints=VALIDATE_CONSTRAINTS,
            solver_name=DEFAULT_SOLVER
        )

        # 4. Run MPC simulation
        logger.info(f"[4/5] Running MPC simulation...")
        logger.info(f"  Expected iterations: ~{len(country_data_slice) // (EXECUTION_HOURS * 4)}")

        if ENABLE_CHECKPOINTING:
            checkpoint_path = project_root / f"checkpoint_{country}_crate{c_rate}.pkl"
            logger.info(f"  Checkpointing enabled: every {CHECKPOINT_INTERVAL_MINUTES} min")

            mpc_results = simulator.run_full_simulation(
                initial_soc_fraction=INITIAL_SOC_FRACTION,
                checkpoint_interval_minutes=CHECKPOINT_INTERVAL_MINUTES,
                checkpoint_path=str(checkpoint_path)
            )

            # Clean up checkpoint file
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        else:
            mpc_results = simulator.run_full_simulation(
                initial_soc_fraction=INITIAL_SOC_FRACTION
            )

        logger.info(f"  ✓ Simulation complete!")
        logger.info(f"  Profit: €{mpc_results['net_profit']:,.2f}")
        logger.info(f"  Revenue: €{mpc_results['total_revenue']:,.2f}")
        logger.info(f"  Degradation: €{mpc_results['total_degradation_cost']:,.2f}")
        

        # 5. Transform and save results
        logger.info(f"[5/5] Transforming and saving results...")

        # Transform to visualization format
        total_bids_df = mpc_results['total_bids_df']
        viz_df = transform_mpc_results_for_viz(
            total_bids_df,
            country_data_slice,
            battery_capacity_kwh=4472.0
        )

        # Extract iteration summary
        iteration_summary = extract_iteration_summary(mpc_results, include_soc_trajectory=True)

        # Build summary metrics
        summary_metrics = {
            'model': 'Model_III_MPC',
            'country': country,
            'test_duration_days': TEST_DURATION_DAYS,
            'alpha': ALPHA,
            'c_rate': c_rate,

            # MPC settings
            "optimizer_settings": {
                "require_sequential": REQUIRE_SEQUENTIAL,
                "lifo_epsilon_kwh": LIFO_EPSILON_KWH,
            },
            'mpc_horizon_hours': HORIZON_HOURS,
            'mpc_execution_hours': EXECUTION_HOURS,
            'mpc_initial_soc_fraction': INITIAL_SOC_FRACTION,
            'mpc_iterations': len(mpc_results['iteration_results']),

            # Financial results
            'total_profit_eur': mpc_results['net_profit'],
            'total_revenue_eur': mpc_results['total_revenue'],
            'total_degradation_eur': mpc_results['total_degradation_cost'],

            # Revenue breakdown
            'revenue_da_eur': mpc_results.get('da_revenue', 0),
            'revenue_afrr_energy_eur': mpc_results.get('afrr_e_revenue', 0),
            'revenue_as_capacity_eur': mpc_results.get('as_revenue', 0),

            # Degradation breakdown
            'degradation_cyclic_eur': mpc_results.get('cyclic_cost', 0),
            'degradation_calendar_eur': mpc_results.get('calendar_cost', 0),

            # SOC metrics
            'initial_soc_kwh': mpc_results.get('soc_trajectory', [INITIAL_SOC_FRACTION * 4472])[0],
            'final_soc_kwh': mpc_results.get('final_soc', INITIAL_SOC_FRACTION * 4472),

            # Timing
            'simulation_time_sec': time.time() - scenario_start,

            # Solver settings
            'solver': DEFAULT_SOLVER,
            'solver_time_limit_sec': DEFAULT_SOLVER_TIME_LIMIT,

            # Data source
            'data_source': 'preprocessed'
        }

        # Generate run name
        run_name = f"{country}_crate{c_rate}"

        # Save results
        output_directory = save_optimization_results(
            viz_df,
            summary_metrics,
            run_name,
            base_output_dir=str(project_root / BASE_OUTPUT_DIR)
        )

        # Save iteration summary
        iteration_csv_path = output_directory / "iteration_summary.csv"
        iteration_summary.to_csv(iteration_csv_path, index=False)

        logger.info(f"  ✓ Results saved to: {output_directory.name}")

        scenario_time = time.time() - scenario_start
        logger.info(f"✓ COMPLETED: {country} | C-rate: {c_rate} | Time: {scenario_time/60:.2f} min")

        return {
            'success': True,
            'country': country,
            'c_rate': c_rate,
            'profit': mpc_results['net_profit'],
            'revenue': mpc_results['total_revenue'],
            'degradation': mpc_results['total_degradation_cost'],
            'final_soc': mpc_results['final_soc'],
            'iterations': len(mpc_results['iteration_results']),
            'solve_time': scenario_time,
            'status': 'SUCCESS',
            'error': None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        scenario_time = time.time() - scenario_start
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"✗ FAILED: {country} | C-rate: {c_rate}")
        logger.error(f"  Error: {error_msg}")
        logger.error(f"  Traceback:\n{traceback.format_exc()}")

        return {
            'success': False,
            'country': country,
            'c_rate': c_rate,
            'profit': None,
            'revenue': None,
            'degradation': None,
            'final_soc': None,
            'iterations': None,
            'solve_time': scenario_time,
            'status': 'FAILED',
            'error': error_msg,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# %%
# ================================================================================
# MAIN BATCH EXECUTION
# ================================================================================

def main():
    """Main batch execution function"""

    # Setup logging
    logger = setup_logging()

    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH EXECUTION: FINAL SUBMISSION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Data mode: {'Preprocessed parquet' if PREPROCESSED_DATA_READY else 'Excel (TechArena2025_Phase2_data.xlsx)'}")
    logger.info(f"Total scenarios: {len(SCENARIOS)}")
    logger.info(f"Test duration: {TEST_DURATION_DAYS} days")
    logger.info(f"Alpha: {ALPHA}")
    logger.info(f"MPC Settings: {HORIZON_HOURS}h horizon / {EXECUTION_HOURS}h execution")
    logger.info(f"Solver: {DEFAULT_SOLVER}")
    logger.info(f"Output directory: {BASE_OUTPUT_DIR}")
    logger.info(f"Generate CSV: {GENERATE_COMPETITION_CSV and not SKIP_CSV_EXPORT}")
    logger.info("=" * 80)
    logger.info("")

    # Console summary
    print("=" * 60)
    print("BATCH EXECUTION: FINAL SUBMISSION")
    print("=" * 60)
    print(f"Data: {'Parquet (fast)' if PREPROCESSED_DATA_READY else 'Excel (competition)'}")
    print(f"Scenarios: {len(SCENARIOS)} ({TEST_DURATION_DAYS} days each)")
    print(f"Solver: {DEFAULT_SOLVER} | Horizon: {HORIZON_HOURS}h / Exec: {EXECUTION_HOURS}h")
    print(f"Output: {BASE_OUTPUT_DIR}/")
    print("=" * 60)

    # Load full market data once if using Excel mode
    full_market_data = None
    if not PREPROCESSED_DATA_READY:
        logger.info("\n" + "=" * 80)
        logger.info("LOADING MARKET DATA FROM EXCEL")
        logger.info("=" * 80)
        excel_path = project_root / INPUT_EXCEL_PATH
        logger.info(f"Loading from: {excel_path}")

        if not excel_path.exists():
            logger.error(f"Excel file not found: {excel_path}")
            raise FileNotFoundError(f"Excel file not found: {excel_path}")

        # Use optimizer's data loading method
        temp_optimizer = BESSOptimizerModelIII(alpha=ALPHA, use_afrr_ev_weighting=USE_AFRR_EV_WEIGHTING)
        full_market_data = temp_optimizer.load_and_preprocess_data(str(excel_path))
        logger.info(f"[OK] Loaded {len(full_market_data)} timesteps for all countries")
        logger.info(f"[OK] Countries available: {list(set([col[0] for col in full_market_data.columns]))}")
        logger.info(f"[OK] aFRR EV weighting: {USE_AFRR_EV_WEIGHTING}")
        logger.info("=" * 80)

    # Execute all scenarios
    batch_start = time.time()
    results_list = []

    for i, (country, c_rate) in enumerate(SCENARIOS, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"SCENARIO {i}/{len(SCENARIOS)}: {country} @ C-rate {c_rate}")
        logger.info(f"{'=' * 80}\n")

        # Console progress
        print(f"\n[{i}/{len(SCENARIOS)}] {country} @ C-rate {c_rate}...", flush=True)

        result = run_scenario(country, c_rate, logger, full_market_data)
        results_list.append(result)

        # Brief summary
        if result['success']:
            logger.info(f"✓ Success | Profit: €{result['profit']:,.2f} | Time: {result['solve_time']/60:.1f} min")
            print(f"  ✓ €{result['profit']:,.0f} | {result['solve_time']/60:.1f}min")
        else:
            logger.info(f"✗ Failed | Error: {result['error']}")
            print(f"  ✗ FAILED: {result['error']}")

    batch_time = time.time() - batch_start

    # Generate batch summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 80)

    summary_df = pd.DataFrame(results_list)
    summary_csv_path = project_root / BASE_OUTPUT_DIR / f"batch_summary_{TIMENOW_STR}.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Statistics
    n_success = summary_df['success'].sum()
    n_failed = len(summary_df) - n_success

    logger.info(f"Total scenarios: {len(SCENARIOS)}")
    logger.info(f"  Successful: {n_success}")
    logger.info(f"  Failed: {n_failed}")
    logger.info(f"Total time: {batch_time/60:.2f} min ({batch_time/3600:.2f} hours)")

    if n_success > 0:
        successful_results = summary_df[summary_df['success']]
        logger.info(f"\nFinancial Summary (Successful Runs):")
        logger.info(f"  Total Profit: €{successful_results['profit'].sum():,.2f}")
        logger.info(f"  Avg Profit: €{successful_results['profit'].mean():,.2f}")
        logger.info(f"  Profit Range: €{successful_results['profit'].min():,.2f} - €{successful_results['profit'].max():,.2f}")

    if n_failed > 0:
        logger.info(f"\nFailed Scenarios:")
        failed_results = summary_df[~summary_df['success']]
        for _, row in failed_results.iterrows():
            logger.info(f"  - {row['country']} @ C-rate {row['c_rate']}: {row['error']}")

    logger.info(f"\nBatch summary saved to: {summary_csv_path}")

    # Generate competition CSV files
    if GENERATE_COMPETITION_CSV and not SKIP_CSV_EXPORT and n_success > 0:
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING COMPETITION CSV FILES")
        logger.info("=" * 80)

        try:
            # Import conversion functions
            from py_script.submission.convert_results import (
                load_result_from_directory,
                convert_operation_file,
                calculate_10year_roi,
                WACC, INFLATION, BATTERY_CAPACITY_KWH
            )
            from py_script.mpc.transform_mpc_results import transform_mpc_results_for_viz

            # Create output directory
            output_dir = project_root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load results from submission_results directories
            logger.info("[1/3] Loading results from scenario directories...")
            loaded_results = []
            for _, row in successful_results.iterrows():
                # Find scenario directory
                scenario_pattern = f"*{row['country'].lower()}*crate{row['c_rate']}*"
                matches = list((project_root / BASE_OUTPUT_DIR).glob(scenario_pattern))
                if matches:
                    scenario_dir = matches[0]
                    try:
                        result_data = load_result_from_directory(scenario_dir)
                        loaded_results.append(result_data)
                        logger.info(f"  [OK] Loaded: {result_data['country']} @ C-rate {result_data['c_rate']}")
                    except Exception as e:
                        logger.warning(f"  [SKIP] {scenario_dir.name}: {e}")

            if not loaded_results:
                logger.error("No results could be loaded for CSV generation!")
            else:
                logger.info(f"[OK] Loaded {len(loaded_results)} scenario results")

                # Generate Operation CSV (combined file)
                logger.info("\n[2/3] Generating Operation CSV...")
                operation_rows = []
                for result in loaded_results:
                    operation_df = convert_operation_file(result['solution_df'])
                    operation_df['Country'] = result['country']
                    operation_df['C-rate'] = result['c_rate']
                    operation_rows.append(operation_df)

                combined_operation_df = pd.concat(operation_rows, ignore_index=True)
                operation_csv_path = output_dir / "TechArena_Phase2_Operation.csv"
                combined_operation_df.to_csv(operation_csv_path, index=False)
                logger.info(f"  [SAVED] {operation_csv_path} ({len(combined_operation_df)} rows)")

                # Generate Configuration CSV
                logger.info("\n[3/3] Generating Configuration & Investment CSVs...")
                config_rows = []
                investment_rows = []

                # Group by country
                country_results = {}
                for result in loaded_results:
                    country = result['country']
                    if country not in country_results:
                        country_results[country] = []
                    country_results[country].append(result)

                for country, country_res in sorted(country_results.items()):
                    # Configuration data
                    for result in sorted(country_res, key=lambda x: x['c_rate']):
                        c_rate = result['c_rate']
                        perf = result['performance']

                        battery_power_mw = BATTERY_CAPACITY_KWH * c_rate / 1000
                        annual_profit = perf['total_profit_eur']
                        yearly_profit_per_mw = (annual_profit / battery_power_mw) / 1000

                        roi_percent, _ = calculate_10year_roi(annual_profit, country)

                        cyclic_cost = perf.get('degradation_cyclic_eur', 0)
                        num_cycles = cyclic_cost / 100 if cyclic_cost > 0 else 0

                        config_rows.append({
                            'Country': country,
                            'C-rate': c_rate,
                            'number of cycles': round(num_cycles, 2),
                            'yearly profits [kEUR/MW]': round(yearly_profit_per_mw, 2),
                            'levelized ROI [%]': round(roi_percent, 2)
                        })

                    # Investment data (best C-rate per country)
                    best_result = max(country_res, key=lambda x: x['performance']['total_profit_eur'])
                    annual_profit = best_result['performance']['total_profit_eur']
                    c_rate = best_result['c_rate']
                    roi_percent, years_df = calculate_10year_roi(annual_profit, country)

                    investment_rows.append({
                        'Country': country,
                        'WACC': WACC[country],
                        'Inflation Rate': INFLATION[country],
                        'Discount Rate': WACC[country],
                        'Yearly Profits (2024) [kEUR]': annual_profit / 1000,
                        'Best C-rate': c_rate,
                        'Levelized ROI [%]': round(roi_percent, 2),
                        'Total Investment [kEUR]': BATTERY_CAPACITY_KWH * 200 / 1000
                    })

                config_df = pd.DataFrame(config_rows)
                config_csv_path = output_dir / "TechArena_Phase2_Configuration.csv"
                config_df.to_csv(config_csv_path, index=False)
                logger.info(f"  [SAVED] {config_csv_path} ({len(config_df)} rows)")

                investment_df = pd.DataFrame(investment_rows)
                investment_csv_path = output_dir / "TechArena_Phase2_Investment.csv"
                investment_df.to_csv(investment_csv_path, index=False)
                logger.info(f"  [SAVED] {investment_csv_path} ({len(investment_df)} rows)")

                logger.info("\n[OK] All competition CSV files generated successfully!")

        except Exception as e:
            logger.error(f"[ERROR] CSV generation failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("BATCH EXECUTION COMPLETE")
    logger.info("=" * 80)

    # Console summary
    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"Success: {n_success}/{len(SCENARIOS)} | Failed: {n_failed}")
    print(f"Total time: {batch_time/60:.1f}min ({batch_time/3600:.2f}h)")
    if n_success > 0:
        successful_results = summary_df[summary_df['success']]
        print(f"Total Profit: €{successful_results['profit'].sum():,.0f}")
        print(f"Avg Profit: €{successful_results['profit'].mean():,.0f}")
    print(f"Results: {summary_csv_path}")
    if GENERATE_COMPETITION_CSV and not SKIP_CSV_EXPORT and n_success > 0:
        print(f"\nCompetition CSV files saved to: output/")
        print("  - TechArena_Phase2_Operation.csv")
        print("  - TechArena_Phase2_Configuration.csv")
        print("  - TechArena_Phase2_Investment.csv")
    print("\nTo analyze and plot results, run:")
    print("  python notebook/py_version/p2d_results_ana.py")
    print("=" * 60)

    return summary_df

# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    summary = main()
    print(f"\n✓ Batch execution complete! Check {BASE_OUTPUT_DIR}/batch_summary_{TIMENOW_STR}.csv for results.")

# %%
