# -*- coding: utf-8 -*-
"""
Phase 2 MPC Testing - Interactive Version

This script provides a flexible testing and validation harness for the MPC (Model Predictive Control) simulation framework.

This version has cell markers (# %%) for block-by-block execution in VS Code.
Run cells individually using Shift+Enter or clicking "Run Cell" above each block.
"""

# %%
# ================================================================================
# [SECTION 1] SETUP & IMPORTS
# ================================================================================

# Standard library imports
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Data processing
import pandas as pd
import numpy as np

# Optimization models
from src.core.optimizer import BESSOptimizerModelIII

# MPC simulation
from src.mpc.mpc_simulator import MPCSimulator
from src.mpc.meta_optimizer import MetaOptimizer
from src.mpc.transform_mpc_results import (
    transform_mpc_results_for_viz,
    extract_iteration_summary
)

# Data loading
from src.data.load_process_market_data import load_preprocessed_country_data

# Standard visualization utilities
from src.visualization.optimization_analysis import (
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid,
    plot_soc_and_power_bids
)

# MPC-specific visualization
from src.visualization.mpc_analysis import (
    plot_iteration_boundaries,
    plot_iteration_performance,
    plot_state_continuity
)

# Results export
from src.validation.results_exporter import save_optimization_results

print("[OK] All imports successful!")
print(f"Project root: {project_root}")

# %%
# ================================================================================
# [SECTION 2] CONFIGURATION
# ================================================================================

# ============================================================================
# Load From Saved Results (OPTIONAL)
# ============================================================================

# Set to True to skip simulation and load from saved results instead
LOAD_FROM_SAVED = False

# Path to saved MPC results directory (only used if LOAD_FROM_SAVED = True)
saved_stamp = "your_timestamp_here"  # e.g., "20251115_141102"
SAVED_RESULTS_DIR = f"validation_results/mpc_validation/{saved_stamp}"

# ============================================================================
# Configuration Files
# ============================================================================

# Define configuration paths
config_dir = project_root / "data" / "p2_config"

# Load MPC configuration
mpc_config_path = config_dir / "mpc_config.json"
with open(mpc_config_path, 'r') as f:
    mpc_config = json.load(f)
    print(f"[OK] Loaded MPC config: {mpc_config_path.name}")

# Load MPC test configuration
mpc_test_config_path = config_dir / "mpc_test_config.json"
with open(mpc_test_config_path, 'r') as f:
    mpc_test_config = json.load(f)
    print(f"[OK] Loaded MPC test config: {mpc_test_config_path.name}")

# Load solver config
solver_config_path = config_dir / "solver_config.json"
with open(solver_config_path, 'r') as f:
    solver_config = json.load(f)
    print(f"[OK] Loaded solver config: {solver_config_path.name}")

# Load aging config
aging_config_path = config_dir / "aging_config.json"
with open(aging_config_path, 'r') as f:
    aging_config = json.load(f)
    print(f"[OK] Loaded aging config: {aging_config_path.name}")

# Load aFRR EV weights config
afrr_ev_config_path = config_dir / "afrr_ev_weights_config.json"
with open(afrr_ev_config_path, 'r') as f:
    afrr_ev_config = json.load(f)
    print(f"[OK] Loaded aFRR EV config: {afrr_ev_config_path.name}")

print("\nConfiguration files loaded successfully!")

# Extract solver settings
DEFAULT_SOLVER = solver_config.get('default_solver', 'cbc')
DEFAULT_SOLVER_TIME_LIMIT = solver_config.get('solver_time_limit_sec', 900)

print(f"\n[SOLVER] Default solver: {DEFAULT_SOLVER}")
print(f"[SOLVER] Time limit: {DEFAULT_SOLVER_TIME_LIMIT}s")

# %%
# ============================================================================
# Extract Scenario Parameters from Config
# ============================================================================

# ===
# Test scenario (from mpc_test_config.json)
TEST_COUNTRY = mpc_test_config['test_scenario']['country'] # DE_LU, CH, AT, CZ, HU
TEST_C_RATE = mpc_test_config['test_scenario']['c_rate']
DATA_SOURCE = mpc_test_config['test_scenario']['data_source'] # 'preprocessed' or 'excel'
# Alpha settings
ALPHA_MODE = mpc_test_config['alpha_settings']['mode']
SINGLE_ALPHA = mpc_test_config['alpha_settings']['single_alpha']
ALPHA_SWEEP_RANGE = mpc_test_config['alpha_settings']['sweep_range']
# Meta-optimizer settings
ENABLE_META_OPTIMIZER = mpc_test_config['meta_optimizer']['enabled']
META_N_JOBS = mpc_test_config['meta_optimizer']['n_jobs']
META_WACC = mpc_test_config['meta_optimizer']['wacc']
META_INFLATION = mpc_test_config['meta_optimizer']['inflation_rate']
META_LIFETIME = mpc_test_config['meta_optimizer']['project_lifetime_years']
# Max iterations (from mpc_test_config.json)
# MAX_ITERATIONS = mpc_test_config['mpc_execution']['max_iterations']

# Visualization settings
ENABLE_STANDARD_PLOTS = mpc_test_config['visualization']['enable_standard_plots']
ENABLE_MPC_PLOTS = mpc_test_config['visualization']['enable_mpc_plots']
MPC_PLOT_OPTIONS = mpc_test_config['visualization']['mpc_plot_options']
SAVE_FORMAT = mpc_test_config['visualization']['save_format']

# Output settings
SAVE_RESULTS = mpc_test_config['output']['save_results']
BASE_OUTPUT_DIR = mpc_test_config['output']['base_output_dir']
AUTO_GENERATE_RUN_NAME = mpc_test_config['output']['auto_generate_run_name']
CUSTOM_RUN_NAME = mpc_test_config['output']['custom_run_name']


# ===
# MPC execution settings (from mpc_config.json)
HORIZON_HOURS = mpc_config['mpc_parameters']['horizon_hours']
EXECUTION_HOURS = mpc_config['mpc_parameters']['execution_hours']
TEST_DURATION_DAYS = mpc_config['mpc_parameters']['duration_days']
INITIAL_SOC_FRACTION = mpc_config['mpc_parameters']['initial_soc_fraction']
VALIDATE_CONSTRAINTS = mpc_config['mpc_parameters']['validate_constraints']


# Optimizer configuration (applies to single-alpha mode only)
ENABLE_CROSS_MARKET_EXCLUSIVITY = True  # Set to False to disable Cst-8 (reduces constraints)
MAX_AS_RATIO = 0.8                      # Max ancillary service ratio (80%)

# SOC operating limits (fraction of battery capacity)
SOC_MIN = 0.1  # Minimum SOC (0% - 100% allowed per challenge rules)
SOC_MAX = 0.9  # Maximum SOC (100%)
REQUIRE_SEQUENTIAL = True          # Enforce sequential activation for charging the BESS
LIFO_EPSILON_KWH = 0            # Tolerance for LIFO segment fullness (kWh)


# Checkpoint configuration (for long-running simulations)
ENABLE_CHECKPOINTING = True             # Enable automatic checkpoint saving
CHECKPOINT_INTERVAL_MINUTES = 5        # Save checkpoint every N minutes
# Note: For meta-optimizer mode, modify MetaOptimizer class to accept these parameters

# Display scenario summary
print("=" * 80)
print("[CONFIG] MPC TEST SCENARIO CONFIGURATION")
print("=" * 80)
print(f"Country:              {TEST_COUNTRY}")
print(f"Test Duration:        {TEST_DURATION_DAYS} days")
print(f"C-Rate:               {TEST_C_RATE}")
print(f"Data Source:          {DATA_SOURCE}")
print()
print("MPC Settings:")
print(f"  Horizon:            {HORIZON_HOURS} hours")
print(f"  Execution:          {EXECUTION_HOURS} hours")
print(f"  Initial SOC:        {INITIAL_SOC_FRACTION * 100:.0f}%")
# print(f"  Max Iterations:     {MAX_ITERATIONS if MAX_ITERATIONS else 'Full duration'}")
print()
print("Optimizer Settings:")
print(f"  Solver:             {DEFAULT_SOLVER.upper()}")
print(f"  Max AS Ratio:       {MAX_AS_RATIO * 100:.0f}%")
print(f"  SOC Limits:         {SOC_MIN * 100:.0f}% - {SOC_MAX * 100:.0f}%")
print(f"  Cross-Market Exclusivity (Cst-8): {ENABLE_CROSS_MARKET_EXCLUSIVITY}")
print(f"  Checkpointing:      {'Enabled' if ENABLE_CHECKPOINTING else 'Disabled'}")
if ENABLE_CHECKPOINTING:
    print(f"    Interval:         {CHECKPOINT_INTERVAL_MINUTES} minutes")
print()
print("Alpha Settings:")
if ALPHA_MODE == 'single':
    print(f"  Mode:               Single alpha")
    print(f"  Alpha:              {SINGLE_ALPHA}")
else:
    print(f"  Mode:               Sweep")
    print(f"  Range:              {ALPHA_SWEEP_RANGE['min']} - {ALPHA_SWEEP_RANGE['max']} (step {ALPHA_SWEEP_RANGE['step']})")
print()
print("Meta-Optimizer:")
print(f"  Enabled:            {ENABLE_META_OPTIMIZER}")
if ENABLE_META_OPTIMIZER:
    print(f"  Parallel Jobs:      {META_N_JOBS}")
    print(f"  WACC:               {META_WACC * 100:.1f}%")
    print(f"  Inflation:          {META_INFLATION * 100:.1f}%")
    print(f"  Project Lifetime:   {META_LIFETIME} years")
print("=" * 80)

# %%
# ============================================================================
# Load Market Data
# ============================================================================

# Calculate number of timesteps
duration_timesteps = TEST_DURATION_DAYS * 96  # 96 timesteps per day (15-min intervals)

if DATA_SOURCE == 'preprocessed':
    # Option 1: Load preprocessed country-specific parquet (FASTEST)
    preprocessed_dir = project_root / "data" / "parquet" / "preprocessed"
    preprocessed_path = preprocessed_dir / f"{TEST_COUNTRY.lower()}.parquet"
    
    if preprocessed_path.exists():
        print(f"[FAST PATH] Loading preprocessed data: {preprocessed_path.name}")
        country_data = load_preprocessed_country_data(TEST_COUNTRY, data_dir=preprocessed_dir)
        print(f"[OK] Loaded {len(country_data)} time steps for {TEST_COUNTRY} (preprocessed)")
    else:
        print(f"ERROR: Preprocessed file not found: {preprocessed_path}")
        print("Falling back to Excel...")
        DATA_SOURCE = 'excel'

if DATA_SOURCE == 'excel':
    # Option 2: Load from Excel (SUBMISSION PATH)
    excel_path = project_root / "data" / "TechArena2025_Phase2_data.xlsx"
    
    if excel_path.exists():
        print(f"[SUBMISSION PATH] Loading from Excel: {excel_path.name}")
        print("   This matches Huawei submission requirements...")
        
        # Create temporary optimizer for data loading
        temp_opt = BESSOptimizerModelIII()
        
        # Load using Phase 2 Excel loader
        print("   Loading Phase 2 market tables from Excel...")
        full_data = temp_opt.load_and_preprocess_data(str(excel_path))
        
        # Extract country-specific data
        print(f"   Extracting country data for {TEST_COUNTRY}...")
        country_data = temp_opt.extract_country_data(full_data, TEST_COUNTRY)
        print(f"[OK] Loaded {len(country_data)} time steps for {TEST_COUNTRY} (Excel)")
    else:
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

# Slice to test duration
if duration_timesteps > len(country_data):
    print(f"\nWARNING: Requested {duration_timesteps} timesteps but only {len(country_data)} available")
    print(f"Using full dataset ({len(country_data) // 96} days)")
    country_data_slice = country_data.copy()
else:
    country_data_slice = country_data.iloc[:duration_timesteps].copy()
    print(f"\n[OK] Sliced to {TEST_DURATION_DAYS} days ({len(country_data_slice)} timesteps)")

# Display data summary
print(f"\nMarket Data Summary:")
print(f"   DA Price:           {country_data_slice['price_day_ahead'].min():.2f} - {country_data_slice['price_day_ahead'].max():.2f} EUR/MWh")
print(f"   FCR Price:          {country_data_slice['price_fcr'].min():.2f} - {country_data_slice['price_fcr'].max():.2f} EUR/MW")
print(f"   aFRR+ Cap Price:    {country_data_slice['price_afrr_pos'].min():.2f} - {country_data_slice['price_afrr_pos'].max():.2f} EUR/MW")
print(f"   aFRR- Cap Price:    {country_data_slice['price_afrr_neg'].min():.2f} - {country_data_slice['price_afrr_neg'].max():.2f} EUR/MW")
print(f"   aFRR+ Energy Price: {country_data_slice['price_afrr_energy_pos'].min():.2f} - {country_data_slice['price_afrr_energy_pos'].max():.2f} EUR/MWh")
print(f"   aFRR- Energy Price: {country_data_slice['price_afrr_energy_neg'].min():.2f} - {country_data_slice['price_afrr_energy_neg'].max():.2f} EUR/MWh")


# %%
# ================================================================================
# [SECTION 3] RUN MPC SIMULATION OR LOAD SAVED RESULTS
# ================================================================================

if LOAD_FROM_SAVED:
    # ============================================================================
    # Load Previously Saved MPC Results
    # ============================================================================

    print("\n" + "=" * 80)
    print("[LOAD] LOADING SAVED MPC RESULTS")
    print("=" * 80)

    # Construct absolute path from project root
    saved_dir = project_root / SAVED_RESULTS_DIR

    if not saved_dir.exists():
        raise FileNotFoundError(f"Saved results directory not found: {saved_dir}")

    print(f"\nLoading from: {saved_dir}")

    # Load performance summary
    perf_file = saved_dir / "performance_summary.json"
    with open(perf_file, 'r') as f:
        perf_summary = json.load(f)
    print(f"  [OK] Loaded performance_summary.json")

    # Load iteration summary
    iter_file = saved_dir / "iteration_summary.csv"
    iteration_summary = pd.read_csv(iter_file)
    print(f"  [OK] Loaded iteration_summary.csv ({len(iteration_summary)} iterations)")

    # Load solution timeseries
    sol_file = saved_dir / "solution_timeseries.csv"
    viz_df = pd.read_csv(sol_file, index_col=0)
    if 'timestamp' in viz_df.columns:
        viz_df['timestamp'] = pd.to_datetime(viz_df['timestamp'])
    print(f"  [OK] Loaded solution_timeseries.csv ({len(viz_df)} timesteps)")

    # Extract parameters from saved results
    TEST_COUNTRY = perf_summary['country']
    TEST_DURATION_DAYS = perf_summary['test_duration_days']
    TEST_C_RATE = perf_summary['c_rate']
    used_alpha = perf_summary['alpha']
    HORIZON_HOURS = perf_summary['mpc_horizon_hours']
    EXECUTION_HOURS = perf_summary['mpc_execution_hours']
    simulation_time = perf_summary['simulation_time_sec']

    # Reconstruct SOC trajectory from iteration boundaries
    # Extract SOC values at end of each iteration from solution timeseries
    soc_trajectory = [perf_summary['initial_soc_kwh']]
    for _, iter_row in iteration_summary.iterrows():
        end_idx = int(iter_row['end_timestep'])  # Convert to Python int for .iloc
        if end_idx < len(viz_df):
            soc_trajectory.append(viz_df['soc_kwh'].iloc[end_idx])
    # Ensure final SOC is included
    if len(soc_trajectory) <= len(iteration_summary):
        soc_trajectory.append(perf_summary['final_soc_kwh'])

    # Reconstruct mpc_results for plotting functions
    mpc_results = {
        'total_revenue': perf_summary['total_revenue_eur'],
        'total_degradation_cost': perf_summary['total_degradation_eur'],
        'net_profit': perf_summary['total_profit_eur'],
        'final_soc': perf_summary['final_soc_kwh'],
        'soc_trajectory': soc_trajectory,
        'soc_15min': viz_df['soc_kwh'].tolist(),
        'iteration_results': iteration_summary.to_dict('records'),
        'da_revenue': perf_summary.get('revenue_da_eur', 0),
        'afrr_e_revenue': perf_summary.get('revenue_afrr_energy_eur', 0),
        'as_revenue': perf_summary.get('revenue_as_capacity_eur', 0),
        'cyclic_cost': perf_summary.get('degradation_cyclic_eur', 0),
        'calendar_cost': perf_summary.get('degradation_calendar_eur', 0),
    }

    # Build summary metrics
    summary_metrics = perf_summary.copy()

    # Set output directory to saved location
    output_directory = saved_dir
    SAVE_RESULTS = False  # Don't overwrite existing results

    # Set visualization settings for loaded results (if not already defined)
    if 'ENABLE_MPC_PLOTS' not in dir():
        ENABLE_MPC_PLOTS = True  # Enable plotting by default
    if 'MPC_PLOT_OPTIONS' not in dir():
        MPC_PLOT_OPTIONS = {
            'iteration_boundaries': True,
            'iteration_performance': True,
            'state_continuity': True
        }
    if 'SAVE_FORMAT' not in dir():
        SAVE_FORMAT = 'html'

    print("\n" + "=" * 80)
    print("[RESULTS] LOADED MPC RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Revenue:          €{mpc_results['total_revenue']:,.2f}")
    print(f"Total Degradation Cost: €{mpc_results['total_degradation_cost']:,.2f}")
    print(f"Net Profit:             €{mpc_results['net_profit']:,.2f}")
    print()
    print(f"Final SOC:              {mpc_results['final_soc']:.2f} kWh")
    print(f"Number of Iterations:   {len(mpc_results['iteration_results'])}")
    print(f"Simulation Time:        {simulation_time:.2f}s ({simulation_time/60:.2f} min)")
    print("=" * 80)

    # Skip to visualization sections
    print("\n[INFO] Skipping simulation - using loaded results")
    print("[INFO] Jump to SECTION 5 or SECTION 6 for visualizations")

else:
    # ============================================================================
    # Initialize and Run MPC Simulation
    # ============================================================================

    print("\n" + "=" * 80)
    print("[RUN] RUNNING MPC SIMULATION")
    print("=" * 80)

    simulation_start = time.time()

    if ENABLE_META_OPTIMIZER and ALPHA_MODE == 'sweep':
        # WARNING: MetaOptimizer API is outdated and needs updating
        # For now, use single-alpha mode instead
        raise NotImplementedError(
            "MetaOptimizer mode is currently disabled due to API mismatch. "
            "Please use single-alpha mode (set ENABLE_META_OPTIMIZER=False in mpc_test_config.json)"
        )

        # Use Meta-Optimizer for alpha sweep
        print(f"\n[MODE] Meta-Optimizer (Alpha Sweep)")
        print(f"   Alpha range: {ALPHA_SWEEP_RANGE['min']} - {ALPHA_SWEEP_RANGE['max']} (step {ALPHA_SWEEP_RANGE['step']})")
        print(f"   Parallel jobs: {META_N_JOBS}")
        print()

        # Generate alpha values
        alpha_values = np.arange(
            ALPHA_SWEEP_RANGE['min'],
            ALPHA_SWEEP_RANGE['max'] + ALPHA_SWEEP_RANGE['step'] / 2,
            ALPHA_SWEEP_RANGE['step']
        )

        # Initialize Meta-Optimizer
        meta_opt = MetaOptimizer(
            optimizer_class=BESSOptimizerModelIII,
            country_data=country_data_slice,
            c_rate=TEST_C_RATE,
            horizon_hours=HORIZON_HOURS,
            execution_hours=EXECUTION_HOURS,
            wacc=META_WACC,
            inflation_rate=META_INFLATION,
            project_lifetime_years=META_LIFETIME,
            n_jobs=META_N_JOBS
        )

        # Run meta-optimization
        best_alpha, best_roi, all_results = meta_opt.optimize_alpha(
            alpha_values,
            initial_soc_fraction=INITIAL_SOC_FRACTION,
            # max_iterations=MAX_ITERATIONS
        )

        # Extract best result
        mpc_results = all_results[best_alpha]['mpc_results']
        used_alpha = best_alpha

        print(f"\n[OK] Meta-Optimization Complete!")
        print(f"   Best Alpha: {best_alpha}")
        print(f"   Best ROI:   {best_roi:.2f}%")

    else:
        # Single alpha MPC simulation
        used_alpha = SINGLE_ALPHA
        print(f"\n[MODE] Single Alpha MPC Simulation")
        print(f"   Alpha: {used_alpha}")
        print(f"   Horizon: {HORIZON_HOURS}h")
        print(f"   Execution: {EXECUTION_HOURS}h")
        print()

        # Initialize optimizer
        optimizer = BESSOptimizerModelIII(alpha=used_alpha)

        # Configure optimizer settings
        optimizer.max_as_ratio = MAX_AS_RATIO
        optimizer.market_params['enable_cross_market_exclusivity'] = ENABLE_CROSS_MARKET_EXCLUSIVITY
        optimizer.battery_params['soc_min'] = SOC_MIN  # Apply SOC limits
        optimizer.battery_params['soc_max'] = SOC_MAX
        optimizer.degradation_params['lifo_epsilon_kwh'] = LIFO_EPSILON_KWH
        optimizer.degradation_params['require_sequential'] = REQUIRE_SEQUENTIAL


        # Initialize MPC simulator
        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=country_data_slice,
            horizon_hours=HORIZON_HOURS,
            execution_hours=EXECUTION_HOURS,
            c_rate=TEST_C_RATE,
            validate_constraints=VALIDATE_CONSTRAINTS,
            solver_name=DEFAULT_SOLVER
        )

        # Run simulation with optional checkpoint saving
        if ENABLE_CHECKPOINTING:
            checkpoint_path = project_root / f"mpc_checkpoint_{TEST_COUNTRY}_{TEST_C_RATE}_{TEST_DURATION_DAYS}_backup.pkl"
            print(f"   Checkpointing enabled: every {CHECKPOINT_INTERVAL_MINUTES} minutes")
            print(f"   Checkpoint file: {checkpoint_path}")
            mpc_results = simulator.run_full_simulation(
                initial_soc_fraction=INITIAL_SOC_FRACTION,
                # max_iterations=MAX_ITERATIONS,
                checkpoint_interval_minutes=CHECKPOINT_INTERVAL_MINUTES,
                checkpoint_path=str(checkpoint_path)
            )
        else:
            print("   Checkpointing disabled")
            mpc_results = simulator.run_full_simulation(
                initial_soc_fraction=INITIAL_SOC_FRACTION,
                # max_iterations=MAX_ITERATIONS
            )

        print(f"\n[OK] MPC Simulation Complete!")

    simulation_time = time.time() - simulation_start

    # Display results summary
    print("\n" + "=" * 80)
    print("[RESULTS] MPC RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Revenue:          €{mpc_results['total_revenue']:,.2f}")
    print(f"Total Degradation Cost: €{mpc_results['total_degradation_cost']:,.2f}")
    print(f"Net Profit:             €{mpc_results['net_profit']:,.2f}")
    print()
    print(f"Final SOC:              {mpc_results['final_soc']:.2f} kWh")
    print(f"Number of Iterations:   {len(mpc_results['iteration_results'])}")
    print(f"Simulation Time:        {simulation_time:.2f}s ({simulation_time/60:.2f} min)")
    print("=" * 80)

# %%
# ================================================================================
# [SECTION 4] TRANSFORM & SAVE RESULTS
# ================================================================================

if not LOAD_FROM_SAVED:
    # ============================================================================
    # Transform MPC Results for Visualization
    # ============================================================================

    print("\n" + "=" * 80)
    print("[TRANSFORM] TRANSFORMING RESULTS FOR VISUALIZATION")
    print("=" * 80)

    # Extract Complete bids DataFrame
    total_bids_df = mpc_results['total_bids_df']
    print(f"Complete bids DataFrame: {total_bids_df.shape[0]} timesteps")

    # Transform to visualization format
    viz_df = transform_mpc_results_for_viz(
        total_bids_df,
        country_data_slice,
        battery_capacity_kwh=4472.0
    )

    print(f"[OK] Transformed to visualization format")
    print(f"   Columns: {len(viz_df.columns)}")
    print(f"   Rows:    {len(viz_df)}")

    # Extract iteration summary
    iteration_summary = extract_iteration_summary(mpc_results, include_soc_trajectory=True)
    print(f"\n[OK] Extracted iteration summary: {len(iteration_summary)} iterations")
else:
    print("\n[INFO] Skipping transform - using loaded visualization data")

#%%
# ============================================================================
# Prepare Summary Metrics
# ============================================================================

if not LOAD_FROM_SAVED:
    # Build summary metrics dictionary
    summary_metrics = {
        'model': 'Model_III_MPC',
        'country': TEST_COUNTRY,
        'test_duration_days': TEST_DURATION_DAYS,
        'alpha': used_alpha,
        'c_rate': TEST_C_RATE,

        # MPC settings
        'mpc_horizon_hours': HORIZON_HOURS,
        'mpc_execution_hours': EXECUTION_HOURS,
        'mpc_initial_soc_fraction': INITIAL_SOC_FRACTION,
        'mpc_iterations': len(mpc_results['iteration_results']),

        # Financial results
        'total_profit_eur': mpc_results['net_profit'],
        'total_revenue_eur': mpc_results['total_revenue'],
        'total_degradation_eur': mpc_results['total_degradation_cost'],

        # Revenue breakdown (if available)
        'revenue_da_eur': mpc_results.get('da_revenue', 0),
        'revenue_afrr_energy_eur': mpc_results.get('afrr_e_revenue', 0),
        'revenue_as_capacity_eur': mpc_results.get('as_revenue', 0),

        # Degradation breakdown (if available)
        'degradation_cyclic_eur': mpc_results.get('cyclic_cost', 0),
        'degradation_calendar_eur': mpc_results.get('calendar_cost', 0),

        # SOC metrics
        'initial_soc_kwh': mpc_results.get('soc_trajectory', [INITIAL_SOC_FRACTION * 4472])[0],
        'final_soc_kwh': mpc_results.get('final_soc', INITIAL_SOC_FRACTION * 4472),

        # Timing
        'simulation_time_sec': simulation_time,

        # Solver settings
        'solver': DEFAULT_SOLVER,
        'solver_time_limit_sec': DEFAULT_SOLVER_TIME_LIMIT,

        # Data source
        'data_source': DATA_SOURCE
    }

    # Add meta-optimizer results if applicable
    if ENABLE_META_OPTIMIZER and ALPHA_MODE == 'sweep':
        # Check if meta-optimizer variables exist (defensive programming)
        required_vars = ['best_alpha', 'best_roi', 'all_results', 'alpha_values']
        if all(var in locals() for var in required_vars):
            summary_metrics['meta_optimizer'] = {
                'enabled': True,
                'best_alpha': best_alpha,
                'best_roi': best_roi,
                'alpha_range': {
                    'min': ALPHA_SWEEP_RANGE['min'],
                    'max': ALPHA_SWEEP_RANGE['max'],
                    'step': ALPHA_SWEEP_RANGE['step']
                },
                'all_alphas': {str(float(a)): float(all_results[a]['roi']) for a in alpha_values}
            }
        else:
            missing = [v for v in required_vars if v not in locals()]
            print(f"[WARNING] Meta-optimizer enabled but missing variables: {missing} - skipping meta-optimizer summary")

    print("[OK] Summary metrics prepared")
    print(f"\nTotal Revenue: €{summary_metrics['total_revenue_eur']:,.2f}")
    print(f"Total Profit:  €{summary_metrics['total_profit_eur']:,.2f}")
else:
    print("[INFO] Using loaded summary metrics")

# %%
# ============================================================================
# Save Results to Disk
# ============================================================================

if SAVE_RESULTS and not LOAD_FROM_SAVED:
    # Generate run name
    if AUTO_GENERATE_RUN_NAME:
        run_name = f"mpc_{TEST_COUNTRY}_{TEST_DURATION_DAYS}d_alpha{used_alpha}"
        if ENABLE_META_OPTIMIZER:
            run_name += "_meta"
    else:
        run_name = CUSTOM_RUN_NAME if CUSTOM_RUN_NAME else "mpc_test"

    # Save using results_exporter
    output_directory = save_optimization_results(
        viz_df,
        summary_metrics,
        run_name,
        base_output_dir=str(project_root / BASE_OUTPUT_DIR)
    )

    # Also save iteration summary
    iteration_csv_path = output_directory / "iteration_summary.csv"
    iteration_summary.to_csv(iteration_csv_path, index=False)
    print(f"   [DATA] iteration_summary.csv")

    print("\n" + "=" * 80)
    print("[SAVE] RESULTS SAVED SUCCESSFULLY")
    print("=" * 80)
    print(f"[OUTPUT] Output directory: {output_directory}")
    print(f"   [DATA] solution_timeseries.csv")
    print(f"   [DATA] iteration_summary.csv")
    print(f"   [DATA] performance_summary.json")
    print(f"   [DATA] plots/ (subdirectory created)")
    print("=" * 80)
elif LOAD_FROM_SAVED:
    print("\n" + "=" * 80)
    print("[INFO] Using loaded results directory for plots")
    print("=" * 80)
    print(f"[OUTPUT] Results directory: {output_directory}")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("[WARNING] RESULTS NOT SAVED (SAVE_RESULTS = False)")
    print("=" * 80)
    output_directory = Path(".")

# %%
# ================================================================================
# [SECTION 5] STANDARD VALIDATION PLOTS
# ================================================================================

if ENABLE_STANDARD_PLOTS:
    # Define plots directory
    plots_dir = output_directory / "plots" if SAVE_RESULTS else Path(".")
    title_suffix = f"MPC {TEST_COUNTRY} - {TEST_DURATION_DAYS}d - {HORIZON_HOURS}h/{EXECUTION_HOURS}h"
    
    print("\n" + "=" * 80)
    print("[PLOTS] GENERATING STANDARD MARKET PLOTS")
    print("=" * 80)
    
    # Plot 1: Day-Ahead Market
    print("\n[1/4] Day-Ahead Market...")
    fig_da = plot_da_market_price_bid(viz_df, title_suffix=title_suffix, use_timestamp=True)
    if SAVE_RESULTS:
        fig_da.write_html(str(plots_dir / f"da_market_price_bid.{SAVE_FORMAT}"))
    fig_da.show()
    print("   [OK] Saved: da_market_price_bid.html")

    # Plot 2: aFRR Energy Market
    print("\n[2/4] aFRR Energy Market...")
    fig_afrr_e = plot_afrr_energy_market_price_bid(viz_df, title_suffix=title_suffix, use_timestamp=True)
    if SAVE_RESULTS:
        fig_afrr_e.write_html(str(plots_dir / f"afrr_energy_market_price_bid.{SAVE_FORMAT}"))
    fig_afrr_e.show()
    print("   [OK] Saved: afrr_energy_market_price_bid.html")

    # Plot 3: Capacity Markets
    print("\n[3/4] Capacity Markets...")
    fig_cap = plot_capacity_markets_price_bid(viz_df, title_suffix=title_suffix, use_timestamp=True)
    if SAVE_RESULTS:
        fig_cap.write_html(str(plots_dir / f"capacity_markets_price_bid.{SAVE_FORMAT}"))
    fig_cap.show()
    print("   [OK] Saved: capacity_markets_price_bid.html")

    # Plot 4: SOC & Power Bids
    print("\n[4/4] SOC & Power Bids...")
    fig_soc = plot_soc_and_power_bids(viz_df, title_suffix=title_suffix, use_timestamp=True)
    if SAVE_RESULTS:
        fig_soc.write_html(str(plots_dir / f"soc_and_power_bids.{SAVE_FORMAT}"))
    fig_soc.show()
    print("   [OK] Saved: soc_and_power_bids.html")

    print("\n" + "=" * 80)
    print("[OK] All standard market plots generated!")
    print("=" * 80)
else:
    print("\n[WARNING] Standard plots disabled (ENABLE_STANDARD_PLOTS = False)")

# %%
viz_df['e_soc']

# %%
# ================================================================================
# [SECTION 6] MPC-SPECIFIC ANALYSIS PLOTS
# ================================================================================

# Path to saved MPC results directory (only used if LOAD_FROM_SAVED = True)
# saved_stamp = "20251116_013155_mpc_ch_3d_alpha1.0"  # e.g., "20251115_141102"
SAVED_RESULTS_DIR = f"validation_results/mpc_validation/20251116_021804_mpc_ch_3d_alpha1.0"

LOAD_FROM_SAVED = True

if LOAD_FROM_SAVED:
    # ============================================================================
    # Load Previously Saved MPC Results
    # ============================================================================

    print("\n" + "=" * 80)
    print("[LOAD] LOADING SAVED MPC RESULTS")
    print("=" * 80)

    # Construct absolute path from project root
    saved_dir = project_root / SAVED_RESULTS_DIR

    if not saved_dir.exists():
        raise FileNotFoundError(f"Saved results directory not found: {saved_dir}")

    print(f"\nLoading from: {saved_dir}")

    # Load performance summary
    perf_file = saved_dir / "performance_summary.json"
    with open(perf_file, 'r') as f:
        perf_summary = json.load(f)
    print(f"  [OK] Loaded performance_summary.json")

    # Load iteration summary
    iter_file = saved_dir / "iteration_summary.csv"
    iteration_summary = pd.read_csv(iter_file)
    print(f"  [OK] Loaded iteration_summary.csv ({len(iteration_summary)} iterations)")

    # Load solution timeseries
    sol_file = saved_dir / "solution_timeseries.csv"
    viz_df = pd.read_csv(sol_file, index_col=0)
    if 'timestamp' in viz_df.columns:
        viz_df['timestamp'] = pd.to_datetime(viz_df['timestamp'])
    print(f"  [OK] Loaded solution_timeseries.csv ({len(viz_df)} timesteps)")

    # Extract parameters from saved results
    TEST_COUNTRY = perf_summary['country']
    TEST_DURATION_DAYS = perf_summary['test_duration_days']
    TEST_C_RATE = perf_summary['c_rate']
    used_alpha = perf_summary['alpha']
    HORIZON_HOURS = perf_summary['mpc_horizon_hours']
    EXECUTION_HOURS = perf_summary['mpc_execution_hours']
    simulation_time = perf_summary['simulation_time_sec']

    # Reconstruct SOC trajectory from iteration boundaries
    # Extract SOC values at end of each iteration from solution timeseries
    soc_trajectory = [perf_summary['initial_soc_kwh']]
    for _, iter_row in iteration_summary.iterrows():
        end_idx = int(iter_row['end_timestep'])  # Convert to Python int for .iloc
        if end_idx < len(viz_df):
            soc_trajectory.append(viz_df['soc_kwh'].iloc[end_idx])
    # Ensure final SOC is included
    if len(soc_trajectory) <= len(iteration_summary):
        soc_trajectory.append(perf_summary['final_soc_kwh'])

    # Reconstruct mpc_results for plotting functions
    mpc_results = {
        'total_revenue': perf_summary['total_revenue_eur'],
        'total_degradation_cost': perf_summary['total_degradation_eur'],
        'net_profit': perf_summary['total_profit_eur'],
        'final_soc': perf_summary['final_soc_kwh'],
        'soc_trajectory': soc_trajectory,
        'soc_15min': viz_df['soc_kwh'].tolist(),
        'iteration_results': iteration_summary.to_dict('records'),
        'da_revenue': perf_summary.get('revenue_da_eur', 0),
        'afrr_e_revenue': perf_summary.get('revenue_afrr_energy_eur', 0),
        'as_revenue': perf_summary.get('revenue_as_capacity_eur', 0),
        'cyclic_cost': perf_summary.get('degradation_cyclic_eur', 0),
        'calendar_cost': perf_summary.get('degradation_calendar_eur', 0),
    }

    # Build summary metrics
    summary_metrics = perf_summary.copy()

    # Set output directory to saved location
    output_directory = saved_dir
    SAVE_RESULTS = False  # Don't overwrite existing results

    # Set visualization settings for loaded results (if not already defined)
    if 'ENABLE_MPC_PLOTS' not in dir():
        ENABLE_MPC_PLOTS = True  # Enable plotting by default
    if 'MPC_PLOT_OPTIONS' not in dir():
        MPC_PLOT_OPTIONS = {
            'iteration_boundaries': True,
            'financial_breakdown': True,
            'soc_evolution': True
        }
    if 'SAVE_FORMAT' not in dir():
        SAVE_FORMAT = 'html'


# %%
mpc_results
# %%

if ENABLE_MPC_PLOTS:
    plots_dir = output_directory / "plots" if SAVE_RESULTS else Path(".")
    title_suffix = f"MPC {TEST_COUNTRY} - {TEST_DURATION_DAYS}d"
    
    print("\n" + "=" * 80)
    print("[MPC PLOTS] GENERATING MPC-SPECIFIC ANALYSIS PLOTS")
    print("=" * 80)
    
    # Plot 1: Iteration Boundaries
    if MPC_PLOT_OPTIONS['iteration_boundaries']:
        print("\n[1/3] Iteration Boundaries...")
        fig_boundaries = plot_iteration_boundaries(
            mpc_results,
            execution_hours=EXECUTION_HOURS,
            title_suffix=title_suffix,
            show_horizons=False
        )
        if SAVE_RESULTS:
            fig_boundaries.write_html(str(plots_dir / f"mpc_iteration_boundaries.{SAVE_FORMAT}"))
        fig_boundaries.show()
        print("   [OK] Saved: mpc_iteration_boundaries.html")

    # Plot 2: Iteration Performance
    if MPC_PLOT_OPTIONS['iteration_performance']:
        print("\n[2/3] Iteration Performance...")
        fig_performance = plot_iteration_performance(
            mpc_results,
            title_suffix=title_suffix,
            show_cumulative=True
        )
        if SAVE_RESULTS:
            fig_performance.write_html(str(plots_dir / f"mpc_iteration_performance.{SAVE_FORMAT}"))
        fig_performance.show()
        print("   [OK] Saved: mpc_iteration_performance.html")

    # Plot 3: State Continuity
    if MPC_PLOT_OPTIONS['state_continuity']:
        print("\n[3/3] State Continuity Check...")
        fig_continuity = plot_state_continuity(
            mpc_results,
            title_suffix=title_suffix,
            tolerance_pct=0.1
        )
        if SAVE_RESULTS:
            fig_continuity.write_html(str(plots_dir / f"mpc_state_continuity.{SAVE_FORMAT}"))
        fig_continuity.show()
        print("   [OK] Saved: mpc_state_continuity.html")

    print("\n" + "=" * 80)
    print("[OK] All MPC analysis plots generated!")
    print("=" * 80)
else:
    print("\n[WARNING] MPC plots disabled (ENABLE_MPC_PLOTS = False)")

# %%
# ================================================================================
# [COMPLETE] NOTEBOOK COMPLETE!
# ================================================================================
#
# ### What was accomplished:
# 1. [OK] Loaded all configuration files (MPC, test, solver, aging, aFRR)
# 2. [OK] Loaded market data (preprocessed or Excel)
# 3. [OK] Ran MPC rolling horizon simulation (with optional meta-optimizer)
# 4. [OK] Transformed results to visualization format
# 5. [OK] Saved results using `results_exporter`
# 6. [OK] Generated standard market participation plots
# 7. [OK] Generated MPC-specific analysis plots
#
# ### Output location:
# All results are saved in the timestamped directory under `validation_results/mpc_validation/`
#
# ### Next steps:
# - Modify parameters in `data/p2_config/mpc_test_config.json` and re-run
# - Enable meta-optimizer to find optimal alpha
# - Test different countries or time horizons
# - Compare MPC results with single-pass optimization
# - Analyze iteration-level performance for optimization opportunities
#
# ### Configuration file reference:
# - `mpc_config.json`: MPC horizon/execution settings
# - `mpc_test_config.json`: Test scenario parameters (country, duration, alpha, visualization options)
# - `solver_config.json`: Solver timeouts and tolerances
# - `aging_config.json`: Degradation model parameters
# - `afrr_ev_weights_config.json`: aFRR activation probabilities
