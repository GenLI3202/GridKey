# -*- coding: utf-8 -*-
"""
Phase 2 Alpha Meta-Optimization - Interactive Analysis Script

This script performs alpha parameter meta-optimization for the MPC-based BESS optimizer.
It runs parallel simulations across multiple alpha values and provides comprehensive
comparative analysis through interactive visualizations.

Key Features:
- Parallel alpha sweep execution (configurable workers)
- Test mode (14-day) vs Full mode (365-day)
- Pareto frontier analysis (Profit vs Aging Cost)
- SOC sensitivity analysis
- Revenue/cost breakdown comparison
- Interactive plot generation

Test Configuration:
- Country: CZ
- C-rate: 0.5
- Rolling horizon: 36h planning / 24h execution
- Alpha range: [0.5, 1.5] with 0.1 step (11 values)
- REQUIRE_SEQUENTIAL: False
- EPSILON: 0

This version has cell markers (# %%) for block-by-block execution in VS Code.
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
import warnings

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Data processing
import pandas as pd
import numpy as np

# Parallel processing
from joblib import Parallel, delayed
from tqdm import tqdm

# Plotly for visualizations
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optimization models
from src.core.optimizer import BESSOptimizerModelIII

# MPC simulation
from src.mpc.mpc_simulator import MPCSimulator
from src.mpc.transform_mpc_results import (
    transform_mpc_results_for_viz,
    extract_iteration_summary
)

# Data loading
from src.data.load_process_market_data import load_preprocessed_country_data

# Visualization
from src.visualization.config import WATERFALL_COLORS, MCKINSEY_COLORS, apply_mckinsey_style

# Results export
from src.validation.results_exporter import save_optimization_results

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Suppress Gurobi logging errors in threading mode (harmless)
import logging
logging.getLogger('gurobipy').setLevel(logging.CRITICAL)

print("=" * 80)
print("ALPHA META-OPTIMIZATION SCRIPT")
print("=" * 80)
print(f"[OK] All imports successful!")
print(f"Project root: {project_root}")
print()

# %%
# ================================================================================
# [SECTION 2] CONFIGURATION
# ================================================================================

print("=" * 80)
print("[SECTION 2] CONFIGURATION")
print("=" * 80)

# ============================================================================
# Meta-Optimization Test Configuration
# ============================================================================

# Test Mode: Choose between quick test and full production run
TEST_MODE = True  # Set to False for full 365-day run

if TEST_MODE:
    TEST_DAYS = 5  # Quick test: 5 days
    SIMULATION_DESCRIPTION = f"{TEST_DAYS}-day test"
else:
    TEST_DAYS = 365  # Full year
    SIMULATION_DESCRIPTION = "Full year (365 days)"

print(f"\n[CONFIG] Test Mode: {TEST_MODE}")
print(f"[CONFIG] Simulation Duration: {SIMULATION_DESCRIPTION}")

# ============================================================================
# Fixed Scenario Parameters (as requested)
# ============================================================================

COUNTRY = "CZ"
C_RATE = 0.5
REQUIRE_SEQUENTIAL = False
EPSILON = 0.0

print(f"\n[CONFIG] Fixed Parameters:")
print(f"  - Country: {COUNTRY}")
print(f"  - C-rate: {C_RATE}")
print(f"  - REQUIRE_SEQUENTIAL: {REQUIRE_SEQUENTIAL}")
print(f"  - EPSILON: {EPSILON}")

# ============================================================================
# Alpha Sweep Configuration
# ============================================================================

ALPHA_MIN = 0.5
ALPHA_MAX = 1.5
ALPHA_STEP = 0.1

# Generate alpha values
alpha_values = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP/2, ALPHA_STEP)
alpha_values = np.round(alpha_values, 2)  # Avoid floating point issues

print(f"\n[CONFIG] Alpha Sweep:")
print(f"  - Range: [{ALPHA_MIN}, {ALPHA_MAX}]")
print(f"  - Step: {ALPHA_STEP}")
print(f"  - Values: {list(alpha_values)}")
print(f"  - Total simulations: {len(alpha_values)}")

# ============================================================================
# Parallel Execution Configuration
# ============================================================================

N_JOBS = 8  # Number of parallel workers (adjust based on your CPU cores)

print(f"\n[CONFIG] Parallel Execution:")
print(f"  - Workers: {N_JOBS}")
print(f"  - Estimated speedup: ~{N_JOBS}x")

# ============================================================================
# MPC Configuration
# ============================================================================

# Load MPC configuration from unified YAML config
from src.utils.config_loader import ConfigLoader

base_mpc_config = ConfigLoader.get_mpc_config()
print(f"\n[OK] Loaded base MPC config from config/Config.yml")

# Extract MPC horizon settings
planning_horizon_hours = base_mpc_config.get("planning_horizon_hours", 36)
execution_horizon_hours = base_mpc_config.get("execution_horizon_hours", 24)

print(f"  - Planning horizon: {planning_horizon_hours}h")
print(f"  - Execution horizon: {execution_horizon_hours}h")

# ============================================================================
# Output Configuration
# ============================================================================

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
duration_str = f"{TEST_DAYS}d" if TEST_MODE else "365d"
output_dir_name = f"alpha_meta_{COUNTRY}_{C_RATE}C_{duration_str}_{timestamp}"
output_base_dir = project_root / "validation_results" / output_dir_name

print(f"\n[CONFIG] Output:")
print(f"  - Base directory: {output_base_dir}")

# ============================================================================
# Investment & ROI Parameters (for future NPV/ROI calculations)
# ============================================================================

WACC = 0.08  # Weighted Average Cost of Capital
INFLATION_RATE = 0.02
PROJECT_LIFETIME_YEARS = 10

print(f"\n[CONFIG] Financial Parameters (for NPV/ROI):")
print(f"  - WACC: {WACC*100}%")
print(f"  - Inflation: {INFLATION_RATE*100}%")
print(f"  - Project lifetime: {PROJECT_LIFETIME_YEARS} years")

print()

# %%
# ================================================================================
# [SECTION 3] PARALLEL ALPHA SWEEP EXECUTION
# ================================================================================

print("=" * 80)
print("[SECTION 3] PARALLEL ALPHA SWEEP EXECUTION")
print("=" * 80)

def run_single_alpha_simulation(alpha, config):
    """
    Run MPC simulation for a single alpha value.

    Parameters
    ----------
    alpha : float
        Alpha parameter (degradation cost weight)
    config : dict
        Configuration dictionary containing all necessary parameters

    Returns
    -------
    dict
        Results dictionary with performance metrics and paths
    """
    # Write progress to file instead of console (to avoid blocking in multiprocessing)
    # Console logging in multiprocessing creates too much overhead

    try:
        # Extract configuration
        country = config['country']
        c_rate = config['c_rate']
        test_days = config['test_days']
        output_base = Path(config['output_base_dir'])
        base_mpc_config = config['base_mpc_config']
        require_sequential = config['require_sequential']
        epsilon = config['epsilon']

        # Create alpha-specific output directory
        alpha_dir = output_base / f"alpha_{alpha:.1f}"
        alpha_dir.mkdir(parents=True, exist_ok=True)

        # Load market data (preprocessed for speed)
        # Use project_root from config to ensure correct path
        project_root = Path(config.get('project_root', Path.cwd()))
        preprocessed_data_dir = project_root / "data" / "parquet" / "preprocessed"
        market_data = load_preprocessed_country_data(country, data_dir=preprocessed_data_dir)

        # Limit data to test period
        start_date = market_data['timestamp'].min()
        end_date = start_date + pd.Timedelta(days=test_days)
        market_data = market_data[market_data['timestamp'] < end_date].copy()

        # Create optimizer instance with alpha parameter
        optimizer = BESSOptimizerModelIII(alpha=alpha)

        # Update LIFO settings
        optimizer.REQUIRE_SEQUENTIAL_FILL = require_sequential
        optimizer.EPSILON = epsilon

        # Extract MPC parameters
        mpc_params = base_mpc_config.get('mpc_parameters', {})
        horizon_hours = mpc_params.get('horizon_hours', 36)
        execution_hours = mpc_params.get('execution_hours', 24)

        # Create MPC simulator
        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=market_data,
            horizon_hours=horizon_hours,
            execution_hours=execution_hours,
            c_rate=c_rate,
            validate_constraints=False  # Disable for speed
        )

        # Run simulation
        start_time = time.time()
        mpc_results = simulator.run_full_simulation()
        runtime = time.time() - start_time

        # Check if simulation succeeded
        if mpc_results is None:
            raise ValueError("MPC simulation returned None - simulation failed")

        # Debug: Check what type mpc_results is
        if not isinstance(mpc_results, dict):
            raise TypeError(f"MPC results is {type(mpc_results)}, expected dict. Value: {str(mpc_results)[:200]}")

        # MPC simulator returns data at top level, not in 'performance' dict
        if 'total_revenue' not in mpc_results:
            raise ValueError(f"MPC simulation completed but no financial data returned. Keys: {list(mpc_results.keys())}")

        # Save results
        results_summary = {
            'alpha': alpha,
            'country': country,
            'c_rate': c_rate,
            'test_days': test_days,
            'runtime_seconds': runtime,
            'output_dir': str(alpha_dir),
            'status': 'success'  # Mark as successful
        }

        # Extract performance metrics from MPC results (top-level keys)
        total_revenue = mpc_results.get('total_revenue', 0)
        total_degradation_cost = mpc_results.get('total_degradation_cost', 0)
        net_profit = mpc_results.get('net_profit', 0)

        results_summary.update({
            'total_profit_eur': net_profit,
            'total_revenue_eur': total_revenue,
            'total_cost_eur': total_degradation_cost,
            'revenue_da_eur': mpc_results.get('da_revenue', 0),
            'revenue_fcr_eur': 0,  # FCR is part of as_revenue
            'revenue_afrr_capacity_eur': 0,  # aFRR capacity is part of as_revenue
            'revenue_afrr_energy_eur': mpc_results.get('afrr_e_revenue', 0),
            'revenue_as_eur': mpc_results.get('as_revenue', 0),  # Combined AS revenue
            'degradation_cyclic_eur': mpc_results.get('cyclic_cost', 0),
            'degradation_calendar_eur': mpc_results.get('calendar_cost', 0),
            'total_aging_cost_eur': total_degradation_cost,
            'num_iterations': mpc_results.get('summary', {}).get('iterations', 0),
            'solver_status': 'optimal'  # MPC only returns if successful
        })

        # Calculate SOC statistics from 15-min SOC array
        if 'soc_15min' in mpc_results and mpc_results['soc_15min']:
            soc_values = mpc_results['soc_15min']
            # Convert to numpy array if it's a list
            if isinstance(soc_values, list):
                soc_values = np.array(soc_values)
            results_summary.update({
                'soc_avg_kwh': float(np.mean(soc_values)),
                'soc_min_kwh': float(np.min(soc_values)),
                'soc_max_kwh': float(np.max(soc_values)),
                'soc_std_kwh': float(np.std(soc_values))
            })

        # Save detailed results
        # Removed print statement - causes Unicode encoding issues on Windows console

        # Save MPC-specific results
        # Convert MPC results to format expected by save_optimization_results
        performance_summary = {
            'total_profit_eur': net_profit,
            'total_revenue_eur': total_revenue,
            'total_cost_eur': total_degradation_cost,
            'revenue_da_eur': mpc_results.get('da_revenue', 0),
            'revenue_afrr_energy_eur': mpc_results.get('afrr_e_revenue', 0),
            'revenue_as_eur': mpc_results.get('as_revenue', 0),
            'degradation_cyclic_eur': mpc_results.get('cyclic_cost', 0),
            'degradation_calendar_eur': mpc_results.get('calendar_cost', 0),
            'num_iterations': mpc_results.get('summary', {}).get('iterations', 0),
            'alpha': alpha,
            'c_rate': c_rate
        }

        # Save total_bids_df (annual bids DataFrame)
        if 'total_bids_df' in mpc_results:
            bids_csv_path = alpha_dir / "solution_timeseries.csv"
            mpc_results['total_bids_df'].to_csv(bids_csv_path, index=False)
            # Removed print statement - causes Unicode encoding issues on Windows console

        # Save performance summary as JSON (ensure UTF-8 encoding for Unicode chars)
        perf_json_path = alpha_dir / "performance_summary.json"
        with open(perf_json_path, 'w', encoding='utf-8') as f:
            json.dump(performance_summary, f, indent=2, ensure_ascii=False)

        # Save iteration summary if available
        if 'iteration_results' in mpc_results:
            try:
                iter_df = extract_iteration_summary(mpc_results['iteration_results'])
                iter_csv_path = alpha_dir / "iteration_summary.csv"
                iter_df.to_csv(iter_csv_path, index=False)
                # Removed print statement - causes Unicode encoding issues on Windows console
            except Exception as iter_err:
                # Don't fail entire simulation if iteration summary fails
                print(f"Warning: Could not save iteration summary for alpha {alpha}: {iter_err}")

        # Print success message (removed Unicode chars for Windows console compatibility)
        profit = results_summary.get('total_profit_eur', 0)
        aging_cost = results_summary.get('total_aging_cost_eur', 0)
        # Removed print statements - causes Unicode encoding issues on Windows console

        return results_summary

    except Exception as e:
        # Print detailed error for debugging
        import traceback
        print(f"\n[ERROR] Alpha {alpha:.1f} failed:")
        print(f"  Error: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()

        # Return error information with all required fields
        return {
            'alpha': alpha,
            'status': 'error',
            'error': str(e),
            'runtime_seconds': 0,
            'total_profit_eur': np.nan,
            'total_revenue_eur': np.nan,
            'total_cost_eur': np.nan,
            'revenue_da_eur': np.nan,
            'revenue_fcr_eur': np.nan,
            'revenue_afrr_capacity_eur': np.nan,
            'revenue_afrr_energy_eur': np.nan,
            'degradation_cyclic_eur': np.nan,
            'degradation_calendar_eur': np.nan,
            'total_aging_cost_eur': np.nan,
            'num_iterations': 0,
            'solver_status': 'error',
            'soc_avg_kwh': np.nan,
            'soc_min_kwh': np.nan,
            'soc_max_kwh': np.nan,
            'soc_std_kwh': np.nan,
            'country': config.get('country', 'unknown'),
            'c_rate': config.get('c_rate', 0),
            'test_days': config.get('test_days', 0),
            'output_dir': str(Path(config.get('output_base_dir', '')) / f"alpha_{alpha:.1f}")
        }

# %%
# ============================================================================
# Execute Parallel Alpha Sweep
# ============================================================================

# Prepare configuration for parallel execution
sweep_config = {
    'country': COUNTRY,
    'c_rate': C_RATE,
    'test_days': TEST_DAYS,
    'output_base_dir': str(output_base_dir),
    'base_mpc_config': base_mpc_config,
    'require_sequential': REQUIRE_SEQUENTIAL,
    'epsilon': EPSILON,
    'project_root': str(project_root)  # Pass project root for data loading
}

# Save sweep configuration
output_base_dir.mkdir(parents=True, exist_ok=True)
config_snapshot_path = output_base_dir / f"{TEST_DAYS}d_sweep_config.json"
with open(config_snapshot_path, 'w', encoding='utf-8') as f:
    json.dump({
        'alpha_values': list(alpha_values),
        'test_mode': TEST_MODE,
        'test_days': TEST_DAYS,
        'country': COUNTRY,
        'c_rate': C_RATE,
        'require_sequential': REQUIRE_SEQUENTIAL,
        'epsilon': EPSILON,
        'planning_horizon_hours': planning_horizon_hours,
        'execution_horizon_hours': execution_horizon_hours,
        'n_jobs': N_JOBS,
        'timestamp': timestamp
    }, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Configuration saved to: {config_snapshot_path}")

# Execute parallel sweep
print(f"\n[START] Running alpha sweep with {N_JOBS} parallel workers...")
print(f"Total alphas to test: {len(alpha_values)}")
# print(f"Expected total runtime: ~{len(alpha_values) * 45 / N_JOBS / 60:.1f} minutes")
print(f"\n{'='*80}")
print("DETAILED PROGRESS (each worker will report its status)")
print(f"{'='*80}\n")

sweep_start_time = time.time()

##################################################
# Run simulations in parallel with multiprocessing (faster, less logging noise)
# verbose=10 shows detailed progress with timestamps
print("\n[TIP] Simulations running in background. Watch for completion messages...")
# print("      Each alpha should complete in 5-15 seconds for 2-day test.\n")

results_list = Parallel(n_jobs=N_JOBS, verbose=10, backend='loky')(
    delayed(run_single_alpha_simulation)(alpha, sweep_config)
    for alpha in alpha_values
)

sweep_runtime = time.time() - sweep_start_time
###################################################


print(f"\n[COMPLETE] Alpha sweep finished!")
print(f"  - Total runtime: {sweep_runtime/60:.1f} minutes")
print(f"  - Average per alpha: {sweep_runtime/len(alpha_values):.1f} seconds")
print(f"  - Speedup vs sequential: ~{N_JOBS:.1f}x")

# %%
# ================================================================================
# [SECTION 4] RESULTS AGGREGATION & ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 4] RESULTS AGGREGATION & ANALYSIS")
print("=" * 80)

# ============================================================================
# Build Comparison DataFrame
# ============================================================================

print("\n[START] Aggregating results...")

# Convert results to DataFrame
results_df = pd.DataFrame(results_list)

# Debug: Check what columns we got
print(f"\n[DEBUG] Results columns: {list(results_df.columns)}")
print(f"[DEBUG] Number of rows: {len(results_df)}")

# Check for errors
if 'status' in results_df.columns:
    n_errors = (results_df['status'] == 'error').sum()
    if n_errors > 0:
        print(f"\n[WARNING] {n_errors}/{len(results_df)} simulations failed!")
        print("\nFailed alphas:")
        error_rows = results_df[results_df['status'] == 'error']
        for _, row in error_rows.iterrows():
            print(f"  Alpha {row['alpha']:.1f}: {row.get('error', 'Unknown error')}")

# Sort by alpha
results_df = results_df.sort_values('alpha').reset_index(drop=True)

# Calculate additional metrics
results_df['net_profit_eur'] = results_df['total_profit_eur']  # Already net profit
results_df['profit_per_day'] = results_df['net_profit_eur'] / TEST_DAYS

# Annualize metrics if in test mode
if TEST_MODE:
    annualization_factor = 365 / TEST_DAYS
    results_df['annual_profit_estimate'] = results_df['net_profit_eur'] * annualization_factor
    results_df['annual_aging_cost_estimate'] = results_df['total_aging_cost_eur'] * annualization_factor
else:
    results_df['annual_profit_estimate'] = results_df['net_profit_eur']
    results_df['annual_aging_cost_estimate'] = results_df['total_aging_cost_eur']

# Calculate NPV (simplified: annualize profit and discount over project lifetime)
discount_rates = [(1 / (1 + WACC) ** year) for year in range(1, PROJECT_LIFETIME_YEARS + 1)]
npv_multiplier = sum(discount_rates)

results_df['npv_eur'] = results_df['annual_profit_estimate'] * npv_multiplier

# Calculate simple ROI (assuming initial investment, to be refined)
# For now, use profit-to-aging-cost ratio as a proxy
results_df['roi_proxy'] = results_df['net_profit_eur'] / results_df['total_aging_cost_eur'].replace(0, 1)

# Save comparison results
comparison_csv_path = output_base_dir / "comparison_results.csv"
results_df.to_csv(comparison_csv_path, index=False)

print(f"[OK] Comparison results saved to: {comparison_csv_path}")

# ============================================================================
# Display Summary Table
# ============================================================================

print("\n" + "=" * 80)
print("ALPHA SWEEP SUMMARY")
print("=" * 80)

# Create summary display
summary_cols = [
    'alpha',
    'net_profit_eur',
    'total_aging_cost_eur',
    'annual_profit_estimate',
    'npv_eur',
    'soc_avg_kwh',
    'runtime_seconds'
]

display_df = results_df[summary_cols].copy()
display_df.columns = [
    'Alpha',
    'Profit (EUR)',
    'Aging Cost (EUR)',
    'Annual Profit Est.',
    'NPV (EUR)',
    'Avg SOC (kWh)',
    'Runtime (s)'
]

print(display_df.to_string(index=False))
print()

# ============================================================================
# Identify Optimal Alpha Values
# ============================================================================

# Check if we have any successful results
successful_results = results_df[results_df['status'] == 'success']

if len(successful_results) > 0:
    # Find best alpha by different metrics (only among successful runs)
    best_profit_idx = successful_results['net_profit_eur'].idxmax()
    best_npv_idx = successful_results['npv_eur'].idxmax()
    best_roi_idx = successful_results['roi_proxy'].idxmax()

    print("=" * 80)
    print("OPTIMAL ALPHA CANDIDATES")
    print("=" * 80)
    print(f"\nBest by Net Profit: alpha = {results_df.loc[best_profit_idx, 'alpha']:.1f}")
    print(f"  - Profit: €{results_df.loc[best_profit_idx, 'net_profit_eur']:,.0f}")
    print(f"  - Aging Cost: €{results_df.loc[best_profit_idx, 'total_aging_cost_eur']:,.0f}")

    print(f"\nBest by NPV: alpha = {results_df.loc[best_npv_idx, 'alpha']:.1f}")
    print(f"  - NPV: €{results_df.loc[best_npv_idx, 'npv_eur']:,.0f}")
    print(f"  - Annual Profit Est.: €{results_df.loc[best_npv_idx, 'annual_profit_estimate']:,.0f}")

    print(f"\nBest by ROI Proxy: alpha = {results_df.loc[best_roi_idx, 'alpha']:.1f}")
    print(f"  - ROI Proxy: {results_df.loc[best_roi_idx, 'roi_proxy']:.2f}")
    print()
else:
    print("=" * 80)
    print("OPTIMAL ALPHA CANDIDATES")
    print("=" * 80)
    print("\n[ERROR] No successful simulations! Cannot identify optimal alpha.")
    print("Please fix the errors above and re-run.")
    print()

# %%
# ================================================================================
# [SECTION 5] PLOTTING FUNCTIONS
# ================================================================================

print("=" * 80)
print("[SECTION 5] PLOTTING FUNCTIONS LOADED")
print("=" * 80)
print("Ready to generate plots interactively.")
print()

# ============================================================================
# Function 1: Pareto Front (Aging Cost vs Profit)
# ============================================================================

def plot_pareto_front(results_df, output_dir=None):
    """
    Create Pareto frontier plot showing trade-off between aging cost and profit.

    Parameters
    ----------
    results_df : pd.DataFrame
        Comparison results DataFrame
    output_dir : Path, optional
        Directory to save plot

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    print("\n[PLOTTING] Generating Pareto Front...")

    # Create figure
    fig = go.Figure()

    # Add scatter trace with alpha as color
    fig.add_trace(go.Scatter(
        x=results_df['annual_aging_cost_estimate'],
        y=results_df['annual_profit_estimate'],
        mode='markers+lines+text',
        marker=dict(
            size=12,
            color=results_df['alpha'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Alpha (α)"),
            line=dict(width=1, color='white')
        ),
        line=dict(width=2, color='rgba(100,100,100,0.3)'),
        text=[f"α={a:.1f}" for a in results_df['alpha']],
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate=(
            "<b>Alpha: %{text}</b><br>" +
            "Aging Cost: €%{x:,.0f}<br>" +
            "Annual Profit: €%{y:,.0f}<br>" +
            "<extra></extra>"
        ),
        name='Alpha values'
    ))

    # Highlight optimal points
    best_profit_alpha = results_df.loc[results_df['net_profit_eur'].idxmax()]
    best_npv_alpha = results_df.loc[results_df['npv_eur'].idxmax()]

    # Add best profit marker
    fig.add_trace(go.Scatter(
        x=[best_profit_alpha['annual_aging_cost_estimate']],
        y=[best_profit_alpha['annual_profit_estimate']],
        mode='markers',
        marker=dict(size=18, color='red', symbol='star', line=dict(width=2, color='white')),
        name=f'Best Profit (α={best_profit_alpha["alpha"]:.1f})',
        hovertemplate=(
            f"<b>Best Net Profit</b><br>" +
            f"Alpha: {best_profit_alpha['alpha']:.1f}<br>" +
            f"Aging Cost: €{best_profit_alpha['annual_aging_cost_estimate']:,.0f}<br>" +
            f"Annual Profit: €{best_profit_alpha['annual_profit_estimate']:,.0f}<br>" +
            "<extra></extra>"
        )
    ))

    # Add best NPV marker
    fig.add_trace(go.Scatter(
        x=[best_npv_alpha['annual_aging_cost_estimate']],
        y=[best_npv_alpha['annual_profit_estimate']],
        mode='markers',
        marker=dict(size=18, color='gold', symbol='diamond', line=dict(width=2, color='white')),
        name=f'Best NPV (α={best_npv_alpha["alpha"]:.1f})',
        hovertemplate=(
            f"<b>Best NPV</b><br>" +
            f"Alpha: {best_npv_alpha['alpha']:.1f}<br>" +
            f"Aging Cost: €{best_npv_alpha['annual_aging_cost_estimate']:,.0f}<br>" +
            f"Annual Profit: €{best_npv_alpha['annual_profit_estimate']:,.0f}<br>" +
            f"NPV: €{best_npv_alpha['npv_eur']:,.0f}<br>" +
            "<extra></extra>"
        )
    ))

    # Update layout
    fig.update_layout(
        title=f"Pareto Frontier: Aging Cost vs Profit<br><sub>{COUNTRY}, {C_RATE} C-rate, {SIMULATION_DESCRIPTION}</sub>",
        xaxis_title="Annual Aging Cost (EUR)",
        yaxis_title="Annual Profit (EUR)",
        template='mckinsey',
        height=600,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "pareto_front.html"
        fig.write_html(str(plot_path))
        print(f"  [SAVED] {plot_path}")

    return fig

# ============================================================================
# Function 2: SOC vs Alpha
# ============================================================================

def plot_soc_vs_alpha(results_df, output_dir=None):
    """
    Plot SOC statistics (avg, min, max) vs alpha values.

    Parameters
    ----------
    results_df : pd.DataFrame
        Comparison results DataFrame
    output_dir : Path, optional
        Directory to save plot

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    print("\n[PLOTTING] Generating SOC vs Alpha...")

    # Create figure
    fig = go.Figure()

    # Add average SOC line
    fig.add_trace(go.Scatter(
        x=results_df['alpha'],
        y=results_df['soc_avg_kwh'],
        mode='lines+markers',
        name='Average SOC',
        line=dict(color=MCKINSEY_COLORS['navy'], width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>Alpha: %{x:.1f}</b><br>" +
            "Avg SOC: %{y:.1f} kWh<br>" +
            "<extra></extra>"
        )
    ))

    # Add min/max range as filled area
    fig.add_trace(go.Scatter(
        x=results_df['alpha'],
        y=results_df['soc_max_kwh'],
        mode='lines',
        name='Max SOC',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=results_df['alpha'],
        y=results_df['soc_min_kwh'],
        mode='lines',
        name='SOC Range',
        fill='tonexty',
        fillcolor='rgba(0, 63, 92, 0.2)',
        line=dict(width=0),
        hovertemplate=(
            "<b>Alpha: %{x:.1f}</b><br>" +
            "Min SOC: %{y:.1f} kWh<br>" +
            "<extra></extra>"
        )
    ))

    # Add individual min/max lines (dashed)
    fig.add_trace(go.Scatter(
        x=results_df['alpha'],
        y=results_df['soc_max_kwh'],
        mode='lines',
        name='Max SOC',
        line=dict(color=MCKINSEY_COLORS['teal'], width=1, dash='dash'),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=results_df['alpha'],
        y=results_df['soc_min_kwh'],
        mode='lines',
        name='Min SOC',
        line=dict(color=MCKINSEY_COLORS['negative'], width=1, dash='dash'),
        hoverinfo='skip'
    ))

    # Add battery capacity reference line
    battery_capacity = 4472  # kWh
    fig.add_hline(
        y=battery_capacity,
        line=dict(color='gray', width=1, dash='dot'),
        annotation_text="Battery Capacity (4,472 kWh)",
        annotation_position="right"
    )

    # Update layout
    fig.update_layout(
        title=f"SOC Sensitivity to Alpha<br><sub>{COUNTRY}, {C_RATE} C-rate, {SIMULATION_DESCRIPTION}</sub>",
        xaxis_title="Alpha (Degradation Cost Weight)",
        yaxis_title="State of Charge (kWh)",
        template='mckinsey',
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / "soc_vs_alpha.html"
        fig.write_html(str(plot_path))
        print(f"  [SAVED] {plot_path}")

    return fig

# ============================================================================
# Function 3: Revenue Breakdown Pie Chart (extracted from p2d)
# ============================================================================

def plot_revenue_breakdown_for_alpha(alpha, output_dir=None):
    """
    Create revenue/cost breakdown pie charts for a specific alpha value.

    This function loads the iteration summary for the specified alpha and
    generates side-by-side pie charts showing revenue sources and cost components.

    Parameters
    ----------
    alpha : float
        Alpha value to plot
    output_dir : Path, optional
        Base output directory (default: use global output_base_dir)

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    print(f"\n[PLOTTING] Generating Revenue Breakdown for alpha={alpha:.1f}...")

    if output_dir is None:
        output_dir = output_base_dir
    else:
        output_dir = Path(output_dir)

    # Load iteration summary for this alpha
    alpha_dir = output_dir / f"alpha_{alpha:.1f}"
    iter_csv_path = alpha_dir / "iteration_summary.csv"

    if not iter_csv_path.exists():
        print(f"  [WARNING] Iteration summary not found: {iter_csv_path}")
        return None

    iter_df = pd.read_csv(iter_csv_path)

    # Calculate detailed revenue breakdown
    da_discharge_revenue = iter_df['da_discharge_revenue'].sum() if 'da_discharge_revenue' in iter_df.columns else 0
    fcr_revenue = iter_df['fcr_revenue'].sum() if 'fcr_revenue' in iter_df.columns else 0
    afrr_pos_cap_revenue = iter_df['afrr_pos_cap_revenue'].sum() if 'afrr_pos_cap_revenue' in iter_df.columns else 0
    afrr_neg_cap_revenue = iter_df['afrr_neg_cap_revenue'].sum() if 'afrr_neg_cap_revenue' in iter_df.columns else 0

    # aFRR energy revenue (try separated columns first)
    if 'afrr_pos_e_revenue' in iter_df.columns and 'afrr_neg_e_revenue' in iter_df.columns:
        afrr_pos_energy_revenue = iter_df['afrr_pos_e_revenue'].sum()
        afrr_neg_energy_revenue = iter_df['afrr_neg_e_revenue'].sum()
    elif 'afrr_e_revenue' in iter_df.columns:
        afrr_energy_revenue = iter_df['afrr_e_revenue'].sum()
        afrr_pos_energy_revenue = afrr_energy_revenue
        afrr_neg_energy_revenue = 0
    else:
        afrr_pos_energy_revenue = 0
        afrr_neg_energy_revenue = 0

    # Calculate cost breakdown
    da_charge_cost = iter_df['da_charge_cost'].sum() if 'da_charge_cost' in iter_df.columns else 0
    cyclic_cost = iter_df['cyclic_cost'].sum() if 'cyclic_cost' in iter_df.columns else 0
    calendar_cost = iter_df['calendar_cost'].sum() if 'calendar_cost' in iter_df.columns else 0

    # Build revenue pie chart data
    revenue_pie_data = []
    if da_discharge_revenue > 0:
        revenue_pie_data.append(('DA Discharge', da_discharge_revenue, WATERFALL_COLORS['revenue_primary']))
    if fcr_revenue > 0:
        revenue_pie_data.append(('FCR Capacity', fcr_revenue, WATERFALL_COLORS['revenue_secondary']))
    if afrr_pos_cap_revenue > 0:
        revenue_pie_data.append(('aFRR+ Capacity', afrr_pos_cap_revenue, WATERFALL_COLORS['revenue_secondary']))
    if afrr_neg_cap_revenue > 0:
        revenue_pie_data.append(('aFRR- Capacity', afrr_neg_cap_revenue, 'rgba(34, 81, 255, 0.85)'))
    if afrr_pos_energy_revenue > 0:
        revenue_pie_data.append(('aFRR+ Energy', afrr_pos_energy_revenue, WATERFALL_COLORS['revenue_tertiary']))
    if afrr_neg_energy_revenue > 0:
        revenue_pie_data.append(('aFRR- Energy', afrr_neg_energy_revenue, 'rgba(0, 169, 244, 0.85)'))

    # Build cost pie chart data
    cost_pie_data = []
    if da_charge_cost > 0:
        cost_pie_data.append(('DA Charge Cost', da_charge_cost, WATERFALL_COLORS['cost_primary']))
    if cyclic_cost > 0:
        cost_pie_data.append(('Cyclic Aging', cyclic_cost, WATERFALL_COLORS['cost_secondary']))
    if calendar_cost > 0:
        cost_pie_data.append(('Calendar Aging', calendar_cost, WATERFALL_COLORS['cost_tertiary']))

    # Create side-by-side pie charts
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=('Revenue Sources', 'Cost Sources'),
        horizontal_spacing=0.02
    )

    # Add revenue pie chart (left)
    if revenue_pie_data:
        rev_labels = [item[0] for item in revenue_pie_data]
        rev_values = [item[1] for item in revenue_pie_data]
        rev_colors = [item[2] for item in revenue_pie_data]

        total_rev = sum(rev_values)
        rev_pcts = [v / total_rev * 100 for v in rev_values]
        text_positions = ['inside' if pct > 15 else 'outside' for pct in rev_pcts]

        fig.add_trace(go.Pie(
            labels=rev_labels,
            values=rev_values,
            marker=dict(colors=rev_colors),
            textinfo='label+percent',
            texttemplate='%{label}<br>%{percent}',
            textposition=text_positions,
            insidetextorientation='radial',
            name='Revenue',
            domain={'x': [0, 0.48], 'y': [0, 1]},
            hovertemplate='%{label}<br>€%{value:,.0f}<br>%{percent}<extra></extra>'
        ), row=1, col=1)
    else:
        fig.add_trace(go.Pie(
            labels=['No Revenue'],
            values=[1],
            marker=dict(colors=['#f0f0f0']),
            textinfo='label',
            name='Revenue',
            domain={'x': [0, 0.48], 'y': [0, 1]}
        ), row=1, col=1)

    # Add cost pie chart (right)
    if cost_pie_data:
        cost_labels = [item[0] for item in cost_pie_data]
        cost_values = [item[1] for item in cost_pie_data]
        cost_colors = [item[2] for item in cost_pie_data]

        total_cost = sum(cost_values)
        cost_pcts = [v / total_cost * 100 for v in cost_values]
        text_positions_cost = ['inside' if pct > 15 else 'outside' for pct in cost_pcts]

        fig.add_trace(go.Pie(
            labels=cost_labels,
            values=cost_values,
            marker=dict(colors=cost_colors),
            textinfo='label+percent',
            texttemplate='%{label}<br>%{percent}',
            textposition=text_positions_cost,
            insidetextorientation='radial',
            name='Costs',
            domain={'x': [0.52, 1], 'y': [0, 1]},
            hovertemplate='%{label}<br>€%{value:,.0f}<br>%{percent}<extra></extra>'
        ), row=1, col=2)
    else:
        fig.add_trace(go.Pie(
            labels=['No Costs'],
            values=[1],
            marker=dict(colors=['#f0f0f0']),
            textinfo='label',
            name='Costs'
        ), row=1, col=2)

    # Update layout
    total_revenue = sum([item[1] for item in revenue_pie_data]) if revenue_pie_data else 0
    total_cost = sum([item[1] for item in cost_pie_data]) if cost_pie_data else 0

    fig.update_layout(
        title=f"Financial Breakdown - Alpha {alpha:.1f}<br><sub>Total Revenue: €{total_revenue:,.0f} | Total Cost: €{total_cost:,.0f} | Net Profit: €{total_revenue - total_cost:,.0f}</sub>",
        font=dict(family='Arial', size=11),
        height=550,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )

    # Update pie chart styling
    fig.update_traces(
        textfont_size=10,
        pull=[0.05 if pct < 5 else 0 for pct in (rev_pcts if revenue_pie_data else [])],
        selector=dict(type='pie', name='Revenue')
    )

    if cost_pie_data:
        fig.update_traces(
            textfont_size=10,
            pull=[0.05 if pct < 5 else 0 for pct in cost_pcts],
            selector=dict(type='pie', name='Costs')
        )

    # Save if output directory provided
    if output_dir:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_path = plots_dir / f"revenue_breakdown_alpha_{alpha:.1f}.html"
        fig.write_html(str(plot_path))
        print(f"  [SAVED] {plot_path}")

    return fig

print("[OK] Plotting functions defined:")
print("  - plot_pareto_front(results_df, output_dir)")
print("  - plot_soc_vs_alpha(results_df, output_dir)")
print("  - plot_revenue_breakdown_for_alpha(alpha, output_dir)")
print()

# %%
# ================================================================================
# [SECTION 6] INTERACTIVE CONTROL PANEL
# ================================================================================

print("=" * 80)
print("[SECTION 6] INTERACTIVE CONTROL PANEL")
print("=" * 80)

def show_menu():
    """Display interactive menu for plot generation."""
    print("\n" + "=" * 80)
    print("INTERACTIVE PLOTTING MENU")
    print("=" * 80)
    print("\n[1] Generate Pareto Front Plot")
    print("[2] Generate SOC vs Alpha Plot")
    print("[3] Generate Revenue Breakdown for Specific Alpha")
    print("[4] Generate All Plots")
    print("[5] Export Results Summary & Exit")
    print("[0] Exit Without Saving")
    print()

def interactive_control():
    """Run interactive control loop for plot generation."""

    while True:
        show_menu()
        choice = input("Enter choice [0-5]: ").strip()

        if choice == '1':
            # Pareto front
            fig = plot_pareto_front(results_df, output_base_dir)
            print("  [OK] Pareto front generated!")

        elif choice == '2':
            # SOC vs Alpha
            fig = plot_soc_vs_alpha(results_df, output_base_dir)
            print("  [OK] SOC vs Alpha plot generated!")

        elif choice == '3':
            # Revenue breakdown for specific alpha
            print("\nAvailable alpha values:")
            print(list(results_df['alpha']))
            alpha_str = input("Enter alpha value: ").strip()
            try:
                alpha = float(alpha_str)
                if alpha in results_df['alpha'].values:
                    fig = plot_revenue_breakdown_for_alpha(alpha, output_base_dir)
                    print("  [OK] Revenue breakdown generated!")
                else:
                    print(f"  [ERROR] Alpha {alpha} not found in results.")
            except ValueError:
                print("  [ERROR] Invalid alpha value.")

        elif choice == '4':
            # Generate all plots
            print("\n[START] Generating all plots...")

            # Pareto front
            fig1 = plot_pareto_front(results_df, output_base_dir)

            # SOC vs Alpha
            fig2 = plot_soc_vs_alpha(results_df, output_base_dir)

            # Revenue breakdown for best profit alpha
            best_alpha = results_df.loc[results_df['net_profit_eur'].idxmax(), 'alpha']
            fig3 = plot_revenue_breakdown_for_alpha(best_alpha, output_base_dir)

            print("\n[COMPLETE] All plots generated!")
            print(f"  - Pareto Front")
            print(f"  - SOC vs Alpha")
            print(f"  - Revenue Breakdown (α={best_alpha:.1f})")

        elif choice == '5':
            # Export and exit
            print("\n[START] Exporting final results summary...")

            summary_path = output_base_dir / "RESULTS_SUMMARY.txt"
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("ALPHA META-OPTIMIZATION RESULTS SUMMARY\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Configuration:\n")
                f.write(f"  - Country: {COUNTRY}\n")
                f.write(f"  - C-rate: {C_RATE}\n")
                f.write(f"  - Duration: {SIMULATION_DESCRIPTION}\n")
                f.write(f"  - Alpha range: [{ALPHA_MIN}, {ALPHA_MAX}] step {ALPHA_STEP}\n")
                f.write(f"  - Total simulations: {len(alpha_values)}\n\n")

                f.write(f"Optimal Alpha Values:\n")
                f.write(f"  - Best Net Profit: alpha = {results_df.loc[results_df['net_profit_eur'].idxmax(), 'alpha']:.1f}\n")
                f.write(f"  - Best NPV: alpha = {results_df.loc[results_df['npv_eur'].idxmax(), 'alpha']:.1f}\n")
                f.write(f"  - Best ROI: alpha = {results_df.loc[results_df['roi_proxy'].idxmax(), 'alpha']:.1f}\n\n")

                f.write("Full Results Table:\n")
                f.write(display_df.to_string(index=False))
                f.write("\n\n")

                f.write(f"Output Directory: {output_base_dir}\n")
                f.write(f"Timestamp: {timestamp}\n")

            print(f"  [SAVED] {summary_path}")
            print(f"\n[COMPLETE] All results saved to: {output_base_dir}")
            print("\nThank you for using the Alpha Meta-Optimization tool!")
            break

        elif choice == '0':
            # Exit without saving
            print("\nExiting without saving additional results.")
            break

        else:
            print("\n[ERROR] Invalid choice. Please enter 0-5.")

# Run interactive control
print("\n[INFO] Results aggregation complete. Ready for interactive plotting.")
print("[INFO] You can now generate plots using the menu below, or run plotting")
print("       functions directly in this notebook/script.")
print()

# Auto-run if desired (comment out for pure interactive mode)
if __name__ == "__main__":
    # Uncomment the line below to launch interactive menu automatically
    # interactive_control()

    # Or generate all plots automatically:
    print("[AUTO] Generating all plots automatically...")
    plot_pareto_front(results_df, output_base_dir)
    plot_soc_vs_alpha(results_df, output_base_dir)
    best_alpha = results_df.loc[results_df['net_profit_eur'].idxmax(), 'alpha']
    plot_revenue_breakdown_for_alpha(best_alpha, output_base_dir)
    print("\n[COMPLETE] Auto-plot generation finished!")
    print(f"Results saved to: {output_base_dir}")

print("\n" + "=" * 80)
print("SCRIPT COMPLETE")
print("=" * 80)
