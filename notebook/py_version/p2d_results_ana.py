# -*- coding: utf-8 -*-
"""
Results Analysis Script for MPC Batch Submission

Comprehensive validation and analysis of the 15-scenario batch results:
- 5 countries: CH, DE_LU, AT, HU, CZ
- 3 C-rates: 0.25, 0.33, 0.5

This script provides multi-level analysis:
1. Batch-wide comparison (all 15 scenarios)
2. Country and C-rate impact analysis
3. Individual scenario deep-dives
4. Comprehensive visualization generation
5. Validation checks and report export

Usage:
- Run all cells for full analysis
- Or run individual sections (# %% markers) for specific analyses
"""

# %%
# ================================================================================
# [SECTION 1] SETUP & IMPORTS
# ================================================================================

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Data processing
import pandas as pd
import numpy as np

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

print("[OK] All imports successful!")
print(f"Project root: {project_root}")

# %%
# ================================================================================
# [SECTION 2] CONFIGURATION
# ================================================================================

# Results directory
RESULTS_BASE_DIR = "submission_results"  # Change to "validation_results/mpc_validation" if needed
RESULTS_PATH = project_root / RESULTS_BASE_DIR

# Analysis mode
ANALYSIS_MODE = "batch"  # "batch" = analyze all 15, "single" = analyze one scenario
SELECTED_SCENARIO = "20251116_183156_ch_crate0.33"  # Used only if ANALYSIS_MODE = "single"

# Visualization settings
GENERATE_PLOTS = True  # Set to False to skip plot generation (faster analysis)
SAVE_PLOTS = True      # Save plots to scenario directories
PLOT_FORMAT = "html"   # "html" or "png"

# Report settings
SAVE_ANALYSIS_REPORT = True  # Export markdown summary report
REPORT_FILENAME = "analysis_report.md"

# Display settings
VERBOSE = True  # Print detailed progress messages

print("=" * 80)
print("[CONFIG] ANALYSIS CONFIGURATION")
print("=" * 80)
print(f"Results Directory:  {RESULTS_BASE_DIR}")
print(f"Analysis Mode:      {ANALYSIS_MODE}")
print(f"Generate Plots:     {GENERATE_PLOTS}")
print(f"Save Report:        {SAVE_ANALYSIS_REPORT}")
print("=" * 80)

# %%
# ================================================================================
# [SECTION 3] HELPER FUNCTIONS
# ================================================================================

def load_scenario_results(scenario_dir):
    """
    Load all result files for a single scenario.

    Parameters
    ----------
    scenario_dir : Path
        Path to scenario directory

    Returns
    -------
    dict
        Dictionary with keys: 'perf_summary', 'iter_summary', 'solution_df', 'scenario_name'
    """
    # Load performance summary
    perf_file = scenario_dir / "performance_summary.json"
    with open(perf_file, 'r') as f:
        perf_summary = json.load(f)

    # Load iteration summary
    iter_file = scenario_dir / "iteration_summary.csv"
    iter_summary = pd.read_csv(iter_file)

    # Load solution timeseries
    sol_file = scenario_dir / "solution_timeseries.csv"
    solution_df = pd.read_csv(sol_file, index_col=0)
    if 'timestamp' in solution_df.columns:
        solution_df['timestamp'] = pd.to_datetime(solution_df['timestamp'])

    return {
        'perf_summary': perf_summary,
        'iter_summary': iter_summary,
        'solution_df': solution_df,
        'scenario_name': scenario_dir.name
    }


def print_scenario_summary(results, verbose=True):
    """Print comprehensive summary for a single scenario."""
    perf = results['perf_summary']
    iter_df = results['iter_summary']

    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {results['scenario_name']}")
    print(f"{'=' * 80}")

    # Basic info
    print(f"\nConfiguration:")
    print(f"  Country:          {perf['country']}")
    print(f"  C-Rate:           {perf['c_rate']}")
    print(f"  Duration:         {perf['test_duration_days']} days")
    print(f"  Alpha:            {perf['alpha']}")
    print(f"  MPC Horizon:      {perf['mpc_horizon_hours']}h / {perf['mpc_execution_hours']}h")
    print(f"  Iterations:       {perf['mpc_iterations']}")

    # Financial results
    print(f"\nFinancial Summary:")
    print(f"  Total Revenue:        >>{perf['total_revenue_eur']:>12,.2f}")
    print(f"  Total Degradation:    >>{perf['total_degradation_eur']:>12,.2f}")
    print(f"  Net Profit:           >>{perf['total_profit_eur']:>12,.2f}")
    profit_margin = (perf['total_profit_eur'] / perf['total_revenue_eur'] * 100) if perf['total_revenue_eur'] > 0 else 0
    print(f"  Profit Margin:        {profit_margin:>13.1f}%")

    # Revenue breakdown
    print(f"\nRevenue Breakdown:")
    print(f"  Day-Ahead:            >>{perf.get('revenue_da_eur', 0):>12,.2f}")
    print(f"  aFRR Energy:          >>{perf.get('revenue_afrr_energy_eur', 0):>12,.2f}")
    print(f"  AS Capacity:          >>{perf.get('revenue_as_capacity_eur', 0):>12,.2f}")

    # # Degradation breakdown
    # print(f"\nDegradation Breakdown:")
    # print(f"  Cyclic:               >>{perf.get('degradation_cyclic_eur', 0):>12,.2f}")
    # print(f"  Calendar:             >>{perf.get('degradation_calendar_eur', 0):>12,.2f}")

    # Battery operation
    print(f"\nBattery Operation:")
    print(f"  Initial SOC:          {perf['initial_soc_kwh']:>12.2f} kWh")
    print(f"  Final SOC:            {perf['final_soc_kwh']:>12.2f} kWh")
    print(f"  SOC Change:           {perf['final_soc_kwh'] - perf['initial_soc_kwh']:>12.2f} kWh")

    # Performance metrics
    print(f"\nPerformance:")
    print(f"  Simulation Time:      {perf['simulation_time_sec']:>12.2f}s ({perf['simulation_time_sec']/60:.2f} min)")
    print(f"  Avg Solve Time:       {iter_df['solve_time'].mean():>12.2f}s/iter")
    print(f"  Solver:               {perf['solver'].upper()}")

    if verbose:
        # Iteration statistics
        print(f"\nIteration Statistics:")
        print(f"  Best Day Profit:      >>{iter_df['profit'].max():>12,.2f} (iter {iter_df['profit'].idxmax()})")
        print(f"  Worst Day Profit:     >>{iter_df['profit'].min():>12,.2f} (iter {iter_df['profit'].idxmin()})")
        print(f"  Avg Daily Profit:     >>{iter_df['profit'].mean():>12,.2f}")
        print(f"  Profit Std Dev:       >>{iter_df['profit'].std():>12,.2f}")

    print(f"{'=' * 80}\n")


def validate_scenario(results):
    """
    Run validation checks on a scenario.

    Returns list of warnings (empty if all checks pass).
    """
    warnings = []
    perf = results['perf_summary']
    iter_df = results['iter_summary']
    sol_df = results['solution_df']

    # Check 1: Iteration count
    expected_iters = perf['test_duration_days']
    if perf['mpc_iterations'] != expected_iters:
        warnings.append(f"Iteration count mismatch: expected {expected_iters}, got {perf['mpc_iterations']}")

    # Check 2: Timestep count
    expected_timesteps = perf['test_duration_days'] * 96
    if len(sol_df) != expected_timesteps:
        warnings.append(f"Timestep count mismatch: expected {expected_timesteps}, got {len(sol_df)}")

    # Check 3: Financial consistency
    revenue_sum = (perf.get('revenue_da_eur', 0) +
                   perf.get('revenue_afrr_energy_eur', 0) +
                   perf.get('revenue_as_capacity_eur', 0))
    if abs(revenue_sum - perf['total_revenue_eur']) > 1.0:  # Allow 1 EUR rounding error
        warnings.append(f"Revenue breakdown mismatch: sum={revenue_sum:.2f}, total={perf['total_revenue_eur']:.2f}")

    deg_sum = (perf.get('degradation_cyclic_eur', 0) +
               perf.get('degradation_calendar_eur', 0))
    if abs(deg_sum - perf['total_degradation_eur']) > 1.0:
        warnings.append(f"Degradation breakdown mismatch: sum={deg_sum:.2f}, total={perf['total_degradation_eur']:.2f}")

    # Check 4: SOC bounds
    if 'soc_kwh' in sol_df.columns:
        soc_min = sol_df['soc_kwh'].min()
        soc_max = sol_df['soc_kwh'].max()
        if soc_min < -1.0 or soc_max > 4473.0:  # Allow small numerical errors
            warnings.append(f"SOC out of bounds: min={soc_min:.2f}, max={soc_max:.2f}")

    # Check 5: Negative profit warning
    if perf['total_profit_eur'] < 0:
        warnings.append(f"NEGATIVE PROFIT: >>{perf['total_profit_eur']:.2f}")

    return warnings


def calculate_detailed_breakdown_from_solution(sol_df, iter_df, perf_summary):
    """
    Calculate detailed financial breakdown for each iteration from solution timeseries.

    This computes what the MPC simulator should have saved but didn't in older runs.

    Parameters
    ----------
    sol_df : pd.DataFrame
        Solution timeseries with power bids, capacity bids, and prices
    iter_df : pd.DataFrame
        Iteration summary with start_timestep, end_timestep
    perf_summary : dict
        Performance summary (for degradation costs if available)

    Returns
    -------
    pd.DataFrame
        Enhanced iteration summary with detailed breakdown columns
    """
    iter_df_enhanced = iter_df.copy()

    # Initialize detailed breakdown columns
    iter_df_enhanced['da_discharge_revenue'] = 0.0
    iter_df_enhanced['da_charge_cost'] = 0.0
    iter_df_enhanced['fcr_revenue'] = 0.0
    iter_df_enhanced['afrr_pos_cap_revenue'] = 0.0
    iter_df_enhanced['afrr_neg_cap_revenue'] = 0.0
    iter_df_enhanced['afrr_e_revenue'] = 0.0
    iter_df_enhanced['afrr_pos_e_revenue'] = 0.0  # Separate tracking
    iter_df_enhanced['afrr_neg_e_revenue'] = 0.0  # Separate tracking
    iter_df_enhanced['cyclic_cost'] = 0.0
    iter_df_enhanced['calendar_cost'] = 0.0

    dt = 0.25  # 15-min timestep in hours

    # Calculate for each iteration
    for idx, row in iter_df_enhanced.iterrows():
        start_ts = int(row['start_timestep'])
        end_ts = int(row['end_timestep'])

        # Slice solution data for this iteration
        iter_sol = sol_df.iloc[start_ts:end_ts]

        if len(iter_sol) == 0:
            continue

        # DA discharge revenue (positive)
        da_discharge = (iter_sol['p_dis'] * iter_sol['price_da_eur_mwh'] / 1000 * dt).sum()
        iter_df_enhanced.at[idx, 'da_discharge_revenue'] = da_discharge

        # DA charge cost (positive value, will be negated in plot)
        da_charge = (iter_sol['p_ch'] * iter_sol['price_da_eur_mwh'] / 1000 * dt).sum()
        iter_df_enhanced.at[idx, 'da_charge_cost'] = da_charge

        # FCR capacity revenue (4-hour blocks, so divide by 4 for 15-min resolution)
        fcr_rev = (iter_sol['c_fcr'] * iter_sol['price_fcr_eur_mw'] / 4).sum()
        iter_df_enhanced.at[idx, 'fcr_revenue'] = fcr_rev

        # aFRR positive capacity revenue
        afrr_pos_cap = (iter_sol['c_afrr_pos'] * iter_sol['price_afrr_cap_pos_eur_mw'] / 4).sum()
        iter_df_enhanced.at[idx, 'afrr_pos_cap_revenue'] = afrr_pos_cap

        # aFRR negative capacity revenue
        afrr_neg_cap = (iter_sol['c_afrr_neg'] * iter_sol['price_afrr_cap_neg_eur_mw'] / 4).sum()
        iter_df_enhanced.at[idx, 'afrr_neg_cap_revenue'] = afrr_neg_cap

        # aFRR energy revenue (preliminary calculation - will be adjusted below)
        # Note: This is the gross revenue before EV weighting
        afrr_e_pos = (iter_sol['p_afrr_pos_e'] * iter_sol['price_afrr_energy_pos_eur_mwh'].fillna(0) / 1000 * dt).sum()
        afrr_e_neg = (iter_sol['p_afrr_neg_e'] * iter_sol['price_afrr_energy_neg_eur_mwh'].fillna(0) / 1000 * dt).sum()
        iter_df_enhanced.at[idx, 'afrr_pos_e_revenue'] = afrr_e_pos
        iter_df_enhanced.at[idx, 'afrr_neg_e_revenue'] = afrr_e_neg
        iter_df_enhanced.at[idx, 'afrr_e_revenue'] = afrr_e_pos + afrr_e_neg

    # Adjust aFRR energy revenue to match performance summary (accounts for EV weighting)
    if 'revenue_afrr_energy_eur' in perf_summary:
        gross_afrr_e = iter_df_enhanced['afrr_e_revenue'].sum()
        actual_afrr_e = perf_summary['revenue_afrr_energy_eur']

        if gross_afrr_e > 0:
            # Apply scaling factor to match actual revenue
            scaling_factor = actual_afrr_e / gross_afrr_e
            iter_df_enhanced['afrr_e_revenue'] *= scaling_factor
            iter_df_enhanced['afrr_pos_e_revenue'] *= scaling_factor
            iter_df_enhanced['afrr_neg_e_revenue'] *= scaling_factor

    # Try to split degradation costs proportionally if we have totals
    if 'degradation_cyclic_eur' in perf_summary and 'degradation_calendar_eur' in perf_summary:
        total_cyclic = perf_summary['degradation_cyclic_eur']
        total_calendar = perf_summary['degradation_calendar_eur']

        # If we have per-iteration profit, we can estimate degradation proportionally
        if 'profit' in iter_df_enhanced.columns:
            # Use absolute values to avoid negative cost allocation when total profit is negative
            total_abs_profit = iter_df_enhanced['profit'].abs().sum()

            for idx, row in iter_df_enhanced.iterrows():
                # Proportional allocation based on absolute profit share
                # This ensures degradation costs are always positive (costs)
                if total_abs_profit > 0:
                    profit_share = abs(row['profit']) / total_abs_profit
                else:
                    profit_share = 1.0 / len(iter_df_enhanced)

                iter_df_enhanced.at[idx, 'cyclic_cost'] = total_cyclic * profit_share
                iter_df_enhanced.at[idx, 'calendar_cost'] = total_calendar * profit_share

    # CRITICAL: Ensure all degradation costs are positive (costs should never be negative)
    # This prevents costs from appearing on the revenue (positive) side of waterfall charts
    iter_df_enhanced['cyclic_cost'] = iter_df_enhanced['cyclic_cost'].abs()
    iter_df_enhanced['calendar_cost'] = iter_df_enhanced['calendar_cost'].abs()
    iter_df_enhanced['da_charge_cost'] = iter_df_enhanced['da_charge_cost'].abs()

    return iter_df_enhanced


print("[OK] Helper functions defined")

# %%
# ================================================================================
# [SECTION 4] BATCH-LEVEL ANALYSIS
# ================================================================================

print("\n" + "=" * 80)
print("[BATCH ANALYSIS] LOADING BATCH SUMMARY")
print("=" * 80)

# Load batch summary
batch_summary_path = RESULTS_PATH / "batch_summary.csv"

if not batch_summary_path.exists():
    print(f"\n[WARNING] Batch summary not found at: {batch_summary_path}")
    print("Attempting to scan directory for individual scenarios...")

    # Scan for scenario directories
    scenario_dirs = [d for d in RESULTS_PATH.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"Found {len(scenario_dirs)} scenario directories")

    if len(scenario_dirs) == 0:
        raise FileNotFoundError(f"No results found in {RESULTS_PATH}")

    # Build batch summary from individual results
    batch_data = []
    for scenario_dir in scenario_dirs:
        try:
            results = load_scenario_results(scenario_dir)
            perf = results['perf_summary']

            batch_data.append({
                'success': True,
                'country': perf['country'],
                'c_rate': perf['c_rate'],
                'profit': perf['total_profit_eur'],
                'revenue': perf['total_revenue_eur'],
                'degradation': perf['total_degradation_eur'],
                'final_soc': perf['final_soc_kwh'],
                'iterations': perf['mpc_iterations'],
                'solve_time': perf['simulation_time_sec'],
                'status': 'SUCCESS',
                'scenario_dir': scenario_dir.name
            })
        except Exception as e:
            print(f"  [ERROR] Failed to load {scenario_dir.name}: {e}")
            batch_data.append({
                'success': False,
                'scenario_dir': scenario_dir.name,
                'status': 'LOAD_ERROR',
                'error': str(e)
            })

    batch_df = pd.DataFrame(batch_data)
else:
    # Load existing batch summary
    batch_df = pd.read_csv(batch_summary_path)
    print(f"[OK] Loaded batch summary: {len(batch_df)} scenarios")

# Display batch summary
print(f"\n{'=' * 80}")
print(f"BATCH SUMMARY ({len(batch_df)} Scenarios)")
print(f"{'=' * 80}\n")

# Overall statistics
n_success = batch_df['success'].sum()
n_failed = len(batch_df) - n_success

print(f"Success Rate:       {n_success}/{len(batch_df)} ({n_success/len(batch_df)*100:.1f}%)")

if n_success > 0:
    successful_df = batch_df[batch_df['success']]

    print(f"\nFinancial Summary (Successful Runs):")
    print(f"  Total Profit:       >>{successful_df['profit'].sum():>12,.2f}")
    print(f"  Avg Profit:         >>{successful_df['profit'].mean():>12,.2f}")
    print(f"  Min Profit:         >>{successful_df['profit'].min():>12,.2f}")
    print(f"  Max Profit:         >>{successful_df['profit'].max():>12,.2f}")
    print(f"  Profit Std Dev:     >>{successful_df['profit'].std():>12,.2f}")

    # Best/worst scenarios
    best_idx = successful_df['profit'].idxmax()
    worst_idx = successful_df['profit'].idxmin()

    print(f"\nBest Scenario:")
    print(f"  {successful_df.loc[best_idx, 'country']} @ C-rate {successful_df.loc[best_idx, 'c_rate']}")
    print(f"  Profit: >>{successful_df.loc[best_idx, 'profit']:,.2f}")

    print(f"\nWorst Scenario:")
    print(f"  {successful_df.loc[worst_idx, 'country']} @ C-rate {successful_df.loc[worst_idx, 'c_rate']}")
    print(f"  Profit: >>{successful_df.loc[worst_idx, 'profit']:,.2f}")

    # Performance stats
    print(f"\nPerformance:")
    print(f"  Total Solve Time:   {successful_df['solve_time'].sum():>12.2f}s ({successful_df['solve_time'].sum()/3600:.2f} hours)")
    print(f"  Avg Solve Time:     {successful_df['solve_time'].mean():>12.2f}s ({successful_df['solve_time'].mean()/60:.2f} min)")
    print(f"  Fastest:            {successful_df['solve_time'].min():>12.2f}s")
    print(f"  Slowest:            {successful_df['solve_time'].max():>12.2f}s")

if n_failed > 0:
    print(f"\n[WARNING] {n_failed} scenarios failed:")
    failed_df = batch_df[~batch_df['success']]
    for _, row in failed_df.iterrows():
        print(f"  - {row.get('country', 'Unknown')} @ C-rate {row.get('c_rate', 'Unknown')}: {row.get('error', 'Unknown error')}")

print(f"\n{'=' * 80}\n")

# %%
# ================================================================================
# [SECTION 5] CROSS-COUNTRY COMPARISON
# ================================================================================

if n_success > 0:
    print("\n" + "=" * 80)
    print("[ANALYSIS] COUNTRY COMPARISON")
    print("=" * 80)

    successful_df = batch_df[batch_df['success']].copy()

    # Aggregate by country
    country_stats = successful_df.groupby('country').agg({
        'profit': ['mean', 'sum', 'std'],
        'revenue': ['mean', 'sum'],
        'degradation': ['mean', 'sum'],
        'solve_time': 'mean'
    }).round(2)

    # Flatten column names
    country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns.values]
    country_stats = country_stats.sort_values('profit_sum', ascending=False)

    print("\nCountry Rankings (sorted by total profit):")
    print(f"{'Country':<10} {'Total Profit':>15} {'Avg Profit':>15} {'Avg Revenue':>15} {'Avg Degrad':>15}")
    print("-" * 80)
    for country, row in country_stats.iterrows():
        print(f"{country:<10} >>{row['profit_sum']:>14,.2f} >>{row['profit_mean']:>14,.2f} >>{row['revenue_mean']:>14,.2f} >>{row['degradation_mean']:>14,.2f}")

    # Visualization: Country comparison
    if GENERATE_PLOTS:
        fig_country = go.Figure()

        fig_country.add_trace(go.Bar(
            x=country_stats.index,
            y=country_stats['profit_sum'],
            name='Total Profit',
            marker_color='#00B4A0',
            text=[f">>{val:,.0f}" for val in country_stats['profit_sum']],
            textposition='outside'
        ))

        fig_country.update_layout(
            title="Total Profit by Country (All C-Rates Combined)",
            xaxis_title="Country",
            yaxis_title="Total Profit (EUR)",
            font=dict(family='Arial', size=12),
            height=500,
            showlegend=False
        )

        if SAVE_PLOTS:
            fig_country.write_html(str(RESULTS_PATH / f"analysis_country_comparison.{PLOT_FORMAT}"))
            print(f"\n[PLOT] Saved: analysis_country_comparison.{PLOT_FORMAT}")

        fig_country.show()

# %% 
# ================================================================================
# [SECTION 6] C-RATE IMPACT ANALYSIS
# ================================================================================

if n_success > 0:
    print("\n" + "=" * 80)
    print("[ANALYSIS] C-RATE IMPACT")
    print("=" * 80)

    # Aggregate by C-rate
    crate_stats = successful_df.groupby('c_rate').agg({
        'profit': ['mean', 'sum', 'std'],
        'revenue': ['mean', 'sum'],
        'degradation': ['mean', 'sum']
    }).round(2)

    crate_stats.columns = ['_'.join(col).strip() for col in crate_stats.columns.values]
    crate_stats = crate_stats.sort_values('profit_sum', ascending=False)

    print("\nC-Rate Rankings (sorted by total profit):")
    print(f"{'C-Rate':<10} {'Total Profit':>15} {'Avg Profit':>15} {'Avg Revenue':>15} {'Avg Degrad':>15}")
    print("-" * 80)
    for crate, row in crate_stats.iterrows():
        print(f"{crate:<10} >>{row['profit_sum']:>14,.2f} >>{row['profit_mean']:>14,.2f} >>{row['revenue_mean']:>14,.2f} >>{row['degradation_mean']:>14,.2f}")

    # Visualization: C-rate comparison
    if GENERATE_PLOTS:
        fig_crate = go.Figure()

        fig_crate.add_trace(go.Bar(
            x=[str(c) for c in crate_stats.index],
            y=crate_stats['profit_sum'],
            name='Total Profit',
            marker_color='#005EB8',
            text=[f">>{val:,.0f}" for val in crate_stats['profit_sum']],
            textposition='outside'
        ))

        fig_crate.update_layout(
            title="Total Profit by C-Rate (All Countries Combined)",
            xaxis_title="C-Rate",
            yaxis_title="Total Profit (EUR)",
            font=dict(family='Arial', size=12),
            height=500,
            showlegend=False
        )

        if SAVE_PLOTS:
            fig_crate.write_html(str(RESULTS_PATH / f"analysis_crate_comparison.{PLOT_FORMAT}"))
            print(f"\n[PLOT] Saved: analysis_crate_comparison.{PLOT_FORMAT}")

        fig_crate.show()

# %%
# ================================================================================
# [SECTION 7] PROFITABILITY HEATMAP
# ================================================================================

if n_success > 0 and GENERATE_PLOTS:
    print("\n" + "=" * 80)
    print("[ANALYSIS] PROFITABILITY HEATMAP")
    print("=" * 80)

    # Pivot table: country >> c-rate
    pivot_profit = successful_df.pivot_table(
        values='profit',
        index='country',
        columns='c_rate',
        aggfunc='sum'
    )

    # Sort by total profit
    pivot_profit['total'] = pivot_profit.sum(axis=1)
    pivot_profit = pivot_profit.sort_values('total', ascending=False)
    pivot_profit = pivot_profit.drop('total', axis=1)

    # Import McKinsey colors for heatmap
    from src.visualization.config import MCKINSEY_COLORS

    # Create McKinsey blue color scale (light to dark blue)
    # Custom colorscale: white -> light blue -> medium blue -> dark blue -> navy
    mckinsey_blue_scale = [
        [0.0, '#f8f9fa'],   # Very light gray (low profit)
        [0.2, '#d4e6f1'],   # Light blue
        [0.4, '#85c1e9'],   # Medium light blue
        [0.6, '#3498db'],   # Medium blue
        [0.8, '#2874a6'],   # Dark blue
        [1.0, '#003f5c']    # Navy (high profit)
    ]

    # Create heatmap with McKinsey blue theme
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_profit.values,
        x=[str(c) for c in pivot_profit.columns],
        y=pivot_profit.index,
        colorscale=mckinsey_blue_scale,
        text=[[f"{val:,.0f}" for val in row] for row in pivot_profit.values],
        texttemplate='%{text}',
        textfont={"size": 11, "color": "black"},
        colorbar=dict(
            title=dict(
                text="Profit (EUR)",
                font=dict(size=12, color=MCKINSEY_COLORS['navy'])
            ),
            tickfont=dict(size=10)
        ),
        hovertemplate='Country: %{y}<br>C-Rate: %{x}<br>Profit: €%{text}<extra></extra>'
    ))

    fig_heatmap.update_layout(
        title=dict(
            text="Profitability Heatmap: Country >> C-Rate",
            font=dict(size=16, color=MCKINSEY_COLORS['navy'])
        ),
        xaxis_title="C-Rate",
        yaxis_title="Country",
        font=dict(family='Arial, Helvetica, sans-serif', size=12),
        height=450,
        width=650,
        plot_bgcolor=MCKINSEY_COLORS['bg_white'],
        paper_bgcolor=MCKINSEY_COLORS['bg_white']
    )

    # Update axes styling
    fig_heatmap.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor=MCKINSEY_COLORS['gray_dark'],
        tickfont=dict(size=11)
    )

    fig_heatmap.update_yaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor=MCKINSEY_COLORS['gray_dark'],
        tickfont=dict(size=11)
    )

    if SAVE_PLOTS:
        fig_heatmap.write_html(str(RESULTS_PATH / f"analysis_profitability_heatmap.{PLOT_FORMAT}"))
        print(f"[PLOT] Saved: analysis_profitability_heatmap.{PLOT_FORMAT}")

    fig_heatmap.show()

# %%
# ================================================================================
# [SECTION 8] SINGLE SCENARIO DEEP DIVE
# ================================================================================

# Determine which scenarios to analyze in detail
if ANALYSIS_MODE == "single":
    # Analyze only the selected scenario
    scenario_path = RESULTS_PATH / SELECTED_SCENARIO
    if not scenario_path.exists():
        raise FileNotFoundError(f"Selected scenario not found: {scenario_path}")
    scenarios_to_analyze = [scenario_path]
    print(f"\n[MODE] Single scenario analysis: {SELECTED_SCENARIO}")

elif ANALYSIS_MODE == "batch":
    # Analyze all successful scenarios
    if n_success > 0:
        scenarios_to_analyze = []
        for _, row in successful_df.iterrows():
            # Try to find scenario directory
            if 'scenario_dir' in row:
                scenario_dir = RESULTS_PATH / row['scenario_dir']
            else:
                # Search by country and c_rate
                pattern = f"*{row['country'].lower()}*crate{row['c_rate']}*"
                matches = list(RESULTS_PATH.glob(pattern))
                if matches:
                    scenario_dir = matches[0]
                else:
                    continue

            if scenario_dir.exists():
                scenarios_to_analyze.append(scenario_dir)

        print(f"\n[MODE] Batch analysis: {len(scenarios_to_analyze)} scenarios")
    else:
        scenarios_to_analyze = []
        print("\n[WARNING] No successful scenarios to analyze")
else:
    scenarios_to_analyze = []



# Analyze each scenario
for i, scenario_dir in enumerate(scenarios_to_analyze, 1):
    print(f"\n{'#' * 80}")
    print(f"# ANALYZING SCENARIO {i}/{len(scenarios_to_analyze)}")
    print(f"{'#' * 80}")

    # Load scenario results
    results = load_scenario_results(scenario_dir)

    # Print summary
    print_scenario_summary(results, verbose=VERBOSE)

    # Run validation checks
    warnings = validate_scenario(results)
    if warnings:
        print(f"[WARNING] Validation issues found:")
        for warning in warnings:
            print(f"  >> {warning}")
    else:
        print(f"[OK] All validation checks passed")

    # Generate plots if enabled
    if GENERATE_PLOTS:
        print(f"\n[PLOTS] Generating visualizations for {results['scenario_name']}...")

        plots_dir = scenario_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        perf = results['perf_summary']
        iter_df = results['iter_summary']
        sol_df = results['solution_df']

        # Calculate detailed breakdown from solution timeseries if not already present
        has_detailed = all(col in iter_df.columns for col in [
            'da_discharge_revenue', 'da_charge_cost', 'fcr_revenue', 'cyclic_cost', 'calendar_cost'
        ])

        if not has_detailed:
            if VERBOSE:
                print("      [INFO] Computing detailed breakdown from solution timeseries...")
            iter_df = calculate_detailed_breakdown_from_solution(sol_df, iter_df, perf)

        title_suffix = f"{perf['country']} @ C-rate {perf['c_rate']} ({perf['test_duration_days']}d)"

        # Standard optimization plots (4)
        print("  [1/10] Day-Ahead Market...")
        fig1 = plot_da_market_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
        if SAVE_PLOTS:
            fig1.write_html(str(plots_dir / f"da_market_price_bid.{PLOT_FORMAT}"))

        print("  [2/10] aFRR Energy Market...")
        fig2 = plot_afrr_energy_market_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
        if SAVE_PLOTS:
            fig2.write_html(str(plots_dir / f"afrr_energy_market_price_bid.{PLOT_FORMAT}"))

        print("  [3/10] Capacity Markets...")
        fig3 = plot_capacity_markets_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
        if SAVE_PLOTS:
            fig3.write_html(str(plots_dir / f"capacity_markets_price_bid.{PLOT_FORMAT}"))

        print("  [4/10] SOC & Power Bids...")
        fig4 = plot_soc_and_power_bids(sol_df, title_suffix=title_suffix, use_timestamp=True)
        if SAVE_PLOTS:
            fig4.write_html(str(plots_dir / f"soc_and_power_bids.{PLOT_FORMAT}"))

        # MPC-specific plots (3)
        # Reconstruct mpc_results dict for plotting functions
        soc_trajectory = [perf['initial_soc_kwh']]
        for _, iter_row in iter_df.iterrows():
            end_idx = int(iter_row['end_timestep'])
            if end_idx < len(sol_df):
                soc_trajectory.append(sol_df['soc_kwh'].iloc[end_idx])
        if len(soc_trajectory) <= len(iter_df):
            soc_trajectory.append(perf['final_soc_kwh'])

        # Build iteration_results with detailed financial breakdown
        # NOTE: iter_df loaded from iteration_summary.csv should contain:
        #   - Aggregated: revenue, degradation_cost, profit
        #   - Detailed revenue: da_discharge_revenue, fcr_revenue, afrr_e_revenue, etc.
        #   - Detailed costs: da_charge_cost, cyclic_cost, calendar_cost
        # These detailed columns enable waterfall-style visualization in plot_iteration_performance()
        mpc_results = {
            'total_revenue': perf['total_revenue_eur'],
            'total_degradation_cost': perf['total_degradation_eur'],
            'net_profit': perf['total_profit_eur'],
            'final_soc': perf['final_soc_kwh'],
            'soc_trajectory': soc_trajectory,
            'soc_15min': sol_df['soc_kwh'].tolist(),
            'iteration_results': iter_df.to_dict('records')  # Passes detailed breakdown to plotting
        }

        # print("  [5/10] Iteration Boundaries...")
        # fig5 = plot_iteration_boundaries(
        #     mpc_results,
        #     execution_hours=perf['mpc_execution_hours'],
        #     title_suffix=title_suffix,
        #     show_horizons=False
        # )
        # if SAVE_PLOTS:
        #     fig5.write_html(str(plots_dir / f"mpc_iteration_boundaries.{PLOT_FORMAT}"))

        print("  [6/10] Iteration Performance...")

        fig6 = plot_iteration_performance(
            mpc_results,
            title_suffix=title_suffix,
            show_cumulative=True
        )
        if SAVE_PLOTS:
            fig6.write_html(str(plots_dir / f"mpc_iteration_performance.{PLOT_FORMAT}"))

        # print("  [7/10] State Continuity...")
        # fig7 = plot_state_continuity(
        #     mpc_results,
        #     title_suffix=title_suffix,
        #     tolerance_pct=0.1
        # )
        # if SAVE_PLOTS:
        #     fig7.write_html(str(plots_dir / f"mpc_state_continuity.{PLOT_FORMAT}"))

        # Custom analysis plots (3)
        print("  [8/10] Financial Breakdown (Revenue & Cost)...")

        # Import waterfall colors for consistency
        from src.visualization.config import WATERFALL_COLORS, MCKINSEY_COLORS
        from plotly.subplots import make_subplots

        # Calculate detailed revenue breakdown from iteration data
        # This matches the waterfall chart breakdown
        da_discharge_revenue = iter_df['da_discharge_revenue'].sum() if 'da_discharge_revenue' in iter_df.columns else 0
        fcr_revenue = iter_df['fcr_revenue'].sum() if 'fcr_revenue' in iter_df.columns else 0
        afrr_pos_cap_revenue = iter_df['afrr_pos_cap_revenue'].sum() if 'afrr_pos_cap_revenue' in iter_df.columns else 0
        afrr_neg_cap_revenue = iter_df['afrr_neg_cap_revenue'].sum() if 'afrr_neg_cap_revenue' in iter_df.columns else 0

        # Split aFRR energy revenue into positive and negative components
        # Extract from solution timeseries for accurate breakdown
        if 'afrr_pos_e_revenue' in iter_df.columns and 'afrr_neg_e_revenue' in iter_df.columns:
            # Use separated columns from detailed breakdown
            afrr_pos_energy_revenue = iter_df['afrr_pos_e_revenue'].sum()
            afrr_neg_energy_revenue = iter_df['afrr_neg_e_revenue'].sum()
        elif 'afrr_e_revenue' in iter_df.columns:
            # Fallback: use total (unable to split)
            afrr_energy_revenue = iter_df['afrr_e_revenue'].sum()
            afrr_pos_energy_revenue = afrr_energy_revenue
            afrr_neg_energy_revenue = 0
        else:
            # Last resort: use performance summary
            afrr_energy_revenue = perf.get('revenue_afrr_energy_eur', 0)
            afrr_pos_energy_revenue = afrr_energy_revenue
            afrr_neg_energy_revenue = 0

        # Calculate cost breakdown
        da_charge_cost = iter_df['da_charge_cost'].sum() if 'da_charge_cost' in iter_df.columns else 0
        cyclic_cost = iter_df['cyclic_cost'].sum() if 'cyclic_cost' in iter_df.columns else perf.get('degradation_cyclic_eur', 0)
        calendar_cost = iter_df['calendar_cost'].sum() if 'calendar_cost' in iter_df.columns else perf.get('degradation_calendar_eur', 0)

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

        # Create side-by-side pie charts with minimal spacing
        fig8 = make_subplots(
            rows=1, cols=2,
            specs=[[{'type':'domain'}, {'type':'domain'}]],
            subplot_titles=('Revenue Sources', 'Cost Sources'),
            horizontal_spacing=0.02  # Minimal spacing between pies
        )

        # Add revenue pie chart (left)
        if revenue_pie_data:
            rev_labels = [item[0] for item in revenue_pie_data]
            rev_values = [item[1] for item in revenue_pie_data]
            rev_colors = [item[2] for item in revenue_pie_data]

            # Calculate percentages for smart positioning
            total_rev = sum(rev_values)
            rev_pcts = [v / total_rev * 100 for v in rev_values]

            # Use 'inside' for large segments (>15%), 'outside' for small ones
            # This prevents label cramping when one segment dominates
            text_positions = ['inside' if pct > 15 else 'outside' for pct in rev_pcts]

            fig8.add_trace(go.Pie(
                labels=rev_labels,
                values=rev_values,
                marker=dict(colors=rev_colors),
                textinfo='label+percent',
                texttemplate='%{label}<br>%{percent}',  # Simplified for large segments
                textposition=text_positions,
                insidetextorientation='radial',  # Better readability for inside labels
                name='Revenue',
                domain={'x': [0, 0.48], 'y': [0, 1]},  # Larger pie (48% width)
                hovertemplate='%{label}<br>€%{value:,.0f}<br>%{percent}<extra></extra>'  # Full info on hover
            ), row=1, col=1)
        else:
            # Empty pie with message
            fig8.add_trace(go.Pie(
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

            # Calculate percentages for smart positioning
            total_cost = sum(cost_values)
            cost_pcts = [v / total_cost * 100 for v in cost_values]

            # Use 'inside' for large segments (>15%), 'outside' for small ones
            text_positions_cost = ['inside' if pct > 15 else 'outside' for pct in cost_pcts]

            fig8.add_trace(go.Pie(
                labels=cost_labels,
                values=cost_values,
                marker=dict(colors=cost_colors),
                textinfo='label+percent',
                texttemplate='%{label}<br>%{percent}',  # Simplified for large segments
                textposition=text_positions_cost,
                insidetextorientation='radial',  # Better readability for inside labels
                name='Costs',
                domain={'x': [0.52, 1], 'y': [0, 1]},  # Larger pie (48% width)
                hovertemplate='%{label}<br>€%{value:,.0f}<br>%{percent}<extra></extra>'  # Full info on hover
            ), row=1, col=2)
        else:
            # Empty pie with message
            fig8.add_trace(go.Pie(
                labels=['No Costs'],
                values=[1],
                marker=dict(colors=['#f0f0f0']),
                textinfo='label',
                name='Costs'
            ), row=1, col=2)

        # Update layout
        total_revenue = sum([item[1] for item in revenue_pie_data]) if revenue_pie_data else 0
        total_cost = sum([item[1] for item in cost_pie_data]) if cost_pie_data else 0

        fig8.update_layout(
            title=f"Financial Breakdown - {title_suffix}<br><sub>Total Revenue: €{total_revenue:,.0f} | Total Cost: €{total_cost:,.0f} | Net Profit: €{total_revenue - total_cost:,.0f}</sub>",
            font=dict(family='Arial', size=11),  # Slightly smaller base font
            height=550,  # 10% larger height
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

        # Update pie chart text font size
        fig8.update_traces(
            textfont_size=10,  # Smaller text to reduce crowding
            pull=[0.05 if pct < 5 else 0 for pct in (rev_pcts if revenue_pie_data else [])],  # Pull out very small slices
            selector=dict(type='pie', name='Revenue')
        )

        if cost_pie_data:
            fig8.update_traces(
                textfont_size=10,
                pull=[0.05 if pct < 5 else 0 for pct in cost_pcts],
                selector=dict(type='pie', name='Costs')
            )

        if SAVE_PLOTS:
            fig8.write_html(str(plots_dir / f"financial_revenue_breakdown.{PLOT_FORMAT}"))

   
        print("  [10/10] Daily Profit Distribution...")

        # Calculate statistics
        mean_profit = iter_df['profit'].mean()
        median_profit = iter_df['profit'].median()
        std_profit = iter_df['profit'].std()

        fig10 = go.Figure()

        # Histogram with McKinsey color
        fig10.add_trace(go.Histogram(
            x=iter_df['profit'],
            nbinsx=30,
            marker_color=WATERFALL_COLORS['revenue_secondary'],  # McKinsey blue
            marker_line_color=MCKINSEY_COLORS['navy'],
            marker_line_width=0.5,
            name='Daily Profit',
            opacity=0.85
        ))

        # Mean line with McKinsey dark blue
        fig10.add_vline(
            x=mean_profit,
            line_dash="dash",
            line_color=MCKINSEY_COLORS['navy'],
            line_width=2,
            annotation_text=f"Mean: €{mean_profit:,.0f}",
            annotation_position="top right",
            annotation=dict(
                font=dict(size=11, color=MCKINSEY_COLORS['navy']),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor=MCKINSEY_COLORS['gray_light'],
                borderwidth=1
            )
        )

        # Apply McKinsey template and styling
        fig10.update_layout(
            template='mckinsey',
            title=dict(
                text=f"Daily Profit Distribution - {title_suffix}<br><sub>Mean: €{mean_profit:,.0f} | Median: €{median_profit:,.0f} | Std Dev: €{std_profit:,.0f}</sub>",
                font=dict(size=16, color=MCKINSEY_COLORS['navy'])
            ),
            xaxis_title="Daily Profit (EUR)",
            yaxis_title="Frequency (Days)",
            font=dict(family='Arial, Helvetica, sans-serif', size=12),
            height=500,
            showlegend=False,
            plot_bgcolor=MCKINSEY_COLORS['bg_light_gray'],
            paper_bgcolor=MCKINSEY_COLORS['bg_white']
        )

        # Update axis styling
        fig10.update_xaxes(
            showgrid=True,
            gridcolor=MCKINSEY_COLORS['gray_light'],
            gridwidth=0.5,
            showline=True,
            linewidth=1,
            linecolor=MCKINSEY_COLORS['gray_dark'],
            tickformat=',',
            zeroline=False
        )

        fig10.update_yaxes(
            showgrid=True,
            gridcolor=MCKINSEY_COLORS['gray_light'],
            gridwidth=0.5,
            showline=True,
            linewidth=1,
            linecolor=MCKINSEY_COLORS['gray_dark'],
            zeroline=False
        )

        if SAVE_PLOTS:
            fig10.write_html(str(plots_dir / f"daily_profit_distribution.{PLOT_FORMAT}"))

        print(f"  [OK] All plots saved to {plots_dir}/")

    print(f"\n{'#' * 80}\n")

# %%
# ================================================================================
# [SECTION 9] EXPORT ANALYSIS REPORT
# ================================================================================

if SAVE_ANALYSIS_REPORT and n_success > 0:
    print("\n" + "=" * 80)
    print("[EXPORT] GENERATING ANALYSIS REPORT")
    print("=" * 80)

    report_path = RESULTS_PATH / REPORT_FILENAME

    with open(report_path, 'w') as f:
        f.write("# MPC Submission Results Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Results Directory:** `{RESULTS_BASE_DIR}/`\n\n")
        f.write("---\n\n")

        # Batch Summary
        f.write("## Batch Summary (15 Scenarios)\n\n")
        f.write(f"- **Success Rate:** {n_success}/{len(batch_df)} ({n_success/len(batch_df)*100:.1f}%)\n")
        f.write(f"- **Total Profit:** >>{successful_df['profit'].sum():,.2f}\n")
        f.write(f"- **Average Profit:** >>{successful_df['profit'].mean():,.2f}\n")
        f.write(f"- **Profit Range:** >>{successful_df['profit'].min():,.2f} - >>{successful_df['profit'].max():,.2f}\n")
        f.write(f"- **Total Simulation Time:** {successful_df['solve_time'].sum()/3600:.2f} hours\n\n")

        # Best/Worst
        best_idx = successful_df['profit'].idxmax()
        worst_idx = successful_df['profit'].idxmin()

        f.write(f"**Best Scenario:** {successful_df.loc[best_idx, 'country']} @ C-rate {successful_df.loc[best_idx, 'c_rate']} >> >>{successful_df.loc[best_idx, 'profit']:,.2f}\n\n")
        f.write(f"**Worst Scenario:** {successful_df.loc[worst_idx, 'country']} @ C-rate {successful_df.loc[worst_idx, 'c_rate']} >> >>{successful_df.loc[worst_idx, 'profit']:,.2f}\n\n")

        # Country Rankings
        f.write("---\n\n")
        f.write("## Country Rankings\n\n")
        f.write("| Rank | Country | Total Profit | Avg Profit | Avg Revenue | Avg Degradation |\n")
        f.write("|------|---------|--------------|------------|-------------|------------------|\n")

        for rank, (country, row) in enumerate(country_stats.iterrows(), 1):
            f.write(f"| {rank} | {country} | >>{row['profit_sum']:,.2f} | >>{row['profit_mean']:,.2f} | >>{row['revenue_mean']:,.2f} | >>{row['degradation_mean']:,.2f} |\n")

        # C-Rate Rankings
        f.write("\n---\n\n")
        f.write("## C-Rate Rankings\n\n")
        f.write("| Rank | C-Rate | Total Profit | Avg Profit | Avg Revenue | Avg Degradation |\n")
        f.write("|------|--------|--------------|------------|-------------|------------------|\n")

        for rank, (crate, row) in enumerate(crate_stats.iterrows(), 1):
            f.write(f"| {rank} | {crate} | >>{row['profit_sum']:,.2f} | >>{row['profit_mean']:,.2f} | >>{row['revenue_mean']:,.2f} | >>{row['degradation_mean']:,.2f} |\n")

        # Individual Scenarios
        f.write("\n---\n\n")
        f.write("## Individual Scenario Details\n\n")

        for _, row in successful_df.sort_values('profit', ascending=False).iterrows():
            f.write(f"### {row['country']} @ C-rate {row['c_rate']}\n\n")
            f.write(f"- **Profit:** >>{row['profit']:,.2f}\n")
            f.write(f"- **Revenue:** >>{row['revenue']:,.2f}\n")
            f.write(f"- **Degradation:** >>{row['degradation']:,.2f}\n")
            f.write(f"- **Final SOC:** {row['final_soc']:.2f} kWh\n")
            f.write(f"- **Iterations:** {row['iterations']}\n")
            f.write(f"- **Solve Time:** {row['solve_time']/60:.2f} min\n\n")

        # Footer
        f.write("\n---\n\n")
        f.write("*Report generated by `p2d_results_ana.py`*\n")

    print(f"[OK] Analysis report saved to: {report_path}")
    print("=" * 80)

# %%
# ================================================================================
# [COMPLETE] ANALYSIS COMPLETE!
# ================================================================================

print("\n" + "=" * 80)
print("[COMPLETE] RESULTS ANALYSIS FINISHED")
print("=" * 80)

if SAVE_ANALYSIS_REPORT:
    print(f"\n=>> Analysis report: {RESULTS_PATH / REPORT_FILENAME}")

if GENERATE_PLOTS and SAVE_PLOTS:
    print(f"\n=>> Plots saved to individual scenario directories under plots/")

print(f"\n Yeah!  All analysis complete!")
print("=" * 80)

# %%
