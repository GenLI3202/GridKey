#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive MPC Results Visualization
======================================

Interactive script to visualize MPC simulation results.
Plots are displayed in browser without saving files.

Usage:
    1. Set the RESULT_DIR variable to your MPC results directory
    2. Run the script: python notebook/py_version/mpc_results_interactive.py
    3. Plots will open in your browser automatically
    4. Comment/uncomment sections to show only the plots you want

Author: Claude Code
Date: 2025-11-15
"""

import sys
from pathlib import Path
import pandas as pd
import json
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.optimization_analysis import (
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid,
    plot_soc_and_power_bids
)
from src.visualization.config import MCKINSEY_COLORS, MCKINSEY_FONTS

# =============================================================================
# CONFIGURATION - EDIT THIS!
# =============================================================================

# Set your MPC results directory here
RESULT_DIR = "validation_results/mpc_validation/20251115_141102_mpc_ch_2d_alpha1.0"

# Choose which plots to display (set to False to skip)
SHOW_MPC_PLOTS = True          # MPC-specific plots (SOC iterations, solver, continuity)
SHOW_MARKET_PLOTS = True       # Standard market plots (DA, aFRR, capacity, SOC)

# Individual plot controls
SHOW_SOC_ITERATIONS = True     # SOC trajectory with MPC iteration boundaries
SHOW_SOLVER_PERFORMANCE = True  # Solver times across iterations
SHOW_STATE_CONTINUITY = True   # SOC continuity validation
SHOW_DA_MARKET = True          # Day-ahead market
SHOW_AFRR_ENERGY = True        # aFRR energy market
SHOW_CAPACITY_MARKETS = True   # FCR and aFRR capacity
SHOW_SOC_POWER_BIDS = True     # Complete battery schedule

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Color palette for consistency
PLOT_COLORS = {
    **MCKINSEY_COLORS,
    'red': MCKINSEY_COLORS['negative'],
    'green': MCKINSEY_COLORS['positive'],
    'orange': '#ffa600',
    'gray': MCKINSEY_COLORS['gray_medium']
}


def load_mpc_results(result_dir: Path):
    """Load MPC results from saved CSV and JSON files."""
    result_dir = Path(result_dir)

    # Load performance summary
    perf_file = result_dir / "performance_summary.json"
    with open(perf_file, 'r') as f:
        perf_summary = json.load(f)

    # Load iteration summary
    iter_file = result_dir / "iteration_summary.csv"
    iter_df = pd.read_csv(iter_file)

    # Load solution timeseries
    sol_file = result_dir / "solution_timeseries.csv"
    sol_df = pd.read_csv(sol_file, index_col=0)

    # Parse timestamps if present
    if 'timestamp' in sol_df.columns:
        sol_df['timestamp'] = pd.to_datetime(sol_df['timestamp'])

    return perf_summary, iter_df, sol_df


def print_summary(perf_summary: dict, iter_df: pd.DataFrame, sol_df: pd.DataFrame):
    """Print comprehensive summary of MPC results."""
    print("\n" + "="*80)
    print("MPC SIMULATION RESULTS SUMMARY")
    print("="*80)

    print("\n[Configuration]")
    print(f"  - Country: {perf_summary['country']}")
    print(f"  - Model: {perf_summary['model']}")
    print(f"  - Test Duration: {perf_summary['test_duration_days']} days")
    print(f"  - Alpha: {perf_summary['alpha']}")
    print(f"  - C-rate: {perf_summary['c_rate']}")
    print(f"  - MPC Horizon: {perf_summary['mpc_horizon_hours']}h")
    print(f"  - MPC Execution: {perf_summary['mpc_execution_hours']}h")
    print(f"  - MPC Iterations: {perf_summary['mpc_iterations']}")

    print("\n[Financial Performance]")
    print(f"  - Total Profit: EUR {perf_summary['total_profit_eur']:,.2f}")
    print(f"  - Total Revenue: EUR {perf_summary['total_revenue_eur']:,.2f}")
    print(f"  - Total Degradation: EUR {perf_summary['total_degradation_eur']:,.2f}")

    print("\n  Revenue Breakdown:")
    print(f"    * Day-Ahead: EUR {perf_summary['revenue_da_eur']:,.2f}")
    print(f"    * aFRR Energy: EUR {perf_summary['revenue_afrr_energy_eur']:,.2f}")
    print(f"    * AS Capacity: EUR {perf_summary['revenue_as_capacity_eur']:,.2f}")

    print("\n[Solver Performance]")
    print(f"  - Total Simulation Time: {perf_summary['simulation_time_sec']:.2f}s")
    print(f"  - Average Solve Time: {iter_df['solve_time'].mean():.2f}s")
    print(f"  - Min/Max Solve Time: {iter_df['solve_time'].min():.2f}s / {iter_df['solve_time'].max():.2f}s")

    print("\n" + "="*80 + "\n")


# =============================================================================
# MPC-SPECIFIC PLOTTING FUNCTIONS
# =============================================================================

def plot_mpc_iteration_boundaries(sol_df, iter_df, perf_summary):
    """Plot SOC trajectory with MPC iteration boundaries."""
    execution_hours = perf_summary['mpc_execution_hours']

    fig = go.Figure()

    # Add SOC trajectory
    fig.add_trace(
        go.Scatter(
            x=sol_df['hour'],
            y=sol_df['soc_pct'],
            mode='lines',
            name='SOC',
            line=dict(color=PLOT_COLORS['navy'], width=2),
            hovertemplate='Hour %{x:.2f}<br>SOC: %{y:.1f}%<extra></extra>'
        )
    )

    # Add iteration boundaries
    for i in range(len(iter_df)):
        hour = i * execution_hours
        fig.add_vline(
            x=hour,
            line=dict(color=PLOT_COLORS['red'], width=1, dash='dash'),
            annotation_text=f"Iter {i}",
            annotation_position="top",
            annotation=dict(font=dict(size=10, color=PLOT_COLORS['red']))
        )

    fig.update_layout(
        title=f"MPC SOC Trajectory - {perf_summary['country']} ({perf_summary['test_duration_days']}d, α={perf_summary['alpha']})",
        xaxis_title="Time (hours)",
        yaxis_title="State of Charge (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        height=500
    )

    return fig


def plot_mpc_solver_performance(iter_df, perf_summary):
    """Plot solver performance across iterations."""
    fig = go.Figure()

    # Add solve time bars
    fig.add_trace(
        go.Bar(
            x=iter_df['iteration'],
            y=iter_df['solve_time'],
            name='Solve Time',
            marker_color=PLOT_COLORS['navy'],
            hovertemplate='Iteration %{x}<br>Solve Time: %{y:.2f}s<extra></extra>'
        )
    )

    # Add mean line
    mean_time = iter_df['solve_time'].mean()
    fig.add_hline(
        y=mean_time,
        line=dict(color=PLOT_COLORS['orange'], width=2, dash='dash'),
        annotation_text=f"Mean: {mean_time:.2f}s",
        annotation_position="right"
    )

    total_time = iter_df['solve_time'].sum()

    fig.update_layout(
        title=f"MPC Solver Performance - {perf_summary['country']}<br><sub>Total: {total_time:.1f}s ({total_time/60:.1f} min)</sub>",
        xaxis_title="MPC Iteration",
        yaxis_title="Solve Time (seconds)",
        template="plotly_white",
        hovermode='x unified',
        showlegend=False,
        height=400
    )

    return fig


def plot_mpc_soc_state_continuity(iter_df, perf_summary):
    """Plot SOC at iteration boundaries to validate continuity."""
    fig = go.Figure()

    # SOC at iteration boundaries
    initial_socs = [perf_summary['initial_soc_kwh']] + iter_df['final_soc'].tolist()
    final_socs = iter_df['final_soc'].tolist() + [perf_summary['final_soc_kwh']]
    iterations = list(range(len(iter_df) + 1))

    # Convert to percentage
    initial_socs_pct = [soc / 4472 * 100 for soc in initial_socs]
    final_socs_pct = [soc / 4472 * 100 for soc in final_socs]

    # Calculate differences
    soc_diff_pct = [0] + [abs(initial_socs_pct[i] - final_socs_pct[i-1]) for i in range(1, len(initial_socs_pct))]
    tolerance_pct = 0.1
    is_continuous = [diff <= tolerance_pct for diff in soc_diff_pct]

    # Plot initial SOC at each iteration
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=initial_socs_pct,
            mode='lines+markers',
            name='Initial SOC',
            line=dict(color=PLOT_COLORS['navy'], width=2),
            marker=dict(size=10, color=[
                PLOT_COLORS['green'] if cont else PLOT_COLORS['red']
                for cont in is_continuous
            ]),
            hovertemplate='Iteration %{x}<br>SOC: %{y:.2f}%<br>Diff: %{customdata:.3f}%<extra></extra>',
            customdata=soc_diff_pct
        )
    )

    n_discontinuities = sum(1 for cont in is_continuous if not cont) - 1
    status_text = "[OK] All transitions smooth" if n_discontinuities == 0 else f"[WARNING] {n_discontinuities} discontinuities"

    fig.update_layout(
        title=f"MPC State Continuity - {perf_summary['country']}<br><sub>{status_text} (Tolerance: {tolerance_pct}%)</sub>",
        xaxis_title="MPC Iteration",
        yaxis_title="State of Charge (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        height=500,
        annotations=[
            dict(
                text="<b>Color Code:</b> Green = Smooth | Red = Discontinuity",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=10, color=PLOT_COLORS['gray'])
            )
        ]
    )

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INTERACTIVE MPC RESULTS VISUALIZATION")
    print("="*80)

    # Load results
    result_dir = Path(RESULT_DIR)
    if not result_dir.exists():
        print(f"\nERROR: Result directory not found: {result_dir}")
        print("Please update RESULT_DIR in the script configuration section.")
        sys.exit(1)

    print(f"\nLoading results from: {result_dir}")
    perf_summary, iter_df, sol_df = load_mpc_results(result_dir)

    # Print summary
    print_summary(perf_summary, iter_df, sol_df)

    # Title suffix for all plots
    title_suffix = f"({perf_summary['country']} {perf_summary['test_duration_days']}d, MPC α={perf_summary['alpha']})"

    # =============================================================================
    # MPC-SPECIFIC PLOTS
    # =============================================================================

    if SHOW_MPC_PLOTS:
        print("="*80)
        print("MPC-SPECIFIC PLOTS")
        print("="*80)

        if SHOW_SOC_ITERATIONS:
            print("\n[1/3] SOC Trajectory with Iteration Boundaries...")
            fig = plot_mpc_iteration_boundaries(sol_df, iter_df, perf_summary)
            fig.show()

        if SHOW_SOLVER_PERFORMANCE:
            print("\n[2/3] Solver Performance...")
            fig = plot_mpc_solver_performance(iter_df, perf_summary)
            fig.show()

        if SHOW_STATE_CONTINUITY:
            print("\n[3/3] State Continuity Validation...")
            fig = plot_mpc_soc_state_continuity(iter_df, perf_summary)
            fig.show()

    # =============================================================================
    # STANDARD OPTIMIZATION PLOTS
    # =============================================================================

    if SHOW_MARKET_PLOTS:
        print("\n" + "="*80)
        print("STANDARD OPTIMIZATION PLOTS")
        print("="*80)

        if SHOW_DA_MARKET:
            print("\n[4/7] Day-Ahead Market...")
            fig = plot_da_market_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
            fig.show()

        if SHOW_AFRR_ENERGY:
            print("\n[5/7] aFRR Energy Market...")
            fig = plot_afrr_energy_market_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
            fig.show()

        if SHOW_CAPACITY_MARKETS:
            print("\n[6/7] Capacity Markets...")
            fig = plot_capacity_markets_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
            fig.show()

        if SHOW_SOC_POWER_BIDS:
            print("\n[7/7] SOC and Power Bids...")
            fig = plot_soc_and_power_bids(sol_df, title_suffix=title_suffix, use_timestamp=True)
            fig.show()

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nAll requested plots have been displayed in your browser.")
    print("Close browser tabs when done, or re-run script to refresh.")
    print("="*80 + "\n")
