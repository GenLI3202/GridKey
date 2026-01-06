#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPC Saved Results Visualization
================================

Quick visualization of MPC simulation results from saved CSV files.
Generates comprehensive plots for MPC validation and analysis.

Usage:
    python py_script/visualization/plot_mpc_saved_results.py <result_dir>

Example:
    python py_script/visualization/plot_mpc_saved_results.py validation_results/mpc_validation/20251115_141102_mpc_ch_2d_alpha1.0
"""

import pandas as pd
import json
from pathlib import Path
import sys
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from py_script.visualization.optimization_analysis import (
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid,
    plot_soc_and_power_bids
)
from py_script.visualization.config import MCKINSEY_COLORS, MCKINSEY_FONTS

# Define additional colors for compatibility
PLOT_COLORS = {
    **MCKINSEY_COLORS,
    'red': MCKINSEY_COLORS['negative'],
    'green': MCKINSEY_COLORS['positive'],
    'orange': '#ffa600',
    'gray': MCKINSEY_COLORS['gray_medium']
}


def load_mpc_results(result_dir: Path):
    """Load MPC results from saved CSV and JSON files."""

    # Load performance summary
    perf_file = result_dir / "performance_summary.json"
    if not perf_file.exists():
        raise FileNotFoundError(f"Performance summary not found: {perf_file}")

    with open(perf_file, 'r') as f:
        perf_summary = json.load(f)

    # Load iteration summary
    iter_file = result_dir / "iteration_summary.csv"
    if not iter_file.exists():
        raise FileNotFoundError(f"Iteration summary not found: {iter_file}")

    iter_df = pd.read_csv(iter_file)

    # Load solution timeseries
    sol_file = result_dir / "solution_timeseries.csv"
    if not sol_file.exists():
        raise FileNotFoundError(f"Solution timeseries not found: {sol_file}")

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

    # Configuration
    print("\n[Configuration]")
    print(f"  - Country: {perf_summary['country']}")
    print(f"  - Model: {perf_summary['model']}")
    print(f"  - Test Duration: {perf_summary['test_duration_days']} days ({len(sol_df)} timesteps)")
    print(f"  - Alpha: {perf_summary['alpha']}")
    print(f"  - C-rate: {perf_summary['c_rate']}")
    print(f"  - MPC Horizon: {perf_summary['mpc_horizon_hours']}h")
    print(f"  - MPC Execution: {perf_summary['mpc_execution_hours']}h")
    print(f"  - MPC Iterations: {perf_summary['mpc_iterations']}")

    # Financial Performance
    print("\n[Financial Performance]")
    print(f"  - Total Profit: EUR {perf_summary['total_profit_eur']:,.2f}")
    print(f"  - Total Revenue: EUR {perf_summary['total_revenue_eur']:,.2f}")
    print(f"  - Total Degradation: EUR {perf_summary['total_degradation_eur']:,.2f}")

    print("\n  Revenue Breakdown:")
    print(f"    * Day-Ahead: EUR {perf_summary['revenue_da_eur']:,.2f}")
    print(f"    * aFRR Energy: EUR {perf_summary['revenue_afrr_energy_eur']:,.2f}")
    print(f"    * AS Capacity: EUR {perf_summary['revenue_as_capacity_eur']:,.2f}")

    print("\n  Degradation Breakdown:")
    print(f"    * Cyclic: EUR {perf_summary['degradation_cyclic_eur']:,.2f}")
    print(f"    * Calendar: EUR {perf_summary['degradation_calendar_eur']:,.2f}")

    # Battery Operation
    print("\n[Battery Operation]")
    print(f"  - Initial SOC: {perf_summary['initial_soc_kwh']:.1f} kWh ({perf_summary['initial_soc_kwh']/4472*100:.1f}%)")
    print(f"  - Final SOC: {perf_summary['final_soc_kwh']:.1f} kWh ({perf_summary['final_soc_kwh']/4472*100:.1f}%)")
    print(f"  - SOC Range: {sol_df['soc_kwh'].min():.1f} - {sol_df['soc_kwh'].max():.1f} kWh")
    print(f"  - Total DA Charge: {sol_df['p_ch_kw'].sum()/1000*0.25:.2f} MWh")
    print(f"  - Total DA Discharge: {sol_df['p_dis_kw'].sum()/1000*0.25:.2f} MWh")

    # Market Participation
    print("\n[Market Participation]")
    n_timesteps = len(sol_df)
    print(f"  - DA Charge: {(sol_df['p_ch_kw'] > 0).sum()}/{n_timesteps} timesteps ({(sol_df['p_ch_kw'] > 0).sum()/n_timesteps*100:.1f}%)")
    print(f"  - DA Discharge: {(sol_df['p_dis_kw'] > 0).sum()}/{n_timesteps} timesteps ({(sol_df['p_dis_kw'] > 0).sum()/n_timesteps*100:.1f}%)")
    print(f"  - aFRR Energy Pos: {(sol_df['p_afrr_pos_e_kw'] > 0).sum()}/{n_timesteps} timesteps")
    print(f"  - aFRR Energy Neg: {(sol_df['p_afrr_neg_e_kw'] > 0).sum()}/{n_timesteps} timesteps")
    print(f"  - FCR Active: {(sol_df['c_fcr_mw'] > 0).sum()}/{n_timesteps} timesteps")
    print(f"  - aFRR Cap Pos Active: {(sol_df['c_afrr_pos_mw'] > 0).sum()}/{n_timesteps} timesteps")
    print(f"  - aFRR Cap Neg Active: {(sol_df['c_afrr_neg_mw'] > 0).sum()}/{n_timesteps} timesteps")

    # Solver Performance
    print("\n[Solver Performance]")
    print(f"  - Total Simulation Time: {perf_summary['simulation_time_sec']:.2f}s ({perf_summary['simulation_time_sec']/60:.2f} min)")
    print(f"  - Average Solve Time: {iter_df['solve_time'].mean():.2f}s")
    print(f"  - Min Solve Time: {iter_df['solve_time'].min():.2f}s")
    print(f"  - Max Solve Time: {iter_df['solve_time'].max():.2f}s")

    print("\n" + "="*80)


def plot_mpc_iteration_boundaries(sol_df: pd.DataFrame, iter_df: pd.DataFrame,
                                   perf_summary: dict, output_dir: Path):
    """Plot SOC trajectory with MPC iteration boundaries."""

    execution_hours = perf_summary['mpc_execution_hours']
    execution_timesteps = execution_hours * 4  # 15-min intervals

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
        title=dict(
            text=f"MPC SOC Trajectory - {perf_summary['country']} ({perf_summary['test_duration_days']}d, α={perf_summary['alpha']})",
            font=dict(size=16, family=MCKINSEY_FONTS['family'])
        ),
        xaxis_title="Time (hours)",
        yaxis_title="State of Charge (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        font=dict(family=MCKINSEY_FONTS['family']),
        height=500
    )

    output_file = output_dir / "mpc_soc_iterations.html"
    fig.write_html(str(output_file))
    print(f"  [OK] Saved: {output_file}")
    return fig


def plot_mpc_solver_performance(iter_df: pd.DataFrame, perf_summary: dict, output_dir: Path):
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
        title=dict(
            text=f"MPC Solver Performance - {perf_summary['country']}<br><sub>Total: {total_time:.1f}s ({total_time/60:.1f} min)</sub>",
            font=dict(size=16, family=MCKINSEY_FONTS['family'])
        ),
        xaxis_title="MPC Iteration",
        yaxis_title="Solve Time (seconds)",
        template="plotly_white",
        hovermode='x unified',
        showlegend=False,
        font=dict(family=MCKINSEY_FONTS['family']),
        height=400
    )

    output_file = output_dir / "mpc_solver_performance.html"
    fig.write_html(str(output_file))
    print(f"  [OK] Saved: {output_file}")
    return fig


def plot_mpc_soc_state_continuity(iter_df: pd.DataFrame, perf_summary: dict, output_dir: Path):
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
        title=dict(
            text=f"MPC State Continuity - {perf_summary['country']}<br><sub>{status_text} (Tolerance: {tolerance_pct}%)</sub>",
            font=dict(size=16, family=MCKINSEY_FONTS['family'])
        ),
        xaxis_title="MPC Iteration",
        yaxis_title="State of Charge (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        font=dict(family=MCKINSEY_FONTS['family']),
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

    output_file = output_dir / "mpc_state_continuity.html"
    fig.write_html(str(output_file))
    print(f"  [OK] Saved: {output_file}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize MPC simulation results from saved files'
    )
    parser.add_argument(
        'result_dir',
        type=str,
        help='Path to MPC results directory'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: result_dir/plots)'
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"ERROR: Result directory not found: {result_dir}")
        return 1

    # Load results
    print(f"\nLoading MPC results from: {result_dir}")
    perf_summary, iter_df, sol_df = load_mpc_results(result_dir)

    # Print summary
    print_summary(perf_summary, iter_df, sol_df)

    # Create plots directory
    plots_dir = Path(args.plots_dir) if args.plots_dir else result_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    # MPC-specific plots
    print("\n[MPC-Specific Plots]")
    print("\n[1/3] SOC Trajectory with Iteration Boundaries...")
    plot_mpc_iteration_boundaries(sol_df, iter_df, perf_summary, plots_dir)

    print("\n[2/3] Solver Performance...")
    plot_mpc_solver_performance(iter_df, perf_summary, plots_dir)

    print("\n[3/3] State Continuity Validation...")
    plot_mpc_soc_state_continuity(iter_df, perf_summary, plots_dir)

    # Standard optimization plots
    print("\n[Standard Optimization Plots]")

    title_suffix = f"({perf_summary['country']} {perf_summary['test_duration_days']}d, MPC α={perf_summary['alpha']})"

    print("\n[4/7] Day-Ahead Market...")
    fig_da = plot_da_market_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
    output_file = plots_dir / "da_market_price_bid.html"
    fig_da.write_html(str(output_file))
    print(f"  [OK] Saved: {output_file}")

    print("\n[5/7] aFRR Energy Market...")
    fig_afrr_e = plot_afrr_energy_market_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
    output_file = plots_dir / "afrr_energy_market_price_bid.html"
    fig_afrr_e.write_html(str(output_file))
    print(f"  [OK] Saved: {output_file}")

    print("\n[6/7] Capacity Markets...")
    fig_cap = plot_capacity_markets_price_bid(sol_df, title_suffix=title_suffix, use_timestamp=True)
    output_file = plots_dir / "capacity_markets_price_bid.html"
    fig_cap.write_html(str(output_file))
    print(f"  [OK] Saved: {output_file}")

    print("\n[7/7] SOC and Power Bids...")
    fig_soc = plot_soc_and_power_bids(sol_df, title_suffix=title_suffix, use_timestamp=True)
    output_file = plots_dir / "soc_and_power_bids.html"
    fig_soc.write_html(str(output_file))
    print(f"  [OK] Saved: {output_file}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {plots_dir}")
    print(f"\nTotal plots generated: 7")
    print("  • 3 MPC-specific plots")
    print("  • 4 standard optimization plots")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
