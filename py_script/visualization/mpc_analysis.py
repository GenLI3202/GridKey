"""
MPC Analysis Visualizations
============================

This module provides specialized visualization functions for analyzing Model Predictive Control
(MPC) simulation results. These plots complement the standard optimization visualizations by
focusing on MPC-specific aspects such as:

- Iteration boundaries and rolling horizon windows
- State continuity across MPC iterations
- Per-iteration performance metrics
- Solver performance over time

These visualizations help validate MPC implementation correctness and diagnose issues with
state propagation, window overlaps, and computational performance.

Functions
---------
- plot_iteration_boundaries: Show SOC trajectory with MPC iteration markers
- plot_iteration_performance: Bar charts of revenue/cost per iteration
- plot_state_continuity: Validate SOC continuity between iterations

Example
-------
>>> from py_script.mpc.mpc_simulator import MPCSimulator
>>> from py_script.mpc.transform_mpc_results import extract_iteration_summary
>>> from py_script.visualization.mpc_analysis import (
...     plot_iteration_boundaries,
...     plot_iteration_performance,
...     plot_state_continuity
... )
>>>
>>> # Run MPC simulation
>>> results = simulator.run_full_simulation(0.5)
>>>
>>> # Create MPC-specific plots
>>> fig1 = plot_iteration_boundaries(results, execution_hours=24)
>>> fig2 = plot_iteration_performance(results)
>>> fig3 = plot_state_continuity(results)
"""

from __future__ import annotations

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..visualization.config import MCKINSEY_COLORS, MCKINSEY_FONTS, WATERFALL_COLORS, WATERFALL_STYLE


def plot_iteration_boundaries(
    mpc_results: Dict,
    execution_hours: int = 24,
    battery_capacity_kwh: float = 4472.0,
    title_suffix: str = "",
    show_horizons: bool = False
) -> go.Figure:
    """
    Plot SOC trajectory with vertical lines marking MPC iteration boundaries.

    This visualization shows the continuous SOC trajectory across the entire simulation
    period with vertical markers indicating where each MPC iteration starts and ends.
    Optionally, can show the prediction horizon extent for each iteration.

    Parameters
    ----------
    mpc_results : dict
        Results dictionary from MPCSimulator.run_full_simulation()
        Required keys:
        - soc_15min: list of SOC values at 15-min resolution (kWh)
        - iteration_results: list of dict with per-iteration metrics
        - soc_trajectory: list of SOC values at iteration boundaries

    execution_hours : int, optional
        Execution window size in hours (default: 24)

    battery_capacity_kwh : float, optional
        Battery capacity for SOC percentage calculation (default: 4472)

    title_suffix : str, optional
        Additional text to append to plot title

    show_horizons : bool, optional
        If True, show shaded regions for each horizon window (default: False)
        Useful for understanding lookahead vs execution windows

    Returns
    -------
    go.Figure
        Plotly figure with SOC trajectory and iteration boundaries

    Example
    -------
    >>> results = simulator.run_full_simulation(0.5)
    >>> fig = plot_iteration_boundaries(results, execution_hours=24, show_horizons=True)
    >>> fig.write_html('validation_results/mpc_iteration_boundaries.html')
    """
    # Extract data
    soc_15min = mpc_results['soc_15min']
    iteration_results = mpc_results['iteration_results']
    timesteps = list(range(len(soc_15min)))
    hours = [t * 0.25 for t in timesteps]  # 15-min → hours

    # Calculate SOC percentage
    soc_pct = [(soc / battery_capacity_kwh) * 100 for soc in soc_15min]

    # Create figure
    fig = go.Figure()

    # Add SOC trajectory
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=soc_pct,
            mode='lines',
            name='SOC',
            line=dict(color=MCKINSEY_COLORS['navy'], width=2),
            hovertemplate='Hour %{x:.2f}<br>SOC: %{y:.1f}%<extra></extra>'
        )
    )

    # Calculate execution window size in timesteps
    execution_timesteps = execution_hours * 4  # 15-min intervals

    # Add iteration boundaries
    for i, iter_result in enumerate(iteration_results):
        # Get start timestep for this iteration
        start_ts = i * execution_timesteps

        # Add vertical line at iteration start
        fig.add_vline(
            x=start_ts * 0.25,
            line=dict(color=MCKINSEY_COLORS['negative'], width=1, dash='dash'),
            annotation_text=f"Iter {i}",
            annotation_position="top",
            annotation=dict(font=dict(size=10, color=MCKINSEY_COLORS['negative']))
        )

        # Optionally show horizon windows
        if show_horizons and 'horizon_hours' in iter_result:
            horizon_hours = iter_result['horizon_hours']
            horizon_timesteps = horizon_hours * 4
            end_ts = start_ts + horizon_timesteps

            # Add shaded region for horizon
            fig.add_vrect(
                x0=start_ts * 0.25,
                x1=min(end_ts * 0.25, max(hours)),
                fillcolor=MCKINSEY_COLORS['gray_light'],
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text=f"{horizon_hours}h horizon" if i == 0 else "",
                annotation_position="top left"
            )

    # Add final boundary
    if iteration_results:
        final_ts = len(iteration_results) * execution_timesteps
        if final_ts < len(timesteps):
            fig.add_vline(
                x=final_ts * 0.25,
                line=dict(color=MCKINSEY_COLORS['negative'], width=1, dash='dash'),
                annotation_text=f"Iter {len(iteration_results)}",
                annotation_position="top",
                annotation=dict(font=dict(size=10, color=MCKINSEY_COLORS['negative']))
            )

    # Update layout
    title_text = f"MPC SOC Trajectory with Iteration Boundaries"
    if title_suffix:
        title_text += f" {title_suffix}"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=MCKINSEY_FONTS['title_size'], family=MCKINSEY_FONTS['family'])),
        xaxis_title="Time (hours)",
        yaxis_title="State of Charge (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        height=500
    )

    return fig


def plot_iteration_performance(
    mpc_results: Dict,
    title_suffix: str = "",
    show_cumulative: bool = True
) -> go.Figure:
    """
    Plot per-iteration revenue, degradation cost, and profit as bar charts.

    This visualization helps understand the financial performance of each MPC iteration
    and identify any anomalies or patterns over time.

    Parameters
    ----------
    mpc_results : dict
        Results dictionary from MPCSimulator.run_full_simulation()
        Required keys:
        - iteration_results: list of dict with per-iteration metrics

    title_suffix : str, optional
        Additional text to append to plot title

    show_cumulative : bool, optional
        If True, add cumulative profit line overlay (default: True)

    Returns
    -------
    go.Figure
        Plotly figure with stacked bars for revenue/cost and optional cumulative line

    Example
    -------
    >>> results = simulator.run_full_simulation(0.5)
    >>> fig = plot_iteration_performance(results, show_cumulative=True)
    >>> fig.write_html('validation_results/mpc_iteration_performance.html')
    """
    from ..mpc.transform_mpc_results import extract_iteration_summary

    # Extract iteration summary
    iter_df = extract_iteration_summary(mpc_results, include_soc_trajectory=False)

    if iter_df.empty:
        raise ValueError("No iteration results found in mpc_results")

    # Create figure with secondary y-axis if showing cumulative
    if show_cumulative:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Waterfall-style stacking: Revenue components (positive) + Cost components (negative)
    # This creates an intuitive visualization where net = revenue stacks - cost stacks

    # Check if detailed breakdown is available AND has non-zero values
    # (extract_iteration_summary always adds these columns with 0.0 defaults)
    has_detailed_breakdown = (
        all(col in iter_df.columns for col in [
            'da_discharge_revenue', 'da_charge_cost', 'cyclic_cost', 'calendar_cost'
        ]) and
        # Check if ANY of the detailed columns have non-zero values
        (iter_df['da_discharge_revenue'].abs().sum() > 0.01 or
         iter_df['da_charge_cost'].abs().sum() > 0.01 or
         iter_df['cyclic_cost'].abs().sum() > 0.01 or
         iter_df['calendar_cost'].abs().sum() > 0.01)
    )

    if has_detailed_breakdown:
        # DETAILED BREAKDOWN (NEW FORMAT) - McKinsey Blue Gradient Theme
        # Reference: https://www.mckinsey.com/industries/energy-and-materials/our-insights/global-energy-perspective#/
        # POSITIVE REVENUE COMPONENTS (stacked above zero) - Blues with 85% opacity
        # 1. DA Discharge Revenue
        fig.add_trace(
            go.Bar(
                x=iter_df['iteration'],
                y=iter_df['da_discharge_revenue'],
                name='DA Discharge',
                marker_color=WATERFALL_COLORS['revenue_primary'],
                hovertemplate='Iteration %{x}<br>DA Discharge: €%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False if show_cumulative else None
        )

        # 2. FCR Capacity Revenue
        if 'fcr_revenue' in iter_df.columns:
            fig.add_trace(
                go.Bar(
                    x=iter_df['iteration'],
                    y=iter_df['fcr_revenue'],
                    name='FCR Capacity',
                    marker_color=WATERFALL_COLORS['revenue_secondary'],
                    hovertemplate='Iteration %{x}<br>FCR: €%{y:,.0f}<extra></extra>'
                ),
                secondary_y=False if show_cumulative else None
            )

        # 3. aFRR Energy Revenue
        if 'afrr_e_revenue' in iter_df.columns:
            fig.add_trace(
                go.Bar(
                    x=iter_df['iteration'],
                    y=iter_df['afrr_e_revenue'],
                    name='aFRR Energy',
                    marker_color=WATERFALL_COLORS['revenue_tertiary'],
                    hovertemplate='Iteration %{x}<br>aFRR Energy: €%{y:,.0f}<extra></extra>'
                ),
                secondary_y=False if show_cumulative else None
            )

        # NEGATIVE COST COMPONENTS (stacked below zero) - Full opacity for costs
        # 4. DA Charge Cost (negative)
        fig.add_trace(
            go.Bar(
                x=iter_df['iteration'],
                y=-iter_df['da_charge_cost'],
                name='DA Charge Cost',
                marker_color=WATERFALL_COLORS['cost_primary'],
                hovertemplate='Iteration %{x}<br>DA Charge: -€%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False if show_cumulative else None
        )

        # 5. Cyclic Aging Cost (negative)
        fig.add_trace(
            go.Bar(
                x=iter_df['iteration'],
                y=-iter_df['cyclic_cost'],
                name='Cyclic Aging',
                marker_color=WATERFALL_COLORS['cost_secondary'],
                hovertemplate='Iteration %{x}<br>Cyclic Aging: -€%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False if show_cumulative else None
        )

        # 6. Calendar Aging Cost (negative)
        fig.add_trace(
            go.Bar(
                x=iter_df['iteration'],
                y=-iter_df['calendar_cost'],
                name='Calendar Aging',
                marker_color=WATERFALL_COLORS['cost_tertiary'],
                hovertemplate='Iteration %{x}<br>Calendar Aging: -€%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False if show_cumulative else None
        )

    else:
        # FALLBACK: AGGREGATED FORMAT (OLD DATA - for backward compatibility)
        # Show aggregated revenue and degradation cost bars
        fig.add_trace(
            go.Bar(
                x=iter_df['iteration'],
                y=iter_df['revenue'],
                name='Revenue (Total)',
                marker_color=WATERFALL_COLORS['total_revenue'],
                hovertemplate='Iteration %{x}<br>Revenue: €%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False if show_cumulative else None
        )

        fig.add_trace(
            go.Bar(
                x=iter_df['iteration'],
                y=-iter_df['degradation_cost'],
                name='Degradation Cost',
                marker_color=WATERFALL_COLORS['total_cost'],
                hovertemplate='Iteration %{x}<br>Degradation: -€%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False if show_cumulative else None
        )

    # NOTE: Net profit is NOT stacked (shown as cumulative line only)
    # Total height of positive stack - total height of negative stack = profit

    # Add cumulative profit line
    if show_cumulative:
        cumulative_profit = iter_df['profit'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=iter_df['iteration'],
                y=cumulative_profit,
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color=WATERFALL_COLORS['cumulative_line'], width=WATERFALL_STYLE['line_width']),
                marker=dict(size=WATERFALL_STYLE['marker_size'], color=WATERFALL_COLORS['cumulative_marker']),
                hovertemplate='Iteration %{x}<br>Cumulative: €%{y:,.0f}<extra></extra>'
            ),
            secondary_y=True
        )

    # Update layout
    title_text = "MPC Per-Iteration Financial Performance"
    if title_suffix:
        title_text += f" {title_suffix}"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=MCKINSEY_FONTS['title_size'], family=MCKINSEY_FONTS['family'])),
        xaxis_title="MPC Iteration",
        barmode='relative',
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        height=500
    )

    if show_cumulative:
        fig.update_yaxes(title_text="Per-Iteration Financial (EUR)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Profit (EUR)", secondary_y=True)
    else:
        fig.update_yaxes(title_text="Financial (EUR)")

    return fig


def plot_state_continuity(
    mpc_results: Dict,
    battery_capacity_kwh: float = 4472.0,
    title_suffix: str = "",
    tolerance_pct: float = 0.1
) -> go.Figure:
    """
    Validate SOC continuity between MPC iterations.

    This plot shows the SOC at the end of each iteration's execution window
    (which becomes the initial SOC for the next iteration) to verify smooth
    state propagation without jumps or discontinuities.

    Parameters
    ----------
    mpc_results : dict
        Results dictionary from MPCSimulator.run_full_simulation()
        Required keys:
        - soc_trajectory: list of SOC values at iteration boundaries (kWh)

    battery_capacity_kwh : float, optional
        Battery capacity for SOC percentage calculation (default: 4472)

    title_suffix : str, optional
        Additional text to append to plot title

    tolerance_pct : float, optional
        Tolerance for flagging discontinuities as percentage of capacity (default: 0.1%)
        If consecutive SOC values differ by more than this, mark as potential issue

    Returns
    -------
    go.Figure
        Plotly figure showing SOC at iteration boundaries with continuity validation

    Example
    -------
    >>> results = simulator.run_full_simulation(0.5)
    >>> fig = plot_state_continuity(results, tolerance_pct=0.1)
    >>> fig.write_html('validation_results/mpc_state_continuity.html')

    Notes
    -----
    - Green markers: smooth transition (within tolerance)
    - Red markers: potential discontinuity (exceeds tolerance)
    - SOC should be continuous - any red markers indicate potential MPC implementation issues
    """
    if 'soc_trajectory' not in mpc_results:
        raise ValueError("mpc_results missing 'soc_trajectory' key. Ensure MPC simulation includes SOC tracking.")

    soc_trajectory = mpc_results['soc_trajectory']

    if not soc_trajectory or len(soc_trajectory) < 2:
        raise ValueError("Need at least 2 SOC trajectory points to check continuity")

    # Calculate SOC percentage
    soc_pct = [(soc / battery_capacity_kwh) * 100 for soc in soc_trajectory]
    iterations = list(range(len(soc_trajectory)))

    # Calculate SOC differences
    soc_diff_pct = [0] + [abs(soc_pct[i] - soc_pct[i-1]) for i in range(1, len(soc_pct))]

    # Identify discontinuities
    is_continuous = [diff <= tolerance_pct for diff in soc_diff_pct]

    # Create figure
    fig = go.Figure()

    # Add SOC trajectory line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=soc_pct,
            mode='lines+markers',
            name='SOC at Iteration Boundary',
            line=dict(color=MCKINSEY_COLORS['navy'], width=2),
            marker=dict(size=10, color=[
                MCKINSEY_COLORS['positive'] if cont else MCKINSEY_COLORS['negative']
                for cont in is_continuous
            ]),
            hovertemplate='Iteration %{x}<br>SOC: %{y:.2f}%<br>Diff: %{customdata:.3f}%<extra></extra>',
            customdata=soc_diff_pct
        )
    )

    # Add tolerance band
    mean_soc = np.mean(soc_pct)
    fig.add_hline(
        y=mean_soc,
        line=dict(color=MCKINSEY_COLORS['gray_medium'], width=1, dash='dot'),
        annotation_text=f"Mean SOC: {mean_soc:.1f}%",
        annotation_position="right"
    )

    # Update layout
    title_text = f"MPC State Continuity Validation (Tolerance: {tolerance_pct}%)"
    if title_suffix:
        title_text += f" {title_suffix}"

    # Count discontinuities
    n_discontinuities = sum(1 for cont in is_continuous if not cont) - 1  # Exclude first point
    status_text = "✓ All transitions smooth" if n_discontinuities == 0 else f"⚠ {n_discontinuities} potential discontinuities"

    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sub>{status_text}</sub>",
            font=dict(size=MCKINSEY_FONTS['title_size'], family=MCKINSEY_FONTS['family'])
        ),
        xaxis_title="MPC Iteration",
        yaxis_title="State of Charge (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        height=500,
        annotations=[
            dict(
                text="<b>Color Code:</b> Green = Smooth | Red = Discontinuity",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                xanchor='center', yanchor='top',
                showarrow=False,
                font=dict(size=10, color=MCKINSEY_COLORS['gray_medium'])
            )
        ]
    )

    return fig


def plot_solver_performance(
    mpc_results: Dict,
    title_suffix: str = "",
    show_status: bool = True
) -> go.Figure:
    """
    Plot solver performance metrics across MPC iterations.

    This visualization shows solve times and solver status for each iteration,
    helping identify computational bottlenecks or solver issues.

    Parameters
    ----------
    mpc_results : dict
        Results dictionary from MPCSimulator.run_full_simulation()
        Required keys:
        - iteration_results: list of dict with per-iteration metrics
          Each dict should contain 'solve_time' and optionally 'solver_status'

    title_suffix : str, optional
        Additional text to append to plot title

    show_status : bool, optional
        If True and solver status available, show status markers (default: True)

    Returns
    -------
    go.Figure
        Plotly figure showing solver performance over iterations

    Example
    -------
    >>> results = simulator.run_full_simulation(0.5)
    >>> fig = plot_solver_performance(results, show_status=True)
    >>> fig.write_html('validation_results/mpc_solver_performance.html')

    Notes
    -----
    Requires 'solve_time' in iteration_results. If not available, raises ValueError.
    """
    from ..mpc.transform_mpc_results import extract_iteration_summary

    # Extract iteration summary
    iter_df = extract_iteration_summary(mpc_results, include_soc_trajectory=False)

    if 'solve_time' not in iter_df.columns:
        raise ValueError("No solver timing data found in iteration_results")

    # Create figure
    fig = go.Figure()

    # Add solve time bars
    fig.add_trace(
        go.Bar(
            x=iter_df['iteration'],
            y=iter_df['solve_time'],
            name='Solve Time',
            marker_color=MCKINSEY_COLORS['navy'],
            hovertemplate='Iteration %{x}<br>Solve Time: %{y:.2f}s<extra></extra>'
        )
    )

    # Add mean line
    mean_time = iter_df['solve_time'].mean()
    fig.add_hline(
        y=mean_time,
        line=dict(color=MCKINSEY_COLORS['dark_blue'], width=2, dash='dash'),
        annotation_text=f"Mean: {mean_time:.2f}s",
        annotation_position="right"
    )

    # Update layout
    title_text = "MPC Solver Performance"
    if title_suffix:
        title_text += f" {title_suffix}"

    total_time = iter_df['solve_time'].sum()
    subtitle = f"Total solve time: {total_time:.1f}s ({total_time/60:.1f} min)"

    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sub>{subtitle}</sub>",
            font=dict(size=MCKINSEY_FONTS['title_size'], family=MCKINSEY_FONTS['family'])
        ),
        xaxis_title="MPC Iteration",
        yaxis_title="Solve Time (seconds)",
        template="plotly_white",
        hovermode='x unified',
        showlegend=False,
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        height=400
    )

    return fig
