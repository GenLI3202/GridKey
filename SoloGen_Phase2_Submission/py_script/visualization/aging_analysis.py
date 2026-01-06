#!/usr/bin/env python3
"""
Aging Analysis Visualizations
==============================

This module provides specialized visualization functions for validating and
analyzing degradation model behaviors from BESSOptimizerModelIII.

The plots focus on:
- Cyclic aging: Visualizing the "stacked tank" segmented SOC model (10 segments)
- Calendar aging: Validating the SOS2 piecewise-linear cost function (5 breakpoints)

These visualizations are essential for debugging the degradation models and
verifying that the optimizer's aging cost calculations are correct.

Author: TechArena 2025 Team
Date: November 2025
"""

from __future__ import annotations

from typing import Dict, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..visualization.config import MCKINSEY_COLORS, MCKINSEY_FONTS

# Constants
BATTERY_CAPACITY = 4472  # kWh
NUM_SEGMENTS = 10
SEGMENT_CAPACITY = BATTERY_CAPACITY / NUM_SEGMENTS  # 447.2 kWh per segment


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _extract_segment_soc_from_solution(solution: Dict, num_timesteps: int) -> pd.DataFrame:
    """
    Extract segmented SOC data from solution dictionary.

    Args:
        solution: Solution dictionary from optimizer.extract_solution()
        num_timesteps: Number of time steps in the optimization horizon

    Returns:
        DataFrame with columns: time_step, segment_1, segment_2, ..., segment_10
    """
    # Extract e_soc_j from solution dictionary
    # Format: e_soc_j[(t, j)] = value (from BESSOptimizerModelII/III)
    e_soc_j = solution.get('e_soc_j', {})

    # Initialize data structure
    data = {'time_step': list(range(num_timesteps))}

    # Extract each segment
    for j in range(1, NUM_SEGMENTS + 1):
        segment_col = f'segment_{j}'
        data[segment_col] = [e_soc_j.get((t, j), 0.0) for t in range(num_timesteps)]

    return pd.DataFrame(data)


def _extract_calendar_data_from_solution(solution: Dict, num_timesteps: int) -> pd.DataFrame:
    """
    Extract calendar aging data from solution dictionary.

    Args:
        solution: Solution dictionary from optimizer.extract_solution()
        num_timesteps: Number of time steps in the optimization horizon

    Returns:
        DataFrame with columns: time_step, soc_kwh, cal_cost_eur_hr
    """
    # Extract total SOC and calendar costs from solution dict
    # Format: e_soc[t] = value, c_cal_cost[t] = value (from BESSOptimizerModelIII)
    e_soc = solution.get('e_soc', {})
    c_cal_cost = solution.get('c_cal_cost', {})

    # Build DataFrame
    data = {
        'time_step': list(range(num_timesteps)),
        'soc_kwh': [e_soc.get(t, 0.0) for t in range(num_timesteps)],
        'cal_cost_eur_hr': [c_cal_cost.get(t, 0.0) for t in range(num_timesteps)]
    }

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Main Plotting Functions
# ---------------------------------------------------------------------------

def plot_stacked_cyclic_soc(
    data: Dict[str, any],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Validate the "stacked tank" logic of the 10 cyclic SOC segments.

    This function creates a stacked area chart showing the energy stored in each
    of the 10 cyclic aging segments over time. The visualization helps validate:
    - Segment ordering (j=1 is bottom, j=10 is top)
    - Stacked tank constraint (e_soc_j[t,j] >= e_soc_j[t,j+1])
    - Total SOC = sum of all segments

    Parameters
    ----------
    data : dict or pd.DataFrame
        PREFERRED: Solution dictionary from optimizer.extract_solution()
        LEGACY: DataFrame from extract_detailed_solution() with segment columns

    title_suffix : str, optional
        Additional text to append to plot title (e.g., "HU Winter 36h")

    save_path : str, optional
        Path to save the plot as an HTML file. If None, plot is not saved.

    Returns
    -------
    go.Figure
        Plotly figure with stacked area chart

    Examples
    --------
    >>> # PREFERRED: Use solution dict directly
    >>> solution = optimizer.extract_solution(model, results)
    >>> fig = plot_stacked_cyclic_soc(solution, title_suffix="HU Winter 36h")
    >>> fig.show()
    >>>
    >>> # LEGACY: Use DataFrame (for backward compatibility)
    >>> df = extract_detailed_solution(solution, test_data, 36)
    >>> fig = plot_stacked_cyclic_soc(df, title_suffix="HU Winter 36h")
    >>> fig.show()
    """
    # Handle both dict and DataFrame inputs
    if isinstance(data, dict):
        # Extract from solution dictionary
        e_soc = data.get('e_soc', {})
        if not e_soc:
            raise ValueError("Solution dict missing 'e_soc' key")
        num_timesteps = len(e_soc)

        # Extract segment data
        df = _extract_segment_soc_from_solution(data, num_timesteps)

        # Add hour column for x-axis
        df['hour'] = [t * 0.25 for t in range(num_timesteps)]

    else:
        # Assume it's a DataFrame (backward compatibility)
        df = data

        # Check if segment columns exist
        segment_cols = [f'segment_{j}' for j in range(1, NUM_SEGMENTS + 1)]
        if not all(col in df.columns for col in segment_cols):
            raise ValueError(
                f"DataFrame missing segment columns. Expected: {segment_cols}\n"
                f"Found columns: {list(df.columns)}"
            )

    # Determine x-axis column
    x_col = 'timestamp' if 'timestamp' in df.columns else 'hour'
    x_values = df[x_col].values
    x_title = 'Time' if x_col == 'timestamp' else 'Hour'

    # Create figure
    fig = go.Figure()

    # Generate color palette for segments (gradient from bottom to top)
    # Bottom segments (j=1) are darker, top segments (j=10) are lighter
    colors = [
        f'rgb({int(0 + i*25)}, {int(63 + i*15)}, {int(92 + i*10)})'
        for i in range(NUM_SEGMENTS)
    ]

    # Add stacked area traces (bottom to top: j=1 to j=10)
    for j in range(1, NUM_SEGMENTS + 1):
        segment_col = f'segment_{j}'
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df[segment_col].values,
                mode='lines',
                name=f'Segment {j}',
                line=dict(width=0.5, color=colors[j-1]),
                fillcolor=colors[j-1],
                fill='tonexty' if j > 1 else 'tozeroy',
                stackgroup='one',
                hovertemplate=f'Segment {j}: %{{y:.2f}} kWh<extra></extra>'
            )
        )

    # Update layout
    fig.update_layout(
        title=f'Cyclic Aging: Stacked SOC Segments (10 Segments){" - " + title_suffix if title_suffix else ""}',
        xaxis_title=x_title,
        yaxis_title='Energy in Segment (kWh)',
        template='mckinsey',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            font=dict(size=9)
        ),
        margin=dict(l=80, r=150, t=100, b=60),
        height=600
    )

    # Add annotations
    fig.add_annotation(
        text="Segment ordering: j=1 (bottom/darkest) → j=10 (top/lightest)",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color=MCKINSEY_COLORS['gray_dark']),
        xanchor='center'
    )

    # Add total SOC line on top for reference
    total_soc = df[[f'segment_{j}' for j in range(1, NUM_SEGMENTS + 1)]].sum(axis=1)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=total_soc.values,
            mode='lines',
            name='Total SOC',
            line=dict(color='black', width=2, dash='dash'),
            hovertemplate='Total SOC: %{y:.2f} kWh<extra></extra>',
            showlegend=True
        )
    )

    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Stacked cyclic SOC plot saved to: {save_path}")

    return fig


def plot_calendar_aging_curve(
    data: Dict[str, any],
    aging_config: Optional[Dict] = None,
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Validate the SOS2 piecewise-linear cost function for calendar aging.

    This function creates a 2D scatter plot showing the relationship between
    total SOC and calendar aging cost. The plot should trace the N-point convex
    curve defined in the aging_config.json, validating that:
    - The SOS2 variables correctly interpolate between breakpoints
    - The piecewise-linear approximation is accurate
    - Higher SOC → Higher calendar cost (as expected)

    Parameters
    ----------
    data : dict or pd.DataFrame
        PREFERRED: Solution dictionary from optimizer.extract_solution()
        LEGACY: DataFrame with 'soc_kwh' and 'cal_cost_eur_hr' columns

    aging_config : dict, optional
        Aging configuration dictionary with 'calendar_aging']['breakpoints'].
        If provided, breakpoints will be overlaid on the plot for comparison.

    title_suffix : str, optional
        Additional text to append to plot title

    save_path : str, optional
        Path to save the plot as an HTML file. If None, plot is not saved.

    Returns
    -------
    go.Figure
        Plotly figure with scatter plot showing SOC vs. calendar cost

    Examples
    --------
    >>> import json
    >>> # PREFERRED: Use solution dict directly
    >>> solution = optimizer.extract_solution(model, results)
    >>>
    >>> # Load aging config for breakpoint overlay
    >>> with open('data/p2_config/aging_config.json') as f:
    >>>     aging_config = json.load(f)
    >>>
    >>> fig = plot_calendar_aging_curve(solution, aging_config, title_suffix="HU Winter")
    >>> fig.show()
    """
    # Handle both dict and DataFrame inputs
    if isinstance(data, dict):
        # Extract from solution dictionary
        e_soc = data.get('e_soc', {})
        c_cal_cost = data.get('c_cal_cost', {})

        if not e_soc:
            raise ValueError("Solution dict missing 'e_soc' key")
        if not c_cal_cost:
            raise ValueError("Solution dict missing 'c_cal_cost' key (Model III required)")

        num_timesteps = len(e_soc)

        # Extract calendar aging data
        df = _extract_calendar_data_from_solution(data, num_timesteps)

        soc_col = 'soc_kwh'
        cal_cost_col = 'cal_cost_eur_hr'

    else:
        # Assume it's a DataFrame (backward compatibility)
        df = data

        # Extract SOC and calendar cost columns
        if 'soc_kwh' in df.columns:
            soc_col = 'soc_kwh'
        elif 'e_soc' in df.columns:
            soc_col = 'e_soc'
        else:
            raise ValueError(
                "DataFrame missing SOC column. Expected 'soc_kwh' or 'e_soc'\n"
                f"Found columns: {list(df.columns)}"
            )

        # Find calendar cost column
        cal_cost_col = None
        for col in ['cal_cost_eur_hr', 'c_cal_cost', 'calendar_cost']:
            if col in df.columns:
                cal_cost_col = col
                break

        if cal_cost_col is None:
            raise ValueError(
                "DataFrame missing calendar cost column.\n"
                f"Expected one of: 'cal_cost_eur_hr', 'c_cal_cost', 'calendar_cost'\n"
                f"Found columns: {list(df.columns)}"
            )

    # Extract data
    soc_kwh = df[soc_col].values
    cal_cost = df[cal_cost_col].values

    # Create figure
    fig = go.Figure()

    # Add scatter plot of actual optimization results
    fig.add_trace(
        go.Scatter(
            x=soc_kwh,
            y=cal_cost,
            mode='markers',
            name='Optimization Results',
            marker=dict(
                size=6,
                color=MCKINSEY_COLORS['navy'],
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='SOC: %{x:.2f} kWh<br>Cost: %{y:.4f} EUR/hr<extra></extra>'
        )
    )

    # Overlay theoretical breakpoints if config provided
    if aging_config is not None:
        breakpoints = aging_config.get('calendar_aging', {}).get('breakpoints', [])

        if breakpoints:
            bp_soc = [bp['soc_kwh'] for bp in breakpoints]
            bp_cost = [bp['cost_eur_hr'] for bp in breakpoints]

            # Add breakpoint markers
            fig.add_trace(
                go.Scatter(
                    x=bp_soc,
                    y=bp_cost,
                    mode='markers+lines',
                    name='Theoretical Breakpoints',
                    marker=dict(
                        size=10,
                        color=MCKINSEY_COLORS['teal'],
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    line=dict(
                        color=MCKINSEY_COLORS['teal'],
                        width=2,
                        dash='dot'
                    ),
                    hovertemplate='Breakpoint<br>SOC: %{x:.0f} kWh<br>Cost: %{y:.2f} EUR/hr<extra></extra>'
                )
            )

    # Update layout
    fig.update_layout(
        title=f'Calendar Aging: SOS2 Piecewise-Linear Cost Function{" - " + title_suffix if title_suffix else ""}',
        xaxis_title='Total State of Charge (kWh)',
        yaxis_title='Calendar Aging Cost (EUR/hr)',
        template='mckinsey',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=0.02
        ),
        height=600
    )

    # Add annotations
    fig.add_annotation(
        text="Expected: Convex piecewise-linear curve with higher cost at higher SOC",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color=MCKINSEY_COLORS['gray_dark']),
        xanchor='center'
    )

    # Add grid for easier reading
    fig.update_xaxes(
        showgrid=True,
        gridcolor=MCKINSEY_COLORS['gray_light'],
        range=[0, BATTERY_CAPACITY * 1.05]
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=MCKINSEY_COLORS['gray_light']
    )

    # Save if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Calendar aging curve plot saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def plot_aging_validation_suite(
    solution: Dict,
    aging_config: Optional[Dict] = None,
    output_dir: str = ".",
    test_name: str = "aging_validation"
) -> Dict[str, go.Figure]:
    """
    Generate complete aging validation plot suite.

    This convenience function generates both cyclic and calendar aging plots
    from a solution dictionary and saves them to the specified directory.

    Parameters
    ----------
    solution : dict
        Solution dictionary from optimizer.extract_solution()
        Must be from BESSOptimizerModelII or ModelIII (contains e_soc_j)
    aging_config : dict, optional
        Aging configuration for breakpoint overlay on calendar curve.
        If None and using Model III, calendar plot will show data without breakpoints.
    output_dir : str, optional
        Directory to save plots (default: current directory)
    test_name : str, optional
        Name prefix for saved files

    Returns
    -------
    dict
        Dictionary mapping plot names to Plotly figures:
        - 'cyclic_soc': Stacked SOC segment plot (if e_soc_j present)
        - 'calendar_curve': Calendar aging cost curve (if c_cal_cost present)

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>>
    >>> # Run optimization with Model III
    >>> optimizer = BESSOptimizerModelIII(alpha=1.0)
    >>> model = optimizer.build_model(test_data, horizon_hours=36)
    >>> solved_model, results = optimizer.solve_model(model)
    >>> solution = optimizer.extract_solution(solved_model, results)
    >>>
    >>> # Load aging config for breakpoint overlay
    >>> with open('data/p2_config/aging_config.json') as f:
    >>>     aging_config = json.load(f)
    >>>
    >>> # Generate all plots (no need for test_data or horizon_hours!)
    >>> figs = plot_aging_validation_suite(
    >>>     solution,
    >>>     aging_config=aging_config,
    >>>     output_dir="results/hu_winter",
    >>>     test_name="hu_winter_36h"
    >>> )
    """
    from pathlib import Path

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Dictionary to store figures
    figures = {}

    # 1. Stacked Cyclic SOC Plot (Model II and III)
    if 'e_soc_j' in solution and solution['e_soc_j']:
        print(f"\nGenerating cyclic SOC plot...")
        cyclic_path = Path(output_dir) / f"{test_name}_cyclic_soc.html"
        fig_cyclic = plot_stacked_cyclic_soc(
            solution,  # Pass solution dict directly!
            title_suffix=test_name.replace('_', ' ').title(),
            save_path=str(cyclic_path)
        )
        figures['cyclic_soc'] = fig_cyclic
    else:
        print(f"\nSkipping cyclic SOC plot (no segment data - Model I?)")

    # 2. Calendar Aging Curve Plot (Model III only)
    if 'c_cal_cost' in solution and solution['c_cal_cost']:
        print(f"Generating calendar aging curve plot...")
        calendar_path = Path(output_dir) / f"{test_name}_calendar_curve.html"
        fig_calendar = plot_calendar_aging_curve(
            solution,  # Pass solution dict directly!
            aging_config=aging_config,
            title_suffix=test_name.replace('_', ' ').title(),
            save_path=str(calendar_path)
        )
        figures['calendar_curve'] = fig_calendar
    else:
        print(f"Skipping calendar aging curve (no calendar cost data - Model I or II?)")

    print(f"\nAging validation plots saved to: {output_dir}")

    return figures


# ---------------------------------------------------------------------------
# Module Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Aging Analysis Visualization Module")
    print("=" * 70)
    print("\nThis module provides degradation model validation plots:")
    print("  1. plot_stacked_cyclic_soc() - Validates 10-segment SOC stacking")
    print("  2. plot_calendar_aging_curve() - Validates SOS2 calendar aging")
    print("\nUsage:")
    print("  from py_script.visualization.aging_analysis import plot_stacked_cyclic_soc")
    print("  fig = plot_stacked_cyclic_soc(df, save_path='cyclic_validation.html')")
    print("\nSee function docstrings for detailed examples.")
