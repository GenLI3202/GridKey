#!/usr/bin/env python3
"""
Model I/II/III Comparison Visualization Suite
==============================================

This module provides comprehensive comparison visualizations for analyzing
the impact of degradation modeling across BESS optimizer Models I, II, and III.

Key Features:
- 8 professional McKinsey-styled comparison plots
- Financial performance analysis (waterfall, revenue breakdown, heatmap)
- Operational behavior comparison (SOC, power dispatch, segment distribution)
- Performance metrics (degradation, computational complexity)
- Dual output format (interactive HTML + presentation-ready PNG)

Author: SoloGen Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import McKinsey styling
from .config import (
    MCKINSEY_COLORS,
    COUNTRY_COLORS,
    WATERFALL_COLORS,
    MCKINSEY_FONTS,
    apply_mckinsey_style
)

# Import existing utilities
from .plot_mpc_saved_results import load_mpc_results
from .optimization_analysis import (
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid
)


# ============================================================================
# Helper Functions
# ============================================================================

def _save_figure(fig: go.Figure, save_path: str, width: int = 1200, height: int = 500):
    """Helper function to save figure as HTML and PNG with directory creation."""
    if not save_path:
        return

    # Create output directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save HTML
    fig.write_html(save_path)

    # Save PNG (optional, may fail if kaleido not installed)
    png_path = save_path.replace('.html', '.png')
    try:
        fig.write_image(png_path, width=width, height=height)
    except Exception as e:
        print(f"   [WARNING] Could not save PNG: {e}")


# ============================================================================
# Data Loading
# ============================================================================

def load_model_results(model_dirs: Dict[str, str]) -> Dict[str, Dict]:
    """
    Load results from multiple model validation runs.

    Parameters
    ----------
    model_dirs : dict
        Dictionary mapping model names to result directories.
        Example: {'I': 'path/to/model_i', 'II': 'path/to/model_ii', 'III': 'path/to/model_iii'}

    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            'I': {'performance': dict, 'timeseries': DataFrame},
            'II': {...},
            'III': {...}
        }
    """
    results = {}

    for model_name, result_dir in model_dirs.items():
        result_path = Path(result_dir)

        if not result_path.exists():
            raise FileNotFoundError(f"Result directory not found: {result_dir}")

        # Load performance summary JSON
        perf_file = result_path / "performance_summary.json"
        if not perf_file.exists():
            raise FileNotFoundError(f"Performance summary not found: {perf_file}")

        with open(perf_file, 'r') as f:
            perf_dict = json.load(f)

        # Load solution timeseries CSV
        sol_file = result_path / "solution_timeseries.csv"
        if not sol_file.exists():
            raise FileNotFoundError(f"Solution timeseries not found: {sol_file}")

        sol_df = pd.read_csv(sol_file)

        results[model_name] = {
            'performance': perf_dict,
            'timeseries': sol_df
        }

        print(f"[OK] Loaded Model {model_name}: {len(sol_df)} timesteps, "
              f"Profit: €{perf_dict.get('total_profit_eur', 0):.2f}")

    return results


# ============================================================================
# Plot 1: Profit Waterfall Comparison
# ============================================================================

def plot_profit_waterfall_comparison(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create side-by-side waterfall charts comparing profit components across models.

    Shows revenue sources (positive) and costs (negative) leading to net profit
    for each model.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text (e.g., "CZ - 36h")
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 3 waterfall subplots
    """
    # Create 1x3 subplot grid
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Model {m}" for m in results.keys()],
        horizontal_spacing=0.08
    )

    model_colors = {'I': MCKINSEY_COLORS['navy'],
                    'II': MCKINSEY_COLORS['dark_blue'],
                    'III': MCKINSEY_COLORS['teal']}

    for idx, (model_name, data) in enumerate(results.items(), start=1):
        perf = data['performance']
        sol_df = data['timeseries']

        # Calculate revenue components
        revenue_da = sol_df['revenue_da_eur'].sum() if 'revenue_da_eur' in sol_df.columns else 0
        revenue_fcr = perf.get('revenue_components', {}).get('revenue_fcr_eur', 0)
        revenue_afrr_cap = perf.get('revenue_components', {}).get('revenue_afrr_capacity_eur', 0)
        revenue_afrr_energy = sol_df['revenue_afrr_energy_eur'].sum() if 'revenue_afrr_energy_eur' in sol_df.columns else 0

        # Calculate cost components
        cost_da_charge = -sol_df[sol_df['revenue_da_eur'] < 0]['revenue_da_eur'].sum() if 'revenue_da_eur' in sol_df.columns else 0
        cost_cyclic = perf.get('degradation_metrics', {}).get('total_cyclic_cost_eur', 0)
        cost_calendar = perf.get('degradation_metrics', {}).get('total_calendar_cost_eur', 0)

        # Build waterfall data
        measures = []
        x_labels = []
        y_values = []

        # Revenue components (positive)
        if revenue_da > 0:
            x_labels.append('DA Discharge')
            y_values.append(revenue_da)
            measures.append('relative')

        if revenue_afrr_energy > 0:
            x_labels.append('aFRR Energy')
            y_values.append(revenue_afrr_energy)
            measures.append('relative')

        if revenue_fcr > 0:
            x_labels.append('FCR Cap.')
            y_values.append(revenue_fcr)
            measures.append('relative')

        if revenue_afrr_cap > 0:
            x_labels.append('aFRR Cap.')
            y_values.append(revenue_afrr_cap)
            measures.append('relative')

        # Cost components (negative)
        if cost_da_charge > 0:
            x_labels.append('DA Charge')
            y_values.append(-cost_da_charge)
            measures.append('relative')

        if cost_cyclic > 0:
            x_labels.append('Cyclic Aging')
            y_values.append(-cost_cyclic)
            measures.append('relative')

        if cost_calendar > 0:
            x_labels.append('Calendar Aging')
            y_values.append(-cost_calendar)
            measures.append('relative')

        # Net profit (total)
        net_profit = perf.get('total_profit_eur', 0)
        x_labels.append('Net Profit')
        y_values.append(net_profit)
        measures.append('total')

        # Create waterfall trace
        fig.add_trace(
            go.Waterfall(
                name=f"Model {model_name}",
                orientation="v",
                measure=measures,
                x=x_labels,
                y=y_values,
                connector={"line": {"color": MCKINSEY_COLORS['gray_medium'], "width": 1}},
                increasing={"marker": {"color": WATERFALL_COLORS['revenue_primary']}},
                decreasing={"marker": {"color": WATERFALL_COLORS['cost_primary']}},
                totals={"marker": {"color": model_colors[model_name]}},
                textposition="outside",
                text=[f"€{v:.0f}" for v in y_values],
                textfont={"size": 9}
            ),
            row=1, col=idx
        )

        # Update subplot y-axis
        fig.update_yaxes(title_text="EUR" if idx == 1 else "", row=1, col=idx)

    # Update layout
    title = f"Profit Waterfall Comparison"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        showlegend=False,
        height=500,
        font=dict(family=MCKINSEY_FONTS['family'])
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1400, height=500)

    return fig


# ============================================================================
# Plot 2: Revenue Components Grouped Bar Chart
# ============================================================================

def plot_revenue_components_grouped(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create grouped bar chart comparing revenue sources across models.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly grouped bar chart
    """
    fig = go.Figure()

    model_colors = {'I': MCKINSEY_COLORS['navy'],
                    'II': MCKINSEY_COLORS['dark_blue'],
                    'III': MCKINSEY_COLORS['teal']}

    # Revenue categories
    categories = ['DA Discharge', 'aFRR Energy', 'FCR Capacity', 'aFRR Capacity']

    for model_name, data in results.items():
        perf = data['performance']
        sol_df = data['timeseries']

        # Extract revenue components
        revenue_da = sol_df['revenue_da_eur'].sum() if 'revenue_da_eur' in sol_df.columns else 0
        # Only positive (discharge revenue)
        revenue_da_discharge = sol_df[sol_df['revenue_da_eur'] > 0]['revenue_da_eur'].sum() if 'revenue_da_eur' in sol_df.columns else 0

        revenue_afrr_energy = sol_df['revenue_afrr_energy_eur'].sum() if 'revenue_afrr_energy_eur' in sol_df.columns else 0
        revenue_fcr = perf.get('revenue_components', {}).get('revenue_fcr_eur', 0)
        revenue_afrr_cap = perf.get('revenue_components', {}).get('revenue_afrr_capacity_eur', 0)

        values = [revenue_da_discharge, revenue_afrr_energy, revenue_fcr, revenue_afrr_cap]

        fig.add_trace(go.Bar(
            name=f'Model {model_name}',
            x=categories,
            y=values,
            marker_color=model_colors[model_name],
            text=[f'€{v:.0f}' for v in values],
            textposition='outside'
        ))

    # Update layout
    title = "Revenue Components by Market"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        xaxis_title="Revenue Source",
        yaxis_title="Revenue (EUR)",
        barmode='group',
        template="mckinsey",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1000, height=500)

    return fig


# ============================================================================
# Plot 3: Profitability Heatmap
# ============================================================================

def plot_profitability_heatmap(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create annotated heatmap showing revenue, costs, and profit across models.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly heatmap
    """
    # Prepare data matrix
    models = list(results.keys())
    metrics = ['Total Revenue', 'Total Costs', 'Net Profit']

    data_matrix = []
    annotations = []

    for metric in metrics:
        row = []
        for model_name in models:
            perf = results[model_name]['performance']

            if metric == 'Total Revenue':
                value = perf.get('total_revenue_eur', 0)
            elif metric == 'Total Costs':
                # Calculate total costs (negative of revenue - profit)
                revenue = perf.get('total_revenue_eur', 0)
                profit = perf.get('total_profit_eur', 0)
                value = revenue - profit
            else:  # Net Profit
                value = perf.get('total_profit_eur', 0)

            row.append(value)
        data_matrix.append(row)

    # Create annotations
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            value = data_matrix[i][j]
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"€{value:.0f}",
                    showarrow=False,
                    font=dict(size=14, color='white' if value > 5000 else 'black')
                )
            )

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=[f'Model {m}' for m in models],
        y=metrics,
        colorscale='RdYlGn',  # Red (low) to Green (high)
        text=[[f"€{v:.0f}" for v in row] for row in data_matrix],
        texttemplate="%{text}",
        textfont={"size": 14},
        showscale=True,
        colorbar=dict(title="EUR")
    ))

    # Update layout
    title = "Financial Performance Matrix"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=400,
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left')
    )

    # Save if path provided
    _save_figure(fig, save_path, width=800, height=400)

    return fig


# ============================================================================
# Plot 4: SOC Trajectory Overlay
# ============================================================================

def plot_soc_trajectory_overlay(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create overlaid SOC trajectories for all models.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly line chart
    """
    fig = go.Figure()

    # Color scheme with Model III in bright blue
    model_colors = {
        'I': MCKINSEY_COLORS['navy'],      # Navy (dashed line)
        'II': '#ff0000',                    # Red (dotted line)
        'III': '#2251ff'                    # Bright blue (solid line)
    }

    # Line styles for better distinction
    model_dash = {
        'I': 'dash',       # Model I: dashed line
        'II': 'dot',       # Model II: dotted line
        'III': 'solid'     # Model III: solid line
    }

    for model_name, data in results.items():
        sol_df = data['timeseries']
        perf = data['performance']
        profit = perf.get('total_profit_eur', 0)

        fig.add_trace(go.Scatter(
            x=sol_df['hour'] if 'hour' in sol_df.columns else sol_df.index * 0.25,
            y=sol_df['soc_pct'] if 'soc_pct' in sol_df.columns else sol_df['soc_kwh'] / 4472 * 100,
            mode='lines',
            name=f'Model {model_name} (€{profit:.0f})',
            line=dict(
                color=model_colors[model_name],
                width=2.5,
                dash=model_dash[model_name]
            ),
            hovertemplate='Hour: %{x:.1f}<br>SOC: %{y:.1f}%<extra></extra>'
        ))

    # Update layout
    title = "SOC Trajectory Comparison"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        xaxis_title="Hour",
        yaxis_title="State of Charge (%)",
        template="mckinsey",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        height=500,
        hovermode='x unified'
    )

    # Add SOC range shading
    fig.add_hrect(y0=0, y1=20, fillcolor=MCKINSEY_COLORS['negative'], opacity=0.1,
                  annotation_text="Low SOC", annotation_position="left")
    fig.add_hrect(y0=80, y1=100, fillcolor=MCKINSEY_COLORS['positive'], opacity=0.1,
                  annotation_text="High SOC", annotation_position="left")

    # Save if path provided
    _save_figure(fig, save_path, width=1200, height=500)

    return fig


# ============================================================================
# Plot 5: Power Dispatch Comparison
# ============================================================================

def plot_power_dispatch_comparison(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create 3-subplot comparison of power dispatch patterns.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 3 subplots
    """
    # Create 1x3 subplot grid
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Model {m}" for m in results.keys()],
        horizontal_spacing=0.06
    )

    for idx, (model_name, data) in enumerate(results.items(), start=1):
        sol_df = data['timeseries']

        # Get hour values
        x_vals = sol_df['hour'] if 'hour' in sol_df.columns else sol_df.index * 0.25

        # DA Discharge (positive)
        if 'p_dis_kw' in sol_df.columns:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=sol_df['p_dis_kw'],
                mode='lines',
                name='DA Discharge',
                line=dict(color=WATERFALL_COLORS['revenue_primary'], width=0),
                fill='tozeroy',
                fillcolor=WATERFALL_COLORS['revenue_primary'],
                showlegend=(idx == 1),
                stackgroup='positive'
            ), row=1, col=idx)

        # aFRR Energy Positive
        if 'p_afrr_pos_e_kw' in sol_df.columns:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=sol_df['p_afrr_pos_e_kw'],
                mode='lines',
                name='aFRR Pos Energy',
                line=dict(color=WATERFALL_COLORS['revenue_tertiary'], width=0),
                fill='tozeroy',
                fillcolor=WATERFALL_COLORS['revenue_tertiary'],
                showlegend=(idx == 1),
                stackgroup='positive'
            ), row=1, col=idx)

        # DA Charge (negative)
        if 'p_ch_kw' in sol_df.columns:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=-sol_df['p_ch_kw'],
                mode='lines',
                name='DA Charge',
                line=dict(color=WATERFALL_COLORS['cost_primary'], width=0),
                fill='tozeroy',
                fillcolor=WATERFALL_COLORS['cost_primary'],
                showlegend=(idx == 1),
                stackgroup='negative'
            ), row=1, col=idx)

        # aFRR Energy Negative
        if 'p_afrr_neg_e_kw' in sol_df.columns:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=-sol_df['p_afrr_neg_e_kw'],
                mode='lines',
                name='aFRR Neg Energy',
                line=dict(color=WATERFALL_COLORS['cost_tertiary'], width=0),
                fill='tozeroy',
                fillcolor=WATERFALL_COLORS['cost_tertiary'],
                showlegend=(idx == 1),
                stackgroup='negative'
            ), row=1, col=idx)

        # Update subplot axes
        fig.update_xaxes(title_text="Hour" if idx == 2 else "", row=1, col=idx)
        fig.update_yaxes(title_text="Power (kW)" if idx == 1 else "", row=1, col=idx)

    # Update layout
    title = "Power Dispatch Patterns"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1400, height=500)

    return fig


# ============================================================================
# Plot 6: Segment Distribution Comparison (Model II vs III)
# ============================================================================

def plot_segment_distribution(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create side-by-side stacked area charts showing LIFO segment distribution.

    Only applicable for Models II and III (with degradation segments).

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 2 subplots
    """
    # Filter to only models with segments
    models_with_segments = {k: v for k, v in results.items() if k in ['II', 'III']}

    if not models_with_segments:
        raise ValueError("No models with segment data found (need Model II or III)")

    n_models = len(models_with_segments)

    # Create subplot grid (vertical layout for easier time comparison)
    # Use shared y-axis to show absolute differences between models
    fig = make_subplots(
        rows=n_models, cols=1,
        subplot_titles=[f"Model {m}" for m in models_with_segments.keys()],
        vertical_spacing=0.08,
        shared_yaxes=True
    )

    # Determine number of active segments by checking which segments have data
    # (config has 6 segments, but solution may have 10 columns with 7-10 empty)
    n_segments = 6  # Default to 6 segments (current config)

    # Verify by checking first model's data
    first_model_data = list(models_with_segments.values())[0]['timeseries']
    for seg_num in range(10, 0, -1):
        seg_col = f'segment_{seg_num}'
        if seg_col in first_model_data.columns and first_model_data[seg_col].max() > 0.01:
            n_segments = seg_num
            break

    # Color gradient for segments (dark to light)
    all_segment_colors = [
        '#003f5c', '#2f4b7c', '#3e5f8a', '#4d7298',
        '#5c85a6', '#6b98b4', '#7aabc2', '#89bed0',
        '#98d1de', '#a7e4ec'
    ]
    segment_colors = all_segment_colors[:n_segments]

    # Calculate max SOC across all models for consistent y-axis
    max_soc = 0
    for model_name, data in models_with_segments.items():
        sol_df = data['timeseries']
        # Sum all segments to get total SOC
        total_soc = 0
        for seg_num in range(1, n_segments + 1):
            seg_col = f'segment_{seg_num}'
            if seg_col in sol_df.columns:
                total_soc += sol_df[seg_col]
        if len(total_soc) > 0:
            max_soc = max(max_soc, total_soc.max())

    for idx, (model_name, data) in enumerate(models_with_segments.items(), start=1):
        sol_df = data['timeseries']

        # Get hour values
        x_vals = sol_df['hour'] if 'hour' in sol_df.columns else sol_df.index * 0.25

        # Plot active segments only (stacked)
        for seg_num in range(1, n_segments + 1):
            seg_col = f'segment_{seg_num}'
            if seg_col in sol_df.columns:
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=sol_df[seg_col],
                    mode='lines',
                    name=f'Segment {seg_num}',
                    line=dict(color=segment_colors[seg_num-1], width=0),
                    fill='tonexty' if seg_num > 1 else 'tozeroy',
                    fillcolor=segment_colors[seg_num-1],
                    showlegend=(idx == 1),
                    stackgroup='one'
                ), row=idx, col=1)

        # Update subplot axes with consistent y-axis range
        fig.update_xaxes(title_text="Hour" if idx == n_models else "", row=idx, col=1)
        fig.update_yaxes(
            title_text="SOC (kWh)",
            row=idx, col=1,
            range=[0, max_soc * 1.05]  # Add 5% margin
        )

    # Update layout
    title = "LIFO Segment Distribution"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=700 if n_models == 2 else 1000,  # Adjust height for vertical layout + legend space
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1200, height=700 if n_models == 2 else 1000)

    return fig


# ============================================================================
# Plot 7: Degradation Metrics Dashboard
# ============================================================================

def plot_degradation_metrics(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create 2x2 grid comparing degradation metrics across models.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 4 subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Equivalent Full Cycles',
            'Total Energy Throughput',
            'Average Depth of Discharge',
            'Degradation Cost Breakdown'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    model_colors = {'I': MCKINSEY_COLORS['navy'],
                    'II': MCKINSEY_COLORS['dark_blue'],
                    'III': MCKINSEY_COLORS['teal']}

    models = list(results.keys())

    # Extract metrics
    cycles = []
    throughput = []
    avg_dod = []
    cyclic_cost = []
    calendar_cost = []

    for model_name in models:
        perf = results[model_name]['performance']
        deg_metrics = perf.get('degradation_metrics', {})

        cycles.append(deg_metrics.get('equivalent_full_cycles', 0))
        throughput.append(deg_metrics.get('total_throughput_kwh', 0))
        avg_dod.append(deg_metrics.get('average_dod', 0))
        cyclic_cost.append(deg_metrics.get('total_cyclic_cost_eur', 0))
        calendar_cost.append(deg_metrics.get('total_calendar_cost_eur', 0))

    # Subplot 1: Equivalent Full Cycles
    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=cycles,
        marker_color=[model_colors[m] for m in models],
        text=[f'{c:.2f}' for c in cycles],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    fig.update_yaxes(title_text="Cycles", row=1, col=1)

    # Subplot 2: Total Throughput
    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=throughput,
        marker_color=[model_colors[m] for m in models],
        text=[f'{t/1000:.1f}k' for t in throughput],
        textposition='outside',
        showlegend=False
    ), row=1, col=2)
    fig.update_yaxes(title_text="kWh", row=1, col=2)

    # Subplot 3: Average DoD
    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=avg_dod,
        marker_color=[model_colors[m] for m in models],
        text=[f'{d:.2f}' for d in avg_dod],
        textposition='outside',
        showlegend=False
    ), row=2, col=1)
    fig.update_yaxes(title_text="DoD", row=2, col=1)

    # Subplot 4: Degradation Cost Breakdown (Stacked)
    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=cyclic_cost,
        name='Cyclic',
        marker_color=WATERFALL_COLORS['cost_secondary'],
        text=[f'€{c:.0f}' if c > 0 else '' for c in cyclic_cost],
        textposition='inside'
    ), row=2, col=2)

    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=calendar_cost,
        name='Calendar',
        marker_color=WATERFALL_COLORS['cost_tertiary'],
        text=[f'€{c:.0f}' if c > 0 else '' for c in calendar_cost],
        textposition='inside'
    ), row=2, col=2)

    fig.update_yaxes(title_text="EUR", row=2, col=2)
    fig.update_layout(barmode='stack')

    # Update layout
    title = "Degradation Metrics Comparison"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5
        )
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1200, height=700)

    return fig


# ============================================================================
# Plot 8: Computational Complexity Dashboard
# ============================================================================

def plot_computational_complexity(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create dashboard comparing computational requirements across models.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 2 subplots + table
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Solve Time',
            'Model Size',
            'Build Time',
            'Solver Status'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    model_colors = {'I': MCKINSEY_COLORS['navy'],
                    'II': MCKINSEY_COLORS['dark_blue'],
                    'III': MCKINSEY_COLORS['teal']}

    models = list(results.keys())

    # Extract metrics
    solve_times = []
    build_times = []
    n_variables = []
    n_constraints = []
    statuses = []

    for model_name in models:
        perf = results[model_name]['performance']

        solve_times.append(perf.get('solve_time_sec', 0))
        build_times.append(perf.get('build_time_sec', 0))
        n_variables.append(perf.get('n_variables', 0))
        n_constraints.append(perf.get('n_constraints', 0))
        statuses.append(perf.get('solver_status', 'unknown'))

    # Subplot 1: Solve Time
    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=solve_times,
        marker_color=[model_colors[m] for m in models],
        text=[f'{t:.1f}s' for t in solve_times],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    # Add margin for text labels at top
    max_solve = max(solve_times)
    fig.update_yaxes(title_text="Seconds", range=[0, max_solve * 1.15], row=1, col=1)

    # Subplot 2: Model Size (Grouped)
    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=n_variables,
        name='Variables',
        marker_color=MCKINSEY_COLORS['cat_3'],
        text=[f'{v:,}' for v in n_variables],
        textposition='outside'
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=n_constraints,
        name='Constraints',
        marker_color=MCKINSEY_COLORS['cat_4'],
        text=[f'{c:,}' for c in n_constraints],
        textposition='outside'
    ), row=1, col=2)

    # Add margin for text labels
    max_count = max(max(n_variables), max(n_constraints))
    fig.update_yaxes(title_text="Count", range=[0, max_count * 1.15], row=1, col=2)

    # Subplot 3: Build Time
    fig.add_trace(go.Bar(
        x=[f'Model {m}' for m in models],
        y=build_times,
        marker_color=[model_colors[m] for m in models],
        text=[f'{t:.3f}s' for t in build_times],
        textposition='outside',
        showlegend=False
    ), row=2, col=1)
    # Add margin for text labels
    max_build = max(build_times)
    fig.update_yaxes(title_text="Seconds", range=[0, max_build * 1.15], row=2, col=1)

    # Subplot 4: Status Table
    fig.add_trace(go.Table(
        header=dict(
            values=['Model', 'Status', 'Total Time (s)'],
            fill_color=MCKINSEY_COLORS['gray_light'],
            align='left',
            font=dict(size=12, color=MCKINSEY_COLORS['navy'])
        ),
        cells=dict(
            values=[
                [f'Model {m}' for m in models],
                statuses,
                [f'{s+b:.1f}' for s, b in zip(solve_times, build_times)]
            ],
            fill_color='white',
            align='left',
            font=dict(size=11)
        )
    ), row=2, col=2)

    # Update layout with caption
    title = "Computational Complexity Comparison"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=750,  # Increased for caption space
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.52,  # Position above Solver Status table
            xanchor="center",
            x=0.75  # Center under right column (Model Size)
        ),
        barmode='group',
        annotations=[
            dict(
                text="Solver: Gurobi 13.0.0 | Hardware: Intel i9-12900K (16 cores, 24 threads) | RAM: 64 GB",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.08,
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(size=10, color=MCKINSEY_COLORS['gray_dark']),
                align="center"
            )
        ]
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1200, height=750)

    return fig


# ============================================================================
# Plot 9: DA Market Comparison (Using Existing Utility + Model Overlay)
# ============================================================================

def plot_da_market_comparison(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create stacked DA market plots for all models using existing visualization utility.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 3 DA market subplots (row-by-row)
    """
    # Create 3x1 subplot grid (rows for easier time comparison)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f"Model {m}" for m in results.keys()],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    model_colors = {
        'I': MCKINSEY_COLORS['navy'],
        'II': '#ff0000',
        'III': '#2251ff'
    }

    for idx, (model_name, data) in enumerate(results.items(), start=1):
        sol_df = data['timeseries']

        # Use existing utility to create individual figure
        individual_fig = plot_da_market_price_bid(sol_df, title_suffix="", use_timestamp=False)

        # Extract traces from individual figure and add to subplot
        for trace in individual_fig.data:
            # Modify colors for model-specific styling if needed
            if 'Charge' in trace.name or 'Discharge' in trace.name:
                # Handle both Bar and Scatter traces
                if hasattr(trace, 'marker'):
                    trace.marker.color = model_colors[model_name]
                if hasattr(trace, 'line'):
                    trace.line.color = model_colors[model_name]

            fig.add_trace(trace, row=idx, col=1, secondary_y=('Price' in trace.name))

        # Update axes
        fig.update_xaxes(title_text="Hour" if idx == 3 else "", row=idx, col=1)
        fig.update_yaxes(title_text="Power (kW)", secondary_y=False, row=idx, col=1)
        fig.update_yaxes(title_text="Price (EUR/MWh)", secondary_y=True, row=idx, col=1)

    # Update layout
    title = "Day-Ahead Market Comparison"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        )
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1200, height=900)

    return fig


# ============================================================================
# Plot 10: Capacity Markets Comparison (Using Existing Utility)
# ============================================================================

def plot_afrr_energy_market_comparison(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create stacked aFRR energy market plots using existing visualization utility.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 3 aFRR energy market subplots (row-by-row)
    """
    # Create 3x1 subplot grid (rows for easier time comparison)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f"Model {m}" for m in results.keys()],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    model_colors = {
        'I': MCKINSEY_COLORS['navy'],
        'II': '#ff0000',
        'III': '#2251ff'
    }

    for idx, (model_name, data) in enumerate(results.items(), start=1):
        sol_df = data['timeseries']

        # Use existing utility to create individual figure
        individual_fig = plot_afrr_energy_market_price_bid(sol_df, title_suffix="", use_timestamp=False)

        # Extract traces from individual figure and add to subplot
        for trace in individual_fig.data:
            # Modify colors for model-specific styling if needed
            if 'Pos' in trace.name or 'Neg' in trace.name:
                # Handle both Bar and Scatter traces
                if hasattr(trace, 'marker'):
                    trace.marker.color = model_colors[model_name]
                if hasattr(trace, 'line'):
                    trace.line.color = model_colors[model_name]

            fig.add_trace(trace, row=idx, col=1, secondary_y=('Price' in trace.name))

        # Update axes
        fig.update_xaxes(title_text="Hour" if idx == 3 else "", row=idx, col=1)
        fig.update_yaxes(title_text="Power (kW)", secondary_y=False, row=idx, col=1)
        fig.update_yaxes(title_text="Price (EUR/MWh)", secondary_y=True, row=idx, col=1)

    # Update layout
    title = "aFRR Energy Market Comparison"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        )
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1200, height=900)

    return fig


# ============================================================================
# Plot 11: Capacity Markets Comparison (Using Existing Utility)
# ============================================================================

def plot_capacity_markets_comparison(
    results: Dict[str, Dict],
    title_suffix: str = "",
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create stacked capacity markets plots using existing visualization utility.

    Parameters
    ----------
    results : dict
        Model results from load_model_results()
    title_suffix : str, optional
        Additional title text
    save_path : str, optional
        Path to save HTML file

    Returns
    -------
    go.Figure
        Plotly figure with 3 capacity market subplots (row-by-row)
    """
    # Create 3x1 subplot grid (rows for easier time comparison)
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f"Model {m}" for m in results.keys()],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    model_colors = {
        'I': MCKINSEY_COLORS['navy'],
        'II': '#ff0000',
        'III': '#2251ff'
    }

    for idx, (model_name, data) in enumerate(results.items(), start=1):
        sol_df = data['timeseries']

        # Use existing utility to create individual figure
        individual_fig = plot_capacity_markets_price_bid(sol_df, title_suffix="", use_timestamp=False)

        # Extract traces from individual figure and add to subplot
        for trace in individual_fig.data:
            # Modify colors for model-specific styling if needed
            if 'FCR' in trace.name or 'aFRR' in trace.name:
                # Handle both Bar and Scatter traces
                if hasattr(trace, 'marker'):
                    trace.marker.color = model_colors[model_name]
                if hasattr(trace, 'line'):
                    trace.line.color = model_colors[model_name]

            fig.add_trace(trace, row=idx, col=1, secondary_y=('Price' in trace.name))

        # Update axes
        fig.update_xaxes(title_text="Hour" if idx == 3 else "", row=idx, col=1)
        fig.update_yaxes(title_text="Capacity (MW)", secondary_y=False, row=idx, col=1)
        fig.update_yaxes(title_text="Price (EUR/MW)", secondary_y=True, row=idx, col=1)

    # Update layout
    title = "Capacity Markets Comparison"
    if title_suffix:
        title += f" - {title_suffix}"

    fig.update_layout(
        title=title,
        template="mckinsey",
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.05,
            xanchor="center",
            x=0.5
        )
    )

    # Save if path provided
    _save_figure(fig, save_path, width=1200, height=1000)

    return fig


# ============================================================================
# Master Function: Generate Complete Comparison Suite
# ============================================================================

def generate_comparison_suite(
    model_dirs: Dict[str, str],
    output_dir: str,
    title_suffix: str = "",
    generate_html: bool = True,
    generate_png: bool = True
) -> Dict[str, go.Figure]:
    """
    Generate all 8 comparison plots and save to output directory.

    Parameters
    ----------
    model_dirs : dict
        Dictionary mapping model names to result directories.
        Example: {'I': 'path/to/model_i', 'II': 'path/to/model_ii', 'III': 'path/to/model_iii'}
    output_dir : str
        Directory to save all plots
    title_suffix : str, optional
        Additional title text (e.g., "CZ - 36h")
    generate_html : bool, default True
        Generate interactive HTML plots
    generate_png : bool, default True
        Generate static PNG plots

    Returns
    -------
    dict
        Dictionary of generated figures {plot_name: figure}
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"GENERATING MODEL COMPARISON SUITE")
    print(f"{'='*80}")
    print(f"Output directory: {output_path}")
    print(f"Title suffix: {title_suffix}")
    print(f"Formats: {'HTML ' if generate_html else ''}{'PNG' if generate_png else ''}")
    print(f"{'='*80}\n")

    # Load all model results
    print("[1/9] Loading model results...")
    results = load_model_results(model_dirs)

    # Generate plots
    figures = {}

    print("[2/9] Generating Plot 1: Profit Waterfall Comparison...")
    save_path = str(output_path / "01_profit_waterfall.html") if generate_html else None
    figures['waterfall'] = plot_profit_waterfall_comparison(results, title_suffix, save_path)

    print("[3/9] Generating Plot 2: Revenue Components Grouped Bar...")
    save_path = str(output_path / "02_revenue_components.html") if generate_html else None
    figures['revenue'] = plot_revenue_components_grouped(results, title_suffix, save_path)

    print("[4/9] Generating Plot 3: Profitability Heatmap...")
    save_path = str(output_path / "03_profitability_heatmap.html") if generate_html else None
    figures['heatmap'] = plot_profitability_heatmap(results, title_suffix, save_path)

    print("[5/9] Generating Plot 4: SOC Trajectory Overlay...")
    save_path = str(output_path / "04_soc_trajectory_overlay.html") if generate_html else None
    figures['soc'] = plot_soc_trajectory_overlay(results, title_suffix, save_path)

    print("[6/9] Generating Plot 5: Power Dispatch Comparison...")
    save_path = str(output_path / "05_power_dispatch_comparison.html") if generate_html else None
    figures['dispatch'] = plot_power_dispatch_comparison(results, title_suffix, save_path)

    # Only generate segment plot if Models II/III are present
    if any(m in results for m in ['II', 'III']):
        print("[7/9] Generating Plot 6: Segment Distribution...")
        save_path = str(output_path / "06_segment_distribution.html") if generate_html else None
        figures['segments'] = plot_segment_distribution(results, title_suffix, save_path)
    else:
        print("[7/9] Skipping Plot 6: No segment data (requires Model II or III)")

    print("[8/9] Generating Plot 7: Degradation Metrics Dashboard...")
    save_path = str(output_path / "07_degradation_metrics.html") if generate_html else None
    figures['degradation'] = plot_degradation_metrics(results, title_suffix, save_path)

    print("[9/9] Generating Plot 8: Computational Complexity Dashboard...")
    save_path = str(output_path / "08_computational_complexity.html") if generate_html else None
    figures['complexity'] = plot_computational_complexity(results, title_suffix, save_path)

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'title_suffix': title_suffix,
        'models': list(model_dirs.keys()),
        'model_dirs': model_dirs,
        'plots_generated': list(figures.keys()),
        'output_formats': [f for f, flag in [('HTML', generate_html), ('PNG', generate_png)] if flag],
        'model_performance': {
            model: {
                'profit_eur': results[model]['performance'].get('total_profit_eur', 0),
                'solve_time_sec': results[model]['performance'].get('solve_time_sec', 0),
                'status': results[model]['performance'].get('solver_status', 'unknown')
            }
            for model in results.keys()
        }
    }

    summary_path = output_path / "comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ COMPARISON SUITE COMPLETE!")
    print(f"{'='*80}")
    print(f"Plots generated: {len(figures)}")
    print(f"Summary saved: {summary_path}")
    print(f"{'='*80}\n")

    return figures


# ============================================================================
# Main Execution (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Model Comparison Visualization Suite")
    print("=" * 80)
    print("This module provides 8 comparison plots for analyzing Models I, II, and III.")
    print("\nUsage:")
    print("  from py_script.visualization.model_comparison import generate_comparison_suite")
    print("  ")
    print("  model_dirs = {")
    print("      'I': 'validation_results/optimizer_validation/model_i_run',")
    print("      'II': 'validation_results/optimizer_validation/model_ii_run',")
    print("      'III': 'validation_results/optimizer_validation/model_iii_run'")
    print("  }")
    print("  ")
    print("  figures = generate_comparison_suite(")
    print("      model_dirs=model_dirs,")
    print("      output_dir='validation_results/model_comparison_cz_36h',")
    print("      title_suffix='CZ - 36h'")
    print("  )")
    print("=" * 80)
