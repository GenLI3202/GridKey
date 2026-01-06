"""
Visualization functions for battery segment LIFO analysis.

This module provides comprehensive plotting tools to analyze and validate
LIFO (Last-In-First-Out) behavior in segmented battery SOC models.

Key visualizations:
1. Segment energy stacked timeline
2. Segment energy deltas (detect parallel charging/discharging)
3. SOC trajectory with power flows
4. Violation annotations and highlights
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def detect_parallel_operations(
    df: pd.DataFrame,
    epsilon: float = 0.1
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect parallel charging and discharging violations.

    Args:
        df: DataFrame with segment_1...segment_10 columns
        epsilon: Tolerance for detecting energy changes (kWh)

    Returns:
        Tuple of (parallel_charging_events, parallel_discharging_events)
    """
    segment_cols = [f'segment_{i}' for i in range(1, 11)]

    parallel_charging = []
    parallel_discharging = []

    for t in range(1, len(df)):
        # Calculate deltas
        deltas = {}
        for i, col in enumerate(segment_cols):
            delta = df[col].iloc[t] - df[col].iloc[t-1]
            if abs(delta) > epsilon:
                deltas[i+1] = delta  # 1-indexed segment number

        # Find charging segments
        charging = {seg: d for seg, d in deltas.items() if d > epsilon}
        if len(charging) > 1:
            parallel_charging.append({
                'timestep': t,
                'hour': df['hour'].iloc[t],
                'segments': list(charging.keys()),
                'deltas': list(charging.values()),
                'soc': df['soc_kwh'].iloc[t]
            })

        # Find discharging segments
        discharging = {seg: abs(d) for seg, d in deltas.items() if d < -epsilon}
        if len(discharging) > 1:
            parallel_discharging.append({
                'timestep': t,
                'hour': df['hour'].iloc[t],
                'segments': list(discharging.keys()),
                'deltas': list(discharging.values()),
                'soc': df['soc_kwh'].iloc[t]
            })

    return parallel_charging, parallel_discharging


def plot_segment_lifo_analysis(
    csv_path: str,
    output_path: Optional[str] = None,
    show_violations: bool = True,
    plot_style: str = 'stacked',  # 'stacked' or 'individual'
    epsilon: float = 0.1,
    width: int = 1400,
    height: int = 1200
) -> go.Figure:
    """
    Create comprehensive LIFO analysis visualization.

    Args:
        csv_path: Path to solution_timeseries.csv
        output_path: Where to save HTML plot (None = auto-generate)
        show_violations: Highlight parallel charging/discharging events
        plot_style: 'stacked' for area chart, 'individual' for line traces
        epsilon: Tolerance for violation detection (kWh)
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    # Load data
    df = pd.read_csv(csv_path)
    segment_cols = [f'segment_{i}' for i in range(1, 11)]
    E_seg = 447.2  # Segment capacity

    # Detect violations
    if show_violations:
        parallel_ch, parallel_dis = detect_parallel_operations(df, epsilon)
    else:
        parallel_ch, parallel_dis = [], []

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'Segment Energy Over Time ({plot_style.capitalize()} View)',
            'Segment Energy Deltas (Δ per timestep)',
            'Total SOC Trajectory',
            'Power Flows (Charge/Discharge)'
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ],
        vertical_spacing=0.08,
        row_heights=[0.35, 0.25, 0.2, 0.2]
    )

    # Color scheme
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]

    # ========================================================================
    # Plot 1: Segment Energy (Stacked or Individual)
    # ========================================================================
    if plot_style == 'stacked':
        # Stacked area chart
        for i, col in enumerate(segment_cols):
            fig.add_trace(
                go.Scatter(
                    x=df['hour'],
                    y=df[col],
                    name=f'Segment {i+1}',
                    mode='lines',
                    stackgroup='segments',
                    fillcolor=colors[i],
                    line=dict(width=0.5, color=colors[i]),
                    hovertemplate=(
                        f'<b>Segment {i+1}</b><br>'
                        'Hour: %{x:.2f}<br>'
                        'Energy: %{y:.2f} kWh<br>'
                        f'Capacity: {E_seg:.2f} kWh<br>'
                        'Fill: %{customdata:.1f}%'
                        '<extra></extra>'
                    ),
                    customdata=df[col] / E_seg * 100,
                    legendgroup='segments',
                    showlegend=True
                ),
                row=1, col=1
            )
    else:
        # Individual line traces
        for i, col in enumerate(segment_cols):
            fig.add_trace(
                go.Scatter(
                    x=df['hour'],
                    y=df[col],
                    name=f'Segment {i+1}',
                    mode='lines',
                    line=dict(width=2, color=colors[i]),
                    hovertemplate=(
                        f'<b>Segment {i+1}</b><br>'
                        'Hour: %{x:.2f}<br>'
                        'Energy: %{y:.2f} kWh<br>'
                        f'Capacity: {E_seg:.2f} kWh<br>'
                        'Fill: %{customdata:.1f}%'
                        '<extra></extra>'
                    ),
                    customdata=df[col] / E_seg * 100,
                    legendgroup='segments',
                    showlegend=True
                ),
                row=1, col=1
            )

        # Add segment capacity reference line
        fig.add_hline(
            y=E_seg,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Segment Capacity ({E_seg} kWh)",
            annotation_position="right",
            row=1, col=1
        )

    # ========================================================================
    # Plot 2: Segment Energy Deltas
    # ========================================================================
    for i, col in enumerate(segment_cols):
        delta = df[col].diff().fillna(0)

        # Color markers by delta sign
        marker_colors = ['green' if d > epsilon else 'red' if d < -epsilon else 'lightgray'
                        for d in delta]

        fig.add_trace(
            go.Scatter(
                x=df['hour'],
                y=delta,
                name=f'Δ Seg {i+1}',
                mode='markers',
                marker=dict(
                    size=5,
                    color=marker_colors,
                    line=dict(width=0.5, color=colors[i])
                ),
                hovertemplate=(
                    f'<b>Segment {i+1} Delta</b><br>'
                    'Hour: %{x:.2f}<br>'
                    'ΔEnergy: %{y:.2f} kWh'
                    '<extra></extra>'
                ),
                legendgroup='deltas',
                showlegend=False
            ),
            row=2, col=1
        )

    # Add zero reference line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1,
                  opacity=0.3, row=2, col=1)

    # Highlight parallel charging events
    if show_violations and parallel_ch:
        violation_hours = [v['hour'] for v in parallel_ch]
        fig.add_trace(
            go.Scatter(
                x=violation_hours,
                y=[0] * len(violation_hours),
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='red',
                    line=dict(width=2)
                ),
                name='Parallel Charging',
                hovertemplate='<b>PARALLEL CHARGING</b><br>Hour: %{x:.2f}<extra></extra>',
                legendgroup='violations',
                showlegend=True
            ),
            row=2, col=1
        )

    # ========================================================================
    # Plot 3: Total SOC
    # ========================================================================
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['soc_kwh'],
            name='Total SOC',
            mode='lines',
            line=dict(width=3, color='black'),
            fill='tozeroy',
            fillcolor='rgba(0,0,0,0.1)',
            hovertemplate=(
                '<b>State of Charge</b><br>'
                'Hour: %{x:.2f}<br>'
                'SOC: %{y:.2f} kWh<br>'
                'SOC: %{customdata:.1f}%'
                '<extra></extra>'
            ),
            customdata=df['soc_pct'],
            legendgroup='soc',
            showlegend=True
        ),
        row=3, col=1
    )

    # Add battery capacity reference
    battery_capacity = 4472.0
    fig.add_hline(
        y=battery_capacity,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text=f"Battery Capacity ({battery_capacity} kWh)",
        annotation_position="right",
        row=3, col=1
    )

    # ========================================================================
    # Plot 4: Power Flows
    # ========================================================================
    # Charging power (positive)
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['p_total_ch_kw'],
            name='Charge Power',
            mode='lines',
            line=dict(width=2, color='green'),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)',
            hovertemplate=(
                '<b>Charging</b><br>'
                'Hour: %{x:.2f}<br>'
                'Power: %{y:.2f} kW'
                '<extra></extra>'
            ),
            legendgroup='power',
            showlegend=True
        ),
        row=4, col=1
    )

    # Discharging power (negative for visual clarity)
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=-df['p_total_dis_kw'],
            name='Discharge Power',
            mode='lines',
            line=dict(width=2, color='red'),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)',
            hovertemplate=(
                '<b>Discharging</b><br>'
                'Hour: %{x:.2f}<br>'
                'Power: %{y:.2f} kW (absolute)'
                '<extra></extra>'
            ),
            legendgroup='power',
            showlegend=True
        ),
        row=4, col=1
    )

    # Add zero reference line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1,
                  opacity=0.3, row=4, col=1)

    # ========================================================================
    # Update Axes
    # ========================================================================
    fig.update_xaxes(title_text="Hour", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_xaxes(title_text="Hour", row=3, col=1)
    fig.update_xaxes(title_text="Hour", row=4, col=1)

    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="ΔEnergy (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="SOC (kWh)", row=3, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=4, col=1)

    # ========================================================================
    # Layout and Annotations
    # ========================================================================
    title_text = "Battery Segment LIFO Analysis"
    if show_violations:
        n_violations = len(parallel_ch) + len(parallel_dis)
        if n_violations > 0:
            title_text += f" - {len(parallel_ch)} Parallel Charging, {len(parallel_dis)} Parallel Discharging Violations"
        else:
            title_text += " - ✓ No Violations Detected"

    fig.update_layout(
        height=height,
        width=width,
        title_text=title_text,
        title_font_size=16,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )

    # Add annotation with violation summary
    if show_violations and (parallel_ch or parallel_dis):
        annotation_text = "<b>Violations Detected:</b><br>"
        if parallel_ch:
            annotation_text += f"• {len(parallel_ch)} Parallel Charging events<br>"
        if parallel_dis:
            annotation_text += f"• {len(parallel_dis)} Parallel Discharging events<br>"
        annotation_text += "<br>See delta plot (Panel 2) for details"

        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255,200,200,0.8)",
            bordercolor="red",
            borderwidth=2,
            font=dict(size=10, color="darkred")
        )

    # ========================================================================
    # Save and Return
    # ========================================================================
    if output_path is None:
        output_path = Path(csv_path).parent / "segment_lifo_analysis.html"

    fig.write_html(str(output_path))
    print(f"\n{'='*80}")
    print(f"Visualization saved to: {output_path}")
    print(f"{'='*80}")

    # Print violation summary
    if show_violations:
        print(f"\nViolation Summary:")
        print(f"  Parallel Charging:     {len(parallel_ch)}")
        print(f"  Parallel Discharging:  {len(parallel_dis)}")
        print(f"  Total:                 {len(parallel_ch) + len(parallel_dis)}")

        if parallel_ch:
            print(f"\nParallel Charging Events:")
            for i, v in enumerate(parallel_ch[:5], 1):
                print(f"  {i}. Hour {v['hour']:.2f}: Segments {v['segments']} "
                      f"({[f'{d:.1f} kWh' for d in v['deltas']]})")
            if len(parallel_ch) > 5:
                print(f"  ... and {len(parallel_ch)-5} more")

    return fig


def plot_segment_comparison(
    csv_paths: List[str],
    labels: List[str],
    output_path: Optional[str] = None,
    width: int = 1400,
    height: int = 800
) -> go.Figure:
    """
    Compare segment behavior across multiple optimization runs.

    Args:
        csv_paths: List of paths to solution_timeseries.csv files
        labels: Labels for each run (e.g., ['Sequential ON', 'Sequential OFF'])
        output_path: Where to save HTML plot
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly Figure object
    """
    n_runs = len(csv_paths)

    fig = make_subplots(
        rows=n_runs, cols=2,
        subplot_titles=[f'{label} - Segment Energy' for label in labels] +
                       [f'{label} - Deltas' for label in labels],
        specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in range(n_runs)],
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for run_idx, (csv_path, label) in enumerate(zip(csv_paths, labels), 1):
        df = pd.read_csv(csv_path)
        segment_cols = [f'segment_{i}' for i in range(1, 11)]

        # Plot stacked segments (left column)
        for i, col in enumerate(segment_cols):
            show_legend = (run_idx == 1)  # Only show legend for first run
            fig.add_trace(
                go.Scatter(
                    x=df['hour'],
                    y=df[col],
                    name=f'Seg {i+1}',
                    mode='lines',
                    stackgroup='segments',
                    fillcolor=colors[i],
                    line=dict(width=0.5, color=colors[i]),
                    showlegend=show_legend,
                    legendgroup=f'seg{i+1}'
                ),
                row=run_idx, col=1
            )

        # Plot deltas (right column)
        for i, col in enumerate(segment_cols):
            delta = df[col].diff().fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=df['hour'],
                    y=delta,
                    name=f'Δ{i+1}',
                    mode='markers',
                    marker=dict(size=4, color=colors[i]),
                    showlegend=False
                ),
                row=run_idx, col=2
            )

        # Add zero line to delta plot
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                     opacity=0.5, row=run_idx, col=2)

    # Update axes
    for i in range(1, n_runs + 1):
        fig.update_xaxes(title_text="Hour", row=i, col=1)
        fig.update_xaxes(title_text="Hour", row=i, col=2)
        fig.update_yaxes(title_text="Energy (kWh)", row=i, col=1)
        fig.update_yaxes(title_text="ΔEnergy (kWh)", row=i, col=2)

    fig.update_layout(
        height=height * n_runs / 2,
        width=width,
        title_text="Segment LIFO Comparison Across Runs",
        showlegend=True,
        hovermode='x unified'
    )

    if output_path:
        fig.write_html(str(output_path))
        print(f"Comparison plot saved to: {output_path}")

    return fig


# =============================================================================
# Command-line interface
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize battery segment LIFO behavior and detect violations'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to solution_timeseries.csv file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output HTML path (default: auto-generate in same directory)'
    )
    parser.add_argument(
        '--style',
        type=str,
        choices=['stacked', 'individual'],
        default='stacked',
        help='Plot style: stacked area or individual lines'
    )
    parser.add_argument(
        '--no-violations',
        action='store_true',
        help='Disable violation detection and highlighting'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
        help='Tolerance for violation detection (kWh), default=0.1'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1400,
        help='Plot width in pixels, default=1400'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1200,
        help='Plot height in pixels, default=1200'
    )

    args = parser.parse_args()

    # Create visualization
    fig = plot_segment_lifo_analysis(
        csv_path=args.csv_path,
        output_path=args.output,
        show_violations=not args.no_violations,
        plot_style=args.style,
        epsilon=args.epsilon,
        width=args.width,
        height=args.height
    )

    print("\n✓ Visualization complete!")
