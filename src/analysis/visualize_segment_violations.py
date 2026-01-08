"""
Visualize segment energy changes to show parallel charging violations.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def visualize_segment_violations(csv_path: str, output_dir: str = None):
    """
    Create visualization showing segment energy over time and parallel charging.

    Args:
        csv_path: Path to solution_timeseries.csv
        output_dir: Where to save plots (defaults to same dir as csv)
    """
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = Path(csv_path).parent

    segment_cols = [f'segment_{i}' for i in range(1, 11)]

    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Segment Energy Over Time (Stacked View)',
            'Segment Energy Deltas (Change per Timestep)',
            'Total SOC and Power'
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}]],
        vertical_spacing=0.12,
        row_heights=[0.4, 0.3, 0.3]
    )

    # Colors for segments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plot 1: Stacked segment energy
    for i, col in enumerate(segment_cols):
        fig.add_trace(
            go.Scatter(
                x=df['hour'],
                y=df[col],
                name=f'Seg {i+1}',
                mode='lines',
                stackgroup='one',
                fillcolor=colors[i],
                line=dict(width=0.5, color=colors[i]),
                hovertemplate='<b>Segment %d</b><br>Hour: %%{x:.2f}<br>Energy: %%{y:.2f} kWh<extra></extra>' % (i+1)
            ),
            row=1, col=1
        )

    # Plot 2: Segment deltas (detect parallel charging)
    for i, col in enumerate(segment_cols):
        delta = df[col].diff().fillna(0)
        fig.add_trace(
            go.Scatter(
                x=df['hour'],
                y=delta,
                name=f'Δ Seg {i+1}',
                mode='markers+lines',
                marker=dict(size=4, color=colors[i]),
                line=dict(width=1, color=colors[i]),
                showlegend=False,
                hovertemplate='<b>Segment %d Delta</b><br>Hour: %%{x:.2f}<br>ΔE: %%{y:.2f} kWh<extra></extra>' % (i+1)
            ),
            row=2, col=1
        )

    # Plot 3: Total SOC and power
    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['soc_kwh'],
            name='SOC',
            mode='lines',
            line=dict(width=2, color='black'),
            hovertemplate='<b>SOC</b><br>Hour: %{x:.2f}<br>Energy: %{y:.2f} kWh<extra></extra>'
        ),
        row=3, col=1,
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=df['p_total_ch_kw'],
            name='P charge',
            mode='lines',
            line=dict(width=1.5, color='green'),
            hovertemplate='<b>Charge Power</b><br>Hour: %{x:.2f}<br>Power: %{y:.2f} kW<extra></extra>'
        ),
        row=3, col=1,
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=df['hour'],
            y=-df['p_total_dis_kw'],  # Negative for visual clarity
            name='P discharge',
            mode='lines',
            line=dict(width=1.5, color='red'),
            hovertemplate='<b>Discharge Power</b><br>Hour: %{x:.2f}<br>Power: %{y:.2f} kW<extra></extra>'
        ),
        row=3, col=1,
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="Hour", row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_xaxes(title_text="Hour", row=3, col=1)

    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="ΔEnergy (kWh/step)", row=2, col=1)
    fig.update_yaxes(title_text="SOC (kWh)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Power (kW)", row=3, col=1, secondary_y=True)

    # Add horizontal line at y=0 for delta plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Segment Energy Analysis - Parallel Charging Detection",
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )

    # Save
    output_path = Path(output_dir) / "segment_violation_analysis.html"
    fig.write_html(str(output_path))
    print(f"\nVisualization saved to: {output_path}")

    # Create detailed violation table
    print("\n" + "="*80)
    print("PARALLEL CHARGING DETECTION (from deltas)")
    print("="*80)

    epsilon = 0.1
    for t in range(1, len(df)):
        deltas = [(i+1, df[col].iloc[t] - df[col].iloc[t-1]) for i, col in enumerate(segment_cols)]
        charging = [(seg, d) for seg, d in deltas if d > epsilon]

        if len(charging) > 1:
            print(f"\nt={t}, hour={df['hour'].iloc[t]:.2f}")
            print(f"  Multiple segments charging: {[(s, f'{d:.2f} kWh') for s, d in charging]}")
            print(f"  Total SOC: {df['soc_kwh'].iloc[t]:.2f} kWh")

    return fig


if __name__ == '__main__':
    csv_path = r'H:\TUM-PC\TUM_CEM_PhD\a_tech_arena_hw\TechArena2025_EMS\validation_results\optimizer_validation\20251114_223822_notebook_test_modeliii_ch_24h_alpha1.0\solution_timeseries.csv'

    visualize_segment_violations(csv_path)
