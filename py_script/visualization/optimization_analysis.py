"""
Optimization Analysis Visualizations
====================================

This script provides a suite of functions for visualizing the results of the
BESS (Battery Energy Storage System) optimization model. These visualizations
are crucial for debugging, validating, and understanding the optimizer's
bidding and scheduling behavior across various electricity markets.

The plots cover:
- Battery operational schedules (charge, discharge, state of charge).
- Market participation and revenue analysis.
- Arbitrage opportunity analysis.
- Constraint validation (e.g., for Cst-8).

These functions typically consume a `solution_data` dictionary produced by the
optimizer and the corresponding market data to provide rich, contextual plots.

Important Notes
---------------
.. note::
    For degradation analysis (cyclic and calendar aging), use the dedicated
    ``py_script.visualization.aging_analysis`` module instead. The aging
    analysis functions work directly with solution dicts from
    ``optimizer.extract_solution()``, providing a simpler API without
    requiring test_data or horizon_hours.

    See: py_script.visualization.aging_analysis.plot_aging_validation_suite()
"""

from __future__ import annotations

from typing import Dict
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ..visualization.config import MCKINSEY_COLORS, MCKINSEY_FONTS

# Constants
TIMESTAMP_COL = "timestamp"


# ---------------------------------------------------------------------------
# Solution Analysis 
# ---------------------------------------------------------------------------


def extract_detailed_solution(solution: dict, test_data: pd.DataFrame, horizon_hours: int):
    """
    Extract all decision variables and market data into a DataFrame.

    .. deprecated:: 2025-11
        This is a LEGACY approach for backward compatibility.

        PREFERRED: Use the solution dict directly from optimizer.extract_solution()
        with the visualization functions in py_script.visualization.aging_analysis.

        The aging analysis functions (plot_stacked_cyclic_soc, plot_calendar_aging_curve)
        now accept solution dicts directly, eliminating the need for this intermediate
        DataFrame conversion and the test_data dependency.

        Example (PREFERRED):
            >>> solution = optimizer.extract_solution(model, results)
            >>> fig = plot_stacked_cyclic_soc(solution, title_suffix="Test")

        Example (LEGACY - still works):
            >>> df = extract_detailed_solution(solution, test_data, horizon_hours)
            >>> fig = plot_stacked_cyclic_soc(df, title_suffix="Test")

    Parameters
    ----------
    solution : dict
        Solution dictionary from optimizer.extract_solution()
    test_data : pd.DataFrame
        Market data DataFrame (required for legacy DataFrame extraction)
    horizon_hours : int
        Optimization horizon in hours

    Returns
    -------
    pd.DataFrame
        DataFrame with all decision variables and market prices
    """

    T = len(solution.get('e_soc', {}))

    # Initialize data dictionary
    data = {
        'time_step': list(range(T)),
        'hour': [t * 0.25 for t in range(T)],  # 15-min intervals
    }

    # SOC and segment data
    data['soc_kwh'] = [solution.get('e_soc', {}).get(t, 0) for t in range(T)]
    data['soc_pct'] = [solution.get('e_soc', {}).get(t, 0) / 4472 * 100 for t in range(T)]

    # Power variables
    data['p_ch_kw'] = [solution.get('p_ch', {}).get(t, 0) for t in range(T)]
    data['p_dis_kw'] = [solution.get('p_dis', {}).get(t, 0) for t in range(T)]
    data['p_total_ch_kw'] = [solution.get('p_total_ch', {}).get(t, 0) for t in range(T)]
    data['p_total_dis_kw'] = [solution.get('p_total_dis', {}).get(t, 0) for t in range(T)]

    # aFRR energy power
    data['p_afrr_pos_e_kw'] = [solution.get('p_afrr_pos_e', {}).get(t, 0) for t in range(T)]
    data['p_afrr_neg_e_kw'] = [solution.get('p_afrr_neg_e', {}).get(t, 0) for t in range(T)]

    # Binary decisions (time-indexed)
    data['y_ch'] = [solution.get('y_ch', {}).get(t, 0) for t in range(T)]
    data['y_dis'] = [solution.get('y_dis', {}).get(t, 0) for t in range(T)]
    data['y_total_ch'] = [solution.get('y_total_ch', {}).get(t, 0) for t in range(T)]
    data['y_total_dis'] = [solution.get('y_total_dis', {}).get(t, 0) for t in range(T)]

    # Block-indexed variables (need to map to time steps)
    block_map = solution.get('block_map', {})

    # Capacity bids (MW)
    data['c_fcr_mw'] = [solution.get('c_fcr', {}).get(block_map.get(t, 0), 0) for t in range(T)]
    data['c_afrr_pos_mw'] = [solution.get('c_afrr_pos', {}).get(block_map.get(t, 0), 0) for t in range(T)]
    data['c_afrr_neg_mw'] = [solution.get('c_afrr_neg', {}).get(block_map.get(t, 0), 0) for t in range(T)]

    # Binary decisions (block-indexed)
    data['y_fcr'] = [solution.get('y_fcr', {}).get(block_map.get(t, 0), 0) for t in range(T)]
    data['y_afrr_pos'] = [solution.get('y_afrr_pos', {}).get(block_map.get(t, 0), 0) for t in range(T)]
    data['y_afrr_neg'] = [solution.get('y_afrr_neg', {}).get(block_map.get(t, 0), 0) for t in range(T)]

    # Market prices (from test_data)
    if len(test_data) >= T:
        data['price_da_eur_mwh'] = test_data['price_day_ahead'].iloc[:T].values
        data['price_fcr_eur_mw'] = test_data['price_fcr'].iloc[:T].values
        # aFRR capacity prices (price_afrr_pos/neg are capacity prices)
        data['price_afrr_cap_pos_eur_mw'] = test_data['price_afrr_pos'].iloc[:T].values
        data['price_afrr_cap_neg_eur_mw'] = test_data['price_afrr_neg'].iloc[:T].values
        # aFRR energy prices
        data['price_afrr_energy_pos_eur_mwh'] = test_data['price_afrr_energy_pos'].iloc[:T].values
        data['price_afrr_energy_neg_eur_mwh'] = test_data['price_afrr_energy_neg'].iloc[:T].values

    # Cst-8 check values
    data['cst8_discharge_sum'] = [
        data['y_total_dis'][t] + data['y_fcr'][t] + data['y_afrr_neg'][t]
        for t in range(T)
    ]
    data['cst8_charge_sum'] = [
        data['y_total_ch'][t] + data['y_fcr'][t] + data['y_afrr_pos'][t]
        for t in range(T)
    ]

    # Revenue calculations (per time step)
    data['revenue_da_eur'] = [
        (data['p_dis_kw'][t] * data['price_da_eur_mwh'][t] / 1000 -
         data['p_ch_kw'][t] * data['price_da_eur_mwh'][t] / 1000) * 0.25
        if 'price_da_eur_mwh' in data else 0
        for t in range(T)
    ]

    data['revenue_afrr_energy_eur'] = [
        (data['p_afrr_pos_e_kw'][t] * data['price_afrr_energy_pos_eur_mwh'][t] / 1000 +
         data['p_afrr_neg_e_kw'][t] * data['price_afrr_energy_neg_eur_mwh'][t] / 1000) * 0.25
        if 'price_afrr_energy_pos_eur_mwh' in data else 0
        for t in range(T)
    ]

    data['revenue_as_capacity_eur'] = [
        (data['c_fcr_mw'][t] * data['price_fcr_eur_mw'][t] +
         data['c_afrr_pos_mw'][t] * data['price_afrr_cap_pos_eur_mw'][t] +
         data['c_afrr_neg_mw'][t] * data['price_afrr_cap_neg_eur_mw'][t]) * 0.25
        if 'price_fcr_eur_mw' in data else 0
        for t in range(T)
    ]

    # ========================================================================
    # Model II/III: Segment SOC data (if available)
    # ========================================================================
    e_soc_j = solution.get('e_soc_j', {})
    if e_soc_j:
        # Extract segments (j=1 to 10)
        for j in range(1, 11):
            segment_col = f'segment_{j}'
            data[segment_col] = [e_soc_j.get((t, j), 0.0) for t in range(T)]

    # ========================================================================
    # Model III: Calendar aging data (if available)
    # ========================================================================
    c_cal_cost = solution.get('c_cal_cost', {})
    if c_cal_cost:
        data['cal_cost_eur_hr'] = [c_cal_cost.get(t, 0.0) for t in range(T)]

    df = pd.DataFrame(data)

    # Add metadata
    df.attrs['horizon_hours'] = horizon_hours
    df.attrs['intervals'] = T
    df.attrs['objective_value'] = solution.get('objective_value', 0)
    df.attrs['status'] = solution.get('status', 'unknown')

    return df


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def plot_da_market_price_bid(df: pd.DataFrame, title_suffix: str = "", use_timestamp: bool = False) -> go.Figure:
    """
    Plot Day-Ahead market with prices (lines) and bids (bars).

    Parameters
    ----------
    df : pd.DataFrame
        Solution DataFrame containing power bids and prices
        Required columns:
        - 'hour' or 'timestamp': time axis
        - 'p_ch_kw', 'p_dis_kw': DA charge/discharge power (kW)
        - 'price_da_eur_mwh': DA energy price (EUR/MWh)
    title_suffix : str, optional
        Additional text to append to plot title
    use_timestamp : bool, optional
        If True, use 'timestamp' column for x-axis; otherwise use 'hour'

    Returns
    -------
    go.Figure
        Plotly figure with dual y-axes (prices left, power right)
    """
    from plotly.subplots import make_subplots

    # Prepare x-axis
    x_col = 'timestamp' if use_timestamp and 'timestamp' in df.columns else 'hour'
    x_values = df[x_col].values
    x_title = 'Time' if use_timestamp else 'Hour'

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ========================================================================
    # PRICES (Lines on left y-axis)
    # ========================================================================

    # DA Price
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['price_da_eur_mwh'].values,
            mode='lines',
            name='DA Price',
            line=dict(color=MCKINSEY_COLORS['navy'], width=2),
            hovertemplate='%{y:.2f} EUR/MWh<extra></extra>'
        ),
        secondary_y=False
    )

    # ========================================================================
    # BIDS (Bars on right y-axis, in MW)
    # ========================================================================

    # DA Charge (positive bars, above zero)
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=df['p_ch_kw'].values / 1000,  # Convert kW to MW
            name='DA Charge',
            marker_color=MCKINSEY_COLORS['navy'],
            opacity=0.6,
            hovertemplate='Charge: %{y:.3f} MW<extra></extra>'
        ),
        secondary_y=True
    )

    # DA Discharge (negative bars, below zero)
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=-df['p_dis_kw'].values / 1000,  # Negative for discharge
            name='DA Discharge',
            marker_color=MCKINSEY_COLORS['navy'],
            opacity=0.4,
            hovertemplate='Discharge: %{y:.3f} MW<extra></extra>'
        ),
        secondary_y=True
    )

    # ========================================================================
    # Layout configuration
    # ========================================================================

    fig.update_layout(
        title=f'Day-Ahead Market: Prices & Bids {title_suffix}',
        xaxis_title=x_title,
        hovermode='x unified',
        barmode='relative',  # Stacks positive/negative bars
        template='plotly_white',
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=600,
        width=1200
    )

    # Set y-axes titles
    fig.update_yaxes(
        title_text="Price (EUR/MWh)",
        secondary_y=False,
        showgrid=True,
        gridcolor=MCKINSEY_COLORS['gray_light']
    )
    fig.update_yaxes(
        title_text="Power (MW)",
        secondary_y=True,
        showgrid=False
    )

    # Add zero line for right y-axis
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=MCKINSEY_COLORS['gray_dark'],
        line_width=1,
        opacity=0.5,
        secondary_y=True
    )

    return fig

def plot_afrr_energy_market_price_bid(df: pd.DataFrame, title_suffix: str = "", use_timestamp: bool = False) -> go.Figure:
    """
    Plot aFRR Energy market with prices (lines) and bids (bars).
    Both aFRR+ and aFRR- bids are shown as positive values.

    Parameters
    ----------
    df : pd.DataFrame
        Solution DataFrame containing power bids and prices
        Required columns:
        - 'hour' or 'timestamp': time axis
        - 'p_afrr_pos_e_kw', 'p_afrr_neg_e_kw': aFRR energy power (kW)
        - 'price_afrr_energy_pos_eur_mwh', 'price_afrr_energy_neg_eur_mwh': aFRR energy prices (EUR/MWh)
    title_suffix : str, optional
        Additional text to append to plot title
    use_timestamp : bool, optional
        If True, use 'timestamp' column for x-axis; otherwise use 'hour'

    Returns
    -------
    go.Figure
        Plotly figure with dual y-axes (prices left, power right)
    """
    from plotly.subplots import make_subplots

    # Prepare x-axis
    x_col = 'timestamp' if use_timestamp and 'timestamp' in df.columns else 'hour'
    x_values = df[x_col].values
    x_title = 'Time' if use_timestamp else 'Hour'

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ========================================================================
    # PRICES (Lines on left y-axis)
    # ========================================================================

    # aFRR+ Energy Price
    if 'price_afrr_energy_pos_eur_mwh' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df['price_afrr_energy_pos_eur_mwh'].values,
                mode='lines',
                name='aFRR+ Energy Price',
                line=dict(color=MCKINSEY_COLORS['positive'], width=2),
                hovertemplate='%{y:.2f} EUR/MWh<extra></extra>'
            ),
            secondary_y=False
        )

    # aFRR- Energy Price
    if 'price_afrr_energy_neg_eur_mwh' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df['price_afrr_energy_neg_eur_mwh'].values,
                mode='lines',
                name='aFRR- Energy Price',
                line=dict(color=MCKINSEY_COLORS['negative'], width=2),
                hovertemplate='%{y:.2f} EUR/MWh<extra></extra>'
            ),
            secondary_y=False
        )

    # ========================================================================
    # BIDS (Bars on right y-axis, in MW) - BOTH SHOWN AS POSITIVE
    # ========================================================================

    # aFRR+ Energy (positive)
    if 'p_afrr_pos_e_kw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df['p_afrr_pos_e_kw'].values / 1000,  # Positive value
                name='aFRR+ Energy Bid',
                marker_color=MCKINSEY_COLORS['positive'],
                opacity=0.6,
                hovertemplate='aFRR+ Energy: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    # aFRR- Energy (also positive - no negation)
    if 'p_afrr_neg_e_kw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df['p_afrr_neg_e_kw'].values / 1000,  # Positive value (no minus sign)
                name='aFRR- Energy Bid',
                marker_color=MCKINSEY_COLORS['negative'],
                opacity=0.6,
                hovertemplate='aFRR- Energy: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    # ========================================================================
    # Layout configuration
    # ========================================================================

    fig.update_layout(
        title=f'aFRR Energy Market: Prices & Bids {title_suffix}',
        xaxis_title=x_title,
        hovermode='x unified',
        barmode='stack',  # Stack both positive bars
        template='plotly_white',
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=600,
        width=1200
    )

    # Set y-axes titles
    fig.update_yaxes(
        title_text="Price (EUR/MWh)",
        secondary_y=False,
        showgrid=True,
        gridcolor=MCKINSEY_COLORS['gray_light']
    )
    fig.update_yaxes(
        title_text="Power (MW)",
        secondary_y=True,
        showgrid=False
    )

    return fig

def plot_soc_and_power_bids(df: pd.DataFrame, title_suffix: str = "", use_timestamp: bool = False) -> go.Figure:
    """
    Plot SOC trajectory (line) with power bids (stacked bars).

    Positive bars (above zero): DA charge + aFRR- energy (both charging - stacked)
    Negative bars (below zero): DA discharge + aFRR+ energy (both discharging - stacked)
    SOC shown as line on left y-axis (both kWh and %)

    Parameters
    ----------
    df : pd.DataFrame
        Solution DataFrame containing SOC and power bids
        Required columns:
        - 'hour' or 'timestamp': time axis
        - 'soc_kwh', 'soc_pct': State of charge in kWh and %
        - 'p_ch_kw', 'p_dis_kw': DA charge/discharge power (kW)
        - 'p_afrr_pos_e_kw', 'p_afrr_neg_e_kw': aFRR energy power (kW)
    title_suffix : str, optional
        Additional text to append to plot title
    use_timestamp : bool, optional
        If True, use 'timestamp' column for x-axis; otherwise use 'hour'

    Returns
    -------
    go.Figure
        Plotly figure with dual y-axes (SOC left, power right)
    """
    from plotly.subplots import make_subplots

    # Prepare x-axis
    x_col = 'timestamp' if use_timestamp and 'timestamp' in df.columns else 'hour'
    x_values = df[x_col].values
    x_title = 'Time' if use_timestamp else 'Hour'

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ========================================================================
    # SOC (Line on left y-axis)
    # ========================================================================

    # SOC in kWh (primary SOC line)
    if 'soc_kwh' in df.columns and 'soc_pct' in df.columns:
        # Create custom hover text with both kWh and %
        hover_text = [f'SOC: {kwh:.2f} kWh ({pct:.1f}%)'
                      for kwh, pct in zip(df['soc_kwh'].values, df['soc_pct'].values)]

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df['soc_kwh'].values,
                mode='lines',
                name='SOC',
                line=dict(color=MCKINSEY_COLORS['dark_blue'], width=3),
                hovertext=hover_text,
                hovertemplate='%{hovertext}<extra></extra>'
            ),
            secondary_y=False
        )

    # ========================================================================
    # POWER BIDS (Bars on right y-axis, in MW)
    # ========================================================================

    # Positive bars (stacked): DA Charge + aFRR- Energy (both charging)
    if 'p_ch_kw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df['p_ch_kw'].values / 1000,  # Convert kW to MW
                name='DA Charge',
                marker_color=MCKINSEY_COLORS['navy'],
                opacity=0.7,
                hovertemplate='DA Charge: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    if 'p_afrr_neg_e_kw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df['p_afrr_neg_e_kw'].values / 1000,  # Positive (charging)
                name='aFRR- Energy',
                marker_color=MCKINSEY_COLORS['negative'],
                opacity=0.7,
                hovertemplate='aFRR- Energy: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    # Negative bars (stacked): DA Discharge + aFRR+ Energy (both discharging)
    if 'p_dis_kw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=-df['p_dis_kw'].values / 1000,  # Negative for discharge
                name='DA Discharge',
                marker_color=MCKINSEY_COLORS['navy'],
                opacity=0.5,
                hovertemplate='DA Discharge: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    if 'p_afrr_pos_e_kw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=-df['p_afrr_pos_e_kw'].values / 1000,  # Negative (discharging)
                name='aFRR+ Energy',
                marker_color=MCKINSEY_COLORS['positive'],
                opacity=0.7,
                hovertemplate='aFRR+ Energy: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    # ========================================================================
    # Layout configuration
    # ========================================================================

    fig.update_layout(
        title=f'Battery Schedule: SOC & Power Bids {title_suffix}',
        xaxis_title=x_title,
        hovermode='x unified',
        barmode='relative',  # Stack positive and negative bars separately
        template='plotly_white',
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=600,
        width=1200
    )

    # Set y-axes titles
    fig.update_yaxes(
        title_text="SOC (kWh)",
        secondary_y=False,
        showgrid=True,
        gridcolor=MCKINSEY_COLORS['gray_light']
    )
    fig.update_yaxes(
        title_text="Power (MW)",
        secondary_y=True,
        showgrid=False
    )

    # Add zero line for power axis (now on secondary y-axis)
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color=MCKINSEY_COLORS['gray_dark'],
        line_width=1,
        opacity=0.5,
        secondary_y=True
    )

    return fig

def plot_capacity_markets_price_bid(df: pd.DataFrame, title_suffix: str = "", use_timestamp: bool = False) -> go.Figure:
    """
    Plot capacity markets (FCR and aFRR capacity) with prices (lines) and bids (bars).

    Parameters
    ----------
    df : pd.DataFrame
        Solution DataFrame containing capacity bids and prices
        Required columns:
        - 'hour' or 'timestamp': time axis
        - 'c_fcr_mw': FCR capacity bid (MW)
        - 'c_afrr_pos_mw', 'c_afrr_neg_mw': aFRR capacity bids (MW)
        - 'price_fcr_eur_mw': FCR capacity price (EUR/MW)
        - 'price_afrr_cap_pos_eur_mw', 'price_afrr_cap_neg_eur_mw': aFRR capacity prices (EUR/MW)
    title_suffix : str, optional
        Additional text to append to plot title
    use_timestamp : bool, optional
        If True, use 'timestamp' column for x-axis; otherwise use 'hour'

    Returns
    -------
    go.Figure
        Plotly figure with dual y-axes (prices left, capacity right)
    """
    from plotly.subplots import make_subplots

    # Prepare x-axis
    x_col = 'timestamp' if use_timestamp and 'timestamp' in df.columns else 'hour'
    x_values = df[x_col].values
    x_title = 'Time' if use_timestamp else 'Hour'

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ========================================================================
    # PRICES (Lines on left y-axis)
    # ========================================================================

    # FCR Capacity Price
    if 'price_fcr_eur_mw' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df['price_fcr_eur_mw'].values,
                mode='lines',
                name='FCR Price',
                line=dict(color=MCKINSEY_COLORS['dark_blue'], width=2),
                hovertemplate='%{y:.2f} EUR/MW<extra></extra>'
            ),
            secondary_y=False
        )

    # aFRR+ Capacity Price
    if 'price_afrr_cap_pos_eur_mw' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df['price_afrr_cap_pos_eur_mw'].values,
                mode='lines',
                name='aFRR+ Capacity Price',
                line=dict(color=MCKINSEY_COLORS['positive'], width=2, dash='dash'),
                hovertemplate='%{y:.2f} EUR/MW<extra></extra>'
            ),
            secondary_y=False
        )

    # aFRR- Capacity Price
    if 'price_afrr_cap_neg_eur_mw' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df['price_afrr_cap_neg_eur_mw'].values,
                mode='lines',
                name='aFRR- Capacity Price',
                line=dict(color=MCKINSEY_COLORS['negative'], width=2, dash='dash'),
                hovertemplate='%{y:.2f} EUR/MW<extra></extra>'
            ),
            secondary_y=False
        )

    # ========================================================================
    # BIDS (Bars on right y-axis, in MW)
    # ========================================================================

    # FCR Capacity
    if 'c_fcr_mw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df['c_fcr_mw'].values,
                name='FCR Capacity',
                marker_color=MCKINSEY_COLORS['dark_blue'],
                opacity=0.6,
                hovertemplate='FCR: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    # aFRR+ Capacity
    if 'c_afrr_pos_mw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df['c_afrr_pos_mw'].values,
                name='aFRR+ Capacity',
                marker_color=MCKINSEY_COLORS['positive'],
                opacity=0.6,
                hovertemplate='aFRR+ Capacity: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    # aFRR- Capacity
    if 'c_afrr_neg_mw' in df.columns:
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=df['c_afrr_neg_mw'].values,
                name='aFRR- Capacity',
                marker_color=MCKINSEY_COLORS['negative'],
                opacity=0.6,
                hovertemplate='aFRR- Capacity: %{y:.3f} MW<extra></extra>'
            ),
            secondary_y=True
        )

    # ========================================================================
    # Layout configuration
    # ========================================================================

    fig.update_layout(
        title=f'Capacity Markets: Prices & Bids {title_suffix}',
        xaxis_title=x_title,
        hovermode='x unified',
        barmode='stack',  # Stack capacity bars
        template='plotly_white',
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['axis_label_size']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=600,
        width=1200
    )

    # Set y-axes titles
    fig.update_yaxes(
        title_text="Price (EUR/MW)",
        secondary_y=False,
        showgrid=True,
        gridcolor=MCKINSEY_COLORS['gray_light']
    )
    fig.update_yaxes(
        title_text="Capacity (MW)",
        secondary_y=True,
        showgrid=False
    )

    return fig



def plot_cst8_validation(df: pd.DataFrame, horizon_hours: int, save: bool = True, plot_dir: str = '.') -> plt.Figure:
    """Plot 5: Cst-8 constraint validation visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    hours = df['hour'].values

    sum_discharge = df['cst8_discharge_sum'].values
    ax1.plot(hours, sum_discharge, linewidth=2, color=MCKINSEY_COLORS['negative'], label='Discharge Binary Sum', marker='o', markersize=3)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Constraint Limit (≤ 1.0)')
    ax1.set_ylabel('Binary Sum', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax1.set_title('Cst-8a: Discharge + AS Reserves Binary Sum', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold', color=MCKINSEY_COLORS['navy'])
    ax1.legend(loc='upper right', fontsize=MCKINSEY_FONTS['legend_size'])
    violations_dis = sum_discharge > 1.000001
    if violations_dis.any():
        ax1.scatter(hours[violations_dis], sum_discharge[violations_dis], color='red', s=100, marker='X', label=f'Violations: {violations_dis.sum()}', zorder=10)

    sum_charge = df['cst8_charge_sum'].values
    ax2.plot(hours, sum_charge, linewidth=2, color=MCKINSEY_COLORS['positive'], label='Charge Binary Sum', marker='o', markersize=3)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Constraint Limit (≤ 1.0)')
    ax2.set_xlabel('Time (hours)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax2.set_ylabel('Binary Sum', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax2.set_title('Cst-8b: Charge + AS Reserves Binary Sum', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold', color=MCKINSEY_COLORS['navy'])
    ax2.legend(loc='upper right', fontsize=MCKINSEY_FONTS['legend_size'])
    ax2.set_xlim(0, horizon_hours)
    violations_ch = sum_charge > 1.000001
    if violations_ch.any():
        ax2.scatter(hours[violations_ch], sum_charge[violations_ch], color='red', s=100, marker='X', label=f'Violations: {violations_ch.sum()}', zorder=10)

    total_violations = violations_dis.sum() + violations_ch.sum()
    status_text = 'PASS' if total_violations == 0 else f'FAIL ({total_violations} violations)'
    fig.suptitle(f'Cst-8 Constraint Validation - {horizon_hours}h Horizon\nStatus: {status_text}', fontsize=MCKINSEY_FONTS['title_size'], fontweight='bold', color='green' if total_violations == 0 else 'red')

    plt.tight_layout()
    if save:
        plt.savefig(f"{plot_dir}/{horizon_hours}h_cst8_validation.png", dpi=300, bbox_inches='tight')
    return fig


def plot_battery_operation_schedule(solution_data: Dict, country_data: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Plot battery charge/discharge schedule with market prices."""
    required_vars = ['p_ch', 'p_dis', 'e_soc']
    if not all(var in solution_data for var in required_vars):
        raise ValueError(f"Solution data must contain {required_vars}")

    time_indices = sorted(solution_data['p_ch'].keys())
    charge_values = [solution_data['p_ch'][t] for t in time_indices]
    discharge_values = [solution_data['p_dis'][t] for t in time_indices]
    soc_values = [solution_data['e_soc'][t] for t in time_indices]
    datetime_values = country_data[TIMESTAMP_COL].iloc[:len(time_indices)].tolist()
    day_ahead_prices = country_data['price_day_ahead'].iloc[:len(time_indices)].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datetime_values, y=charge_values, mode='lines', name='Charge (kW)', line=dict(color='green', width=2), fill='tozeroy', fillcolor='rgba(0,255,0,0.3)'))
    fig.add_trace(go.Scatter(x=datetime_values, y=[-d for d in discharge_values], mode='lines', name='Discharge (kW)', line=dict(color='red', width=2), fill='tozeroy', fillcolor='rgba(255,0,0,0.3)'))
    fig.add_trace(go.Scatter(x=datetime_values, y=soc_values, mode='lines', name='State of Charge (kWh)', line=dict(color='blue', width=2, dash='dash'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=datetime_values, y=day_ahead_prices, mode='lines', name='Day-Ahead Price (€/MWh)', line=dict(color='orange', width=1), yaxis='y3', opacity=0.7))

    fig.update_layout(
        title=f'Battery Operation Schedule {title_suffix}', xaxis_title='Time',
        yaxis=dict(title='Power (kW)', side='left'),
        yaxis2=dict(title='SoC (kWh)', side='right', overlaying='y', showgrid=False),
        yaxis3=dict(title='Price (€/MWh)', side='right', overlaying='y', position=0.95, showgrid=False),
        hovermode='x unified', width=1000, height=500
    )
    return fig

def plot_market_price_bid_comparison(solution_data: Dict, country_data: pd.DataFrame, market_type: str = 'day_ahead', title_suffix: str = "") -> go.Figure:
    """Plot market prices vs battery bids for arbitrage analysis."""
    required_vars = ['p_ch', 'p_dis']
    if not all(var in solution_data for var in required_vars):
        raise ValueError(f"Solution data must contain {required_vars}")

    price_col_map = {'day_ahead': 'price_day_ahead', 'fcr': 'price_fcr', 'afrr': 'price_afrr_pos'}
    if market_type not in price_col_map:
        raise ValueError(f"Market type must be one of {list(price_col_map.keys())}")
    price_col = price_col_map[market_type]

    time_indices = sorted(solution_data['p_ch'].keys())
    datetime_values = country_data[TIMESTAMP_COL].iloc[:len(time_indices)].tolist()
    market_prices = country_data[price_col].iloc[:len(time_indices)].tolist()
    charge_vals = [solution_data['p_ch'][t] for t in time_indices]
    discharge_vals = [solution_data['p_dis'][t] for t in time_indices]
    net_power = [c - d for c, d in zip(charge_vals, discharge_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datetime_values, y=market_prices, mode='lines', name=f'{market_type.replace("_", " ").title()} Price (€/MWh)', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=datetime_values, y=charge_vals, mode='lines', name='Battery Charging (kW)', line=dict(color='green', width=1.5, dash='dot'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=datetime_values, y=discharge_vals, mode='lines', name='Battery Discharging (kW)', line=dict(color='red', width=1.5, dash='dot'), yaxis='y2'))
    fig.add_trace(go.Scatter(x=datetime_values, y=net_power, mode='lines', name='Net Power (kW)', line=dict(color='purple', width=1, dash='dash'), yaxis='y2'))

    fig.update_layout(
        title=f'{market_type.replace("_", " ").title()} Market: Price vs Battery Actions {title_suffix}',
        xaxis_title='Time',
        yaxis=dict(title='Market Price (€/MWh)', side='left'),
        yaxis2=dict(title='Battery Power (kW)', side='right', overlaying='y'),
        hovermode='x unified', width=1000, height=500
    )
    return fig

def plot_arbitrage_opportunities(solution_data: Dict, country_data: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Plot arbitrage opportunities highlighting profitable periods."""
    required_vars = ['p_ch', 'p_dis']
    if not all(var in solution_data for var in required_vars):
        raise ValueError(f"Solution data must contain {required_vars}")

    time_indices = sorted(solution_data['p_ch'].keys())
    datetime_values = country_data[TIMESTAMP_COL].iloc[:len(time_indices)].tolist()
    day_ahead_prices = country_data['price_day_ahead'].iloc[:len(time_indices)].tolist()
    charge_values = [solution_data['p_ch'][t] for t in time_indices]
    discharge_values = [solution_data['p_dis'][t] for t in time_indices]

    charge_periods = [i for i, c in enumerate(charge_values) if c > 1e-6]
    discharge_periods = [i for i, d in enumerate(discharge_values) if d > 1e-6]
    price_rolling_mean = pd.Series(day_ahead_prices).rolling(window=4, center=True).mean()
    price_deviations = [p - pm if pd.notna(pm) else 0 for p, pm in zip(day_ahead_prices, price_rolling_mean)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datetime_values, y=day_ahead_prices, mode='lines', name='Day-Ahead Price (€/MWh)', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=datetime_values, y=price_rolling_mean.tolist(), mode='lines', name='Price Moving Average', line=dict(color='gray', width=1, dash='dash'), opacity=0.7))
    if charge_periods:
        fig.add_trace(go.Scatter(x=[datetime_values[i] for i in charge_periods], y=[day_ahead_prices[i] for i in charge_periods], mode='markers', name='Battery Charging (Buy)', marker=dict(color='green', size=8, symbol='circle')))
    if discharge_periods:
        fig.add_trace(go.Scatter(x=[datetime_values[i] for i in discharge_periods], y=[day_ahead_prices[i] for i in discharge_periods], mode='markers', name='Battery Discharging (Sell)', marker=dict(color='red', size=8, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=datetime_values, y=price_deviations, mode='lines', name='Price Deviation from Average', line=dict(color='orange', width=1), yaxis='y2', opacity=0.5))

    fig.update_layout(
        title=f'Arbitrage Opportunities Analysis {title_suffix}', xaxis_title='Time',
        yaxis=dict(title='Price (€/MWh)', side='left'),
        yaxis2=dict(title='Price Deviation (€/MWh)', side='right', overlaying='y', showgrid=False),
        hovermode='x unified', width=1000, height=500
    )
    return fig

def plot_revenue_breakdown(solution_data: Dict, country_data: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Plot revenue breakdown by market and time period."""
    required_vars = ['p_ch', 'p_dis']
    if not all(var in solution_data for var in required_vars):
        raise ValueError(f"Solution data must contain {required_vars}")

    time_indices = sorted(solution_data['p_ch'].keys())
    datetime_values = country_data[TIMESTAMP_COL].iloc[:len(time_indices)].tolist()
    charge_values = [solution_data['p_ch'][t] for t in time_indices]
    discharge_values = [solution_data['p_dis'][t] for t in time_indices]
    da_prices = country_data['price_day_ahead'].iloc[:len(time_indices)].tolist()
    da_revenue = [(d - c) * p * 0.25 for c, d, p in zip(charge_values, discharge_values, da_prices)]
    cumulative_revenue = pd.Series(da_revenue).cumsum().tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datetime_values, y=da_revenue, mode='lines', name='Instantaneous Revenue (€/15min)', line=dict(color='green', width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=datetime_values, y=cumulative_revenue, mode='lines', name='Cumulative Revenue (€)', line=dict(color='blue', width=2), yaxis='y2'))

    fig.update_layout(
        title=f'Revenue Analysis {title_suffix}', xaxis_title='Time',
        yaxis=dict(title='Instantaneous Revenue (€/15min)', side='left'),
        yaxis2=dict(title='Cumulative Revenue (€)', side='right', overlaying='y'),
        hovermode='x unified', width=1000, height=500
    )
    return fig

def plot_battery_efficiency_analysis(solution_data: Dict, country_data: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Plot battery efficiency and cycling analysis."""
    required_vars = ['p_ch', 'p_dis', 'e_soc']
    if not all(var in solution_data for var in required_vars):
        raise ValueError(f"Solution data must contain {required_vars}")

    time_indices = sorted(solution_data['p_ch'].keys())
    datetime_values = country_data[TIMESTAMP_COL].iloc[:len(time_indices)].tolist()
    charge_values = [solution_data['p_ch'][t] for t in time_indices]
    discharge_values = [solution_data['p_dis'][t] for t in time_indices]
    soc_values = [solution_data['e_soc'][t] for t in time_indices]

    total_charge = sum(charge_values) * 0.25
    total_discharge = sum(discharge_values) * 0.25
    round_trip_efficiency = total_discharge / total_charge if total_charge > 0 else 0
    soc_range = max(soc_values) - min(soc_values)
    soc_utilization = soc_range / max(soc_values) if max(soc_values) > 0 else 0
    soc_diff = [abs(soc_values[i] - soc_values[i-1]) for i in range(1, len(soc_values))]
    avg_soc_change = sum(soc_diff) / len(soc_diff) if soc_diff else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datetime_values, y=soc_values, mode='lines', name='State of Charge (MWh)', line=dict(color='blue', width=2)))
    charge_mask = [c > 1e-6 for c in charge_values]
    if any(charge_mask):
        fig.add_trace(go.Scatter(x=[datetime_values[i] for i, m in enumerate(charge_mask) if m], y=[soc_values[i] for i, m in enumerate(charge_mask) if m], mode='markers', name='Charging Periods', marker=dict(color='green', size=4, symbol='circle')))
    discharge_mask = [d > 1e-6 for d in discharge_values]
    if any(discharge_mask):
        fig.add_trace(go.Scatter(x=[datetime_values[i] for i, m in enumerate(discharge_mask) if m], y=[soc_values[i] for i, m in enumerate(discharge_mask) if m], mode='markers', name='Discharging Periods', marker=dict(color='red', size=4, symbol='triangle-up')))

    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"Round-trip Efficiency: {round_trip_efficiency:.2%}<br>SoC Utilization: {soc_utilization:.2%}<br>Avg SoC Change: {avg_soc_change:.2f} MWh",
        showarrow=False, bgcolor="white", bordercolor="black", borderwidth=1
    )
    fig.update_layout(
        title=f'Battery Efficiency Analysis {title_suffix}', xaxis_title='Time',
        yaxis_title='State of Charge (MWh)', hovermode='x unified', width=1000, height=500
    )
    return fig

def plot_power_scheduling_overview(df: pd.DataFrame, horizon_hours: int, save: bool = True, plot_dir: str = '.') -> plt.Figure:
    """Plot 1: Power scheduling overview with SOC profile."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1, 1], hspace=0.3)
    hours = df['hour'].values

    ax1 = fig.add_subplot(gs[0])
    p_ch = df['p_ch_kw'].values / 1000
    p_dis = -df['p_dis_kw'].values / 1000
    p_afrr_pos_e = df['p_afrr_pos_e_kw'].values / 1000
    p_afrr_neg_e = -df['p_afrr_neg_e_kw'].values / 1000
    ax1.fill_between(hours, 0, p_ch, step='post', alpha=0.6, color=MCKINSEY_COLORS['positive'], label='Charge (DA)')
    ax1.fill_between(hours, 0, p_dis, step='post', alpha=0.6, color=MCKINSEY_COLORS['negative'], label='Discharge (DA)')
    ax1.fill_between(hours, 0, p_afrr_pos_e, step='post', alpha=0.4, color='#4ECDC4', label='aFRR+ Energy')
    ax1.fill_between(hours, 0, p_afrr_neg_e, step='post', alpha=0.4, color='#FF6B6B', label='aFRR- Energy')
    ax1.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Power (MW)', fontsize=MCKINSEY_FONTS['axis_label_size'], color=MCKINSEY_COLORS['gray_dark'])
    ax1.set_title(f'Battery Power Scheduling - {horizon_hours}h Horizon', fontsize=MCKINSEY_FONTS['title_size'], fontweight='bold', color=MCKINSEY_COLORS['navy'])
    ax1.grid(True, alpha=0.3, color=MCKINSEY_COLORS['gray_light'])
    ax1.legend(loc='upper right', fontsize=MCKINSEY_FONTS['legend_size'])
    ax1.set_xlim(0, horizon_hours)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    c_fcr = df['c_fcr_mw'].values
    c_afrr_pos = df['c_afrr_pos_mw'].values
    c_afrr_neg = df['c_afrr_neg_mw'].values
    ax2.fill_between(hours, 0, c_fcr, step='post', alpha=0.7, color='#FFD700', label='FCR Capacity')
    ax2.fill_between(hours, c_fcr, c_fcr + c_afrr_pos, step='post', alpha=0.7, color='#4ECDC4', label='aFRR+ Capacity')
    ax2.fill_between(hours, c_fcr + c_afrr_pos, c_fcr + c_afrr_pos + c_afrr_neg, step='post', alpha=0.7, color='#FF6B6B', label='aFRR- Capacity')
    ax2.set_ylabel('Capacity (MW)', fontsize=MCKINSEY_FONTS['axis_label_size'], color=MCKINSEY_COLORS['gray_dark'])
    ax2.set_title('Ancillary Service Capacity Reservations', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold', color=MCKINSEY_COLORS['dark_blue'])
    ax2.grid(True, alpha=0.3, color=MCKINSEY_COLORS['gray_light'])
    ax2.legend(loc='upper right', fontsize=MCKINSEY_FONTS['legend_size'])

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    soc_pct = df['soc_pct'].values
    ax3.plot(hours, soc_pct, linewidth=2.5, color=MCKINSEY_COLORS['navy'], label='SOC')
    ax3.fill_between(hours, 0, soc_pct, alpha=0.2, color=MCKINSEY_COLORS['navy'])
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Min SOC')
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Max SOC')
    ax3.set_xlabel('Time (hours)', fontsize=MCKINSEY_FONTS['axis_label_size'], color=MCKINSEY_COLORS['gray_dark'])
    ax3.set_ylabel('SOC (%)', fontsize=MCKINSEY_FONTS['axis_label_size'], color=MCKINSEY_COLORS['gray_dark'])
    ax3.set_title('State of Charge Trajectory', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold', color=MCKINSEY_COLORS['dark_blue'])
    ax3.grid(True, alpha=0.3, color=MCKINSEY_COLORS['gray_light'])
    ax3.legend(loc='upper right', fontsize=MCKINSEY_FONTS['legend_size'])
    ax3.set_ylim(-5, 105)
    ax3.set_xlim(0, horizon_hours)

    plt.tight_layout()
    if save:
        plt.savefig(f"{plot_dir}/{horizon_hours}h_power_scheduling_overview.png", dpi=300, bbox_inches='tight')
    return fig

def plot_market_participation_timeline(df: pd.DataFrame, horizon_hours: int, save: bool = True, plot_dir: str = '.') -> plt.Figure:
    """Plot 2: Market participation timeline with binary decisions."""
    fig, ax = plt.subplots(figsize=(16, 6))
    binary_vars = pd.DataFrame({
        'DA Charge': df['y_ch'].values, 'DA Discharge': df['y_dis'].values,
        'FCR Reserve': df['y_fcr'].values, 'aFRR+ Reserve': df['y_afrr_pos'].values,
        'aFRR- Reserve': df['y_afrr_neg'].values,
    })
    im = ax.imshow(binary_vars.T, aspect='auto', cmap='YlGn', interpolation='nearest', vmin=0, vmax=1, extent=[0, horizon_hours, -0.5, 4.5])
    ax.set_yticks(range(5))
    ax.set_yticklabels(binary_vars.columns, fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax.set_xlabel('Time (hours)', fontsize=MCKINSEY_FONTS['axis_label_size'], color=MCKINSEY_COLORS['gray_dark'])
    ax.set_title(f'Market Participation Timeline - {horizon_hours}h Horizon\n(0 = Inactive, 1 = Active)', fontsize=MCKINSEY_FONTS['title_size'], fontweight='bold', color=MCKINSEY_COLORS['navy'])
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Decision Value', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax.set_xticks(np.arange(0, horizon_hours + 1, 6))
    ax.grid(True, which='major', axis='x', alpha=0.3, color='white', linewidth=1)
    plt.tight_layout()
    if save:
        plt.savefig(f"{plot_dir}/{horizon_hours}h_market_participation_timeline.png", dpi=300, bbox_inches='tight')
    return fig

def plot_price_action_correlation(df: pd.DataFrame, horizon_hours: int, save: bool = True, plot_dir: str = '.') -> plt.Figure:
    """Plot 3: Price-action correlation analysis."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    hours = df['hour'].values

    ax1 = fig.add_subplot(gs[0])
    ax1_price = ax1.twinx()
    ax1_price.plot(hours, df['price_da_eur_mwh'].values, linewidth=2, color=MCKINSEY_COLORS['navy'], label='DA Price', alpha=0.7)
    ax1_price.set_ylabel('DA Price (EUR/MWh)', fontsize=MCKINSEY_FONTS['axis_label_size'], color=MCKINSEY_COLORS['navy'])
    p_net = (df['p_dis_kw'].values - df['p_ch_kw'].values) / 1000
    ax1.bar(hours, p_net, width=0.25, color=['green' if p >= 0 else 'red' for p in p_net], alpha=0.5, label='Net Power (MW)')
    ax1.set_ylabel('Net Power (MW)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax1.set_title('Day-Ahead Market: Price vs Battery Action', fontsize=MCKINSEY_FONTS['title_size'], fontweight='bold', color=MCKINSEY_COLORS['navy'])
    ax1.set_xlim(0, horizon_hours)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_price.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=MCKINSEY_FONTS['legend_size'])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2_price = ax2.twinx()
    ax2_price.plot(hours, df['price_fcr_eur_mw'].values, linewidth=2, color='#FFD700', label='FCR Price', alpha=0.7)
    ax2_price.set_ylabel('FCR Price (EUR/MW)', fontsize=MCKINSEY_FONTS['axis_label_size'], color='#FFD700')
    ax2.bar(hours, df['c_fcr_mw'].values, width=0.25, color='#FFD700', alpha=0.5, label='FCR Bid (MW)')
    ax2.set_ylabel('FCR Capacity (MW)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax2.set_title('FCR Market: Price vs Capacity Bid', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold', color=MCKINSEY_COLORS['dark_blue'])
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_price.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=MCKINSEY_FONTS['legend_size'])

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3_price = ax3.twinx()
    ax3_price.plot(hours, df['price_afrr_cap_pos_eur_mw'].values, linewidth=2, color='#4ECDC4', label='aFRR+ Price', alpha=0.7, linestyle='--')
    ax3_price.plot(hours, df['price_afrr_cap_neg_eur_mw'].values, linewidth=2, color='#FF6B6B', label='aFRR- Price', alpha=0.7, linestyle='--')
    ax3_price.set_ylabel('aFRR Price (EUR/MW)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    width = 0.12
    ax3.bar(hours - width/2, df['c_afrr_pos_mw'].values, width=width, color='#4ECDC4', alpha=0.5, label='aFRR+ Bid')
    ax3.bar(hours + width/2, df['c_afrr_neg_mw'].values, width=width, color='#FF6B6B', alpha=0.5, label='aFRR- Bid')
    ax3.set_ylabel('aFRR Capacity (MW)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax3.set_xlabel('Time (hours)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax3.set_title('aFRR Markets: Price vs Capacity Bids', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold', color=MCKINSEY_COLORS['dark_blue'])
    ax3.set_xlim(0, horizon_hours)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_price.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=MCKINSEY_FONTS['legend_size'])

    plt.tight_layout()
    if save:
        plt.savefig(f"{plot_dir}/{horizon_hours}h_price_action_correlation.png", dpi=300, bbox_inches='tight')
    return fig

def plot_revenue_breakdown_v2(df: pd.DataFrame, summary: dict, horizon_hours: int, save: bool = True, plot_dir: str = '.') -> plt.Figure:
    """Plot 4: Revenue breakdown by market and time (Cst8 version)."""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    hours = df['hour'].values

    ax1 = fig.add_subplot(gs[0, 0])
    rev_da = df['revenue_da_eur'].values
    rev_afrr_e = df['revenue_afrr_energy_eur'].values
    rev_as_cap = df['revenue_as_capacity_eur'].values
    ax1.fill_between(hours, 0, rev_da, step='post', alpha=0.6, color=MCKINSEY_COLORS['navy'], label='DA Energy')
    ax1.fill_between(hours, rev_da, rev_da + rev_afrr_e, step='post', alpha=0.6, color='#4ECDC4', label='aFRR Energy')
    ax1.fill_between(hours, rev_da + rev_afrr_e, rev_da + rev_afrr_e + rev_as_cap, step='post', alpha=0.6, color='#FFD700', label='AS Capacity')
    ax1.set_ylabel('Revenue (EUR/interval)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax1.set_title('Instantaneous Revenue by Market', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold')
    ax1.legend(loc='upper left', fontsize=MCKINSEY_FONTS['legend_size'])
    ax1.set_xlim(0, horizon_hours)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cum_total = np.cumsum(rev_da + rev_afrr_e + rev_as_cap)
    ax2.plot(hours, np.cumsum(rev_da), linewidth=2.5, color=MCKINSEY_COLORS['navy'], label='DA Energy')
    ax2.plot(hours, np.cumsum(rev_afrr_e), linewidth=2.5, color='#4ECDC4', label='aFRR Energy')
    ax2.plot(hours, np.cumsum(rev_as_cap), linewidth=2.5, color='#FFD700', label='AS Capacity')
    ax2.plot(hours, cum_total, linewidth=3, color='black', linestyle='--', label='Total', alpha=0.7)
    ax2.set_xlabel('Time (hours)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax2.set_ylabel('Cumulative Revenue (EUR)', fontsize=MCKINSEY_FONTS['axis_label_size'])
    ax2.set_title('Cumulative Revenue Over Time', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold')
    ax2.legend(loc='upper left', fontsize=MCKINSEY_FONTS['legend_size'])
    ax2.set_xlim(0, horizon_hours)

    ax3 = fig.add_subplot(gs[0, 1])
    revenues = [summary['total_revenue_da'], summary['total_revenue_afrr_e'], summary['total_revenue_as_cap']]
    labels = ['DA Energy', 'aFRR Energy', 'AS Capacity']
    colors = [MCKINSEY_COLORS['navy'], '#4ECDC4', '#FFD700']
    filtered_data = [(r, l, c) for r, l, c in zip(revenues, labels, colors) if abs(r) > 0.01]
    if filtered_data:
        revenues_f, labels_f, colors_f = zip(*filtered_data)
        wedges, texts, autotexts = ax3.pie(revenues_f, labels=labels_f, colors=colors_f, autopct='%1.1f%%', startangle=90, textprops={'fontsize': MCKINSEY_FONTS['legend_size']})
        for autotext in autotexts:
            autotext.set_color('white')
    ax3.set_title(f'Total Revenue\n{sum(revenues):.2f} EUR', fontsize=MCKINSEY_FONTS['subtitle_size'], fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    stats_text = f"REVENUE SUMMARY\n{'='*30}\n\nTotal Revenue: {sum(revenues):.2f} EUR\n\nBy Market:\n• DA Energy:    {revenues[0]:>10.2f} EUR\n• aFRR Energy:  {revenues[1]:>10.2f} EUR\n• AS Capacity:  {revenues[2]:>10.2f} EUR\n\nObjective Value: {summary['objective_value']:.2f} EUR"
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Revenue Analysis - {horizon_hours}h Horizon', fontsize=MCKINSEY_FONTS['title_size'], fontweight='bold', color=MCKINSEY_COLORS['navy'])
    plt.tight_layout()
    if save:
        plt.savefig(f"{plot_dir}/{horizon_hours}h_revenue_breakdown.png", dpi=300, bbox_inches='tight')
    return fig


def plot_comprehensive_strategy_analysis(
    df: pd.DataFrame,
    title_suffix: str = "",
    use_timestamp: bool = True,
    battery_capacity_kwh: float = 4472.0
) -> go.Figure:
    """
    Comprehensive 5-panel strategy visualization showing complete optimization behavior.
    
    This function creates a detailed multi-panel visualization inspired by the AS revenue fix
    analysis, showing:
    1. SOC trajectory with reference lines
    2. Charge/discharge power decisions
    3. FCR capacity bids vs prices
    4. Day-ahead market prices with discharge events
    5. Cumulative revenue breakdown by market
    
    Parameters
    ----------
    df : pd.DataFrame
        Solution DataFrame from extract_detailed_solution()
        Required columns:
        - 'timestamp' or 'hour': time axis
        - 'e_soc', 'soc_pct': state of charge
        - 'p_ch_kw', 'p_dis_kw': charge/discharge power
        - 'c_fcr_mw', 'price_fcr_eur_mw': FCR capacity and prices
        - 'price_da_eur_mwh': day-ahead prices
        - 'revenue_da_eur', 'revenue_as_capacity_eur': revenue components
    title_suffix : str, optional
        Additional text for plot title
    use_timestamp : bool, optional
        If True, use timestamp for x-axis; otherwise use hour
    battery_capacity_kwh : float, optional
        Battery capacity in kWh for SOC reference lines (default: 4472.0)
        
    Returns
    -------
    go.Figure
        Plotly figure with 5 subplots
        
    Example
    -------
    >>> df = extract_detailed_solution(solution, test_data, 32)
    >>> fig = plot_comprehensive_strategy_analysis(df, title_suffix="CH, 5-day MPC")
    >>> fig.show()
    """
    from plotly.subplots import make_subplots
    
    # Determine x-axis
    x_col = 'timestamp' if use_timestamp and 'timestamp' in df.columns else 'hour'
    x_values = df[x_col].values
    x_title = 'Time' if use_timestamp else 'Hour'
    
    # Create subplot structure
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=(
            'SOC Trajectory',
            'Power Decisions: Charge & Discharge',
            'FCR Capacity Bids vs Price',
            'Day-Ahead Market Price',
            'Revenue Components (Cumulative)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.20, 0.20, 0.20, 0.15, 0.25],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # ========================================================================
    # Panel 1: SOC Trajectory
    # ========================================================================
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['soc_kwh'].values,
            name='SOC',
            line=dict(color=MCKINSEY_COLORS['navy'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)',
            hovertemplate='<b>SOC</b><br>%{x}<br>SOC: %{y:.2f} kWh (%{customdata:.1f}%)<extra></extra>',
            customdata=df['soc_pct'].values,
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add SOC reference lines
    fig.add_hline(
        y=battery_capacity_kwh * 0.5,
        line_dash="dash",
        line_color="green",
        opacity=0.5,
        row=1, col=1,
        annotation_text="50% SOC",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=battery_capacity_kwh,
        line_dash="dash",
        line_color="red",
        opacity=0.3,
        row=1, col=1,
        annotation_text="100% SOC",
        annotation_position="right"
    )
    
    # ========================================================================
    # Panel 2: Power Decisions (Charge as negative, Discharge as positive)
    # ========================================================================
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=-df['p_ch_kw'].values / 1000,  # Convert to MW, negative for charge
            name='Charge',
            line=dict(color='green', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 0, 0.3)',
            hovertemplate='<b>Charge</b><br>%{x}<br>Power: %{y:.2f} MW<extra></extra>',
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['p_dis_kw'].values / 1000,  # Convert to MW
            name='Discharge',
            line=dict(color='red', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(200, 0, 0, 0.3)',
            hovertemplate='<b>Discharge</b><br>%{x}<br>Power: %{y:.2f} MW<extra></extra>',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Add zero line for clarity
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, opacity=0.5, row=2, col=1)
    
    # ========================================================================
    # Panel 3: FCR Capacity Bids vs Price (dual y-axis)
    # ========================================================================
    
    # FCR Price (line, secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['price_fcr_eur_mw'].values,
            name='FCR Price',
            line=dict(color='purple', width=1, dash='dot'),
            hovertemplate='<b>FCR Price</b><br>%{x}<br>Price: %{y:.2f} EUR/MW<extra></extra>',
            showlegend=True
        ),
        row=3, col=1,
        secondary_y=True
    )
    
    # FCR Bid (bar, primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['c_fcr_mw'].values,
            name='FCR Bid',
            line=dict(color='orange', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.3)',
            hovertemplate='<b>FCR Bid</b><br>%{x}<br>Capacity: %{y:.2f} MW<extra></extra>',
            showlegend=True
        ),
        row=3, col=1,
        secondary_y=False
    )
    
    # ========================================================================
    # Panel 4: Day-Ahead Price with Discharge Highlighting
    # ========================================================================
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=df['price_da_eur_mwh'].values,
            name='DA Price',
            line=dict(color='darkblue', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 139, 0.2)',
            hovertemplate='<b>DA Price</b><br>%{x}<br>Price: %{y:.2f} EUR/MWh<extra></extra>',
            showlegend=True
        ),
        row=4, col=1
    )
    
    # Highlight discharge periods
    discharge_mask = df['p_dis_kw'] > 100  # Threshold for significant discharge
    if discharge_mask.any():
        # Find continuous discharge periods
        discharge_starts = df[discharge_mask & ~discharge_mask.shift(1, fill_value=False)]
        discharge_ends = df[discharge_mask & ~discharge_mask.shift(-1, fill_value=False)]
        
        for start, end in zip(discharge_starts[x_col], discharge_ends[x_col]):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
                row=4, col=1
            )
    
    # ========================================================================
    # Panel 5: Revenue Components (Cumulative)
    # ========================================================================
    
    # Calculate cumulative revenues
    cumulative_da = df['revenue_da_eur'].cumsum()
    
    # Handle different column names for AS capacity revenue
    if 'revenue_as_capacity_eur' in df.columns:
        cumulative_as = df['revenue_as_capacity_eur'].cumsum()
    elif 'revenue_fcr_eur' in df.columns:
        cumulative_as = df['revenue_fcr_eur'].cumsum()
    else:
        cumulative_as = pd.Series([0] * len(df))
    
    # Handle aFRR energy revenue
    if 'revenue_afrr_energy_eur' in df.columns:
        cumulative_afrr = df['revenue_afrr_energy_eur'].cumsum()
    else:
        cumulative_afrr = pd.Series([0] * len(df))
    
    cumulative_total = cumulative_da + cumulative_as + cumulative_afrr
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=cumulative_da.values,
            name='Cumulative DA Revenue',
            line=dict(color='green', width=2),
            hovertemplate='<b>DA Revenue</b><br>%{x}<br>Total: %{y:.2f} EUR<extra></extra>',
            showlegend=True
        ),
        row=5, col=1
    )
    
    if cumulative_as.sum() > 0.01:  # Only show if non-zero
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=cumulative_as.values,
                name='Cumulative AS Capacity Revenue',
                line=dict(color='orange', width=2),
                hovertemplate='<b>AS Capacity Revenue</b><br>%{x}<br>Total: %{y:.2f} EUR<extra></extra>',
                showlegend=True
            ),
            row=5, col=1
        )
    
    if cumulative_afrr.sum() > 0.01:  # Only show if non-zero
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=cumulative_afrr.values,
                name='Cumulative aFRR Energy Revenue',
                line=dict(color='teal', width=2),
                hovertemplate='<b>aFRR Energy Revenue</b><br>%{x}<br>Total: %{y:.2f} EUR<extra></extra>',
                showlegend=True
            ),
            row=5, col=1
        )
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=cumulative_total.values,
            name='Total Revenue',
            line=dict(color='purple', width=3, dash='dash'),
            hovertemplate='<b>Total Revenue</b><br>%{x}<br>Total: %{y:.2f} EUR<extra></extra>',
            showlegend=True
        ),
        row=5, col=1
    )
    
    # ========================================================================
    # Layout Configuration
    # ========================================================================
    
    # Get financial summary for subtitle
    total_revenue = cumulative_total.iloc[-1] if len(cumulative_total) > 0 else 0
    initial_soc = df['soc_kwh'].iloc[0] if len(df) > 0 else 0
    final_soc = df['soc_kwh'].iloc[-1] if len(df) > 0 else 0
    
    fig.update_layout(
        height=1600,
        title_text=f"<b>Comprehensive Optimization Strategy Analysis</b> {title_suffix}<br>" +
                   f"<sub>Total Revenue: {total_revenue:.2f} EUR | " +
                   f"SOC: {initial_soc:.0f} → {final_soc:.0f} kWh " +
                   f"({100*initial_soc/battery_capacity_kwh:.1f}% → {100*final_soc/battery_capacity_kwh:.1f}%)</sub>",
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family=MCKINSEY_FONTS['family'], size=MCKINSEY_FONTS['tick_label_size']),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Y-axis labels
    fig.update_yaxes(title_text="SOC (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Capacity (MW)", secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_text="Price (EUR/MW)", secondary_y=True, row=3, col=1)
    fig.update_yaxes(title_text="Price (EUR/MWh)", row=4, col=1)
    fig.update_yaxes(title_text="Revenue (EUR)", row=5, col=1)
    
    # X-axis labels (only on bottom plot)
    fig.update_xaxes(title_text=x_title, row=5, col=1)
    
    # Grid styling
    for i in range(1, 6):
        fig.update_yaxes(showgrid=True, gridcolor='lightgray', row=i, col=1)
        fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=i, col=1)
    
    return fig



# =================================================================================
# ARCHIVED OLD PLOT CODE - Kept for reference
# =================================================================================


# def plot_battery_operation_schedule(solution_data: Dict, country_data: pd.DataFrame, 
#                                    title_suffix: str = "") -> go.Figure:
#     """Plot battery charge/discharge schedule with market prices.
    
#     Parameters
#     ----------
#     solution_data : dict
#         Solution dictionary containing charge ('p_ch'), discharge ('p_dis'), soc ('e_soc') data
#     country_data : pd.DataFrame
#         Market data with datetime, price_day_ahead, price_fcr, price_afrr_pos columns
#     title_suffix : str
#         Additional text for plot title
        
#     Returns
#     -------
#     go.Figure
#         Plotly figure with battery operation schedule
#     """
#     # Check if we have the required data
#     required_vars = ['p_ch', 'p_dis', 'e_soc']
#     for var in required_vars:
#         if var not in solution_data:
#             raise ValueError(f"Solution data must contain '{var}' key")
    
#     # Extract time series data
#     time_indices = sorted(solution_data['p_ch'].keys())
    
#     charge_values = [solution_data['p_ch'][t] for t in time_indices]
#     discharge_values = [solution_data['p_dis'][t] for t in time_indices]
#     soc_values = [solution_data['e_soc'][t] for t in time_indices]
    
#     # Get corresponding datetime values from country_data
#     if 'timestamp' in country_data.columns:
#         datetime_values = country_data['timestamp'].iloc[:len(time_indices)].tolist()
#     else:
#         # Generate datetime index if not available
#         datetime_values = pd.date_range(start='2024-01-01', periods=len(time_indices), freq='15min').tolist()
    
#     day_ahead_prices = country_data['price_day_ahead'].iloc[:len(time_indices)].tolist()
    
#     # Create figure with secondary y-axes
#     fig = go.Figure()
    
#     # Add battery charge (positive values)
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=charge_values,
#         mode='lines',
#         name='Charge (kW)',
#         line=dict(color='green', width=2),
#         fill='tozeroy',
#         fillcolor='rgba(0,255,0,0.3)'
#     ))
    
#     # Add battery discharge (negative for visualization)
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=[-d for d in discharge_values],  # Negative for discharge
#         mode='lines',
#         name='Discharge (kW)',
#         line=dict(color='red', width=2),
#         fill='tozeroy',
#         fillcolor='rgba(255,0,0,0.3)'
#     ))
    
#     # Add SoC on secondary y-axis
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=soc_values,
#         mode='lines',
#         name='State of Charge (kWh)',
#         line=dict(color='blue', width=2, dash='dash'),
#         yaxis='y2'
#     ))
    
#     # Add day-ahead prices on tertiary y-axis
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=day_ahead_prices,
#         mode='lines',
#         name='Day-Ahead Price (€/MWh)',
#         line=dict(color='orange', width=1),
#         yaxis='y3',
#         opacity=0.7
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title=f'Battery Operation Schedule {title_suffix}',
#         xaxis_title='Time',
#         yaxis=dict(
#             title='Power (kW)',
#             side='left'
#         ),
#         yaxis2=dict(
#             title='SoC (kWh)',
#             side='right',
#             overlaying='y',
#             showgrid=False
#         ),
#         yaxis3=dict(
#             title='Price (€/MWh)',
#             side='right',
#             overlaying='y',
#             position=0.95,
#             showgrid=False
#         ),
#         hovermode='x unified',
#         width=1000,
#         height=500
#     )
    
#     return fig


# def plot_market_price_bid_comparison(solution_data: Dict, country_data: pd.DataFrame, 
#                                     market_type: str = 'day_ahead', 
#                                     title_suffix: str = "") -> go.Figure:
#     """Plot market prices vs battery bids for arbitrage analysis.
    
#     Parameters
#     ----------
#     solution_data : dict
#         Solution dictionary containing charge ('p_ch'), discharge ('p_dis') data
#     country_data : pd.DataFrame
#         Market data with price information
#     market_type : str
#         Type of market ('day_ahead', 'fcr', 'afrr')
#     title_suffix : str
#         Additional text for plot title
        
#     Returns
#     -------
#     go.Figure
#         Plotly figure comparing market prices and bids
#     """
#     # Check required variables
#     required_vars = ['p_ch', 'p_dis']
#     for var in required_vars:
#         if var not in solution_data:
#             raise ValueError(f"Solution data must contain '{var}' key")
    
#     # Determine price column
#     price_col_map = {
#         'day_ahead': 'price_day_ahead',
#         'fcr': 'price_fcr', 
#         'afrr': 'price_afrr_pos'  # Use positive aFRR as representative
#     }
    
#     if market_type not in price_col_map:
#         raise ValueError(f"Market type must be one of {list(price_col_map.keys())}")
    
#     price_col = price_col_map[market_type]
    
#     # Extract time series data
#     time_indices = sorted(solution_data['p_ch'].keys())
    
#     # Get datetime values
#     if 'timestamp' in country_data.columns:
#         datetime_values = country_data['timestamp'].iloc[:len(time_indices)].tolist()
#     else:
#         datetime_values = pd.date_range(start='2024-01-01', periods=len(time_indices), freq='15min').tolist()
    
#     market_prices = country_data[price_col].iloc[:len(time_indices)].tolist()
    
#     # Calculate net power (charge - discharge)
#     charge_vals = [solution_data['p_ch'][t] for t in time_indices]
#     discharge_vals = [solution_data['p_dis'][t] for t in time_indices]
#     net_power = [c - d for c, d in zip(charge_vals, discharge_vals)]
    
#     fig = go.Figure()
    
#     # Add market prices
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=market_prices,
#         mode='lines',
#         name=f'{market_type.replace("_", " ").title()} Price (€/MWh)',
#         line=dict(color='blue', width=2)
#     ))
    
#     # Add battery actions
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=charge_vals,
#         mode='lines',
#         name='Battery Charging (kW)',
#         line=dict(color='green', width=1.5, dash='dot'),
#         yaxis='y2'
#     ))
    
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=discharge_vals,
#         mode='lines',
#         name='Battery Discharging (kW)',
#         line=dict(color='red', width=1.5, dash='dot'),
#         yaxis='y2'
#     ))
    
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=net_power,
#         mode='lines',
#         name='Net Power (kW)',
#         line=dict(color='purple', width=1, dash='dash'),
#         yaxis='y2'
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title=f'{market_type.replace("_", " ").title()} Market: Price vs Battery Actions {title_suffix}',
#         xaxis_title='Time',
#         yaxis=dict(
#             title='Market Price (€/MWh)',
#             side='left'
#         ),
#         yaxis2=dict(
#             title='Battery Power (kW)',
#             side='right',
#             overlaying='y'
#         ),
#         hovermode='x unified',
#         width=1000,
#         height=500
#     )
    
#     return fig


# def plot_arbitrage_opportunities(solution_data: Dict, country_data: pd.DataFrame,
#                                 title_suffix: str = "") -> go.Figure:
#     """Plot arbitrage opportunities highlighting profitable periods.
    
#     Parameters
#     ----------
#     solution_data : dict
#         Solution dictionary with charge ('p_ch'), discharge ('p_dis') data
#     country_data : pd.DataFrame
#         Market data with price information
#     title_suffix : str
#         Additional text for plot title
        
#     Returns
#     -------
#     go.Figure
#         Plotly figure showing arbitrage analysis
#     """
#     # Check required variables
#     required_vars = ['p_ch', 'p_dis']
#     for var in required_vars:
#         if var not in solution_data:
#             raise ValueError(f"Solution data must contain '{var}' key")
    
#     # Extract time series
#     time_indices = sorted(solution_data['p_ch'].keys())
    
#     # Get datetime values
#     if 'timestamp' in country_data.columns:
#         datetime_values = country_data['timestamp'].iloc[:len(time_indices)].tolist()
#     else:
#         datetime_values = pd.date_range(start='2024-01-01', periods=len(time_indices), freq='15min').tolist()
    
#     day_ahead_prices = country_data['price_day_ahead'].iloc[:len(time_indices)].tolist()
    
#     charge_values = [solution_data['p_ch'][t] for t in time_indices]
#     discharge_values = [solution_data['p_dis'][t] for t in time_indices]
    
#     # Calculate arbitrage indicators
#     charge_periods = [i for i, c in enumerate(charge_values) if c > 1e-6]
#     discharge_periods = [i for i, d in enumerate(discharge_values) if d > 1e-6]
    
#     # Calculate price differences for arbitrage analysis
#     price_rolling_mean = pd.Series(day_ahead_prices).rolling(window=4, center=True).mean()
#     price_deviations = [p - pm if pd.notna(pm) else 0 for p, pm in zip(day_ahead_prices, price_rolling_mean)]
    
#     fig = go.Figure()
    
#     # Add price line
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=day_ahead_prices,
#         mode='lines',
#         name='Day-Ahead Price (€/MWh)',
#         line=dict(color='blue', width=2)
#     ))
    
#     # Add rolling mean
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=price_rolling_mean.tolist(),
#         mode='lines',
#         name='Price Moving Average',
#         line=dict(color='gray', width=1, dash='dash'),
#         opacity=0.7
#     ))
    
#     # Highlight charging periods (low prices)
#     if charge_periods:
#         charge_times = [datetime_values[i] for i in charge_periods]
#         charge_prices = [day_ahead_prices[i] for i in charge_periods]
#         fig.add_trace(go.Scatter(
#             x=charge_times,
#             y=charge_prices,
#             mode='markers',
#             name='Battery Charging (Buy)',
#             marker=dict(color='green', size=8, symbol='circle')
#         ))
    
#     # Highlight discharging periods (high prices)
#     if discharge_periods:
#         discharge_times = [datetime_values[i] for i in discharge_periods]
#         discharge_prices = [day_ahead_prices[i] for i in discharge_periods]
#         fig.add_trace(go.Scatter(
#             x=discharge_times,
#             y=discharge_prices,
#             mode='markers',
#             name='Battery Discharging (Sell)',
#             marker=dict(color='red', size=8, symbol='triangle-up')
#         ))
    
#     # Add price deviation as background color
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=price_deviations,
#         mode='lines',
#         name='Price Deviation from Average',
#         line=dict(color='orange', width=1),
#         yaxis='y2',
#         opacity=0.5
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title=f'Arbitrage Opportunities Analysis {title_suffix}',
#         xaxis_title='Time',
#         yaxis=dict(
#             title='Price (€/MWh)',
#             side='left'
#         ),
#         yaxis2=dict(
#             title='Price Deviation (€/MWh)',
#             side='right',
#             overlaying='y',
#             showgrid=False
#         ),
#         hovermode='x unified',
#         width=1000,
#         height=500
#     )
    
#     return fig


# def plot_revenue_breakdown(solution_data: Dict, country_data: pd.DataFrame,
#                           title_suffix: str = "") -> go.Figure:
#     """Plot revenue breakdown by market and time period.
    
#     Parameters
#     ----------
#     solution_data : dict
#         Solution dictionary with charge ('p_ch'), discharge ('p_dis') data
#     country_data : pd.DataFrame
#         Market data for revenue calculation
#     title_suffix : str
#         Additional text for plot title
        
#     Returns
#     -------
#     go.Figure
#         Plotly figure showing revenue breakdown
#     """
#     # Check required variables
#     required_vars = ['p_ch', 'p_dis']
#     for var in required_vars:
#         if var not in solution_data:
#             raise ValueError(f"Solution data must contain '{var}' key")
    
#     # Calculate revenue streams
#     time_indices = sorted(solution_data['p_ch'].keys())
    
#     # Get datetime values
#     if 'timestamp' in country_data.columns:
#         datetime_values = country_data['timestamp'].iloc[:len(time_indices)].tolist()
#     else:
#         datetime_values = pd.date_range(start='2024-01-01', periods=len(time_indices), freq='15min').tolist()
    
#     # Day-ahead revenue calculation
#     charge_values = [solution_data['p_ch'][t] for t in time_indices]
#     discharge_values = [solution_data['p_dis'][t] for t in time_indices]
#     da_prices = country_data['price_day_ahead'].iloc[:len(time_indices)].tolist()
    
#     # Calculate instantaneous revenue (simplified)
#     da_revenue = []
#     for i, (c, d, price) in enumerate(zip(charge_values, discharge_values, da_prices)):
#         # Revenue = discharge * price - charge * price (simplified, no efficiency loss here)
#         instant_revenue = (d - c) * price * 0.25  # 0.25 for 15-min to 1-hour conversion
#         da_revenue.append(instant_revenue)
    
#     # Cumulative revenue
#     cumulative_revenue = pd.Series(da_revenue).cumsum().tolist()
    
#     # Daily aggregation
#     df_temp = pd.DataFrame({
#         'datetime': datetime_values,
#         'revenue': da_revenue
#     })
#     df_temp['date'] = pd.to_datetime(df_temp['datetime']).dt.date
#     daily_revenue = df_temp.groupby('date')['revenue'].sum().reset_index()
    
#     # Create subplots
#     fig = go.Figure()
    
#     # Instantaneous revenue
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=da_revenue,
#         mode='lines',
#         name='Instantaneous Revenue (€/15min)',
#         line=dict(color='green', width=1),
#         opacity=0.7
#     ))
    
#     # Cumulative revenue
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=cumulative_revenue,
#         mode='lines',
#         name='Cumulative Revenue (€)',
#         line=dict(color='blue', width=2),
#         yaxis='y2'
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title=f'Revenue Analysis {title_suffix}',
#         xaxis_title='Time',
#         yaxis=dict(
#             title='Instantaneous Revenue (€/15min)',
#             side='left'
#         ),
#         yaxis2=dict(
#             title='Cumulative Revenue (€)',
#             side='right',
#             overlaying='y'
#         ),
#         hovermode='x unified',
#         width=1000,
#         height=500
#     )
    
#     return fig


# def plot_battery_efficiency_analysis(solution_data: Dict, country_data: pd.DataFrame,
#                                     title_suffix: str = "") -> go.Figure:
#     """Plot battery efficiency and cycling analysis.
    
#     Parameters
#     ----------
#     solution_data : dict
#         Solution dictionary with charge ('p_ch'), discharge ('p_dis'), soc ('e_soc') data
#     country_data : pd.DataFrame
#         Market data
#     title_suffix : str
#         Additional text for plot title
        
#     Returns
#     -------
#     go.Figure
#         Plotly figure showing efficiency analysis
#     """
#     # Check required variables
#     required_vars = ['p_ch', 'p_dis', 'e_soc']
#     for var in required_vars:
#         if var not in solution_data:
#             raise ValueError(f"Solution data must contain '{var}' key")
    
#     # Extract time series
#     time_indices = sorted(solution_data['p_ch'].keys())
    
#     # Get datetime values
#     if 'timestamp' in country_data.columns:
#         datetime_values = country_data['timestamp'].iloc[:len(time_indices)].tolist()
#     else:
#         datetime_values = pd.date_range(start='2024-01-01', periods=len(time_indices), freq='15min').tolist()
    
#     charge_values = [solution_data['p_ch'][t] for t in time_indices]
#     discharge_values = [solution_data['p_dis'][t] for t in time_indices]
#     soc_values = [solution_data['e_soc'][t] for t in time_indices]
    
#     # Calculate efficiency metrics
#     total_charge = sum(charge_values) * 0.25  # Convert to MWh
#     total_discharge = sum(discharge_values) * 0.25  # Convert to MWh
#     round_trip_efficiency = total_discharge / total_charge if total_charge > 0 else 0
    
#     # Calculate SoC utilization
#     soc_range = max(soc_values) - min(soc_values)
#     soc_utilization = soc_range / max(soc_values) if max(soc_values) > 0 else 0
    
#     # Calculate cycling frequency (simplified)
#     soc_diff = [abs(soc_values[i] - soc_values[i-1]) for i in range(1, len(soc_values))]
#     avg_soc_change = sum(soc_diff) / len(soc_diff) if soc_diff else 0
    
#     fig = go.Figure()
    
#     # SoC profile
#     fig.add_trace(go.Scatter(
#         x=datetime_values,
#         y=soc_values,
#         mode='lines',
#         name='State of Charge (MWh)',
#         line=dict(color='blue', width=2)
#     ))
    
#     # Add charge/discharge indicators
#     charge_mask = [c > 1e-6 for c in charge_values]
#     discharge_mask = [d > 1e-6 for d in discharge_values]
    
#     # Charging periods
#     if any(charge_mask):
#         charge_times = [datetime_values[i] for i, mask in enumerate(charge_mask) if mask]
#         charge_soc = [soc_values[i] for i, mask in enumerate(charge_mask) if mask]
#         fig.add_trace(go.Scatter(
#             x=charge_times,
#             y=charge_soc,
#             mode='markers',
#             name='Charging Periods',
#             marker=dict(color='green', size=4, symbol='circle')
#         ))
    
#     # Discharging periods
#     if any(discharge_mask):
#         discharge_times = [datetime_values[i] for i, mask in enumerate(discharge_mask) if mask]
#         discharge_soc = [soc_values[i] for i, mask in enumerate(discharge_mask) if mask]
#         fig.add_trace(go.Scatter(
#             x=discharge_times,
#             y=discharge_soc,
#             mode='markers',
#             name='Discharging Periods',
#             marker=dict(color='red', size=4, symbol='triangle-up')
#         ))
    
#     # Add efficiency metrics as annotations
#     fig.add_annotation(
#         x=0.02, y=0.98,
#         xref="paper", yref="paper",
#         text=f"Round-trip Efficiency: {round_trip_efficiency:.2%}<br>"
#              f"SoC Utilization: {soc_utilization:.2%}<br>"
#              f"Avg SoC Change: {avg_soc_change:.2f} MWh",
#         showarrow=False,
#         bgcolor="white",
#         bordercolor="black",
#         borderwidth=1
#     )
    
#     # Update layout
#     fig.update_layout(
#         title=f'Battery Efficiency Analysis {title_suffix}',
#         xaxis_title='Time',
#         yaxis_title='State of Charge (MWh)',
#         hovermode='x unified',
#         width=1000,
#         height=500
#     )
    
#     return fig


# ---------------------------------------------------------------------------
# Segment LIFO Analysis and Visualization
# ---------------------------------------------------------------------------


def detect_parallel_segment_operations(
    solution_df: pd.DataFrame,
    epsilon: float = 0.1
) -> tuple[list[dict], list[dict]]:
    """
    Detect parallel charging and discharging violations in segmented SOC model.

    In a proper LIFO (Last-In-First-Out) battery model, only one segment should
    be actively charging or discharging at any given time. This function detects
    violations where multiple segments change energy simultaneously.

    Parameters
    ----------
    solution_df : pd.DataFrame
        Solution DataFrame with segment_1...segment_10 columns
    epsilon : float, default=0.1
        Tolerance for detecting energy changes (kWh). Changes smaller than
        this are ignored to avoid numerical noise.

    Returns
    -------
    tuple[list[dict], list[dict]]
        - parallel_charging_events: List of dictionaries with violation details
        - parallel_discharging_events: List of dictionaries with violation details

    Examples
    --------
    >>> parallel_ch, parallel_dis = detect_parallel_segment_operations(df)
    >>> print(f"Found {len(parallel_ch)} parallel charging violations")
    >>> for event in parallel_ch:
    ...     print(f"Hour {event['hour']}: Segments {event['segments']}")

    See Also
    --------
    plot_segment_lifo_analysis : Main visualization function
    """
    segment_cols = [f'segment_{i}' for i in range(1, 11)]

    parallel_charging = []
    parallel_discharging = []

    for t in range(1, len(solution_df)):
        # Calculate deltas
        deltas = {}
        for i, col in enumerate(segment_cols):
            delta = solution_df[col].iloc[t] - solution_df[col].iloc[t-1]
            if abs(delta) > epsilon:
                deltas[i+1] = delta  # 1-indexed segment number

        # Find charging segments
        charging = {seg: d for seg, d in deltas.items() if d > epsilon}
        if len(charging) > 1:
            parallel_charging.append({
                'timestep': t,
                'hour': solution_df['hour'].iloc[t],
                'segments': list(charging.keys()),
                'deltas': list(charging.values()),
                'soc': solution_df['soc_kwh'].iloc[t]
            })

        # Find discharging segments
        discharging = {seg: abs(d) for seg, d in deltas.items() if d < -epsilon}
        if len(discharging) > 1:
            parallel_discharging.append({
                'timestep': t,
                'hour': solution_df['hour'].iloc[t],
                'segments': list(discharging.keys()),
                'deltas': list(discharging.values()),
                'soc': solution_df['soc_kwh'].iloc[t]
            })

    return parallel_charging, parallel_discharging


def plot_segment_lifo_analysis(
    solution_df: pd.DataFrame,
    show_violations: bool = True,
    plot_style: str = 'stacked',
    epsilon: float = 0.1,
    width: int = 1400,
    height: int = 1200,
    title_suffix: str = ''
) -> go.Figure:
    """
    Create comprehensive LIFO (Last-In-First-Out) analysis visualization for battery segments.

    This function generates a 4-panel visualization to analyze and validate LIFO behavior
    in segmented battery SOC models. The visualization is critical for detecting violations
    of the LIFO constraint, where multiple segments receive/release energy simultaneously.

    Parameters
    ----------
    solution_df : pd.DataFrame
        Solution DataFrame with segment_1...segment_10 columns
    show_violations : bool, default=True
        Highlight parallel charging/discharging events
    plot_style : {'stacked', 'individual'}, default='stacked'
        Plot style for segment energy
    epsilon : float, default=0.1
        Tolerance for violation detection (kWh)
    width : int, default=1400
        Plot width in pixels
    height : int, default=1200
        Plot height in pixels
    title_suffix : str, default=''
        Additional text for plot title

    Returns
    -------
    go.Figure
        Plotly Figure with 4 subplots

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv("solution_timeseries.csv")
    >>> fig = plot_segment_lifo_analysis(df, title_suffix="CH 24h")
    >>> fig.write_html("segment_analysis.html")
    """
    from plotly.subplots import make_subplots

    segment_cols = [f'segment_{i}' for i in range(1, 11)]
    E_seg = 447.2  # Segment capacity in kWh

    # Detect violations
    if show_violations:
        parallel_ch, parallel_dis = detect_parallel_segment_operations(solution_df, epsilon)
    else:
        parallel_ch, parallel_dis = [], []

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'Segment Energy ({plot_style.capitalize()})',
            'Energy Deltas (Δ per timestep)',
            'Total SOC',
            'Power Flows'
        ),
        specs=[[{}], [{}], [{}], [{}]],
        vertical_spacing=0.08,
        row_heights=[0.35, 0.25, 0.2, 0.2]
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Panel 1: Segment Energy
    if plot_style == 'stacked':
        for i, col in enumerate(segment_cols):
            fig.add_trace(
                go.Scatter(x=solution_df['hour'], y=solution_df[col],
                          name=f'Seg {i+1}', mode='lines', stackgroup='segments',
                          fillcolor=colors[i], line=dict(width=0.5, color=colors[i])),
                row=1, col=1
            )
    else:
        for i, col in enumerate(segment_cols):
            fig.add_trace(
                go.Scatter(x=solution_df['hour'], y=solution_df[col],
                          name=f'Seg {i+1}', mode='lines',
                          line=dict(width=2, color=colors[i])),
                row=1, col=1
            )
        fig.add_hline(y=E_seg, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # Panel 2: Deltas
    for i, col in enumerate(segment_cols):
        delta = solution_df[col].diff().fillna(0)
        fig.add_trace(
            go.Scatter(x=solution_df['hour'], y=delta, name=f'Δ{i+1}',
                      mode='markers', marker=dict(size=4, color=colors[i]),
                      showlegend=False),
            row=2, col=1
        )

    fig.add_hline(y=0, line_color="black", opacity=0.3, row=2, col=1)

    if show_violations and parallel_ch:
        fig.add_trace(
            go.Scatter(x=[v['hour'] for v in parallel_ch], y=[0]*len(parallel_ch),
                      mode='markers', marker=dict(symbol='x', size=12, color='red'),
                      name='Parallel Ch', showlegend=True),
            row=2, col=1
        )

    # Panel 3: SOC
    fig.add_trace(
        go.Scatter(x=solution_df['hour'], y=solution_df['soc_kwh'],
                  name='SOC', mode='lines', line=dict(width=3, color='black'),
                  fill='tozeroy'),
        row=3, col=1
    )
    fig.add_hline(y=4472, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    # Panel 4: Power
    fig.add_trace(
        go.Scatter(x=solution_df['hour'], y=solution_df['p_total_ch_kw'],
                  name='Charge', mode='lines', line=dict(width=2, color='green'),
                  fill='tozeroy'),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=solution_df['hour'], y=-solution_df['p_total_dis_kw'],
                  name='Discharge', mode='lines', line=dict(width=2, color='red'),
                  fill='tozeroy'),
        row=4, col=1
    )
    fig.add_hline(y=0, line_color="black", opacity=0.3, row=4, col=1)

    # Layout
    title_text = "Segment LIFO Analysis"
    if title_suffix:
        title_text += f" - {title_suffix}"
    if show_violations:
        if parallel_ch or parallel_dis:
            title_text += f" [{len(parallel_ch)} Violations]"
        else:
            title_text += " [OK]"

    fig.update_layout(
        height=height, width=width, title_text=title_text,
        showlegend=True, hovermode='x unified',
        legend=dict(x=1.05, y=0.99, xanchor='left', yanchor='top')
    )

    fig.update_xaxes(title_text="Hour", row=4, col=1)
    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="ΔE (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="SOC (kWh)", row=3, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=4, col=1)

    return fig
