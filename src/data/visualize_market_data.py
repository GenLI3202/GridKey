"""Market data visualization utilities for TechArena 2025 Phase II.

This module provides comprehensive visualization functions for electricity market data
across multiple markets: Day-Ahead, FCR, aFRR Capacity, and aFRR Energy.

Data Format
-----------
Supports both wide and tidy formats:
- Day-ahead & FCR: columns [timestamp, DE_LU/DE, AT, CH, HU, CZ]  
- aFRR: columns [timestamp, DE_Pos, DE_Neg, AT_Pos, AT_Neg, ...]

Key Visualization Functions
----------------------------

**Time Series (Trend Analysis):**
- plot_price_time_series_mckinsey() - Multi-market time series for one country
- plot_day_ahead_trend() - Day-ahead prices over time

**Box Plot Distributions (NEW Phase II):**
- plot_all_markets_distribution() - 2x2 grid showing all 4 markets
- plot_country_market_comparison() - Single country across all markets
- plot_market_distribution() - Flexible single-market distribution
- Individual wrappers: plot_day_ahead_distribution(), plot_fcr_distribution(), etc.

**Advanced Visualizations:**
- plot_da_price_heatmap_mckinsey() - Hour x Month heatmap
- plot_da_price_ridgeline_mckinsey() - Multi-country ridge plot
- plot_price_statistics_mckinsey() - Statistical summary table

Typical Usage
-------------
>>> from pathlib import Path
>>> from src.data.load_process_market_data import load_phase2_market_tables
>>> from src.data.visualize_market_data import (
...     plot_all_markets_distribution,
...     plot_country_market_comparison,
...     plot_price_time_series_mckinsey
... )
>>> 
>>> # Load Phase 2 data
>>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
>>> 
>>> # Example 1: Compare all markets side-by-side
>>> fig1 = plot_all_markets_distribution(tables, countries=['DE', 'AT', 'CH'])
>>> fig1.show()
>>> 
>>> # Example 2: Single country across all markets
>>> fig2 = plot_country_market_comparison(tables, country='DE')
>>> fig2.show()
>>> 
>>> # Example 3: Time series trends
>>> fig3 = plot_price_time_series_mckinsey(tables, country='DE', time_range='Q1')
>>> fig3.show()

All functions return Plotly Figure instances for embedding in notebooks or dashboards.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from .load_process_market_data import (
    wide_to_tidy_day_ahead,
    wide_to_tidy_fcr,
    wide_to_tidy_afrr,
    load_market_tables,
    load_phase2_market_tables,
    TIMESTAMP_COL as _TIMESTAMP_COL,
    COUNTRY_COL as _COUNTRY_COL,
    PRICE_COL_MWH as _PRICE_COL_MWH,
    PRICE_COL_MW as _PRICE_COL_MW,
    DIRECTION_COL as _DIRECTION_COL,
)   

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAY_AHEAD_SHEET = "Day-ahead prices"
FCR_SHEET = "FCR prices"
AFRR_SHEET = "aFRR capacity prices"

TIMESTAMP_COL = "timestamp"
COUNTRY_COL = "country"
PRICE_COL_MWH = "price_eur_mwh"  # For day-ahead (energy prices) AND aFRR-Energy
PRICE_COL_MW = "price_eur_mw"    # For FCR and aFRR (capacity prices)
DIRECTION_COL = "direction"

AFRR_DIRECTION_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "up": "positive",
    "+": "positive",
    "upward": "positive",
    "negative": "negative",
    "neg": "negative",
    "down": "negative",
    "-": "negative",
    "downward": "negative",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _is_tidy_format(df: pd.DataFrame, expected_columns: list) -> bool:
    """Check if DataFrame is in tidy format based on expected column names."""
    return all(col in df.columns for col in expected_columns)


def wide_to_tidy_day_ahead(day_ahead_df: pd.DataFrame) -> pd.DataFrame:
    """Convert day-ahead DataFrame from wide format to tidy format."""
    country_cols = [col for col in day_ahead_df.columns if col != TIMESTAMP_COL]
    tidy_df = day_ahead_df.melt(
        id_vars=[TIMESTAMP_COL], value_vars=country_cols,
        var_name=COUNTRY_COL, value_name=PRICE_COL_MWH
    ).dropna(subset=[PRICE_COL_MWH])
    return tidy_df.sort_values([COUNTRY_COL, TIMESTAMP_COL]).reset_index(drop=True)



# ---------------------------------------------------------------------------
# Analytical summaries
# ---------------------------------------------------------------------------


def summarize_day_ahead(day_ahead_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize day-ahead market data volatility by country.
    """
    summary = day_ahead_df.groupby("country").agg(
        mean_price=("price_eur_mwh", "mean"),
        median_price=("price_eur_mwh", "median"),
        std_dev_price=("price_eur_mwh", "std"),
        # var_price=("price_eur_mwh", "var"),
        min_price=("price_eur_mwh", "min"),
        max_price=("price_eur_mwh", "max"),
        price_range=("price_eur_mwh", lambda x: x.max() - x.min()),
    ).reset_index().sort_values("country")
    return summary

def summarize_fcr(fcr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute average FCR price per country."""
    # Get country columns (all except timestamp)
    summary = fcr_df.groupby("country").agg(
        mean_price_eur_mw=("price_eur_mwh", "mean"),
        median_price_eur_mw=("price_eur_mwh", "median"),
        std_dev_price_eur_mw=("price_eur_mwh", "std"),
        min_price_eur_mw=("price_eur_mwh", "min"),
        max_price_eur_mw=("price_eur_mwh", "max"),
        price_range_eur_mw=("price_eur_mwh", lambda x: x.max() - x.min()),
    ).reset_index().sort_values("country")
    return summary

def summarize_afrr(afrr_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize aFRR prices by country and direction."""
    summary = afrr_df.groupby(["country", "direction"]).agg(
        mean_price_eur_mw=("price_eur_mwh", "mean"),
        median_price_eur_mw=("price_eur_mwh", "median"),
        std_dev_price_eur_mw=("price_eur_mwh", "std"),
        min_price_eur_mw=("price_eur_mwh", "min"),
        max_price_eur_mw=("price_eur_mwh", "max"),
        price_range_eur_mw=("price_eur_mwh", lambda x: x.max() - x.min()),
    ).reset_index().sort_values(["country", "direction"])
    return summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
#
# UNIFIED PLOTTING ARCHITECTURE (Phase 2)
# ========================================
# Core function: plot_market_distribution(market_df, market_type)
#   - Supports all market types: 'day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy'
#   - Automatically handles wide ↔ tidy format conversion
#   - Uses appropriate price column (EUR/MWh vs EUR/MW) per market type
#
# Convenience wrappers (backward compatible):
#   - plot_day_ahead_distribution(df) → wraps market_type='day_ahead'
#   - plot_fcr_distribution(df) → wraps market_type='fcr'
#   - plot_afrr_capacity_distribution(df) → wraps market_type='afrr_capacity'
#   - plot_afrr_energy_distribution(df) → wraps market_type='afrr_energy'
#   - plot_afrr_distribution(df) → auto-detects capacity vs energy
#
# Format conversion functions imported from load_process_market_data:
#   - wide_to_tidy_day_ahead, wide_to_tidy_fcr, wide_to_tidy_afrr
# ---------------------------------------------------------------------------


def plot_market_distribution(market_df: pd.DataFrame, market_type: str = 'day_ahead') -> go.Figure:
    """Box plot comparing price distributions across countries for any market.
    
    Parameters
    ----------
    market_df : pd.DataFrame
        Market data in wide or tidy format
    market_type : str, optional
        Market type: 'day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy' (default: 'day_ahead')
        
    Returns
    -------
    go.Figure
        Box plot of price distributions
    """
    
    # Determine price column and conversion function based on market type
    if market_type in ['day_ahead', 'afrr_energy']:
        price_col = PRICE_COL_MWH
        convert_func = wide_to_tidy_day_ahead  # Works for both DA and aFRR energy (EUR/MWh)
        price_unit = "EUR/MWh"
        tidy_columns = [TIMESTAMP_COL, COUNTRY_COL, PRICE_COL_MWH]
    elif market_type in ['fcr', 'afrr_capacity']:
        price_col = PRICE_COL_MW
        convert_func = wide_to_tidy_fcr
        price_unit = "EUR/MW"
        tidy_columns = [TIMESTAMP_COL, COUNTRY_COL, PRICE_COL_MW]
    else:
        raise ValueError(f"Unknown market_type: {market_type}")
    
    # Check if data is in tidy format
    if not _is_tidy_format(market_df, tidy_columns):
        # Data is in wide format, convert to tidy format
        melted = convert_func(market_df.copy())
    else:
        # Data is already in tidy format
        melted = market_df.copy()
    
    # Special handling for day-ahead: rename DE_LU to DE for consistency
    if market_type == 'day_ahead' and COUNTRY_COL in melted.columns:
        if 'DE_LU' in melted[COUNTRY_COL].values:
            melted[COUNTRY_COL] = melted[COUNTRY_COL].replace('DE_LU', 'DE')
    
    # Rename columns for plotting
    melted = melted.rename(columns={price_col: 'price', COUNTRY_COL: 'country'})
    
    # Drop NaN values
    melted = melted.dropna(subset=['price'])
    
    # Ensure price column is numeric
    melted['price'] = pd.to_numeric(melted['price'], errors='coerce')
    melted = melted.dropna(subset=['price'])
    
    if melted.empty:
        # Return empty figure if no valid data
        fig = go.Figure()
        fig.add_annotation(
            text="No valid price data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=f"{market_type.replace('_', ' ').title()} Price Distribution by Country")
        return fig
    
    # Create box plot
    fig = px.box(
        melted,
        x='country',
        y='price',
        color='country',
        title=f"{market_type.replace('_', ' ').title()} Price Distribution by Country",
        points="suspectedoutliers",
    )
    fig.update_layout(showlegend=False, xaxis_title="Country", yaxis_title=f"Price [{price_unit}]")
    return fig

def plot_day_ahead_distribution(day_ahead_df: pd.DataFrame) -> go.Figure:
    """Box plot comparing Day-Ahead price distributions across countries.
    
    Convenience wrapper for plot_market_distribution with market_type='day_ahead'.
    """
    return plot_market_distribution(day_ahead_df, market_type='day_ahead')


def plot_fcr_distribution(fcr_df: pd.DataFrame) -> go.Figure:
    """Box plot comparing FCR price distributions across countries.
    
    Convenience wrapper for plot_market_distribution with market_type='fcr'.
    """
    return plot_market_distribution(fcr_df, market_type='fcr')


def plot_afrr_capacity_distribution(afrr_capacity_df: pd.DataFrame) -> go.Figure:
    """Box plot comparing aFRR capacity price distributions across countries.
    
    Convenience wrapper for plot_market_distribution with market_type='afrr_capacity'.
    """
    return plot_market_distribution(afrr_capacity_df, market_type='afrr_capacity')


def plot_afrr_energy_distribution(afrr_energy_df: pd.DataFrame) -> go.Figure:
    """Box plot comparing aFRR energy price distributions across countries.
    
    Convenience wrapper for plot_market_distribution with market_type='afrr_energy'.
    """
    return plot_market_distribution(afrr_energy_df, market_type='afrr_energy')


def plot_day_ahead_trend(day_ahead_df: pd.DataFrame, *, countries: Optional[Iterable[str]] = None) -> go.Figure:
    """Line plot of Day-Ahead prices across 2024, optionally filtered by country."""
    
    # Check if data is in tidy format
    if _is_tidy_format(day_ahead_df, [TIMESTAMP_COL, COUNTRY_COL, PRICE_COL_MWH]):
        # Data is already in tidy format
        melted = day_ahead_df.copy()
        melted = melted.rename(columns={PRICE_COL_MWH: 'price', COUNTRY_COL: 'country'})
        
        # Filter countries if specified
        if countries:
            melted = melted[melted['country'].isin(countries)]
    else:
        # Data is in wide format
        all_country_cols = [col for col in day_ahead_df.columns if col != TIMESTAMP_COL]
        
        if countries:
            # Filter to specified countries
            available_countries = [col for col in all_country_cols if col in countries]
        else:
            available_countries = all_country_cols
        
        # Convert wide format to long format for plotting
        melted = day_ahead_df.melt(
            id_vars=[TIMESTAMP_COL],
            value_vars=available_countries,
            var_name='country',
            value_name='price'
        )
    
    # Clean data
    melted = melted.dropna(subset=['price'])
    melted['price'] = pd.to_numeric(melted['price'], errors='coerce')
    melted = melted.dropna(subset=['price'])
    
    if melted.empty:
        # Return empty figure if no valid data
        fig = go.Figure()
        fig.add_annotation(
            text="No valid price data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Day-Ahead Price Trend (2024)")
        return fig
    
    fig = px.line(
        melted,
        x=TIMESTAMP_COL,
        y='price',
        color='country',
        title="Day-Ahead Price Trend (2024)",
    )
    fig.update_layout(xaxis_title="Timestamp", yaxis_title="Price [EUR/MWh]")
    return fig


def plot_afrr_distribution(afrr_df: pd.DataFrame) -> go.Figure:
    """Box plots for aFRR positive and negative capacity/energy prices by country.
    
    This function wraps plot_market_distribution for backward compatibility.
    For aFRR capacity, use plot_afrr_capacity_distribution().
    For aFRR energy, use plot_afrr_energy_distribution().
    """
    # Try to detect if it's capacity or energy based on data range
    # Capacity prices are typically in EUR/MW (higher values)
    # Energy prices are typically in EUR/MWh (similar to day-ahead)
    
    # Get first non-timestamp column to check values
    price_cols = [col for col in afrr_df.columns if col != TIMESTAMP_COL]
    if price_cols:
        sample_values = afrr_df[price_cols[0]].dropna().abs()
        mean_val = sample_values.mean() if len(sample_values) > 0 else 0
        
        # Heuristic: capacity prices typically > 100, energy prices < 500
        if mean_val > 100:
            return plot_market_distribution(afrr_df, market_type='afrr_capacity')
        else:
            return plot_market_distribution(afrr_df, market_type='afrr_energy')
    
    # Default to capacity if can't determine
    return plot_market_distribution(afrr_df, market_type='afrr_capacity')

def plot_day_ahead_heatmap(day_ahead_df: pd.DataFrame, country: str) -> go.Figure:
    """Hourly-by-month heatmap to reveal charging/discharging windows for a country."""
    
    # Check if data is in tidy format
    if _is_tidy_format(day_ahead_df, [TIMESTAMP_COL, COUNTRY_COL, PRICE_COL_MWH]):
        # Data is in tidy format
        if country not in day_ahead_df[COUNTRY_COL].unique():
            raise ValueError(f"No Day-Ahead data available for country '{country}'.")
        
        # Filter for the specific country
        country_df = day_ahead_df[day_ahead_df[COUNTRY_COL] == country].copy()
        if country_df.empty:
            raise ValueError(f"No Day-Ahead data available for country '{country}'.")
        
        # Use the price column for values
        price_column = PRICE_COL_MWH
        
    else:
        # Data is in wide format (original logic)
        if country not in day_ahead_df.columns:
            raise ValueError(f"No Day-Ahead data available for country '{country}'.")
        
        # Create a subset with just timestamp and the selected country
        country_df = day_ahead_df[[TIMESTAMP_COL, country]].dropna()
        if country_df.empty:
            raise ValueError(f"No Day-Ahead data available for country '{country}'.")
        
        # Use the country column for values
        price_column = country

    enriched = country_df.assign(
        hour=lambda d: d[TIMESTAMP_COL].dt.hour,
        month=lambda d: d[TIMESTAMP_COL].dt.month,
    )
    
    pivot = (
        enriched.pivot_table(
            index="month",
            columns="hour",
            values=price_column,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        labels=dict(x="Hour of Day", y="Month", color="Avg Price [EUR/MWh]"),
        title=f"Hourly vs Monthly Day-Ahead Prices · {country}",
    )
    return fig



# ---------------------------------------------------------------------------
# Convenience utilities
# ---------------------------------------------------------------------------


def ensure_csv_exports(tables: Dict[str, pd.DataFrame], directory: Path) -> None:
    """Persist tidy tables to CSV for faster reloads during experimentation."""
    directory.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        df.to_csv(directory / f"{key}.csv", index=False)


# ===========================================================================
# PHASE 2 EXTENSIONS
# ===========================================================================

# Phase 2 Constants
AFRR_ENERGY_SHEET = "aFRR energy prices"

# Validation Constants
PRICE_BOUNDS = {
    'day_ahead': (-500, 2000),    # EUR/MWh (allow extreme scarcity prices)
    'fcr': (0, 10000),             # EUR/MW (capacity always non-negative)
    'afrr_capacity': (0, 10000),   # EUR/MW
    'afrr_energy': (-500, 2000)    # EUR/MWh (allow extreme scarcity prices)
}
ZERO_THRESHOLD_PCT = 95  # Flag if >95% zeros

# ===========================================================================
# VIEW 1: DATA EXPLORATION VISUALIZATIONS (McKinsey Style)
# ===========================================================================

def _filter_by_time_range(df: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """Filter DataFrame by time range string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' column
    time_range : str
        One of: 'full', 'Q1', 'Q2', 'Q3', 'Q4', or 'YYYY-MM'

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if time_range == 'full':
        return df
    elif time_range in ['Q1', 'Q2', 'Q3', 'Q4']:
        quarter_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        months = quarter_map[time_range]
        return df[df[TIMESTAMP_COL].dt.month.isin(months)]
    else:
        # Assume format 'YYYY-MM'
        return df[df[TIMESTAMP_COL].dt.strftime('%Y-%m') == time_range]


def plot_price_time_series_mckinsey(
    tables: Dict[str, pd.DataFrame],
    country: str,
    time_range: str = 'full',
    markets: list = None
) -> go.Figure:
    """Plot multi-market price time series with McKinsey styling.

    Module A: Electricity Price Time Series

    Creates an interactive multi-series line chart showing DA, FCR, aFRR capacity,
    and aFRR energy prices for a selected country.

    Parameters
    ----------
    tables : dict
        Dictionary with keys: 'day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy'
    country : str
        Country code (DE, AT, CH, HU, CZ)
    time_range : str, optional
        'full', 'Q1', 'Q2', 'Q3', 'Q4', or 'YYYY-MM' (default: 'full')
    markets : list, optional
        List of markets to plot. Default: all

    Returns
    -------
    go.Figure
        McKinsey-styled Plotly figure

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> fig = plot_price_time_series_mckinsey(tables, country='DE', time_range='Q1')
    >>> fig.show()
    """
    from src.visualization.config import MCKINSEY_COLORS, get_country_color, apply_mckinsey_style

    if markets is None:
        markets = ['day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy']

    fig = go.Figure()

    # Add DA prices (15-min, EUR/MWh)
    if 'day_ahead' in markets and 'day_ahead' in tables:
        df_da = tables['day_ahead']
        country_col = 'DE_LU' if country == 'DE' else country

        if country_col in df_da.columns:
            df_filtered = _filter_by_time_range(df_da, time_range)

            fig.add_trace(go.Scatter(
                x=df_filtered[TIMESTAMP_COL],
                y=df_filtered[country_col],
                mode='lines',
                name='Day-Ahead',
                line=dict(color=MCKINSEY_COLORS['cat_1'], width=1.5),
                hovertemplate='%{y:.2f} EUR/MWh<extra></extra>'
            ))

    # Add FCR prices (4-hour blocks, EUR/MW) - on secondary axis
    if 'fcr' in markets and 'fcr' in tables:
        df_fcr = tables['fcr']

        if country in df_fcr.columns:
            df_filtered = _filter_by_time_range(df_fcr, time_range)

            fig.add_trace(go.Scatter(
                x=df_filtered[TIMESTAMP_COL],
                y=df_filtered[country],
                mode='lines',
                name='FCR Capacity',
                line=dict(color=MCKINSEY_COLORS['cat_2'], width=1.5, dash='dot'),
                hovertemplate='%{y:.2f} EUR/MW<extra></extra>',
                yaxis='y2'  # Secondary axis
            ))

    # Add aFRR capacity (4-hour blocks, EUR/MW, Pos/Neg)
    if 'afrr_capacity' in markets and 'afrr_capacity' in tables:
        df_afrr_cap = tables['afrr_capacity']
        df_filtered = _filter_by_time_range(df_afrr_cap, time_range)

        if f'{country}_Pos' in df_afrr_cap.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered[TIMESTAMP_COL],
                y=df_filtered[f'{country}_Pos'],
                mode='lines',
                name='aFRR Cap (Pos)',
                line=dict(color=MCKINSEY_COLORS['positive'], width=1.5, dash='dash'),
                hovertemplate='%{y:.2f} EUR/MW<extra></extra>',
                yaxis='y2'
            ))

        if f'{country}_Neg' in df_afrr_cap.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered[TIMESTAMP_COL],
                y=df_filtered[f'{country}_Neg'],
                mode='lines',
                name='aFRR Cap (Neg)',
                line=dict(color=MCKINSEY_COLORS['negative'], width=1.5, dash='dash'),
                hovertemplate='%{y:.2f} EUR/MW<extra></extra>',
                yaxis='y2'
            ))

    # Add aFRR energy (15-min, EUR/MWh, Pos/Neg) - NEW
    if 'afrr_energy' in markets and 'afrr_energy' in tables:
        df_afrr_energy = tables['afrr_energy']
        df_filtered = _filter_by_time_range(df_afrr_energy, time_range)

        if f'{country}_Pos' in df_afrr_energy.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered[TIMESTAMP_COL],
                y=df_filtered[f'{country}_Pos'],
                mode='lines',
                name='aFRR Energy (Pos)',
                line=dict(color=MCKINSEY_COLORS['teal'], width=1.5),
                hovertemplate='%{y:.2f} EUR/MWh<extra></extra>'
            ))

        if f'{country}_Neg' in df_afrr_energy.columns:
            fig.add_trace(go.Scatter(
                x=df_filtered[TIMESTAMP_COL],
                y=df_filtered[f'{country}_Neg'],
                mode='lines',
                name='aFRR Energy (Neg)',
                line=dict(color=MCKINSEY_COLORS['cat_5'], width=1.5),
                hovertemplate='%{y:.2f} EUR/MWh<extra></extra>'
            ))

    # Apply McKinsey styling and layout
    fig = apply_mckinsey_style(
        fig,
        title=f'Electricity Market Prices - {country} ({time_range})'
    )

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Energy Price (EUR/MWh)',
        yaxis2=dict(
            title='Capacity Price (EUR/MW)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig


def plot_all_markets_distribution(
    tables: Dict[str, pd.DataFrame],
    countries: list = None,
    layout: str = '2x2'
) -> go.Figure:
    """Plot box distributions for all 4 markets in a single figure.

    Creates a multi-panel visualization showing price distributions across
    day-ahead, FCR, aFRR capacity, and aFRR energy markets.

    Parameters
    ----------
    tables : dict
        Dictionary with keys: 'day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy'
    countries : list, optional
        List of country codes to include (default: all available)
    layout : str, optional
        Layout style: '2x2' (grid) or 'horizontal' (1x4) (default: '2x2')

    Returns
    -------
    go.Figure
        Multi-panel box plot figure

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> fig = plot_all_markets_distribution(tables, countries=['DE', 'AT', 'CH'])
    >>> fig.show()
    """
    from plotly.subplots import make_subplots
    from src.visualization.config import COUNTRY_COLORS, get_country_color

    if countries is None:
        countries = ['DE', 'AT', 'CH', 'HU', 'CZ']

    # Define market configurations
    market_configs = [
        ('day_ahead', 'Day-Ahead Energy', 'EUR/MWh', PRICE_COL_MWH),
        ('fcr', 'FCR Capacity', 'EUR/MW', PRICE_COL_MW),
        ('afrr_capacity', 'aFRR Capacity', 'EUR/MW', PRICE_COL_MW),
        ('afrr_energy', 'aFRR Energy', 'EUR/MWh', PRICE_COL_MWH)
    ]

    # Create subplot layout
    if layout == '2x2':
        rows, cols = 2, 2
        subplot_titles = [config[1] for config in market_configs]
    else:  # horizontal
        rows, cols = 1, 4
        subplot_titles = [config[1] for config in market_configs]

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Plot each market
    for idx, (market_key, title, unit, price_col) in enumerate(market_configs):
        if market_key not in tables:
            continue

        df = tables[market_key]
        row = idx // cols + 1
        col = idx % cols + 1

        # Convert to tidy format for plotting
        if market_key == 'day_ahead':
            tidy_df = wide_to_tidy_day_ahead(df)
            # The conversion function returns 'price_eur_mwh' column
            actual_price_col = 'price_eur_mwh'
            # Special handling: Day-ahead has 'DE_LU' instead of 'DE'
            # Rename DE_LU to DE for consistency with other markets
            if 'DE_LU' in tidy_df[COUNTRY_COL].values:
                tidy_df[COUNTRY_COL] = tidy_df[COUNTRY_COL].replace('DE_LU', 'DE')
        elif market_key == 'fcr':
            tidy_df = wide_to_tidy_fcr(df)
            # The conversion function returns 'price_eur_mw' column
            actual_price_col = 'price_eur_mw'
        elif market_key == 'afrr_capacity':
            tidy_df = wide_to_tidy_afrr(df)
            # The conversion function returns 'price_eur_mw' column
            actual_price_col = 'price_eur_mw'
        elif market_key == 'afrr_energy':
            tidy_df = wide_to_tidy_afrr(df)
            # For aFRR energy, we need to convert the price column name
            # The wide_to_tidy_afrr returns 'price_eur_mw' but for energy it should be 'price_eur_mwh'
            # Let's check what column actually exists
            if 'price_eur_mw' in tidy_df.columns:
                # Rename it to match energy pricing
                tidy_df = tidy_df.rename(columns={'price_eur_mw': 'price_eur_mwh'})
                actual_price_col = 'price_eur_mwh'
            else:
                actual_price_col = 'price_eur_mwh'
        else:
            continue

        # Filter countries
        if COUNTRY_COL in tidy_df.columns:
            tidy_df = tidy_df[tidy_df[COUNTRY_COL].isin(countries)]
        
        # For aFRR markets, we might have direction column - need to aggregate both directions
        if 'direction' in tidy_df.columns:
            # Combine both Pos and Neg into the country data
            for country in countries:
                country_data = tidy_df[tidy_df[COUNTRY_COL] == country]
                if len(country_data) == 0:
                    continue

                fig.add_trace(
                    go.Box(
                        y=country_data[actual_price_col],
                        name=country,
                        marker_color=get_country_color(country),
                        showlegend=(idx == 0),  # Only show legend once
                        legendgroup=country,
                        boxmean='sd'  # Show mean and std dev
                    ),
                    row=row, col=col
                )
        else:
            # Regular markets without direction
            for country in countries:
                country_data = tidy_df[tidy_df[COUNTRY_COL] == country]
                if len(country_data) == 0:
                    continue

                fig.add_trace(
                    go.Box(
                        y=country_data[actual_price_col],
                        name=country,
                        marker_color=get_country_color(country),
                        showlegend=(idx == 0),  # Only show legend once
                        legendgroup=country,
                        boxmean='sd'  # Show mean and std dev
                    ),
                    row=row, col=col
                )

        # Update axes labels
        fig.update_xaxes(title_text="Country", row=row, col=col)
        fig.update_yaxes(title_text=f"Price ({unit})", row=row, col=col)

    # Update overall layout
    fig.update_layout(
        title_text="Market Price Distributions - Multi-Market Comparison",
        height=800 if layout == '2x2' else 400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15 if layout == '2x2' else -0.3,
            xanchor='center',
            x=0.5
        )
    )

    return fig


def plot_country_market_comparison(
    tables: Dict[str, pd.DataFrame],
    country: str,
    market_types: list = None
) -> go.Figure:
    """Compare price distributions across multiple markets for a single country.

    Creates a side-by-side box plot comparison showing how prices vary
    across different market types for a specific country.

    Parameters
    ----------
    tables : dict
        Dictionary with keys: 'day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy'
    country : str
        Country code (DE, AT, CH, HU, CZ)
    market_types : list, optional
        List of markets to include (default: all)

    Returns
    -------
    go.Figure
        Box plot comparison figure

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> fig = plot_country_market_comparison(tables, country='DE')
    >>> fig.show()
    """
    from src.visualization.config import MCKINSEY_COLORS, apply_mckinsey_style

    if market_types is None:
        market_types = ['day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy']

    fig = go.Figure()

    colors = [
        MCKINSEY_COLORS['cat_1'],
        MCKINSEY_COLORS['cat_2'],
        MCKINSEY_COLORS['teal'],
        MCKINSEY_COLORS['cat_5']
    ]

    # Define market configurations
    market_configs = {
        'day_ahead': ('Day-Ahead', 'EUR/MWh', 'DE_LU' if country == 'DE' else country),
        'fcr': ('FCR', 'EUR/MW', country),
        'afrr_capacity': ('aFRR Cap', 'EUR/MW', None),  # Has Pos/Neg
        'afrr_energy': ('aFRR Energy', 'EUR/MWh', None)  # Has Pos/Neg
    }

    x_labels = []
    color_idx = 0

    for market_key in market_types:
        if market_key not in tables or market_key not in market_configs:
            continue

        df = tables[market_key]
        label, unit, col_name = market_configs[market_key]

        # Handle markets with Pos/Neg directions
        if market_key in ['afrr_capacity', 'afrr_energy']:
            for direction in ['Pos', 'Neg']:
                col = f'{country}_{direction}'
                if col in df.columns:
                    prices = df[col].dropna()
                    if len(prices) > 0:
                        fig.add_trace(go.Box(
                            y=prices,
                            name=f'{label} ({direction})',
                            marker_color=colors[color_idx % len(colors)],
                            boxmean='sd'
                        ))
                        x_labels.append(f'{label}<br>({direction})')
                        color_idx += 1
        else:
            # Regular markets (day-ahead, FCR)
            if col_name and col_name in df.columns:
                prices = df[col_name].dropna()
                if len(prices) > 0:
                    fig.add_trace(go.Box(
                        y=prices,
                        name=label,
                        marker_color=colors[color_idx % len(colors)],
                        boxmean='sd'
                    ))
                    x_labels.append(label)
                    color_idx += 1

    # Apply styling
    fig = apply_mckinsey_style(
        fig,
        title=f'Price Distribution Comparison - {country} (All Markets)'
    )

    fig.update_layout(
        xaxis_title='Market Type',
        yaxis_title='Price (EUR)',
        height=500,
        showlegend=False
    )

    return fig


def plot_da_price_distribution_mckinsey(
    day_ahead_df: pd.DataFrame,
    country: str,
    bins: int = 50
) -> go.Figure:
    """Plot day-ahead price distribution with McKinsey styling.

    Module B: Price Distribution (DA)

    Creates a histogram with KDE overlay showing the frequency distribution
    of day-ahead prices.

    Parameters
    ----------
    day_ahead_df : pd.DataFrame
        Wide-format day-ahead data
    country : str
        Country code
    bins : int, optional
        Number of histogram bins (default: 50)

    Returns
    -------
    go.Figure
        Histogram with KDE overlay

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> fig = plot_da_price_distribution_mckinsey(tables['day_ahead'], country='DE')
    >>> fig.show()
    """
    from src.visualization.config import MCKINSEY_COLORS, apply_mckinsey_style
    import numpy as np
    from scipy import stats

    # Get prices for country
    country_col = 'DE_LU' if country == 'DE' else country

    if country_col not in day_ahead_df.columns:
        raise ValueError(f"Country {country} not found in day-ahead data")

    prices = day_ahead_df[country_col].dropna()

    # Create figure
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=prices,
        nbinsx=bins,
        name='Frequency',
        marker_color=MCKINSEY_COLORS['navy'],
        opacity=0.7,
        hovertemplate='Price: %{x:.2f} EUR/MWh<br>Count: %{y}<extra></extra>'
    ))

    # KDE overlay
    try:
        kde = stats.gaussian_kde(prices)
        x_range = np.linspace(prices.min(), prices.max(), 200)
        kde_values = kde(x_range)

        # Scale KDE to match histogram height
        kde_scaled = kde_values * len(prices) * (prices.max() - prices.min()) / bins

        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_scaled,
            mode='lines',
            name='Density',
            line=dict(color=MCKINSEY_COLORS['teal'], width=2),
            yaxis='y2',
            hovertemplate='Price: %{x:.2f} EUR/MWh<extra></extra>'
        ))
    except:
        pass  # Skip KDE if scipy not available or data issue

    # Add vertical lines for mean and median
    mean_price = prices.mean()
    median_price = prices.median()

    fig.add_vline(
        x=mean_price,
        line_dash="dash",
        line_color=MCKINSEY_COLORS['gray_dark'],
        annotation_text=f"Mean: {mean_price:.1f}",
        annotation_position="top"
    )

    fig.add_vline(
        x=median_price,
        line_dash="dot",
        line_color=MCKINSEY_COLORS['gray_dark'],
        annotation_text=f"Median: {median_price:.1f}",
        annotation_position="bottom"
    )

    # Apply McKinsey styling
    fig = apply_mckinsey_style(
        fig,
        title=f'Day-Ahead Price Distribution - {country}'
    )

    fig.update_layout(
        xaxis_title='Price (EUR/MWh)',
        yaxis_title='Frequency',
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=400,
        showlegend=True
    )

    return fig


def plot_da_price_distribution_multi_country_mckinsey(
    day_ahead_df: pd.DataFrame,
    countries: list = None,
    bins: int = 50
) -> go.Figure:
    """Plot day-ahead price distribution comparison across multiple countries.

    Module B: Multi-Country Price Distribution (DA)

    Creates overlaid histograms with KDE curves for comparing price distributions
    across countries, using McKinsey styling.

    Parameters
    ----------
    day_ahead_df : pd.DataFrame
        Wide-format day-ahead data
    countries : list, optional
        List of country codes to compare (default: ['DE', 'AT', 'CH', 'HU', 'CZ'])
    bins : int, optional
        Number of histogram bins (default: 50)

    Returns
    -------
    go.Figure
        Overlaid histograms with KDE curves

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> fig = plot_da_price_distribution_multi_country_mckinsey(tables['day_ahead'])
    >>> fig.show()
    """
    from src.visualization.config import COUNTRY_COLORS, apply_mckinsey_style, get_country_color
    import numpy as np
    from scipy import stats

    if countries is None:
        countries = ['DE', 'AT', 'CH', 'HU', 'CZ']

    fig = go.Figure()

    # Add histogram and KDE for each country
    for country in countries:
        country_col = 'DE_LU' if country == 'DE' else country

        if country_col not in day_ahead_df.columns:
            print(f"Warning: Country {country} not found, skipping")
            continue

        prices = day_ahead_df[country_col].dropna()
        color = get_country_color(country)

        # Histogram (semi-transparent)
        fig.add_trace(go.Histogram(
            x=prices,
            nbinsx=bins,
            name=f'{country} (Histogram)',
            marker_color=color,
            opacity=0.3,
            showlegend=False,
            hovertemplate=f'{country}<br>Price: %{{x:.2f}} EUR/MWh<br>Count: %{{y}}<extra></extra>'
        ))

        # KDE overlay (prominent line)
        try:
            kde = stats.gaussian_kde(prices)
            x_range = np.linspace(prices.min(), prices.max(), 200)
            kde_values = kde(x_range)

            # Scale KDE to match histogram height approximately
            kde_scaled = kde_values * len(prices) * (prices.max() - prices.min()) / bins

            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_scaled,
                mode='lines',
                name=f'{country}',
                line=dict(color=color, width=2.5),
                hovertemplate=f'{country}<br>Price: %{{x:.2f}} EUR/MWh<extra></extra>'
            ))
        except Exception as e:
            print(f"Warning: Could not compute KDE for {country}: {e}")

    # Apply McKinsey styling
    fig = apply_mckinsey_style(
        fig,
        title='Day-Ahead Price Distribution - Multi-Country Comparison'
    )

    fig.update_layout(
        xaxis_title='Price (EUR/MWh)',
        yaxis_title='Frequency',
        barmode='overlay',  # Overlay histograms
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig


def plot_da_price_ridgeline_mckinsey(
    day_ahead_df: pd.DataFrame,
    countries: list = None,
    bins: int = 100
):
    """Plot day-ahead price distribution as ridgeline plot (joy plot) with McKinsey styling.

    Module B: Multi-Country Price Distribution (Ridgeline)

    Creates a ridgeline (joy plot) visualization using matplotlib showing 
    overlapping KDE curves for comparing price distributions across countries. 
    This format provides better visual separation than overlaid histograms.

    Parameters
    ----------
    day_ahead_df : pd.DataFrame
        Wide-format day-ahead data
    countries : list, optional
        List of country codes to compare (default: ['DE', 'AT', 'CH', 'HU', 'CZ'])
    bins : int, optional
        Number of points for KDE calculation (default: 100)

    Returns
    -------
    matplotlib.figure.Figure
        Ridgeline plot with vertically offset KDE curves

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> fig = plot_da_price_ridgeline_mckinsey(tables['day_ahead'])
    >>> fig.show()  # or plt.show()
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from src.visualization.config import COUNTRY_COLORS, get_country_color
    import numpy as np
    from scipy import stats

    if countries is None:
        countries = ['DE', 'AT', 'CH', 'HU', 'CZ']

    # Prepare data
    all_prices = []
    kde_data = []
    
    for country in countries:
        country_col = 'DE_LU' if country == 'DE' else country
        
        if country_col not in day_ahead_df.columns:
            print(f"Warning: Country {country} not found, skipping")
            continue
            
        prices = day_ahead_df[country_col].dropna()
        
        if len(prices) < 10:
            print(f"Warning: Insufficient data for {country}, skipping")
            continue
            
        all_prices.extend(prices.values)
        kde_data.append({
            'country': country,
            'prices': prices.values,
            'mean': prices.mean(),
            'color': get_country_color(country)
        })
    
    if not kde_data:
        raise ValueError("No valid data found for specified countries")
    
    # Calculate global x range
    x_min, x_max = np.percentile(all_prices, [1, 99])
    x = np.linspace(x_min, x_max, bins)
    
    # Create figure with McKinsey styling
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # McKinsey color palette - convert hex to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Plot each country's distribution
    vertical_spacing = 0.05  # Spacing as fraction of max density
    
    for idx, data in enumerate(kde_data):
        # Calculate KDE
        kde = stats.gaussian_kde(data['prices'])
        density = kde(x)
        
        # Normalize to 0-1 range for consistent height
        density_norm = density / density.max()
        
        # Vertical offset
        y_offset = idx * vertical_spacing
        y = density_norm + y_offset
        
        # Get color
        color_hex = data['color']
        color_rgb = hex_to_rgb(color_hex)
        
        # Plot filled curve
        ax.fill_between(x, y_offset, y, 
                        alpha=0.7, 
                        color=color_rgb,
                        linewidth=0)
        
        # Plot outline
        ax.plot(x, y, 
               color=color_rgb, 
               linewidth=2.5,
               solid_capstyle='round')
        
        # Plot baseline
        ax.plot(x, np.full_like(x, y_offset), 
               color=color_rgb, 
               linewidth=1.5, 
               alpha=0.5)
        
        # Add mean line
        mean_val = data['mean']
        density_at_mean = kde(mean_val)[0] / kde(x).max()
        ax.plot([mean_val, mean_val], 
               [y_offset, y_offset + density_at_mean * 0.7],
               color=color_rgb,
               linewidth=2.5,
               linestyle='--',
               alpha=0.8)
        
        # Add country label
        ax.text(x_min - (x_max - x_min) * 0.02, 
               y_offset + 0.4,
               data['country'],
               fontsize=14,
               fontweight='bold',
               color=color_rgb,
               ha='right',
               va='center',
               family='sans-serif')
    
    # Styling
    ax.set_xlim(x_min - (x_max - x_min) * 0.05, x_max + (x_max - x_min) * 0.02)
    ax.set_ylim(-0.01, len(kde_data) * vertical_spacing + 1.1)
    
    # Remove y-axis
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # X-axis styling
    ax.set_xlabel('Price (EUR/MWh)', fontsize=12, fontweight='bold', family='sans-serif')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='x', labelsize=11, length=6, width=1.5)
    
    # Title
    ax.set_title('Day-Ahead Price Distribution - Multi-Country Comparison',
                fontsize=16, fontweight='bold', pad=20, family='sans-serif')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)
    
    # Background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def plot_da_price_heatmap_mckinsey(
    day_ahead_df: pd.DataFrame,
    country: str
) -> go.Figure:
    """Plot hour-of-day vs month heatmap with McKinsey styling.

    Module C: DA Price Heatmap

    Creates a 2D heatmap showing average day-ahead prices by hour and month.

    Parameters
    ----------
    day_ahead_df : pd.DataFrame
        Wide-format day-ahead data with timestamp column
    country : str
        Country code

    Returns
    -------
    go.Figure
        Heatmap visualization

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> fig = plot_da_price_heatmap_mckinsey(tables['day_ahead'], country='DE')
    >>> fig.show()
    """
    from src.visualization.config import MCKINSEY_COLORS, apply_mckinsey_style

    # Get data for country
    country_col = 'DE_LU' if country == 'DE' else country

    if country_col not in day_ahead_df.columns:
        raise ValueError(f"Country {country} not found in day-ahead data")

    df = day_ahead_df[[TIMESTAMP_COL, country_col]].copy()
    df['hour'] = df[TIMESTAMP_COL].dt.hour
    df['month'] = df[TIMESTAMP_COL].dt.month

    # Pivot to create hour x month matrix
    pivot = df.pivot_table(
        index='hour',
        columns='month',
        values=country_col,
        aggfunc='mean'
    )

    # Create custom colorscale (diverging: negative=blue, zero=white, positive=red)
    colorscale = [
        [0.0, '#003f5c'],   # Dark blue (negative)
        [0.25, '#2f4b7c'],  # Blue
        [0.5, '#ffffff'],   # White (zero)
        [0.75, '#ff6361'],  # Coral
        [1.0, '#bc5090']    # Purple (high positive)
    ]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=list(range(24)),
        colorscale=colorscale,
        colorbar=dict(
            title=dict(
                text='Avg Price<br>(EUR/MWh)',
                side='right'
            )
        ),
        hovertemplate='Month: %{x}<br>Hour: %{y}:00<br>Avg Price: %{z:.2f} EUR/MWh<extra></extra>'
    ))

    # Apply McKinsey styling
    fig = apply_mckinsey_style(
        fig,
        title=f'Day-Ahead Price Pattern - {country}'
    )

    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Hour of Day',
        height=500,
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2  # Show every 2 hours
        )
    )

    return fig


def calculate_price_statistics_mckinsey(
    tables: Dict[str, pd.DataFrame],
    country: str,
    market: str = 'day_ahead'
) -> pd.DataFrame:
    """Calculate comprehensive price statistics.

    Module D: Price Statistics

    Calculates descriptive statistics for a selected market and country.

    Parameters
    ----------
    tables : dict
        All market tables
    country : str
        Country code
    market : str, optional
        One of: 'day_ahead', 'fcr', 'afrr_capacity_pos', 'afrr_capacity_neg',
                'afrr_energy_pos', 'afrr_energy_neg' (default: 'day_ahead')

    Returns
    -------
    pd.DataFrame
        Statistics table ready for display

    Example
    -------
    >>> tables = load_phase2_market_tables(Path("data/TechArena2025_Phase2_data.xlsx"))
    >>> stats = calculate_price_statistics_mckinsey(tables, country='DE', market='day_ahead')
    >>> print(stats)
    """
    # Get appropriate data
    if market == 'day_ahead':
        col = 'DE_LU' if country == 'DE' else country
        data = tables['day_ahead'][col]
        unit = 'EUR/MWh'
    elif market == 'fcr':
        data = tables['fcr'][country]
        unit = 'EUR/MW'
    elif market.startswith('afrr_capacity'):
        direction = 'Pos' if 'pos' in market else 'Neg'
        data = tables['afrr_capacity'][f'{country}_{direction}']
        unit = 'EUR/MW'
    elif market.startswith('afrr_energy'):
        direction = 'Pos' if 'pos' in market else 'Neg'
        data = tables['afrr_energy'][f'{country}_{direction}']
        unit = 'EUR/MWh'
    else:
        raise ValueError(f"Unknown market: {market}")

    # Calculate statistics
    stats = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range',
                   '10th Percentile', '90th Percentile'],
        'Value': [
            f"{data.mean():.2f}",
            f"{data.median():.2f}",
            f"{data.std():.2f}",
            f"{data.min():.2f}",
            f"{data.max():.2f}",
            f"{data.max() - data.min():.2f}",
            f"{data.quantile(0.1):.2f}",
            f"{data.quantile(0.9):.2f}"
        ],
        'Unit': [unit] * 8
    }

    return pd.DataFrame(stats)


def plot_price_statistics_mckinsey(
    stats_df: pd.DataFrame,
    country: str,
    market: str
) -> go.Figure:
    """Display statistics as a clean table figure.

    Module D: Price Statistics (Visualization)

    Creates a professional table visualization for price statistics.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Statistics from calculate_price_statistics_mckinsey()
    country : str
        Country code
    market : str
        Market name for title

    Returns
    -------
    go.Figure
        Table visualization

    Example
    -------
    >>> stats = calculate_price_statistics_mckinsey(tables, 'DE', 'day_ahead')
    >>> fig = plot_price_statistics_mckinsey(stats, 'DE', 'day_ahead')
    >>> fig.show()
    """
    from src.visualization.config import MCKINSEY_COLORS, MCKINSEY_FONTS

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>', '<b>Unit</b>'],
            fill_color=MCKINSEY_COLORS['navy'],
            font=dict(color='white', size=MCKINSEY_FONTS['axis_label_size']),
            align='left',
            height=30
        ),
        cells=dict(
            values=[stats_df['Metric'], stats_df['Value'], stats_df['Unit']],
            fill_color=[[MCKINSEY_COLORS['bg_light_gray'], 'white'] * 4],
            font=dict(size=MCKINSEY_FONTS['tick_label_size']),
            align='left',
            height=25
        )
    )])

    fig.update_layout(
        title=f'Price Statistics - {market.replace("_", " ").title()} - {country}',
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


# ---------------------------------------------------------------------------
# CLI bootstrap
# ---------------------------------------------------------------------------


def _cli_example(workbook: Path) -> None:
    tables = load_market_tables(workbook)
    volatility = summarize_day_ahead(tables["day_ahead"])
    fcr_summary = summarize_fcr(tables["fcr"])
    afrr_summary = summarize_afrr(tables["afrr"])

    print("Top 5 arbitrage opportunities (Day-Ahead volatility):")
    print(volatility.head())
    print("\nFCR average prices by country:")
    print(fcr_summary.sort_values("mean", ascending=False))
    print("\naFRR summary (positive vs negative):")
    print(afrr_summary)


# if __name__ == "__main__":
#     default_path = Path(__file__).resolve().parents[1] / "SoloGen_TechArena2025_Phase1" / "input" / "TechArena2025_data.xlsx"
#     if default_path.exists():
#         _cli_example(default_path)
#     else:
#         raise SystemExit(
#             "Unable to locate the default workbook. Pass a valid path or adjust the script."
#         )
