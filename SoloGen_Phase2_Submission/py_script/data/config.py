#!/usr/bin/env python3
"""
McKinsey-Style Visualization Configuration for TechArena 2025
==============================================================

This module provides professional visualization styling following McKinsey
presentation standards for all dashboard and report visualizations.

Key Features:
- Clean, minimal design with high data-ink ratio
- Professional color palette (navy blues, grays, strategic accents)
- Reusable Plotly template for consistent styling
- Typography standards for executive-friendly presentations

Author: SoloGen Team
Date: October 2025
"""

import plotly.graph_objects as go
import plotly.io as pio

# ============================================================================
# Color Palette
# ============================================================================

MCKINSEY_COLORS = {
    # Primary colors
    'navy': '#003f5c',           # Main data series, titles
    'dark_blue': '#2f4b7c',      # Secondary data, accents
    'teal': '#00a99d',           # Highlights, key insights

    # Supporting colors
    'gray_dark': '#505050',      # Axes, labels
    'gray_medium': '#808080',    # Gridlines (when used)
    'gray_light': '#d3d3d3',     # Backgrounds, subtle elements

    # Categorical palette (for multiple countries)
    'cat_1': '#003f5c',  # Germany (navy)
    'cat_2': '#2f4b7c',  # Austria (dark blue)
    'cat_3': '#00a99d',  # Switzerland (teal)
    'cat_4': '#bc5090',  # Hungary (purple)
    'cat_5': '#ff6361',  # Czech Republic (coral)

    # Diverging colors (for Pos/Neg, charge/discharge)
    'positive': '#00a99d',  # Teal
    'negative': '#ff6361',  # Coral

    # Backgrounds
    'bg_white': '#ffffff',
    'bg_light_gray': '#f8f9fa',
}

# Country-specific color mapping
COUNTRY_COLORS = {
    'DE': MCKINSEY_COLORS['cat_1'],
    'DE_LU': MCKINSEY_COLORS['cat_1'],  # Alias for Germany
    'AT': MCKINSEY_COLORS['cat_2'],
    'CH': MCKINSEY_COLORS['cat_3'],
    'HU': MCKINSEY_COLORS['cat_4'],
    'CZ': MCKINSEY_COLORS['cat_5'],
}

# ============================================================================
# Typography
# ============================================================================

MCKINSEY_FONTS = {
    'family': 'Arial, Helvetica, sans-serif',
    'title_size': 16,
    'subtitle_size': 14,
    'axis_label_size': 12,
    'tick_label_size': 10,
    'legend_size': 10,
}

# ============================================================================
# Grid and Axes
# ============================================================================

MCKINSEY_GRID = {
    'show_grid': True,
    'grid_color': MCKINSEY_COLORS['gray_light'],
    'grid_width': 0.5,
    'show_minor_grid': False,
    'axis_line_width': 1,
    'axis_line_color': MCKINSEY_COLORS['gray_dark'],
}

# ============================================================================
# Plotly Template
# ============================================================================

def create_mckinsey_template():
    """
    Create a reusable Plotly template with McKinsey styling.

    This template can be applied globally to all Plotly figures for
    consistent professional appearance across all visualizations.

    Returns
    -------
    plotly.graph_objects.layout.Template
        McKinsey-styled Plotly template

    Example
    -------
    >>> import plotly.io as pio
    >>> pio.templates.default = "mckinsey"
    >>> fig = px.line(df, x='date', y='price')
    >>> fig.show()  # Automatically uses McKinsey styling
    """
    template = go.layout.Template()

    # Layout defaults
    template.layout = go.Layout(
        font=dict(
            family=MCKINSEY_FONTS['family'],
            size=MCKINSEY_FONTS['axis_label_size'],
            color=MCKINSEY_COLORS['gray_dark']
        ),
        title=dict(
            font=dict(
                size=MCKINSEY_FONTS['title_size'],
                color=MCKINSEY_COLORS['navy'],
                family=MCKINSEY_FONTS['family']
            ),
            x=0.05,  # Left-aligned titles
            xanchor='left'
        ),
        paper_bgcolor=MCKINSEY_COLORS['bg_white'],
        plot_bgcolor=MCKINSEY_COLORS['bg_light_gray'],

        # X-axis styling
        xaxis=dict(
            showgrid=MCKINSEY_GRID['show_grid'],
            gridcolor=MCKINSEY_GRID['grid_color'],
            gridwidth=MCKINSEY_GRID['grid_width'],
            showline=True,
            linewidth=MCKINSEY_GRID['axis_line_width'],
            linecolor=MCKINSEY_GRID['axis_line_color'],
            ticks='outside',
            tickfont=dict(size=MCKINSEY_FONTS['tick_label_size']),
            title_font=dict(size=MCKINSEY_FONTS['axis_label_size'])
        ),

        # Y-axis styling
        yaxis=dict(
            showgrid=MCKINSEY_GRID['show_grid'],
            gridcolor=MCKINSEY_GRID['grid_color'],
            gridwidth=MCKINSEY_GRID['grid_width'],
            showline=True,
            linewidth=MCKINSEY_GRID['axis_line_width'],
            linecolor=MCKINSEY_GRID['axis_line_color'],
            ticks='outside',
            tickfont=dict(size=MCKINSEY_FONTS['tick_label_size']),
            title_font=dict(size=MCKINSEY_FONTS['axis_label_size']),
            zeroline=False
        ),

        # Legend styling
        legend=dict(
            font=dict(size=MCKINSEY_FONTS['legend_size']),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=MCKINSEY_COLORS['gray_light'],
            borderwidth=1
        ),

        # Margins - generous white space
        margin=dict(l=80, r=60, t=80, b=60),

        # Hover label styling
        hoverlabel=dict(
            bgcolor=MCKINSEY_COLORS['bg_white'],
            font_size=MCKINSEY_FONTS['tick_label_size'],
            font_family=MCKINSEY_FONTS['family']
        )
    )

    # Color sequence for categorical data
    template.layout.colorway = [
        MCKINSEY_COLORS['cat_1'],
        MCKINSEY_COLORS['cat_2'],
        MCKINSEY_COLORS['cat_3'],
        MCKINSEY_COLORS['cat_4'],
        MCKINSEY_COLORS['cat_5'],
    ]

    return template


# ============================================================================
# Template Registration
# ============================================================================

# Register the McKinsey template globally
pio.templates["mckinsey"] = create_mckinsey_template()

# Optionally set as default (can be overridden by user)
# pio.templates.default = "mckinsey"


# ============================================================================
# Utility Functions
# ============================================================================

def get_country_color(country_code: str) -> str:
    """
    Get the McKinsey color for a specific country.

    Parameters
    ----------
    country_code : str
        Country code (DE, AT, CH, HU, CZ, or DE_LU)

    Returns
    -------
    str
        Hex color code

    Example
    -------
    >>> color = get_country_color('DE')
    >>> print(color)
    '#003f5c'
    """
    return COUNTRY_COLORS.get(country_code, MCKINSEY_COLORS['gray_dark'])


def apply_mckinsey_style(fig: go.Figure, title: str = None) -> go.Figure:
    """
    Apply McKinsey styling to an existing Plotly figure.

    This function can be used to style figures that weren't created
    with the McKinsey template.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to style
    title : str, optional
        Title to set (if different from existing)

    Returns
    -------
    go.Figure
        Styled figure

    Example
    -------
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6]))
    >>> fig = apply_mckinsey_style(fig, title="Sample Chart")
    """
    # Apply template
    fig.update_layout(template="mckinsey")

    # Set title if provided
    if title:
        fig.update_layout(title=title)

    return fig


if __name__ == "__main__":
    # Demo: Show McKinsey template in action
    import pandas as pd
    import numpy as np

    print("McKinsey Visualization Configuration")
    print("=" * 50)
    print(f"\nColor Palette:")
    for name, color in MCKINSEY_COLORS.items():
        print(f"  {name:15s}: {color}")

    print(f"\nCountry Colors:")
    for country, color in COUNTRY_COLORS.items():
        print(f"  {country:15s}: {color}")

    print(f"\nTemplate registered as: 'mckinsey'")
    print(f"Available templates: {list(pio.templates.keys())}")

    # Create a sample figure
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'price_de': np.random.uniform(20, 100, 100),
        'price_at': np.random.uniform(15, 90, 100)
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price_de'],
        mode='lines',
        name='Germany',
        line=dict(color=COUNTRY_COLORS['DE'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price_at'],
        mode='lines',
        name='Austria',
        line=dict(color=COUNTRY_COLORS['AT'], width=2)
    ))

    fig = apply_mckinsey_style(fig, title='Sample Price Comparison')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (EUR/MWh)'
    )

    # Save demo figure
    fig.write_html('mckinsey_template_demo.html')
    print(f"\nDemo figure saved to: mckinsey_template_demo.html")
