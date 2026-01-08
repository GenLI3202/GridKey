# %% [markdown]
# # TechArena 2025 Phase 2: Electricity Market Analysis
# ## McKinsey-Style Data Visualization
# 
# **Purpose**: Comprehensive analysis of European electricity market data across 4 markets:
# - Day-Ahead Energy Market (15-min, EUR/MWh)
# - FCR Capacity Market (4-hour blocks, EUR/MW)
# - aFRR Capacity Market (4-hour blocks, EUR/MW, Pos/Neg)
# - aFRR Energy Market (15-min, EUR/MWh, Pos/Neg)
# 
# **Countries**: Germany (DE), Austria (AT), Switzerland (CH), Hungary (HU), Czech Republic (CZ)
# 
# ---

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
# Import standard libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
import plotly.graph_objects as go
import plotly.io as pio
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add parent directory to path for imports
# # Get the current notebook directory first
# current_dir = os.path.dirname(os.path.abspath('phase2_market_analysis_mckinsey.ipynb'))
# # Go up one level to the main project directory
# project_root = os.path.dirname(current_dir)
# sys.path.insert(0, project_root)
# py_script_dir = os.path.join(project_root, 'py_script')
# sys.path.insert(0, py_script_dir)   


import importlib

# Reload load_process_market_data
if 'src.data.load_process_market_data' in sys.modules:
    import src.data.load_process_market_data
    importlib.reload(src.data.load_process_market_data)
# Reload visualize_market_data
if 'src.data.visualize_market_data' in sys.modules:
    import src.data.visualize_market_data
    importlib.reload(src.data.visualize_market_data)
# Reload visualization.config (if it exists)
if 'src.data.config' in sys.modules:
    import src.data.config
    importlib.reload(src.data.config)


# Import custom modules

from src.data.load_process_market_data import (
    load_phase2_market_tables,
    validate_phase2_data,
    convert_afrr_energy_zero_to_nan,
    TIMESTAMP_COL, COUNTRY_COL, PRICE_COL_MWH, PRICE_COL_MW, DIRECTION_COL, AFRR_DIRECTION_ALIASES
)

from src.data.visualize_market_data import (
    # McKinsey plotting functions
    plot_price_time_series_mckinsey,
    plot_da_price_distribution_mckinsey,
    plot_da_price_distribution_multi_country_mckinsey,
    plot_da_price_ridgeline_mckinsey,
    plot_da_price_heatmap_mckinsey,
    calculate_price_statistics_mckinsey,
    plot_price_statistics_mckinsey,
    # Basic plotting functions
    plot_market_distribution,
    plot_all_markets_distribution,
    plot_country_market_comparison,
    plot_fcr_distribution,
    plot_afrr_distribution,
    plot_day_ahead_trend,
    summarize_day_ahead,
    summarize_fcr,
    summarize_afrr
)

from src.data.config import (
    MCKINSEY_COLORS,
    COUNTRY_COLORS,
    apply_mckinsey_style,
    get_country_color
)

# Configure Plotly to use McKinsey template
pio.templates.default = "mckinsey"

print("✓ Libraries imported successfully")
print(f"✓ McKinsey template activated")

# %%
project_root

# %%
# Load Phase 2 market data from Excel file
from pathlib import Path

# Path to the Phase 2 Excel workbook
phase2_data_path = project_root / 'data' / 'TechArena2025_Phase2_data.xlsx'

print(f"Loading Phase 2 market data from: {phase2_data_path}")
print(f"File exists: {phase2_data_path.exists()}\n")

# Load all Phase 2 market tables (day_ahead, fcr, afrr_capacity, afrr_energy)
p2_market_tables = load_phase2_market_tables(phase2_data_path)

# Display what was loaded
print("Loaded tables:")
for table_name, table_df in p2_market_tables.items():
    print(f"  ✓ {table_name:20s}: {len(table_df):,} rows x {len(table_df.columns)} columns")

# %%
tables = p2_market_tables.copy()

# %%
tables["day_ahead"]

# %%
# Load metadata
with open(f'{project_root}/data/metadata.json', 'r') as f:
    import json
    metadata = json.load(f)

print(f"\n📅 Data Time Range: {metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
print(f"📊 Processing Date: {metadata['processing_timestamp'][:10]}")

# %%
# Quick data preview
print("Data Preview - Day-Ahead Market:\n")
display(tables['day_ahead'].head(5))

print("\nData Preview - aFRR Energy Market:\n")
display(tables['afrr_energy'].head(5))

# %% [markdown]
# ## 2. Day-Ahead Market Analysis
# 
# The day-ahead market is the primary energy trading market, with 15-minute resolution pricing in EUR/MWh.

# %% [markdown]
# ### 2.1 Multi-Country Price Comparison

# %%
tables.keys()  # List available tables

# %%
# Plot day-ahead price distribution across all countries
table_name = 'day_ahead'
fig = plot_market_distribution(tables[table_name], table_name)
fig.update_layout(height=500, title=f'{table_name} Price Distribution by Country (2024)')
fig.show()

# %%
# NEW: Multi-Market Distribution Comparison (2x2 Grid)
# Shows all 4 markets: Day-Ahead, FCR, aFRR Capacity, aFRR Energy

fig1 = plot_all_markets_distribution(tables, countries=['DE', 'AT', 'CH', 'HU', 'CZ'])
fig1.show()
fig1.write_html('multi_market_distribution_comparison.html')

# TODO: aFRR-C and aFRR-E should have both directions.

# %%
fig2 = plot_country_market_comparison(tables, country='DE')
fig2.show()

# %%
fig3 = plot_price_time_series_mckinsey(tables, country='CZ', time_range='full',)
fig3.update_layout(height=600, width=1200)
fig3.show()

# %%


# %% [markdown]
# # NEW: Multi-Market Distribution Comparison (2x2 Grid)
# fig1 = plot_all_markets_distribution(tables, countries=['DE', 'AT', 'CH'])
# fig1.show()
# import importlib
# 
# # Remove from cache
# if 'py_script.data.visualize_market_data' in sys.modules:
#     del sys.modules['py_script.data.visualize_market_data']
# if 'py_script.data.load_process_market_data' in sys.modules:
#     del sys.modules['py_script.data.load_process_market_data']
# 
# # Reimport
# from src.data.visualize_market_data import plot_all_markets_distribution
# 
# # Test the fixed function
# fig1 = plot_all_markets_distribution(tables, countries=['DE', 'AT', 'CH'])
# fig1.show()rtlib
# if 'py_script.data.visualize_market_data' in sys.modules:
#     import src.data.visualize_market_data
#     importlib.reload(py_script.data.visualize_market_data)
#     from src.data.visualize_market_data import plot_all_markets_distribution
# 
# # Test the fixed function
# fig1 = plot_all_markets_distribution(tables, countries=['DE', 'AT', 'CH'])
# fig1.show()

# %%
# Germany price time series (full year)
country_name = 'DE'
market_type = 'afrr_capacity' # also 'fcr', 'afrr_capacity', 'afrr_energy' or 'day_ahead'

fig = plot_price_time_series_mckinsey(
    tables, 
    country=country_name, 
    time_range='full',
    markets=[market_type]  # Only day-ahead for clarity
)
fig.update_layout(width=1200, height=500)
fig.show()

# %%
# Day-Ahead price distribution - 5 countries comparison (Ridgeline Plot, McKinsey style)
# Using matplotlib for better control and cleaner visualization
import matplotlib.pyplot as plt

fig = plot_da_price_ridgeline_mckinsey(
    tables['day_ahead'], 
    countries=['DE', 'AT', 'CH', 'HU', 'CZ'],
    bins=100
)
plt.show()

# %%
plot_da_price_distribution_multi_country_mckinsey(
    tables['day_ahead'], 
    countries=['DE', 'AT', 'CH', 'HU', 'CZ'],
    bins=50
)

# %%
tables.keys()

# %%
# Germany price heatmap (Hour x Month pattern)
fig = plot_da_price_heatmap_mckinsey(
    tables['day_ahead'],
    country='HU'
)
fig.show()

# %%
# Germany price statistics table
stats_de = calculate_price_statistics_mckinsey(
    tables, 
    country='DE', 
    market='day_ahead'
)

fig = plot_price_statistics_mckinsey(stats_de, 'DE', 'Day-Ahead')
fig.show()

# Also show as DataFrame
display(stats_de)

# %% [markdown]
# ### 2.3 Quarterly Analysis (Germany)

# %%
# Compare all quarters for Germany
from plotly.subplots import make_subplots

quarters = ['Q1', 'Q2', 'Q3', 'Q4']
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
)

for idx, quarter in enumerate(quarters):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    # Get quarterly data
    quarter_map = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
    months = quarter_map[quarter]
    df_q = tables['day_ahead'][tables['day_ahead']['timestamp'].dt.month.isin(months)]
    
    country_col = 'DE_LU'
    fig.add_trace(
        go.Scatter(
            x=df_q['timestamp'],
            y=df_q[country_col],
            mode='lines',
            name=quarter,
            line=dict(color=COUNTRY_COLORS['DE'], width=1),
            showlegend=(idx==0)
        ),
        row=row, col=col
    )

fig.update_xaxes(title_text='Date', row=2)
fig.update_yaxes(title_text='Price (EUR/MWh)')
fig = apply_mckinsey_style(fig, title='Germany Day-Ahead Prices - Quarterly Breakdown')
fig.update_layout(height=600, showlegend=True)
fig.show()

# %% [markdown]
# ### 2.4 Cross-Country Comparison

# %%
# Compare statistics across all countries
countries = ['DE', 'AT', 'CH', 'HU', 'CZ']
stats_all = []

for country in countries:
    stats = calculate_price_statistics_mckinsey(
        tables, 
        country=country, 
        market='day_ahead'
    )
    stats['Country'] = country
    stats_all.append(stats)

# Combine into single DataFrame
df_stats_combined = pd.concat(stats_all, ignore_index=True)

# Pivot for better visualization
df_pivot = df_stats_combined.pivot(index='Metric', columns='Country', values='Value')

print("Day-Ahead Price Statistics - All Countries:\n")
display(df_pivot)

# %%
# Visualize mean prices across countries
mean_prices = []
for country in countries:
    col = 'DE_LU' if country == 'DE' else country
    mean_prices.append(tables['day_ahead'][col].mean())

fig = go.Figure()
fig.add_trace(go.Bar(
    x=countries,
    y=mean_prices,
    marker_color=[get_country_color(c) for c in countries],
    text=[f'{p:.1f}' for p in mean_prices],
    textposition='outside'
))

fig = apply_mckinsey_style(fig, title='Average Day-Ahead Prices by Country (2024)')
fig.update_layout(
    xaxis_title='Country',
    yaxis_title='Mean Price (EUR/MWh)',
    height=400,
    showlegend=False
)
fig.show()

# %% [markdown]
# ## 3. FCR Capacity Market Analysis
# 
# Frequency Containment Reserve (FCR) capacity market with 4-hour block pricing in EUR/MW.

# %%
# FCR price distribution across countries
fig = plot_fcr_distribution(tables['fcr'])
fig.update_layout(height=500, title='FCR Capacity Price Distribution by Country (2024)')
fig.show()

# %%
# FCR time series for Germany
fig = plot_price_time_series_mckinsey(
    tables, 
    country='DE', 
    time_range='full',
    markets=['fcr']
)
fig.update_layout(height=500)
fig.show()

# %%
# FCR statistics for Germany
stats_fcr_de = calculate_price_statistics_mckinsey(
    tables, 
    country='DE', 
    market='fcr'
)

fig = plot_price_statistics_mckinsey(stats_fcr_de, 'DE', 'FCR Capacity')
fig.show()

# %% [markdown]
# ## 4. aFRR Capacity Market Analysis
# 
# Automatic Frequency Restoration Reserve capacity market with positive and negative directions.

# %%
# aFRR capacity distribution (Pos vs Neg)
fig = plot_afrr_distribution(tables['afrr_capacity'])
fig.update_layout(height=500, title='aFRR Capacity Price Distribution (Positive vs Negative)')
fig.show()

# %%
# aFRR capacity time series (Germany - both directions)
fig = plot_price_time_series_mckinsey(
    tables, 
    country='DE', 
    time_range='full',
    markets=['afrr_capacity']
)
fig.update_layout(height=500)
fig.show()

# %%
# Compare Positive vs Negative aFRR capacity for Germany
df_afrr_cap = tables['afrr_capacity']

fig = go.Figure()

fig.add_trace(go.Box(
    y=df_afrr_cap['DE_Pos'],
    name='Positive',
    marker_color=MCKINSEY_COLORS['positive']
))

fig.add_trace(go.Box(
    y=df_afrr_cap['DE_Neg'],
    name='Negative',
    marker_color=MCKINSEY_COLORS['negative']
))

fig = apply_mckinsey_style(fig, title='Germany aFRR Capacity - Positive vs Negative Distribution')
fig.update_layout(
    yaxis_title='Price (EUR/MW)',
    height=400,
    showlegend=True
)
fig.show()

# %% [markdown]
# ## 5. aFRR Energy Market Analysis (NEW in Phase 2)
# 
# aFRR energy activation prices with 15-minute resolution in EUR/MWh.

# %%
# aFRR energy distribution (Pos vs Neg)
fig = plot_afrr_distribution(tables['afrr_energy'])
fig.update_layout(height=500, title='aFRR Energy Price Distribution (Positive vs Negative)')
fig.show()

# %%
# aFRR energy time series (Germany)
fig = plot_price_time_series_mckinsey(
    tables, 
    country='DE', 
    time_range='Q1',  # Use Q1 for clearer visualization
    markets=['afrr_energy']
)
fig.update_layout(height=500)
fig.show()

# %%
# Check for zero activation prices (common when no activation occurs)
df_afrr_energy = tables['afrr_energy']

zero_analysis = []
for country in ['DE', 'AT', 'CH', 'HU', 'CZ']:
    for direction in ['Pos', 'Neg']:
        col = f'{country}_{direction}'
        if col in df_afrr_energy.columns:
            total = len(df_afrr_energy)
            zeros = (df_afrr_energy[col] == 0).sum()
            zero_pct = zeros / total * 100
            mean_non_zero = df_afrr_energy[df_afrr_energy[col] != 0][col].mean()
            
            zero_analysis.append({
                'Country': country,
                'Direction': direction,
                'Zero Count': zeros,
                'Zero %': f'{zero_pct:.1f}%',
                'Mean (Non-Zero)': f'{mean_non_zero:.2f}' if not pd.isna(mean_non_zero) else 'N/A'
            })

df_zero_analysis = pd.DataFrame(zero_analysis)

print("aFRR Energy Price - Zero Activation Analysis:\n")
print("(High % of zeros indicates periods with no reserve activation)\n")
display(df_zero_analysis)

# %% [markdown]
# ## 6. Multi-Market Comparison (Germany)

# %%
# All markets for Germany in one view (Q1 only for clarity)
fig = plot_price_time_series_mckinsey(
    tables, 
    country='DE', 
    time_range='Q1',
    markets=['day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy']
)
fig.update_layout(height=600)
fig.show()

# %% [markdown]
# ## 7. Price Volatility Analysis

# %%
# Calculate daily volatility (standard deviation) for day-ahead prices
df_da = tables['day_ahead'].copy()
df_da['date'] = df_da['timestamp'].dt.date

volatility_data = []
for country in ['DE', 'AT', 'CH', 'HU', 'CZ']:
    col = 'DE_LU' if country == 'DE' else country
    daily_vol = df_da.groupby('date')[col].std()
    volatility_data.append({
        'country': country,
        'mean_daily_volatility': daily_vol.mean(),
        'max_daily_volatility': daily_vol.max(),
        'dates': pd.to_datetime(daily_vol.index),
        'volatility': daily_vol.values
    })

# Plot volatility over time
fig = go.Figure()

for data in volatility_data:
    fig.add_trace(go.Scatter(
        x=data['dates'],
        y=data['volatility'],
        mode='lines',
        name=data['country'],
        line=dict(color=get_country_color(data['country']), width=1.5)
    ))

fig = apply_mckinsey_style(fig, title='Day-Ahead Price Volatility (Daily Std Dev)')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Daily Standard Deviation (EUR/MWh)',
    height=500,
    hovermode='x unified'
)
fig.show()

# %%
# Volatility comparison bar chart
countries_vol = [d['country'] for d in volatility_data]
mean_vols = [d['mean_daily_volatility'] for d in volatility_data]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=countries_vol,
    y=mean_vols,
    marker_color=[get_country_color(c) for c in countries_vol],
    text=[f'{v:.1f}' for v in mean_vols],
    textposition='outside'
))

fig = apply_mckinsey_style(fig, title='Average Daily Price Volatility by Country')
fig.update_layout(
    xaxis_title='Country',
    yaxis_title='Mean Daily Std Dev (EUR/MWh)',
    height=400,
    showlegend=False
)
fig.show()

# %% [markdown]
# ## 8. Price Correlation Analysis

# %%
# Calculate correlation matrix for day-ahead prices
df_da = tables['day_ahead'].copy()

# Rename DE_LU to DE for cleaner display
df_da = df_da.rename(columns={'DE_LU': 'DE'})

price_cols = ['DE', 'AT', 'CH', 'HU', 'CZ']
corr_matrix = df_da[price_cols].corr()

print("Day-Ahead Price Correlation Matrix:\n")
display(corr_matrix.round(3))

# %%
# Visualize correlation matrix as heatmap
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    zmin=-1,
    zmax=1,
    text=corr_matrix.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 12},
    colorbar=dict(title='Correlation')
))

fig = apply_mckinsey_style(fig, title='Day-Ahead Price Correlation Matrix')
fig.update_layout(
    height=500,
    xaxis_title='Country',
    yaxis_title='Country'
)
fig.show()

# %% [markdown]
# ## 9. Summary Report

# %%
# Generate comprehensive summary report
print("="*80)
print("TECHARENA 2025 PHASE 2 - MARKET ANALYSIS SUMMARY REPORT")
print("="*80)

print(f"\n📅 Analysis Period: {metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
print(f"📊 Total Data Points: {len(tables['day_ahead']):,} timestamps (15-min resolution)")

print("\n" + "="*80)
print("DAY-AHEAD MARKET SUMMARY")
print("="*80)

for country in ['DE', 'AT', 'CH', 'HU', 'CZ']:
    col = 'DE_LU' if country == 'DE' else country
    prices = tables['day_ahead'][col]
    print(f"\n{country}:")
    print(f"  Mean Price:      {prices.mean():8.2f} EUR/MWh")
    print(f"  Median Price:    {prices.median():8.2f} EUR/MWh")
    print(f"  Std Dev:         {prices.std():8.2f} EUR/MWh")
    print(f"  Price Range:     [{prices.min():6.2f}, {prices.max():6.2f}] EUR/MWh")
    print(f"  Negative Hours:  {(prices < 0).sum():8d} ({(prices < 0).sum() / len(prices) * 100:.1f}%)")

print("\n" + "="*80)
print("FCR CAPACITY MARKET SUMMARY")
print("="*80)

for country in ['DE', 'AT', 'CH', 'HU', 'CZ']:
    if country in tables['fcr'].columns:
        prices = tables['fcr'][country]
        print(f"\n{country}:")
        print(f"  Mean Price:      {prices.mean():8.2f} EUR/MW")
        print(f"  Median Price:    {prices.median():8.2f} EUR/MW")
        print(f"  Price Range:     [{prices.min():6.2f}, {prices.max():6.2f}] EUR/MW")

print("\n" + "="*80)
print("aFRR ENERGY MARKET SUMMARY (NEW)")
print("="*80)

for country in ['DE', 'AT', 'CH', 'HU', 'CZ']:
    print(f"\n{country}:")
    for direction in ['Pos', 'Neg']:
        col = f'{country}_{direction}'
        if col in tables['afrr_energy'].columns:
            prices = tables['afrr_energy'][col]
            non_zero = prices[prices != 0]
            zero_pct = (prices == 0).sum() / len(prices) * 100
            print(f"  {direction}:")
            print(f"    Zero Activation: {zero_pct:6.1f}%")
            if len(non_zero) > 0:
                print(f"    Mean (non-zero): {non_zero.mean():8.2f} EUR/MWh")
                print(f"    Max Price:       {non_zero.max():8.2f} EUR/MWh")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Find country with highest/lowest mean day-ahead price
mean_prices_dict = {}
for country in ['DE', 'AT', 'CH', 'HU', 'CZ']:
    col = 'DE_LU' if country == 'DE' else country
    mean_prices_dict[country] = tables['day_ahead'][col].mean()

highest_price_country = max(mean_prices_dict, key=mean_prices_dict.get)
lowest_price_country = min(mean_prices_dict, key=mean_prices_dict.get)

print(f"\n✓ Highest mean DA price: {highest_price_country} ({mean_prices_dict[highest_price_country]:.2f} EUR/MWh)")
print(f"✓ Lowest mean DA price:  {lowest_price_country} ({mean_prices_dict[lowest_price_country]:.2f} EUR/MWh)")

# Price correlation
corr_matrix = df_da[['DE', 'AT', 'CH', 'HU', 'CZ']].corr()
min_corr = corr_matrix.min().min()
max_corr = corr_matrix[corr_matrix < 1].max().max()

print(f"\n✓ Price correlation range: [{min_corr:.2f}, {max_corr:.2f}]")
print(f"  → Strong price coupling observed across European markets")

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)

# %% [markdown]
# ## 10. Export Analysis Results

# %%
# Optional: Export key figures to HTML for sharing
output_dir = Path('validation_results/market_data_analysis')
output_dir.mkdir(exist_ok=True)

print("Exporting figures to HTML...\n")

# Example: Export Germany heatmap
fig = plot_da_price_heatmap_mckinsey(tables['day_ahead'], 'DE')
fig.write_html(output_dir / 'de_price_heatmap.html')
print("✓ Exported: de_price_heatmap.html")

# Export multi-market comparison
fig = plot_price_time_series_mckinsey(
    tables, country='DE', time_range='Q1',
    markets=['day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy']
)
fig.write_html(output_dir / 'de_all_markets_q1.html')
print("✓ Exported: de_all_markets_q1.html")

print(f"\n📁 All exports saved to: {output_dir.absolute()}")

# %%
# Develop a new feature here, add ML model to predict activation rate

import numpy as np


# %% [markdown]
# ---
# 
# ## Analysis Complete
# 
# This notebook provides a comprehensive overview of the Phase 2 electricity market data using professional McKinsey-style visualizations. 
# 
# **Key Findings:**
# 1. Day-ahead prices show strong seasonal and hourly patterns
# 2. High price correlation across neighboring European markets
# 3. aFRR energy prices have significant zero-activation periods
# 4. FCR capacity prices show lower volatility than energy markets
# 
# **Next Steps:**
# - Use these insights for battery optimization strategy
# - Identify arbitrage opportunities in multi-market participation
# - Analyze seasonal patterns for long-term investment decisions


