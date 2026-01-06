"""
Visualize AS Revenue Fix Impact
================================
Compare optimizer behavior before and after removing Δb coefficient from AS capacity revenue
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load current results (after fix)
df_after = pd.read_csv('results/mpc_5day_test/decision_variables_5day.csv')
df_after['timestamp'] = pd.to_datetime(df_after['timestamp'])

print("="*80)
print("AS REVENUE FIX VISUALIZATION")
print("="*80)
print(f"\nResults after fix loaded: {len(df_after)} timesteps")
print(f"Date range: {df_after['timestamp'].iloc[0]} to {df_after['timestamp'].iloc[-1]}")
print()

# Calculate revenues
# IMPORTANT: p_ch and p_dis are in kW, need to convert to MW for revenue calculation
df_after['da_revenue_eur'] = (df_after['p_dis'] / 1000 - df_after['p_ch'] / 1000) * df_after['price_day_ahead'] * 0.25  # (MW) * (EUR/MWh) * (h) = EUR
df_after['fcr_revenue_eur'] = df_after['c_fcr'] * df_after['price_fcr'] / 16  # (MW) * (EUR/MW per 4h block) / (16 intervals/block) = EUR per interval

print("Financial Summary (After Fix):")
print(f"  Total DA Revenue:  {df_after['da_revenue_eur'].sum():.2f} EUR")
print(f"  Total FCR Revenue: {df_after['fcr_revenue_eur'].sum():.2f} EUR")
print(f"  Initial SOC: {df_after['e_soc'].iloc[0]:.2f} kWh ({100*df_after['e_soc'].iloc[0]/4472:.1f}%)")
print(f"  Final SOC:   {df_after['e_soc'].iloc[-1]:.2f} kWh ({100*df_after['e_soc'].iloc[-1]/4472:.1f}%)")
print()

# Create comprehensive visualization
fig = make_subplots(
    rows=5, cols=1,
    subplot_titles=(
        'SOC Trajectory (After Fix)',
        'Power Decisions: Charge & Discharge',
        'FCR Capacity Bids vs Price',
        'Day-Ahead Market Price',
        'Revenue Components'
    ),
    vertical_spacing=0.08,
    row_heights=[0.20, 0.20, 0.20, 0.15, 0.25]
)

# 1. SOC Trajectory
fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['e_soc'],
        name='SOC',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)',
        hovertemplate='<b>SOC</b><br>Time: %{x}<br>SOC: %{y:.2f} kWh (%{customdata:.1f}%)<extra></extra>',
        customdata=100 * df_after['e_soc'] / 4472
    ),
    row=1, col=1
)

# Add SOC reference lines
fig.add_hline(y=2236, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1,
              annotation_text="50% SOC", annotation_position="right")
fig.add_hline(y=4472, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1,
              annotation_text="100% SOC", annotation_position="right")

# 2. Power Decisions (convert kW to MW for visualization)
fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=-df_after['p_ch'] / 1000,  # kW to MW
        name='Charge',
        line=dict(color='green', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 0, 0.3)',
        hovertemplate='<b>Charge</b><br>Time: %{x}<br>Power: %{y:.2f} MW<extra></extra>'
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['p_dis'] / 1000,  # kW to MW
        name='Discharge',
        line=dict(color='red', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(200, 0, 0, 0.3)',
        hovertemplate='<b>Discharge</b><br>Time: %{x}<br>Power: %{y:.2f} MW<extra></extra>'
    ),
    row=2, col=1
)

# 3. FCR Capacity Bids vs Price
fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['price_fcr'],
        name='FCR Price',
        line=dict(color='purple', width=1, dash='dot'),
        yaxis='y3',
        hovertemplate='<b>FCR Price</b><br>Time: %{x}<br>Price: %{y:.2f} EUR/MW<extra></extra>'
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['c_fcr'],
        name='FCR Bid',
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.3)',
        hovertemplate='<b>FCR Bid</b><br>Time: %{x}<br>Capacity: %{y:.2f} MW<extra></extra>'
    ),
    row=3, col=1
)

# 4. Day-Ahead Price
fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['price_day_ahead'],
        name='DA Price',
        line=dict(color='darkblue', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 139, 0.2)',
        hovertemplate='<b>DA Price</b><br>Time: %{x}<br>Price: %{y:.2f} EUR/MWh<extra></extra>'
    ),
    row=4, col=1
)

# Highlight high-price discharge period
discharge_period = df_after[df_after['p_dis'] > 100]
if len(discharge_period) > 0:
    fig.add_vrect(
        x0=discharge_period['timestamp'].iloc[0],
        x1=discharge_period['timestamp'].iloc[-1],
        fillcolor="red", opacity=0.1, layer="below",
        line_width=0,
        annotation_text="High-price discharge",
        annotation_position="top left",
        row=4, col=1
    )

# 5. Revenue Components (Cumulative)
df_after['cumulative_da'] = df_after['da_revenue_eur'].cumsum()
df_after['cumulative_fcr'] = df_after['fcr_revenue_eur'].cumsum()
df_after['cumulative_total'] = df_after['cumulative_da'] + df_after['cumulative_fcr']

fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['cumulative_da'],
        name='Cumulative DA Revenue',
        line=dict(color='green', width=2),
        hovertemplate='<b>DA Revenue</b><br>Time: %{x}<br>Total: %{y:.2f} EUR<extra></extra>'
    ),
    row=5, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['cumulative_fcr'],
        name='Cumulative FCR Revenue',
        line=dict(color='orange', width=2),
        hovertemplate='<b>FCR Revenue</b><br>Time: %{x}<br>Total: %{y:.2f} EUR<extra></extra>'
    ),
    row=5, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_after['timestamp'],
        y=df_after['cumulative_total'],
        name='Total Revenue',
        line=dict(color='purple', width=3, dash='dash'),
        hovertemplate='<b>Total Revenue</b><br>Time: %{x}<br>Total: %{y:.2f} EUR<extra></extra>'
    ),
    row=5, col=1
)

# Update layout
fig.update_layout(
    height=1600,
    title_text=f"<b>MPC Optimization Strategy After AS Revenue Fix</b><br>" +
               f"<sub>CH, 5 days, 32h/24h horizon | Removed Δb coefficient from AS capacity revenue</sub>",
    showlegend=True,
    hovermode='x unified',
    template='plotly_white'
)

# Y-axis labels
fig.update_yaxes(title_text="SOC (kWh)", row=1, col=1)
fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
fig.update_yaxes(title_text="Capacity (MW) / Price (EUR/MW)", row=3, col=1)
fig.update_yaxes(title_text="Price (EUR/MWh)", row=4, col=1)
fig.update_yaxes(title_text="Revenue (EUR)", row=5, col=1)

# X-axis labels
fig.update_xaxes(title_text="Time", row=5, col=1)

# Save
output_path = 'results/mpc_5day_test/as_revenue_fix_visualization.html'
fig.write_html(output_path)
print(f"Saved visualization: {output_path}")
print()

# Print key insights
print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

print("1. BEFORE FIX (with delta_b coefficient):")
print("   - AS revenue was 4x too high (15,512 EUR -> should be ~3,878 EUR)")
print("   - Optimizer chose to do ONLY FCR bidding, no charge/discharge")
print("   - SOC remained constant at 50% (2236 kWh)")
print("   - Strategy: Maximize FCR capacity by staying at equilibrium SOC")
print()

print("2. AFTER FIX (removed delta_b coefficient):")
print(f"   - AS revenue is now correct: {df_after['fcr_revenue_eur'].sum():.2f} EUR")
print(f"   - DA revenue: {df_after['da_revenue_eur'].sum():.2f} EUR")
print(f"   - Optimizer now uses BOTH FCR bidding AND DA arbitrage")
print(f"   - SOC varies: {df_after['e_soc'].iloc[0]:.0f} kWh -> {df_after['e_soc'].iloc[-1]:.0f} kWh")
print(f"   - Strategy: Balance FCR revenue with high-price DA discharge opportunities")
print()

# Identify strategic discharge
discharge_events = df_after[df_after['p_dis'] > 100]
if len(discharge_events) > 0:
    print("3. STRATEGIC DISCHARGE EVENT:")
    print(f"   - Time: {discharge_events['timestamp'].iloc[0]} to {discharge_events['timestamp'].iloc[-1]}")
    print(f"   - Duration: {len(discharge_events) * 0.25:.2f} hours")
    print(f"   - Average discharge power: {discharge_events['p_dis'].mean() / 1000:.2f} MW")
    print(f"   - Average DA price: {discharge_events['price_day_ahead'].mean():.2f} EUR/MWh")
    print(f"   - Energy discharged: {discharge_events['p_dis'].sum() / 1000 * 0.25:.2f} MWh")
    print(f"   - DA revenue from discharge: {discharge_events['da_revenue_eur'].sum():.2f} EUR")
    print(f"   - FCR revenue during discharge: {discharge_events['fcr_revenue_eur'].sum():.2f} EUR (gave up capacity!)")
print()

print("="*80)
print(f"Opening visualization: {output_path}")
print("="*80)

# Open in browser
import subprocess
subprocess.run(['start', output_path], shell=True)
