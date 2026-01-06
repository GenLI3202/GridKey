"""
Alpha Meta-Optimization Results Analysis Script

Analyzes existing alpha meta-optimization results without running new simulations.
Loads performance summaries from each alpha directory and generates comparison plots.

Author: Generated with Claude Code
Date: 2024-11-19
"""

# %%
# ================================================================================
# [SECTION 1] SETUP & IMPORTS
# ================================================================================

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import McKinsey styling
from py_script.visualization.config import (
    MCKINSEY_COLORS,
    COUNTRY_COLORS,
    apply_mckinsey_style,
    MCKINSEY_FONTS
)
import plotly.io as pio

# Set McKinsey template as default
pio.templates.default = "mckinsey"

print("=" * 80)
print("[SECTION 1] SETUP COMPLETE")
print("=" * 80)

# %%
# ================================================================================
# [SECTION 2] CONFIGURATION
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 2] CONFIGURATION")
print("=" * 80)

# *** CHANGE THIS to your results directory ***
RESULTS_DIR = project_root / "validation_results/alpha_meta_seq_CZ_0.5C_365d_20251117_051129"

# Financial parameters (for NPV calculation)
WACC = 0.08  # Weighted Average Cost of Capital
INFLATION_RATE = 0.02
PROJECT_LIFETIME_YEARS = 10

print(f"\n[CONFIG] Results directory: {RESULTS_DIR}")
print(f"[CONFIG] Directory exists: {RESULTS_DIR.exists()}")

# %%
# ================================================================================
# [SECTION 3] LOAD EXISTING RESULTS
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 3] LOAD EXISTING RESULTS")
print("=" * 80)

def load_existing_results(results_dir):
    """Load all existing alpha results from directory structure."""

    results_dir = Path(results_dir)

    # Load sweep configuration
    config_path = results_dir / "sweep_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            sweep_config = json.load(f)
        print(f"\n[LOADED] Sweep configuration:")
        print(f"  - Country: {sweep_config.get('country')}")
        print(f"  - C-rate: {sweep_config.get('c_rate')}")
        print(f"  - Test days: {sweep_config.get('test_days')}")
        print(f"  - Alpha values (planned): {sweep_config.get('alpha_values')}")
    else:
        print(f"\n[WARNING] No sweep_config.json found")
        sweep_config = {}

    # Find all alpha_* directories
    alpha_dirs = sorted(results_dir.glob("alpha_*"))
    print(f"\n[FOUND] {len(alpha_dirs)} alpha directories:")

    results_list = []

    for alpha_dir in alpha_dirs:
        # Extract alpha value from directory name
        try:
            alpha_value = float(alpha_dir.name.replace("alpha_", ""))
        except ValueError:
            print(f"  [SKIP] Invalid directory name: {alpha_dir.name}")
            continue

        # Load performance summary
        perf_path = alpha_dir / "performance_summary.json"
        if not perf_path.exists():
            print(f"  [SKIP] Alpha {alpha_value:.1f} - No performance_summary.json")
            continue

        with open(perf_path, 'r') as f:
            perf = json.load(f)

        # Load timeseries for SOC statistics (if exists)
        timeseries_path = alpha_dir / "solution_timeseries.csv"
        soc_stats = {}
        if timeseries_path.exists():
            try:
                df = pd.read_csv(timeseries_path)
                if 'e_soc' in df.columns:
                    soc_values = df['e_soc'].values
                    soc_stats = {
                        'soc_avg_kwh': float(np.mean(soc_values)),
                        'soc_min_kwh': float(np.min(soc_values)),
                        'soc_max_kwh': float(np.max(soc_values)),
                        'soc_std_kwh': float(np.std(soc_values))
                    }
            except Exception as e:
                print(f"  [WARNING] Alpha {alpha_value:.1f} - Could not load SOC stats: {e}")

        # Combine results
        result = {
            'alpha': alpha_value,
            'status': 'success',
            'total_profit_eur': perf.get('total_profit_eur', 0),
            'total_revenue_eur': perf.get('total_revenue_eur', 0),
            'total_cost_eur': perf.get('total_cost_eur', 0),
            'revenue_da_eur': perf.get('revenue_da_eur', 0),
            'revenue_afrr_energy_eur': perf.get('revenue_afrr_energy_eur', 0),
            'revenue_as_eur': perf.get('revenue_as_eur', 0),
            'degradation_cyclic_eur': perf.get('degradation_cyclic_eur', 0),
            'degradation_calendar_eur': perf.get('degradation_calendar_eur', 0),
            'total_aging_cost_eur': perf.get('total_cost_eur', 0),
            'num_iterations': perf.get('num_iterations', 0),
            'country': sweep_config.get('country', 'CZ'),
            'c_rate': perf.get('c_rate', sweep_config.get('c_rate', 0.5)),
            'test_days': sweep_config.get('test_days', 365),
            'output_dir': str(alpha_dir),
            **soc_stats
        }

        results_list.append(result)
        print(f"  [OK] Alpha {alpha_value:.1f} - Profit: €{result['total_profit_eur']:,.0f}")

    print(f"\n[SUMMARY] Loaded {len(results_list)} alpha results successfully")

    return results_list, sweep_config


# Load the results
results_list, sweep_config = load_existing_results(RESULTS_DIR)

if not results_list:
    print("\n[ERROR] No results found! Check your RESULTS_DIR path.")
    sys.exit(1)

# Create DataFrame
df = pd.DataFrame(results_list)
df = df.sort_values('alpha').reset_index(drop=True)

print("\n[LOADED] Results DataFrame:")
print(df[['alpha', 'total_profit_eur', 'total_aging_cost_eur', 'status']].to_string(index=False))

# %%
# ================================================================================
# [SECTION 4] CALCULATE DERIVED METRICS
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 4] CALCULATE DERIVED METRICS")
print("=" * 80)

# Test days from config or first result
TEST_DAYS = sweep_config.get('test_days', df['test_days'].iloc[0])
COUNTRY = sweep_config.get('country', df['country'].iloc[0])
C_RATE = sweep_config.get('c_rate', df['c_rate'].iloc[0])

# Calculate derived metrics
df['net_profit_eur'] = df['total_profit_eur']
df['profit_per_day'] = df['net_profit_eur'] / TEST_DAYS

# Annual estimates (already full year)
df['annual_profit_estimate'] = df['net_profit_eur']
df['annual_aging_cost_estimate'] = df['total_aging_cost_eur']

# Calculate NPV
discount_rates = [(1 / (1 + WACC) ** year) for year in range(1, PROJECT_LIFETIME_YEARS + 1)]
npv_multiplier = sum(discount_rates)
df['npv_eur'] = df['annual_profit_estimate'] * npv_multiplier

# ROI proxy
df['roi_proxy'] = df['net_profit_eur'] / df['total_aging_cost_eur'].replace(0, 1)

print(f"\n[CALCULATED] Derived metrics:")
print(f"  - Test days: {TEST_DAYS}")
print(f"  - NPV multiplier (WACC={WACC}, {PROJECT_LIFETIME_YEARS}y): {npv_multiplier:.2f}")
print(f"  - Profit per day range: €{df['profit_per_day'].min():.0f} - €{df['profit_per_day'].max():.0f}")

# Save enhanced results
output_csv = RESULTS_DIR / "comparison_results.csv"
df.to_csv(output_csv, index=False)
print(f"\n[SAVED] Enhanced results to: {output_csv}")

# %%
# ================================================================================
# [SECTION 5] SUMMARY STATISTICS
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 5] SUMMARY STATISTICS")
print("=" * 80)

# Find best configurations
best_profit_idx = df['total_profit_eur'].idxmax()
best_npv_idx = df['npv_eur'].idxmax()
best_roi_idx = df['roi_proxy'].idxmax()

print(f"\n[BEST CONFIGURATIONS]")
print(f"\n1. Best Profit:")
print(f"   Alpha: {df.loc[best_profit_idx, 'alpha']:.1f}")
print(f"   Profit: €{df.loc[best_profit_idx, 'total_profit_eur']:,.0f}")
print(f"   NPV: €{df.loc[best_profit_idx, 'npv_eur']:,.0f}")

print(f"\n2. Best NPV:")
print(f"   Alpha: {df.loc[best_npv_idx, 'alpha']:.1f}")
print(f"   Profit: €{df.loc[best_npv_idx, 'total_profit_eur']:,.0f}")
print(f"   NPV: €{df.loc[best_npv_idx, 'npv_eur']:,.0f}")

print(f"\n3. Best ROI Proxy:")
print(f"   Alpha: {df.loc[best_roi_idx, 'alpha']:.1f}")
print(f"   ROI: {df.loc[best_roi_idx, 'roi_proxy']:.2f}")
print(f"   Profit: €{df.loc[best_roi_idx, 'total_profit_eur']:,.0f}")

# Revenue breakdown for best profit
print(f"\n[REVENUE BREAKDOWN - Best Profit Alpha {df.loc[best_profit_idx, 'alpha']:.1f}]")
print(f"  Day-ahead: €{df.loc[best_profit_idx, 'revenue_da_eur']:,.0f}")
print(f"  aFRR Energy: €{df.loc[best_profit_idx, 'revenue_afrr_energy_eur']:,.0f}")
print(f"  Ancillary Services: €{df.loc[best_profit_idx, 'revenue_as_eur']:,.0f}")
print(f"  TOTAL Revenue: €{df.loc[best_profit_idx, 'total_revenue_eur']:,.0f}")
print(f"\n  Cyclic Aging: €{df.loc[best_profit_idx, 'degradation_cyclic_eur']:,.0f}")
print(f"  Calendar Aging: €{df.loc[best_profit_idx, 'degradation_calendar_eur']:,.0f}")
print(f"  TOTAL Cost: €{df.loc[best_profit_idx, 'total_aging_cost_eur']:,.0f}")

# %%
# ================================================================================
# [SECTION 6] PLOTTING FUNCTIONS
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 6] GENERATE PLOTS")
print("=" * 80)

plots_dir = RESULTS_DIR / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Plot 1: Pareto Front
# -------------------------
print("\n[1/5] Generating Pareto Front...")

fig = go.Figure()

# Main scatter - use Teal-Navy gradient for alpha values
fig.add_trace(go.Scatter(
    x=df['annual_aging_cost_estimate'],
    y=df['annual_profit_estimate'],
    mode='markers+lines',
    marker=dict(
        size=12,
        color=df['alpha'],
        colorscale=[[0, MCKINSEY_COLORS['teal']], [1, MCKINSEY_COLORS['navy']]], # Teal to Navy
        showscale=True,
        colorbar=dict(title="Alpha", tickfont=dict(size=MCKINSEY_FONTS['tick_label_size'])),
        line=dict(width=0.5, color='white')  # White outline for clarity
    ),
    line=dict(color=MCKINSEY_COLORS['gray_medium'], width=1.5, dash='dot'),
    text=[f"alpha={a:.1f}" for a in df['alpha']],
    hovertemplate='<b>Alpha: %{text}</b><br>' +
                  'Aging Cost: €%{x:,.0f}/yr<br>' +
                  'Profit: €%{y:,.0f}/yr<br>' +
                  '<extra></extra>',
    name='Alpha sweep'
))

# Highlight best profit - using categorical color
best_profit = df.loc[best_profit_idx]
fig.add_trace(go.Scatter(
    x=[best_profit['annual_aging_cost_estimate']],
    y=[best_profit['annual_profit_estimate']],
    mode='markers',
    marker=dict(size=20, color=MCKINSEY_COLORS['cat_4'], symbol='star',
                line=dict(width=2, color='white')),
    name=f'Best Profit (α={best_profit["alpha"]:.1f})',
    hovertemplate=f'<b>Best Profit</b><br>Alpha: {best_profit["alpha"]:.1f}<br>' +
                 f'Profit: €{best_profit["annual_profit_estimate"]:,.0f}/yr<br>' +
                 '<extra></extra>'
))

# Highlight best NPV - using teal
best_npv = df.loc[best_npv_idx]
fig.add_trace(go.Scatter(
    x=[best_npv['annual_aging_cost_estimate']],
    y=[best_npv['annual_profit_estimate']],
    mode='markers',
    marker=dict(size=20, color=MCKINSEY_COLORS['teal'], symbol='diamond',
                line=dict(width=2, color='white')),
    name=f'Best NPV (α={best_npv["alpha"]:.1f})',
    hovertemplate=f'<b>Best NPV</b><br>Alpha: {best_npv["alpha"]:.1f}<br>' +
                 f'NPV: €{best_npv["npv_eur"]:,.0f}<br>' +
                 '<extra></extra>'
))

fig = apply_mckinsey_style(fig, title=f"Pareto Front: Aging Cost vs Profit ({COUNTRY}, C-rate={C_RATE})")
fig.update_layout(
    xaxis_title="Annual Aging Cost (EUR)",
    yaxis_title="Annual Profit (EUR)",
    hovermode='closest',
    width=1000,
    height=700
)

output_path = plots_dir / "pareto_front.html"
fig.write_html(str(output_path))
print(f"  [SAVED] {output_path}")

# -------------------------
# Plot 2: SOC vs Alpha
# -------------------------
if 'soc_avg_kwh' in df.columns and df['soc_avg_kwh'].notna().any():
    print("\n[2/5] Generating SOC vs Alpha...")

    fig = go.Figure()

    # Min/Max range (plot first so it's in background) - darker shading
    if 'soc_max_kwh' in df.columns and 'soc_min_kwh' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['alpha'],
            y=df['soc_max_kwh'],
            mode='lines',
            name='Max SOC',
            line=dict(color=MCKINSEY_COLORS['gray_medium'], width=1.5, dash='dash'),
            showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=df['alpha'],
            y=df['soc_min_kwh'],
            mode='lines',
            name='Min SOC',
            line=dict(color=MCKINSEY_COLORS['gray_medium'], width=1.5, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.3)',  # Darker gray fill (30% opacity)
            showlegend=True
        ))

    # Average SOC - main line (McKinsey navy)
    fig.add_trace(go.Scatter(
        x=df['alpha'],
        y=df['soc_avg_kwh'],
        mode='lines+markers',
        name='Average SOC',
        line=dict(color=MCKINSEY_COLORS['navy'], width=3),
        marker=dict(size=8, color=MCKINSEY_COLORS['navy'],
                    line=dict(width=1, color='white'))
    ))

    fig = apply_mckinsey_style(fig, title=f"SOC Statistics vs Alpha ({COUNTRY}, C-rate={C_RATE})")
    fig.update_layout(
        xaxis_title="Alpha (Degradation Weight)",
        yaxis_title="State of Charge (kWh)",
        hovermode='x unified',
        width=1000,
        height=600,
        # Move legend to top right inside plot area
        legend=dict(
            x=0.98,  # Right side (0.98 = 98% from left)
            y=0.98,  # Top (0.98 = 98% from bottom)
            xanchor='right',
            yanchor='top',
            font=dict(size=12),  # Larger legend font
            bgcolor='rgba(255,255,255,0.9)',  # Semi-transparent white background
            bordercolor=MCKINSEY_COLORS['gray_medium'],
            borderwidth=1
        )
    )

    output_path = plots_dir / "soc_vs_alpha.html"
    fig.write_html(str(output_path))
    print(f"  [SAVED] {output_path}")
else:
    print("\n[2/5] SKIPPED - No SOC data available")

# -------------------------
# Plot 3: Profit vs Alpha
# -------------------------
print("\n[3/5] Generating Profit vs Alpha...")

fig = go.Figure()

# Main profit line (McKinsey teal for positive financial metric)
fig.add_trace(go.Scatter(
    x=df['alpha'],
    y=df['total_profit_eur'],
    mode='lines+markers',
    name='Net Profit',
    line=dict(color=MCKINSEY_COLORS['teal'], width=3),
    marker=dict(size=8, color=MCKINSEY_COLORS['teal'],
                line=dict(width=1, color='white'))
))

# Mark best profit (categorical purple for emphasis)
fig.add_trace(go.Scatter(
    x=[best_profit['alpha']],
    y=[best_profit['total_profit_eur']],
    mode='markers',
    marker=dict(size=20, color=MCKINSEY_COLORS['cat_4'], symbol='star',
                line=dict(width=2, color='white')),
    name='Best Profit',
    showlegend=True
))

fig = apply_mckinsey_style(fig, title=f"Net Profit vs Alpha ({COUNTRY}, C-rate={C_RATE})")
fig.update_layout(
    xaxis_title="Alpha (Degradation Weight)",
    yaxis_title="Annual Net Profit (EUR)",
    hovermode='x unified',
    width=1000,
    height=600
)

output_path = plots_dir / "profit_vs_alpha.html"
fig.write_html(str(output_path))
print(f"  [SAVED] {output_path}")

# -------------------------
# Plot 4: Revenue Breakdown
# -------------------------
print("\n[4/5] Generating Revenue Breakdown...")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Revenue Components', 'Cost Components'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
)

# Revenue components (using McKinsey categorical colors)
fig.add_trace(go.Bar(
    x=df['alpha'],
    y=df['revenue_da_eur'],
    name='Day-ahead',
    marker_color=MCKINSEY_COLORS['cat_1']  # Navy
), row=1, col=1)

fig.add_trace(go.Bar(
    x=df['alpha'],
    y=df['revenue_afrr_energy_eur'],
    name='aFRR Energy',
    marker_color=MCKINSEY_COLORS['cat_2']  # Dark blue
), row=1, col=1)

fig.add_trace(go.Bar(
    x=df['alpha'],
    y=df['revenue_as_eur'],
    name='Ancillary Services',
    marker_color=MCKINSEY_COLORS['cat_3']  # Teal
), row=1, col=1)

# Cost components (using negative/darker colors)
fig.add_trace(go.Bar(
    x=df['alpha'],
    y=df['degradation_cyclic_eur'],
    name='Cyclic Aging',
    marker_color=MCKINSEY_COLORS['cat_5'],  # Coral
    showlegend=True
), row=1, col=2)

fig.add_trace(go.Bar(
    x=df['alpha'],
    y=df['degradation_calendar_eur'],
    name='Calendar Aging',
    marker_color=MCKINSEY_COLORS['cat_4'],  # Purple
    showlegend=True
), row=1, col=2)

fig = apply_mckinsey_style(fig, title=f"Revenue & Cost Breakdown vs Alpha ({COUNTRY}, C-rate={C_RATE})")
fig.update_layout(
    barmode='stack',
    width=1400,
    height=600
)

fig.update_xaxes(title_text="Alpha", row=1, col=1)
fig.update_xaxes(title_text="Alpha", row=1, col=2)
fig.update_yaxes(title_text="Revenue (EUR)", row=1, col=1)
fig.update_yaxes(title_text="Cost (EUR)", row=1, col=2)

output_path = plots_dir / "revenue_cost_breakdown.html"
fig.write_html(str(output_path))
print(f"  [SAVED] {output_path}")

# -------------------------
# Plot 5: NPV vs Alpha
# -------------------------
print("\n[5/5] Generating NPV vs Alpha...")

fig = go.Figure()

# Main NPV line (McKinsey dark blue for long-term value)
fig.add_trace(go.Scatter(
    x=df['alpha'],
    y=df['npv_eur'],
    mode='lines+markers',
    name='NPV',
    line=dict(color=MCKINSEY_COLORS['dark_blue'], width=3),
    marker=dict(size=8, color=MCKINSEY_COLORS['dark_blue'],
                line=dict(width=1, color='white'))
))

# Mark best NPV (teal for emphasis)
fig.add_trace(go.Scatter(
    x=[best_npv['alpha']],
    y=[best_npv['npv_eur']],
    mode='markers',
    marker=dict(size=20, color=MCKINSEY_COLORS['teal'], symbol='diamond',
                line=dict(width=2, color='white')),
    name='Best NPV',
    showlegend=True
))

fig = apply_mckinsey_style(fig, title=f"Net Present Value vs Alpha ({COUNTRY}, C-rate={C_RATE}, WACC={WACC}, {PROJECT_LIFETIME_YEARS}y)")
fig.update_layout(
    xaxis_title="Alpha (Degradation Weight)",
    yaxis_title=f"NPV (EUR, {PROJECT_LIFETIME_YEARS} years)",
    hovermode='x unified',
    width=1000,
    height=600
)

output_path = plots_dir / "npv_vs_alpha.html"
fig.write_html(str(output_path))
print(f"  [SAVED] {output_path}")

# %%
# ================================================================================
# [SECTION 7] SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 7] ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n[SUMMARY] Alpha Meta-Optimization Analysis")
print(f"  Results directory: {RESULTS_DIR}")
print(f"  Country: {COUNTRY} | C-rate: {C_RATE} | Test days: {TEST_DAYS}")
print(f"  Alpha values analyzed: {len(df)}")
print(f"  Alpha range: {df['alpha'].min():.1f} - {df['alpha'].max():.1f}")

print(f"\n[KEY FINDINGS]")
print(f"  Best Profit: alpha={df.loc[best_profit_idx, 'alpha']:.1f} -> EUR {df.loc[best_profit_idx, 'total_profit_eur']:,.0f}/yr")
print(f"  Best NPV: alpha={df.loc[best_npv_idx, 'alpha']:.1f} -> EUR {df.loc[best_npv_idx, 'npv_eur']:,.0f} ({PROJECT_LIFETIME_YEARS}y)")
print(f"  Best ROI: alpha={df.loc[best_roi_idx, 'alpha']:.1f} -> {df.loc[best_roi_idx, 'roi_proxy']:.2f}")

print(f"\n[OUTPUTS GENERATED]")
print(f"  - {output_csv.relative_to(project_root)}")
print(f"  - {plots_dir.relative_to(project_root)}/pareto_front.html")
print(f"  - {plots_dir.relative_to(project_root)}/soc_vs_alpha.html (if SOC data available)")
print(f"  - {plots_dir.relative_to(project_root)}/profit_vs_alpha.html")
print(f"  - {plots_dir.relative_to(project_root)}/revenue_cost_breakdown.html")
print(f"  - {plots_dir.relative_to(project_root)}/npv_vs_alpha.html")

print("\n" + "=" * 80)
print("[DONE] Analysis complete!")
print("=" * 80)

# %%
