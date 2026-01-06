# -*- coding: utf-8 -*-
"""
Phase 2 Model I/II/III Comparison - Interactive Analysis
=========================================================

This interactive script generates comprehensive comparison visualizations
analyzing the impact of degradation modeling across Models I, II, and III.

Quick start:
1. Update MODEL_DIRS configuration below with your result directories
2. Run cells one by one using Shift+Enter
3. Inspect each plot in the Interactive Window
4. All plots will be saved to the output directory

Features:
- 8 professional McKinsey-styled comparison plots
- Financial analysis (waterfall, revenue breakdown, heatmap)
- Operational comparison (SOC, power dispatch, segments)
- Performance metrics (degradation, computational complexity)
- Dual output format (interactive HTML + presentation PNG)
"""

# %%
# ============================================================================
# IMPORTS
# ============================================================================

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from py_script.visualization.model_comparison import (
    load_model_results,
    plot_profit_waterfall_comparison,
    plot_revenue_components_grouped,
    plot_profitability_heatmap,
    plot_soc_trajectory_overlay,
    plot_power_dispatch_comparison,
    plot_segment_distribution,
    plot_degradation_metrics,
    plot_computational_complexity,
    generate_comparison_suite
)

print("[OK] All imports successful!")

# %%
# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# ============================================================================

# Result directory paths for each model
# Update these to match your validation result directories
MODEL_DIRS = {
    'I': str(project_root / 'validation_results' / 'optimizer_validation' / '20251117_185926_interactive_modeli_cz_36h_eps0'),
    'II': str(project_root / 'validation_results' / 'optimizer_validation' / '20251117_191434_interactive_modelii_cz_36h_eps0'),
    'III': str(project_root / 'validation_results' / 'optimizer_validation' / '20251117_195521_interactive_modeliii_cz_36h_eps0')
}

# Output configuration
OUTPUT_DIR = str(project_root / 'validation_results' / 'model_comparison_cz_36h')
TITLE_SUFFIX = "CZ - 36h"
GENERATE_HTML = True
GENERATE_PNG = True

print("=" * 80)
print("[CONFIG] MODEL COMPARISON CONFIGURATION")
print("=" * 80)
print(f"Model I:   {MODEL_DIRS['I']}")
print(f"Model II:  {MODEL_DIRS['II']}")
print(f"Model III: {MODEL_DIRS['III']}")
print(f"")
print(f"Output:    {OUTPUT_DIR}")
print(f"Title:     {TITLE_SUFFIX}")
print(f"Formats:   {'HTML ' if GENERATE_HTML else ''}{'PNG' if GENERATE_PNG else ''}")
print("=" * 80)

# %%
# ============================================================================
# LOAD MODEL RESULTS
# ============================================================================

print("\n[LOAD] Loading model results...")

results = load_model_results(MODEL_DIRS)

print(f"\n[OK] Loaded {len(results)} model results")

# Display summary
print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

for model_name, data in results.items():
    perf = data['performance']
    sol_df = data['timeseries']

    print(f"\n[Model {model_name}]")
    print(f"  Profit:           €{perf.get('total_profit_eur', 0):,.2f}")
    print(f"  Revenue:          €{perf.get('total_revenue_eur', 0):,.2f}")
    print(f"  Solve Time:       {perf.get('solve_time_sec', 0):.2f}s")
    print(f"  Build Time:       {perf.get('build_time_sec', 0):.3f}s")
    print(f"  Variables:        {perf.get('n_variables', 0):,}")
    print(f"  Constraints:      {perf.get('n_constraints', 0):,}")
    print(f"  Solver Status:    {perf.get('solver_status', 'unknown')}")
    print(f"  Timesteps:        {len(sol_df)}")

    # Degradation metrics (if available)
    if 'degradation_metrics' in perf:
        deg = perf['degradation_metrics']
        print(f"  Cyclic Cost:      €{deg.get('total_cyclic_cost_eur', 0):.2f}")
        print(f"  Calendar Cost:    €{deg.get('total_calendar_cost_eur', 0):.2f}")
        print(f"  Full Cycles:      {deg.get('equivalent_full_cycles', 0):.2f}")

print("\n" + "=" * 80)

# %%
# ============================================================================
# PLOT 1: PROFIT WATERFALL COMPARISON
# ============================================================================

print("\n[1/8] Generating Profit Waterfall Comparison...")

fig1 = plot_profit_waterfall_comparison(
    results,
    title_suffix=TITLE_SUFFIX,
    save_path=str(Path(OUTPUT_DIR) / "01_profit_waterfall.html") if GENERATE_HTML else None
)

fig1.show()

print("   [OK] Waterfall chart complete")

# %%
# ============================================================================
# PLOT 2: REVENUE COMPONENTS GROUPED BAR
# ============================================================================

print("\n[2/8] Generating Revenue Components Comparison...")

fig2 = plot_revenue_components_grouped(
    results,
    title_suffix=TITLE_SUFFIX,
    save_path=str(Path(OUTPUT_DIR) / "02_revenue_components.html") if GENERATE_HTML else None
)

fig2.show()

print("   [OK] Revenue components chart complete")

# %%
# ============================================================================
# PLOT 3: PROFITABILITY HEATMAP
# ============================================================================

print("\n[3/8] Generating Profitability Heatmap...")

fig3 = plot_profitability_heatmap(
    results,
    title_suffix=TITLE_SUFFIX,
    save_path=str(Path(OUTPUT_DIR) / "03_profitability_heatmap.html") if GENERATE_HTML else None
)

fig3.show()

print("   [OK] Profitability heatmap complete")

# %%
# ============================================================================
# PLOT 4: SOC TRAJECTORY OVERLAY
# ============================================================================

print("\n[4/8] Generating SOC Trajectory Overlay...")

fig4 = plot_soc_trajectory_overlay(
    results,
    title_suffix=TITLE_SUFFIX,
    save_path=str(Path(OUTPUT_DIR) / "04_soc_trajectory_overlay.html") if GENERATE_HTML else None
)

fig4.show()

print("   [OK] SOC trajectory overlay complete")

# %%
# ============================================================================
# PLOT 5: POWER DISPATCH COMPARISON
# ============================================================================

print("\n[5/8] Generating Power Dispatch Comparison...")

fig5 = plot_power_dispatch_comparison(
    results,
    title_suffix=TITLE_SUFFIX,
    save_path=str(Path(OUTPUT_DIR) / "05_power_dispatch_comparison.html") if GENERATE_HTML else None
)

fig5.show()

print("   [OK] Power dispatch comparison complete")

# %%
# ============================================================================
# PLOT 6: SEGMENT DISTRIBUTION (Model II vs III)
# ============================================================================

print("\n[6/8] Generating Segment Distribution Comparison...")

try:
    fig6 = plot_segment_distribution(
        results,
        title_suffix=TITLE_SUFFIX,
        save_path=str(Path(OUTPUT_DIR) / "06_segment_distribution.html") if GENERATE_HTML else None
    )
    fig6.show()
    print("   [OK] Segment distribution complete")
except ValueError as e:
    print(f"   [SKIP] {e}")

# %%
# ============================================================================
# PLOT 7: DEGRADATION METRICS DASHBOARD
# ============================================================================

print("\n[7/8] Generating Degradation Metrics Dashboard...")

fig7 = plot_degradation_metrics(
    results,
    title_suffix=TITLE_SUFFIX,
    save_path=str(Path(OUTPUT_DIR) / "07_degradation_metrics.html") if GENERATE_HTML else None
)

fig7.show()

print("   [OK] Degradation metrics dashboard complete")

# %%
# ============================================================================
# PLOT 8: COMPUTATIONAL COMPLEXITY DASHBOARD
# ============================================================================

print("\n[8/8] Generating Computational Complexity Dashboard...")

fig8 = plot_computational_complexity(
    results,
    title_suffix=TITLE_SUFFIX,
    save_path=str(Path(OUTPUT_DIR) / "08_computational_complexity.html") if GENERATE_HTML else None
)

fig8.show()

print("   [OK] Computational complexity dashboard complete")

# %%
# ============================================================================
# GENERATE ALL PLOTS AT ONCE (BATCH MODE)
# ============================================================================

# Uncomment and run this cell to generate all 8 plots in one go

# print("\n" + "=" * 80)
# print("GENERATING ALL PLOTS (BATCH MODE)")
# print("=" * 80)
#
# all_figures = generate_comparison_suite(
#     model_dirs=MODEL_DIRS,
#     output_dir=OUTPUT_DIR,
#     title_suffix=TITLE_SUFFIX,
#     generate_html=GENERATE_HTML,
#     generate_png=GENERATE_PNG
# )
#
# print(f"\n[OK] Generated {len(all_figures)} plots")
# print(f"[OK] Output saved to: {OUTPUT_DIR}")

# %%
# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

import pandas as pd

print("\n" + "=" * 80)
print("COMPARISON SUMMARY TABLE")
print("=" * 80)

# Build comparison DataFrame
summary_data = []

for model_name in ['I', 'II', 'III']:
    if model_name in results:
        perf = results[model_name]['performance']
        deg = perf.get('degradation_metrics', {})

        row = {
            'Model': model_name,
            'Profit (EUR)': perf.get('total_profit_eur', 0),
            'Revenue (EUR)': perf.get('total_revenue_eur', 0),
            'Cyclic Cost (EUR)': deg.get('total_cyclic_cost_eur', 0),
            'Calendar Cost (EUR)': deg.get('total_calendar_cost_eur', 0),
            'Solve Time (s)': perf.get('solve_time_sec', 0),
            'Variables': perf.get('n_variables', 0),
            'Constraints': perf.get('n_constraints', 0),
            'Full Cycles': deg.get('equivalent_full_cycles', 0),
            'Status': perf.get('solver_status', 'unknown')
        }
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

# Calculate profit differences
if len(summary_df) > 1:
    baseline_profit = summary_df.loc[summary_df['Model'] == 'I', 'Profit (EUR)'].values[0]
    summary_df['Profit vs Model I (%)'] = ((summary_df['Profit (EUR)'] - baseline_profit) / baseline_profit * 100).round(2)

print(summary_df.to_string(index=False))

# Save summary to CSV
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
summary_csv_path = output_path / "comparison_summary.csv"
summary_df.to_csv(summary_csv_path, index=False)

print(f"\n[OK] Summary saved to: {summary_csv_path}")
print("=" * 80)

# %%
# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

if len(results) == 3:
    profit_i = results['I']['performance'].get('total_profit_eur', 0)
    profit_ii = results['II']['performance'].get('total_profit_eur', 0)
    profit_iii = results['III']['performance'].get('total_profit_eur', 0)

    solve_i = results['I']['performance'].get('solve_time_sec', 0)
    solve_ii = results['II']['performance'].get('solve_time_sec', 0)
    solve_iii = results['III']['performance'].get('solve_time_sec', 0)

    cyclic_ii = results['II']['performance'].get('degradation_metrics', {}).get('total_cyclic_cost_eur', 0)
    cyclic_iii = results['III']['performance'].get('degradation_metrics', {}).get('total_cyclic_cost_eur', 0)
    calendar_iii = results['III']['performance'].get('degradation_metrics', {}).get('total_calendar_cost_eur', 0)

    print(f"\n1. PROFITABILITY IMPACT:")
    print(f"   - Model II vs I: {((profit_ii - profit_i) / profit_i * 100):.1f}% profit reduction")
    print(f"   - Model III vs I: {((profit_iii - profit_i) / profit_i * 100):.1f}% profit reduction")
    print(f"   - Cyclic aging cost: €{cyclic_iii:.2f} ({cyclic_iii / profit_i * 100:.1f}% of Model I profit)")
    print(f"   - Calendar aging cost: €{calendar_iii:.2f} ({calendar_iii / profit_i * 100:.1f}% of Model I profit)")

    print(f"\n2. COMPUTATIONAL COMPLEXITY:")
    print(f"   - Model II vs I: {solve_ii / solve_i:.1f}x longer solve time")
    print(f"   - Model III vs I: {solve_iii / solve_i:.1f}x longer solve time")
    print(f"   - Model III vs II: {solve_iii / solve_ii:.1f}x longer solve time")

    print(f"\n3. MODEL SIZE:")
    vars_i = results['I']['performance'].get('n_variables', 0)
    vars_iii = results['III']['performance'].get('n_variables', 0)
    print(f"   - Variables increased: {vars_i:,} -> {vars_iii:,} ({(vars_iii - vars_i) / vars_i * 100:.1f}%)")

    print(f"\n4. DEGRADATION AWARENESS:")
    cycles_ii = results['II']['performance'].get('degradation_metrics', {}).get('equivalent_full_cycles', 0)
    cycles_iii = results['III']['performance'].get('degradation_metrics', {}).get('equivalent_full_cycles', 0)
    print(f"   - Model II cycling: {cycles_ii:.2f} equivalent full cycles")
    print(f"   - Model III cycling: {cycles_iii:.2f} equivalent full cycles")
    print(f"   - Calendar aging reduces aggressive cycling by {((cycles_ii - cycles_iii) / cycles_ii * 100):.1f}%")

print("\n" + "=" * 80)

# %%
# ============================================================================
# COMPLETE - SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("[COMPLETE] MODEL COMPARISON ANALYSIS COMPLETE!")
print("=" * 80)
print(f"Models analyzed: {len(results)}")
print(f"Plots generated: 8")
print(f"Output directory: {OUTPUT_DIR}")
print(f"")
print("Files generated:")
print("  - 01_profit_waterfall.html (+ .png)")
print("  - 02_revenue_components.html (+ .png)")
print("  - 03_profitability_heatmap.html (+ .png)")
print("  - 04_soc_trajectory_overlay.html (+ .png)")
print("  - 05_power_dispatch_comparison.html (+ .png)")
print("  - 06_segment_distribution.html (+ .png)")
print("  - 07_degradation_metrics.html (+ .png)")
print("  - 08_computational_complexity.html (+ .png)")
print("  - comparison_summary.json")
print("  - comparison_summary.csv")
print("=" * 80)
