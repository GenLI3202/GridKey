"""
Example: How to use segment LIFO visualization functions

These functions are now part of py_script.visualization.optimization_analysis
"""

import pandas as pd
from py_script.visualization.optimization_analysis import (
    plot_segment_lifo_analysis,
    detect_parallel_segment_operations
)

# ============================================================================
# Example 1: Load and visualize existing results
# ============================================================================

# Path to solution file
solution_csv = "validation_results/optimizer_validation/20251114_223822_notebook_test_modeliii_ch_24h_alpha1.0/solution_timeseries.csv"

# Load data
df = pd.read_csv(solution_csv)

# Detect violations
parallel_ch, parallel_dis = detect_parallel_segment_operations(df, epsilon=0.1)

print(f"\n{'='*80}")
print("LIFO VIOLATION CHECK")
print(f"{'='*80}")
print(f"Parallel charging violations:     {len(parallel_ch)}")
print(f"Parallel discharging violations:  {len(parallel_dis)}")

if len(parallel_ch) > 0:
    print(f"\nFirst 3 violations:")
    for i, v in enumerate(parallel_ch[:3], 1):
        print(f"  {i}. Hour {v['hour']:.2f}: Segments {v['segments']}")

# Create visualization
fig = plot_segment_lifo_analysis(
    solution_df=df,
    show_violations=True,     # Highlight violations
    plot_style='stacked',     # or 'individual'
    epsilon=0.1,              # Tolerance
    width=1400,
    height=1200,
    title_suffix='CH 24h Test'
)

# Save
output_html = solution_csv.replace('.csv', '_segment_lifo.html')
fig.write_html(output_html)
print(f"\nVisualization saved to: {output_html}")

# ============================================================================
# Example 2: Use in notebook after optimization
# ============================================================================

# After running optimizer.optimize():
# solution = optimizer.extract_solution(model, results)
# solution_df = pd.DataFrame(solution)  # or use extract_detailed_solution()

# fig = plot_segment_lifo_analysis(solution_df, title_suffix=run_name)
# fig.write_html(f"{validation_dir}/segment_lifo.html")

print(f"\n{'='*80}\n")
