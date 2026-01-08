"""
Example usage of segment LIFO plotting functions.

This script demonstrates how to use the plotting functions to visualize
battery segment behavior and detect LIFO violations.
"""

from pathlib import Path
from plot_segment_lifo import (
    plot_segment_lifo_analysis,
    plot_segment_comparison,
    detect_parallel_operations
)
import pandas as pd


def example_single_run():
    """Example: Analyze a single optimization run."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Run Analysis")
    print("="*80)

    # Path to your solution file
    csv_path = Path("validation_results/optimizer_validation") / \
               "20251114_223822_notebook_test_modeliii_ch_24h_alpha1.0" / \
               "solution_timeseries.csv"

    # Create visualization with stacked area chart
    fig = plot_segment_lifo_analysis(
        csv_path=str(csv_path),
        output_path=None,  # Auto-generate name
        show_violations=True,  # Highlight violations
        plot_style='stacked',  # or 'individual'
        epsilon=0.1,  # Tolerance for violation detection
        width=1400,
        height=1200
    )

    print("\nVisualization created! Open the HTML file in your browser.")


def example_detect_violations_only():
    """Example: Detect violations without plotting."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Violation Detection Only")
    print("="*80)

    csv_path = Path("validation_results/optimizer_validation") / \
               "20251114_223822_notebook_test_modeliii_ch_24h_alpha1.0" / \
               "solution_timeseries.csv"

    # Load data
    df = pd.read_csv(csv_path)

    # Detect violations
    parallel_ch, parallel_dis = detect_parallel_operations(df, epsilon=0.1)

    print(f"\nResults:")
    print(f"  Parallel Charging Events:     {len(parallel_ch)}")
    print(f"  Parallel Discharging Events:  {len(parallel_dis)}")

    if parallel_ch:
        print(f"\nFirst 3 Parallel Charging Events:")
        for i, event in enumerate(parallel_ch[:3], 1):
            print(f"  {i}. Hour {event['hour']:.2f}h")
            print(f"     Segments: {event['segments']}")
            print(f"     Deltas:   {[f'{d:.2f} kWh' for d in event['deltas']]}")
            print(f"     SOC:      {event['soc']:.2f} kWh")


def example_compare_runs():
    """Example: Compare two runs (with/without sequential activation)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Multiple Runs")
    print("="*80)

    # Paths to two different runs
    csv_paths = [
        "validation_results/run1_sequential_on/solution_timeseries.csv",
        "validation_results/run2_sequential_off/solution_timeseries.csv"
    ]
    labels = [
        "Sequential Activation ON",
        "Sequential Activation OFF"
    ]

    # Check if files exist
    existing_paths = [p for p in csv_paths if Path(p).exists()]
    if len(existing_paths) < 2:
        print("\nNote: This example requires 2 solution files to compare.")
        print("Update the paths in this script to match your actual files.")
        return

    # Create comparison plot
    fig = plot_segment_comparison(
        csv_paths=csv_paths,
        labels=labels,
        output_path="validation_results/segment_comparison.html",
        width=1400,
        height=800
    )

    print("\nComparison plot created!")


def example_programmatic_usage():
    """Example: Use in a script/notebook for analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Programmatic Usage")
    print("="*80)

    csv_path = Path("validation_results/optimizer_validation") / \
               "20251114_223822_notebook_test_modeliii_ch_24h_alpha1.0" / \
               "solution_timeseries.csv"

    # Load data
    df = pd.read_csv(csv_path)

    # Extract segment data
    segment_cols = [f'segment_{i}' for i in range(1, 11)]
    segment_data = df[segment_cols]

    # Check max energy in each segment
    print("\nMaximum energy in each segment:")
    for i, col in enumerate(segment_cols, 1):
        max_energy = df[col].max()
        print(f"  Segment {i:2d}: {max_energy:7.2f} kWh "
              f"({max_energy/447.2*100:5.1f}% of capacity)")

    # Check for parallel operations
    parallel_ch, parallel_dis = detect_parallel_operations(df)

    print(f"\nViolation check:")
    print(f"  Parallel charging:     {len(parallel_ch)} events")
    print(f"  Parallel discharging:  {len(parallel_dis)} events")

    if len(parallel_ch) == 0 and len(parallel_dis) == 0:
        print("\n  ✓ PASS: No violations detected - LIFO constraint enforced correctly")
    else:
        print(f"\n  ✗ FAIL: Found {len(parallel_ch) + len(parallel_dis)} violations")

    # Calculate total throughput per segment
    print("\nThroughput per segment (sum of absolute deltas):")
    for i, col in enumerate(segment_cols, 1):
        delta = df[col].diff().fillna(0)
        throughput = delta.abs().sum()
        print(f"  Segment {i:2d}: {throughput:8.2f} kWh")


if __name__ == '__main__':
    # Run examples
    example_single_run()
    example_detect_violations_only()
    example_programmatic_usage()

    # Uncomment to run comparison example (requires 2 runs)
    # example_compare_runs()

    print("\n" + "="*80)
    print("Examples complete! Check the generated HTML files.")
    print("="*80 + "\n")
