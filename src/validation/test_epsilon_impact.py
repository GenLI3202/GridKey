"""
Test script to systematically evaluate the impact of epsilon parameter on solver performance.

Based on notebook p2b_optimizer.ipynb code (Sections 1-3).
Tests epsilon values: 0.0, 0.5, 1.0, 2.0, 5.0, 10.0 kWh

Author: Analysis script for epsilon tolerance investigation
Date: 2025-11-14
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.optimizer import BESSOptimizerModelIII
from src.data.load_process_market_data import load_preprocessed_country_data

# =============================================================================
# Configuration
# =============================================================================

TEST_COUNTRY = "CH"
TEST_C_RATE = 0.5
TEST_ALPHA = 1.0
TEST_TIME_HORIZON_HOURS = 24
TEST_START_STEP = 0
MAX_AS_RATIO = 0.8
REQUIRE_SEQUENTIAL = True  # Test with strict LIFO enforcement

# Epsilon values to test (kWh)
EPSILON_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

print("=" * 80)
print("EPSILON PARAMETER IMPACT TEST")
print("=" * 80)
print(f"Country: {TEST_COUNTRY}")
print(f"Time Horizon: {TEST_TIME_HORIZON_HOURS} hours")
print(f"C-Rate: {TEST_C_RATE}")
print(f"Alpha: {TEST_ALPHA}")
print(f"Require Sequential Activation: {REQUIRE_SEQUENTIAL}")
print(f"Testing epsilon values: {EPSILON_VALUES}")
print("=" * 80)

# =============================================================================
# Load Market Data (once)
# =============================================================================

print("\nLoading market data...")
preprocessed_dir = project_root / "data" / "parquet" / "preprocessed"
country_data = load_preprocessed_country_data(TEST_COUNTRY, data_dir=preprocessed_dir)
print(f"[OK] Loaded {len(country_data)} time steps")

# Extract time window
horizon_steps = TEST_TIME_HORIZON_HOURS * 4
end_step = TEST_START_STEP + horizon_steps
data_slice = country_data.iloc[TEST_START_STEP:end_step].copy()
data_slice.reset_index(drop=True, inplace=True)
print(f"[OK] Time window: {len(data_slice)} steps ({TEST_TIME_HORIZON_HOURS} hours)")

# =============================================================================
# Test Loop
# =============================================================================

results = []

for epsilon in EPSILON_VALUES:
    print("\n" + "=" * 80)
    print(f"Testing epsilon = {epsilon} kWh")
    print("=" * 80)

    try:
        # Step 1: Initialize optimizer
        print("  [1/3] Initializing optimizer...")
        optimizer = BESSOptimizerModelIII(
            alpha=TEST_ALPHA,
            require_sequential_segment_activation=REQUIRE_SEQUENTIAL,
            use_afrr_ev_weighting=False
        )

        # Override epsilon parameter
        optimizer.degradation_params['lifo_epsilon_kwh'] = epsilon
        optimizer.max_as_ratio = MAX_AS_RATIO

        print(f"      [OK] Optimizer initialized (epsilon={epsilon} kWh)")

        # Step 2: Build model
        print("  [2/3] Building optimization model...")
        build_start = time.time()
        model = optimizer.build_optimization_model(data_slice, c_rate=TEST_C_RATE)
        build_time = time.time() - build_start

        n_vars = model.nvariables()
        n_constrs = model.nconstraints()
        print(f"      [OK] Model built in {build_time:.2f}s ({n_vars} vars, {n_constrs} constrs)")

        # Step 3: Solve model
        print("  [3/3] Solving optimization model...")
        solve_start = time.time()
        solved_model, solver_results = optimizer.solve_model(model)
        solve_time = time.time() - solve_start

        # Extract solution
        solution_dict = optimizer.extract_solution(solved_model, solver_results)

        # Collect results
        status = solution_dict.get('status', 'unknown')
        objective = solution_dict.get('objective_value', 0)

        # Degradation metrics
        deg_metrics = solution_dict.get('degradation_metrics', {})
        cyclic_cost = deg_metrics.get('total_cyclic_cost_eur', 0)
        calendar_cost = deg_metrics.get('total_calendar_cost_eur', 0)
        total_deg = deg_metrics.get('total_degradation_cost_eur', 0)
        equiv_cycles = deg_metrics.get('equivalent_full_cycles', 0)

        # Segment usage (count how many segments were used)
        throughput_per_seg = deg_metrics.get('throughput_per_segment_kwh', {})
        segments_used = sum(1 for v in throughput_per_seg.values() if abs(v) > 1.0)  # >1 kWh threshold

        print(f"      [OK] Solved in {solve_time:.2f}s (status: {status})")
        print(f"         Objective: {objective:.2f} EUR")
        print(f"         Cyclic cost: {cyclic_cost:.2f} EUR")
        print(f"         Segments used: {segments_used}/10")

        # Store results
        results.append({
            'epsilon_kwh': epsilon,
            'build_time_sec': build_time,
            'solve_time_sec': solve_time,
            'total_time_sec': build_time + solve_time,
            'status': status,
            'n_variables': n_vars,
            'n_constraints': n_constrs,
            'objective_value': objective,
            'cyclic_cost_eur': cyclic_cost,
            'calendar_cost_eur': calendar_cost,
            'total_degradation_eur': total_deg,
            'equivalent_cycles': equiv_cycles,
            'segments_used': segments_used,
            'success': True
        })

    except Exception as e:
        print(f"      [ERROR] Error: {str(e)}")
        results.append({
            'epsilon_kwh': epsilon,
            'error': str(e),
            'success': False
        })

# =============================================================================
# Analysis & Report
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Create DataFrame
df_results = pd.DataFrame(results)

# Display results table
if df_results['success'].all():
    print("\n[OK] All tests completed successfully\n")

    # Sort by solve time for easy comparison
    df_display = df_results[['epsilon_kwh', 'solve_time_sec', 'total_time_sec',
                              'objective_value', 'cyclic_cost_eur', 'segments_used']].copy()
    df_display = df_display.sort_values('solve_time_sec')

    print(df_display.to_string(index=False))

    # Find optimal epsilon
    fastest_idx = df_results['solve_time_sec'].idxmin()
    fastest_epsilon = df_results.loc[fastest_idx, 'epsilon_kwh']
    fastest_time = df_results.loc[fastest_idx, 'solve_time_sec']

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"Fastest solve: epsilon = {fastest_epsilon} kWh ({fastest_time:.2f} sec)")

    # Compare to slowest
    slowest_idx = df_results['solve_time_sec'].idxmax()
    slowest_epsilon = df_results.loc[slowest_idx, 'epsilon_kwh']
    slowest_time = df_results.loc[slowest_idx, 'solve_time_sec']
    speedup = slowest_time / fastest_time

    print(f"Slowest solve: epsilon = {slowest_epsilon} kWh ({slowest_time:.2f} sec)")
    print(f"Speedup: {speedup:.2f}x faster")

    # Check if epsilon affects solution quality
    obj_std = df_results['objective_value'].std()
    cyclic_std = df_results['cyclic_cost_eur'].std()

    print(f"\nSolution Quality Variation:")
    print(f"  Objective value std: {obj_std:.2f} EUR ({obj_std/df_results['objective_value'].mean()*100:.2f}%)")
    print(f"  Cyclic cost std: {cyclic_std:.2f} EUR ({cyclic_std/df_results['cyclic_cost_eur'].mean()*100:.2f}%)")

    # Segment usage analysis
    print(f"\nSegment Usage:")
    for _, row in df_results.iterrows():
        print(f"  epsilon={row['epsilon_kwh']:4.1f} kWh: {row['segments_used']}/10 segments used")

else:
    print("\n⚠️ Some tests failed:")
    failed = df_results[~df_results['success']]
    print(failed[['epsilon_kwh', 'error']].to_string(index=False))

# Save results
output_dir = project_root / "validation_results" / "epsilon_tests"
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"epsilon_test_{timestamp}.csv"
df_results.to_csv(output_file, index=False)

print(f"\n[SAVE] Results saved to: {output_file}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
