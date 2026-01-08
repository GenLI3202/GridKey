"""
Analyze segment time series to detect LIFO violations.

LIFO (Last-In-First-Out) principle for battery segments:
- Charging: Must fill segment j-1 COMPLETELY before segment j receives ANY energy
- Discharging: Must empty segment j COMPLETELY before segment j-1 discharges ANY energy

This script detects violations of these rules.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def check_lifo_violations(csv_path: str, epsilon: float = 0.01):
    """
    Check for LIFO violations in segment charging/discharging.

    Args:
        csv_path: Path to solution_timeseries.csv
        epsilon: Tolerance for "full" segment (447.2 - epsilon is considered full)

    Returns:
        dict with violation analysis
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Segment columns
    segment_cols = [f'segment_{i}' for i in range(1, 11)]
    E_seg = 447.2  # Segment capacity in kWh

    violations = []
    n_timesteps = len(df)

    print(f"\n{'='*80}")
    print(f"LIFO VIOLATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Data file: {Path(csv_path).name}")
    print(f"Time steps: {n_timesteps}")
    print(f"Segment capacity: {E_seg} kWh")
    print(f"Epsilon tolerance: {epsilon} kWh")
    print(f"{'='*80}\n")

    for t in range(n_timesteps):
        row = df.iloc[t]
        hour = row['hour']
        soc_total = row['soc_kwh']

        # Get segment energies
        seg_energies = [row[col] for col in segment_cols]

        # Check LIFO constraint: segment j can have energy only if segment j-1 is full
        for j in range(1, 10):  # j = 1 to 9 (segments 2 to 10 in 1-indexed)
            seg_j_minus_1 = seg_energies[j-1]  # segment j (0-indexed)
            seg_j = seg_energies[j]  # segment j+1 (0-indexed)

            # Violation: segment j+1 has energy but segment j is not full
            if seg_j > epsilon and seg_j_minus_1 < (E_seg - epsilon):
                violation = {
                    'timestep': t,
                    'hour': hour,
                    'segment_lower': j,  # 0-indexed segment that should be full
                    'segment_upper': j+1,  # 0-indexed segment that shouldn't have energy yet
                    'energy_lower': seg_j_minus_1,
                    'energy_upper': seg_j,
                    'shortfall_lower': E_seg - seg_j_minus_1,
                    'soc_total': soc_total,
                    'type': 'LIFO_VIOLATION'
                }
                violations.append(violation)

        # Check for parallel charging/discharging (multiple segments changing simultaneously)
        if t > 0:
            prev_energies = [df.iloc[t-1][col] for col in segment_cols]
            delta_energies = [seg_energies[i] - prev_energies[i] for i in range(10)]

            # Count how many segments are charging (positive delta)
            charging_segments = [i+1 for i, delta in enumerate(delta_energies) if delta > epsilon]
            discharging_segments = [i+1 for i, delta in enumerate(delta_energies) if delta < -epsilon]

            if len(charging_segments) > 1:
                violations.append({
                    'timestep': t,
                    'hour': hour,
                    'type': 'PARALLEL_CHARGING',
                    'segments': charging_segments,
                    'deltas': [delta_energies[i-1] for i in charging_segments],
                    'soc_total': soc_total
                })

            if len(discharging_segments) > 1:
                violations.append({
                    'timestep': t,
                    'hour': hour,
                    'type': 'PARALLEL_DISCHARGING',
                    'segments': discharging_segments,
                    'deltas': [delta_energies[i-1] for i in discharging_segments],
                    'soc_total': soc_total
                })

    # Print violations
    print(f"\nVIOLATION SUMMARY:")
    print(f"{'-'*80}")

    lifo_violations = [v for v in violations if v['type'] == 'LIFO_VIOLATION']
    parallel_ch_violations = [v for v in violations if v['type'] == 'PARALLEL_CHARGING']
    parallel_dis_violations = [v for v in violations if v['type'] == 'PARALLEL_DISCHARGING']

    print(f"LIFO violations: {len(lifo_violations)}")
    print(f"Parallel charging violations: {len(parallel_ch_violations)}")
    print(f"Parallel discharging violations: {len(parallel_dis_violations)}")
    print(f"Total violations: {len(violations)}")
    print(f"{'-'*80}\n")

    # Print detailed LIFO violations
    if lifo_violations:
        print(f"\nDETAILED LIFO VIOLATIONS:")
        print(f"{'-'*80}")
        for i, v in enumerate(lifo_violations[:10], 1):  # Show first 10
            print(f"\nViolation {i}:")
            print(f"  Time step: {v['timestep']}, Hour: {v['hour']:.2f}")
            print(f"  Segment {v['segment_lower']+1} (lower): {v['energy_lower']:.2f} kWh (should be {E_seg:.2f})")
            print(f"  Segment {v['segment_upper']+1} (upper): {v['energy_upper']:.2f} kWh (should be 0)")
            print(f"  Shortfall in segment {v['segment_lower']+1}: {v['shortfall_lower']:.2f} kWh")
            print(f"  Total SOC: {v['soc_total']:.2f} kWh")

        if len(lifo_violations) > 10:
            print(f"\n... and {len(lifo_violations)-10} more violations")

    # Print detailed parallel charging violations
    if parallel_ch_violations:
        print(f"\n\nDETAILED PARALLEL CHARGING VIOLATIONS:")
        print(f"{'-'*80}")
        for i, v in enumerate(parallel_ch_violations[:10], 1):
            print(f"\nViolation {i}:")
            print(f"  Time step: {v['timestep']}, Hour: {v['hour']:.2f}")
            print(f"  Segments charging simultaneously: {v['segments']}")
            print(f"  Energy deltas: {[f'{d:.2f}' for d in v['deltas']]} kWh")
            print(f"  Total SOC: {v['soc_total']:.2f} kWh")

        if len(parallel_ch_violations) > 10:
            print(f"\n... and {len(parallel_ch_violations)-10} more violations")

    # Sample segment state at a few time points
    print(f"\n\nSAMPLE SEGMENT STATES:")
    print(f"{'-'*80}")
    sample_times = [0, n_timesteps//4, n_timesteps//2, 3*n_timesteps//4, n_timesteps-1]

    for t in sample_times:
        row = df.iloc[t]
        print(f"\nt={t}, hour={row['hour']:.2f}, SOC={row['soc_kwh']:.2f} kWh ({row['soc_pct']:.1f}%)")
        seg_energies = [row[col] for col in segment_cols]
        print(f"  Segments: {[f'{e:.1f}' for e in seg_energies]}")

    return {
        'total_violations': len(violations),
        'lifo_violations': len(lifo_violations),
        'parallel_charging_violations': len(parallel_ch_violations),
        'parallel_discharging_violations': len(parallel_dis_violations),
        'violations': violations,
        'df': df
    }


if __name__ == '__main__':
    # Path to the solution file
    csv_path = r'H:\TUM-PC\TUM_CEM_PhD\a_tech_arena_hw\TechArena2025_EMS\validation_results\optimizer_validation\20251114_223822_notebook_test_modeliii_ch_24h_alpha1.0\solution_timeseries.csv'

    results = check_lifo_violations(csv_path, epsilon=0.1)

    if results['total_violations'] == 0:
        print("\n" + "="*80)
        print("✓ NO VIOLATIONS FOUND - LIFO constraint properly enforced!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ VIOLATIONS DETECTED - See details above")
        print("="*80)
