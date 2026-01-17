#!/usr/bin/env python3
"""
Test Aging Analysis Visualizations
===================================

This script tests the new aging analysis visualization functions by running
a Model III optimization and generating the two new plots:
1. Stacked cyclic SOC segments (10 segments)
2. Calendar aging cost curve (SOS2 validation)

The test uses a short 36-hour horizon for Hungary winter data.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.optimizer import BESSOptimizerModelIII
from src.visualization.optimization_analysis import extract_detailed_solution
from src.visualization.aging_analysis import (
    plot_stacked_cyclic_soc,
    plot_calendar_aging_curve,
    plot_aging_validation_suite
)


def main():
    """Test the aging analysis visualization functions."""

    print("=" * 80)
    print("Testing Aging Analysis Visualizations")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========================================================================
    # Configuration
    # ========================================================================

    country = 'HU'
    horizon_hours = 36
    horizon_steps = horizon_hours * 4  # 15-min intervals
    alpha = 0.5  # Degradation weight

    print(f"Configuration:")
    print(f"  Country: {country}")
    print(f"  Horizon: {horizon_hours} hours ({horizon_steps} steps)")
    print(f"  Alpha: {alpha}")
    print(f"  Model: BESSOptimizerModelIII (Cyclic + Calendar Aging)")
    print()

    # ========================================================================
    # Initialize Optimizer & Load Data
    # ========================================================================

    print("[1/5] Initializing Model III and loading data...")
    print("-" * 80)

    # Uses default ConfigLoader (loads from config/Config.yml)
    optimizer = BESSOptimizerModelIII(
        alpha=alpha,
        use_afrr_ev_weighting=True
    )
    optimizer.max_as_ratio = 0.8

    print(f"  BESS Capacity: {optimizer.battery_params['capacity_kwh']} kWh")
    print(f"  Cyclic Segments: {optimizer.degradation_params.get('num_segments', 10)}")
    print(f"  Calendar Breakpoints: {len(optimizer.degradation_config['calendar_aging']['breakpoints'])}")

    # Load data
    from src.data.load_process_market_data import load_preprocessed_country_data
    country_data = load_preprocessed_country_data(country)

    # Extract winter period (36 hours)
    test_data = country_data.iloc[:horizon_steps].copy().reset_index(drop=True)

    print(f"  Loaded {len(test_data)} timesteps")
    print(f"  DA Price range: {test_data['price_day_ahead'].min():.2f} - {test_data['price_day_ahead'].max():.2f} EUR/MWh")
    print()

    # ========================================================================
    # Run Optimization
    # ========================================================================

    print("[2/5] Building and solving Model III...")
    print("-" * 80)

    import time as time_module

    build_start = time_module.time()
    model = optimizer.build_optimization_model(test_data, c_rate=0.5)
    build_time = time_module.time() - build_start

    print(f"  Model built in {build_time:.2f}s")
    print(f"  Variables: {model.nvariables():,}")
    print(f"  Constraints: {model.nconstraints():,}")

    print(f"\n  Solving...")
    solve_start = time_module.time()
    solved_model, solver_results = optimizer.solve_model(model)
    solution = optimizer.extract_solution(solved_model, solver_results)
    solve_time = time_module.time() - solve_start

    print(f"  Status: {solution['status']}")
    print(f"  Objective: {solution['objective_value']:.2f} EUR")
    print(f"  Solve time: {solve_time:.2f}s")

    if solution['status'] not in ['optimal', 'feasible']:
        print(f"\n[ERROR] Optimization failed. Cannot generate visualizations.")
        return

    # Check degradation metrics
    if 'degradation_metrics' in solution:
        metrics = solution['degradation_metrics']
        print(f"\n  Degradation Costs:")
        if 'cost_breakdown' in metrics:
            print(f"    Cyclic:   {metrics['cost_breakdown']['cyclic_eur']:.4f} EUR")
            print(f"    Calendar: {metrics['cost_breakdown']['calendar_eur']:.4f} EUR")
            print(f"    Total:    {metrics['cost_breakdown']['total_eur']:.4f} EUR")
        else:
            print(f"    Total: {metrics.get('total_degradation_cost_eur', 0):.4f} EUR")
    print()

    # ========================================================================
    # Extract Detailed Solution
    # ========================================================================

    print("[3/5] Extracting detailed solution...")
    print("-" * 80)

    df = extract_detailed_solution(solution, test_data, horizon_hours)

    print(f"  DataFrame shape: {df.shape}")
    print(f"  SOC range: {df['soc_kwh'].min():.2f} - {df['soc_kwh'].max():.2f} kWh")

    # Check if segment columns exist
    segment_cols = [f'segment_{j}' for j in range(1, 11)]
    has_segments = all(col in df.columns for col in segment_cols)
    print(f"  Segment columns present: {has_segments}")

    if has_segments:
        # Verify segment stacking
        total_from_segments = df[segment_cols].sum(axis=1)
        soc_match = ((total_from_segments - df['soc_kwh']).abs() < 0.01).all()
        print(f"  Segment sum matches SOC: {soc_match}")

        # Print segment ranges
        print(f"\n  Segment SOC Ranges (kWh):")
        for j in range(1, 11):
            col = f'segment_{j}'
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"    Segment {j:2d}: {min_val:6.2f} - {max_val:6.2f}")
    else:
        print(f"  [WARNING] Segment columns not found in DataFrame!")
        print(f"  Available columns: {list(df.columns)[:20]}...")

    # Check if calendar cost column exists
    has_cal_cost = 'cal_cost_eur_hr' in df.columns or 'c_cal_cost' in df.columns
    print(f"  Calendar cost column present: {has_cal_cost}")
    print()

    # ========================================================================
    # Generate Visualizations - Method 1: Individual Functions
    # ========================================================================

    print("[4/5] Testing individual plot functions...")
    print("-" * 80)

    output_dir = Path("results/aging_visualization_test")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Test 1: Stacked Cyclic SOC Plot
    print(f"\n  Test 1: plot_stacked_cyclic_soc()")
    try:
        if has_segments:
            fig_cyclic = plot_stacked_cyclic_soc(
                df,
                title_suffix="HU Winter 36h Test",
                save_path=str(output_dir / "test_cyclic_soc.html")
            )
            print(f"    [OK] Plot generated successfully")
            print(f"    [OK] Saved to: {output_dir / 'test_cyclic_soc.html'}")
        else:
            print(f"    [SKIP] No segment columns available")
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Calendar Aging Curve Plot
    print(f"\n  Test 2: plot_calendar_aging_curve()")
    try:
        if has_cal_cost:
            # Load aging config from unified YAML
            from src.utils.config_loader import ConfigLoader
            aging_config = ConfigLoader.get_aging_config()

            fig_calendar = plot_calendar_aging_curve(
                df,
                aging_config=aging_config,
                title_suffix="HU Winter 36h Test",
                save_path=str(output_dir / "test_calendar_curve.html")
            )
            print(f"    [OK] Plot generated successfully")
            print(f"    [OK] Saved to: {output_dir / 'test_calendar_curve.html'}")

            # Print breakpoints for reference
            breakpoints = aging_config['calendar_aging']['breakpoints']
            print(f"\n    Reference Breakpoints:")
            for i, bp in enumerate(breakpoints):
                print(f"      {i+1}. SOC={bp['soc_kwh']:4.0f} kWh -> Cost={bp['cost_eur_hr']:.2f} EUR/hr")
        else:
            print(f"    [SKIP] No calendar cost column available")
    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    print()

    # ========================================================================
    # Generate Visualizations - Method 2: Convenience Function
    # ========================================================================

    print("[5/5] Testing convenience function plot_aging_validation_suite()...")
    print("-" * 80)

    try:
        # Load aging config from unified YAML
        aging_config = ConfigLoader.get_aging_config()

        # Generate both plots using convenience function
        figures = plot_aging_validation_suite(
            solution,
            test_data,
            horizon_hours,
            aging_config=aging_config,
            output_dir=str(output_dir / "suite_output"),
            test_name="hu_winter_36h"
        )

        print(f"  [OK] Generated {len(figures)} plots:")
        for name, fig in figures.items():
            print(f"    - {name}")

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  1. test_cyclic_soc.html       - Stacked 10-segment SOC validation")
    print(f"  2. test_calendar_curve.html    - SOS2 calendar aging curve")
    print(f"  3. suite_output/               - Output from convenience function")
    print(f"\nPlease open the HTML files in a browser to validate the plots.")
    print("=" * 80)


if __name__ == "__main__":
    main()
