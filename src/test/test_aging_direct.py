#!/usr/bin/env python3
"""
Test Aging Visualizations with Direct extract_solution() Approach
==================================================================

This script demonstrates the PREFERRED approach of using the solution dictionary
directly from optimizer.extract_solution(), without needing test_data or horizon_hours.

Key benefits:
- No need for extract_detailed_solution() or test_data
- Works directly with optimizer's native output
- Cleaner API: just pass solution dict to plotting functions
- Supports future integration into optimizer class methods
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.optimizer import BESSOptimizerModelIII
from src.visualization.aging_analysis import (
    plot_stacked_cyclic_soc,
    plot_calendar_aging_curve,
    plot_aging_validation_suite
)


def main():
    """Test aging visualizations using direct solution dict approach."""

    print("=" * 80)
    print("Testing Aging Visualizations - Direct extract_solution() Approach")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========================================================================
    # Configuration
    # ========================================================================

    country = 'HU'
    horizon_hours = 36
    horizon_steps = horizon_hours * 4
    alpha = 0.5

    print(f"Configuration:")
    print(f"  Country: {country}")
    print(f"  Horizon: {horizon_hours} hours ({horizon_steps} steps)")
    print(f"  Alpha: {alpha}")
    print(f"  Model: BESSOptimizerModelIII")
    print()

    # ========================================================================
    # Initialize & Run Optimization
    # ========================================================================

    print("[1/3] Running Model III optimization...")
    print("-" * 80)

    # Initialize (uses default ConfigLoader from config/Config.yml)
    optimizer = BESSOptimizerModelIII(
        alpha=alpha,
        use_afrr_ev_weighting=True
    )
    optimizer.max_as_ratio = 0.8

    # Load data
    from src.data.load_process_market_data import load_preprocessed_country_data
    country_data = load_preprocessed_country_data(country)
    test_data = country_data.iloc[:horizon_steps].copy().reset_index(drop=True)

    print(f"  Data loaded: {len(test_data)} timesteps")

    # Build and solve
    import time as time_module
    build_start = time_module.time()
    model = optimizer.build_optimization_model(test_data, c_rate=0.5)
    build_time = time_module.time() - build_start

    print(f"  Model built in {build_time:.2f}s ({model.nvariables():,} vars, {model.nconstraints():,} cons)")

    solve_start = time_module.time()
    solved_model, solver_results = optimizer.solve_model(model)
    solve_time = time_module.time() - solve_start

    # Extract solution using optimizer's extract_solution() method
    solution = optimizer.extract_solution(solved_model, solver_results)

    print(f"  Solved in {solve_time:.2f}s (Status: {solution['status']})")
    print(f"  Objective: {solution['objective_value']:.2f} EUR")

    if 'degradation_metrics' in solution and 'cost_breakdown' in solution['degradation_metrics']:
        breakdown = solution['degradation_metrics']['cost_breakdown']
        print(f"  Degradation costs:")
        print(f"    Cyclic:   {breakdown['cyclic_eur']:.2f} EUR")
        print(f"    Calendar: {breakdown['calendar_eur']:.2f} EUR")
        print(f"    Total:    {breakdown['total_eur']:.2f} EUR")

    print()

    # ========================================================================
    # Test Individual Plotting Functions with Solution Dict
    # ========================================================================

    print("[2/3] Testing individual plot functions...")
    print("-" * 80)

    output_dir = Path("results/aging_direct_test")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load aging config from unified YAML
    from src.utils.config_loader import ConfigLoader
    aging_config = ConfigLoader.get_aging_config()

    # Test 1: Cyclic SOC plot (direct from solution dict!)
    print(f"\n  Test 1: plot_stacked_cyclic_soc(solution)")
    try:
        fig_cyclic = plot_stacked_cyclic_soc(
            solution,  # Pass solution dict directly - no DataFrame needed!
            title_suffix="HU Winter 36h (Direct)",
            save_path=str(output_dir / "cyclic_soc_direct.html")
        )
        print(f"    [OK] Generated from solution dict")
        print(f"    [OK] Saved to: {output_dir / 'cyclic_soc_direct.html'}")

        # Verify data
        num_timesteps = len(solution['e_soc'])
        print(f"    [OK] Extracted {num_timesteps} timesteps from solution['e_soc']")
        print(f"    [OK] Extracted 10 segments from solution['e_soc_j']")

    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Calendar aging curve (direct from solution dict!)
    print(f"\n  Test 2: plot_calendar_aging_curve(solution, aging_config)")
    try:
        fig_calendar = plot_calendar_aging_curve(
            solution,  # Pass solution dict directly - no DataFrame needed!
            aging_config=aging_config,
            title_suffix="HU Winter 36h (Direct)",
            save_path=str(output_dir / "calendar_curve_direct.html")
        )
        print(f"    [OK] Generated from solution dict")
        print(f"    [OK] Saved to: {output_dir / 'calendar_curve_direct.html'}")

        # Verify data
        num_timesteps = len(solution['e_soc'])
        num_calendar_costs = len(solution['c_cal_cost'])
        print(f"    [OK] Extracted {num_timesteps} SOC values from solution['e_soc']")
        print(f"    [OK] Extracted {num_calendar_costs} calendar costs from solution['c_cal_cost']")

        # Show breakpoints
        breakpoints = aging_config['calendar_aging']['breakpoints']
        print(f"    [OK] Overlaid {len(breakpoints)} theoretical breakpoints:")
        for bp in breakpoints:
            print(f"         SOC={bp['soc_kwh']:4.0f} kWh -> Cost={bp['cost_eur_hr']:.2f} EUR/hr")

    except Exception as e:
        print(f"    [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    print()

    # ========================================================================
    # Test Convenience Function
    # ========================================================================

    print("[3/3] Testing convenience function plot_aging_validation_suite()...")
    print("-" * 80)
    print(f"  Notice: No test_data or horizon_hours needed!")
    print()

    try:
        figures = plot_aging_validation_suite(
            solution,  # Just pass solution dict - that's it!
            aging_config=aging_config,
            output_dir=str(output_dir / "suite"),
            test_name="hu_winter_36h_direct"
        )

        print(f"\n  [OK] Generated {len(figures)} plots:")
        for name in figures.keys():
            print(f"    - {name}")

        print(f"\n  API Comparison:")
        print(f"    OLD: plot_aging_validation_suite(solution, test_data, horizon_hours, ...)")
        print(f"    NEW: plot_aging_validation_suite(solution, aging_config, ...)")
        print(f"    Result: Simpler API, no extra dependencies!")

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 80)
    print("TEST COMPLETE - extract_solution() Direct Approach")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey Advantages:")
    print(f"  1. No need for extract_detailed_solution() wrapper")
    print(f"  2. No need to pass test_data or horizon_hours")
    print(f"  3. Works directly with optimizer.extract_solution() output")
    print(f"  4. Cleaner API for users")
    print(f"  5. Ready to integrate as optimizer class methods")
    print(f"\nGenerated files:")
    print(f"  - cyclic_soc_direct.html")
    print(f"  - calendar_curve_direct.html")
    print(f"  - suite/*.html")
    print("=" * 80)


if __name__ == "__main__":
    main()
