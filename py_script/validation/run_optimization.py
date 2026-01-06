#!/usr/bin/env python3
"""
General-Purpose BESS Optimization Runner
=========================================

Flexible command-line tool for running BESS optimization with any configuration.
Replaces hardcoded test scripts with a parameterized, reusable utility.

Features:
- Supports all three models (I, II, III)
- Flexible time windows (any hours, any start point)
- Any country from available data
- Configurable solver, alpha, c-rate, daily cycles
- Auto-generates standard visualizations
- Saves results in organized timestamped directories

Usage Examples:
    # Run 36h HU winter with Model III
    python run_optimization.py --model III --country HU --hours 36 --start-step 0 --alpha 0.5 --plots

    # Run 48h DE summer with Model II
    python run_optimization.py --model II --country DE --hours 48 --start-step 4000 --alpha 1.0

    # Quick 12h test with Model I
    python run_optimization.py --model I --country AT --hours 12 --c-rate 0.33 --cycles 1.5

Author: TechArena 2025 Team
Date: November 2025
"""

import sys
import argparse
import json
import time as time_module
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from py_script.core.optimizer import (
    BESSOptimizerModelI,
    BESSOptimizerModelII,
    BESSOptimizerModelIII
)
from py_script.visualization.optimization_analysis import (
    extract_detailed_solution,
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid,
    plot_soc_and_power_bids
)
from py_script.validation.results_exporter import save_optimization_results


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BESS optimization with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Model III: 36h HU winter, alpha=0.5, with plots
  python run_optimization.py --model III --country HU --hours 36 --start-step 0 --alpha 0.5 --plots

  # Model II: 48h DE, alpha=1.0, custom c-rate
  python run_optimization.py --model II --country DE --hours 48 --c-rate 0.33 --alpha 1.0

  # Model I: 24h AT with daily cycle limit
  python run_optimization.py --model I --country AT --hours 24 --cycles 1.5

  # Quick test: 12h with CBC solver
  python run_optimization.py --model I --country CH --hours 12 --solver cbc
        """
    )

    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['I', 'II', 'III'],
        help='Optimizer model to use (I: base, II: +cyclic aging, III: +calendar aging)'
    )
    parser.add_argument(
        '--country',
        type=str,
        required=True,
        choices=['DE_LU', 'AT', 'CH', 'HU', 'CZ'],
        help='Country code for market data'
    )
    parser.add_argument(
        '--hours',
        type=int,
        required=True,
        help='Time horizon in hours (will be converted to 15-min steps)'
    )

    # Optional arguments
    parser.add_argument(
        '--start-step',
        type=int,
        default=0,
        help='Starting time step (15-min intervals, default: 0)'
    )
    parser.add_argument(
        '--c-rate',
        type=float,
        default=0.5,
        help='Battery C-rate (default: 0.5)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Degradation cost weight for Model II/III (default: 1.0 for II/III, N/A for I)'
    )
    parser.add_argument(
        '--cycles',
        type=float,
        default=None,
        help='Daily cycle limit for Model I (ignored for Model II/III)'
    )
    parser.add_argument(
        '--solver',
        type=str,
        default=None,
        choices=['cplex', 'gurobi', 'highs', 'cbc', 'glpk'],
        help='Solver to use (default: auto-detect best available)'
    )
    parser.add_argument(
        '--max-as-ratio',
        type=float,
        default=0.8,
        help='Maximum ancillary service ratio (default: 0.8 = 80%%)'
    )
    parser.add_argument(
        '--use-ev-weighting',
        action='store_true',
        help='Enable expected value weighting for aFRR energy bids (Model I/II/III)'
    )

    # Output control
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate and save all standard plots'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_results',
        help='Base output directory (default: validation_results)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Custom run name (default: auto-generated from params)'
    )

    # Data source
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/parquet',
        help='Directory containing market data parquet files (default: data/parquet)'
    )

    return parser.parse_args()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    args = parse_args()

    # Print header
    print("=" * 80)
    print("BESS Optimization Runner")
    print("=" * 80)
    print(f"Model: {args.model} | Country: {args.country} | Hours: {args.hours}")
    print(f"Start Step: {args.start_step} | C-Rate: {args.c_rate}")
    if args.alpha is not None:
        print(f"Alpha: {args.alpha}")
    if args.cycles is not None:
        print(f"Daily Cycles: {args.cycles}")
    print("=" * 80)

    # ========================================================================
    # 1. Initialize Optimizer
    # ========================================================================

    print("\n[1/5] Initializing optimizer...")

    # Set default alpha for Model II/III if not specified
    if args.model in ['II', 'III'] and args.alpha is None:
        args.alpha = 1.0
        print(f"  Using default alpha={args.alpha} for Model {args.model}")

    # Initialize the appropriate model
    if args.model == 'I':
        optimizer = BESSOptimizerModelI()
        print(f"  Initialized Model I (Base 4-market optimization)")
    elif args.model == 'II':
        optimizer = BESSOptimizerModelII(
            alpha=args.alpha,
            use_afrr_ev_weighting=args.use_ev_weighting
        )
        print(f"  Initialized Model II (Base + Cyclic Aging, alpha={args.alpha})")
    else:  # Model III
        optimizer = BESSOptimizerModelIII(
            alpha=args.alpha,
            use_afrr_ev_weighting=args.use_ev_weighting
        )
        print(f"  Initialized Model III (Base + Cyclic + Calendar Aging, alpha={args.alpha})")

    # Configure optimizer
    optimizer.max_as_ratio = args.max_as_ratio
    print(f"  Battery Capacity: {optimizer.battery_params['capacity_kwh']} kWh")
    print(f"  Max AS Ratio: {optimizer.max_as_ratio * 100:.0f}%")
    print(f"  EV Weighting: {'Enabled' if args.use_ev_weighting else 'Disabled'}")

    # ========================================================================
    # 2. Load Market Data
    # ========================================================================

    print(f"\n[2/5] Loading market data...")

    try:
        # Try to load Phase 2 parquet data if available
        # Try Phase 2 preprocessed parquet first (fastest)
        from py_script.data.load_process_market_data import load_preprocessed_country_data

        preprocessed_path = Path("data/parquet/preprocessed") / f"{args.country.lower()}.parquet"

        if preprocessed_path.exists():
            print(f"  Loading from preprocessed parquet: {preprocessed_path}")
            import pandas as pd
            country_data = load_preprocessed_country_data(args.country)
        else:
            # Fallback: Try old data_dir location
            data_parquet_path = Path(args.data_dir) / f"{args.country.lower()}_market_data.parquet"

            if data_parquet_path.exists():
                print(f"  Loading from parquet: {data_parquet_path}")
                import pandas as pd
                country_data = pd.read_parquet(data_parquet_path)
            else:
                # Last resort: Load from Excel (submission path)
                excel_path = Path("data/TechArena2025_Phase2_data.xlsx")
                if excel_path.exists():
                    print(f"  Loading from Excel: {excel_path}")
                    temp_opt = BESSOptimizerModelI()
                    full_data = temp_opt.load_and_preprocess_data(str(excel_path))
                    country_data = temp_opt.extract_country_data(full_data, args.country)
                else:
                    raise FileNotFoundError(f"No data found. Please generate preprocessed files or provide Excel workbook.")

        print(f"  Loaded {len(country_data)} total time steps for {args.country}")

        # Extract time window
        horizon_steps = args.hours * 4  # 15-min intervals
        end_step = args.start_step + horizon_steps

        if end_step > len(country_data):
            print(f"  ERROR: Requested end step {end_step} exceeds available data {len(country_data)}")
            sys.exit(1)

        data_slice = country_data.iloc[args.start_step:end_step].copy()
        data_slice.reset_index(drop=True, inplace=True)

        print(f"  Extracted time window: steps {args.start_step} to {end_step} ({args.hours} hours)")
        print(f"  Time steps: {len(data_slice)}")

        # Show data summary
        print(f"\n  Market Data Summary:")
        print(f"    DA Price:    {data_slice['price_day_ahead'].min():.2f} - {data_slice['price_day_ahead'].max():.2f} EUR/MWh")
        print(f"    FCR Price:   {data_slice['price_fcr'].min():.2f} - {data_slice['price_fcr'].max():.2f} EUR/MW")
        print(f"    aFRR+ Price: {data_slice['price_afrr_pos'].min():.2f} - {data_slice['price_afrr_pos'].max():.2f} EUR/MW")

    except FileNotFoundError as e:
        print(f"  ERROR: Market data not found: {e}")
        print(f"  Please ensure one of the following:")
        print(f"    1. Preprocessed parquet files: data/parquet/preprocessed/<country>.parquet")
        print(f"    2. Excel workbook: data/TechArena2025_Phase2_data.xlsx")
        print(f"  Generate preprocessed files with: python py_script/data/generate_preprocessed_country_data.py")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================================================
    # 3. Build and Solve Model
    # ========================================================================

    print(f"\n[3/5] Building and solving optimization model...")

    try:
        # Build model
        build_start = time_module.time()

        if args.model == 'I' and args.cycles is not None:
            model = optimizer.build_optimization_model(
                data_slice,
                c_rate=args.c_rate,
                daily_cycle_limit=args.cycles
            )
            print(f"  Model I built with daily_cycle_limit={args.cycles}")
        else:
            model = optimizer.build_optimization_model(
                data_slice,
                c_rate=args.c_rate
            )

        build_time = time_module.time() - build_start
        print(f"  Model built in {build_time:.2f}s")
        print(f"    Variables:   {model.nvariables()}")
        print(f"    Constraints: {model.nconstraints()}")

        # Solve model
        print(f"\n  Solving with solver: {args.solver or 'auto-detect'}...")
        solve_start = time_module.time()

        solved_model, solver_results = optimizer.solve_model(model, args.solver)
        solution = optimizer.extract_solution(solved_model, solver_results)

        solve_time = time_module.time() - solve_start

        # Check status
        if solution['status'] not in ['optimal', 'feasible']:
            print(f"  ERROR: Optimization failed with status: {solution['status']}")
            print(f"  Termination condition: {solution.get('termination_condition')}")
            sys.exit(1)

        print(f"  Optimization successful!")
        print(f"    Status:          {solution['status']}")
        print(f"    Objective Value: {solution['objective_value']:.2f} EUR")
        print(f"    Solver:          {solution.get('solver', 'unknown')}")
        print(f"    Build Time:      {build_time:.2f}s")
        print(f"    Solve Time:      {solve_time:.2f}s")
        print(f"    Total Time:      {build_time + solve_time:.2f}s")

        # Store timing in solution
        solution['build_time'] = build_time
        solution['solve_time'] = solve_time
        solution['total_time'] = build_time + solve_time

    except Exception as e:
        print(f"  ERROR: Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================================================
    # 4. Extract and Save Results
    # ========================================================================

    print(f"\n[4/5] Extracting and saving results...")

    # Extract detailed solution DataFrame
    solution_df = extract_detailed_solution(solution, data_slice, args.hours)
    print(f"  Extracted solution DataFrame: {solution_df.shape}")

    # Calculate revenue breakdown
    revenue_da = solution_df['revenue_da_eur'].sum() if 'revenue_da_eur' in solution_df.columns else 0
    revenue_fcr = solution_df['revenue_fcr_eur'].sum() if 'revenue_fcr_eur' in solution_df.columns else 0
    revenue_afrr_cap = solution_df['revenue_afrr_capacity_eur'].sum() if 'revenue_afrr_capacity_eur' in solution_df.columns else 0
    revenue_afrr_energy = solution_df['revenue_afrr_energy_eur'].sum() if 'revenue_afrr_energy_eur' in solution_df.columns else 0
    total_revenue = revenue_da + revenue_fcr + revenue_afrr_cap + revenue_afrr_energy

    print(f"\n  Revenue Summary:")
    print(f"    DA Market:       {revenue_da:.2f} EUR")
    print(f"    FCR Capacity:    {revenue_fcr:.2f} EUR")
    print(f"    aFRR Capacity:   {revenue_afrr_cap:.2f} EUR")
    print(f"    aFRR Energy:     {revenue_afrr_energy:.2f} EUR")
    print(f"    Total Revenue:   {total_revenue:.2f} EUR")

    # Print degradation metrics if available
    if 'degradation_metrics' in solution:
        print(f"\n  Degradation Metrics:")
        dm = solution['degradation_metrics']
        if 'total_cyclic_cost_eur' in dm:
            print(f"    Cyclic Cost:     {dm['total_cyclic_cost_eur']:.2f} EUR")
        if 'total_calendar_cost_eur' in dm:
            print(f"    Calendar Cost:   {dm['total_calendar_cost_eur']:.2f} EUR")
        if 'total_degradation_cost_eur' in dm:
            print(f"    Total Degrad.:   {dm['total_degradation_cost_eur']:.2f} EUR")
        if 'equivalent_full_cycles' in dm:
            print(f"    Equiv. Cycles:   {dm['equivalent_full_cycles']:.3f}")

    # Prepare summary metrics
    summary_metrics = {
        'model': args.model,
        'country': args.country,
        'time_horizon_hours': args.hours,
        'start_step': args.start_step,
        'c_rate': args.c_rate,
        'max_as_ratio': args.max_as_ratio,
        'use_ev_weighting': args.use_ev_weighting,
        'total_profit_eur': solution['objective_value'],
        'total_revenue_eur': total_revenue,
        'revenue_da_eur': revenue_da,
        'revenue_fcr_eur': revenue_fcr,
        'revenue_afrr_capacity_eur': revenue_afrr_cap,
        'revenue_afrr_energy_eur': revenue_afrr_energy,
        'solver_status': solution['status'],
        'solver_name': solution.get('solver', 'unknown'),
        'solve_time_sec': solve_time,
        'build_time_sec': build_time,
        'total_time_sec': build_time + solve_time,
        'n_variables': model.nvariables(),
        'n_constraints': model.nconstraints()
    }

    # Add model-specific parameters
    if args.model in ['II', 'III']:
        summary_metrics['alpha'] = args.alpha
    if args.model == 'I' and args.cycles is not None:
        summary_metrics['daily_cycle_limit'] = args.cycles

    # Add degradation metrics
    if 'degradation_metrics' in solution:
        summary_metrics['degradation_metrics'] = solution['degradation_metrics']

    # Generate run name if not provided
    if args.run_name is None:
        run_name = f"model{args.model}_{args.country}_{args.hours}h_step{args.start_step}"
        if args.model in ['II', 'III']:
            run_name += f"_alpha{args.alpha}"
    else:
        run_name = args.run_name

    # Save results using results_exporter
    output_dir = save_optimization_results(
        solution_df,
        summary_metrics,
        run_name,
        base_output_dir=args.output_dir
    )

    print(f"  Results saved to: {output_dir}")

    # ========================================================================
    # 5. Generate Visualizations
    # ========================================================================

    if args.plots:
        print(f"\n[5/5] Generating visualizations...")

        plots_dir = output_dir / "plots"
        title_suffix = f"({args.country}, {args.hours}h, Model {args.model})"

        try:
            # Plot 1: Day-Ahead Market
            print("  [1/4] Day-Ahead Market plot...")
            fig_da = plot_da_market_price_bid(solution_df, title_suffix=title_suffix, use_timestamp=False)
            fig_da.write_html(str(plots_dir / "da_market_price_bid.html"))
            print("        Saved: da_market_price_bid.html")
        except Exception as e:
            print(f"        ERROR: {e}")

        try:
            # Plot 2: aFRR Energy Market
            print("  [2/4] aFRR Energy Market plot...")
            fig_afrr_e = plot_afrr_energy_market_price_bid(solution_df, title_suffix=title_suffix, use_timestamp=False)
            fig_afrr_e.write_html(str(plots_dir / "afrr_energy_market_price_bid.html"))
            print("        Saved: afrr_energy_market_price_bid.html")
        except Exception as e:
            print(f"        ERROR: {e}")

        try:
            # Plot 3: Capacity Markets
            print("  [3/4] Capacity Markets plot...")
            fig_cap = plot_capacity_markets_price_bid(solution_df, title_suffix=title_suffix, use_timestamp=False)
            fig_cap.write_html(str(plots_dir / "capacity_markets_price_bid.html"))
            print("        Saved: capacity_markets_price_bid.html")
        except Exception as e:
            print(f"        ERROR: {e}")

        try:
            # Plot 4: SOC & Power Bids
            print("  [4/4] SOC & Power Bids plot...")
            fig_soc = plot_soc_and_power_bids(solution_df, title_suffix=title_suffix, use_timestamp=False)
            fig_soc.write_html(str(plots_dir / "soc_and_power_bids.html"))
            print("        Saved: soc_and_power_bids.html")
        except Exception as e:
            print(f"        ERROR: {e}")

        print(f"\n  All plots saved to: {plots_dir}")
    else:
        print(f"\n[5/5] Skipping visualizations (use --plots to generate)")

    # ========================================================================
    # Done
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Results directory: {output_dir}")
    print(f"  - solution_timeseries.csv")
    print(f"  - performance_summary.json")
    if args.plots:
        print(f"  - plots/*.html (4 files)")
    print("=" * 80)


if __name__ == "__main__":
    main()
