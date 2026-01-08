#!/usr/bin/env python3
"""
General-Purpose BESS Optimization Comparator
=============================================

Flexible command-line tool for comparing different BESS optimization configurations.
Replaces hardcoded comparison scripts with a parameterized, reusable utility.

Comparison Types:
- single-vs-mpc: Compare single-shot optimization vs Model Predictive Control
- models: Compare different models (I vs II vs III)
- alpha: Compare different degradation weights
- countries: Compare different market countries
- c-rates: Compare different C-rates

Features:
- Runs multiple optimizations with systematic parameter variations
- Generates side-by-side comparison plots
- Creates summary comparison table
- Saves all results in organized directory structure

Usage Examples:
    # Compare 32h single vs MPC (replaces test_single_32h_vs_mpc.py)
    python compare_optimizations.py --compare-type single-vs-mpc --hours 32 --country HU

    # Compare Model I vs Model II vs Model III
    python compare_optimizations.py --compare-type models --models I II III --hours 24 --country DE

    # Compare different alphas for Model III
    python compare_optimizations.py --compare-type alpha --alphas 0.5 1.0 1.5 --hours 36 --country HU

    # Compare countries with Model II
    python compare_optimizations.py --compare-type countries --countries DE_LU AT CH --hours 48 --model II

Author: TechArena 2025 Team
Date: November 2025
"""

import sys
import argparse
import json
import time as time_module
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.optimizer import (
    BESSOptimizerModelI,
    BESSOptimizerModelII,
    BESSOptimizerModelIII
)
from src.visualization.optimization_analysis import extract_detailed_solution
from src.validation.results_exporter import save_optimization_results

# Try to import MPC simulator if available
try:
    from src.mpc.mpc_simulator import MPCSimulator
    MPC_AVAILABLE = True
except ImportError:
    MPC_AVAILABLE = False
    print("Warning: MPC simulator not available. single-vs-mpc comparison disabled.")


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare BESS optimization configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Comparison Types:
  single-vs-mpc : Compare single optimization vs MPC approach
  models        : Compare different models (I, II, III)
  alpha         : Compare different degradation weights
  countries     : Compare different markets
  c-rates       : Compare different C-rates

Examples:
  # Compare 32h single vs MPC for HU
  python compare_optimizations.py --compare-type single-vs-mpc --hours 32 --country HU

  # Compare all three models for 24h DE
  python compare_optimizations.py --compare-type models --models I II III --hours 24 --country DE

  # Alpha sensitivity for Model III
  python compare_optimizations.py --compare-type alpha --alphas 0.5 0.75 1.0 1.25 1.5 --hours 36 --country HU --model III

  # Compare countries
  python compare_optimizations.py --compare-type countries --countries DE_LU AT CH HU --hours 48 --model II --alpha 1.0
        """
    )

    # Required arguments
    parser.add_argument(
        '--compare-type',
        type=str,
        required=True,
        choices=['single-vs-mpc', 'models', 'alpha', 'countries', 'c-rates'],
        help='Type of comparison to perform'
    )
    parser.add_argument(
        '--hours',
        type=int,
        required=True,
        help='Time horizon in hours'
    )

    # Comparison-specific arguments
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['I', 'II', 'III'],
        help='Models to compare (for --compare-type models)'
    )
    parser.add_argument(
        '--alphas',
        type=float,
        nargs='+',
        help='Alpha values to compare (for --compare-type alpha)'
    )
    parser.add_argument(
        '--countries',
        type=str,
        nargs='+',
        choices=['DE_LU', 'AT', 'CH', 'HU', 'CZ'],
        help='Countries to compare (for --compare-type countries)'
    )
    parser.add_argument(
        '--c-rates',
        type=float,
        nargs='+',
        help='C-rates to compare (for --compare-type c-rates)'
    )

    # Common parameters (used as baseline when not being compared)
    parser.add_argument(
        '--model',
        type=str,
        default='III',
        choices=['I', 'II', 'III'],
        help='Model to use (when not comparing models, default: III)'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='HU',
        choices=['DE_LU', 'AT', 'CH', 'HU', 'CZ'],
        help='Country (when not comparing countries, default: HU)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Alpha value (when not comparing alpha, default: 1.0)'
    )
    parser.add_argument(
        '--c-rate',
        type=float,
        default=0.5,
        help='C-rate (when not comparing c-rates, default: 0.5)'
    )

    # MPC-specific parameters
    parser.add_argument(
        '--mpc-horizon',
        type=int,
        default=32,
        help='MPC optimization horizon in hours (default: 32)'
    )
    parser.add_argument(
        '--mpc-update',
        type=int,
        default=24,
        help='MPC update interval in hours (default: 24)'
    )

    # Other parameters
    parser.add_argument(
        '--start-step',
        type=int,
        default=0,
        help='Starting time step (default: 0)'
    )
    parser.add_argument(
        '--solver',
        type=str,
        default=None,
        help='Solver to use (default: auto-detect)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/parquet',
        help='Market data directory (default: data/parquet)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_results/comparisons',
        help='Output directory (default: validation_results/comparisons)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating individual result plots'
    )

    return parser.parse_args()


# ============================================================================
# Utility Functions
# ============================================================================

def load_market_data_flexible(data_dir: str, country: str) -> pd.DataFrame:
    """
    Load market data from preprocessed parquet (Phase 2 pipeline).

    Args:
        data_dir: Directory containing market data
        country: Country code

    Returns:
        DataFrame with market data for the country
    """
    from src.data.load_process_market_data import load_preprocessed_country_data

    # Try preprocessed country parquet first (Phase 2 fast path)
    preprocessed_path = Path("data/parquet/preprocessed") / f"{country.lower()}.parquet"

    if preprocessed_path.exists():
        return load_preprocessed_country_data(country)

    # Fallback: Try old data_dir location
    data_parquet_path = Path(data_dir) / f"{country.lower()}_market_data.parquet"
    if data_parquet_path.exists():
        return pd.read_parquet(data_parquet_path)

    # Last resort: Load from Excel
    excel_path = Path("data/TechArena2025_Phase2_data.xlsx")
    if excel_path.exists():
        temp_opt = BESSOptimizerModelI()
        full_data = temp_opt.load_and_preprocess_data(str(excel_path))
        return temp_opt.extract_country_data(full_data, country)

    raise FileNotFoundError(f"No market data found for {country}. Please generate preprocessed files or provide Excel workbook.")


def run_single_optimization(
    model_type: str,
    country: str,
    data_slice: pd.DataFrame,
    c_rate: float,
    alpha: float = None,
    solver: str = None,
    **kwargs
) -> Tuple[Dict[str, Any], pd.DataFrame, float]:
    """
    Run a single optimization and return results.

    Returns:
        Tuple of (solution_dict, solution_df, solve_time)
    """
    # Initialize optimizer
    if model_type == 'I':
        optimizer = BESSOptimizerModelI()
    elif model_type == 'II':
        optimizer = BESSOptimizerModelII(alpha=alpha or 1.0)
    else:  # Model III
        optimizer = BESSOptimizerModelIII(alpha=alpha or 1.0)

    # Build model
    model = optimizer.build_optimization_model(data_slice, c_rate=c_rate)

    # Solve
    start = time_module.time()
    solved_model, solver_results = optimizer.solve_model(model, solver)
    solution = optimizer.extract_solution(solved_model, solver_results)
    solve_time = time_module.time() - start

    # Extract DataFrame
    hours = len(data_slice) // 4
    solution_df = extract_detailed_solution(solution, data_slice, hours)

    return solution, solution_df, solve_time


def run_mpc_simulation(
    model_type: str,
    country: str,
    data: pd.DataFrame,
    c_rate: float,
    horizon_hours: int,
    update_hours: int,
    alpha: float = None,
    solver: str = None
) -> Tuple[Dict[str, Any], pd.DataFrame, float]:
    """
    Run MPC simulation.

    Returns:
        Tuple of (aggregated_solution_dict, full_trajectory_df, total_solve_time)
    """
    if not MPC_AVAILABLE:
        raise RuntimeError("MPC simulator not available")

    # Initialize base optimizer
    if model_type == 'I':
        optimizer = BESSOptimizerModelI()
    elif model_type == 'II':
        optimizer = BESSOptimizerModelII(alpha=alpha or 1.0)
    else:  # Model III
        optimizer = BESSOptimizerModelIII(alpha=alpha or 1.0)

    # Initialize MPC simulator
    mpc_sim = MPCSimulator(
        optimizer=optimizer,
        horizon_hours=horizon_hours,
        update_interval_hours=update_hours
    )

    # Run simulation
    start = time_module.time()
    mpc_results = mpc_sim.run_simulation(
        market_data=data,
        c_rate=c_rate,
        solver_name=solver
    )
    total_time = time_module.time() - start

    # Extract results
    trajectory_df = mpc_results['trajectory']
    total_profit = mpc_results['total_profit']

    # Create aggregated solution dict for consistency
    solution = {
        'status': 'optimal' if mpc_results['success'] else 'failed',
        'objective_value': total_profit,
        'solve_time': total_time,
        'mpc_iterations': mpc_results.get('iterations', 0)
    }

    return solution, trajectory_df, total_time


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_single_vs_mpc(args) -> pd.DataFrame:
    """Compare single-shot optimization vs MPC."""
    if not MPC_AVAILABLE:
        print("ERROR: MPC simulator not available. Cannot run single-vs-mpc comparison.")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Comparing: Single-Shot vs MPC")
    print(f"{'='*80}")
    print(f"Country: {args.country} | Hours: {args.hours} | Model: {args.model}")
    print(f"MPC Horizon: {args.mpc_horizon}h | MPC Update: {args.mpc_update}h")

    # Load data
    print(f"\nLoading market data...")
    country_data = load_market_data_flexible(args.data_dir, args.country)
    horizon_steps = args.hours * 4
    data_slice = country_data.iloc[args.start_step:args.start_step + horizon_steps].copy()
    data_slice.reset_index(drop=True, inplace=True)

    results = []

    # Run single-shot
    print(f"\n[1/2] Running single-shot optimization ({args.hours}h)...")
    try:
        solution_single, df_single, time_single = run_single_optimization(
            args.model, args.country, data_slice, args.c_rate, args.alpha, args.solver
        )
        print(f"  Status: {solution_single['status']}")
        print(f"  Profit: {solution_single['objective_value']:.2f} EUR")
        print(f"  Time:   {time_single:.2f}s")

        results.append({
            'approach': 'Single-Shot',
            'profit_eur': solution_single['objective_value'],
            'solve_time_sec': time_single,
            'status': solution_single['status']
        })
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            'approach': 'Single-Shot',
            'profit_eur': None,
            'solve_time_sec': None,
            'status': 'error'
        })

    # Run MPC
    print(f"\n[2/2] Running MPC simulation (horizon={args.mpc_horizon}h, update={args.mpc_update}h)...")
    try:
        solution_mpc, df_mpc, time_mpc = run_mpc_simulation(
            args.model, args.country, data_slice, args.c_rate,
            args.mpc_horizon, args.mpc_update, args.alpha, args.solver
        )
        print(f"  Status: {solution_mpc['status']}")
        print(f"  Profit: {solution_mpc['objective_value']:.2f} EUR")
        print(f"  Time:   {time_mpc:.2f}s")
        print(f"  Iterations: {solution_mpc.get('mpc_iterations', 0)}")

        results.append({
            'approach': 'MPC',
            'profit_eur': solution_mpc['objective_value'],
            'solve_time_sec': time_mpc,
            'status': solution_mpc['status'],
            'mpc_iterations': solution_mpc.get('mpc_iterations', 0)
        })
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'approach': 'MPC',
            'profit_eur': None,
            'solve_time_sec': None,
            'status': 'error'
        })

    return pd.DataFrame(results)


def compare_models(args) -> pd.DataFrame:
    """Compare different optimizer models."""
    print(f"\n{'='*80}")
    print(f"Comparing Models: {', '.join(args.models)}")
    print(f"{'='*80}")
    print(f"Country: {args.country} | Hours: {args.hours} | C-Rate: {args.c_rate}")

    # Load data
    print(f"\nLoading market data...")
    country_data = load_market_data_flexible(args.data_dir, args.country)
    horizon_steps = args.hours * 4
    data_slice = country_data.iloc[args.start_step:args.start_step + horizon_steps].copy()
    data_slice.reset_index(drop=True, inplace=True)

    results = []

    for i, model in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Running Model {model}...")
        try:
            solution, df, solve_time = run_single_optimization(
                model, args.country, data_slice, args.c_rate, args.alpha, args.solver
            )
            print(f"  Status: {solution['status']}")
            print(f"  Profit: {solution['objective_value']:.2f} EUR")
            print(f"  Time:   {solve_time:.2f}s")

            result = {
                'model': model,
                'profit_eur': solution['objective_value'],
                'solve_time_sec': solve_time,
                'status': solution['status']
            }

            # Add degradation metrics if available
            if 'degradation_metrics' in solution:
                dm = solution['degradation_metrics']
                if 'total_cyclic_cost_eur' in dm:
                    result['cyclic_cost_eur'] = dm['total_cyclic_cost_eur']
                if 'total_calendar_cost_eur' in dm:
                    result['calendar_cost_eur'] = dm['total_calendar_cost_eur']

            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'model': model,
                'profit_eur': None,
                'solve_time_sec': None,
                'status': 'error'
            })

    return pd.DataFrame(results)


def compare_alpha(args) -> pd.DataFrame:
    """Compare different alpha values."""
    print(f"\n{'='*80}")
    print(f"Comparing Alpha Values: {', '.join(map(str, args.alphas))}")
    print(f"{'='*80}")
    print(f"Model: {args.model} | Country: {args.country} | Hours: {args.hours}")

    if args.model == 'I':
        print("WARNING: Model I does not use alpha parameter. Results will be identical.")

    # Load data
    print(f"\nLoading market data...")
    country_data = load_market_data_flexible(args.data_dir, args.country)
    horizon_steps = args.hours * 4
    data_slice = country_data.iloc[args.start_step:args.start_step + horizon_steps].copy()
    data_slice.reset_index(drop=True, inplace=True)

    results = []

    for i, alpha in enumerate(args.alphas, 1):
        print(f"\n[{i}/{len(args.alphas)}] Running with alpha={alpha}...")
        try:
            solution, df, solve_time = run_single_optimization(
                args.model, args.country, data_slice, args.c_rate, alpha, args.solver
            )
            print(f"  Status: {solution['status']}")
            print(f"  Profit: {solution['objective_value']:.2f} EUR")
            print(f"  Time:   {solve_time:.2f}s")

            result = {
                'alpha': alpha,
                'profit_eur': solution['objective_value'],
                'solve_time_sec': solve_time,
                'status': solution['status']
            }

            # Add degradation metrics
            if 'degradation_metrics' in solution:
                dm = solution['degradation_metrics']
                if 'total_cyclic_cost_eur' in dm:
                    result['cyclic_cost_eur'] = dm['total_cyclic_cost_eur']
                if 'total_calendar_cost_eur' in dm:
                    result['calendar_cost_eur'] = dm['total_calendar_cost_eur']
                if 'total_degradation_cost_eur' in dm:
                    result['total_degrad_cost_eur'] = dm['total_degradation_cost_eur']

            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'alpha': alpha,
                'profit_eur': None,
                'solve_time_sec': None,
                'status': 'error'
            })

    return pd.DataFrame(results)


def compare_countries(args) -> pd.DataFrame:
    """Compare different countries."""
    print(f"\n{'='*80}")
    print(f"Comparing Countries: {', '.join(args.countries)}")
    print(f"{'='*80}")
    print(f"Model: {args.model} | Hours: {args.hours} | C-Rate: {args.c_rate}")

    results = []

    for i, country in enumerate(args.countries, 1):
        print(f"\n[{i}/{len(args.countries)}] Running {country}...")
        try:
            # Load country-specific data
            country_data = load_market_data_flexible(args.data_dir, country)
            horizon_steps = args.hours * 4
            data_slice = country_data.iloc[args.start_step:args.start_step + horizon_steps].copy()
            data_slice.reset_index(drop=True, inplace=True)

            solution, df, solve_time = run_single_optimization(
                args.model, country, data_slice, args.c_rate, args.alpha, args.solver
            )
            print(f"  Status: {solution['status']}")
            print(f"  Profit: {solution['objective_value']:.2f} EUR")
            print(f"  Time:   {solve_time:.2f}s")

            results.append({
                'country': country,
                'profit_eur': solution['objective_value'],
                'solve_time_sec': solve_time,
                'status': solution['status']
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'country': country,
                'profit_eur': None,
                'solve_time_sec': None,
                'status': 'error'
            })

    return pd.DataFrame(results)


def compare_c_rates(args) -> pd.DataFrame:
    """Compare different C-rates."""
    print(f"\n{'='*80}")
    print(f"Comparing C-Rates: {', '.join(map(str, args.c_rates))}")
    print(f"{'='*80}")
    print(f"Model: {args.model} | Country: {args.country} | Hours: {args.hours}")

    # Load data
    print(f"\nLoading market data...")
    country_data = load_market_data_flexible(args.data_dir, args.country)
    horizon_steps = args.hours * 4
    data_slice = country_data.iloc[args.start_step:args.start_step + horizon_steps].copy()
    data_slice.reset_index(drop=True, inplace=True)

    results = []

    for i, c_rate in enumerate(args.c_rates, 1):
        print(f"\n[{i}/{len(args.c_rates)}] Running with C-rate={c_rate}...")
        try:
            solution, df, solve_time = run_single_optimization(
                args.model, args.country, data_slice, c_rate, args.alpha, args.solver
            )
            print(f"  Status: {solution['status']}")
            print(f"  Profit: {solution['objective_value']:.2f} EUR")
            print(f"  Time:   {solve_time:.2f}s")

            battery_cap = 4472  # kWh
            max_power = c_rate * battery_cap

            results.append({
                'c_rate': c_rate,
                'max_power_kw': max_power,
                'profit_eur': solution['objective_value'],
                'solve_time_sec': solve_time,
                'status': solution['status']
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'c_rate': c_rate,
                'max_power_kw': c_rate * 4472,
                'profit_eur': None,
                'solve_time_sec': None,
                'status': 'error'
            })

    return pd.DataFrame(results)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    args = parse_args()

    # Print header
    print("=" * 80)
    print("BESS Optimization Comparator")
    print("=" * 80)

    # Route to appropriate comparison function
    if args.compare_type == 'single-vs-mpc':
        comparison_df = compare_single_vs_mpc(args)
    elif args.compare_type == 'models':
        if not args.models:
            print("ERROR: --models required for models comparison")
            sys.exit(1)
        comparison_df = compare_models(args)
    elif args.compare_type == 'alpha':
        if not args.alphas:
            print("ERROR: --alphas required for alpha comparison")
            sys.exit(1)
        comparison_df = compare_alpha(args)
    elif args.compare_type == 'countries':
        if not args.countries:
            print("ERROR: --countries required for countries comparison")
            sys.exit(1)
        comparison_df = compare_countries(args)
    elif args.compare_type == 'c-rates':
        if not args.c_rates:
            print("ERROR: --c-rates required for c-rates comparison")
            sys.exit(1)
        comparison_df = compare_c_rates(args)
    else:
        print(f"ERROR: Unknown comparison type: {args.compare_type}")
        sys.exit(1)

    # Print results table
    print(f"\n{'='*80}")
    print("Comparison Results")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{args.compare_type}_comparison.csv"
    output_path = output_dir / filename

    comparison_df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
