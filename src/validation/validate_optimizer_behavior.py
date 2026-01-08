"""
Flexible Optimizer Behavior Validation Script

This script provides a command-line interface to run the BESS optimizer with
various configurations and scenarios. It is designed for flexible, ad-hoc testing,
behavioral validation, and "what-if" analysis without the rigidity of the formal
pytest structure.

Purpose:
- Validate optimizer behavior under specific market conditions or constraints.
- Compare results between different models (e.g., Model I vs. Model II).
- Debug specific issues by isolating scenarios (e.g., a single day, a specific bug).
- Generate quick plots and summaries for analysis.

How to Use:
- Run from the command line with different arguments.
- Example:
  python validate_optimizer_behavior.py --country DE --year 2023 --week 5 \
  --model II --alpha 1.2 --plot-soc

Based on the test refactoring plan.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to access 'core' and 'data' modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.optimizer import BESSOptimizerModelI, BESSOptimizerModelII
from data.data_processor import DataProcessor
# Assuming a plotting utility exists after the previous refactoring
from analysis.optimization_analysis import plot_soc_and_power, plot_revenue_streams

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate BESS Optimizer Behavior")
    parser.add_argument("--country", type=str, default="DE", help="Country code (e.g., DE, FR)")
    parser.add_argument("--year", type=int, default=2023, help="Year to process")
    parser.add_argument("--week", type=int, required=True, help="Week number to process (1-52)")
    parser.add_argument("--model", type=str, choices=['I', 'II'], default='II', help="Optimizer model to use")
    parser.add_argument("--alpha", type=float, default=1.0, help="Degradation cost factor for Model II")
    parser.add_argument("--c-rate", type=float, default=0.5, help="C-rate for the simulation")
    parser.add_argument("--cycle-limit", type=float, default=None, help="Daily cycle limit for Model I")
    parser.add_argument("--plot-soc", action='store_true', help="Generate and show SOC/Power plot")
    parser.add_argument("--plot-revenue", action='store_true', help="Generate and show revenue plot")
    parser.add_argument("--output-path", type=str, default="validation_output.xlsx", help="Path to save result summary")
    return parser.parse_args()

def main():
    """Main execution function."""
    args = get_args()
    print(f"--- Running Optimizer Validation for {args.country}, Year {args.year}, Week {args.week} ---")
    print(f"Model: {args.model}, Alpha: {args.alpha}, C-Rate: {args.c_rate}\n")

    # 1. Load and prepare data
    try:
        data_processor = DataProcessor(country=args.country, year=args.year)
        market_data = data_processor.get_market_data_for_week(args.week)
        print(f"Successfully loaded data for week {args.week}. Total timesteps: {len(market_data)}")
    except FileNotFoundError:
        print(f"ERROR: Data not found for {args.country}, {args.year}. Ensure data is processed.")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # 2. Initialize the correct optimizer model
    if args.model == 'I':
        optimizer = BESSOptimizerModelI()
        model_params = {'daily_cycle_limit': args.cycle_limit}
        print("Using Model I with daily cycle limit:", args.cycle_limit)
    else: # Model II
        optimizer = BESSOptimizerModelII(alpha=args.alpha)
        model_params = {}
        print(f"Using Model II with alpha (degradation cost factor): {args.alpha}")

    # 3. Run optimization
    print("\nStarting optimization...")
    results = optimizer.optimize(
        country_data=market_data,
        c_rate=args.c_rate,
        **model_params
    )
    print("Optimization finished. Status:", results['status'])

    if results['status'] not in ['optimal', 'feasible']:
        print("WARNING: Optimization did not find an optimal solution. Results may be unreliable.")
        # Decide whether to exit or proceed
        # sys.exit(1)

    # 4. Process and save results
    summary_df = results['summary_df']
    summary_df.to_excel(args.output_path)
    print(f"\nResults summary saved to '{args.output_path}'")

    if 'degradation_metrics' in results:
        print("\nDegradation Metrics (Model II):")
        for key, value in results['degradation_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  - {key.replace('_', ' ').title()}: {value:.2f}")

    # 5. Generate plots if requested
    if args.plot_soc:
        print("Generating SOC and Power plot...")
        fig_soc = plot_soc_and_power(summary_df, title=f"SOC & Power | Week {args.week} | Model {args.model}")
        fig_soc.show()

    if args.plot_revenue:
        print("Generating Revenue Streams plot...")
        fig_revenue = plot_revenue_streams(summary_df, title=f"Revenue | Week {args.week} | Model {args.model}")
        fig_revenue.show()

    print("\n--- Validation run complete ---")

if __name__ == '__main__':
    main()
