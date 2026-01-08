"""
Compare Single 32h Model III Optimization vs MPC Results
=========================================================
Run Model III directly on a single 32-hour instance and compare
decision variables with MPC results
"""

from src.core.optimizer import BESSOptimizerModelIII
from src.visualization.optimization_analysis import (
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid,
    plot_soc_and_power_bids,
    extract_detailed_solution
)
import pandas as pd
import json
from datetime import datetime
import os

def main():
    print("=" * 80)
    print("SINGLE 32H MODEL III vs MPC COMPARISON")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load MPC results for comparison
    print("[1/4] Loading MPC results...")
    mpc_df = pd.read_csv('results/mpc_5day_test/decision_variables_5day.csv')
    print(f"  MPC results: {len(mpc_df)} timesteps")
    print(f"  MPC SOC range: {mpc_df['e_soc'].min():.2f} - {mpc_df['e_soc'].max():.2f} kWh")
    print()

    # Initialize optimizer
    print("[2/4] Running single 32h Model III optimization...")
    optimizer = BESSOptimizerModelIII(alpha=1.0)

    # Load same data as MPC
    from src.data.load_process_market_data import load_preprocessed_country_data
    country_data = load_preprocessed_country_data('CH')

    # Take first 32 hours (128 timesteps) - same as MPC first horizon
    horizon_steps = 32 * 4  # 32 hours * 4 intervals/hour
    test_data = country_data.iloc[:horizon_steps].reset_index(drop=True)

    print(f"  Data: {len(test_data)} timesteps (32 hours)")
    print(f"  Date range: {test_data['timestamp'].iloc[0]} to {test_data['timestamp'].iloc[-1]}")
    print()

    # Build and solve
    print("  Building Model III...")
    model = optimizer.build_optimization_model(
        test_data,
        c_rate=0.5,
        daily_cycle_limit=None
    )

    print(f"  Variables: {model.nvariables()}")
    print(f"  Constraints: {model.nconstraints()}")
    print()

    print("  Solving...")
    solved_model, solver_results = optimizer.solve_model(model)
    solution = optimizer.extract_solution(solved_model, solver_results)

    print(f"  Status: {solution['status']}")
    print(f"  Solve time: {solution['solve_time']:.2f} seconds")
    print()

    # Extract detailed solution using the same function as visualizations
    print("[3/4] Extracting solution details...")
    solution_df = extract_detailed_solution(solution, test_data, horizon_hours=32)

    print(f"  Solution DataFrame: {solution_df.shape}")
    print(f"  Columns: {list(solution_df.columns[:10])}...")
    print()

    # Compare with MPC
    print("[4/4] Comparing with MPC results...")
    print("-" * 80)

    # Take first 128 timesteps from MPC for comparison
    mpc_32h = mpc_df.iloc[:horizon_steps].copy()

    comparison = {
        'metric': [],
        'single_32h': [],
        'mpc_first_32h': [],
        'difference': [],
        'match': []
    }

    # SOC comparison
    soc_single = solution_df['soc_kwh'].values
    soc_mpc = mpc_32h['e_soc'].values
    soc_diff = abs(soc_single - soc_mpc).max()

    comparison['metric'].append('SOC (kWh)')
    comparison['single_32h'].append(f"{soc_single.min():.2f} - {soc_single.max():.2f}")
    comparison['mpc_first_32h'].append(f"{soc_mpc.min():.2f} - {soc_mpc.max():.2f}")
    comparison['difference'].append(f"max diff: {soc_diff:.4f}")
    comparison['match'].append('PASS' if soc_diff < 0.1 else 'FAIL')

    # Power bids comparison
    for var in [('p_ch', 'p_ch_kw'), ('p_dis', 'p_dis_kw')]:
        mpc_var, single_var = var
        vals_single = solution_df[single_var].values / 1000  # Convert to MW
        vals_mpc = mpc_32h[mpc_var].values
        diff = abs(vals_single - vals_mpc).max()

        comparison['metric'].append(f'{mpc_var} (MW)')
        comparison['single_32h'].append(f"sum: {vals_single.sum():.2f}")
        comparison['mpc_first_32h'].append(f"sum: {vals_mpc.sum():.2f}")
        comparison['difference'].append(f"max diff: {diff:.4f}")
        comparison['match'].append('PASS' if diff < 0.001 else 'FAIL')

    # Capacity bids comparison
    for var in ['c_fcr', 'c_afrr_pos', 'c_afrr_neg']:
        vals_single = solution_df[f'{var}_mw'].values
        vals_mpc = mpc_32h[var].values
        diff = abs(vals_single - vals_mpc).max()

        comparison['metric'].append(f'{var} (MW)')
        comparison['single_32h'].append(f"mean: {vals_single.mean():.4f}")
        comparison['mpc_first_32h'].append(f"mean: {vals_mpc.mean():.4f}")
        comparison['difference'].append(f"max diff: {diff:.4f}")
        comparison['match'].append('PASS' if diff < 0.001 else 'FAIL')

    comp_df = pd.DataFrame(comparison)
    print("\nComparison Table:")
    print(comp_df.to_string(index=False))
    print()

    # Financial comparison
    print("Financial Comparison:")
    print("-" * 80)

    single_revenue = solution.get('total_revenue', 0)
    single_degrad = solution.get('degradation_metrics', {}).get('total_degradation_cost_eur', 0)
    single_profit = single_revenue - single_degrad

    print(f"Single 32h optimization:")
    print(f"  Revenue:     {single_revenue:.2f} EUR")
    print(f"  Degradation: {single_degrad:.2f} EUR")
    print(f"  Profit:      {single_profit:.2f} EUR")
    print()

    # Load MPC financial data
    with open('results/mpc_5day_test/financial_summary.json', 'r') as f:
        mpc_financial = json.load(f)

    # MPC first iteration should have solved same 32h window
    mpc_rev_first = mpc_financial['financial']['as_revenue'] / 5  # Approximate first iteration
    mpc_deg_first = mpc_financial['financial']['calendar_cost'] / 5
    mpc_profit_first = mpc_rev_first - mpc_deg_first

    print(f"MPC first iteration (approximate):")
    print(f"  Revenue:     {mpc_rev_first:.2f} EUR")
    print(f"  Degradation: {mpc_deg_first:.2f} EUR")
    print(f"  Profit:      {mpc_profit_first:.2f} EUR")
    print()

    # Save results
    output_dir = 'results/mpc_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Save single 32h solution
    solution_df.to_csv(f'{output_dir}/single_32h_solution.csv', index=False)
    print(f"Saved: {output_dir}/single_32h_solution.csv")

    # Save comparison
    comp_df.to_csv(f'{output_dir}/comparison_table.csv', index=False)
    print(f"Saved: {output_dir}/comparison_table.csv")

    # Create visualizations for single 32h
    print("\nCreating visualizations for single 32h optimization...")

    # Prepare viz_df with correct column names
    viz_df = solution_df.copy()
    viz_df['timestamp'] = test_data['timestamp'].iloc[:len(viz_df)].values

    print("  [1/4] Day-Ahead Market plot...")
    fig1 = plot_da_market_price_bid(viz_df, title_suffix="(Single 32h)", use_timestamp=True)
    fig1.write_html(f'{output_dir}/single_32h_da_market.html')

    print("  [2/4] aFRR Energy Market plot...")
    fig2 = plot_afrr_energy_market_price_bid(viz_df, title_suffix="(Single 32h)", use_timestamp=True)
    fig2.write_html(f'{output_dir}/single_32h_afrr_energy.html')

    print("  [3/4] Capacity Markets plot...")
    fig3 = plot_capacity_markets_price_bid(viz_df, title_suffix="(Single 32h)", use_timestamp=True)
    fig3.write_html(f'{output_dir}/single_32h_capacity_markets.html')

    print("  [4/4] SOC and Power Bids plot...")
    fig4 = plot_soc_and_power_bids(viz_df, title_suffix="(Single 32h)", use_timestamp=True)
    fig4.write_html(f'{output_dir}/single_32h_soc_power.html')

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Summary
    all_match = all(comp_df['match'] == 'PASS')

    if all_match:
        print("\n[PASS] RESULT: Single 32h optimization MATCHES MPC first iteration!")
        print("  This confirms MPC is correctly solving each horizon window.")
    else:
        print("\n[FAIL] RESULT: Differences found between single 32h and MPC!")
        print("  Mismatches:")
        mismatches = comp_df[comp_df['match'] == 'FAIL']
        for _, row in mismatches.iterrows():
            print(f"    - {row['metric']}: {row['difference']}")

    print(f"\nOutput directory: {output_dir}")
    print("  - single_32h_solution.csv")
    print("  - comparison_table.csv")
    print("  - single_32h_*.html (4 visualization plots)")

    # Open key plots
    print("\nOpening visualizations...")
    import subprocess
    subprocess.run(['start', f'{output_dir}\\single_32h_soc_power.html'], shell=True)
    subprocess.run(['start', f'{output_dir}\\single_32h_capacity_markets.html'], shell=True)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return solution_df, comp_df

if __name__ == '__main__':
    solution_df, comp_df = main()
