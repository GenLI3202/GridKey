"""
MPC 5-Day Test with Comprehensive Visualizations
=================================================
- Horizon: 32 hours
- Execution: 24 hours
- Duration: 5 days
- Outputs: Decision variables CSV + Interactive HTML plots
"""

from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
from py_script.visualization.optimization_analysis import (
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid,
    plot_soc_and_power_bids,
    extract_detailed_solution
)
import pandas as pd
import json
import time
from datetime import datetime
import os

def main():
    print("=" * 80)
    print("MPC 5-DAY TEST WITH VISUALIZATIONS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    config = {
        'country': 'CH',
        'num_days': 5,
        'horizon_hours': 32,      # From mpc_config.json
        'execution_hours': 24,     # From mpc_config.json
        'alpha': 1.0,
        'c_rate': 0.5,
        'initial_soc_fraction': 0.5
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create output directory
    output_dir = 'results/mpc_5day_test'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Initialize optimizer
    print("[1/5] Initializing Model III optimizer...")
    optimizer = BESSOptimizerModelIII(alpha=config['alpha'])

    # Load data
    print(f"[2/5] Loading Phase 2 data ({config['country']})...")
    from py_script.data.load_process_market_data import load_preprocessed_country_data
    country_data = load_preprocessed_country_data(config['country'])

    # Take 5 days
    num_steps = config['num_days'] * 96  # 96 timesteps per day
    country_data = country_data.iloc[:num_steps].reset_index(drop=True)

    print(f"  Data loaded: {len(country_data)} timesteps ({len(country_data)/96:.1f} days)")
    print(f"  Date range: {country_data['timestamp'].iloc[0]} to {country_data['timestamp'].iloc[-1]}")
    print()

    # Run MPC simulation
    print("[3/5] Running MPC simulation...")
    simulator = MPCSimulator(
        optimizer_model=optimizer,
        full_data=country_data,
        horizon_hours=config['horizon_hours'],
        execution_hours=config['execution_hours'],
        c_rate=config['c_rate'],
        validate_constraints=False
    )

    start_time = time.time()
    results = simulator.run_full_simulation(
        initial_soc_fraction=config['initial_soc_fraction']
    )
    total_time = time.time() - start_time

    print(f"\n  Simulation completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print()

    # Extract and save decision variables
    print("[4/5] Extracting decision variables...")
    annual_bids_df = results.get('annual_bids_df')

    if annual_bids_df is not None:
        # Add timestamp column from country_data
        annual_bids_df['timestamp'] = country_data['timestamp'].iloc[:len(annual_bids_df)].values

        # Add market prices for visualization
        annual_bids_df['price_day_ahead'] = country_data['price_day_ahead'].iloc[:len(annual_bids_df)].values
        annual_bids_df['price_fcr'] = country_data['price_fcr'].iloc[:len(annual_bids_df)].values
        annual_bids_df['price_afrr_pos'] = country_data['price_afrr_pos'].iloc[:len(annual_bids_df)].values
        annual_bids_df['price_afrr_neg'] = country_data['price_afrr_neg'].iloc[:len(annual_bids_df)].values
        annual_bids_df['price_afrr_energy_pos'] = country_data['price_afrr_energy_pos'].iloc[:len(annual_bids_df)].values
        annual_bids_df['price_afrr_energy_neg'] = country_data['price_afrr_energy_neg'].iloc[:len(annual_bids_df)].values

        # NOTE: Capacity bids (c_fcr, c_afrr_pos, c_afrr_neg) are already in annual_bids_df from MPC
        # DO NOT overwrite them here!

        # Save to CSV
        csv_path = f'{output_dir}/decision_variables_5day.csv'
        annual_bids_df.to_csv(csv_path, index=False)
        print(f"  Saved decision variables: {csv_path}")
        print(f"  Shape: {annual_bids_df.shape}")
        print(f"  Columns: {list(annual_bids_df.columns)}")
    else:
        print("  ERROR: No annual_bids_df found in results!")
        return

    # Save financial summary
    financial_summary = {
        'test_id': 'MPC_5DAY_VISUALIZATION',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'runtime_seconds': total_time,
        'financial': {
            'total_revenue': results['total_revenue'],
            'da_revenue': results['da_revenue'],
            'afrr_e_revenue': results['afrr_e_revenue'],
            'as_revenue': results['as_revenue'],
            'total_degradation_cost': results['total_degradation_cost'],
            'cyclic_cost': results['cyclic_cost'],
            'calendar_cost': results['calendar_cost'],
            'net_profit': results['net_profit'],
        },
        'soc': {
            'initial_soc_kwh': results['soc_trajectory'][0],
            'final_soc_kwh': results['final_soc'],
            'soc_trajectory': results['soc_trajectory'],
        },
        'iterations': len(results['iteration_results']),
    }

    json_path = f'{output_dir}/financial_summary.json'
    with open(json_path, 'w') as f:
        json.dump(financial_summary, f, indent=2)
    print(f"  Saved financial summary: {json_path}")
    print()

    # Create visualizations
    print("[5/5] Creating visualizations...")

    # Prepare data for visualization functions
    # These functions expect specific column names
    viz_df = annual_bids_df.copy()

    # Add required columns for visualization
    viz_df['hour'] = viz_df['timestep'] * 0.25
    viz_df['soc_kwh'] = viz_df['e_soc']
    viz_df['soc_pct'] = viz_df['e_soc'] / 4472 * 100
    viz_df['p_ch_kw'] = viz_df['p_ch'] * 1000  # Convert MW to kW
    viz_df['p_dis_kw'] = viz_df['p_dis'] * 1000
    viz_df['p_afrr_pos_e_kw'] = viz_df['p_afrr_pos_e'] * 1000
    viz_df['p_afrr_neg_e_kw'] = viz_df['p_afrr_neg_e'] * 1000
    viz_df['c_fcr_mw'] = viz_df['c_fcr']
    viz_df['c_afrr_pos_mw'] = viz_df['c_afrr_pos']
    viz_df['c_afrr_neg_mw'] = viz_df['c_afrr_neg']
    viz_df['price_da_eur_mwh'] = viz_df['price_day_ahead']
    viz_df['price_fcr_eur_mw'] = viz_df['price_fcr']
    viz_df['price_afrr_cap_pos_eur_mw'] = viz_df['price_afrr_pos']
    viz_df['price_afrr_cap_neg_eur_mw'] = viz_df['price_afrr_neg']
    viz_df['price_afrr_energy_pos_eur_mwh'] = viz_df['price_afrr_energy_pos']
    viz_df['price_afrr_energy_neg_eur_mwh'] = viz_df['price_afrr_energy_neg']

    # Generate plots
    plots = []

    print("  [5.1] Day-Ahead Market plot...")
    fig1 = plot_da_market_price_bid(viz_df, title_suffix=f"(5 days, {config['horizon_hours']}h/{config['execution_hours']}h)", use_timestamp=True)
    plot1_path = f'{output_dir}/da_market_price_bid.html'
    fig1.write_html(plot1_path)
    plots.append(('Day-Ahead Market', plot1_path))

    print("  [5.2] aFRR Energy Market plot...")
    fig2 = plot_afrr_energy_market_price_bid(viz_df, title_suffix=f"(5 days, {config['horizon_hours']}h/{config['execution_hours']}h)", use_timestamp=True)
    plot2_path = f'{output_dir}/afrr_energy_market_price_bid.html'
    fig2.write_html(plot2_path)
    plots.append(('aFRR Energy Market', plot2_path))

    print("  [5.3] Capacity Markets plot...")
    fig3 = plot_capacity_markets_price_bid(viz_df, title_suffix=f"(5 days, {config['horizon_hours']}h/{config['execution_hours']}h)", use_timestamp=True)
    plot3_path = f'{output_dir}/capacity_markets_price_bid.html'
    fig3.write_html(plot3_path)
    plots.append(('Capacity Markets', plot3_path))

    print("  [5.4] SOC and Power Bids plot...")
    fig4 = plot_soc_and_power_bids(viz_df, title_suffix=f"(5 days, {config['horizon_hours']}h/{config['execution_hours']}h)", use_timestamp=True)
    plot4_path = f'{output_dir}/soc_and_power_bids.html'
    fig4.write_html(plot4_path)
    plots.append(('SOC and Power Bids', plot4_path))

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    print("\nFinancial Summary:")
    print(f"  Total Revenue:      {results['total_revenue']:.2f} EUR")
    print(f"    - DA Energy:      {results['da_revenue']:.2f} EUR")
    print(f"    - aFRR Energy:    {results['afrr_e_revenue']:.2f} EUR")
    print(f"    - AS Capacity:    {results['as_revenue']:.2f} EUR")
    print(f"  Total Degradation:  {results['total_degradation_cost']:.2f} EUR")
    print(f"    - Cyclic:         {results['cyclic_cost']:.2f} EUR")
    print(f"    - Calendar:       {results['calendar_cost']:.2f} EUR")
    print(f"  Net Profit:         {results['net_profit']:.2f} EUR")

    print("\nSOC Trajectory:")
    print(f"  Initial SOC:        {results['soc_trajectory'][0]:.2f} kWh ({100*results['soc_trajectory'][0]/4472:.1f}%)")
    print(f"  Final SOC:          {results['final_soc']:.2f} kWh ({100*results['final_soc']/4472:.1f}%)")
    print(f"  Iterations:         {len(results['iteration_results'])}")

    print("\nOutput Files:")
    print(f"  Decision Variables: {csv_path}")
    print(f"  Financial Summary:  {json_path}")
    print("\nVisualization Plots:")
    for name, path in plots:
        print(f"  {name}: {path}")

    print(f"\nRuntime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results

if __name__ == '__main__':
    results = main()
