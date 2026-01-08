#!/usr/bin/env python3
"""
36-hour optimization for Hungary (HU) in winter with full visualization.

Runs Model III optimization for 36 hours using HU winter market data,
saves detailed solution, and generates all four visualization plots.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(str(Path(__file__).parent / 'py_script'))

from src.core.optimizer import BESSOptimizerModelIII
from src.visualization.optimization_analysis import (
    plot_da_market_price_bid,
    plot_afrr_energy_market_price_bid,
    plot_capacity_markets_price_bid,
    plot_soc_and_power_bids
)

def main():
    """Run 36h HU winter optimization and generate visualizations."""

    print("="*80)
    print("36-Hour Hungary Winter Optimization - Model III")
    print("="*80)

    # ========================================================================
    # Configuration
    # ========================================================================

    country = 'HU'
    horizon_hours = 36
    horizon_steps = horizon_hours * 4  # 15-min intervals

    # Use Phase 2 preprocessed data (fast path)
    from src.data.load_process_market_data import load_preprocessed_country_data

    # Winter period: Use January data (week 1-2)
    start_step = 0
    end_step = start_step + horizon_steps

    print(f"\nConfiguration:")
    print(f"  Country: {country}")
    print(f"  Horizon: {horizon_hours} hours ({horizon_steps} steps)")
    print(f"  Time steps: {start_step} to {end_step}")
    print(f"  Season: Winter (January)")
    print(f"  Alpha: 0.5 (degradation weight)")

    # ========================================================================
    # Initialize Optimizer & Load Data
    # ========================================================================

    print("\n" + "="*80)
    print("Initializing Optimizer & Loading Data...")
    print("="*80)

    # Initialize Model III optimizer with lower alpha
    optimizer = BESSOptimizerModelIII(alpha=0.5, use_afrr_ev_weighting=True)

    # Configuration
    optimizer.max_as_ratio = 0.8  # Max 80% total AS capacity

    print(f"[OK] BESS Configuration:")
    print(f"  Capacity:     {optimizer.battery_params['capacity_kwh']} kWh")
    print(f"  Efficiency:   {optimizer.battery_params['efficiency']}")
    print(f"  Initial SOC:  {optimizer.battery_params['initial_soc']*100:.0f}%")
    print(f"  Max AS Ratio: {optimizer.max_as_ratio*100:.0f}%")

    # Load data
    try:
        print(f"\n[OK] Loading preprocessed data for: {country}")
        country_data = load_preprocessed_country_data(country)

        # Extract winter period (36 hours)
        data_slice = country_data.iloc[start_step:end_step].copy()
        data_slice.reset_index(drop=True, inplace=True)

        print(f"[OK] Extracted {len(data_slice)} time steps for optimization")

        # Show data summary
        print(f"\nMarket Data Summary:")
        print(f"  DA Price:    {data_slice['price_day_ahead'].min():.2f} - {data_slice['price_day_ahead'].max():.2f} EUR/MWh")
        print(f"  FCR Price:   {data_slice['price_fcr'].min():.2f} - {data_slice['price_fcr'].max():.2f} EUR/MW")
        print(f"  aFRR+ Price: {data_slice['price_afrr_pos'].min():.2f} - {data_slice['price_afrr_pos'].max():.2f} EUR/MW")

    except Exception as e:
        print(f"[FAIL] Error loading market data: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # Run Optimization
    # ========================================================================

    print("\n" + "="*80)
    print("Building and Solving Model...")
    print("="*80)

    import time as time_module

    try:
        # Build model
        build_start = time_module.time()
        model = optimizer.build_optimization_model(data_slice, c_rate=0.5)
        build_time = time_module.time() - build_start

        print(f"[OK] Model built in {build_time:.2f}s")
        print(f"  Variables:   {model.nvariables()}")
        print(f"  Constraints: {model.nconstraints()}")

        # Solve model
        print(f"\n[OK] Solving optimization problem...")
        solve_start = time_module.time()
        solved_model, solver_results = optimizer.solve_model(model)
        solution = optimizer.extract_solution(solved_model, solver_results)
        solve_time = time_module.time() - solve_start

        if solution['status'] == 'optimal':
            print(f"[OK] Optimization successful!")
            print(f"  Status:          {solution['status']}")
            print(f"  Objective Value: {solution['objective_value']:.2f} EUR")
            print(f"  Build Time:      {build_time:.2f}s")
            print(f"  Solve Time:      {solve_time:.2f}s")
            print(f"  Total Time:      {build_time + solve_time:.2f}s")

            # Store timing info in solution for later saving
            solution['build_time'] = build_time
            solution['solve_time'] = solve_time
            solution['total_time'] = build_time + solve_time
        else:
            print(f"[FAIL] Optimization failed: {solution['status']}")
            return

    except Exception as e:
        print(f"[FAIL] Optimization error: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # Extract and Save Detailed Solution
    # ========================================================================

    print("\n" + "="*80)
    print("Preparing Detailed Solution...")
    print("="*80)

    # Use extraction function from visualization module
    from src.visualization.optimization_analysis import extract_detailed_solution
    df = extract_detailed_solution(solution, data_slice, horizon_hours)

    # Save to CSV
    output_dir = Path("results/model_iii_validation")
    output_dir.mkdir(exist_ok=True, parents=True)

    csv_file = output_dir / f"solution_36h_hu_winter.csv"
    df.to_csv(csv_file, index=False)
    print(f"[OK] Saved detailed solution: {csv_file}")

    # Save metadata with timing information
    metadata = {
        'country': country,
        'horizon_hours': horizon_hours,
        'horizon_steps': horizon_steps,
        'start_step': start_step,
        'end_step': end_step,
        'alpha': optimizer.degradation_params['alpha'],
        'max_as_ratio': optimizer.max_as_ratio,
        'optimization': {
            'status': solution['status'],
            'objective_value': solution['objective_value'],
            'build_time_seconds': solution['build_time'],
            'solve_time_seconds': solution['solve_time'],
            'total_time_seconds': solution['total_time'],
            'n_variables': model.nvariables(),
            'n_constraints': model.nconstraints()
        },
        'bess_config': optimizer.battery_params,
        'timestamp': datetime.now().isoformat()
    }

    metadata_file = output_dir / f"solution_36h_hu_winter_metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata: {metadata_file}")

    # Print solution summary
    print(f"\nSolution Summary:")
    print(f"  Total Revenue:      {df['revenue_da_eur'].sum() + df['revenue_afrr_energy_eur'].sum() + df['revenue_as_capacity_eur'].sum():.2f} EUR")
    print(f"    DA Revenue:       {df['revenue_da_eur'].sum():.2f} EUR")
    print(f"    aFRR-E Revenue:   {df['revenue_afrr_energy_eur'].sum():.2f} EUR")
    print(f"    AS Cap Revenue:   {df['revenue_as_capacity_eur'].sum():.2f} EUR")
    print(f"  DA Energy:")
    print(f"    Charge:           {df['p_ch_kw'].sum()/1000:.2f} MWh")
    print(f"    Discharge:        {df['p_dis_kw'].sum()/1000:.2f} MWh")
    print(f"  aFRR Energy:")
    print(f"    Positive:         {df['p_afrr_pos_e_kw'].sum()/1000:.2f} MWh")
    print(f"    Negative:         {df['p_afrr_neg_e_kw'].sum()/1000:.2f} MWh")
    print(f"  Max Capacities:")
    print(f"    FCR:              {df['c_fcr_mw'].max():.3f} MW")
    print(f"    aFRR+:            {df['c_afrr_pos_mw'].max():.3f} MW")
    print(f"    aFRR-:            {df['c_afrr_neg_mw'].max():.3f} MW")

    # ========================================================================
    # Generate Visualizations
    # ========================================================================

    print("\n" + "="*80)
    print("Generating Visualizations...")
    print("="*80)

    plot_dir = output_dir / "hu_winter_36h_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    title_suffix = f"(HU Winter, 36h)"

    # Plot 1: Day-Ahead Market
    print("\n[1/4] Generating Day-Ahead Market plot...")
    try:
        fig_da = plot_da_market_price_bid(df, title_suffix=title_suffix, use_timestamp=False)
        html_file = plot_dir / "da_market_price_bid.html"
        fig_da.write_html(str(html_file))
        print(f"      [OK] Saved: {html_file}")
    except Exception as e:
        print(f"      [FAIL] Error: {e}")

    # Plot 2: aFRR Energy Market
    print("[2/4] Generating aFRR Energy Market plot...")
    try:
        fig_afrr_e = plot_afrr_energy_market_price_bid(df, title_suffix=title_suffix, use_timestamp=False)
        html_file = plot_dir / "afrr_energy_market_price_bid.html"
        fig_afrr_e.write_html(str(html_file))
        print(f"      [OK] Saved: {html_file}")
    except Exception as e:
        print(f"      [FAIL] Error: {e}")

    # Plot 3: Capacity Markets
    print("[3/4] Generating Capacity Markets plot...")
    try:
        fig_cap = plot_capacity_markets_price_bid(df, title_suffix=title_suffix, use_timestamp=False)
        html_file = plot_dir / "capacity_markets_price_bid.html"
        fig_cap.write_html(str(html_file))
        print(f"      [OK] Saved: {html_file}")
    except Exception as e:
        print(f"      [FAIL] Error: {e}")

    # Plot 4: SOC & Power Bids
    print("[4/4] Generating SOC & Power Bids plot...")
    try:
        fig_soc = plot_soc_and_power_bids(df, title_suffix=title_suffix, use_timestamp=False)
        html_file = plot_dir / "soc_and_power_bids.html"
        fig_soc.write_html(str(html_file))
        print(f"      [OK] Saved: {html_file}")
    except Exception as e:
        print(f"      [FAIL] Error: {e}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  CSV:   {csv_file}")
    print(f"  Plots: {plot_dir}")
    print("\nGenerated plots:")
    print(f"  1. da_market_price_bid.html")
    print(f"  2. afrr_energy_market_price_bid.html")
    print(f"  3. capacity_markets_price_bid.html")
    print(f"  4. soc_and_power_bids.html")
    print("="*80)

if __name__ == "__main__":
    main()
