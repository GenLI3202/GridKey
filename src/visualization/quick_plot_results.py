#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick visualization of 36h HU winter optimization results.
Displays DA market and power bid data for inspection.
"""

import pandas as pd
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add py_script to path
sys.path.append(str(Path(__file__).parent / 'py_script'))

from src.visualization.optimization_analysis import (
    plot_da_market_price_bid,
    plot_soc_and_power_bids
)

def main():
    """Load and plot optimization results."""

    # Load solution data
    csv_file = Path("results/model_iii_validation/solution_36h_hu_winter.csv")

    if not csv_file.exists():
        print(f"ERROR: Solution file not found: {csv_file}")
        return

    print("Loading solution data...")
    df = pd.read_csv(csv_file)

    print(f"Loaded {len(df)} time steps")
    print(f"\nData summary:")
    print(f"  Time range: {df['hour'].min():.1f}h - {df['hour'].max():.1f}h")
    print(f"  SOC range: {df['soc_kwh'].min():.1f} - {df['soc_kwh'].max():.1f} kWh")
    print(f"  DA Price range: {df['price_da_eur_mwh'].min():.2f} - {df['price_da_eur_mwh'].max():.2f} EUR/MWh")
    print(f"  Total DA discharge: {df['p_dis_kw'].sum()/1000:.2f} MWh")
    print(f"  Total DA charge: {df['p_ch_kw'].sum()/1000:.2f} MWh")

    # Create output directory
    output_dir = Path("results/model_iii_validation/quick_check_plots")
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*80)
    print("Generating Plots...")
    print("="*80)

    # Plot 1: Day-Ahead Market
    print("\n[1/2] Day-Ahead Market: Price vs Bids...")
    try:
        fig_da = plot_da_market_price_bid(df, title_suffix="(HU Winter 36h)", use_timestamp=False)
        html_file = output_dir / "da_market_check.html"
        fig_da.write_html(str(html_file))
        print(f"      ✓ Saved: {html_file}")

        # Also open in browser
        import webbrowser
        webbrowser.open(str(html_file.absolute()))
        print(f"      ✓ Opened in browser")

    except Exception as e:
        print(f"      ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Plot 2: SOC and Power Bids
    print("\n[2/2] Battery Schedule: SOC & All Power Bids...")
    try:
        fig_soc = plot_soc_and_power_bids(df, title_suffix="(HU Winter 36h)", use_timestamp=False)
        html_file = output_dir / "soc_power_check.html"
        fig_soc.write_html(str(html_file))
        print(f"      ✓ Saved: {html_file}")

        # Also open in browser
        import webbrowser
        webbrowser.open(str(html_file.absolute()))
        print(f"      ✓ Opened in browser")

    except Exception as e:
        print(f"      ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")
    print("\nBoth plots should now be open in your browser.")
    print("="*80)

    # Print some key observations
    print("\n📊 Key Observations:")
    print(f"  • DA market participation: {(df['p_dis_kw'] > 0).sum()} discharge periods, {(df['p_ch_kw'] > 0).sum()} charge periods")
    print(f"  • aFRR energy participation: {(df['p_afrr_pos_e_kw'] > 0).sum()} positive, {(df['p_afrr_neg_e_kw'] > 0).sum()} negative")
    print(f"  • FCR capacity active: {(df['c_fcr_mw'] > 0).sum()}/{len(df)} time steps")
    print(f"  • Max power: {max(df['p_total_dis_kw'].max(), df['p_total_ch_kw'].max()):.2f} kW")

    # Check for any unusual patterns
    if df['p_ch_kw'].sum() == 0 and df['p_dis_kw'].sum() > 0:
        print("\n⚠️  Note: Battery only discharges, never charges in DA market")
        print("    This might indicate starting SOC was sufficient for the entire period")

if __name__ == "__main__":
    main()
