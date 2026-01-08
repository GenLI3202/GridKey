"""
Convert MPC submission results to Phase 2 submission format

This script converts existing MPC simulation results to the required
Excel format with multiple sheets for submission.

Output Files:
1. TechArena_Phase2_Operation.xlsx - 15 sheets (5 countries × 3 C-rates)
2. TechArena_Phase2_Configuration.xlsx - 5 sheets (one per country)
3. TechArena_Phase2_Investment.xlsx - 5 sheets (one per country)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


# Country-specific financial parameters
WACC = {
    'DE_LU': 0.083,
    'AT': 0.083,
    'CH': 0.083,
    'CZ': 0.12,
    'HU': 0.15
}

INFLATION = {
    'DE_LU': 0.02,
    'AT': 0.033,
    'CH': 0.001,
    'CZ': 0.029,
    'HU': 0.046
}

# Battery parameters
BATTERY_CAPACITY_KWH = 4472  # kWh
INVESTMENT_COST_PER_KWH = 200  # EUR/kWh
ANNUAL_DEGRADATION_RATE = 0.03  # 3% per year (100% → 70% over 10 years)
PROJECT_YEARS = 10


def convert_operation_file(solution_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert solution_timeseries.csv to TechArena_Phase2_Operation format.

    Args:
        solution_df: DataFrame from solution_timeseries.csv

    Returns:
        DataFrame in submission format with required columns
    """
    operation_df = pd.DataFrame({
        'Timestamp': solution_df['timestamp'],
        'Stored energy [MWh]': solution_df['soc_kwh'] / 1000,
        'SoC [-]': solution_df['soc_pct'] / 100,  # Convert 0-100% to 0-1
        'Charge [MWh]': solution_df['p_ch_kw'] / 1000 * 0.25,  # 15min interval
        'Discharge [MWh]': solution_df['p_dis_kw'] / 1000 * 0.25,

        # Separate DA from aFRR energy
        'Day-ahead buy [MWh]': (solution_df['p_ch_kw'] - solution_df['p_afrr_neg_e_kw']).clip(lower=0) / 1000 * 0.25,
        'Day-ahead sell [MWh]': (solution_df['p_dis_kw'] - solution_df['p_afrr_pos_e_kw']).clip(lower=0) / 1000 * 0.25,

        'FCR Capacity [MW]': solution_df['c_fcr_mw'],
        'aFRR Capacity POS [MW]': solution_df['c_afrr_pos_mw'],
        'aFRR Capacity NEG [MW]': solution_df['c_afrr_neg_mw'],

        # Phase 2 Addition: aFRR Energy bids
        'aFRR-E POS [MWh]': solution_df['p_afrr_pos_e_kw'] / 1000 * 0.25,
        'aFRR-E NEG [MWh]': solution_df['p_afrr_neg_e_kw'] / 1000 * 0.25
    })

    return operation_df


def calculate_10year_roi(annual_profit: float, country: str) -> Tuple[float, pd.DataFrame]:
    """
    Calculate 10-year NPV and ROI with capacity degradation.

    Args:
        annual_profit: First year profit in EUR
        country: Country code for WACC and inflation

    Returns:
        Tuple of (ROI percentage, year-by-year DataFrame)
    """
    wacc = WACC[country]
    infl = INFLATION[country]
    total_investment = BATTERY_CAPACITY_KWH * INVESTMENT_COST_PER_KWH

    # Year-by-year calculation
    years_data = []
    npv = -total_investment

    for year in range(2024, 2024 + PROJECT_YEARS):
        year_num = year - 2023

        # Capacity degrades linearly: 100% → 70% over 10 years
        capacity_factor = 1 - ANNUAL_DEGRADATION_RATE * (year_num - 1)
        capacity_factor = max(0.7, capacity_factor)  # Floor at 70%

        # Profit scales with capacity
        yearly_profit = annual_profit * capacity_factor

        # Apply inflation
        yearly_profit_inflated = yearly_profit * ((1 + infl) ** (year_num - 1))

        # Discount to present value
        discount_factor = (1 + wacc) ** year_num
        pv_profit = yearly_profit_inflated / discount_factor

        npv += pv_profit

        years_data.append({
            'Year': year,
            'Capacity [%]': capacity_factor * 100,
            'Yearly Profit [kEUR]': yearly_profit / 1000,
            'Inflated Profit [kEUR]': yearly_profit_inflated / 1000,
            'PV Contribution [kEUR]': pv_profit / 1000
        })

    roi_percent = (npv / total_investment) * 100

    years_df = pd.DataFrame(years_data)
    return roi_percent, years_df


def load_result_from_directory(result_dir: Path) -> Dict:
    """
    Load MPC result from a directory.

    Args:
        result_dir: Path to result directory

    Returns:
        Dictionary with result data
    """
    # Load performance summary
    perf_path = result_dir / 'performance_summary.json'
    with open(perf_path, 'r') as f:
        perf = json.load(f)

    # Load solution timeseries
    sol_path = result_dir / 'solution_timeseries.csv'
    solution_df = pd.read_csv(sol_path)
    solution_df['timestamp'] = pd.to_datetime(solution_df['timestamp'])

    # Extract scenario info from directory name or performance file
    country = perf.get('country', 'UNKNOWN')
    c_rate = perf.get('c_rate', 0.0)

    return {
        'country': country,
        'c_rate': c_rate,
        'performance': perf,
        'solution_df': solution_df,
        'result_dir': result_dir
    }


def generate_operation_excel(results: List[Dict], output_path: Path):
    """
    Generate TechArena_Phase2_Operation.xlsx with 15 sheets.

    Args:
        results: List of result dictionaries
        output_path: Path to save Excel file
    """
    print(f"\n[GENERATING] Operation Excel file...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for result in results:
            country = result['country']
            c_rate = result['c_rate']
            sheet_name = f"{country}_{c_rate}"

            # Convert to operation format
            operation_df = convert_operation_file(result['solution_df'])

            # Write to sheet
            operation_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  [OK] Sheet: {sheet_name} ({len(operation_df)} rows)")

    print(f"[SAVED] {output_path}")


def generate_configuration_excel(results: List[Dict], output_path: Path):
    """
    Generate TechArena_Phase2_Configuration.xlsx with 5 sheets.

    Args:
        results: List of result dictionaries
        output_path: Path to save Excel file
    """
    print(f"\n[GENERATING] Configuration Excel file...")

    # Group by country
    country_results = {}
    for result in results:
        country = result['country']
        if country not in country_results:
            country_results[country] = []
        country_results[country].append(result)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for country, country_res in sorted(country_results.items()):
            config_data = []

            for result in sorted(country_res, key=lambda x: x['c_rate']):
                c_rate = result['c_rate']
                perf = result['performance']

                # Calculate metrics
                battery_power_mw = BATTERY_CAPACITY_KWH * c_rate / 1000
                annual_profit = perf['total_profit_eur']
                yearly_profit_per_mw = (annual_profit / battery_power_mw) / 1000  # kEUR/MW

                # Calculate ROI
                roi_percent, _ = calculate_10year_roi(annual_profit, country)

                # Estimate number of cycles from degradation
                cyclic_cost = perf.get('degradation_cyclic_eur', 0)
                # Rough estimate: assuming linear cost per cycle
                num_cycles = cyclic_cost / 100 if cyclic_cost > 0 else 0  # Placeholder

                config_data.append({
                    'C-rate': c_rate,
                    'number of cycles': round(num_cycles, 2),
                    'yearly profits [kEUR/MW]': round(yearly_profit_per_mw, 2),
                    'levelized ROI [%]': round(roi_percent, 2)
                })

            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name=country, index=False)
            print(f"  [OK] Sheet: {country} ({len(config_df)} configurations)")

    print(f"[SAVED] {output_path}")


def generate_investment_excel(results: List[Dict], output_path: Path):
    """
    Generate TechArena_Phase2_Investment.xlsx with 5 sheets.

    Args:
        results: List of result dictionaries
        output_path: Path to save Excel file
    """
    print(f"\n[GENERATING] Investment Excel file...")

    # Group by country
    country_results = {}
    for result in results:
        country = result['country']
        if country not in country_results:
            country_results[country] = []
        country_results[country].append(result)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for country, country_res in sorted(country_results.items()):
            # Use best C-rate result for this country
            best_result = max(country_res, key=lambda x: x['performance']['total_profit_eur'])
            annual_profit = best_result['performance']['total_profit_eur']
            c_rate = best_result['c_rate']

            # Calculate 10-year ROI
            roi_percent, years_df = calculate_10year_roi(annual_profit, country)

            # Create investment sheet
            investment_data = []

            # Header section
            investment_data.append(['WACC', WACC[country]])
            investment_data.append(['Inflation Rate', INFLATION[country]])
            investment_data.append(['Discount Rate', WACC[country]])  # WACC = discount rate
            investment_data.append(['Yearly Profits (2024) [kEUR]', annual_profit / 1000])
            investment_data.append(['Best C-rate', c_rate])
            investment_data.append([])  # Empty row

            # Year-by-year table
            investment_data.append(['Year', 'Capacity [%]', 'Yearly Profit [kEUR]',
                                   'Inflated Profit [kEUR]', 'PV Contribution [kEUR]'])
            for _, row in years_df.iterrows():
                investment_data.append([
                    row['Year'],
                    row['Capacity [%]'],
                    row['Yearly Profit [kEUR]'],
                    row['Inflated Profit [kEUR]'],
                    row['PV Contribution [kEUR]']
                ])

            investment_data.append([])  # Empty row
            investment_data.append(['Levelized ROI [%]', round(roi_percent, 2)])
            investment_data.append(['Total Investment [kEUR]', BATTERY_CAPACITY_KWH * INVESTMENT_COST_PER_KWH / 1000])

            # Convert to DataFrame and write
            inv_df = pd.DataFrame(investment_data)
            inv_df.to_excel(writer, sheet_name=country, index=False, header=False)
            print(f"  [OK] Sheet: {country} (ROI: {roi_percent:.2f}%)")

    print(f"[SAVED] {output_path}")


def main(submission_results_dir: Path = None, output_dir: Path = None):
    """
    Main conversion function.

    Args:
        submission_results_dir: Path to submission_results from main repo
        output_dir: Path to save output Excel files
    """
    if submission_results_dir is None:
        # Default to main repo's submission_results
        submission_results_dir = project_root.parent / 'submission_results'

    if output_dir is None:
        output_dir = project_root / 'output'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SUBMISSION RESULTS CONVERSION")
    print("="*80)
    print(f"Source: {submission_results_dir}")
    print(f"Output: {output_dir}")
    print("="*80)

    # Load all result directories
    print("\n[LOADING] Scanning for result directories...")
    result_dirs = [d for d in submission_results_dir.iterdir() if d.is_dir()]
    print(f"Found {len(result_dirs)} result directories")

    # Load results
    results = []
    for result_dir in sorted(result_dirs):
        try:
            result = load_result_from_directory(result_dir)
            results.append(result)
            print(f"  [OK] Loaded: {result['country']} C-rate {result['c_rate']}")
        except Exception as e:
            print(f"  [FAILED] {result_dir.name} - {e}")

    print(f"\n[SUCCESS] Loaded {len(results)} results")

    # Generate Excel files
    generate_operation_excel(results, output_dir / 'TechArena_Phase2_Operation.xlsx')
    generate_configuration_excel(results, output_dir / 'TechArena_Phase2_Configuration.xlsx')
    generate_investment_excel(results, output_dir / 'TechArena_Phase2_Investment.xlsx')

    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"\nGenerated files in {output_dir}:")
    print("  1. TechArena_Phase2_Operation.xlsx")
    print("  2. TechArena_Phase2_Configuration.xlsx")
    print("  3. TechArena_Phase2_Investment.xlsx")


if __name__ == "__main__":
    main()
