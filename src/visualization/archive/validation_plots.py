"""
BESS Validation Plotting Utilities
===================================

Comprehensive plotting functions for BESS model validation results.

Generates:
1. SOC trajectory plots
2. Power profile visualizations
3. Revenue breakdown charts
4. Configuration heatmaps
5. Constraint validation dashboards

Author: BESS Optimization Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ValidationPlotter:
    """
    Plotting utilities for BESS validation results.
    """

    def __init__(self, output_dir: str = "validation_week_results"):
        """Initialize plotter with output directory."""
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.plot_dir / "soc_trajectories").mkdir(exist_ok=True)
        (self.plot_dir / "power_profiles").mkdir(exist_ok=True)
        (self.plot_dir / "revenue_breakdown").mkdir(exist_ok=True)
        (self.plot_dir / "configuration_heatmaps").mkdir(exist_ok=True)

        logger.info(f"Validation plotter initialized. Output: {self.plot_dir}")

    def plot_soc_trajectory(self, scenario_name: str, detailed_result: Dict,
                           save: bool = True) -> Optional[plt.Figure]:
        """
        Plot SOC trajectory for a single scenario.

        Args:
            scenario_name: Name of the scenario
            detailed_result: Detailed results dict with solution data
            save: Whether to save the plot

        Returns:
            matplotlib Figure object
        """
        solution = detailed_result['solution']
        e_soc = solution.get('e_soc', {})

        if not e_soc:
            logger.warning(f"No SOC data for scenario {scenario_name}")
            return None

        # Get battery capacity from correctness validation
        correctness = detailed_result.get('correctness_validation', {})
        E_nom = 4472  # Default, can extract from model if available

        # Extract SOC values and convert to percentage
        times = sorted(e_soc.keys())
        soc_values = [e_soc[t] / E_nom * 100 for t in times]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot SOC trajectory
        ax.plot(times, soc_values, linewidth=2, label='SOC (%)', color='steelblue')
        ax.fill_between(times, 0, soc_values, alpha=0.3, color='steelblue')

        # Add horizontal lines for limits
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Min SOC')
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Max SOC')

        # Formatting
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('State of Charge (%)', fontsize=12)
        ax.set_title(f'SOC Trajectory - {scenario_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Add statistics text box
        stats_text = f'Min: {min(soc_values):.1f}%\nMax: {max(soc_values):.1f}%\nMean: {np.mean(soc_values):.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            filename = self.plot_dir / "soc_trajectories" / f"{scenario_name}_soc.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved SOC trajectory plot: {filename}")

        return fig

    def plot_power_profile(self, scenario_name: str, detailed_result: Dict,
                          save: bool = True) -> Optional[plt.Figure]:
        """
        Plot power profile (charge/discharge) for a single scenario.

        Args:
            scenario_name: Name of the scenario
            detailed_result: Detailed results dict with solution data
            save: Whether to save the plot

        Returns:
            matplotlib Figure object
        """
        solution = detailed_result['solution']
        p_ch = solution.get('p_ch', {})
        p_dis = solution.get('p_dis', {})

        if not p_ch and not p_dis:
            logger.warning(f"No power data for scenario {scenario_name}")
            return None

        # Extract power values (convert kW to MW)
        times = sorted(set(list(p_ch.keys()) + list(p_dis.keys())))
        charge_values = [p_ch.get(t, 0) / 1000 for t in times]  # Convert to MW
        discharge_values = [-p_dis.get(t, 0) / 1000 for t in times]  # Negative for discharge, convert to MW

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot power profiles
        ax.bar(times, charge_values, width=0.8, label='Charging', color='green', alpha=0.7)
        ax.bar(times, discharge_values, width=0.8, label='Discharging', color='orange', alpha=0.7)

        # Formatting
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Power (MW)', fontsize=12)
        ax.set_title(f'Power Profile - {scenario_name}', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best')

        # Add statistics text box
        total_charge = sum([v for v in charge_values if v > 0])
        total_discharge = sum([abs(v) for v in discharge_values if v < 0])
        stats_text = f'Total Charge: {total_charge:.1f} MWh\nTotal Discharge: {total_discharge:.1f} MWh'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            filename = self.plot_dir / "power_profiles" / f"{scenario_name}_power.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved power profile plot: {filename}")

        return fig

    def plot_revenue_breakdown(self, scenario_name: str, detailed_result: Dict,
                              save: bool = True) -> Optional[plt.Figure]:
        """
        Plot revenue breakdown (DA vs AS markets) for a single scenario.

        Args:
            scenario_name: Name of the scenario
            detailed_result: Detailed results dict
            save: Whether to save the plot

        Returns:
            matplotlib Figure object
        """
        correctness = detailed_result.get('correctness_validation', {})
        obj_val = correctness.get('objective_validation', {})

        da_revenue = obj_val.get('da_profit', 0)
        fcr_revenue = obj_val.get('fcr_revenue', 0)
        afrr_pos_revenue = obj_val.get('afrr_pos_revenue', 0)
        afrr_neg_revenue = obj_val.get('afrr_neg_revenue', 0)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: DA vs AS
        revenue_types = ['Day-Ahead', 'Ancillary Services']
        revenues = [da_revenue, fcr_revenue + afrr_pos_revenue + afrr_neg_revenue]
        colors = ['steelblue', 'coral']

        ax1.bar(revenue_types, revenues, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Revenue (€)', fontsize=12)
        ax1.set_title('Revenue: DA vs AS', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(revenues):
            ax1.text(i, v, f'€{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Subplot 2: Detailed AS breakdown
        as_types = ['FCR', 'aFRR+', 'aFRR-']
        as_revenues = [fcr_revenue, afrr_pos_revenue, afrr_neg_revenue]
        as_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        # Filter out zero revenues for cleaner pie chart
        filtered_types = [t for t, r in zip(as_types, as_revenues) if r > 0.01]
        filtered_revenues = [r for r in as_revenues if r > 0.01]

        if filtered_revenues:
            ax2.pie(filtered_revenues, labels=filtered_types, autopct='%1.1f%%',
                   colors=as_colors, startangle=90, textprops={'fontsize': 10})
            ax2.set_title('AS Revenue Breakdown', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No AS Revenue', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('AS Revenue Breakdown', fontsize=12, fontweight='bold')

        # Overall title
        total_revenue = da_revenue + fcr_revenue + afrr_pos_revenue + afrr_neg_revenue
        fig.suptitle(f'Revenue Breakdown - {scenario_name}\nTotal: €{total_revenue:.2f}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            filename = self.plot_dir / "revenue_breakdown" / f"{scenario_name}_revenue.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved revenue breakdown plot: {filename}")

        return fig

    def plot_configuration_heatmap(self, results_df: pd.DataFrame, country: str,
                                   metric: str = 'total_revenue',
                                   save: bool = True) -> Optional[plt.Figure]:
        """
        Plot heatmap of configuration performance for a specific country.

        Args:
            results_df: DataFrame with all scenario results
            country: Country to plot
            metric: Metric to visualize (default: total_revenue)
            save: Whether to save the plot

        Returns:
            matplotlib Figure object
        """
        # Filter for country
        country_data = results_df[results_df['country'] == country]

        if country_data.empty:
            logger.warning(f"No data for country {country}")
            return None

        # Pivot for heatmap (C-rate vs Daily Cycles)
        heatmap_data = country_data.pivot(index='n_cycles', columns='c_rate', values=metric)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu',
                   cbar_kws={'label': metric.replace('_', ' ').title()},
                   ax=ax, linewidths=1, linecolor='white')

        ax.set_xlabel('C-Rate', fontsize=12)
        ax.set_ylabel('Daily Cycles', fontsize=12)
        ax.set_title(f'Configuration Performance - {country}\n{metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')

        # Highlight best configuration
        best_idx = country_data[metric].idxmax()
        best_row = country_data.loc[best_idx]
        best_text = f'Best: C={best_row["c_rate"]}, N={best_row["n_cycles"]}, {metric}={best_row[metric]:.2f}'
        ax.text(0.5, -0.15, best_text, transform=ax.transAxes,
               ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()

        if save:
            filename = self.plot_dir / "configuration_heatmaps" / f"{country}_{metric}_heatmap.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved configuration heatmap: {filename}")

        return fig

    def plot_all_countries_comparison(self, results_df: pd.DataFrame,
                                      metric: str = 'total_revenue',
                                      save: bool = True) -> Optional[plt.Figure]:
        """
        Plot comparison of all countries' best configurations.

        Args:
            results_df: DataFrame with all scenario results
            metric: Metric to compare
            save: Whether to save the plot

        Returns:
            matplotlib Figure object
        """
        # Get best configuration for each country
        best_configs = []
        for country in results_df['country'].unique():
            country_data = results_df[results_df['country'] == country]
            best_idx = country_data[metric].idxmax()
            best_row = country_data.loc[best_idx]
            best_configs.append({
                'country': country,
                'best_value': best_row[metric],
                'c_rate': best_row['c_rate'],
                'n_cycles': best_row['n_cycles']
            })

        best_df = pd.DataFrame(best_configs).sort_values('best_value', ascending=False)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bar chart
        bars = ax.bar(best_df['country'], best_df['best_value'],
                     color=plt.cm.viridis(np.linspace(0.3, 0.9, len(best_df))),
                     alpha=0.8, edgecolor='black')

        # Add configuration labels on bars
        for i, (idx, row) in enumerate(best_df.iterrows()):
            ax.text(i, row['best_value'], f"C={row['c_rate']}\nN={row['n_cycles']}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Country', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Best Configuration per Country\n{metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            filename = self.plot_dir / f"all_countries_{metric}_comparison.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved country comparison plot: {filename}")

        return fig

    def plot_constraint_validation_dashboard(self, results_df: pd.DataFrame,
                                            save: bool = True) -> Optional[plt.Figure]:
        """
        Plot dashboard showing constraint validation results.

        Args:
            results_df: DataFrame with all scenario results
            save: Whether to save the plot

        Returns:
            matplotlib Figure object
        """
        # Extract constraint pass rates
        constraint_cols = [col for col in results_df.columns if col.endswith('_pass') and col != 'all_constraints_pass']

        if not constraint_cols:
            logger.warning("No constraint validation columns found")
            return None

        # Calculate pass rates
        pass_rates = {}
        for col in constraint_cols:
            constraint_name = col.replace('_pass', '').replace('Cst', 'Constraint ')
            pass_rates[constraint_name] = results_df[col].sum() / len(results_df) * 100

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Pass rates bar chart
        constraints = list(pass_rates.keys())
        rates = list(pass_rates.values())
        colors = ['green' if r == 100 else 'orange' if r >= 90 else 'red' for r in rates]

        ax1.barh(constraints, rates, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Pass Rate (%)', fontsize=12)
        ax1.set_title('Constraint Validation Pass Rates', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 105)
        ax1.axvline(x=100, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(rates):
            ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

        # Subplot 2: Overall status pie chart
        status_counts = results_df['status'].value_counts()
        colors_status = {'PASS': 'green', 'FAIL': 'red', 'ERROR': 'gray'}
        colors_pie = [colors_status.get(status, 'blue') for status in status_counts.index]

        ax2.pie(status_counts.values, labels=status_counts.index,
               autopct='%1.1f%%', colors=colors_pie, startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Overall Validation Status', fontsize=12, fontweight='bold')

        # Overall title
        total_pass = (results_df['status'] == 'PASS').sum()
        fig.suptitle(f'Constraint Validation Dashboard\n{total_pass}/{len(results_df)} Scenarios Passed All Constraints',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            filename = self.plot_dir / "validation_summary_dashboard.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved validation dashboard: {filename}")

        return fig

    def generate_all_plots(self, results_csv: str, detailed_results_dir: str):
        """
        Generate all validation plots from results.

        Args:
            results_csv: Path to validation_summary.csv
            detailed_results_dir: Path to directory with detailed JSON results
        """
        logger.info("Generating all validation plots...")

        # Load summary results
        results_df = pd.read_csv(results_csv)

        # 1. Constraint validation dashboard
        logger.info("Creating constraint validation dashboard...")
        self.plot_constraint_validation_dashboard(results_df)

        # 2. Country comparison
        logger.info("Creating country comparison plot...")
        self.plot_all_countries_comparison(results_df, metric='total_revenue')

        # 3. Configuration heatmaps for each country
        logger.info("Creating configuration heatmaps...")
        for country in results_df['country'].unique():
            self.plot_configuration_heatmap(results_df, country, metric='total_revenue')

        # 4. Individual scenario plots (SOC, power, revenue) for selected scenarios
        logger.info("Creating individual scenario plots...")
        detailed_dir = Path(detailed_results_dir)

        # Plot for best scenario of each country
        for country in results_df['country'].unique():
            country_data = results_df[results_df['country'] == country]
            best_idx = country_data['total_revenue'].idxmax()
            best_scenario = country_data.loc[best_idx, 'scenario']

            # Load detailed result
            result_file = detailed_dir / f"{best_scenario}_detailed.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    detailed_result = json.load(f)

                self.plot_soc_trajectory(best_scenario, detailed_result)
                self.plot_power_profile(best_scenario, detailed_result)
                self.plot_revenue_breakdown(best_scenario, detailed_result)

        logger.info("All plots generated successfully!")

def main():
    """Main entry point for plotting script."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate BESS validation plots')
    parser.add_argument('--results', type=str,
                       default='validation_week_results/results/validation_summary.csv',
                       help='Path to validation_summary.csv')
    parser.add_argument('--detailed', type=str,
                       default='validation_week_results/results',
                       help='Path to detailed results directory')
    parser.add_argument('--output', type=str,
                       default='validation_week_results',
                       help='Output directory')

    args = parser.parse_args()

    # Create plotter
    plotter = ValidationPlotter(output_dir=args.output)

    # Generate all plots
    plotter.generate_all_plots(args.results, args.detailed)

    print("\n" + "="*80)
    print("PLOTTING COMPLETE!")
    print(f"Plots saved to: {plotter.plot_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
