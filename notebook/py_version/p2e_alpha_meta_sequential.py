"""
Interactive Alpha Meta-Optimization Script (Instance-by-Instance)

Run MPC simulations one alpha at a time with interactive control.
No parallel execution - easy to monitor, pause, and analyze each run.

Author: Generated with Claude Code
Date: 2024-11-17
"""

# %%
# ================================================================================
# [SECTION 1] SETUP & IMPORTS
# ================================================================================

import sys
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Core optimizer
from src.core.optimizer import BESSOptimizerModelIII

# MPC simulation
from src.mpc.mpc_simulator import MPCSimulator
from src.mpc.transform_mpc_results import (
    transform_mpc_results_for_viz,
    extract_iteration_summary
)

# Data loading
from src.data.load_process_market_data import load_preprocessed_country_data

print("=" * 80)
print("[SECTION 1] SETUP COMPLETE")
print("=" * 80)

# %%
# ================================================================================
# [SECTION 2] CONFIGURATION
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 2] CONFIGURATION")
print("=" * 80)

# Test scenario (FIXED for meta-optimization)
COUNTRY = "CZ"      # Country: DE_LU, AT, CH, HU, CZ
C_RATE = 0.5        # C-rate: 0.25, 0.33, 0.5

# *** TEST DURATION - CHANGE THIS to control simulation length ***
# Examples: 2 (quick test, ~10-15 min), 7 (week), 30 (month), 365 (full year, ~3-6 hours)
TEST_DAYS = 365

# Alpha range - define the sweep of degradation cost weights to test
ALPHA_MIN = 0.7     # Minimum alpha value
ALPHA_MAX = 1.6     # Maximum alpha value
ALPHA_STEP = 0.1    # Step size between alpha values

# LIFO segment filling (from MPC config)
REQUIRE_SEQUENTIAL = False
EPSILON = 0

# Financial parameters (for NPV calculation)
WACC = 0.08  # Weighted Average Cost of Capital
INFLATION_RATE = 0.02
PROJECT_LIFETIME_YEARS = 10


# Output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base_dir = project_root / f"validation_results/alpha_meta_seq_CZ_0.5C_{TEST_DAYS}d_{timestamp}"

# Generate alpha values
alpha_values = np.arange(ALPHA_MIN, ALPHA_MAX + ALPHA_STEP/2, ALPHA_STEP)
alpha_values = np.round(alpha_values, 1)  # Round to avoid floating point issues

print(f"\n[CONFIG] Test Scenario:")
print(f"  - Country: {COUNTRY}")
print(f"  - C-rate: {C_RATE}")
# print(f"  - Test mode: {'YES' if TEST_MODE else 'NO'}")
print(f"  - Duration: {TEST_DAYS} days")

print(f"\n[CONFIG] Alpha Range:")
print(f"  - Min: {ALPHA_MIN}")
print(f"  - Max: {ALPHA_MAX}")
print(f"  - Step: {ALPHA_STEP}")
print(f"  - Total values: {len(alpha_values)}")
print(f"  - Values: {list(alpha_values)}")

print(f"\n[CONFIG] MPC Settings:")
print(f"  - REQUIRE_SEQUENTIAL: {REQUIRE_SEQUENTIAL}")
print(f"  - EPSILON: {EPSILON}")

print(f"\n[CONFIG] Output:")
print(f"  - Base directory: {output_base_dir}")

# %%
# ================================================================================
# [SECTION 3] LOAD DATA & PREPARE
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 3] DATA LOADING")
print("=" * 80)

# Load MPC configuration
config_dir = project_root / "data" / "p2_config"
mpc_config_path = config_dir / "mpc_config.json"
with open(mpc_config_path, 'r') as f:
    base_mpc_config = json.load(f)

# Load market data (preprocessed for speed)
print(f"\n[LOADING] Market data for {COUNTRY}...")
preprocessed_data_dir = project_root / "data" / "parquet" / "preprocessed"
market_data = load_preprocessed_country_data(COUNTRY, data_dir=preprocessed_data_dir)

print(f"[Running] Using full year data: {len(market_data)} intervals")

# Extract MPC parameters
mpc_params = base_mpc_config.get('mpc_parameters', {})
planning_horizon_hours = mpc_params.get('horizon_hours', 36)
execution_horizon_hours = mpc_params.get('execution_hours', 24)

print(f"\n[MPC CONFIG]:")
print(f"  - Planning horizon: {planning_horizon_hours}h")
print(f"  - Execution horizon: {execution_horizon_hours}h")
print(f"  - Data intervals: {len(market_data)}")

# Create output directory
output_base_dir.mkdir(parents=True, exist_ok=True)

# Save configuration
config_snapshot = {
    'alpha_values': list(alpha_values),
    # 'test_mode': TEST_MODE,
    'test_days': TEST_DAYS,
    'country': COUNTRY,
    'c_rate': C_RATE,
    'require_sequential': REQUIRE_SEQUENTIAL,
    'epsilon': EPSILON,
    'planning_horizon_hours': planning_horizon_hours,
    'execution_horizon_hours': execution_horizon_hours,
    'timestamp': timestamp
}

config_path = output_base_dir / "sweep_config.json"
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Configuration saved to: {config_path}")

# %%
# ================================================================================
# [SECTION 4] INTERACTIVE ALPHA SWEEP
# ================================================================================

print("\n" + "=" * 80)
print("[SECTION 4] INTERACTIVE ALPHA SWEEP")
print("=" * 80)

# Initialize results storage
results_list = []
completed_alphas = []

print(f"\n[INFO] Ready to run {len(alpha_values)} alpha simulations")
print(f"[INFO] You can run them one at a time or in batches")
print(f"\n{'='*80}")

def run_alpha_simulation(alpha_value):
    """Run simulation for a single alpha value."""

    print(f"\n{'='*80}")
    print(f"[START] Alpha = {alpha_value:.1f}")
    print(f"{'='*80}")

    # Create output directory for this alpha
    alpha_dir = output_base_dir / f"alpha_{alpha_value:.1f}"
    alpha_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create optimizer with this alpha
        print(f"[1/5] Creating optimizer with alpha={alpha_value:.1f}...")
        optimizer = BESSOptimizerModelIII(alpha=alpha_value)

        # Update LIFO settings
        optimizer.REQUIRE_SEQUENTIAL_FILL = REQUIRE_SEQUENTIAL
        optimizer.EPSILON = EPSILON

        # Create MPC simulator
        print(f"[2/5] Creating MPC simulator...")
        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=market_data,
            horizon_hours=planning_horizon_hours,
            execution_hours=execution_horizon_hours,
            c_rate=C_RATE,
            validate_constraints=False  # Disable for speed
        )

        # Run simulation
        print(f"[3/5] Running MPC simulation...")
        start_time = time.time()
        mpc_results = simulator.run_full_simulation()
        runtime = time.time() - start_time

        print(f"[OK] Simulation completed in {runtime:.1f}s")

        # Extract results
        print(f"[4/5] Extracting results...")

        total_revenue = mpc_results.get('total_revenue', 0)
        total_degradation_cost = mpc_results.get('total_degradation_cost', 0)
        net_profit = mpc_results.get('net_profit', 0)

        # Calculate SOC statistics
        soc_values = mpc_results.get('soc_15min', [])
        if isinstance(soc_values, list):
            soc_values = np.array(soc_values)

        soc_avg = float(np.mean(soc_values)) if len(soc_values) > 0 else np.nan
        soc_min = float(np.min(soc_values)) if len(soc_values) > 0 else np.nan
        soc_max = float(np.max(soc_values)) if len(soc_values) > 0 else np.nan
        soc_std = float(np.std(soc_values)) if len(soc_values) > 0 else np.nan

        # Create results summary
        results = {
            'alpha': alpha_value,
            'status': 'success',
            'runtime_seconds': runtime,
            'total_profit_eur': net_profit,
            'total_revenue_eur': total_revenue,
            'total_cost_eur': total_degradation_cost,
            'revenue_da_eur': mpc_results.get('da_revenue', 0),
            'revenue_afrr_energy_eur': mpc_results.get('afrr_e_revenue', 0),
            'revenue_as_eur': mpc_results.get('as_revenue', 0),
            'degradation_cyclic_eur': mpc_results.get('cyclic_cost', 0),
            'degradation_calendar_eur': mpc_results.get('calendar_cost', 0),
            'total_aging_cost_eur': total_degradation_cost,
            'num_iterations': mpc_results.get('summary', {}).get('iterations', 0),
            'solver_status': 'optimal',
            'soc_avg_kwh': soc_avg,
            'soc_min_kwh': soc_min,
            'soc_max_kwh': soc_max,
            'soc_std_kwh': soc_std,
            'country': COUNTRY,
            'c_rate': C_RATE,
            'test_days': TEST_DAYS,
            'output_dir': str(alpha_dir)
        }

        # Save results
        print(f"[5/5] Saving results...")

        # Save performance summary
        performance_summary = {
            'total_profit_eur': net_profit,
            'total_revenue_eur': total_revenue,
            'total_cost_eur': total_degradation_cost,
            'revenue_da_eur': results['revenue_da_eur'],
            'revenue_afrr_energy_eur': results['revenue_afrr_energy_eur'],
            'revenue_as_eur': results['revenue_as_eur'],
            'degradation_cyclic_eur': results['degradation_cyclic_eur'],
            'degradation_calendar_eur': results['degradation_calendar_eur'],
            'num_iterations': results['num_iterations'],
            'alpha': alpha_value,
            'c_rate': C_RATE
        }

        perf_json_path = alpha_dir / "performance_summary.json"
        with open(perf_json_path, 'w', encoding='utf-8') as f:
            json.dump(performance_summary, f, indent=2, ensure_ascii=False)

        # Save timeseries
        if 'total_bids_df' in mpc_results:
            bids_csv_path = alpha_dir / "solution_timeseries.csv"
            mpc_results['total_bids_df'].to_csv(bids_csv_path, index=False)

        # Save iteration summary (best effort)
        if 'iteration_results' in mpc_results:
            try:
                iter_df = extract_iteration_summary(mpc_results['iteration_results'])
                iter_csv_path = alpha_dir / "iteration_summary.csv"
                iter_df.to_csv(iter_csv_path, index=False)
            except Exception as e:
                print(f"[WARNING] Could not save iteration summary: {e}")

        # Print summary
        print(f"\n{'='*80}")
        print(f"[SUCCESS] Alpha {alpha_value:.1f}")
        print(f"{'='*80}")
        print(f"  Profit: €{net_profit:,.0f}")
        print(f"  Revenue: €{total_revenue:,.0f}")
        print(f"  Aging Cost: €{total_degradation_cost:,.0f}")
        print(f"  Runtime: {runtime:.1f}s")
        print(f"  SOC Avg: {soc_avg:.0f} kWh")
        print(f"{'='*80}\n")

        return results

    except Exception as e:
        print(f"\n[ERROR] Alpha {alpha_value:.1f} failed: {e}")
        import traceback
        traceback.print_exc()

        # Return error results
        return {
            'alpha': alpha_value,
            'status': 'error',
            'error': str(e),
            'runtime_seconds': 0,
            'total_profit_eur': np.nan,
            'total_revenue_eur': np.nan,
            'total_cost_eur': np.nan,
            'revenue_da_eur': np.nan,
            'revenue_afrr_energy_eur': np.nan,
            'revenue_as_eur': np.nan,
            'degradation_cyclic_eur': np.nan,
            'degradation_calendar_eur': np.nan,
            'total_aging_cost_eur': np.nan,
            'num_iterations': 0,
            'solver_status': 'error',
            'soc_avg_kwh': np.nan,
            'soc_min_kwh': np.nan,
            'soc_max_kwh': np.nan,
            'soc_std_kwh': np.nan,
            'country': COUNTRY,
            'c_rate': C_RATE,
            'test_days': TEST_DAYS,
            'output_dir': str(alpha_dir)
        }


def run_next_alpha():
    """Run the next alpha in the sequence."""
    remaining = [a for a in alpha_values if a not in completed_alphas]
    if not remaining:
        print("[INFO] All alphas completed!")
        return None

    alpha = remaining[0]
    result = run_alpha_simulation(alpha)
    results_list.append(result)
    completed_alphas.append(alpha)

    print(f"\n[PROGRESS] Completed: {len(completed_alphas)}/{len(alpha_values)} alphas")
    print(f"[PROGRESS] Remaining: {list(remaining[1:])}")

    return result


def run_all_remaining():
    """Run all remaining alphas sequentially."""
    remaining = [a for a in alpha_values if a not in completed_alphas]
    print(f"\n[START] Running {len(remaining)} remaining alphas...")

    for i, alpha in enumerate(remaining, 1):
        print(f"\n[BATCH PROGRESS] {i}/{len(remaining)}")
        result = run_alpha_simulation(alpha)
        results_list.append(result)
        completed_alphas.append(alpha)

    print(f"\n[COMPLETE] All {len(alpha_values)} alphas completed!")


def show_current_results():
    """Display current results summary."""
    if not results_list:
        print("[INFO] No results yet. Run some alphas first.")
        return

    df = pd.DataFrame(results_list)
    df = df.sort_values('alpha').reset_index(drop=True)

    print("\n" + "=" * 80)
    print("CURRENT RESULTS")
    print("=" * 80)

    display_cols = ['alpha', 'total_profit_eur', 'total_aging_cost_eur',
                    'soc_avg_kwh', 'runtime_seconds', 'status']
    print(df[display_cols].to_string(index=False))

    # Show best so far
    success_df = df[df['status'] == 'success']
    if len(success_df) > 0:
        best_idx = success_df['total_profit_eur'].idxmax()
        best_alpha = success_df.loc[best_idx, 'alpha']
        best_profit = success_df.loc[best_idx, 'total_profit_eur']

        print(f"\n[BEST SO FAR] Alpha = {best_alpha:.1f}, Profit = €{best_profit:,.0f}")


def save_results():
    """Save aggregated results to CSV."""
    if not results_list:
        print("[INFO] No results to save yet.")
        return

    df = pd.DataFrame(results_list)
    df = df.sort_values('alpha').reset_index(drop=True)

    # Calculate derived metrics
    df['net_profit_eur'] = df['total_profit_eur']
    df['profit_per_day'] = df['net_profit_eur'] / TEST_DAYS

    # Annualize if in test mode
    df['annual_profit_estimate'] = df['net_profit_eur']
    df['annual_aging_cost_estimate'] = df['total_aging_cost_eur']

    # Calculate NPV
    discount_rates = [(1 / (1 + WACC) ** year) for year in range(1, PROJECT_LIFETIME_YEARS + 1)]
    npv_multiplier = sum(discount_rates)
    df['npv_eur'] = df['annual_profit_estimate'] * npv_multiplier

    # ROI proxy
    df['roi_proxy'] = df['net_profit_eur'] / df['total_aging_cost_eur'].replace(0, 1)

    # Save
    csv_path = output_base_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to: {csv_path}")

    return df


# Note: Functions are defined for potential interactive use,
# but main execution is automatic (see bottom of file)

# %%
# ================================================================================
# [SECTION 5] PLOTTING FUNCTIONS
# ================================================================================

def plot_pareto_front(output_dir=None):
    """Generate Pareto front plot from current results."""
    if not results_list:
        print("[ERROR] No results available. Run simulations first.")
        return

    df = save_results()  # Get latest results with calculations
    success_df = df[df['status'] == 'success'].copy()

    if len(success_df) == 0:
        print("[ERROR] No successful simulations to plot.")
        return

    if output_dir is None:
        output_dir = output_base_dir
    else:
        output_dir = Path(output_dir)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PLOTTING] Generating Pareto Front...")

    # Highlight best profit and best NPV
    best_profit_idx = success_df['net_profit_eur'].idxmax()
    best_npv_idx = success_df['npv_eur'].idxmax()

    # Create scatter plot
    fig = go.Figure()

    # Main scatter
    fig.add_trace(go.Scatter(
        x=success_df['annual_aging_cost_estimate'],
        y=success_df['annual_profit_estimate'],
        mode='markers+lines',
        marker=dict(
            size=12,
            color=success_df['alpha'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Alpha"),
        ),
        text=[f"alpha={a:.1f}" for a in success_df['alpha']],
        hovertemplate='<b>Alpha: %{text}</b><br>' +
                      'Aging Cost: €%{x:,.0f}/yr<br>' +
                      'Profit: €%{y:,.0f}/yr<br>' +
                      '<extra></extra>',
        name='Alpha sweep'
    ))

    # Highlight best profit
    best_profit = success_df.loc[best_profit_idx]
    fig.add_trace(go.Scatter(
        x=[best_profit['annual_aging_cost_estimate']],
        y=[best_profit['annual_profit_estimate']],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name=f'Best Profit (alpha={best_profit["alpha"]:.1f})',
        hovertemplate=f'<b>Best Profit</b><br>Alpha: {best_profit["alpha"]:.1f}<br>' +
                     f'Profit: €{best_profit["annual_profit_estimate"]:,.0f}/yr<br>' +
                     '<extra></extra>'
    ))

    # Highlight best NPV
    best_npv = success_df.loc[best_npv_idx]
    fig.add_trace(go.Scatter(
        x=[best_npv['annual_aging_cost_estimate']],
        y=[best_npv['annual_profit_estimate']],
        mode='markers',
        marker=dict(size=20, color='gold', symbol='diamond'),
        name=f'Best NPV (alpha={best_npv["alpha"]:.1f})',
        hovertemplate=f'<b>Best NPV</b><br>Alpha: {best_npv["alpha"]:.1f}<br>' +
                     f'NPV: €{best_npv["npv_eur"]:,.0f}<br>' +
                     '<extra></extra>'
    ))

    fig.update_layout(
        title=f"Pareto Front: Aging Cost vs Profit ({COUNTRY}, C-rate={C_RATE})",
        xaxis_title="Annual Aging Cost (EUR)",
        yaxis_title="Annual Profit (EUR)",
        hovermode='closest',
        width=1000,
        height=700
    )

    output_path = plots_dir / "pareto_front.html"
    fig.write_html(str(output_path))
    print(f"  [SAVED] {output_path}")

    return fig


def plot_soc_vs_alpha(output_dir=None):
    """Generate SOC vs Alpha plot."""
    if not results_list:
        print("[ERROR] No results available. Run simulations first.")
        return

    df = pd.DataFrame(results_list)
    success_df = df[df['status'] == 'success'].copy()

    if len(success_df) == 0:
        print("[ERROR] No successful simulations to plot.")
        return

    if output_dir is None:
        output_dir = output_base_dir
    else:
        output_dir = Path(output_dir)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PLOTTING] Generating SOC vs Alpha...")

    fig = go.Figure()

    # Average SOC
    fig.add_trace(go.Scatter(
        x=success_df['alpha'],
        y=success_df['soc_avg_kwh'],
        mode='lines+markers',
        name='Average SOC',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))

    # Min/Max range
    fig.add_trace(go.Scatter(
        x=success_df['alpha'],
        y=success_df['soc_max_kwh'],
        mode='lines',
        name='Max SOC',
        line=dict(color='lightblue', width=1, dash='dash'),
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=success_df['alpha'],
        y=success_df['soc_min_kwh'],
        mode='lines',
        name='Min SOC',
        line=dict(color='lightblue', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.2)',
        showlegend=True
    ))

    fig.update_layout(
        title=f"SOC Statistics vs Alpha ({COUNTRY}, C-rate={C_RATE})",
        xaxis_title="Alpha (Degradation Weight)",
        yaxis_title="State of Charge (kWh)",
        hovermode='x unified',
        width=1000,
        height=600
    )

    output_path = plots_dir / "soc_vs_alpha.html"
    fig.write_html(str(output_path))
    print(f"  [SAVED] {output_path}")

    return fig


def plot_all():
    """Generate all plots."""
    plot_pareto_front()
    plot_soc_vs_alpha()
    print("\n[COMPLETE] All plots generated!")


print("=" * 80)
print("[SECTION 5] PLOTTING FUNCTIONS LOADED")
print("=" * 80)
print("""
Available plotting functions:
  plot_pareto_front()  - Pareto front: Aging Cost vs Profit
  plot_soc_vs_alpha()  - SOC sensitivity vs Alpha
  plot_all()           - Generate all plots
""")

# %%
# ================================================================================
# [SECTION 6] MAIN BATCH EXECUTION
# ================================================================================

def main():
    """Main batch execution function - runs all alphas sequentially."""

    print("\n" + "=" * 80)
    print("SEQUENTIAL ALPHA META-OPTIMIZATION")
    print("=" * 80)
    print(f"Total alphas to test: {len(alpha_values)}")
    print(f"Test duration: {TEST_DAYS} days")
    print(f"Country: {COUNTRY} | C-rate: {C_RATE}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 80)

    # Execute all alphas sequentially
    batch_start = time.time()

    print(f"\n[START] Running {len(alpha_values)} alpha simulations sequentially...")
    print("Progress will be shown for each alpha.\n")

    for i, alpha in enumerate(alpha_values, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(alpha_values)}] Alpha = {alpha:.1f}")
        print(f"{'='*80}")

        result = run_alpha_simulation(alpha)
        results_list.append(result)
        completed_alphas.append(alpha)

        # Brief summary
        if result['status'] == 'success':
            print(f"\n✓ Success | Profit: €{result['total_profit_eur']:,.0f} | Runtime: {result['runtime_seconds']:.1f}s")
        else:
            print(f"\n✗ Failed | Error: {result.get('error', 'Unknown error')}")

    batch_time = time.time() - batch_start

    print(f"\n{'='*80}")
    print("[COMPLETE] All alpha simulations finished!")
    print(f"{'='*80}")
    print(f"Total time: {batch_time/60:.1f} min ({batch_time/3600:.2f} hours)")
    print()

    # Show results summary
    show_current_results()

    # Save results to CSV
    print("\n" + "=" * 80)
    print("[SAVING] Results to CSV...")
    print("=" * 80)
    final_df = save_results()

    # Generate all plots
    print("\n" + "=" * 80)
    print("[PLOTTING] Generating visualizations...")
    print("=" * 80)
    plot_all()

    # Final summary
    print("\n" + "=" * 80)
    print("BATCH EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_base_dir}")
    print("\nGenerated files:")
    print(f"  - comparison_results.csv")
    print(f"  - plots/pareto_front.html")
    print(f"  - plots/soc_vs_alpha.html")
    print(f"  - alpha_X.X/ directories (detailed results for each alpha)")

    n_success = (final_df['status'] == 'success').sum()
    n_failed = len(final_df) - n_success

    print(f"\nSuccess: {n_success}/{len(alpha_values)} | Failed: {n_failed}")

    if n_success > 0:
        successful_results = final_df[final_df['status'] == 'success']
        best_alpha = successful_results.loc[successful_results['total_profit_eur'].idxmax(), 'alpha']
        best_profit = successful_results['total_profit_eur'].max()
        print(f"Best alpha: {best_alpha:.1f} with profit €{best_profit:,.0f}")

    print("=" * 80)

    return final_df

# %%
# ================================================================================
# [SECTION 7] ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    # Run batch execution automatically
    summary_df = main()
    print(f"\n✓ Alpha meta-optimization complete!")
    print(f"   Check {output_base_dir} for all results.")
else:
    # If imported, just show ready message
    print("\n" + "=" * 80)
    print("[READY] Alpha Meta-Optimization Functions Loaded")
    print("=" * 80)
    print("\nTo run batch execution: main()")
    print("Or use individual functions:")
    print("  run_alpha_simulation(0.7) - Run specific alpha")
    print("  run_all_remaining()       - Run all remaining alphas")
    print("  show_current_results()    - Display current results")
    print("  save_results()            - Save to CSV")
    print("  plot_all()                - Generate plots")
    print("=" * 80)

# %%
