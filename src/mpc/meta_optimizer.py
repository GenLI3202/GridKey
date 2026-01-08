"""
Meta-Optimizer for BESS Degradation Price Parameter Tuning
===========================================================

This module implements the outer layer of the three-layer optimization framework.
It performs a parameter sweep over the degradation price (α) to find the optimal
balance between revenue and battery lifetime.

Three-Layer Architecture:
--------------------------
1. Inner Layer: BESSOptimizerModelIII (single MILP solve)
2. Middle Layer: MPCSimulator (rolling horizon simulation)
3. Outer Layer: MetaOptimizer (THIS MODULE - alpha sweep)

Meta-Optimization Algorithm:
-----------------------------
For each α in [α_min, α_max]:
    1. Create BESSOptimizerModelIII with alpha=α
    2. Run full-year MPC simulation
    3. Calculate 10-year ROI based on annual profit
    4. Track best α

Return: α* that maximizes 10-year ROI

Financial Model:
----------------
- Fixed 10-year project lifetime (per competition rules)
- NPV calculation with WACC and inflation
- No battery replacement (cost already captured in degradation term)

References:
-----------
- doc/p2_model/p2_bi_model_ggdp.tex: Section "Meta-Optimization"
- whole_project_description.md: "Important Update" on fixed lifetime

Author: Gen's BESS Optimization Team
Phase II Development: November 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import required modules
import sys
from ..core.optimizer import BESSOptimizerModelIII
from .mpc_simulator import MPCSimulator

logger = logging.getLogger(__name__)


class MetaOptimizer:
    """Meta-optimizer for finding optimal degradation price parameter (α).

    Performs a parameter sweep to maximize 10-year Return on Investment (ROI)
    by finding the optimal trade-off between revenue and degradation cost.

    Key Features:
    -------------
    - Systematic alpha parameter sweep
    - 10-year NPV calculation with WACC and inflation
    - Parallel execution support (optional)
    - Comprehensive results tracking and export

    Attributes:
        full_data: Full year market data
        country_config: Financial parameters (WACC, inflation, investment cost)
        alpha_values: List of alpha values to test
        c_rate: C-rate configuration
        mpc_config: MPC simulation parameters

    Example:
        # Define alpha sweep
        alpha_values = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        # Create meta-optimizer
        meta_opt = MetaOptimizer(
            full_data=data,
            country_config={'wacc': 0.05, 'inflation': 0.02, 'investment_eur_per_kwh': 200},
            alpha_values=alpha_values,
            c_rate=0.5
        )

        # Find optimal alpha
        results = meta_opt.find_optimal_alpha()
        print(f"Optimal alpha: {results['best_alpha']}")
        print(f"Best ROI: {results['best_roi']:.2%}")
    """

    def __init__(
        self,
        full_data: pd.DataFrame,
        country_config: Dict[str, Any],
        alpha_values: List[float],
        c_rate: float = 0.5,
        mpc_config: Optional[Dict[str, Any]] = None,
        degradation_config_path: Optional[str] = None,
        use_afrr_ev_weighting: bool = False,
    ) -> None:
        """Initialize meta-optimizer.

        Args:
            full_data: Full year market data DataFrame
            country_config: Dict with keys:
                - 'wacc': Weighted average cost of capital (e.g., 0.05)
                - 'inflation': Annual inflation rate (e.g., 0.02)
                - 'investment_eur_per_kwh': BESS investment cost (e.g., 200)
                - 'capacity_kwh': Battery capacity (e.g., 4472)
            alpha_values: List of alpha values to test
            c_rate: C-rate configuration (default 0.5)
            mpc_config: MPC parameters (horizon, execution window)
            degradation_config_path: Path to aging_config.json
            use_afrr_ev_weighting: Enable Expected Value weighting for aFRR energy bids (default False)
        """
        self.full_data = full_data
        self.country_config = country_config
        self.alpha_values = sorted(alpha_values)
        self.c_rate = c_rate
        self.degradation_config_path = degradation_config_path
        self.use_afrr_ev_weighting = use_afrr_ev_weighting

        # MPC configuration
        self.mpc_config = mpc_config or {
            'horizon_hours': 48,
            'execution_hours': 24,
            'validate_constraints': False,  # Disable for speed in meta-opt
        }

        # Validate country config
        required_keys = ['wacc', 'inflation', 'investment_eur_per_kwh', 'capacity_kwh']
        for key in required_keys:
            if key not in country_config:
                raise KeyError(f"Missing '{key}' in country_config")

        # Calculate total investment
        self.total_investment = (
            country_config['capacity_kwh'] * country_config['investment_eur_per_kwh']
        )

        logger.info("=" * 80)
        logger.info("META-OPTIMIZER INITIALIZED")
        logger.info("=" * 80)
        logger.info("  Alpha values to test: %d (min=%.2f, max=%.2f)",
                   len(alpha_values), min(alpha_values), max(alpha_values))
        logger.info("  C-rate: %.2f", c_rate)
        logger.info("  MPC horizon: %d hours", self.mpc_config['horizon_hours'])
        logger.info("  MPC execution: %d hours", self.mpc_config['execution_hours'])
        logger.info("  Financial parameters:")
        logger.info("    - WACC: %.2f%%", country_config['wacc'] * 100)
        logger.info("    - Inflation: %.2f%%", country_config['inflation'] * 100)
        logger.info("    - Investment cost: %.0f EUR/kWh", country_config['investment_eur_per_kwh'])
        logger.info("    - Total investment: %.0f EUR", self.total_investment)
        logger.info("=" * 80)

    def _calculate_10_year_roi(
        self,
        annual_profit: float,
        annual_degradation_cost: float
    ) -> float:
        """Calculate 10-year Return on Investment (ROI).

        Uses Net Present Value (NPV) approach with WACC and inflation.

        IMPORTANT: Per competition rules (whole_project_description.md "Important Update"),
        we use a FIXED 10-year lifetime, NOT battery replacement based on SOH.
        The degradation cost is already subtracted from annual_profit in the
        optimization objective, so we only use it for reporting.

        Args:
            annual_profit: Annual net profit (EUR) - already includes degradation cost deduction
            annual_degradation_cost: Annual degradation cost (EUR) - for reporting only

        Returns:
            ROI as a fraction (e.g., 0.50 for 50% ROI)
        """
        wacc = self.country_config['wacc']
        inflation = self.country_config['inflation']

        # Calculate NPV over 10 years
        npv = 0.0
        for year in range(1, 11):  # Years 1-10
            # Profit in year m (adjusted for inflation)
            profit_year_m = annual_profit * ((1 + inflation) ** (year - 1))

            # Discount to present value
            pv_profit_year_m = profit_year_m / ((1 + wacc) ** year)

            npv += pv_profit_year_m

        # ROI = (NPV - Initial Investment) / Initial Investment
        roi = (npv - self.total_investment) / self.total_investment

        return roi

    def _run_single_alpha(self, alpha: float) -> Dict[str, Any]:
        """Run full simulation for a single alpha value.

        Args:
            alpha: Degradation price parameter

        Returns:
            Dictionary with simulation results and ROI
        """
        logger.info("")
        logger.info("─" * 80)
        logger.info("META-OPTIMIZATION: Testing alpha = %.4f", alpha)
        logger.info("─" * 80)

        try:
            # 1. Create optimizer with this alpha
            optimizer = BESSOptimizerModelIII(
                alpha=alpha,
                degradation_config_path=self.degradation_config_path,
                use_afrr_ev_weighting=self.use_afrr_ev_weighting,
            )

            # 2. Create MPC simulator
            simulator = MPCSimulator(
                optimizer_model=optimizer,
                full_data=self.full_data,
                horizon_hours=self.mpc_config['horizon_hours'],
                execution_hours=self.mpc_config['execution_hours'],
                c_rate=self.c_rate,
                validate_constraints=self.mpc_config.get('validate_constraints', False),
            )

            # 3. Run full simulation
            start_time = datetime.now()
            annual_results = simulator.run_full_simulation()
            simulation_time = (datetime.now() - start_time).total_seconds()

            # 4. Extract metrics
            # Note: net_profit already has degradation cost subtracted
            annual_profit = annual_results['net_profit']
            annual_degradation_cost = annual_results['total_degradation_cost']

            # 5. Calculate 10-year ROI
            roi_10_year = self._calculate_10_year_roi(
                annual_profit,
                annual_degradation_cost
            )

            # 6. Create result summary
            result = {
                'alpha': alpha,
                'roi_10_year': roi_10_year,
                'annual_revenue_eur': annual_results['total_revenue'],
                'annual_profit_eur': annual_profit,
                'annual_degradation_cost_eur': annual_degradation_cost,
                'annual_cyclic_cost_eur': annual_results['cyclic_cost'],
                'annual_calendar_cost_eur': annual_results['calendar_cost'],
                'final_soc_kwh': annual_results['final_soc'],
                'simulation_time_sec': simulation_time,
                'iterations': len(annual_results['iteration_results']),
                'status': 'success',
            }

            logger.info("  Results for alpha=%.4f:", alpha)
            logger.info("    - Annual revenue: %.0f EUR", annual_results['total_revenue'])
            logger.info("    - Annual degradation: %.0f EUR", annual_degradation_cost)
            logger.info("    - Annual profit: %.0f EUR", annual_profit)
            logger.info("    - 10-year ROI: %.2f%%", roi_10_year * 100)
            logger.info("    - Simulation time: %.1f seconds", simulation_time)

            return result

        except Exception as e:
            logger.error("  ERROR running alpha=%.4f: %s", alpha, str(e))
            return {
                'alpha': alpha,
                'status': 'failed',
                'error': str(e),
                'roi_10_year': -np.inf,
            }

    def find_optimal_alpha(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Find optimal alpha value through parameter sweep.

        Args:
            parallel: Whether to run alpha tests in parallel (default False)
            max_workers: Max parallel workers (default: CPU count - 1)

        Returns:
            Dictionary with:
                - best_alpha: Optimal alpha value
                - best_roi: Best 10-year ROI
                - all_results: List of results for all alpha values
                - summary_df: Pandas DataFrame with all results
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("STARTING META-OPTIMIZATION PARAMETER SWEEP")
        logger.info("=" * 80)
        logger.info("  Testing %d alpha values", len(self.alpha_values))
        logger.info("  Parallel execution: %s", "Enabled" if parallel else "Disabled")

        start_time = datetime.now()
        all_results = []

        if parallel and len(self.alpha_values) > 1:
            # Parallel execution
            if max_workers is None:
                max_workers = max(1, multiprocessing.cpu_count() - 1)

            logger.info("  Max workers: %d", max_workers)
            logger.info("")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_alpha = {
                    executor.submit(self._run_single_alpha, alpha): alpha
                    for alpha in self.alpha_values
                }

                # Collect results as they complete
                for future in as_completed(future_to_alpha):
                    alpha = future_to_alpha[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        logger.error("Exception for alpha=%.4f: %s", alpha, str(e))
                        all_results.append({
                            'alpha': alpha,
                            'status': 'failed',
                            'error': str(e),
                            'roi_10_year': -np.inf,
                        })

        else:
            # Sequential execution
            for alpha in self.alpha_values:
                result = self._run_single_alpha(alpha)
                all_results.append(result)

        # Sort results by alpha
        all_results.sort(key=lambda x: x['alpha'])

        # Find best result
        successful_results = [r for r in all_results if r['status'] == 'success']

        if not successful_results:
            logger.error("No successful simulations! Cannot find optimal alpha.")
            return {
                'status': 'failed',
                'error': 'All simulations failed',
                'all_results': all_results,
            }

        best_result = max(successful_results, key=lambda x: x['roi_10_year'])
        best_alpha = best_result['alpha']
        best_roi = best_result['roi_10_year']

        total_time = (datetime.now() - start_time).total_seconds()

        # Create summary DataFrame
        summary_df = pd.DataFrame(all_results)

        logger.info("")
        logger.info("=" * 80)
        logger.info("META-OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info("  Total time: %.1f seconds (%.1f minutes)",
                   total_time, total_time / 60)
        logger.info("  Successful runs: %d / %d",
                   len(successful_results), len(all_results))
        logger.info("")
        logger.info("  OPTIMAL SOLUTION:")
        logger.info("    - Best alpha: %.4f", best_alpha)
        logger.info("    - Best 10-year ROI: %.2f%%", best_roi * 100)
        logger.info("    - Annual revenue: %.0f EUR",
                   best_result.get('annual_revenue_eur', 0))
        logger.info("    - Annual degradation: %.0f EUR",
                   best_result.get('annual_degradation_cost_eur', 0))
        logger.info("    - Annual profit: %.0f EUR",
                   best_result.get('annual_profit_eur', 0))
        logger.info("=" * 80)

        return {
            'status': 'success',
            'best_alpha': best_alpha,
            'best_roi': best_roi,
            'best_result': best_result,
            'all_results': all_results,
            'summary_df': summary_df,
            'meta_optimization_time_sec': total_time,
            'config': {
                'alpha_values': self.alpha_values,
                'c_rate': self.c_rate,
                'mpc_config': self.mpc_config,
                'country_config': self.country_config,
            },
        }

    def export_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        prefix: str = "meta_opt"
    ) -> None:
        """Export meta-optimization results to files.

        Args:
            results: Results dictionary from find_optimal_alpha()
            output_dir: Output directory
            prefix: Filename prefix (default "meta_opt")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Export summary CSV
        if 'summary_df' in results and results['summary_df'] is not None:
            csv_path = output_dir / f"{prefix}_summary_{timestamp}.csv"
            results['summary_df'].to_csv(csv_path, index=False)
            logger.info("Exported summary CSV: %s", csv_path)

        # 2. Export detailed results JSON
        json_path = output_dir / f"{prefix}_results_{timestamp}.json"

        # Prepare JSON-serializable version
        json_results = {
            'status': results['status'],
            'best_alpha': results.get('best_alpha'),
            'best_roi': results.get('best_roi'),
            'best_result': results.get('best_result'),
            'all_results': results.get('all_results', []),
            'meta_optimization_time_sec': results.get('meta_optimization_time_sec'),
            'config': results.get('config', {}),
            'timestamp': timestamp,
        }

        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info("Exported detailed JSON: %s", json_path)

        # 3. Export best alpha configuration
        if results['status'] == 'success':
            best_config_path = output_dir / f"{prefix}_best_alpha.txt"
            with open(best_config_path, 'w') as f:
                f.write(f"Optimal Alpha: {results['best_alpha']:.6f}\n")
                f.write(f"10-Year ROI: {results['best_roi']:.4%}\n")
                f.write(f"Timestamp: {timestamp}\n")

            logger.info("Exported best alpha config: %s", best_config_path)

        logger.info("All results exported to: %s", output_dir)
