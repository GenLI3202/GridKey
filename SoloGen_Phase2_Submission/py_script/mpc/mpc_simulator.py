"""
MPC Simulator for BESS Rolling Horizon Optimization
====================================================

This module implements the Model Predictive Control (MPC) simulation layer for
Battery Energy Storage System (BESS) optimization. It uses a receding horizon
approach to solve the full-year optimization problem in manageable chunks.

Three-Layer Architecture:
--------------------------
1. Inner Layer: BESSOptimizerModelIII (single MILP solve for short horizon)
2. Middle Layer: MPCSimulator (THIS MODULE - rolling horizon simulation)
3. Outer Layer: MetaOptimizer (alpha parameter sweep)

MPC Algorithm:
--------------
For each day/window in the year:
    1. Extract data for optimization horizon (e.g., 48 hours)
    2. Set initial SOC from previous window's final state
    3. Solve MILP for this horizon
    4. Execute/record only the first execution window (e.g., 24 hours)
    5. Update state for next iteration

This approach makes the full-year problem computationally tractable while
maintaining near-optimal solutions through lookahead.

References:
-----------
- Collath et al. (2023): Rolling horizon implementation for BESS
- doc/p2_model/p2_bi_model_ggdp.tex: Section "Rolling Horizon (MPC)"

Author: Gen's BESS Optimization Team
Phase II Development: November 2025
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import the optimizer (Model III)
from ..core.optimizer import BESSOptimizerModelIII
from ..validation import validate_solution

logger = logging.getLogger(__name__)


class MPCSimulator:
    """MPC-based rolling horizon simulator for BESS optimization.

    Implements the middle layer of the three-layer optimization framework.
    Uses Model Predictive Control to solve the full-year problem in rolling windows.

    Attributes:
        optimizer_model: Instance of BESSOptimizerModelIII
        full_data: Full year market data DataFrame
        horizon_hours: Optimization horizon length (default 48h)
        execution_hours: Execution/commitment window (default 24h)
        total_steps: Total number of 15-min intervals in data
        horizon_steps: Number of intervals in horizon
        execution_steps: Number of intervals in execution window

    Example:
        optimizer = BESSOptimizerModelIII(alpha=1.0)
        simulator = MPCSimulator(optimizer, full_data, horizon_hours=48, execution_hours=24)
        results = simulator.run_full_simulation()
        print(f"Annual profit: {results['total_revenue']:.2f} EUR")
        print(f"Total degradation: {results['total_degradation_cost']:.2f} EUR")
    """

    def __init__(
        self,
        optimizer_model: BESSOptimizerModelIII,
        full_data: pd.DataFrame,
        horizon_hours: int = 48,
        execution_hours: int = 24,
        c_rate: float = 0.5,
        validate_constraints: bool = True,
        solver_name: str = None,
    ) -> None:
        """Initialize MPC simulator.

        Args:
            optimizer_model: Instance of BESSOptimizerModelIII (or ModelII, ModelI)
            full_data: Full year DataFrame from load_and_preprocess_data()
            horizon_hours: Optimization horizon (default 48h for 2-day lookahead)
            execution_hours: Execution window (default 24h for 1-day commitment)
            c_rate: C-rate configuration (default 0.5)
            validate_constraints: Whether to run post-solve validation (default True)
            solver_name: Solver to use ('gurobi', 'cplex', 'cbc', etc.). If None, auto-detect.
        """
        self.optimizer = optimizer_model
        self.full_data = full_data
        self.horizon_hours = horizon_hours
        self.execution_hours = execution_hours
        self.c_rate = c_rate
        self.validate_constraints = validate_constraints
        self.solver_name = solver_name

        # Get parameters from optimizer
        self.time_step_hours = self.optimizer.market_params['time_step_hours']
        self.battery_params = self.optimizer.battery_params
        self.degradation_params = getattr(self.optimizer, 'degradation_params', {})

        # Calculate step counts
        self.total_steps = len(full_data)
        self.horizon_steps = int(horizon_hours / self.time_step_hours)
        self.execution_steps = int(execution_hours / self.time_step_hours)

        # Segment parameters (for Model II and III)
        self.num_segments = self.degradation_params.get('num_segments', 10)
        self.segment_capacity = self.degradation_params.get('segment_capacity_kwh',
                                                             self.battery_params['capacity_kwh'] / 10)

        logger.info("=" * 80)
        logger.info("MPC SIMULATOR INITIALIZED")
        logger.info("=" * 80)
        logger.info("  Optimizer: %s", type(self.optimizer).__name__)
        logger.info("  Total data: %d intervals (%.1f days)",
                   self.total_steps, self.total_steps * self.time_step_hours / 24)
        logger.info("  Horizon: %d hours (%d intervals)",
                   self.horizon_hours, self.horizon_steps)
        logger.info("  Execution window: %d hours (%d intervals)",
                   self.execution_hours, self.execution_steps)
        logger.info("  Expected iterations: ~%d",
                   int(np.ceil(self.total_steps / self.execution_steps)))
        logger.info("  C-rate: %.2f", self.c_rate)
        logger.info("  Validation: %s", "Enabled" if validate_constraints else "Disabled")
        logger.info("=" * 80)

    def _get_initial_segment_soc(self, total_soc_kwh: float) -> Dict[int, float]:
        """Convert total SOC to segment-wise SOC distribution.

        Uses top-down filling strategy: fill from shallowest segment (1) down to
        deepest segment (J). This matches the "stacked tank" model in Model II/III.

        Args:
            total_soc_kwh: Total energy in battery (kWh)

        Returns:
            Dictionary mapping segment index j to energy in segment (kWh)
            e.g., {1: 447.2, 2: 447.2, ..., 10: 447.2} for 50% SOC
        """
        segment_soc = {}
        remaining_soc = total_soc_kwh

        # Fill from segment 1 (shallowest) to segment J (deepest)
        for j in range(1, self.num_segments + 1):
            energy_in_this_segment = min(remaining_soc, self.segment_capacity)
            segment_soc[j] = energy_in_this_segment
            remaining_soc -= energy_in_this_segment

        # Verify total
        total_filled = sum(segment_soc.values())
        if abs(total_filled - total_soc_kwh) > 1e-6:
            logger.warning(
                "SOC distribution mismatch: requested %.2f kWh, filled %.2f kWh",
                total_soc_kwh, total_filled
            )

        return segment_soc

    def _extract_window_results(
        self,
        solution: Dict[str, Any],
        execution_length: int,
        country_data_window: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extract and aggregate results for the execution window.

        Args:
            solution: Full solution dict from solve_model()
            execution_length: Number of time steps to extract (usually self.execution_steps)
            country_data_window: Market data for this window (for price calculation)

        Returns:
            Dictionary with profit, costs, and energy metrics for this window
        """
        # Extract power and price data
        p_ch = solution.get('p_ch', {})
        p_dis = solution.get('p_dis', {})
        p_afrr_pos_e = solution.get('p_afrr_pos_e', {})
        p_afrr_neg_e = solution.get('p_afrr_neg_e', {})

        c_fcr = solution.get('c_fcr', {})
        c_afrr_pos = solution.get('c_afrr_pos', {})
        c_afrr_neg = solution.get('c_afrr_neg', {})

        # Time step duration
        dt = self.time_step_hours

        # Initialize accumulators (detailed breakdown for waterfall visualization)
        da_discharge_revenue = 0.0  # Positive: discharge revenue
        da_charge_cost = 0.0        # Negative/cost: charge cost
        da_revenue = 0.0            # Net: discharge - charge
        afrr_e_revenue = 0.0
        fcr_revenue = 0.0
        afrr_pos_cap_revenue = 0.0
        afrr_neg_cap_revenue = 0.0
        as_revenue = 0.0
        total_revenue = 0.0

        # Calculate DA energy revenue for execution window
        for t in range(execution_length):
            if t >= len(country_data_window):
                logger.warning("Execution window exceeds data window at t=%d", t)
                break

            price_da = country_data_window['price_day_ahead'].iloc[t]

            # DA market revenue (separate discharge and charge for waterfall visualization)
            da_discharge_revenue += p_dis.get(t, 0) * price_da / 1000 * dt
            da_charge_cost += p_ch.get(t, 0) * price_da / 1000 * dt
            da_revenue += (p_dis.get(t, 0) - p_ch.get(t, 0)) * price_da / 1000 * dt

            # aFRR energy market revenue
            # NOTE: Handle NaN prices from preprocessing (0 price = market not activated)
            price_afrr_pos = country_data_window['price_afrr_energy_pos'].iloc[t]
            price_afrr_neg = country_data_window['price_afrr_energy_neg'].iloc[t]

            # Only add revenue if price is not NaN (market was activated)
            if pd.notna(price_afrr_pos):
                afrr_e_revenue += p_afrr_pos_e.get(t, 0) * price_afrr_pos / 1000 * dt
            if pd.notna(price_afrr_neg):
                afrr_e_revenue -= p_afrr_neg_e.get(t, 0) * price_afrr_neg / 1000 * dt

        # Calculate AS capacity revenue for execution window (separate by market for waterfall viz)
        # Note: AS prices are block-based, need to map time to blocks
        block_duration_hours = self.optimizer.market_params['block_duration_hours']
        for t in range(execution_length):
            if t >= len(country_data_window):
                break

            block_id = int(country_data_window['block_id'].iloc[t])

            # AS capacity revenue (only count once per block)
            # Check if this is the first timestep of this block in our window
            if t == 0 or int(country_data_window['block_id'].iloc[t-1]) != block_id:
                price_fcr = country_data_window['price_fcr'].iloc[t]
                price_afrr_pos = country_data_window['price_afrr_pos'].iloc[t]
                price_afrr_neg = country_data_window['price_afrr_neg'].iloc[t]

                # Separate revenue by capacity market
                fcr_block_revenue = c_fcr.get(block_id, 0) * price_fcr * block_duration_hours
                afrr_pos_block_revenue = c_afrr_pos.get(block_id, 0) * price_afrr_pos * block_duration_hours
                afrr_neg_block_revenue = c_afrr_neg.get(block_id, 0) * price_afrr_neg * block_duration_hours

                fcr_revenue += fcr_block_revenue
                afrr_pos_cap_revenue += afrr_pos_block_revenue
                afrr_neg_cap_revenue += afrr_neg_block_revenue

                as_revenue += fcr_block_revenue + afrr_pos_block_revenue + afrr_neg_block_revenue

        total_revenue = da_revenue + afrr_e_revenue + as_revenue

        # Extract degradation costs (if available)
        degradation_metrics = solution.get('degradation_metrics', {})
        cyclic_cost = degradation_metrics.get('total_cyclic_cost_eur', 0.0)
        calendar_cost = degradation_metrics.get('total_calendar_cost_eur', 0.0)

        # Scale degradation costs to execution window
        # (solution costs are for full horizon, we only execute part of it)
        scale_factor = execution_length / len(solution.get('p_ch', {})) if solution.get('p_ch') else 1.0
        cyclic_cost_window = cyclic_cost * scale_factor
        calendar_cost_window = calendar_cost * scale_factor

        return {
            # Aggregated (backward compatibility)
            'da_revenue': da_revenue,
            'afrr_e_revenue': afrr_e_revenue,
            'as_revenue': as_revenue,
            'total_revenue': total_revenue,
            'cyclic_cost': cyclic_cost_window,
            'calendar_cost': calendar_cost_window,
            'total_degradation_cost': cyclic_cost_window + calendar_cost_window,
            # Detailed breakdown for waterfall visualization
            'da_discharge_revenue': da_discharge_revenue,
            'da_charge_cost': da_charge_cost,
            'fcr_revenue': fcr_revenue,
            'afrr_pos_cap_revenue': afrr_pos_cap_revenue,
            'afrr_neg_cap_revenue': afrr_neg_cap_revenue,
        }

    def run_full_simulation(
        self,
        initial_soc_fraction: float = 0.5,
        max_iterations: Optional[int] = None,
        checkpoint_interval_minutes: Optional[float] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute full-year rolling horizon simulation.

        Main MPC loop that solves the optimization problem in rolling windows.

        Args:
            initial_soc_fraction: Initial state of charge (fraction of capacity, default 0.5)
            max_iterations: Maximum iterations (for testing, None = full year)
            checkpoint_interval_minutes: Save checkpoint every N minutes (None = no checkpointing)
            checkpoint_path: Path to save checkpoint file (default: './mpc_checkpoint.pkl')

        Returns:
            Dictionary with annual results:
                - total_revenue: Total annual revenue (EUR)
                - total_degradation_cost: Total degradation cost (EUR)
                - final_soc: Final state of charge (kWh)
                - soc_total_bids_df: List of SOC values over time
                - iteration_results: Detailed results per iteration
                - validation_reports: Constraint validation reports (if enabled)
        """
        logger.info("=" * 80)
        logger.info("STARTING FULL-YEAR MPC SIMULATION")
        logger.info("=" * 80)

        # Initialize state
        current_total_soc = initial_soc_fraction * self.battery_params['capacity_kwh']
        logger.info("Initial SOC: %.2f kWh (%.1f%%)",
                   current_total_soc,
                   100 * current_total_soc / self.battery_params['capacity_kwh'])

        # Storage for results
        iteration_results = []
        all_soc_total_bids_df = [current_total_soc]
        validation_reports = []

        # Aggregated results
        total_da_revenue = 0.0
        total_afrr_e_revenue = 0.0
        total_as_revenue = 0.0
        total_cyclic_cost = 0.0
        total_calendar_cost = 0.0

        # CRITICAL: Bid aggregation for annual DataFrame (for submission)
        all_bids_list = []  # List of dicts for each timestep
        all_soc_15min = []  # SOC at every 15-min interval

        # Checkpoint settings
        if checkpoint_interval_minutes is not None:
            import pickle
            checkpoint_path = checkpoint_path or './mpc_checkpoint.pkl'
            last_checkpoint_time = datetime.now()
            checkpoint_interval_seconds = checkpoint_interval_minutes * 60
            logger.info("Checkpointing enabled: every %.1f minutes to %s",
                       checkpoint_interval_minutes, checkpoint_path)
        else:
            checkpoint_path = None
            logger.info("Checkpointing disabled")

        # MPC loop
        iteration = 0
        for t_start in range(0, self.total_steps, self.execution_steps):
            iteration += 1

            # Check max iterations (for testing)
            if max_iterations is not None and iteration > max_iterations:
                logger.info("Reached maximum iterations (%d). Stopping.", max_iterations)
                break

            # Define window bounds
            t_end_horizon = min(t_start + self.horizon_steps, self.total_steps)
            t_end_execution = min(t_start + self.execution_steps, self.total_steps)
            actual_execution_length = t_end_execution - t_start

            if actual_execution_length == 0:
                break  # No more data

            # Log progress
            progress_pct = 100 * t_start / self.total_steps
            logger.info("")
            logger.info("─" * 80)
            logger.info("MPC Iteration %d | Progress: %.1f%% | Time step: %d / %d",
                       iteration, progress_pct, t_start, self.total_steps)
            logger.info("  Horizon: [%d, %d) | Execution: [%d, %d)",
                       t_start, t_end_horizon, t_start, t_end_execution)

            # 1. Slice data for this window
            data_window = self.full_data.iloc[t_start:t_end_horizon].copy()
            data_window = data_window.reset_index(drop=True)

            # 2. Get initial segment SOC from current total SOC
            initial_segment_soc = self._get_initial_segment_soc(current_total_soc)
            logger.info("  Initial SOC: %.2f kWh (%.1f%%)",
                       current_total_soc,
                       100 * current_total_soc / self.battery_params['capacity_kwh'])

            # 3. Build and solve model for this horizon
            try:
                # CRITICAL FIX: Set initial SOC in battery_params BEFORE building model
                # This ensures E_soc_init reads the updated value
                # NOTE: optimizer expects FRACTION (0-1), not absolute kWh!
                initial_soc_fraction = current_total_soc / self.battery_params['capacity_kwh']
                self.optimizer.battery_params['initial_soc'] = initial_soc_fraction
                logger.info("  Set battery_params['initial_soc'] = %.4f (%.2f kWh) for this iteration",
                           initial_soc_fraction, current_total_soc)

                # Build model
                model = self.optimizer.build_optimization_model(
                    data_window,
                    self.c_rate,
                    daily_cycle_limit=None
                )

                # Set initial segment SOC values (Model II/III only)
                if hasattr(model, 'e_soc_j'):
                    for j in model.J:
                        # Fix initial value for each segment
                        model.e_soc_j[0, j].setlb(initial_segment_soc[j])
                        model.e_soc_j[0, j].setub(initial_segment_soc[j])

                    # CRITICAL FIX: Initialize binary variables at t=0 to match segment SOC
                    # This prevents conflicts with LIFO constraints
                    if hasattr(model, 'z_segment_active'):
                        for j in model.J:
                            if initial_segment_soc[j] > 1e-6:  # Segment has energy (tolerance: 1 Wh)
                                model.z_segment_active[0, j].fix(1)
                                logger.debug("    Fixed z_segment_active[0,%d] = 1 (SOC=%.2f kWh)",
                                           j, initial_segment_soc[j])
                            else:  # Segment is empty
                                model.z_segment_active[0, j].fix(0)
                                logger.debug("    Fixed z_segment_active[0,%d] = 0 (empty)", j)

                # Solve
                solved_model, solver_results = self.optimizer.solve_model(model, solver_name=self.solver_name)
                solution = self.optimizer.extract_solution(solved_model, solver_results)

                if solution['status'] not in ['optimal', 'feasible']:
                    logger.error("Solver failed at iteration %d: %s",
                               iteration, solution.get('termination_condition'))
                    logger.error("Stopping simulation.")
                    break

                logger.info("  Solve status: %s (%.2f seconds)",
                           solution['status'], solution['solve_time'])

                # Validate constraints (if enabled)
                validation_report = None
                if self.validate_constraints:
                    try:
                        validation_report = validate_solution(model, solution)
                        if not validation_report['summary']['all_passed']:
                            logger.warning(
                                "  Validation: %d violations found",
                                validation_report['summary']['total_violations']
                            )
                        validation_reports.append({
                            'iteration': iteration,
                            't_start': t_start,
                            'report': validation_report,
                        })
                    except Exception as e:
                        logger.warning("  Validation failed: %s", str(e))

            except Exception as e:
                logger.error("Error building/solving model at iteration %d: %s",
                           iteration, str(e))
                logger.error("Stopping simulation.")
                break

            # 4. Extract and record results for execution window
            window_results = self._extract_window_results(
                solution,
                actual_execution_length,
                data_window
            )

            # Aggregate
            total_da_revenue += window_results['da_revenue']
            total_afrr_e_revenue += window_results['afrr_e_revenue']
            total_as_revenue += window_results['as_revenue']
            total_cyclic_cost += window_results['cyclic_cost']
            total_calendar_cost += window_results['calendar_cost']

            logger.info("  Window revenue: %.2f EUR (DA: %.2f, aFRR-E: %.2f, AS: %.2f)",
                       window_results['total_revenue'],
                       window_results['da_revenue'],
                       window_results['afrr_e_revenue'],
                       window_results['as_revenue'])

            if window_results['total_degradation_cost'] > 0:
                logger.info("  Window degradation: %.2f EUR (Cyclic: %.2f, Calendar: %.2f)",
                           window_results['total_degradation_cost'],
                           window_results['cyclic_cost'],
                           window_results['calendar_cost'])

            # Store iteration results
            iteration_results.append({
                'iteration': iteration,
                't_start': t_start,
                't_end_execution': t_end_execution,
                'start_timestep': t_start,
                'end_timestep': t_start + actual_execution_length,
                # Flatten financial data for easy access
                'revenue': window_results['total_revenue'],
                'degradation_cost': window_results['total_degradation_cost'],
                'profit': window_results['total_revenue'] - window_results['total_degradation_cost'],
                'window_results': window_results,  # Keep full results for reference
                'solve_time': solution['solve_time'],
                'validation': validation_report,
                'solution': solution,  # Store full solution for bid extraction
            })

            # CRITICAL: Extract bids from execution window for annual aggregation
            for t_exec in range(actual_execution_length):
                bid_row = {
                    'timestep': t_start + t_exec,
                    # Day-ahead bids (MW)
                    'p_ch': solution['p_ch'].get(t_exec, 0.0),
                    'p_dis': solution['p_dis'].get(t_exec, 0.0),
                    # aFRR energy bids (MW)
                    'p_afrr_pos_e': solution.get('p_afrr_pos_e', {}).get(t_exec, 0.0),
                    'p_afrr_neg_e': solution.get('p_afrr_neg_e', {}).get(t_exec, 0.0),
                    # SOC (kWh)
                    'e_soc': solution['e_soc'].get(t_exec, 0.0) if 'e_soc' in solution else None,
                }
                all_bids_list.append(bid_row)

                # Extract SOC for 15-min total_bids_df
                if 'e_soc' in solution:
                    all_soc_15min.append(solution['e_soc'].get(t_exec, current_total_soc))
                elif hasattr(model, 'e_soc_j'):
                    # Model II/III: sum segments
                    soc_sum = sum(solution.get(f'e_soc_j[{t_exec},{j}]', 0.0) for j in range(1, 11))
                    all_soc_15min.append(soc_sum)

            # Extract capacity bids (block-level, need to map to timesteps)
            # For AS capacity markets, bids are per 4-hour block
            # CRITICAL FIX: Use actual block_id from data, not local index
            for t_exec in range(actual_execution_length):
                block_id = int(data_window['block_id'].iloc[t_exec])
                all_bids_list[-actual_execution_length + t_exec]['c_fcr'] = solution['c_fcr'].get(block_id, 0.0)
                all_bids_list[-actual_execution_length + t_exec]['c_afrr_pos'] = solution['c_afrr_pos'].get(block_id, 0.0)
                all_bids_list[-actual_execution_length + t_exec]['c_afrr_neg'] = solution['c_afrr_neg'].get(block_id, 0.0)

            # 5. Update state for next iteration
            # Get final SOC from the last executed timestep
            last_execution_step = actual_execution_length - 1

            if hasattr(model, 'e_soc_j'):
                # Model II/III: Sum segment SOCs
                current_total_soc = sum(
                    pyo.value(model.e_soc_j[last_execution_step, j])
                    for j in model.J
                )
            else:
                # Model I: Use total SOC
                current_total_soc = pyo.value(model.e_soc[last_execution_step])

            all_soc_total_bids_df.append(current_total_soc)

            logger.info("  Final SOC: %.2f kWh (%.1f%%)",
                       current_total_soc,
                       100 * current_total_soc / self.battery_params['capacity_kwh'])

            # Checkpoint saving (if enabled)
            if checkpoint_interval_minutes is not None:
                current_time = datetime.now()
                elapsed_since_checkpoint = (current_time - last_checkpoint_time).total_seconds()

                if elapsed_since_checkpoint >= checkpoint_interval_seconds:
                    logger.info("  [CHECKPOINT] Saving intermediate results (%.1f min since last save)...",
                               elapsed_since_checkpoint / 60)

                    # Build checkpoint data
                    checkpoint_data = {
                        'iteration': iteration,
                        'current_total_soc': current_total_soc,
                        'iteration_results': iteration_results.copy(),
                        'all_soc_total_bids_df': all_soc_total_bids_df.copy(),
                        'all_bids_list': all_bids_list.copy(),
                        'all_soc_15min': all_soc_15min.copy(),
                        'total_da_revenue': total_da_revenue,
                        'total_afrr_e_revenue': total_afrr_e_revenue,
                        'total_as_revenue': total_as_revenue,
                        'total_cyclic_cost': total_cyclic_cost,
                        'total_calendar_cost': total_calendar_cost,
                        'timestamp': current_time.isoformat(),
                    }

                    # Save checkpoint
                    import pickle
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(checkpoint_data, f)

                    last_checkpoint_time = current_time
                    logger.info("  [CHECKPOINT] Saved at iteration %d", iteration)

        # Calculate final results
        total_revenue = total_da_revenue + total_afrr_e_revenue + total_as_revenue
        total_degradation_cost = total_cyclic_cost + total_calendar_cost
        net_profit = total_revenue - total_degradation_cost

        # CRITICAL: Create annual bid DataFrame for submission
        import pandas as pd
        total_bids_df = pd.DataFrame(all_bids_list)

        logger.info("")
        logger.info("=" * 80)
        logger.info("MPC SIMULATION COMPLETE")
        logger.info("=" * 80)
        logger.info("  Iterations completed: %d", len(iteration_results))
        logger.info("  Total revenue: %.2f EUR", total_revenue)
        logger.info("    - DA energy: %.2f EUR", total_da_revenue)
        logger.info("    - aFRR energy: %.2f EUR", total_afrr_e_revenue)
        logger.info("    - AS capacity: %.2f EUR", total_as_revenue)
        logger.info("  Total degradation: %.2f EUR", total_degradation_cost)
        logger.info("    - Cyclic aging: %.2f EUR", total_cyclic_cost)
        logger.info("    - Calendar aging: %.2f EUR", total_calendar_cost)
        logger.info("  Net profit: %.2f EUR", net_profit)
        logger.info("  Final SOC: %.2f kWh (%.1f%%)",
                   current_total_soc,
                   100 * current_total_soc / self.battery_params['capacity_kwh'])
        logger.info("  Annual bids DataFrame: %d rows x %d columns",
                   len(total_bids_df), len(total_bids_df.columns))
        logger.info("=" * 80)

        return {
            # Financial totals (for ROI calculation)
            'total_revenue': total_revenue,
            'da_revenue': total_da_revenue,
            'afrr_e_revenue': total_afrr_e_revenue,
            'as_revenue': total_as_revenue,
            'total_degradation_cost': total_degradation_cost,
            'cyclic_cost': total_cyclic_cost,
            'calendar_cost': total_calendar_cost,
            'net_profit': net_profit,

            # State total_bids_df
            'final_soc': current_total_soc,
            'soc_total_bids_df': all_soc_total_bids_df,  # Iteration boundaries
            'soc_15min': all_soc_15min,  # Every 15-min interval

            # CRITICAL: Annual bid DataFrame for submission
            'total_bids_df': total_bids_df,

            # Detailed results
            'iteration_results': iteration_results,
            'validation_reports': validation_reports if self.validate_constraints else None,
            'summary': {
                'iterations': len(iteration_results),
                'alpha': self.degradation_params.get('alpha', 'N/A'),
                'c_rate': self.c_rate,
                'horizon_hours': self.horizon_hours,
                'execution_hours': self.execution_hours,
                'total_timesteps': len(total_bids_df),
            },
        }
