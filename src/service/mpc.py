"""
MPC rolling horizon helper for OptimizerService.

Reuses patterns from src/mpc/mpc_simulator:
- _get_initial_segment_soc(): SOC segment distribution
- SOC state management between windows
- Binary variable fixing (LIFO constraints)

But adapted for:
- OptimizationInput slices (not DataFrame)
- Service layer integration
- 12h demo scope
"""

from typing import Dict, List, Any, Optional
import logging
from .models import OptimizationInput
from .adapter import DataAdapter

logger = logging.getLogger(__name__)


class MPCRollingHorizon:
    """
    MPC rolling horizon optimization (12h demo).

    Args:
        optimizer: BESSOptimizerModelIII instance
        adapter: DataAdapter instance
        horizon_hours: Optimization window size (default 6h)
        execution_hours: Commit execution window (default 4h)
    """

    def __init__(
        self,
        optimizer,
        adapter: DataAdapter,
        horizon_hours: int = 6,
        execution_hours: int = 4,
    ):
        self.optimizer = optimizer
        self.adapter = adapter
        self.horizon_hours = horizon_hours
        self.execution_hours = execution_hours

        # Get parameters from optimizer
        self.time_step_hours = self.optimizer.market_params['time_step_hours']
        self.battery_params = self.optimizer.battery_params
        self.degradation_params = getattr(self.optimizer, 'degradation_params', {})

        # Segment parameters (Model II/III)
        self.num_segments = self.degradation_params.get('num_segments', 10)
        self.segment_capacity = self.degradation_params.get(
            'segment_capacity_kwh',
            self.battery_params['capacity_kwh'] / 10
        )

        # Calculate step counts
        self.horizon_steps = int(horizon_hours / self.time_step_hours)  # 24 for 6h
        self.execution_steps = int(execution_hours / self.time_step_hours)  # 16 for 4h

        logger.info(
            "MPCRollingHorizon initialized: horizon=%dh, execution=%dh, "
            "horizon_steps=%d, execution_steps=%d",
            horizon_hours, execution_hours, self.horizon_steps, self.execution_steps
        )

    def _get_initial_segment_soc(self, total_soc_kwh: float) -> Dict[int, float]:
        """
        Convert total SOC to segment-wise distribution (shallow to deep filling).

        Args:
            total_soc_kwh: Total energy in battery (kWh)

        Returns:
            Dictionary mapping segment index j to energy in segment (kWh)
        """
        segment_soc = {}
        remaining_soc = total_soc_kwh

        # Fill from segment 1 (shallowest) to segment J (deepest)
        for j in range(1, self.num_segments + 1):
            energy = min(remaining_soc, self.segment_capacity)
            segment_soc[j] = energy
            remaining_soc -= energy

        return segment_soc

    def solve_12h(
        self,
        opt_input_12h: OptimizationInput,
        c_rate: float,
    ) -> Dict[str, Any]:
        """
        Execute 12h MPC rolling horizon optimization.

        Strategy:
        - Iteration 1: Optimize [0h-6h], commit [0h-4h]
        - Iteration 2: Optimize [4h-10h], commit [4h-8h]
        - Iteration 3: Optimize [8h-12h], commit [8h-12h]

        Args:
            opt_input_12h: 12-hour OptimizationInput (48 timesteps)
            c_rate: Battery C-rate

        Returns:
            Aggregated solution dict with 12h schedule
        """
        total_timesteps = 12 * 4  # 48
        execution_timesteps = self.execution_steps  # 16

        # Initialize state
        current_total_soc = (
            opt_input_12h.initial_soc * self.battery_params['capacity_kwh']
        )

        logger.info(
            "MPC 12h starting: initial_soc=%.2f kWh (%.1f%%)",
            current_total_soc,
            100 * current_total_soc / self.battery_params['capacity_kwh']
        )

        # Aggregated results storage
        all_solutions = {
            'p_ch': {},
            'p_dis': {},
            'p_afrr_pos_e': {},
            'p_afrr_neg_e': {},
            'c_fcr': {},
            'c_afrr_pos': {},
            'c_afrr_neg': {},
            'e_soc': {},
            'profit_da': 0.0,
            'profit_afrr_energy': 0.0,
            'profit_as_capacity': 0.0,
            'cost_cyclic': 0.0,
            'cost_calendar': 0.0,
            'solve_time': 0.0,
            'solver': 'unknown',
            'status': 'optimal',
        }

        # Renewable energy
        has_renewable = opt_input_12h.renewable_generation is not None
        if has_renewable:
            all_solutions.update({
                'p_renewable_self': {},
                'p_renewable_export': {},
                'p_renewable_curtail': {}
            })

        # MPC main loop: 3 iterations (0h, 4h, 8h)
        iteration = 0
        for t_start in range(0, total_timesteps, execution_timesteps):
            iteration += 1
            t_end_horizon = min(t_start + self.horizon_steps, total_timesteps)
            t_end_execute = min(t_start + execution_timesteps, total_timesteps)

            logger.info(
                "MPC Iteration %d: Window [%.1fh - %.1fh], Execute [%.1fh - %.1fh]",
                iteration,
                t_start / 4, t_end_horizon / 4,
                t_start / 4, t_end_execute / 4
            )

            # 1. Extract window data
            opt_input_window = self._slice_opt_input(
                opt_input_12h, t_start, t_end_horizon
            )

            # 2. Set initial SOC
            initial_soc_fraction = current_total_soc / self.battery_params['capacity_kwh']
            self.optimizer.battery_params['initial_soc'] = initial_soc_fraction

            # 3. Convert to DataFrame and solve
            country_data = self.adapter.to_country_data(opt_input_window)
            model = self.optimizer.build_optimization_model(
                country_data, c_rate, daily_cycle_limit=None
            )

            # 4. Fix initial segment SOC (Model II/III)
            if hasattr(model, 'e_soc_j'):
                initial_segment_soc = self._get_initial_segment_soc(current_total_soc)
                for j in model.J:
                    model.e_soc_j[0, j].setlb(initial_segment_soc[j])
                    model.e_soc_j[0, j].setub(initial_segment_soc[j])

                # Fix binary variables to match segment SOC
                if hasattr(model, 'z_segment_active'):
                    for j in model.J:
                        if initial_segment_soc[j] > 1e-6:  # Segment has energy
                            model.z_segment_active[0, j].fix(1)
                        else:  # Segment is empty
                            model.z_segment_active[0, j].fix(0)

            # 5. Solve
            model, solver_results = self.optimizer.solve_model(model)
            solution = self.optimizer.extract_solution(model, solver_results)

            if solution['status'] not in ['optimal', 'feasible']:
                logger.error(
                    "MPC Iteration %d failed: %s",
                    iteration, solution['status']
                )
                all_solutions['status'] = solution['status']
                break

            logger.info(
                "MPC Iteration %d success: %s, solve_time=%.2fs",
                iteration, solution['status'], solution.get('solve_time', 0)
            )

            # 6. Commit execution window results
            execute_length = t_end_execute - t_start
            for t_rel in range(execute_length):
                t_abs = t_start + t_rel

                all_solutions['p_ch'][t_abs] = solution['p_ch'].get(t_rel, 0.0)
                all_solutions['p_dis'][t_abs] = solution['p_dis'].get(t_rel, 0.0)
                all_solutions['p_afrr_pos_e'][t_abs] = solution.get('p_afrr_pos_e', {}).get(t_rel, 0.0)
                all_solutions['p_afrr_neg_e'][t_abs] = solution.get('p_afrr_neg_e', {}).get(t_rel, 0.0)

                # Get SOC from solution
                soc_value = solution['e_soc'].get(t_rel, current_total_soc)
                all_solutions['e_soc'][t_abs] = soc_value

                # Capacity bids (block-based)
                # Map relative timestep to block
                rel_block = t_rel // 16
                all_solutions['c_fcr'][t_abs] = solution.get('c_fcr', {}).get(rel_block, 0.0)
                all_solutions['c_afrr_pos'][t_abs] = solution.get('c_afrr_pos', {}).get(rel_block, 0.0)
                all_solutions['c_afrr_neg'][t_abs] = solution.get('c_afrr_neg', {}).get(rel_block, 0.0)

                # Renewable
                if has_renewable:
                    all_solutions['p_renewable_self'][t_abs] = solution.get('p_renewable_self', {}).get(t_rel, 0.0)
                    all_solutions['p_renewable_export'][t_abs] = solution.get('p_renewable_export', {}).get(t_rel, 0.0)
                    all_solutions['p_renewable_curtail'][t_abs] = solution.get('p_renewable_curtail', {}).get(t_rel, 0.0)

            # 7. Update SOC (use last timestep of execution window)
            current_total_soc = all_solutions['e_soc'][t_end_execute - 1]

            logger.info(
                "MPC Iteration %d: final_soc=%.2f kWh (%.1f%%), revenue=%.2f EUR",
                iteration,
                current_total_soc,
                100 * current_total_soc / self.battery_params['capacity_kwh'],
                solution.get('profit_da', 0.0) + solution.get('profit_afrr_energy', 0.0) + solution.get('profit_as_capacity', 0.0)
            )

            # 8. Aggregate metrics (scale by execution window ratio)
            scale = execute_length / len(solution.get('p_ch', {}))
            all_solutions['profit_da'] += solution.get('profit_da', 0.0) * scale
            all_solutions['profit_afrr_energy'] += solution.get('profit_afrr_energy', 0.0) * scale
            all_solutions['profit_as_capacity'] += solution.get('profit_as_capacity', 0.0) * scale
            all_solutions['cost_cyclic'] += solution.get('cost_cyclic', 0.0) * scale
            all_solutions['cost_calendar'] += solution.get('cost_calendar', 0.0) * scale
            all_solutions['solve_time'] += solution.get('solve_time', 0.0)

        # Set solver name
        all_solutions['solver'] = getattr(solver_results, 'solver', {}).get('name', 'highs')

        logger.info(
            "MPC 12h complete: status=%s, total_time=%.2fs, total_profit=%.2f EUR",
            all_solutions['status'],
            all_solutions['solve_time'],
            all_solutions['profit_da'] + all_solutions['profit_afrr_energy'] + all_solutions['profit_as_capacity']
        )

        return all_solutions

    def _slice_opt_input(
        self, opt_input: OptimizationInput, t_start: int, t_end: int
    ) -> OptimizationInput:
        """
        Slice OptimizationInput to window [t_start, t_end).

        Args:
            opt_input: Full OptimizationInput (12h)
            t_start: Start timestep (15-min based)
            t_end: End timestep (15-min based)

        Returns:
            Window-specific OptimizationInput
        """
        # 15-min prices
        da_prices = opt_input.da_prices[t_start:t_end]
        afrr_ep = opt_input.afrr_energy_pos[t_start:t_end]
        afrr_en = opt_input.afrr_energy_neg[t_start:t_end]

        # 4-h block prices (need to include complete blocks)
        block_start = t_start // 16
        block_end = (t_end + 15) // 16  # Round up to ensure coverage
        fcr_prices = opt_input.fcr_prices[block_start:block_end]
        afrr_cap_pos = opt_input.afrr_capacity_pos[block_start:block_end]
        afrr_cap_neg = opt_input.afrr_capacity_neg[block_start:block_end]

        # Renewable
        renewable = None
        if opt_input.renewable_generation:
            renewable = opt_input.renewable_generation[t_start:t_end]

        return OptimizationInput(
            time_horizon_hours=(t_end - t_start) / 4,
            da_prices=da_prices,
            afrr_energy_pos=afrr_ep,
            afrr_energy_neg=afrr_en,
            fcr_prices=fcr_prices,
            afrr_capacity_pos=afrr_cap_pos,
            afrr_capacity_neg=afrr_cap_neg,
            renewable_generation=renewable,
            battery_capacity_kwh=opt_input.battery_capacity_kwh,
            c_rate=opt_input.c_rate,
            efficiency=opt_input.efficiency,
            initial_soc=opt_input.initial_soc,
            model_type=opt_input.model_type,
            alpha=opt_input.alpha,
        )
