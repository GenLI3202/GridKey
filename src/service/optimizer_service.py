"""
Optimizer Service Module
========================

This module provides the unified service layer wrapper for the BESS Optimizer,
orchestrating the complete optimization workflow:
1. Data validation and adaptation
2. Model selection and construction
3. Solving and result extraction
4. Result formatting for API/Agent consumption

Usage:
    service = OptimizerService()
    result = service.optimize(
        market_prices=price_service.get_market_prices("DE_LU", 48),
        generation_forecast=weather_service.get_generation_forecast("Munich", 48),
        model_type="III",
        c_rate=0.5,
        alpha=1.0
    )
"""

from typing import Optional, Dict, Any
import logging

from .models import OptimizationInput, OptimizationResult, ModelType, ScheduleEntry, RenewableUtilization
from .adapter import DataAdapter

logger = logging.getLogger(__name__)


class OptimizerService:
    """
    Unified Optimizer Service interface (Section 6.5 of Blueprint).

    Orchestrates the complete optimization workflow:
    1. Data validation and adaptation
    2. Model selection and construction
    3. Solving and result extraction
    4. Result formatting for API/Agent consumption

    Usage:
        service = OptimizerService()
        result = service.optimize(
            market_prices=price_service.get_market_prices("DE_LU", 48),
            generation_forecast=weather_service.get_generation_forecast("Munich", 48),
            model_type="III",
            c_rate=0.5,
            alpha=1.0
        )
    """

    def __init__(self):
        self.adapter = DataAdapter()
        self._optimizer_cache: Dict[str, Any] = {}

    def optimize(
        self,
        market_prices: dict,
        generation_forecast: Optional[dict] = None,
        model_type: str = "III",
        c_rate: float = 0.5,
        alpha: float = 1.0,
        daily_cycle_limit: float = 1.0,
        time_horizon_hours: int = 48,
    ) -> OptimizationResult:
        """
        Run complete optimization and return structured result.

        Args:
            market_prices: Market price data from Price Service
            generation_forecast: Renewable forecast from Weather Service (optional)
            model_type: "I", "II", "III", or "III-renew"
            c_rate: Battery C-rate (0.25, 0.33, 0.5)
            alpha: Degradation cost weight
            daily_cycle_limit: Maximum daily cycles (default 1.0)
            time_horizon_hours: Optimization horizon

        Returns:
            OptimizationResult with schedule, metrics, and metadata

        Raises:
            ValueError: If input validation fails
            RuntimeError: If solver fails or times out
        """
        logger.info(f"Starting optimization: model={model_type}, c_rate={c_rate}, alpha={alpha}")

        # 1. Load battery config
        battery_config = self._load_battery_config()
        battery_config["c_rate"] = c_rate

        # 2. Adapt input data
        opt_input = self.adapter.adapt(
            market_prices=market_prices,
            generation_forecast=generation_forecast,
            battery_config=battery_config,
            time_horizon_hours=time_horizon_hours,
        )
        opt_input.model_type = ModelType(model_type)
        opt_input.alpha = alpha

        # 3. Get or create optimizer
        optimizer = self._get_optimizer(model_type, alpha)

        # 4. Convert to legacy format
        country_data = self.adapter.to_country_data(opt_input)

        # 5. Build and solve model
        model = optimizer.build_optimization_model(country_data, c_rate, daily_cycle_limit)
        model, solver_results = optimizer.solve_model(model)

        # 6. Extract solution
        solution = optimizer.extract_solution(model, solver_results)

        # 7. Convert to OptimizationResult
        return self._build_result(solution, opt_input, solver_results)

    def optimize_from_input(self, opt_input: OptimizationInput) -> OptimizationResult:
        """
        Run optimization from a pre-built OptimizationInput.
        Useful for API endpoints that receive JSON input.
        """
        optimizer = self._get_optimizer(opt_input.model_type.value, opt_input.alpha)
        country_data = self.adapter.to_country_data(opt_input)

        model = optimizer.build_optimization_model(
            country_data, opt_input.c_rate, daily_cycle_limit=1.0
        )
        model, solver_results = optimizer.solve_model(model)
        solution = optimizer.extract_solution(model, solver_results)

        return self._build_result(solution, opt_input, solver_results)

    def optimize_12h_mpc(
        self,
        market_prices: dict,
        generation_forecast: Optional[dict] = None,
        model_type: str = "III",
        c_rate: float = 0.5,
        alpha: float = 1.0,
        horizon_hours: int = 6,
        execution_hours: int = 4,
    ) -> OptimizationResult:
        """
        Solve 12h problem using MPC rolling horizon.

        Uses MPCRollingHorizon to break 12h into 6h optimization windows,
        committing 4h at a time. Total of 3 iterations.

        Args:
            market_prices: Market price data (12h: 48 values @ 15-min)
            generation_forecast: Renewable forecast (optional)
            model_type: "I", "II", "III", or "III-renew"
            c_rate: Battery C-rate
            alpha: Degradation cost weight
            horizon_hours: MPC optimization window (default 6h)
            execution_hours: Commit execution window (default 4h)

        Returns:
            OptimizationResult with 12h complete schedule

        Estimated time: ~15-20 seconds (3 iterations × ~5 sec)
        """
        from .mpc import MPCRollingHorizon

        logger.info(
            "Starting MPC 12h optimization: model=%s, c_rate=%s, alpha=%s",
            model_type, c_rate, alpha
        )

        # 1. Adapt input to 12h OptimizationInput
        opt_input_12h = self.adapter.adapt(
            market_prices=market_prices,
            generation_forecast=generation_forecast,
            battery_config=self._load_battery_config(),
            time_horizon_hours=12,
        )
        opt_input_12h.model_type = ModelType(model_type)
        opt_input_12h.alpha = alpha

        # 2. Get optimizer
        optimizer = self._get_optimizer(model_type, alpha)

        # 3. Create MPC helper
        mpc = MPCRollingHorizon(
            optimizer=optimizer,
            adapter=self.adapter,
            horizon_hours=horizon_hours,
            execution_hours=execution_hours,
        )

        # 4. Run MPC
        solution = mpc.solve_12h(opt_input_12h, c_rate)

        # 5. Build result
        return self._build_result(solution, opt_input_12h, None)

    def _get_optimizer(self, model_type: str, alpha: float):
        """Get or create optimizer instance."""
        from src.core.optimizer import (
            BESSOptimizerModelI,
            BESSOptimizerModelII,
            BESSOptimizerModelIII,
            BESSOptimizerModelIIIRenew,
        )

        cache_key = f"{model_type}_{alpha}"
        if cache_key not in self._optimizer_cache:
            if model_type == "I":
                self._optimizer_cache[cache_key] = BESSOptimizerModelI()
            elif model_type == "II":
                self._optimizer_cache[cache_key] = BESSOptimizerModelII(alpha=alpha)
            elif model_type == "III":
                self._optimizer_cache[cache_key] = BESSOptimizerModelIII(alpha=alpha)
            elif model_type == "III-renew":
                self._optimizer_cache[cache_key] = BESSOptimizerModelIIIRenew(alpha=alpha)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        return self._optimizer_cache[cache_key]

    def _load_battery_config(self) -> dict:
        """Load battery configuration (default values per GEMINI.md)."""
        return {
            "capacity_kwh": 4472,
            "c_rate": 0.5,
            "efficiency": 0.95,
            "initial_soc": 0.5,
        }

    def _build_result(
        self,
        solution: Dict[str, Any],
        opt_input: OptimizationInput,
        solver_results,
    ) -> OptimizationResult:
        """Convert internal solution dict to OptimizationResult."""
        from datetime import datetime, timedelta

        # Handle error/failed status
        if solution.get('status') in ('error', 'failed'):
            # Return minimal result for failed optimizations
            return OptimizationResult(
                objective_value=0.0,
                net_profit=0.0,
                revenue_breakdown={},
                degradation_cost=0.0,
                cyclic_aging_cost=0.0,
                calendar_aging_cost=0.0,
                schedule=[],
                soc_trajectory=[],
                solve_time_seconds=solution.get('solve_time', 0.0),
                solver_name=solution.get('solver', 'unknown'),
                model_type=opt_input.model_type,
                status=solution.get('status', 'failed'),
            )

        # Build schedule entries
        schedule = []
        n_timesteps = opt_input.time_horizon_hours * 4  # 15-min resolution
        base_time = datetime(2024, 1, 1, 0, 0)

        for t in range(n_timesteps):
            timestamp = base_time + timedelta(minutes=15 * t)
            p_ch = solution.get('p_ch', {}).get(t, 0.0)
            p_dis = solution.get('p_dis', {}).get(t, 0.0)
            soc = solution.get('e_soc', {}).get(t, 0.5)

            # Determine action and market
            if p_dis > 0.001:
                action = "discharge"
                power = p_dis
            elif p_ch > 0.001:
                action = "charge"
                power = p_ch
            else:
                action = "idle"
                power = 0.0

            entry = ScheduleEntry(
                timestamp=timestamp,
                action=action,
                power_kw=power,
                market="da",  # Simplified; could be enhanced
                soc_after=max(0.0, min(1.0, soc / opt_input.battery_capacity_kwh)),
            )

            # Add renewable fields if present
            if 'p_renewable_self' in solution:
                entry.renewable_action = "self_consume"
                entry.renewable_power_kw = solution.get('p_renewable_self', {}).get(t, 0.0)

            schedule.append(entry)

        # Build SOC trajectory (normalized to [0, 1])
        soc_trajectory = [
            max(0.0, min(1.0, solution.get('e_soc', {}).get(t, 0.5) / opt_input.battery_capacity_kwh))
            for t in range(n_timesteps)
        ]

        # Revenue breakdown
        revenue_breakdown = {
            'da': solution.get('profit_da', 0.0),
            'afrr_energy': solution.get('profit_afrr_energy', 0.0),
            'fcr': solution.get('profit_as_capacity', 0.0),
        }
        if 'profit_renewable_export' in solution:
            revenue_breakdown['renewable_export'] = solution['profit_renewable_export']

        # Degradation costs
        cyclic_cost = solution.get('cost_cyclic', 0.0)
        calendar_cost = solution.get('cost_calendar', 0.0)
        degradation_cost = cyclic_cost + calendar_cost

        # Renewable utilization (if applicable)
        renewable_util = None
        if 'renewable_utilization' in solution:
            ru = solution['renewable_utilization']
            renewable_util = RenewableUtilization(
                total_generation_kwh=ru.get('total_generation_kwh', 0.0),
                self_consumption_kwh=ru.get('self_consumption_kwh', 0.0),
                export_kwh=ru.get('export_kwh', 0.0),
                curtailment_kwh=ru.get('curtailment_kwh', 0.0),
                utilization_rate=ru.get('utilization_rate', 0.0),
            )

        return OptimizationResult(
            objective_value=solution.get('objective_value', 0.0),
            net_profit=solution.get('objective_value', 0.0) - degradation_cost,
            revenue_breakdown=revenue_breakdown,
            degradation_cost=degradation_cost,
            cyclic_aging_cost=cyclic_cost,
            calendar_aging_cost=calendar_cost,
            schedule=schedule,
            soc_trajectory=soc_trajectory,
            renewable_utilization=renewable_util,
            solve_time_seconds=solution.get('solve_time', 0.0),
            solver_name=solution.get('solver', 'unknown'),
            model_type=opt_input.model_type,
            status=solution.get('status', 'optimal'),
        )
