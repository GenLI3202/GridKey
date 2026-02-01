### Phase 3: Service Layer (Sequential)

This component is the **critical path** — it depends on both Phase 1 and Phase 2.

---

#### 3.1 [NEW] optimizer_service.py — OptimizerService Class

Location: `src/service/optimizer_service.py`

**Depends on:** `models.py` (1.1), `adapter.py` (2.2), `BESSOptimizerModelIIIRenew` (2.1)

##### File Structure

```
src/
├── core/
│   └── optimizer.py              # Existing (extend with renewables)
└── service/                      # NEW directory
    ├── __init__.py
    ├── models.py                 # Data models (Phase 1)
    ├── adapter.py                # DataAdapter (Phase 2)
    └── optimizer_service.py      # OptimizerService (Phase 3)
```

```python
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
                soc_after=soc / opt_input.battery_capacity_kwh,
            )

            # Add renewable fields if present
            if 'p_renewable_self' in solution:
                entry.renewable_action = "self_consume"
                entry.renewable_power_kw = solution.get('p_renewable_self', {}).get(t, 0.0)

            schedule.append(entry)

        # Build SOC trajectory (normalized to [0, 1])
        soc_trajectory = [
            solution.get('e_soc', {}).get(t, 0.5) / opt_input.battery_capacity_kwh
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
```

---

#### 3.2 Key Implementation Notes

> [!IMPORTANT]
> **API Consistency:** The `build_optimization_model()` method requires three parameters: `country_data`, `c_rate`, and `daily_cycle_limit`. The `solve_model()` method returns a **tuple** `(model, results)`.

> [!NOTE]
> **Model Selection:** Use `BESSOptimizerModelIIIRenew` for `"III-renew"` model type. This class already exists in `src/core/optimizer.py` (lines 2397-2681).

---

#### 3.3 Test File: `test_optimizer_service.py`

Location: `src/test/test_optimizer_service.py`

```python
"""
Tests for src/service/optimizer_service.py — OptimizerService workflow.
"""

import pytest
from unittest.mock import Mock, patch

from src.service.optimizer_service import OptimizerService
from src.service.models import ModelType, OptimizationInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service():
    return OptimizerService()


@pytest.fixture
def sample_market_prices():
    """192 entries = 48h at 15-min resolution, 12 blocks for 4-hour prices."""
    return {
        'day_ahead': [50.0 + i * 0.1 for i in range(192)],
        'afrr_energy_pos': [40.0] * 192,
        'afrr_energy_neg': [30.0] * 192,
        'fcr': [100.0] * 12,
        'afrr_capacity_pos': [5.0] * 12,
        'afrr_capacity_neg': [10.0] * 12,
    }


@pytest.fixture
def sample_generation_forecast():
    """Synthetic renewable generation (15-min resolution)."""
    return {
        'generation_kw': [500.0] * 192,  # Constant 500 kW
    }


# ---------------------------------------------------------------------------
# Test: Service Initialization
# ---------------------------------------------------------------------------

class TestServiceInit:
    def test_creates_adapter(self, service):
        assert service.adapter is not None

    def test_empty_cache(self, service):
        assert service._optimizer_cache == {}


# ---------------------------------------------------------------------------
# Test: Model Factory (_get_optimizer)
# ---------------------------------------------------------------------------

class TestGetOptimizer:
    def test_model_i(self, service):
        opt = service._get_optimizer("I", 1.0)
        assert opt.__class__.__name__ == "BESSOptimizerModelI"

    def test_model_ii(self, service):
        opt = service._get_optimizer("II", 0.5)
        assert opt.__class__.__name__ == "BESSOptimizerModelII"

    def test_model_iii(self, service):
        opt = service._get_optimizer("III", 1.0)
        assert opt.__class__.__name__ == "BESSOptimizerModelIII"

    def test_model_iii_renew(self, service):
        opt = service._get_optimizer("III-renew", 1.0)
        assert opt.__class__.__name__ == "BESSOptimizerModelIIIRenew"

    def test_unknown_model_raises(self, service):
        with pytest.raises(ValueError, match="Unknown model type"):
            service._get_optimizer("IV", 1.0)

    def test_caching(self, service):
        opt1 = service._get_optimizer("III", 1.0)
        opt2 = service._get_optimizer("III", 1.0)
        assert opt1 is opt2

    def test_different_alpha_different_instance(self, service):
        opt1 = service._get_optimizer("III", 1.0)
        opt2 = service._get_optimizer("III", 0.5)
        assert opt1 is not opt2


# ---------------------------------------------------------------------------
# Test: Battery Config
# ---------------------------------------------------------------------------

class TestBatteryConfig:
    def test_default_values(self, service):
        config = service._load_battery_config()
        assert config['capacity_kwh'] == 4472
        assert config['c_rate'] == 0.5
        assert config['efficiency'] == 0.95
        assert config['initial_soc'] == 0.5


# ---------------------------------------------------------------------------
# Test: End-to-End (Mocked Solver)
# ---------------------------------------------------------------------------

class TestOptimizeEndToEnd:
    @patch.object(OptimizerService, '_get_optimizer')
    def test_calls_optimizer_pipeline(self, mock_get_opt, service, sample_market_prices):
        # Setup mock optimizer
        mock_optimizer = Mock()
        mock_model = Mock()
        mock_results = Mock()
        mock_results._solve_time = 1.5
        mock_results._solver_name = 'mock'
        mock_results.solver.termination_condition.name = 'optimal'

        mock_optimizer.build_optimization_model.return_value = mock_model
        mock_optimizer.solve_model.return_value = (mock_model, mock_results)
        mock_optimizer.extract_solution.return_value = {
            'status': 'optimal',
            'objective_value': 100.0,
            'solve_time': 1.5,
            'solver': 'mock',
            'profit_da': 80.0,
            'profit_afrr_energy': 20.0,
            'cost_cyclic': 5.0,
            'e_soc': {t: 2236.0 for t in range(192)},
            'p_ch': {t: 0.0 for t in range(192)},
            'p_dis': {t: 0.0 for t in range(192)},
        }
        mock_get_opt.return_value = mock_optimizer

        result = service.optimize(sample_market_prices)

        assert result.status == 'optimal'
        assert result.objective_value == 100.0
        mock_optimizer.build_optimization_model.assert_called_once()
        mock_optimizer.solve_model.assert_called_once()
        mock_optimizer.extract_solution.assert_called_once()
```

---
