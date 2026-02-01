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

    @patch.object(OptimizerService, '_get_optimizer')
    def test_optimize_from_input(self, mock_get_opt, service):
        # Setup mock optimizer
        mock_optimizer = Mock()
        mock_model = Mock()
        mock_results = Mock()
        mock_results.solver.termination_condition.name = 'optimal'

        mock_optimizer.build_optimization_model.return_value = mock_model
        mock_optimizer.solve_model.return_value = (mock_model, mock_results)
        mock_optimizer.extract_solution.return_value = {
            'status': 'optimal',
            'objective_value': 50.0,
            'solve_time': 1.0,
            'solver': 'test',
            'profit_da': 50.0,
            'cost_cyclic': 2.0,
            'cost_calendar': 1.0,
            'e_soc': {t: 2236.0 for t in range(192)},
            'p_ch': {t: 0.0 for t in range(192)},
            'p_dis': {t: 0.0 for t in range(192)},
        }
        mock_get_opt.return_value = mock_optimizer

        # Create OptimizationInput
        opt_input = OptimizationInput(
            time_horizon_hours=48,
            da_prices=[50.0] * 192,
            afrr_energy_pos=[40.0] * 192,
            afrr_energy_neg=[30.0] * 192,
            fcr_prices=[100.0] * 12,
            afrr_capacity_pos=[5.0] * 12,
            afrr_capacity_neg=[10.0] * 12,
            model_type=ModelType.MODEL_III,
        )

        result = service.optimize_from_input(opt_input)

        assert result.status == 'optimal'
        assert result.objective_value == 50.0
        assert result.net_profit == 47.0  # 50 - 2 - 1

    @patch.object(OptimizerService, '_get_optimizer')
    def test_handles_failed_optimization(self, mock_get_opt, service, sample_market_prices):
        # Setup mock optimizer that fails
        mock_optimizer = Mock()
        mock_model = Mock()
        mock_results = Mock()

        mock_optimizer.build_optimization_model.return_value = mock_model
        mock_optimizer.solve_model.return_value = (mock_model, mock_results)
        mock_optimizer.extract_solution.return_value = {
            'status': 'error',
            'solve_time': 0.1,
            'solver': 'test',
        }
        mock_get_opt.return_value = mock_optimizer

        result = service.optimize(sample_market_prices)

        assert result.status == 'error'
        assert result.objective_value == 0.0
        assert result.net_profit == 0.0

    @patch.object(OptimizerService, '_get_optimizer')
    def test_build_result_with_renewable_utilization(self, mock_get_opt, service, sample_market_prices):
        # Setup mock optimizer with renewable output
        mock_optimizer = Mock()
        mock_model = Mock()
        mock_results = Mock()

        mock_optimizer.build_optimization_model.return_value = mock_model
        mock_optimizer.solve_model.return_value = (mock_model, mock_results)
        mock_optimizer.extract_solution.return_value = {
            'status': 'optimal',
            'objective_value': 150.0,
            'solve_time': 2.0,
            'solver': 'test',
            'profit_da': 100.0,
            'profit_afrr_energy': 20.0,
            'profit_renewable_export': 30.0,
            'cost_cyclic': 5.0,
            'cost_calendar': 2.0,
            'e_soc': {t: 2236.0 for t in range(192)},
            'p_ch': {t: 0.0 for t in range(192)},
            'p_dis': {t: 0.0 for t in range(192)},
            'renewable_utilization': {
                'total_generation_kwh': 1000.0,
                'self_consumption_kwh': 700.0,
                'export_kwh': 200.0,
                'curtailment_kwh': 100.0,
                'utilization_rate': 0.9,
            },
        }
        mock_get_opt.return_value = mock_optimizer

        result = service.optimize(sample_market_prices)

        assert result.renewable_utilization is not None
        assert result.renewable_utilization.total_generation_kwh == 1000.0
        assert result.renewable_utilization.utilization_rate == 0.9
        assert 'renewable_export' in result.revenue_breakdown
        assert result.revenue_breakdown['renewable_export'] == 30.0
