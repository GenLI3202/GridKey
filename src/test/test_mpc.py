# src/test/test_mpc.py
"""
Unit tests for MPC rolling horizon optimization.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.service.mpc import MPCRollingHorizon
from src.service.models import OptimizationInput, ModelType
from src.service.adapter import DataAdapter


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    optimizer = Mock()
    optimizer.market_params = {'time_step_hours': 0.25}  # 15 minutes
    optimizer.battery_params = {
        'capacity_kwh': 4472,
        'c_rate': 0.5,
        'efficiency': 0.95,
        'initial_soc': 0.5,
    }
    optimizer.degradation_params = {
        'num_segments': 6,
        'segment_capacity_kwh': 4472 / 6,
    }
    return optimizer


@pytest.fixture
def mock_adapter():
    """Create a mock adapter."""
    adapter = Mock(spec=DataAdapter)
    return adapter


@pytest.fixture
def opt_input_12h():
    """Create a 12h OptimizationInput for testing."""
    return OptimizationInput(
        time_horizon_hours=12,
        da_prices=[50.0] * 48,  # 12h * 4 = 48 timesteps
        afrr_energy_pos=[40.0] * 48,
        afrr_energy_neg=[30.0] * 48,
        fcr_prices=[100.0] * 3,  # 12h / 4h = 3 blocks
        afrr_capacity_pos=[5.0] * 3,
        afrr_capacity_neg=[10.0] * 3,
        battery_capacity_kwh=4472,
        c_rate=0.5,
        efficiency=0.95,
        initial_soc=0.5,
        model_type=ModelType.MODEL_III,
        alpha=1.0,
    )


class TestMPCRollingHorizon:
    """Test MPCRollingHorizon class."""

    def test_init(self, mock_optimizer, mock_adapter):
        """Test MPCRollingHorizon initialization."""
        mpc = MPCRollingHorizon(
            optimizer=mock_optimizer,
            adapter=mock_adapter,
            horizon_hours=6,
            execution_hours=4,
        )

        assert mpc.optimizer == mock_optimizer
        assert mpc.adapter == mock_adapter
        assert mpc.horizon_hours == 6
        assert mpc.execution_hours == 4
        assert mpc.horizon_steps == 24  # 6h * 4
        assert mpc.execution_steps == 16  # 4h * 4

    def test_get_initial_segment_soc(self, mock_optimizer, mock_adapter):
        """Test SOC segment distribution."""
        mpc = MPCRollingHorizon(
            optimizer=mock_optimizer,
            adapter=mock_adapter,
        )

        # Test with 50% SOC
        total_soc = 4472 * 0.5  # 2236 kWh
        segment_soc = mpc._get_initial_segment_soc(total_soc)

        # Should fill segments 1-3 completely, segment 4 partially
        seg_capacity = 4472 / 6  # ~745.33 kWh
        assert abs(segment_soc[1] - seg_capacity) < 1e-6
        assert abs(segment_soc[2] - seg_capacity) < 1e-6
        assert abs(segment_soc[3] - seg_capacity) < 1e-6

        # Verify total
        total = sum(segment_soc.values())
        assert abs(total - total_soc) < 1e-6

    def test_slice_opt_input(self, mock_optimizer, mock_adapter, opt_input_12h):
        """Test OptimizationInput slicing."""
        mpc = MPCRollingHorizon(
            optimizer=mock_optimizer,
            adapter=mock_adapter,
        )

        # Slice first 6 hours (timesteps 0-24)
        sliced = mpc._slice_opt_input(opt_input_12h, 0, 24)

        assert sliced.time_horizon_hours == 6
        assert len(sliced.da_prices) == 24
        assert len(sliced.fcr_prices) == 2  # 0-24 covers 2 blocks

    def test_solve_12h(self, mock_optimizer, mock_adapter, opt_input_12h):
        """Test 12h MPC solving."""
        # Setup mocks
        mock_model = Mock()
        mock_model.J = [1, 2, 3, 4, 5, 6]  # Add segments

        # Mock e_soc_j as a subscriptable dict with setlb/setub methods
        mock_soc_var = Mock()
        mock_soc_var.setlb = Mock()
        mock_soc_var.setub = Mock()
        mock_model.e_soc_j = {(0, j): mock_soc_var for j in mock_model.J}

        # Mock z_segment_active as a subscriptable dict with fix method
        mock_binary_var = Mock()
        mock_binary_var.fix = Mock()
        mock_model.z_segment_active = {(0, j): mock_binary_var for j in mock_model.J}

        mock_solver_results = Mock()
        mock_solver_results.solver = {'name': 'highs'}

        # Mock solution for each iteration
        def mock_build(*args, **kwargs):
            return mock_model

        def mock_solve(model):
            return model, mock_solver_results

        def mock_extract(model, results):
            return {
                'status': 'optimal',
                'p_ch': {i: 0.0 for i in range(24)},
                'p_dis': {i: 100.0 for i in range(24)},
                'p_afrr_pos_e': {},
                'p_afrr_neg_e': {},
                'e_soc': {i: 2236.0 for i in range(24)},  # 50% SOC
                'c_fcr': {0: 1.0},
                'c_afrr_pos': {0: 0.5},
                'c_afrr_neg': {0: 0.0},
                'profit_da': 100.0,
                'profit_afrr_energy': 50.0,
                'profit_as_capacity': 25.0,
                'cost_cyclic': 20.0,
                'cost_calendar': 5.0,
                'solve_time': 5.0,
            }

        mock_optimizer.build_optimization_model = mock_build
        mock_optimizer.solve_model = mock_solve
        mock_optimizer.extract_solution = mock_extract
        mock_adapter.to_country_data = Mock(return_value=Mock())

        # Run MPC
        mpc = MPCRollingHorizon(
            optimizer=mock_optimizer,
            adapter=mock_adapter,
            horizon_hours=6,
            execution_hours=4,
        )

        solution = mpc.solve_12h(opt_input_12h, c_rate=0.5)

        # Verify results
        assert solution['status'] == 'optimal'
        assert len(solution['e_soc']) == 48  # 12h * 4 timesteps
        assert solution['solver'] == 'highs'
        assert solution['solve_time'] > 0  # Should accumulate from 3 iterations


class TestMPCAPITegration:
    """Integration tests for MPC API endpoint."""

    def test_optimize_request_mpc_validation(self):
        """Test OptimizeRequestMPC validates 12h data."""
        from src.api.main import OptimizeRequestMPC, MarketPrices12h
        from pydantic import ValidationError

        # Valid request
        valid_request = OptimizeRequestMPC(
            market_prices=MarketPrices12h(
                day_ahead=[50.0] * 48,
                afrr_energy_pos=[40.0] * 48,
                afrr_energy_neg=[30.0] * 48,
                fcr=[100.0, 105.0, 110.0],
                afrr_capacity_pos=[5.0, 6.0, 7.0],
                afrr_capacity_neg=[10.0, 11.0, 12.0],
            )
        )
        assert valid_request.market_prices.day_ahead[0] == 50.0

        # Invalid: wrong length
        with pytest.raises(ValidationError):
            OptimizeRequestMPC(
                market_prices=MarketPrices12h(
                    day_ahead=[50.0] * 47,  # Wrong length!
                    afrr_energy_pos=[40.0] * 48,
                    afrr_energy_neg=[30.0] * 48,
                    fcr=[100.0, 105.0, 110.0],
                    afrr_capacity_pos=[5.0, 6.0, 7.0],
                    afrr_capacity_neg=[10.0, 11.0, 12.0],
                )
            )

    def test_mpc_endpoint_exists(self):
        """Test that the MPC endpoint is registered."""
        from src.api.main import app
        import inspect

        # Get all routes
        routes = [route.path for route in app.routes]

        assert "/api/v1/optimize-mpc" in routes
