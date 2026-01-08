"""
Unit and integration tests for BESS Optimizer Models.

This file consolidates formal, reusable unit and integration tests for the core
optimizer models (Model I, Model II, etc.). It uses the pytest framework for
structured, maintainable, and scalable testing.

Purpose:
- Verify the mathematical correctness of model formulations.
- Ensure physical constraints (SOC, power limits) are respected.
- Validate the implementation of new features (e.g., degradation costs).
- Compare behavior between different model versions.

Based on the test refactoring plan.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path to access the 'core' module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.optimizer import BESSOptimizerModelII, BESSOptimizerModelI


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def minimal_country_data():
    """
    Create minimal test dataset for quick tests (16 timesteps = 4 hours).
    Using module scope to create it only once per test module run.
    """
    timestamps = pd.date_range('2024-01-01', periods=16, freq='15min')

    # Create price variation to enable optimization
    prices_da = [20, 30, 40, 50, 60, 70, 80, 90,  # Rising
                  90, 80, 70, 60, 50, 40, 30, 20]  # Falling

    return pd.DataFrame({
        'timestamp': timestamps,
        'price_day_ahead': prices_da,
        'price_fcr': [0.0] * 16,
        'price_afrr_pos': [0.0] * 16,
        'price_afrr_neg': [0.0] * 16,
        'price_afrr_energy_pos': [0.0] * 16,
        'price_afrr_energy_neg': [0.0] * 16,
        'hour': timestamps.hour.tolist(),
        'day_of_year': [1] * 16,
        'month': [1] * 16,
        'year': [2024] * 16,
        'block_of_day': [i//16 for i in range(16)],
        'block_id': [0] * 16,
        'day_id': [0] * 16,
    })


@pytest.fixture(scope="module")
def one_day_data():
    """
    Create 1-day test dataset (96 timesteps = 24 hours).
    Using module scope for efficiency.
    """
    timestamps = pd.date_range('2024-01-01', periods=96, freq='15min')

    # Create realistic daily price pattern
    hour_of_day = timestamps.hour.to_numpy()
    base_price = 40
    hourly_variation = 30 * np.sin((hour_of_day - 6) * np.pi / 12)
    prices_da = base_price + hourly_variation

    return pd.DataFrame({
        'timestamp': timestamps,
        'price_day_ahead': prices_da,
        'price_fcr': [100.0] * 96,
        'price_afrr_pos': [50.0] * 96,
        'price_afrr_neg': [50.0] * 96,
        'price_afrr_energy_pos': prices_da + 10,
        'price_afrr_energy_neg': prices_da - 10,
        'hour': timestamps.hour.tolist(),
        'day_of_year': [1] * 96,
        'month': [1] * 96,
        'year': [2024] * 96,
        'block_of_day': [i//16 for i in range(96)],
        'block_id': [i//16 for i in range(96)],
        'day_id': [0] * 96,
    })


# ============================================================================
# Test Class 1: Model II Initialization
# ============================================================================

class TestModelIIInitialization:
    """Test Model (ii) initialization and configuration loading."""

    def test_load_degradation_config(self):
        """Degradation config should load and expose monotone marginal costs."""
        optimizer = BESSOptimizerModelII(
            degradation_config_path='data/phase2_aging_config/aging_config.json'
        )
        assert optimizer.degradation_params['num_segments'] == 10
        assert len(optimizer.degradation_params['marginal_costs']) == 10
        assert optimizer.degradation_params['marginal_costs'][0] == pytest.approx(0.0052)
        assert optimizer.degradation_params['marginal_costs'][-1] == pytest.approx(0.0990)

    def test_invalid_config_path(self):
        """Should raise FileNotFoundError for invalid config path."""
        with pytest.raises(FileNotFoundError):
            BESSOptimizerModelII(degradation_config_path='nonexistent.json')

    def test_segment_capacity_calculation(self):
        """Segment capacity should be total capacity divided by number of segments."""
        optimizer = BESSOptimizerModelII()
        expected_seg_cap = 4472 / 10
        assert optimizer.degradation_params['segment_capacity_kwh'] == pytest.approx(expected_seg_cap)

    def test_cost_sum_equals_full_cycle_cost(self):
        """Sum of marginal costs times segment size should equal cost per full cycle."""
        optimizer = BESSOptimizerModelII()
        costs = optimizer.degradation_params['marginal_costs']
        seg_cap = optimizer.degradation_params['segment_capacity_kwh']
        total_cost = sum(costs) * seg_cap
        expected_cost = 232.92  # EUR per full cycle
        assert total_cost == pytest.approx(expected_cost, rel=0.01)


# ============================================================================
# Test Class 2: Model II Building
# ============================================================================

class TestModelIIModelBuilding:
    """Test Pyomo model construction for Model (ii)."""

    def test_segment_set_creation(self, minimal_country_data):
        """Segment set J should have 10 elements (1 to 10)."""
        optimizer = BESSOptimizerModelII()
        model = optimizer.build_optimization_model(minimal_country_data, c_rate=0.5)
        assert list(model.J) == list(range(1, 11))

    def test_segment_variables_exist(self, minimal_country_data):
        """All segment-related variables and components should exist."""
        optimizer = BESSOptimizerModelII()
        model = optimizer.build_optimization_model(minimal_country_data, c_rate=0.5)
        import pyomo.environ as pyo
        assert hasattr(model, 'p_ch_j')
        assert hasattr(model, 'p_dis_j')
        assert hasattr(model, 'e_soc_j')
        assert hasattr(model, 'e_soc')
        assert isinstance(model.e_soc, (pyo.base.expression.IndexedExpression, pyo.base.expression._GeneralExpressionData))
        assert hasattr(model, 'stacked_tank_ordering')
        assert hasattr(model, 'segment_soc_dynamics')

    def test_objective_includes_degradation_cost(self, minimal_country_data):
        """Objective should include degradation cost term."""
        import pyomo.environ as pyo
        optimizer = BESSOptimizerModelII(alpha=1.0)
        model = optimizer.build_optimization_model(minimal_country_data, c_rate=0.5)
        assert hasattr(model, 'objective')
        assert model.objective.sense == pyo.maximize
        assert hasattr(model, 'alpha')
        assert hasattr(model, 'c_cost')


# ============================================================================
# Test Class 3: Model Comparison
# ============================================================================

class TestModelIIVsModelI:
    """Integration tests comparing Model (i) and Model (ii) behavior."""

    def test_model_ii_ignores_cycle_limit_parameter(self, minimal_country_data):
        """Model (ii) should ignore daily_cycle_limit and not create the constraint."""
        optimizer = BESSOptimizerModelII(alpha=1.0)
        model = optimizer.build_optimization_model(
            minimal_country_data, c_rate=0.5, daily_cycle_limit=1.5
        )
        assert not hasattr(model, 'daily_cycle_limit') or not model.daily_cycle_limit.active


# ============================================================================
# Test Class 4: Degradation Metrics
# ============================================================================

class TestModelIIDegradationMetrics:
    """Test degradation metric calculation."""

    def test_metrics_calculation_with_zero_discharge(self):
        """Metrics should handle case with no discharge (zero cost)."""
        import pyomo.environ as pyo
        optimizer = BESSOptimizerModelII(alpha=1.0)
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=[0, 1])
        model.J = pyo.Set(initialize=range(1, 11))
        model.D = pyo.Set(initialize=[0])
        model.eta_dis = pyo.Param(initialize=0.95)
        model.dt = pyo.Param(initialize=0.25)
        model.E_nom = pyo.Param(initialize=4472)
        model.c_cost = pyo.Param(model.J, initialize={j: 0.01 * j for j in range(1, 11)})
        p_dis_j = {}
        metrics = optimizer._calculate_degradation_metrics(model, p_dis_j)
        assert metrics['total_cyclic_cost_eur'] == 0.0
        assert metrics['equivalent_full_cycles'] == 0.0

    def test_metrics_structure(self):
        """Degradation metrics should have all required keys."""
        import pyomo.environ as pyo
        optimizer = BESSOptimizerModelII(alpha=1.5)
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=[0])
        model.J = pyo.Set(initialize=range(1, 11))
        model.D = pyo.Set(initialize=[0])
        model.eta_dis = pyo.Param(initialize=0.95)
        model.dt = pyo.Param(initialize=0.25)
        model.E_nom = pyo.Param(initialize=4472)
        model.c_cost = pyo.Param(model.J, initialize={j: 0.01 * j for j in range(1, 11)})
        model.alpha = pyo.Param(initialize=1.5)
        p_dis_j = {(0, 1): 100.0}
        metrics = optimizer._calculate_degradation_metrics(model, p_dis_j)
        required_keys = [
            'total_cyclic_cost_eur', 'equivalent_full_cycles', 'total_throughput_kwh',
            'throughput_per_segment_kwh', 'cost_per_segment_eur', 'average_dod', 'alpha'
        ]
        assert all(key in metrics for key in required_keys)


# ============================================================================
# Integration Tests (Slow)
# ============================================================================

@pytest.mark.slow
class TestModelIIIntegration:
    """Integration tests requiring a solver, marked as slow."""

    def test_one_day_optimization_run(self, one_day_data):
        """Test a full 1-day optimization with degradation cost."""
        pytest.skip("Integration test is slow and requires a solver. Run manually if needed.")
        # optimizer = BESSOptimizerModelII(alpha=1.0, enforce_segment_binary=False)
        # results = optimizer.optimize(one_day_data, c_rate=0.5)
        # assert results['status'] in ['optimal', 'feasible']
        # assert 'degradation_metrics' in results
        # assert results['degradation_metrics']['total_cyclic_cost_eur'] > 0


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    """
    Allows running tests directly from the command line.
    Example: python src/test_scripts/test_optimizer_core.py
    """
    # Find the workspace root to configure pytest
    workspace_root = Path(__file__).resolve().parents[2]
    
    # Run pytest on this file
    # Use '-v' for verbose output and '--tb=short' for concise traceback
    # Use '-k "not slow"' to skip slow tests by default
    pytest.main([
        str(Path(__file__).resolve()),
        '-v',
        '--tb=short',
        '-m', 'not slow',
        f'--rootdir={workspace_root}'
    ])
