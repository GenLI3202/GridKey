"""
Tests for BESSOptimizerModelIIIRenew — Renewable integration on top of Model III.

These tests use a small synthetic dataset (12 timesteps = 3 hours) to keep
solve times short while still exercising all renewable constraints.

Requires a working MILP solver (HiGHS / CBC / Gurobi / CPLEX).
"""

import math
import pytest
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.core.optimizer import BESSOptimizerModelIII, BESSOptimizerModelIIIRenew


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_country_data(n_timesteps: int = 12, include_renewable: bool = True):
    """Build a minimal country_data DataFrame for testing.

    12 timesteps = 3 hours = 0.75 blocks (we need at least 1 full block,
    so we use 16 timesteps = 4 hours for block-aware tests,
    but 12 is enough for basic variable/constraint tests).
    """
    # Use 16 timesteps (1 full 4-hour block) for cleaner block alignment
    if n_timesteps < 16:
        n_timesteps = 16

    data = {
        'price_day_ahead':       [50.0 + i * 2 for i in range(n_timesteps)],
        'price_afrr_energy_pos': [float('nan')] * n_timesteps,  # Not activated
        'price_afrr_energy_neg': [float('nan')] * n_timesteps,
        'price_fcr':             [100.0] * n_timesteps,
        'price_afrr_pos':        [5.0]  * n_timesteps,
        'price_afrr_neg':        [10.0] * n_timesteps,
        'w_afrr_pos':            [1.0]  * n_timesteps,
        'w_afrr_neg':            [1.0]  * n_timesteps,
        'block_id':              [i // 16 for i in range(n_timesteps)],
        'day_id':                [1]     * n_timesteps,
        'block_of_day':          [0]     * n_timesteps,
        'hour':                  [(i * 15 // 60) for i in range(n_timesteps)],
        'day_of_year':           [1]     * n_timesteps,
        'month':                 [1]     * n_timesteps,
        'year':                  [2024]  * n_timesteps,
    }

    if include_renewable:
        # Simple solar-like pattern: ramp up then flat
        data['p_renewable_forecast_kw'] = [
            200.0 + 50 * min(i, 8) for i in range(n_timesteps)
        ]

    return pd.DataFrame(data)


def _solve_model(model):
    """Solve a Pyomo model using the first available solver."""
    for solver_name in ['appsi_highs', 'highs', 'cbc', 'glpk']:
        solver = pyo.SolverFactory(solver_name)
        if solver.available():
            results = solver.solve(model, tee=False)
            results._solve_time = 0.0
            results._solver_name = solver_name
            return results
    pytest.skip("No MILP solver available")


# ---------------------------------------------------------------------------
# Test: Model builds successfully
# ---------------------------------------------------------------------------

class TestModelBuild:
    def test_builds_with_renewable(self):
        optimizer = BESSOptimizerModelIIIRenew()
        country_data = _make_country_data(include_renewable=True)
        model = optimizer.build_optimization_model(country_data, c_rate=0.5)

        assert hasattr(model, 'P_renewable')
        assert hasattr(model, 'p_renewable_self')
        assert hasattr(model, 'p_renewable_export')
        assert hasattr(model, 'p_renewable_curtail')
        assert hasattr(model, 'renewable_balance')
        assert hasattr(model, 'profit_renewable_export')

    def test_fallback_without_renewable_column(self):
        optimizer = BESSOptimizerModelIIIRenew()
        country_data = _make_country_data(include_renewable=False)
        model = optimizer.build_optimization_model(country_data, c_rate=0.5)

        # Should fall back to plain Model III — no renewable variables
        assert not hasattr(model, 'P_renewable')
        assert not hasattr(model, 'p_renewable_self')

    def test_fallback_with_all_nan_renewable(self):
        optimizer = BESSOptimizerModelIIIRenew()
        country_data = _make_country_data(include_renewable=True)
        country_data['p_renewable_forecast_kw'] = float('nan')
        model = optimizer.build_optimization_model(country_data, c_rate=0.5)

        assert not hasattr(model, 'P_renewable')

    def test_variable_count_exceeds_model_iii(self):
        """Model III-Renew should have more variables than plain Model III."""
        country_data = _make_country_data(include_renewable=True)

        opt3 = BESSOptimizerModelIII()
        model3 = opt3.build_optimization_model(country_data, c_rate=0.5)

        opt3r = BESSOptimizerModelIIIRenew()
        model3r = opt3r.build_optimization_model(country_data, c_rate=0.5)

        # 3 new variables per timestep
        n_t = len(list(model3.T))
        assert model3r.nvariables() == model3.nvariables() + 3 * n_t


# ---------------------------------------------------------------------------
# Test: Solve and verify constraints
# ---------------------------------------------------------------------------

class TestSolveAndConstraints:
    @pytest.fixture
    def solved(self):
        optimizer = BESSOptimizerModelIIIRenew()
        country_data = _make_country_data(n_timesteps=16, include_renewable=True)
        model = optimizer.build_optimization_model(country_data, c_rate=0.5)
        results = _solve_model(model)
        solution = optimizer.extract_solution(model, results)
        return model, solution

    def test_solve_succeeds(self, solved):
        _, solution = solved
        assert solution['status'] in ('optimal', 'feasible')

    def test_renewable_balance_holds(self, solved):
        """Cst-R1: self + export + curtail == P_renewable for each t."""
        model, _ = solved
        for t in model.T:
            p_self = pyo.value(model.p_renewable_self[t])
            p_export = pyo.value(model.p_renewable_export[t])
            p_curtail = pyo.value(model.p_renewable_curtail[t])
            p_forecast = pyo.value(model.P_renewable[t])
            assert abs(p_self + p_export + p_curtail - p_forecast) < 1e-4, \
                f"Renewable balance violated at t={t}"

    def test_total_charge_includes_renewable_self(self, solved):
        """Cst-R2: p_total_ch >= p_renewable_self."""
        model, _ = solved
        for t in model.T:
            total_ch = pyo.value(model.p_total_ch[t])
            p_self = pyo.value(model.p_renewable_self[t])
            assert total_ch >= p_self - 1e-4, \
                f"Total charge ({total_ch}) < renewable self ({p_self}) at t={t}"

    def test_all_renewable_vars_non_negative(self, solved):
        model, _ = solved
        for t in model.T:
            assert pyo.value(model.p_renewable_self[t]) >= -1e-6
            assert pyo.value(model.p_renewable_export[t]) >= -1e-6
            assert pyo.value(model.p_renewable_curtail[t]) >= -1e-6


# ---------------------------------------------------------------------------
# Test: Extract solution
# ---------------------------------------------------------------------------

class TestExtractSolution:
    @pytest.fixture
    def solution(self):
        optimizer = BESSOptimizerModelIIIRenew()
        country_data = _make_country_data(n_timesteps=16, include_renewable=True)
        model = optimizer.build_optimization_model(country_data, c_rate=0.5)
        results = _solve_model(model)
        return optimizer.extract_solution(model, results)

    def test_has_renewable_keys(self, solution):
        assert 'p_renewable_self' in solution
        assert 'p_renewable_export' in solution
        assert 'p_renewable_curtail' in solution
        assert 'profit_renewable_export' in solution
        assert 'renewable_utilization' in solution
        assert 'revenue_renewable_export' in solution

    def test_renewable_utilization_fields(self, solution):
        ru = solution['renewable_utilization']
        assert 'total_generation_kwh' in ru
        assert 'self_consumption_kwh' in ru
        assert 'export_kwh' in ru
        assert 'curtailment_kwh' in ru
        assert 'utilization_rate' in ru
        assert 0.0 <= ru['utilization_rate'] <= 1.0

    def test_utilization_math(self, solution):
        """self + export + curtail should equal total generation."""
        ru = solution['renewable_utilization']
        total = ru['self_consumption_kwh'] + ru['export_kwh'] + ru['curtailment_kwh']
        assert abs(total - ru['total_generation_kwh']) < 1e-3

    def test_export_revenue_non_negative(self, solution):
        """With positive DA prices and renewable, export revenue >= 0."""
        assert solution['profit_renewable_export'] >= -1e-6

    def test_fallback_solution_has_no_renewable_keys(self):
        """Without renewable data, solution should match plain Model III."""
        optimizer = BESSOptimizerModelIIIRenew()
        country_data = _make_country_data(include_renewable=False)
        model = optimizer.build_optimization_model(country_data, c_rate=0.5)
        results = _solve_model(model)
        solution = optimizer.extract_solution(model, results)

        assert 'p_renewable_self' not in solution
        assert 'renewable_utilization' not in solution


# ---------------------------------------------------------------------------
# Test: Zero renewable forecast
# ---------------------------------------------------------------------------

class TestZeroRenewable:
    def test_zero_forecast_no_export_revenue(self):
        """With zero renewable forecast, export revenue should be zero."""
        optimizer = BESSOptimizerModelIIIRenew()
        country_data = _make_country_data(include_renewable=True)
        country_data['p_renewable_forecast_kw'] = 0.0

        model = optimizer.build_optimization_model(country_data, c_rate=0.5)
        results = _solve_model(model)
        solution = optimizer.extract_solution(model, results)

        assert solution['status'] in ('optimal', 'feasible')
        assert abs(solution['profit_renewable_export']) < 1e-4
        assert abs(solution['renewable_utilization']['total_generation_kwh']) < 1e-4
