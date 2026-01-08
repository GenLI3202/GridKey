"""
Rolling Horizon MPC Framework for BESS Optimization
====================================================

This module provides the Model Predictive Control (MPC) and meta-optimization
framework for large-scale BESS optimization.

Three-Layer Architecture:
--------------------------
1. Inner Layer: BESSOptimizerModelIII (from core.optimizer)
2. Middle Layer: MPCSimulator (rolling horizon simulation)
3. Outer Layer: MetaOptimizer (alpha parameter sweep)

Main Components:
----------------
- MPCSimulator: Receding horizon controller for full-year simulation
- MetaOptimizer: Parameter sweep for optimal degradation price (alpha)

Usage Example:
--------------
    from src.rolling_horizon import MPCSimulator, MetaOptimizer
    from src.core.optimizer import BESSOptimizerModelIII

    # Create optimizer
    optimizer = BESSOptimizerModelIII(alpha=1.0)

    # Run MPC simulation
    simulator = MPCSimulator(optimizer, full_data)
    results = simulator.run_full_simulation()

    # Or run meta-optimization
    meta_opt = MetaOptimizer(
        full_data=data,
        country_config={'wacc': 0.05, 'inflation': 0.02, ...},
        alpha_values=[0.5, 1.0, 1.5, 2.0]
    )
    best = meta_opt.find_optimal_alpha()
"""

from .mpc_simulator import MPCSimulator
from .meta_optimizer import MetaOptimizer

__all__ = ['MPCSimulator', 'MetaOptimizer']
