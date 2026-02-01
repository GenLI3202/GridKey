"""
Optimizer Service Module
========================

This module contains the service layer for the BESS Optimizer,
providing Pydantic data models for type-safe API I/O and the
OptimizerService wrapper for Module D.

Data Models:
- ModelType: Enum of optimization model variants (I, II, III, III-renew)
- OptimizationInput: Standardised optimizer input
- ScheduleEntry: Single timestep schedule item
- RenewableUtilization: Renewable energy utilization breakdown
- OptimizationResult: Standardised optimizer output
"""

from .models import (
    ModelType,
    OptimizationInput,
    ScheduleEntry,
    RenewableUtilization,
    OptimizationResult,
)

__all__ = [
    'ModelType',
    'OptimizationInput',
    'ScheduleEntry',
    'RenewableUtilization',
    'OptimizationResult',
]

__version__ = '1.0.0'
