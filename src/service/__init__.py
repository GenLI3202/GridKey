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

Service:
- OptimizerService: Unified service wrapper for the optimization workflow
"""

from .models import (
    ModelType,
    OptimizationInput,
    ScheduleEntry,
    RenewableUtilization,
    OptimizationResult,
)
from .optimizer_service import OptimizerService

__all__ = [
    'ModelType',
    'OptimizationInput',
    'ScheduleEntry',
    'RenewableUtilization',
    'OptimizationResult',
    'OptimizerService',
]

__version__ = '1.0.0'
