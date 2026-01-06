"""
Core BESS Optimization Module
==============================

This module contains the core optimization logic for the Battery Energy Storage System (BESS).

Main Classes:
- BESSOptimizerV2: Phase II BESS optimization model with improved constraint handling
- DataProcessingError, DataValidationError: Custom exception classes
"""

from .optimizer import (
    BESSOptimizerV2,
    BESSOptimizerModelII,
    BESSOptimizerV3,
    BESSOptimizer_Phase2_ModelII,
)
from ..data.exceptions import (
    DataProcessingError,
    DataLoadingError,
    DataValidationError,
    VisualizationError,
    format_exception_for_logging
)

__all__ = [
    'BESSOptimizerV2',
    'BESSOptimizerModelII',
    'BESSOptimizerV3',
    'BESSOptimizer_Phase2_ModelII',
    'DataProcessingError',
    'DataLoadingError',
    'DataValidationError',
    'VisualizationError',
    'format_exception_for_logging'
]

__version__ = '2.0.0'
