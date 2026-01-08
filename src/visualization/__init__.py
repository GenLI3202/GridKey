"""
Visualization Module
====================

This module contains visualization utilities and plotting functions for BESS analysis.

Main Components:
- config: McKinsey-style visualization templates and configurations
- validation_plots: Validation and results visualization functions
"""

from .config import (
    create_mckinsey_template,
    get_country_color,
    apply_mckinsey_style
)

__all__ = [
    'create_mckinsey_template',
    'get_country_color',
    'apply_mckinsey_style'
]

__version__ = '2.0.0'
