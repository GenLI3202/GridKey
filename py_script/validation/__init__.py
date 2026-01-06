"""
BESS Optimization Validation Module
====================================

This module provides post-solve validation tools for BESS optimization models.

Main Components:
----------------
- ConstraintValidator: Validates commented-out constraints (Cst-8, Cst-9)
- validate_solution: Convenience function for quick validation

Usage:
------
    from py_script.validation import ConstraintValidator, validate_solution

    # After solving a model
    validator = ConstraintValidator(model, solution)
    report = validator.generate_validation_report()

    # Or use convenience function
    report = validate_solution(model, solution)
"""

from .constraint_validator import ConstraintValidator, validate_solution

__all__ = ['ConstraintValidator', 'validate_solution']
