#!/usr/bin/env python3
"""
Custom Exceptions for TechArena 2025 Phase 2 Data Processing
=============================================================

This module defines custom exception classes for better error handling
and debugging throughout the data processing pipeline.

Exception Hierarchy:
    DataProcessingError (base)
    ├── DataLoadingError
    ├── DataValidationError
    └── VisualizationError

Author: SoloGen Team
Date: October 2025
"""

from typing import Dict, Any


class DataProcessingError(Exception):
    """
    Base exception for all data processing errors.

    All custom exceptions in the data processing pipeline inherit
    from this base class for easier exception handling.
    """
    pass


class DataLoadingError(DataProcessingError):
    """
    Raised when data loading from Excel or other sources fails.

    Examples
    --------
    - File not found
    - Sheet missing from Excel workbook
    - Data parsing errors
    - Invalid file format

    Example
    -------
    >>> raise DataLoadingError("Excel file not found: data.xlsx")
    """
    pass


class DataValidationError(DataProcessingError):
    """
    Raised when data validation fails with critical errors.

    This exception carries the full validation report for detailed
    error analysis and debugging.

    Attributes
    ----------
    report : dict
        Validation report containing errors, warnings, and statistics

    Example
    -------
    >>> report = {'errors': ['Price out of bounds'], 'warnings': [], 'passed': False}
    >>> raise DataValidationError(report)
    """

    def __init__(self, validation_report: Dict[str, Any]):
        """
        Initialize DataValidationError with validation report.

        Parameters
        ----------
        validation_report : dict
            Dictionary containing validation results with keys:
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'stats': dict of validation statistics
            - 'passed': bool indicating if validation passed
        """
        self.report = validation_report
        num_errors = len(validation_report.get('errors', []))
        num_warnings = len(validation_report.get('warnings', []))

        message = (
            f"Data validation failed with {num_errors} error(s) "
            f"and {num_warnings} warning(s)"
        )

        super().__init__(message)

    def get_errors(self):
        """Get list of validation errors."""
        return self.report.get('errors', [])

    def get_warnings(self):
        """Get list of validation warnings."""
        return self.report.get('warnings', [])

    def get_stats(self):
        """Get validation statistics."""
        return self.report.get('stats', {})

    def __str__(self):
        """Pretty print error with details."""
        lines = [super().__str__()]

        if self.get_errors():
            lines.append("\nErrors:")
            for i, error in enumerate(self.get_errors(), 1):
                lines.append(f"  {i}. {error}")

        if self.get_warnings():
            lines.append("\nWarnings:")
            for i, warning in enumerate(self.get_warnings(), 1):
                lines.append(f"  {i}. {warning}")

        return "\n".join(lines)


class VisualizationError(DataProcessingError):
    """
    Raised when visualization creation fails.

    Examples
    --------
    - Missing data for plot
    - Invalid plot parameters
    - Plotly rendering errors
    - Country not found in dataset

    Example
    -------
    >>> raise VisualizationError("Country 'XX' not found in dataset")
    """
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def format_exception_for_logging(exc: Exception) -> str:
    """
    Format an exception for clean logging output.

    Parameters
    ----------
    exc : Exception
        Exception to format

    Returns
    -------
    str
        Formatted exception message

    Example
    -------
    >>> try:
    ...     raise DataLoadingError("File not found")
    ... except DataLoadingError as e:
    ...     print(format_exception_for_logging(e))
    DataLoadingError: File not found
    """
    return f"{exc.__class__.__name__}: {str(exc)}"


if __name__ == "__main__":
    # Demo: Show exception classes in action
    print("TechArena 2025 Custom Exceptions Demo")
    print("=" * 50)

    # Example 1: DataLoadingError
    print("\n1. DataLoadingError Example:")
    try:
        raise DataLoadingError("Excel file not found: missing_file.xlsx")
    except DataLoadingError as e:
        print(f"  Caught: {format_exception_for_logging(e)}")

    # Example 2: DataValidationError
    print("\n2. DataValidationError Example:")
    sample_report = {
        'passed': False,
        'errors': [
            'Day-ahead price 1500 EUR/MWh exceeds upper bound 1000',
            'Row count mismatch: aFRR energy (35000) != day-ahead (35135)'
        ],
        'warnings': [
            'aFRR energy DE_Pos: 96.5% zeros'
        ],
        'stats': {
            'day_ahead_rows': 35135,
            'afrr_energy_rows': 35000
        }
    }

    try:
        raise DataValidationError(sample_report)
    except DataValidationError as e:
        print(f"  {e}")
        print(f"\n  Error count: {len(e.get_errors())}")
        print(f"  Warning count: {len(e.get_warnings())}")

    # Example 3: VisualizationError
    print("\n3. VisualizationError Example:")
    try:
        raise VisualizationError("Failed to create plot: Country 'XX' not in dataset")
    except VisualizationError as e:
        print(f"  Caught: {format_exception_for_logging(e)}")

    print("\n" + "=" * 50)
    print("Exception hierarchy:")
    print("  DataProcessingError (base)")
    print("  ├── DataLoadingError")
    print("  ├── DataValidationError")
    print("  └── VisualizationError")
