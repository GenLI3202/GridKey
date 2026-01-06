"""
Market Data Processing Module
==============================

This module handles loading, processing, and transforming market data for BESS optimization.

Main Functions:
- load_market_tables: Load market data from Excel workbooks
- convert_tables_to_tidy: Convert wide-format tables to tidy format
- Visualization functions for market price analysis
"""

from .load_process_market_data import (
    MarketTables,
    load_market_tables,
    # load_data,
    convert_tables_to_tidy,
    wide_to_tidy_day_ahead,
    wide_to_tidy_fcr,
    wide_to_tidy_afrr,
    load_phase2_market_tables,
    validate_phase2_data
)

__all__ = [
    'MarketTables',
    'load_market_tables',
    'load_data',
    'convert_tables_to_tidy',
    'wide_to_tidy_day_ahead',
    'wide_to_tidy_fcr',
    'wide_to_tidy_afrr',
    'load_phase2_market_tables',
    'validate_phase2_data'
]

__version__ = '2.0.0'
