#!/usr/bin/env python3
"""
Investment Analysis Module for TechArena 2025 Phase 1
=====================================================

This module provides the InvestmentAnalyzer class for performing detailed
Discounted Cash Flow (DCF) analysis for Battery Energy Storage System (BESS)
investments according to TechArena 2025 competition specifications.

Key Features:
- Fixed BESS capacity (4,472 kWh) with configurable C-rates
- Country-specific financial parameters (WACC, inflation)
- 10-year DCF analysis with nominal cash flows
- NPV and levelized ROI calculations
- Excel-compatible output formatting

Competition Specifications:
- BESS Capacity: 4,472 kWh (fixed)
- BESS Power: 2,236 kW (rated)
- Investment Cost: 200 EUR/kWh
- C-rate Options: 0.25C, 0.33C, 0.50C
- Analysis Period: 2024-2033 (10 years operation)
- Investment Year: 2023

Author: SoloGen Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

class InvestmentAnalyzer:
    """
    Investment analyzer for BESS deployment analysis with proper DCF methodology.
    
    This class handles all investment calculations according to competition specifications,
    including proper BESS configuration, country-specific financial parameters, and 
    10-year DCF analysis with nominal cash flows.
    """
    
    def __init__(self):
        """Initialize the investment analyzer with competition specifications."""
        
        # BESS Technical Specifications (from competition)
        self.bess_specs = {
            'energy_capacity_kwh': 4472,  # Fixed capacity as per competition
            'rated_power_kw': 2236,       # Rated power at 0.5C
            'investment_cost_per_kwh': 200  # EUR/kWh
        }
        
        # C-rate configurations (power = C-rate * energy_capacity)
        self.c_rate_configs = {
            0.25: {'power_kw': 1118, 'power_mw': 1.118},
            0.33: {'power_kw': 1476, 'power_mw': 1.476}, 
            0.50: {'power_kw': 2236, 'power_mw': 2.236}
        }
        
        # Country-specific financial parameters
        self.financial_params = {
            'DE': {'wacc': 8.3, 'inflation': 2},
            'AT': {'wacc': 8.3, 'inflation': 3.3},
            'CH': {'wacc': 8.3, 'inflation': 0.1},
            'CZ': {'wacc': 12.0, 'inflation': 2.9},
            'HU': {'wacc': 15.0, 'inflation': 4.6}
        }
        
        # Analysis parameters
        self.analysis_years = 10  # 2024-2033
        self.investment_year = 2023
        self.first_operation_year = 2024
    
    def get_bess_configuration(self, c_rate: float) -> Dict[str, float]:
        """
        Get BESS configuration for a given C-rate.
        
        Args:
            c_rate: C-rate value (0.25, 0.33, or 0.50)
            
        Returns:
            Dictionary with BESS configuration parameters
        """
        if c_rate not in self.c_rate_configs:
            raise ValueError(f"Unsupported C-rate: {c_rate}. Supported values: {list(self.c_rate_configs.keys())}")
        
        config = self.c_rate_configs[c_rate].copy()
        config.update({
            'energy_capacity_kwh': self.bess_specs['energy_capacity_kwh'],
            'energy_capacity_mwh': self.bess_specs['energy_capacity_kwh'] / 1000,
            'c_rate': c_rate,
            'total_investment_eur': self.bess_specs['energy_capacity_kwh'] * self.bess_specs['investment_cost_per_kwh'],
            'total_investment_keur': (self.bess_specs['energy_capacity_kwh'] * self.bess_specs['investment_cost_per_kwh']) / 1000
        })
        
        return config
    
    def calculate_dcf_analysis(self, country: str, annual_revenue_2024: float, c_rate: float) -> Dict[str, Any]:
        """
        Perform 10-year DCF analysis for BESS investment.
        
        Args:
            country: Country code (DE, AT, CH, CZ, HU)
            annual_revenue_2024: Annual revenue in EUR for 2024
            c_rate: C-rate configuration
            
        Returns:
            Dictionary with DCF analysis results
        """
        if country not in self.financial_params:
            raise ValueError(f"Unsupported country: {country}")
        
        # Get parameters
        bess_config = self.get_bess_configuration(c_rate)
        financial_params = self.financial_params[country]
        wacc = financial_params['wacc'] / 100  # Convert to decimal
        inflation = financial_params['inflation'] / 100  # Convert to decimal
        
        # Calculate nominal discount rate: r_nominal = (1 + r_real)(1 + π) - 1
        # For simplicity, we use WACC as nominal discount rate directly
        nominal_discount_rate = wacc
        
        # DCF calculation
        total_pv = 0
        annual_cash_flows = []
        
        for year_offset in range(self.analysis_years):
            year = self.first_operation_year + year_offset
            
            # Apply inflation growth: Revenue_year = Revenue_2024 * (1 + π)^(year - 2024)
            nominal_revenue = annual_revenue_2024 * ((1 + inflation) ** year_offset)
            
            # Discount to present value: PV = CF / (1 + r)^t
            discount_factor = (1 + nominal_discount_rate) ** (year_offset + 1)  # +1 because investment is at t=0
            pv_revenue = nominal_revenue / discount_factor
            
            total_pv += pv_revenue
            annual_cash_flows.append({
                'year': year,
                'year_offset': year_offset,
                'nominal_revenue': nominal_revenue,
                'discount_factor': discount_factor,
                'present_value': pv_revenue
            })
        
        # Calculate NPV
        initial_investment = bess_config['total_investment_eur']
        npv = total_pv - initial_investment
        
        # Calculate Levelized ROI
        levelized_roi = (annual_revenue_2024 / initial_investment) * 100
        
        return {
            'country': country,
            'c_rate': c_rate,
            'bess_config': bess_config,
            'financial_params': financial_params,
            'annual_revenue_2024': annual_revenue_2024,
            'initial_investment_eur': initial_investment,
            'total_pv_revenues': total_pv,
            'npv': npv,
            'levelized_roi': levelized_roi,
            'annual_cash_flows': annual_cash_flows,
            'wacc_percent': financial_params['wacc'],
            'inflation_percent': financial_params['inflation']
        }
    
    def analyze_investment(self, country: str, c_rate: float, annual_revenue_2024: float) -> Dict[str, Any]:
        """
        Main method to perform complete investment analysis.
        
        Args:
            country: Country code
            c_rate: C-rate configuration
            annual_revenue_2024: Annual revenue for 2024 in EUR
            
        Returns:
            Complete investment analysis results
        """
        return self.calculate_dcf_analysis(country, annual_revenue_2024, c_rate)
    
    def format_for_excel(self, analysis_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Format analysis results for Excel output following competition format.
        
        Args:
            analysis_result: Results from analyze_investment method
            
        Returns:
            DataFrame formatted for Excel output
        """
        data = []
        
        # Extract key values
        wacc = analysis_result['wacc_percent']
        inflation = analysis_result['inflation_percent']
        annual_revenue = analysis_result['annual_revenue_2024']
        bess_config = analysis_result['bess_config']
        energy_capacity_mwh = bess_config['energy_capacity_mwh']
        
        # Header section with parameters
        data.append({'Col1': 'WACC', 'Col2': f'{wacc}%', 'Col3': '', 'Col4': ''})
        data.append({'Col1': 'Inflation Rate', 'Col2': f'{inflation}%', 'Col3': '', 'Col4': ''})
        data.append({'Col1': 'Discount Rate', 'Col2': f'{wacc}%', 'Col3': '', 'Col4': ''})
        data.append({'Col1': 'Yearly Profits (2024)', 'Col2': f'{annual_revenue:,.0f}', 'Col3': '', 'Col4': ''})
        data.append({'Col1': '', 'Col2': '', 'Col3': '', 'Col4': ''})  # Empty row
        
        # Table header
        data.append({'Col1': 'Year', 'Col2': 'Initial Investment [kEUR/MWh]', 'Col3': 'Yearly profits [kEUR/MWh]', 'Col4': ''})
        
        # Investment year (2023)
        capex_per_mwh_keur = bess_config['total_investment_keur'] / energy_capacity_mwh
        data.append({'Col1': '2023', 'Col2': f'{capex_per_mwh_keur:,.0f}', 'Col3': '', 'Col4': ''})
        
        # Operation years (2024-2033)
        for cash_flow in analysis_result['annual_cash_flows']:
            year = cash_flow['year']
            nominal_revenue = cash_flow['nominal_revenue']
            profit_per_mwh_keur = (nominal_revenue / 1000) / energy_capacity_mwh
            
            data.append({
                'Col1': str(year),
                'Col2': '',  # No additional investment
                'Col3': f'{profit_per_mwh_keur:,.0f}',
                'Col4': ''
            })
        
        return pd.DataFrame(data)
    
    def generate_summary_report(self, country_results: Dict[str, Any]) -> str:
        """
        Generate a summary report for all countries and configurations.
        
        Args:
            country_results: Dictionary with results for each country
            
        Returns:
            Formatted summary report string
        """
        report = []
        report.append("=" * 80)
        report.append("BESS INVESTMENT ANALYSIS SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        for country, result in country_results.items():
            report.append(f"Country: {country}")
            report.append(f"  C-rate: {result['c_rate']}")
            report.append(f"  Annual Revenue (2024): €{result['annual_revenue_2024']:,.0f}")
            report.append(f"  Initial Investment: €{result['initial_investment_eur']:,.0f}")
            report.append(f"  NPV (10-year): €{result['npv']:,.0f}")
            report.append(f"  Levelized ROI: {result['levelized_roi']:.1f}%")
            report.append(f"  WACC: {result['wacc_percent']:.1f}%")
            report.append(f"  Inflation: {result['inflation_percent']:.1f}%")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

if __name__ == "__main__":
    # Example usage
    analyzer = InvestmentAnalyzer()
    
    # Test with sample data
    test_result = analyzer.analyze_investment(
        country='AT',
        c_rate=0.5,
        annual_revenue_2024=150000  # €150k annual revenue
    )
    
    print("Investment Analysis Test:")
    print(f"NPV: €{test_result['npv']:,.0f}")
    print(f"Levelized ROI: {test_result['levelized_roi']:.1f}%")
    
    # Test Excel formatting
    excel_df = analyzer.format_for_excel(test_result)
    print("\nExcel format preview:")
    print(excel_df.head(10))