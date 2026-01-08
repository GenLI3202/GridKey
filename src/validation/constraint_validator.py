"""
Constraint Validator for BESS Optimization Models
==================================================

This module provides post-solve validation for constraints that are commented out
in the optimization model for performance reasons. Specifically, it validates:

- Cst-8: Cross-Market Mutual Exclusivity
- Cst-9: Minimum Bid Size Constraints

These constraints are intentionally not enforced during optimization (to reduce
MILP complexity and solve time), but are validated after solving to ensure the
solution is physically and economically feasible.

Author: Gen's BESS Optimization Team
Phase II Development: November 2025
"""

import pyomo.environ as pyo
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """Post-solve validator for BESS optimization model constraints.

    Validates constraints that are commented out in the model for performance:
    - Cst-8: Cross-market mutual exclusivity (charge/discharge vs AS capacity)
    - Cst-9: Minimum bid sizes for DA and aFRR energy markets
    - Calendar aging SOS2 properties
    - Segment SOC ordering (stacked tank)

    These validations run AFTER optimization completes and do not affect
    the solve process. Violations are reported but do not invalidate the solution.

    Usage:
        validator = ConstraintValidator(model, solution, tolerance=1e-6)
        report = validator.generate_validation_report()
        if report['total_violations'] > 0:
            logger.warning(f"Found {report['total_violations']} constraint violations")
    """

    def __init__(
        self,
        model: pyo.ConcreteModel,
        solution: Dict[str, Any],
        tolerance: float = 1e-6
    ) -> None:
        """Initialize the constraint validator.

        Args:
            model: Solved Pyomo model
            solution: Solution dictionary from solve_model()
            tolerance: Numerical tolerance for constraint violations (default 1e-6)
        """
        self.model = model
        self.solution = solution
        self.tolerance = tolerance

        # Extract model parameters
        self.P_max_config = pyo.value(model.P_max_config)
        self.min_bid_da = pyo.value(model.min_bid_da) * 1000  # Convert MW to kW
        self.min_bid_afrr_e = pyo.value(model.min_bid_afrr_e) * 1000 if hasattr(model, 'min_bid_afrr_e') else 100  # kW

        logger.info("Initialized ConstraintValidator with tolerance=%.2e", tolerance)

    def _safe_value(self, component: Any) -> Optional[float]:
        """Safely extract value from Pyomo component."""
        try:
            return pyo.value(component)
        except (TypeError, ValueError, AttributeError):
            return None

    def check_cst8_cross_market_exclusivity(self) -> Dict[str, Any]:
        """Validate Cst-8: Cross-Market Mutual Exclusivity constraints.

        Checks two constraints (commented out in model for performance):
        1. y_total_dis[t] + y_fcr[b] + y_afrr_neg[b] <= 1
        2. y_total_ch[t] + y_fcr[b] + y_afrr_pos[b] <= 1

        These prevent conflicting bids like:
        - Discharging in DA/aFRR-E while reserving capacity for charging AS (FCR/aFRR-neg)
        - Charging in DA/aFRR-E while reserving capacity for discharging AS (FCR/aFRR-pos)

        Returns:
            Dict with violation counts and details
        """
        logger.info("Checking Cst-8: Cross-Market Mutual Exclusivity...")

        violations_discharge = []
        violations_charge = []

        # Extract binary variables from solution
        y_total_ch = self.solution.get('y_total_ch', {})
        y_total_dis = self.solution.get('y_total_dis', {})
        y_fcr = self.solution.get('y_fcr', {})
        y_afrr_pos = self.solution.get('y_afrr_pos', {})
        y_afrr_neg = self.solution.get('y_afrr_neg', {})

        # Check if binaries are available
        if not y_total_ch and not y_total_dis:
            logger.warning("Total binaries (y_total_ch, y_total_dis) not found in solution. Skipping Cst-8 check.")
            return {
                'checked': False,
                'reason': 'Total binaries not available (likely commented out in model)',
                'violations_discharge': [],
                'violations_charge': [],
                'count_discharge': 0,
                'count_charge': 0,
            }

        # Check for each time step
        for t in self.model.T:
            # Get block for this time step
            block = int(pyo.value(self.model.block_map[t]))

            # Extract binary values (default to 0 if not found)
            y_t_dis = y_total_dis.get(t, 0)
            y_t_ch = y_total_ch.get(t, 0)
            y_b_fcr = y_fcr.get(block, 0)
            y_b_afrr_pos = y_afrr_pos.get(block, 0)
            y_b_afrr_neg = y_afrr_neg.get(block, 0)

            # Check Constraint 1: y_total_dis + y_fcr + y_afrr_neg <= 1
            sum_discharge = y_t_dis + y_b_fcr + y_b_afrr_neg
            if sum_discharge > 1 + self.tolerance:
                violations_discharge.append({
                    'time': t,
                    'block': block,
                    'y_total_dis': y_t_dis,
                    'y_fcr': y_b_fcr,
                    'y_afrr_neg': y_b_afrr_neg,
                    'sum': sum_discharge,
                    'violation': sum_discharge - 1,
                })

            # Check Constraint 2: y_total_ch + y_fcr + y_afrr_pos <= 1
            sum_charge = y_t_ch + y_b_fcr + y_b_afrr_pos
            if sum_charge > 1 + self.tolerance:
                violations_charge.append({
                    'time': t,
                    'block': block,
                    'y_total_ch': y_t_ch,
                    'y_fcr': y_b_fcr,
                    'y_afrr_pos': y_b_afrr_pos,
                    'sum': sum_charge,
                    'violation': sum_charge - 1,
                })

        result = {
            'checked': True,
            'violations_discharge': violations_discharge,
            'violations_charge': violations_charge,
            'count_discharge': len(violations_discharge),
            'count_charge': len(violations_charge),
        }

        if violations_discharge:
            logger.warning(
                "Found %d violations of Cst-8 (discharge): "
                "y_total_dis + y_fcr + y_afrr_neg > 1",
                len(violations_discharge)
            )

        if violations_charge:
            logger.warning(
                "Found %d violations of Cst-8 (charge): "
                "y_total_ch + y_fcr + y_afrr_pos > 1",
                len(violations_charge)
            )

        if not violations_discharge and not violations_charge:
            logger.info("Cst-8 validation passed: No cross-market exclusivity violations")

        return result

    def check_cst9_min_bid_sizes(self) -> Dict[str, Any]:
        """Validate Cst-9: Minimum Bid Size Constraints.

        Checks minimum bid sizes for DA and aFRR energy markets:
        - DA charge: If p_ch[t] > 0, then p_ch[t] >= MinBid_da (0.1 MW = 100 kW)
        - DA discharge: If p_dis[t] > 0, then p_dis[t] >= MinBid_da
        - aFRR-E positive: If p_afrr_pos_e[t] > 0, then p_afrr_pos_e[t] >= MinBid_afrr_e
        - aFRR-E negative: If p_afrr_neg_e[t] > 0, then p_afrr_neg_e[t] >= MinBid_afrr_e

        Returns:
            Dict with violation counts and details
        """
        logger.info("Checking Cst-9: Minimum Bid Size Constraints...")

        violations = {
            'p_ch': [],
            'p_dis': [],
            'p_afrr_pos_e': [],
            'p_afrr_neg_e': [],
        }

        # Extract power variables from solution
        p_ch = self.solution.get('p_ch', {})
        p_dis = self.solution.get('p_dis', {})
        p_afrr_pos_e = self.solution.get('p_afrr_pos_e', {})
        p_afrr_neg_e = self.solution.get('p_afrr_neg_e', {})

        # Check DA charge bids
        for t, power in p_ch.items():
            if power > self.tolerance and power < self.min_bid_da - self.tolerance:
                violations['p_ch'].append({
                    'time': t,
                    'power_kw': power,
                    'min_required_kw': self.min_bid_da,
                    'violation_kw': self.min_bid_da - power,
                })

        # Check DA discharge bids
        for t, power in p_dis.items():
            if power > self.tolerance and power < self.min_bid_da - self.tolerance:
                violations['p_dis'].append({
                    'time': t,
                    'power_kw': power,
                    'min_required_kw': self.min_bid_da,
                    'violation_kw': self.min_bid_da - power,
                })

        # Check aFRR-E positive bids
        for t, power in p_afrr_pos_e.items():
            if power > self.tolerance and power < self.min_bid_afrr_e - self.tolerance:
                violations['p_afrr_pos_e'].append({
                    'time': t,
                    'power_kw': power,
                    'min_required_kw': self.min_bid_afrr_e,
                    'violation_kw': self.min_bid_afrr_e - power,
                })

        # Check aFRR-E negative bids
        for t, power in p_afrr_neg_e.items():
            if power > self.tolerance and power < self.min_bid_afrr_e - self.tolerance:
                violations['p_afrr_neg_e'].append({
                    'time': t,
                    'power_kw': power,
                    'min_required_kw': self.min_bid_afrr_e,
                    'violation_kw': self.min_bid_afrr_e - power,
                })

        # Count total violations
        total_violations = sum(len(v) for v in violations.values())

        result = {
            'checked': True,
            'violations': violations,
            'count_p_ch': len(violations['p_ch']),
            'count_p_dis': len(violations['p_dis']),
            'count_p_afrr_pos_e': len(violations['p_afrr_pos_e']),
            'count_p_afrr_neg_e': len(violations['p_afrr_neg_e']),
            'total_count': total_violations,
        }

        if total_violations > 0:
            logger.warning(
                "Found %d minimum bid size violations: "
                "p_ch=%d, p_dis=%d, p_afrr_pos_e=%d, p_afrr_neg_e=%d",
                total_violations,
                len(violations['p_ch']),
                len(violations['p_dis']),
                len(violations['p_afrr_pos_e']),
                len(violations['p_afrr_neg_e'])
            )
        else:
            logger.info("Cst-9 validation passed: No minimum bid size violations")

        return result

    def check_calendar_aging_sos2(self) -> Dict[str, Any]:
        """Validate calendar aging SOS2 properties.

        For Model (iii), checks:
        1. At most 2 adjacent λ_cal[t,i] variables are non-zero (SOS2 property)
        2. Sum of λ_cal[t,i] equals 1 for all t

        Returns:
            Dict with violation counts and details
        """
        # Check if this is Model (iii)
        if not hasattr(self.model, 'lambda_cal'):
            return {
                'checked': False,
                'reason': 'Calendar aging not enabled (Model I or II)',
            }

        logger.info("Checking calendar aging SOS2 properties...")

        violations_sos2 = []
        violations_sum = []

        lambda_cal = self.solution.get('lambda_cal', {})

        for t in self.model.T:
            # Extract non-zero lambda values for this timestep
            lambdas_t = {}
            for i in self.model.I:
                val = lambda_cal.get((t, i), 0)
                if abs(val) > self.tolerance:
                    lambdas_t[i] = val

            # Check SOS2 property: At most 2 adjacent variables non-zero
            if len(lambdas_t) > 2:
                violations_sos2.append({
                    'time': t,
                    'num_nonzero': len(lambdas_t),
                    'nonzero_indices': list(lambdas_t.keys()),
                    'values': lambdas_t,
                })
            elif len(lambdas_t) == 2:
                # Check if they are adjacent
                indices = sorted(lambdas_t.keys())
                if indices[1] - indices[0] != 1:
                    violations_sos2.append({
                        'time': t,
                        'num_nonzero': 2,
                        'nonzero_indices': indices,
                        'values': lambdas_t,
                        'reason': 'Non-adjacent variables',
                    })

            # Check sum constraint: Σλ = 1
            lambda_sum = sum(lambdas_t.values())
            if abs(lambda_sum - 1.0) > self.tolerance:
                violations_sum.append({
                    'time': t,
                    'sum': lambda_sum,
                    'violation': abs(lambda_sum - 1.0),
                })

        result = {
            'checked': True,
            'violations_sos2': violations_sos2,
            'violations_sum': violations_sum,
            'count_sos2': len(violations_sos2),
            'count_sum': len(violations_sum),
        }

        if violations_sos2:
            logger.warning(
                "Found %d SOS2 violations in calendar aging",
                len(violations_sos2)
            )

        if violations_sum:
            logger.warning(
                "Found %d sum violations in calendar aging (Σλ != 1)",
                len(violations_sum)
            )

        if not violations_sos2 and not violations_sum:
            logger.info("Calendar aging SOS2 validation passed")

        return result

    def check_segment_ordering(self) -> Dict[str, Any]:
        """Validate segment SOC ordering (stacked tank constraint).

        For Model (ii) and (iii), checks:
        e_soc_j[t,j] >= e_soc_j[t,j+1] for all j

        This ensures the "stacked tank" model is physically consistent:
        shallower segments must be at least as full as deeper segments.

        Returns:
            Dict with violation counts and details
        """
        # Check if this is Model (ii) or (iii)
        if not hasattr(self.model, 'e_soc_j'):
            return {
                'checked': False,
                'reason': 'Segment variables not present (Model I)',
            }

        logger.info("Checking segment SOC ordering (stacked tank)...")

        violations = []
        e_soc_j = self.solution.get('e_soc_j', {})

        for t in self.model.T:
            for j in self.model.J:
                if j == max(self.model.J):  # Skip last segment (deepest)
                    continue

                soc_j = e_soc_j.get((t, j), 0)
                soc_j_plus_1 = e_soc_j.get((t, j + 1), 0)

                # Check e_soc_j[t,j] >= e_soc_j[t,j+1]
                if soc_j < soc_j_plus_1 - self.tolerance:
                    violations.append({
                        'time': t,
                        'segment': j,
                        'soc_j': soc_j,
                        'soc_j_plus_1': soc_j_plus_1,
                        'violation': soc_j_plus_1 - soc_j,
                    })

        result = {
            'checked': True,
            'violations': violations,
            'count': len(violations),
        }

        if violations:
            logger.warning(
                "Found %d segment ordering violations (e_soc_j[j] < e_soc_j[j+1])",
                len(violations)
            )
        else:
            logger.info("Segment ordering validation passed")

        return result

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report.

        Runs all validation checks and aggregates results.

        Returns:
            Dict containing all validation results and summary
        """
        logger.info("=" * 80)
        logger.info("CONSTRAINT VALIDATION REPORT")
        logger.info("=" * 80)

        report = {
            'cst8_cross_market': self.check_cst8_cross_market_exclusivity(),
            'cst9_min_bids': self.check_cst9_min_bid_sizes(),
            'calendar_sos2': self.check_calendar_aging_sos2(),
            'segment_ordering': self.check_segment_ordering(),
        }

        # Calculate total violations
        total_violations = 0

        if report['cst8_cross_market']['checked']:
            total_violations += report['cst8_cross_market']['count_discharge']
            total_violations += report['cst8_cross_market']['count_charge']

        if report['cst9_min_bids']['checked']:
            total_violations += report['cst9_min_bids']['total_count']

        if report['calendar_sos2']['checked']:
            total_violations += report['calendar_sos2']['count_sos2']
            total_violations += report['calendar_sos2']['count_sum']

        if report['segment_ordering']['checked']:
            total_violations += report['segment_ordering']['count']

        report['summary'] = {
            'total_violations': total_violations,
            'checks_performed': sum(1 for check in report.values()
                                   if isinstance(check, dict) and check.get('checked', False)),
            'all_passed': total_violations == 0,
        }

        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("  Total violations: %d", total_violations)
        logger.info("  Checks performed: %d", report['summary']['checks_performed'])
        logger.info("  Status: %s", "PASS" if total_violations == 0 else "FAIL")
        logger.info("=" * 80)

        return report


def validate_solution(
    model: pyo.ConcreteModel,
    solution: Dict[str, Any],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """Convenience function to validate a solution.

    Args:
        model: Solved Pyomo model
        solution: Solution dictionary from solve_model()
        tolerance: Numerical tolerance

    Returns:
        Validation report dict
    """
    validator = ConstraintValidator(model, solution, tolerance)
    return validator.generate_validation_report()
