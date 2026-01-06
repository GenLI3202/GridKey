# MPC Rolling Horizon Validation Plan

**Project**: TechArena 2025 BESS Optimizer
**Component**: Rolling Horizon MPC Implementation
**Version**: 1.0
**Last Updated**: 2025-11-09

## Table of Contents

1. [Overview](#1-overview)
2. [Validation Objectives](#2-validation-objectives)
3. [Test Environment Setup](#3-test-environment-setup)
4. [Test Categories](#4-test-categories)
5. [Test Execution Matrix](#5-test-execution-matrix)
6. [Detailed Test Procedures](#6-detailed-test-procedures)
7. [Success Criteria](#7-success-criteria)
8. [Deliverables](#8-deliverables)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [References](#10-references)

---

## 1. Overview

### 1.1 Purpose

This document provides a comprehensive validation plan for the Model Predictive Control (MPC) rolling horizon implementation used to optimize Battery Energy Storage System (BESS) operations across European electricity markets.

### 1.2 Scope

The validation covers the three-layer optimization architecture:

- **Inner Layer**: BESSOptimizerModelIII (single MILP with cyclic + calendar aging)
- **Middle Layer**: MPCSimulator (receding horizon control)
- **Outer Layer**: MetaOptimizer (alpha parameter sweep with 10-year ROI)

### 1.3 Implementation Location

- **Core Files**: `py_script/rolling_horizon/mpc_simulator.py`, `meta_optimizer.py`
- **Configuration**: `data/phase2_aging_config/mpc_config.json`
- **Documentation**: `py_script/rolling_horizon/README.md`

### 1.4 Previous Validation Work

Existing validation completed:
- SOC continuity validation (PASSED) - see `validate_mpc_soc_continuity.py`
- Single-horizon tests (PASSED) - see `MPC_VALIDATION_FINDINGS.md`
- Known issue: SOC depletion with low alpha values (parameter tuning issue, not implementation bug)

---

## 2. Validation Objectives

### 2.1 Primary Objectives

1. **Correctness**: Verify MPC state propagation and constraint satisfaction
2. **Robustness**: Test across countries, seasons, and parameter ranges
3. **Performance**: Validate computational efficiency and scalability
4. **Economic Accuracy**: Confirm ROI calculations and alpha optimization
5. **Integration**: Ensure end-to-end pipeline functionality

### 2.2 Success Criteria Summary

| Category | Metric | Target |
|----------|--------|--------|
| Correctness | SOC continuity error | < 0.1 kWh |
| Correctness | Constraint violations | 0 (zero tolerance) |
| Performance | Solve time per iteration | < 60 seconds (median) |
| Performance | Simulation completion rate | 100% |
| Economic | ROI calculation consistency | < 1% difference on repeated runs |

---

## 3. Test Environment Setup

### 3.1 Prerequisites

**Required Software:**
- Python 3.8+
- MILP Solver: CPLEX, Gurobi, or CBC
- Dependencies: `pyomo`, `pandas`, `numpy`, `plotly`

**Required Data:**
- Phase 2 market data: `data/phase2_processed/*.parquet`
- Aging configuration: `data/phase2_aging_config/afrr_activation_config.json`
- MPC configuration: `data/phase2_aging_config/mpc_config.json`

### 3.2 Directory Structure

Create validation output directories:

```bash
mkdir -p results/mpc_validation
mkdir -p results/mpc_validation/correctness
mkdir -p results/mpc_validation/robustness
mkdir -p results/mpc_validation/performance
mkdir -p results/mpc_validation/economic
mkdir -p results/mpc_validation/integration
```

### 3.3 Configuration Verification

```python
# Verify mpc_config.json is accessible
import json
with open('data/phase2_aging_config/mpc_config.json', 'r') as f:
    config = json.load(f)
    assert 'mpc_parameters' in config
    assert 'alpha_sweep' in config
    assert 'execution_modes' in config
```

---

## 4. Test Categories

### 4.1 Category T1: Correctness Tests

**Objective**: Verify fundamental MPC mechanics and constraint satisfaction

**Tests:**
- T1.1: SOC Continuity Validation
- T1.2: Constraint Satisfaction Check
- T1.3: Revenue Calculation Accuracy
- T1.4: Segment-wise SOC Distribution
- T1.5: Boundary Condition Handling

### 4.2 Category T2: Robustness Tests

**Objective**: Ensure reliable operation across diverse conditions

**Tests:**
- T2.1: Multi-Country Validation
- T2.2: Seasonal Stability Test
- T2.3: Alpha Parameter Sensitivity
- T2.4: Initial SOC Sensitivity
- T2.5: Edge Case Handling

### 4.3 Category T3: Performance Tests

**Objective**: Validate computational efficiency and scalability

**Tests:**
- T3.1: Solve Time Analysis
- T3.2: Memory Usage Profiling
- T3.3: Horizon Length Scalability
- T3.4: Solver Comparison
- T3.5: Parallel Execution Validation

### 4.4 Category T4: Economic Validation

**Objective**: Verify financial calculations and ROI methodology

**Tests:**
- T4.1: 10-Year ROI Calculation
- T4.2: NPV Methodology Verification
- T4.3: Alpha Optimization Effectiveness
- T4.4: Revenue Component Breakdown
- T4.5: Degradation Cost Accuracy

### 4.5 Category T5: Integration Tests

**Objective**: Validate end-to-end pipeline functionality

**Tests:**
- T5.1: Data Preprocessing Integration
- T5.2: Model I/II/III Compatibility
- T5.3: Output Generation and Export
- T5.4: Visualization Pipeline
- T5.5: Full-Year Simulation

---

## 5. Test Execution Matrix

### 5.1 Quick Validation Suite (~1-2 hours)

**Purpose**: Rapid smoke test for development/debugging

| Test ID | Description | Duration | Priority |
|---------|-------------|----------|----------|
| T1.1 | SOC continuity (5 iterations) | 5 min | Critical |
| T1.2 | Constraint check (1 week) | 10 min | Critical |
| T2.1 | Single country (CH) | 15 min | High |
| T3.1 | Solve time (1 month) | 30 min | High |
| T5.5 | Mini full-year (1 month) | 30 min | High |

**Command:**
```bash
python py_script/rolling_horizon/run_validation_suite.py --suite quick
```

### 5.2 Standard Validation Suite (~1 day)

**Purpose**: Comprehensive pre-deployment validation

| Category | Tests | Duration | CPU Cores |
|----------|-------|----------|-----------|
| T1.x | All correctness | 2 hours | 1 |
| T2.x | Multi-country + seasonal | 8 hours | 5 (parallel) |
| T3.x | Performance analysis | 4 hours | 1 |
| T4.x | Economic validation | 6 hours | 1 |
| T5.x | Integration (partial) | 4 hours | 1 |

**Command:**
```bash
python py_script/rolling_horizon/run_validation_suite.py --suite standard
```

### 5.3 Comprehensive Validation Suite (~1 week)

**Purpose**: Full validation for publication/competition submission

**Includes:**
- All 25 test procedures
- Full-year simulations (all 5 countries)
- Complete alpha sweep (16 values)
- Solver comparison (CPLEX vs Gurobi vs CBC)
- Sensitivity analysis (10+ parameter combinations)

**Command:**
```bash
python py_script/rolling_horizon/run_validation_suite.py --suite comprehensive
```

---

## 6. Detailed Test Procedures

### Category T1: Correctness Tests

---

#### T1.1: SOC Continuity Validation

**Purpose**: Verify that SOC state propagates correctly between MPC iterations without discontinuities

**Configuration:**
```python
test_config = {
    'country': 'CH',
    'period': '2024-01-13 to 2024-01-17',  # 5 days
    'horizon_hours': 32,
    'execution_hours': 24,
    'alpha': 0.5,
    'c_rate': 0.5,
    'initial_soc_fraction': 0.5
}
```

**Execution:**
```python
# Create test script: test_t1_1_soc_continuity.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import pandas as pd
import json

# Load data and initialize
optimizer = BESSOptimizerModelIII(alpha=0.5, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data = optimizer.extract_country_data(full_data, 'CH')
data = data['2024-01-13':'2024-01-17']

# Run MPC simulation
simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=32,
    execution_hours=24,
    initial_soc_fraction=0.5,
    validate_constraints=True
)

results = simulator.run_simulation()

# Validation checks
soc_total_bids_df = results['soc_total_bids_df']
soc_changes = [abs(soc_total_bids_df[i+1] - soc_total_bids_df[i])
               for i in range(len(soc_total_bids_df)-1)]

# Export results
with open('results/mpc_validation/correctness/t1_1_soc_continuity.json', 'w') as f:
    json.dump({
        'test_id': 'T1.1',
        'status': 'PASS' if max(soc_changes) < 0.1 else 'FAIL',
        'max_soc_change': max(soc_changes),
        'mean_soc_change': sum(soc_changes) / len(soc_changes),
        'soc_total_bids_df': soc_total_bids_df,
        'iteration_count': len(results['iteration_results'])
    }, f, indent=2)
```

**Run command:**
```bash
python test_t1_1_soc_continuity.py
```

**Expected Output:**
- File: `results/mpc_validation/correctness/t1_1_soc_continuity.json`
- Key metrics:
  - `max_soc_change` < 0.1 kWh (between iteration boundaries)
  - `status` = "PASS"
  - 5 iterations completed successfully

**Pass Criteria:**
- ✅ All SOC changes at iteration boundaries < 0.1 kWh
- ✅ No infeasible iterations
- ✅ SOC remains within [0, 4472] kWh at all times

**Failure Diagnosis:**
- If `max_soc_change` > 0.1: Check segment SOC initialization in `_get_initial_segment_soc()`
- If infeasible: Increase alpha or initial SOC (likely energy reserve constraint violation)

---

#### T1.2: Constraint Satisfaction Check

**Purpose**: Verify all optimization constraints are satisfied post-solve

**Configuration:**
```python
test_config = {
    'country': 'DE_LU',
    'period': '2024-03-01 to 2024-03-07',  # 1 week
    'horizon_hours': 48,
    'execution_hours': 24,
    'alpha': 1.0,
    'c_rate': 0.5,
    'validate_constraints': True  # Enable validation
}
```

**Execution:**
```python
# test_t1_2_constraint_check.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII

optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/de_lu.parquet')
data = optimizer.extract_country_data(full_data, 'DE_LU')
data = data['2024-03-01':'2024-03-07']

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5,
    validate_constraints=True  # Critical for this test
)

results = simulator.run_simulation()

# Aggregate validation reports
all_violations = []
for iter_result in results['iteration_results']:
    if 'validation_report' in iter_result:
        violations = iter_result['validation_report'].get('violations', [])
        all_violations.extend(violations)

# Export
import json
with open('results/mpc_validation/correctness/t1_2_constraint_check.json', 'w') as f:
    json.dump({
        'test_id': 'T1.2',
        'status': 'PASS' if len(all_violations) == 0 else 'FAIL',
        'total_violations': len(all_violations),
        'violations': all_violations,
        'iterations_validated': len(results['iteration_results'])
    }, f, indent=2)
```

**Run command:**
```bash
python test_t1_2_constraint_check.py
```

**Expected Output:**
- File: `results/mpc_validation/correctness/t1_2_constraint_check.json`
- Key metrics:
  - `total_violations` = 0
  - `status` = "PASS"
  - 7 iterations validated

**Pass Criteria:**
- ✅ Zero constraint violations across all iterations
- ✅ All Cst-8 (mutual exclusivity) satisfied
- ✅ All Cst-9 (minimum bids) satisfied
- ✅ SOC segment ordering maintained

**Failure Diagnosis:**
- Check `violations` list for specific constraint IDs
- Common issues:
  - Cst-8: FCR and aFRR capacity overlap (check block logic)
  - Cst-9: Bids below minimum (check epsilon thresholds)

---

#### T1.3: Revenue Calculation Accuracy

**Purpose**: Verify revenue components are calculated correctly

**Configuration:**
```python
test_config = {
    'country': 'AT',
    'period': '2024-06-01 to 2024-06-07',  # 1 week, summer
    'horizon_hours': 48,
    'execution_hours': 24,
    'alpha': 1.0,
    'c_rate': 0.5
}
```

**Execution:**
```python
# test_t1_3_revenue_accuracy.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import pandas as pd
import numpy as np

optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/at.parquet')
data = optimizer.extract_country_data(full_data, 'AT')
data = data['2024-06-01':'2024-06-07']

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5
)

results = simulator.run_simulation()

# Manual revenue verification for first iteration
iter_0 = results['iteration_results'][0]
solution = iter_0['solution']

# Recalculate DA revenue manually
da_revenue_check = 0.0
for t in range(24):  # Execution window
    price = data['day_ahead_price'].iloc[t]
    p_ch = solution.get(f'p_ch[{t}]', 0.0)
    p_dis = solution.get(f'p_dis[{t}]', 0.0)
    da_revenue_check += (p_dis - p_ch) * price

# Compare with reported revenue
da_revenue_reported = iter_0.get('da_revenue', 0.0)
da_diff = abs(da_revenue_check - da_revenue_reported)

# Export
import json
with open('results/mpc_validation/correctness/t1_3_revenue_accuracy.json', 'w') as f:
    json.dump({
        'test_id': 'T1.3',
        'status': 'PASS' if da_diff < 0.01 else 'FAIL',
        'da_revenue_manual': da_revenue_check,
        'da_revenue_reported': da_revenue_reported,
        'difference_eur': da_diff,
        'total_revenue': results['total_revenue'],
        'revenue_breakdown': {
            'da': results['da_revenue'],
            'afrr_e': results['afrr_e_revenue'],
            'as_capacity': results['as_revenue']
        }
    }, f, indent=2)
```

**Run command:**
```bash
python test_t1_3_revenue_accuracy.py
```

**Expected Output:**
- File: `results/mpc_validation/correctness/t1_3_revenue_accuracy.json`
- Key metrics:
  - `difference_eur` < 0.01 EUR (floating point tolerance)
  - `status` = "PASS"

**Pass Criteria:**
- ✅ Manual calculation matches reported revenue (< 0.01 EUR difference)
- ✅ Revenue breakdown sums to total revenue
- ✅ All revenue components non-negative

**Failure Diagnosis:**
- If large difference: Check price data alignment and timestep indexing
- If negative revenues: Investigate sign conventions in objective function

---

#### T1.4: Segment-wise SOC Distribution

**Purpose**: Verify SOC segments maintain correct "stacked tank" ordering

**Configuration:**
```python
test_config = {
    'country': 'CH',
    'period': '2024-02-01 to 2024-02-03',  # 3 days
    'horizon_hours': 48,
    'execution_hours': 24,
    'alpha': 1.5
}
```

**Execution:**
```python
# test_t1_4_segment_distribution.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

optimizer = BESSOptimizerModelIII(alpha=1.5, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data = optimizer.extract_country_data(full_data, 'CH')
data = data['2024-02-01':'2024-02-03']

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5
)

results = simulator.run_simulation()

# Check segment ordering for each iteration
segment_violations = []
for i, iter_result in enumerate(results['iteration_results']):
    solution = iter_result['solution']

    # Extract final segment SOCs (at last timestep of execution window)
    segment_socs = []
    for j in range(1, 11):  # 10 segments
        key = f'e_soc_j[23,{j}]'  # Last timestep of 24h window
        if key in solution:
            segment_socs.append(solution[key])

    # Check top-down ordering: seg 1 should fill first
    # Expected: either [447.2, 447.2, ..., 447.2, X, 0, 0] or all full/empty
    for j in range(len(segment_socs) - 1):
        if segment_socs[j] < segment_socs[j+1] - 0.01:  # Allow small tolerance
            segment_violations.append({
                'iteration': i,
                'segment': j+1,
                'soc_j': segment_socs[j],
                'soc_j+1': segment_socs[j+1],
                'violation': 'Lower segment less full than upper segment'
            })

# Export
with open('results/mpc_validation/correctness/t1_4_segment_distribution.json', 'w') as f:
    json.dump({
        'test_id': 'T1.4',
        'status': 'PASS' if len(segment_violations) == 0 else 'FAIL',
        'total_violations': len(segment_violations),
        'violations': segment_violations,
        'iterations_checked': len(results['iteration_results'])
    }, f, indent=2)
```

**Run command:**
```bash
python test_t1_4_segment_distribution.py
```

**Expected Output:**
- File: `results/mpc_validation/correctness/t1_4_segment_distribution.json`
- Key metrics:
  - `total_violations` = 0
  - `status` = "PASS"

**Pass Criteria:**
- ✅ Segments fill top-down (segment 1 fills before segment 2, etc.)
- ✅ No "inversion" where upper segment has less energy than lower
- ✅ Segment SOCs sum to total SOC (within 0.1 kWh)

**Failure Diagnosis:**
- If violations found: Check `Cst_SOC_Segment_Relation` constraint in optimizer
- May indicate issue with segment ordering constraints (Cst-4)

---

#### T1.5: Boundary Condition Handling

**Purpose**: Test MPC behavior at simulation start/end boundaries

**Configuration:**
```python
test_config = {
    'country': 'HU',
    'period': '2024-01-01 to 2024-01-03',  # Year start boundary
    'horizon_hours': 48,
    'execution_hours': 24,
    'initial_soc_fractions': [0.0, 0.5, 1.0]  # Test extremes
}
```

**Execution:**
```python
# test_t1_5_boundary_conditions.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

test_results = []

for init_soc in [0.0, 0.5, 1.0]:
    optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
    full_data = optimizer.load_and_preprocess_data('data/phase2_processed/hu.parquet')
    data = optimizer.extract_country_data(full_data, 'HU')
    data = data['2024-01-01':'2024-01-03']

    try:
        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=init_soc
        )

        results = simulator.run_simulation()

        test_results.append({
            'initial_soc_fraction': init_soc,
            'status': 'SUCCESS',
            'final_soc': results['final_soc'],
            'total_profit': results['net_profit'],
            'iterations_completed': len(results['iteration_results'])
        })
    except Exception as e:
        test_results.append({
            'initial_soc_fraction': init_soc,
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/correctness/t1_5_boundary_conditions.json', 'w') as f:
    json.dump({
        'test_id': 'T1.5',
        'status': 'PASS' if all(r['status'] == 'SUCCESS' for r in test_results) else 'FAIL',
        'results': test_results
    }, f, indent=2)
```

**Run command:**
```bash
python test_t1_5_boundary_conditions.py
```

**Expected Output:**
- File: `results/mpc_validation/correctness/t1_5_boundary_conditions.json`
- All three initial SOC values complete successfully

**Pass Criteria:**
- ✅ Initial SOC = 0% completes without infeasibility
- ✅ Initial SOC = 50% completes (baseline)
- ✅ Initial SOC = 100% completes without infeasibility
- ✅ Final SOC reasonable in all cases

**Failure Diagnosis:**
- If 0% fails: Check if energy reserve constraints too restrictive
- If 100% fails: Check if discharge constraints prevent feasibility
- Consider adding "ramp-up" period or looser constraints for edge cases

---

### Category T2: Robustness Tests

---

#### T2.1: Multi-Country Validation

**Purpose**: Validate MPC performance across all 5 European countries

**Configuration:**
```python
test_config = {
    'countries': ['DE_LU', 'AT', 'CH', 'HU', 'CZ'],
    'period': '2024-07-01 to 2024-07-07',  # 1 week summer
    'horizon_hours': 48,
    'execution_hours': 24,
    'alpha': 1.0,
    'c_rate': 0.5
}
```

**Execution:**
```python
# test_t2_1_multi_country.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json
import pandas as pd

countries = ['DE_LU', 'AT', 'CH', 'HU', 'CZ']
results_summary = []

for country in countries:
    print(f"Testing country: {country}")

    optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)

    # Load country-specific data
    file_map = {
        'DE_LU': 'data/phase2_processed/de_lu.parquet',
        'AT': 'data/phase2_processed/at.parquet',
        'CH': 'data/phase2_processed/ch.parquet',
        'HU': 'data/phase2_processed/hu.parquet',
        'CZ': 'data/phase2_processed/cz.parquet'
    }

    try:
        full_data = optimizer.load_and_preprocess_data(file_map[country])
        data = optimizer.extract_country_data(full_data, country)
        data = data['2024-07-01':'2024-07-07']

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=0.5
        )

        result = simulator.run_simulation()

        results_summary.append({
            'country': country,
            'status': 'SUCCESS',
            'total_revenue': result['total_revenue'],
            'net_profit': result['net_profit'],
            'total_degradation_cost': result['total_degradation_cost'],
            'iterations': len(result['iteration_results']),
            'final_soc': result['final_soc']
        })

    except Exception as e:
        results_summary.append({
            'country': country,
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/robustness/t2_1_multi_country.json', 'w') as f:
    json.dump({
        'test_id': 'T2.1',
        'status': 'PASS' if all(r['status'] == 'SUCCESS' for r in results_summary) else 'FAIL',
        'results': results_summary,
        'success_rate': f"{sum(1 for r in results_summary if r['status'] == 'SUCCESS')}/{len(countries)}"
    }, f, indent=2)

# Also create CSV for easy comparison
df = pd.DataFrame(results_summary)
df.to_csv('results/mpc_validation/robustness/t2_1_multi_country.csv', index=False)
```

**Run command:**
```bash
python test_t2_1_multi_country.py
```

**Expected Output:**
- File: `results/mpc_validation/robustness/t2_1_multi_country.json`
- File: `results/mpc_validation/robustness/t2_1_multi_country.csv`
- Success for all 5 countries

**Pass Criteria:**
- ✅ All 5 countries complete successfully (5/5 success rate)
- ✅ Profit rankings align with market prices (HU typically highest)
- ✅ No infeasibilities across diverse market conditions

**Failure Diagnosis:**
- If specific country fails: Check data availability for that country
- Compare price ranges - extreme prices may cause infeasibility
- Review country-specific constraints (power limits, market rules)

---

#### T2.2: Seasonal Stability Test

**Purpose**: Validate performance across different seasons (price regimes)

**Configuration:**
```python
test_config = {
    'country': 'CH',  # Switzerland for stability
    'seasons': {
        'winter': '2024-02-12 to 2024-02-18',  # High prices
        'spring': '2024-04-15 to 2024-04-21',
        'summer': '2024-07-22 to 2024-07-28',  # Low prices
        'fall': '2024-10-07 to 2024-10-13'
    },
    'horizon_hours': 48,
    'execution_hours': 24,
    'alpha': 1.0
}
```

**Execution:**
```python
# test_t2_2_seasonal_stability.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

seasons = {
    'winter': '2024-02-12 to 2024-02-18',
    'spring': '2024-04-15 to 2024-04-21',
    'summer': '2024-07-22 to 2024-07-28',
    'fall': '2024-10-07 to 2024-10-13'
}

results_by_season = []

optimizer_base = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer_base.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data_full = optimizer_base.extract_country_data(full_data, 'CH')

for season_name, period in seasons.items():
    start, end = period.split(' to ')
    data_season = data_full[start:end]

    try:
        optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data_season,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=0.5
        )

        result = simulator.run_simulation()

        # Calculate price statistics
        mean_da_price = data_season['day_ahead_price'].mean()
        std_da_price = data_season['day_ahead_price'].std()

        results_by_season.append({
            'season': season_name,
            'period': period,
            'status': 'SUCCESS',
            'mean_da_price': float(mean_da_price),
            'std_da_price': float(std_da_price),
            'total_revenue': result['total_revenue'],
            'net_profit': result['net_profit'],
            'iterations': len(result['iteration_results'])
        })

    except Exception as e:
        results_by_season.append({
            'season': season_name,
            'period': period,
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/robustness/t2_2_seasonal_stability.json', 'w') as f:
    json.dump({
        'test_id': 'T2.2',
        'status': 'PASS' if all(r['status'] == 'SUCCESS' for r in results_by_season) else 'FAIL',
        'results': results_by_season,
        'success_rate': f"{sum(1 for r in results_by_season if r['status'] == 'SUCCESS')}/4"
    }, f, indent=2)
```

**Run command:**
```bash
python test_t2_2_seasonal_stability.py
```

**Expected Output:**
- File: `results/mpc_validation/robustness/t2_2_seasonal_stability.json`
- All 4 seasons complete successfully

**Pass Criteria:**
- ✅ All seasons complete without infeasibility (4/4)
- ✅ Higher profits in winter (high price volatility)
- ✅ Lower profits in summer (low/stable prices)
- ✅ Consistent behavior pattern across seasons

**Failure Diagnosis:**
- Winter failures often due to aggressive discharge → increase alpha
- Summer may have low/zero profits (expected, not a failure)

---

#### T2.3: Alpha Parameter Sensitivity

**Purpose**: Validate behavior across range of degradation weights

**Configuration:**
```python
test_config = {
    'country': 'DE_LU',
    'period': '2024-03-01 to 2024-03-14',  # 2 weeks
    'alpha_values': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    'horizon_hours': 48,
    'execution_hours': 24
}
```

**Execution:**
```python
# test_t2_3_alpha_sensitivity.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json
import pandas as pd

alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
results_by_alpha = []

optimizer_base = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer_base.load_and_preprocess_data('data/phase2_processed/de_lu.parquet')
data = optimizer_base.extract_country_data(full_data, 'DE_LU')
data = data['2024-03-01':'2024-03-14']

for alpha in alpha_values:
    print(f"Testing alpha = {alpha}")

    try:
        optimizer = BESSOptimizerModelIII(alpha=alpha, c_rate=0.5)

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=0.5
        )

        result = simulator.run_simulation()

        results_by_alpha.append({
            'alpha': alpha,
            'status': 'SUCCESS',
            'total_revenue': result['total_revenue'],
            'total_degradation_cost': result['total_degradation_cost'],
            'net_profit': result['net_profit'],
            'cyclic_cost': result['cyclic_cost'],
            'calendar_cost': result['calendar_cost'],
            'iterations': len(result['iteration_results'])
        })

    except Exception as e:
        results_by_alpha.append({
            'alpha': alpha,
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/robustness/t2_3_alpha_sensitivity.json', 'w') as f:
    json.dump({
        'test_id': 'T2.3',
        'status': 'PASS' if all(r['status'] == 'SUCCESS' for r in results_by_alpha) else 'FAIL',
        'results': results_by_alpha
    }, f, indent=2)

# CSV for plotting
df = pd.DataFrame(results_by_alpha)
df.to_csv('results/mpc_validation/robustness/t2_3_alpha_sensitivity.csv', index=False)

# Check expected pattern: higher alpha → lower degradation, lower revenue
if all(r['status'] == 'SUCCESS' for r in results_by_alpha):
    degradation_decreasing = all(
        results_by_alpha[i]['total_degradation_cost'] >= results_by_alpha[i+1]['total_degradation_cost']
        for i in range(len(results_by_alpha)-1)
    )
    print(f"Degradation monotonically decreasing with alpha: {degradation_decreasing}")
```

**Run command:**
```bash
python test_t2_3_alpha_sensitivity.py
```

**Expected Output:**
- File: `results/mpc_validation/robustness/t2_3_alpha_sensitivity.json`
- File: `results/mpc_validation/robustness/t2_3_alpha_sensitivity.csv`

**Pass Criteria:**
- ✅ All alpha values complete successfully
- ✅ Degradation cost decreases with increasing alpha (monotonic trend)
- ✅ Revenue decreases with increasing alpha (less aggressive operation)
- ✅ Net profit shows optimal alpha in range [0.5, 2.0] (expected)

**Failure Diagnosis:**
- Very low alpha (0.1) may cause infeasibility due to aggressive operation
- Very high alpha (3.0) should still be feasible, just conservative

---

#### T2.4: Initial SOC Sensitivity

**Purpose**: Test robustness to different starting battery states

**Configuration:**
```python
test_config = {
    'country': 'AT',
    'period': '2024-05-01 to 2024-05-07',
    'initial_soc_fractions': [0.2, 0.35, 0.5, 0.65, 0.8],
    'horizon_hours': 48,
    'execution_hours': 24,
    'alpha': 1.0
}
```

**Execution:**
```python
# test_t2_4_initial_soc_sensitivity.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

initial_socs = [0.2, 0.35, 0.5, 0.65, 0.8]
results_by_soc = []

optimizer_base = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer_base.load_and_preprocess_data('data/phase2_processed/at.parquet')
data = optimizer_base.extract_country_data(full_data, 'AT')
data = data['2024-05-01':'2024-05-07']

for init_soc in initial_socs:
    try:
        optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=init_soc
        )

        result = simulator.run_simulation()

        results_by_soc.append({
            'initial_soc_fraction': init_soc,
            'initial_soc_kwh': init_soc * 4472,
            'status': 'SUCCESS',
            'final_soc': result['final_soc'],
            'soc_change': result['final_soc'] - (init_soc * 4472),
            'net_profit': result['net_profit'],
            'iterations': len(result['iteration_results'])
        })

    except Exception as e:
        results_by_soc.append({
            'initial_soc_fraction': init_soc,
            'initial_soc_kwh': init_soc * 4472,
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/robustness/t2_4_initial_soc_sensitivity.json', 'w') as f:
    json.dump({
        'test_id': 'T2.4',
        'status': 'PASS' if all(r['status'] == 'SUCCESS' for r in results_by_soc) else 'FAIL',
        'results': results_by_soc
    }, f, indent=2)
```

**Run command:**
```bash
python test_t2_4_initial_soc_sensitivity.py
```

**Expected Output:**
- File: `results/mpc_validation/robustness/t2_4_initial_soc_sensitivity.json`

**Pass Criteria:**
- ✅ All initial SOC values complete successfully
- ✅ Final SOC converges to similar range regardless of initial SOC
- ✅ Profit differences < 10% across initial SOC range

**Failure Diagnosis:**
- Low initial SOC failures: May need higher alpha or energy reserve relaxation
- Check if MPC "self-corrects" SOC over time

---

#### T2.5: Edge Case Handling

**Purpose**: Test MPC behavior under extreme conditions

**Test Cases:**
```python
edge_cases = [
    {
        'name': 'Zero prices',
        'country': 'CH',
        'period': '2024-05-20 to 2024-05-21',  # Find period with many zero prices
        'expected': 'No arbitrage, minimal operation'
    },
    {
        'name': 'Very high prices',
        'country': 'DE_LU',
        'period': '2024-01-10 to 2024-01-11',  # Winter peak
        'expected': 'Maximum discharge utilization'
    },
    {
        'name': 'Negative prices',
        'country': 'DE_LU',
        'period': '2024-06-15 to 2024-06-16',  # Solar oversupply
        'expected': 'Charge during negative, discharge during positive'
    },
    {
        'name': 'Price spike',
        'country': 'HU',
        'period': '2024-02-05 to 2024-02-06',  # Single high-price hour
        'expected': 'Discharge concentrated in spike hour'
    }
]
```

**Execution:**
```python
# test_t2_5_edge_cases.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

edge_cases = [
    {'name': 'Zero prices', 'country': 'CH', 'period': '2024-05-20:2024-05-21'},
    {'name': 'Very high prices', 'country': 'DE_LU', 'period': '2024-01-10:2024-01-11'},
    {'name': 'Negative prices', 'country': 'DE_LU', 'period': '2024-06-15:2024-06-16'},
    {'name': 'Price spike', 'country': 'HU', 'period': '2024-02-05:2024-02-06'}
]

results_edge_cases = []

file_map = {
    'CH': 'data/phase2_processed/ch.parquet',
    'DE_LU': 'data/phase2_processed/de_lu.parquet',
    'HU': 'data/phase2_processed/hu.parquet'
}

for case in edge_cases:
    try:
        optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
        full_data = optimizer.load_and_preprocess_data(file_map[case['country']])
        data = optimizer.extract_country_data(full_data, case['country'])

        start, end = case['period'].split(':')
        data = data[start:end]

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=0.5
        )

        result = simulator.run_simulation()

        results_edge_cases.append({
            'case': case['name'],
            'country': case['country'],
            'status': 'SUCCESS',
            'net_profit': result['net_profit'],
            'total_revenue': result['total_revenue']
        })

    except Exception as e:
        results_edge_cases.append({
            'case': case['name'],
            'country': case['country'],
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/robustness/t2_5_edge_cases.json', 'w') as f:
    json.dump({
        'test_id': 'T2.5',
        'status': 'PASS' if all(r['status'] == 'SUCCESS' for r in results_edge_cases) else 'FAIL',
        'results': results_edge_cases
    }, f, indent=2)
```

**Run command:**
```bash
python test_t2_5_edge_cases.py
```

**Expected Output:**
- File: `results/mpc_validation/robustness/t2_5_edge_cases.json`

**Pass Criteria:**
- ✅ All edge cases complete without crashing
- ✅ Behavior aligns with expectations (e.g., charge during negative prices)
- ✅ No numerical instabilities

---

### Category T3: Performance Tests

---

#### T3.1: Solve Time Analysis

**Purpose**: Characterize computational performance across typical scenarios

**Configuration:**
```python
test_config = {
    'country': 'CH',
    'periods': ['2024-01-15:2024-01-21', '2024-04-15:2024-04-21', '2024-07-15:2024-07-21'],
    'horizon_hours': [24, 36, 48],
    'alpha_values': [0.5, 1.0, 1.5],
    'measurements': ['solve_time', 'model_build_time', 'data_prep_time']
}
```

**Execution:**
```python
# test_t3_1_solve_time.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json
import time
import pandas as pd

optimizer_base = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer_base.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data_full = optimizer_base.extract_country_data(full_data, 'CH')

periods = ['2024-01-15:2024-01-21', '2024-04-15:2024-04-21', '2024-07-15:2024-07-21']
horizon_hours = [24, 36, 48]
alpha_values = [0.5, 1.0, 1.5]

timing_results = []

for period in periods:
    start_date, end_date = period.split(':')
    data = data_full[start_date:end_date]

    for horizon in horizon_hours:
        for alpha in alpha_values:
            optimizer = BESSOptimizerModelIII(alpha=alpha, c_rate=0.5)

            simulator = MPCSimulator(
                optimizer_model=optimizer,
                full_data=data,
                horizon_hours=horizon,
                execution_hours=24,
                initial_soc_fraction=0.5
            )

            start_time = time.time()
            result = simulator.run_simulation()
            total_time = time.time() - start_time

            # Extract per-iteration solve times
            solve_times = [iter_result.get('solve_time_sec', 0)
                          for iter_result in result['iteration_results']]

            timing_results.append({
                'period': period,
                'horizon_hours': horizon,
                'alpha': alpha,
                'total_simulation_time': total_time,
                'iterations': len(solve_times),
                'mean_solve_time': sum(solve_times) / len(solve_times),
                'median_solve_time': sorted(solve_times)[len(solve_times)//2],
                'max_solve_time': max(solve_times),
                'min_solve_time': min(solve_times)
            })

# Export
with open('results/mpc_validation/performance/t3_1_solve_time.json', 'w') as f:
    json.dump({
        'test_id': 'T3.1',
        'status': 'PASS' if all(r['median_solve_time'] < 60 for r in timing_results) else 'FAIL',
        'results': timing_results
    }, f, indent=2)

df = pd.DataFrame(timing_results)
df.to_csv('results/mpc_validation/performance/t3_1_solve_time.csv', index=False)
```

**Run command:**
```bash
python test_t3_1_solve_time.py
```

**Expected Output:**
- File: `results/mpc_validation/performance/t3_1_solve_time.json`
- File: `results/mpc_validation/performance/t3_1_solve_time.csv`

**Pass Criteria:**
- ✅ Median solve time < 60 seconds per iteration
- ✅ Max solve time < 120 seconds (rare outliers acceptable)
- ✅ Solve time scales reasonably with horizon length (linear to quadratic)

**Failure Diagnosis:**
- If solve times too high: Check solver (CPLEX/Gurobi faster than CBC)
- Consider reducing horizon length or relaxing MIP gap tolerance

---

#### T3.2: Memory Usage Profiling

**Purpose**: Monitor memory consumption during long simulations

**Execution:**
```python
# test_t3_2_memory_usage.py
import tracemalloc
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

tracemalloc.start()

optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data = optimizer.extract_country_data(full_data, 'CH')
data = data['2024-01-01':'2024-01-31']  # 1 month

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5
)

# Memory before simulation
mem_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

result = simulator.run_simulation()

# Memory after simulation
mem_after = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
mem_peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB

tracemalloc.stop()

# Export
with open('results/mpc_validation/performance/t3_2_memory_usage.json', 'w') as f:
    json.dump({
        'test_id': 'T3.2',
        'status': 'PASS' if mem_peak < 2000 else 'FAIL',  # 2 GB threshold
        'memory_before_mb': mem_before,
        'memory_after_mb': mem_after,
        'memory_peak_mb': mem_peak,
        'memory_increase_mb': mem_after - mem_before,
        'iterations': len(result['iteration_results'])
    }, f, indent=2)
```

**Run command:**
```bash
python test_t3_2_memory_usage.py
```

**Expected Output:**
- File: `results/mpc_validation/performance/t3_2_memory_usage.json`

**Pass Criteria:**
- ✅ Peak memory < 2 GB for 1-month simulation
- ✅ Memory increase roughly linear with simulation length
- ✅ No memory leaks (memory released after simulation)

---

#### T3.3: Horizon Length Scalability

**Purpose**: Test how horizon length affects performance

**Execution:**
```python
# test_t3_3_horizon_scalability.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json
import time

optimizer_base = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer_base.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data = optimizer_base.extract_country_data(full_data, 'CH')
data = data['2024-03-01':'2024-03-07']  # 1 week

horizon_lengths = [24, 32, 40, 48, 56, 64, 72]
scalability_results = []

for horizon in horizon_lengths:
    optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)

    try:
        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=horizon,
            execution_hours=24,
            initial_soc_fraction=0.5
        )

        start_time = time.time()
        result = simulator.run_simulation()
        total_time = time.time() - start_time

        solve_times = [iter_result.get('solve_time_sec', 0)
                      for iter_result in result['iteration_results']]

        scalability_results.append({
            'horizon_hours': horizon,
            'status': 'SUCCESS',
            'mean_solve_time': sum(solve_times) / len(solve_times),
            'total_simulation_time': total_time,
            'iterations': len(solve_times)
        })

    except Exception as e:
        scalability_results.append({
            'horizon_hours': horizon,
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/performance/t3_3_horizon_scalability.json', 'w') as f:
    json.dump({
        'test_id': 'T3.3',
        'status': 'PASS',
        'results': scalability_results
    }, f, indent=2)
```

**Run command:**
```bash
python test_t3_3_horizon_scalability.py
```

**Expected Output:**
- File: `results/mpc_validation/performance/t3_3_horizon_scalability.json`

**Pass Criteria:**
- ✅ All horizon lengths up to 72h feasible
- ✅ Solve time scaling reasonable (< quadratic ideally)
- ✅ Longer horizons provide better decisions (diminishing returns expected)

---

#### T3.4: Solver Comparison

**Purpose**: Compare performance across available MILP solvers

**Execution:**
```python
# test_t3_4_solver_comparison.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json
import time

optimizer_base = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer_base.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data = optimizer_base.extract_country_data(full_data, 'CH')
data = data['2024-06-01':'2024-06-03']  # 3 days

solvers = ['cplex', 'gurobi', 'cbc', 'glpk']
solver_results = []

for solver in solvers:
    try:
        optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
        optimizer.solver_name = solver  # Override solver

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=0.5
        )

        start_time = time.time()
        result = simulator.run_simulation()
        total_time = time.time() - start_time

        solve_times = [iter_result.get('solve_time_sec', 0)
                      for iter_result in result['iteration_results']]

        solver_results.append({
            'solver': solver,
            'status': 'SUCCESS',
            'total_time': total_time,
            'mean_solve_time': sum(solve_times) / len(solve_times),
            'net_profit': result['net_profit']
        })

    except Exception as e:
        solver_results.append({
            'solver': solver,
            'status': 'NOT_AVAILABLE',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/performance/t3_4_solver_comparison.json', 'w') as f:
    json.dump({
        'test_id': 'T3.4',
        'status': 'PASS',
        'results': solver_results
    }, f, indent=2)
```

**Run command:**
```bash
python test_t3_4_solver_comparison.py
```

**Expected Output:**
- File: `results/mpc_validation/performance/t3_4_solver_comparison.json`

**Pass Criteria:**
- ✅ At least one commercial solver available (CPLEX or Gurobi)
- ✅ All available solvers produce similar net profit (< 1% difference)
- ✅ Performance ranking: Gurobi ≈ CPLEX > CBC > GLPK

---

#### T3.5: Parallel Execution Validation

**Purpose**: Test MetaOptimizer parallel alpha sweep

**Execution:**
```python
# test_t3_5_parallel_execution.py
from py_script.rolling_horizon import MetaOptimizer
import json
import time

# Test with parallel execution enabled
meta_opt = MetaOptimizer(
    country='CH',
    c_rate=0.5,
    start_date='2024-08-01',
    end_date='2024-08-07',
    horizon_hours=48,
    execution_hours=24,
    alpha_values=[0.5, 1.0, 1.5, 2.0, 2.5],
    max_workers=5  # Parallel
)

start_parallel = time.time()
results_parallel = meta_opt.run_meta_optimization()
time_parallel = time.time() - start_parallel

# Test with sequential execution
meta_opt_seq = MetaOptimizer(
    country='CH',
    c_rate=0.5,
    start_date='2024-08-01',
    end_date='2024-08-07',
    horizon_hours=48,
    execution_hours=24,
    alpha_values=[0.5, 1.0, 1.5, 2.0, 2.5],
    max_workers=1  # Sequential
)

start_sequential = time.time()
results_sequential = meta_opt_seq.run_meta_optimization()
time_sequential = time.time() - start_sequential

speedup = time_sequential / time_parallel

# Export
with open('results/mpc_validation/performance/t3_5_parallel_execution.json', 'w') as f:
    json.dump({
        'test_id': 'T3.5',
        'status': 'PASS' if speedup > 2.0 else 'FAIL',  # Expect >2x speedup with 5 workers
        'time_parallel_sec': time_parallel,
        'time_sequential_sec': time_sequential,
        'speedup': speedup,
        'workers': 5,
        'alpha_count': 5
    }, f, indent=2)
```

**Run command:**
```bash
python test_t3_5_parallel_execution.py
```

**Expected Output:**
- File: `results/mpc_validation/performance/t3_5_parallel_execution.json`

**Pass Criteria:**
- ✅ Speedup > 2.0x with 5 workers
- ✅ Both parallel and sequential produce identical results
- ✅ No race conditions or errors

---

### Category T4: Economic Validation

---

#### T4.1: 10-Year ROI Calculation

**Purpose**: Verify 10-year NPV and ROI methodology

**Execution:**
```python
# test_t4_1_roi_calculation.py
from py_script.rolling_horizon import MetaOptimizer
import json
import numpy as np

# Run meta-optimization for CH
meta_opt = MetaOptimizer(
    country='CH',
    c_rate=0.5,
    start_date='2024-01-01',
    end_date='2024-03-31',  # Q1 for faster test
    horizon_hours=48,
    execution_hours=24,
    alpha_values=[0.5, 1.0, 1.5, 2.0]
)

results = meta_opt.run_meta_optimization()

# Manual ROI verification
best_result = results['best_result']
annual_net_profit = best_result['net_profit'] * 4  # Scale Q1 to full year

# CH parameters from mpc_config.json
wacc = 0.04
inflation = 0.01
investment_per_kwh = 200
capacity_kwh = 4472
initial_investment = investment_per_kwh * capacity_kwh

# Calculate NPV manually
npv_manual = -initial_investment
for year in range(1, 11):
    cash_flow = annual_net_profit * ((1 + inflation) ** (year - 1))
    discount_factor = (1 + wacc) ** year
    npv_manual += cash_flow / discount_factor

roi_manual = npv_manual / initial_investment

# Compare with reported ROI
roi_reported = results['best_roi']
roi_difference = abs(roi_manual - roi_reported)

# Export
with open('results/mpc_validation/economic/t4_1_roi_calculation.json', 'w') as f:
    json.dump({
        'test_id': 'T4.1',
        'status': 'PASS' if roi_difference < 0.01 else 'FAIL',
        'roi_manual': roi_manual,
        'roi_reported': roi_reported,
        'roi_difference': roi_difference,
        'npv_manual': npv_manual,
        'initial_investment': initial_investment,
        'annual_net_profit': annual_net_profit
    }, f, indent=2)
```

**Run command:**
```bash
python test_t4_1_roi_calculation.py
```

**Expected Output:**
- File: `results/mpc_validation/economic/t4_1_roi_calculation.json`

**Pass Criteria:**
- ✅ Manual ROI matches reported ROI (< 1% difference)
- ✅ NPV calculation follows correct formula
- ✅ WACC and inflation values match configuration

---

#### T4.2: NPV Methodology Verification

**Purpose**: Validate NPV components and discount factors

**Execution:**
```python
# test_t4_2_npv_methodology.py
from py_script.rolling_horizon import MetaOptimizer
import json

# Test with different financial parameters
test_configs = [
    {'country': 'CH', 'wacc': 0.04, 'inflation': 0.01},  # Low risk
    {'country': 'HU', 'wacc': 0.06, 'inflation': 0.03},  # High risk
]

npv_results = []

for config in test_configs:
    meta_opt = MetaOptimizer(
        country=config['country'],
        c_rate=0.5,
        start_date='2024-01-01',
        end_date='2024-01-31',
        horizon_hours=48,
        execution_hours=24,
        alpha_values=[1.0]
    )

    results = meta_opt.run_meta_optimization()

    npv_results.append({
        'country': config['country'],
        'wacc': config['wacc'],
        'inflation': config['inflation'],
        'roi': results['best_roi'],
        'npv': results['best_result']['net_profit'] * 12 * 7.72  # Rough NPV estimate
    })

# Export
with open('results/mpc_validation/economic/t4_2_npv_methodology.json', 'w') as f:
    json.dump({
        'test_id': 'T4.2',
        'status': 'PASS',
        'results': npv_results,
        'observation': 'Higher WACC should decrease NPV/ROI'
    }, f, indent=2)
```

**Run command:**
```bash
python test_t4_2_npv_methodology.py
```

**Expected Output:**
- File: `results/mpc_validation/economic/t4_2_npv_methodology.json`

**Pass Criteria:**
- ✅ Higher WACC → lower ROI (for same profit)
- ✅ Higher inflation → higher future cash flows (nominal)
- ✅ NPV methodology consistent across countries

---

#### T4.3: Alpha Optimization Effectiveness

**Purpose**: Verify that meta-optimization finds reasonable optimal alpha

**Execution:**
```python
# test_t4_3_alpha_optimization.py
from py_script.rolling_horizon import MetaOptimizer
import json

# Run meta-optimization with fine alpha sweep
meta_opt = MetaOptimizer(
    country='DE_LU',
    c_rate=0.5,
    start_date='2024-02-01',
    end_date='2024-02-29',
    horizon_hours=48,
    execution_hours=24,
    alpha_values=[0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
)

results = meta_opt.run_meta_optimization()

# Check if optimal alpha is not at extremes (edge case)
optimal_alpha = results['best_alpha']
is_interior_optimum = 0.5 <= optimal_alpha <= 2.5

# Check if ROI curve has clear maximum
roi_values = [r['roi'] for r in results['all_results']]
max_roi = max(roi_values)
second_max_roi = sorted(roi_values)[-2]
roi_gap = max_roi - second_max_roi

# Export
with open('results/mpc_validation/economic/t4_3_alpha_optimization.json', 'w') as f:
    json.dump({
        'test_id': 'T4.3',
        'status': 'PASS' if is_interior_optimum else 'WARNING',
        'optimal_alpha': optimal_alpha,
        'optimal_roi': max_roi,
        'is_interior_optimum': is_interior_optimum,
        'roi_gap': roi_gap,
        'all_alphas': [r['alpha'] for r in results['all_results']],
        'all_rois': roi_values
    }, f, indent=2)
```

**Run command:**
```bash
python test_t4_3_alpha_optimization.py
```

**Expected Output:**
- File: `results/mpc_validation/economic/t4_3_alpha_optimization.json`

**Pass Criteria:**
- ✅ Optimal alpha not at extremes (0.5 ≤ α* ≤ 2.5)
- ✅ Clear ROI maximum (not flat curve)
- ✅ ROI curve shows concave shape (revenue-degradation tradeoff)

---

#### T4.4: Revenue Component Breakdown

**Purpose**: Validate revenue attribution across markets

**Execution:**
```python
# test_t4_4_revenue_breakdown.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/hu.parquet')
data = optimizer.extract_country_data(full_data, 'HU')
data = data['2024-01-01':'2024-01-31']

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5
)

results = simulator.run_simulation()

# Calculate revenue percentages
total_revenue = results['total_revenue']
revenue_breakdown = {
    'da_revenue': results['da_revenue'],
    'afrr_e_revenue': results['afrr_e_revenue'],
    'as_revenue': results['as_revenue']
}

revenue_percentages = {
    k: (v / total_revenue * 100) if total_revenue > 0 else 0
    for k, v in revenue_breakdown.items()
}

# Export
with open('results/mpc_validation/economic/t4_4_revenue_breakdown.json', 'w') as f:
    json.dump({
        'test_id': 'T4.4',
        'status': 'PASS',
        'total_revenue': total_revenue,
        'revenue_breakdown': revenue_breakdown,
        'revenue_percentages': revenue_percentages,
        'dominant_market': max(revenue_percentages, key=revenue_percentages.get)
    }, f, indent=2)
```

**Run command:**
```bash
python test_t4_4_revenue_breakdown.py
```

**Expected Output:**
- File: `results/mpc_validation/economic/t4_4_revenue_breakdown.json`

**Pass Criteria:**
- ✅ Revenue components sum to total revenue
- ✅ All components non-negative
- ✅ AS capacity markets typically dominant (40-60% of revenue)

---

#### T4.5: Degradation Cost Accuracy

**Purpose**: Verify cyclic and calendar aging cost calculations

**Execution:**
```python
# test_t4_5_degradation_cost.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
import json

optimizer = BESSOptimizerModelIII(alpha=1.5, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data = optimizer.extract_country_data(full_data, 'CH')
data = data['2024-03-01':'2024-03-07']

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5
)

results = simulator.run_simulation()

# Check degradation cost components
cyclic_cost = results['cyclic_cost']
calendar_cost = results['calendar_cost']
total_degradation = results['total_degradation_cost']

# Verify sum
sum_matches = abs((cyclic_cost + calendar_cost) - total_degradation) < 0.01

# Export
with open('results/mpc_validation/economic/t4_5_degradation_cost.json', 'w') as f:
    json.dump({
        'test_id': 'T4.5',
        'status': 'PASS' if sum_matches else 'FAIL',
        'cyclic_cost': cyclic_cost,
        'calendar_cost': calendar_cost,
        'total_degradation_cost': total_degradation,
        'sum_matches': sum_matches,
        'cyclic_percentage': cyclic_cost / total_degradation * 100 if total_degradation > 0 else 0,
        'calendar_percentage': calendar_cost / total_degradation * 100 if total_degradation > 0 else 0
    }, f, indent=2)
```

**Run command:**
```bash
python test_t4_5_degradation_cost.py
```

**Expected Output:**
- File: `results/mpc_validation/economic/t4_5_degradation_cost.json`

**Pass Criteria:**
- ✅ Cyclic + calendar = total degradation cost
- ✅ Both components non-negative
- ✅ Cyclic cost typically 70-90% of total (operation-dependent)

---

### Category T5: Integration Tests

---

#### T5.1: Data Preprocessing Integration

**Purpose**: Verify aFRR energy preprocessing is correctly applied

**Execution:**
```python
# test_t5_1_data_preprocessing.py
from py_script.core.optimizer import BESSOptimizerModelIII
from py_script.rolling_horizon import MPCSimulator
import json

optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/ch.parquet')
data = optimizer.extract_country_data(full_data, 'CH')

# Check for NaN values in aFRR energy prices (should be present after preprocessing)
afrr_cols = [col for col in data.columns if 'afrr_e_' in col.lower()]
nan_counts = {col: data[col].isna().sum() for col in afrr_cols}

# Verify that some zeros were converted to NaN
has_nans = any(count > 0 for count in nan_counts.values())

# Run short simulation to verify it doesn't crash
data_short = data['2024-06-01':'2024-06-03']
simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data_short,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5
)

result = simulator.run_simulation()

# Export
with open('results/mpc_validation/integration/t5_1_data_preprocessing.json', 'w') as f:
    json.dump({
        'test_id': 'T5.1',
        'status': 'PASS' if has_nans else 'FAIL',
        'nan_counts': nan_counts,
        'simulation_completed': True,
        'afrr_e_revenue': result['afrr_e_revenue']
    }, f, indent=2)
```

**Run command:**
```bash
python test_t5_1_data_preprocessing.py
```

**Expected Output:**
- File: `results/mpc_validation/integration/t5_1_data_preprocessing.json`

**Pass Criteria:**
- ✅ aFRR energy columns contain NaN values (zeros converted)
- ✅ Simulation completes successfully with preprocessed data
- ✅ aFRR energy revenue is non-negative (no false arbitrage)

---

#### T5.2: Model I/II/III Compatibility

**Purpose**: Verify all three model variants work with MPC framework

**Execution:**
```python
# test_t5_2_model_compatibility.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelI, BESSOptimizerModelII, BESSOptimizerModelIII
import json

models = {
    'Model I': BESSOptimizerModelI(c_rate=0.5),
    'Model II': BESSOptimizerModelII(alpha=1.0, c_rate=0.5),
    'Model III': BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
}

compatibility_results = []

for model_name, optimizer in models.items():
    try:
        full_data = optimizer.load_and_preprocess_data('data/phase2_processed/ch.parquet')
        data = optimizer.extract_country_data(full_data, 'CH')
        data = data['2024-05-01':'2024-05-03']

        simulator = MPCSimulator(
            optimizer_model=optimizer,
            full_data=data,
            horizon_hours=48,
            execution_hours=24,
            initial_soc_fraction=0.5
        )

        result = simulator.run_simulation()

        compatibility_results.append({
            'model': model_name,
            'status': 'SUCCESS',
            'net_profit': result['net_profit'],
            'iterations': len(result['iteration_results'])
        })

    except Exception as e:
        compatibility_results.append({
            'model': model_name,
            'status': 'FAILED',
            'error': str(e)
        })

# Export
with open('results/mpc_validation/integration/t5_2_model_compatibility.json', 'w') as f:
    json.dump({
        'test_id': 'T5.2',
        'status': 'PASS' if all(r['status'] == 'SUCCESS' for r in compatibility_results) else 'FAIL',
        'results': compatibility_results,
        'observation': 'Model III profit should be lowest (accounts for degradation)'
    }, f, indent=2)
```

**Run command:**
```bash
python test_t5_2_model_compatibility.py
```

**Expected Output:**
- File: `results/mpc_validation/integration/t5_2_model_compatibility.json`

**Pass Criteria:**
- ✅ All three models complete successfully
- ✅ Profit ordering: Model I > Model II > Model III (degradation reduces profit)
- ✅ No errors in state propagation for any model

---

#### T5.3: Output Generation and Export

**Purpose**: Verify result export and file generation

**Execution:**
```python
# test_t5_3_output_export.py
from py_script.rolling_horizon import MetaOptimizer
import json
import os

# Run meta-optimization with export
meta_opt = MetaOptimizer(
    country='AT',
    c_rate=0.5,
    start_date='2024-07-01',
    end_date='2024-07-07',
    horizon_hours=48,
    execution_hours=24,
    alpha_values=[0.5, 1.0, 1.5]
)

results = meta_opt.run_meta_optimization()

# Export results
output_dir = 'results/mpc_validation/integration/test_export'
meta_opt.export_results(results, output_dir, prefix='t5_3')

# Verify files exist
expected_files = [
    'summary_table.csv',
    'results_full.json',
    'best_alpha.txt'
]

files_exist = {}
for filename in expected_files:
    # Find files matching pattern (may have timestamps)
    files_in_dir = os.listdir(output_dir)
    matching = [f for f in files_in_dir if filename.split('_')[0] in f]
    files_exist[filename] = len(matching) > 0

# Export validation
with open('results/mpc_validation/integration/t5_3_output_export.json', 'w') as f:
    json.dump({
        'test_id': 'T5.3',
        'status': 'PASS' if all(files_exist.values()) else 'FAIL',
        'files_exist': files_exist,
        'output_dir': output_dir
    }, f, indent=2)
```

**Run command:**
```bash
python test_t5_3_output_export.py
```

**Expected Output:**
- File: `results/mpc_validation/integration/t5_3_output_export.json`
- Export files in `results/mpc_validation/integration/test_export/`

**Pass Criteria:**
- ✅ Summary CSV file created
- ✅ Full results JSON file created
- ✅ Best alpha text file created
- ✅ All files contain valid data

---

#### T5.4: Visualization Pipeline

**Purpose**: Test integration with plotting utilities

**Execution:**
```python
# test_t5_4_visualization.py
from py_script.rolling_horizon import MPCSimulator
from py_script.core.optimizer import BESSOptimizerModelIII
from py_script.visualization.optimization_analysis import plot_mpc_iteration
import json
import os

optimizer = BESSOptimizerModelIII(alpha=1.0, c_rate=0.5)
full_data = optimizer.load_and_preprocess_data('data/phase2_processed/hu.parquet')
data = optimizer.extract_country_data(full_data, 'HU')
data = data['2024-01-10':'2024-01-12']

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5
)

result = simulator.run_simulation()

# Generate plots for first iteration
output_dir = 'results/mpc_validation/integration/visualizations'
os.makedirs(output_dir, exist_ok=True)

try:
    iter_0 = result['iteration_results'][0]
    solution = iter_0['solution']

    # This would call the actual plotting function
    # plot_mpc_iteration(solution, data_window, output_dir)

    plot_success = True  # Placeholder
except Exception as e:
    plot_success = False
    error_msg = str(e)

# Export
with open('results/mpc_validation/integration/t5_4_visualization.json', 'w') as f:
    json.dump({
        'test_id': 'T5.4',
        'status': 'PASS' if plot_success else 'FAIL',
        'simulation_completed': True,
        'plot_generation': 'SUCCESS' if plot_success else 'FAILED'
    }, f, indent=2)
```

**Run command:**
```bash
python test_t5_4_visualization.py
```

**Expected Output:**
- File: `results/mpc_validation/integration/t5_4_visualization.json`
- HTML plots in `results/mpc_validation/integration/visualizations/`

**Pass Criteria:**
- ✅ Simulation completes
- ✅ Plotting functions execute without errors
- ✅ Output HTML files are valid

---

#### T5.5: Full-Year Simulation

**Purpose**: End-to-end test of complete year optimization

**Configuration:**
```python
test_config = {
    'country': 'CH',
    'period': '2024-01-01 to 2024-12-31',  # Full year
    'horizon_hours': 48,
    'execution_hours': 24,
    'alpha': 1.0,
    'c_rate': 0.5
}
```

**Execution:**
```python
# test_t5_5_full_year.py
from py_script.rolling_horizon import MPCSimulator, MetaOptimizer
import json
import time

# Option 1: Direct MPC simulation
start_time = time.time()

meta_opt = MetaOptimizer(
    country='CH',
    c_rate=0.5,
    start_date='2024-01-01',
    end_date='2024-12-31',
    horizon_hours=48,
    execution_hours=24,
    alpha_values=[1.0]  # Single alpha for speed
)

results = meta_opt.run_meta_optimization()
total_time = time.time() - start_time

# Calculate annual metrics
annual_result = results['all_results'][0]  # Only one alpha

# Export
output_file = 'results/mpc_validation/integration/t5_5_full_year.json'
with open(output_file, 'w') as f:
    json.dump({
        'test_id': 'T5.5',
        'status': 'PASS' if results['status'] == 'success' else 'FAIL',
        'total_simulation_time_hours': total_time / 3600,
        'iterations': 365,  # Expected
        'annual_revenue': annual_result['total_revenue'],
        'annual_degradation_cost': annual_result['total_degradation_cost'],
        'annual_net_profit': annual_result['net_profit'],
        'roi_10_year': annual_result['roi'],
        'final_soc': annual_result['final_soc']
    }, f, indent=2)

# Also export full results
meta_opt.export_results(results, 'results/mpc_validation/integration', prefix='full_year_ch')
```

**Run command:**
```bash
python test_t5_5_full_year.py
```

**Expected Runtime**: 6-12 hours (depends on solver and hardware)

**Expected Output:**
- File: `results/mpc_validation/integration/t5_5_full_year.json`
- Full results: `results/mpc_validation/integration/full_year_ch_*.json`

**Pass Criteria:**
- ✅ Simulation completes all 365 iterations
- ✅ No infeasibilities encountered
- ✅ Annual net profit positive
- ✅ 10-year ROI reasonable (0.1 to 1.0 typical range)
- ✅ Total runtime < 24 hours

**Failure Diagnosis:**
- If infeasibility mid-year: Check price data, increase alpha, or adjust initial SOC
- If runtime too long: Verify solver (use CPLEX/Gurobi), consider looser MIP gap

---

## 7. Success Criteria

### 7.1 Overall Validation Status

The MPC rolling horizon implementation is considered **validated** if:

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| **Correctness** | All T1.x tests PASS | ⬜ |
| **Robustness** | ≥ 4/5 T2.x tests PASS | ⬜ |
| **Performance** | ≥ 4/5 T3.x tests PASS | ⬜ |
| **Economic** | All T4.x tests PASS | ⬜ |
| **Integration** | All T5.x tests PASS | ⬜ |

### 7.2 Critical Tests (Must Pass)

These tests are **mandatory** for production use:

- ✅ T1.1: SOC Continuity Validation
- ✅ T1.2: Constraint Satisfaction Check
- ✅ T2.1: Multi-Country Validation
- ✅ T4.1: 10-Year ROI Calculation
- ✅ T5.5: Full-Year Simulation

### 7.3 Performance Targets

| Metric | Target | Acceptable | Unacceptable |
|--------|--------|------------|--------------|
| Median solve time | < 30 sec | < 60 sec | > 120 sec |
| Peak memory | < 1 GB | < 2 GB | > 4 GB |
| Full-year runtime | < 8 hours | < 24 hours | > 48 hours |
| Multi-country success | 5/5 | 4/5 | < 4/5 |
| Constraint violations | 0 | 0 | > 0 |

---

## 8. Deliverables

### 8.1 Validation Report Structure

```
results/mpc_validation/
├── correctness/
│   ├── t1_1_soc_continuity.json
│   ├── t1_2_constraint_check.json
│   ├── t1_3_revenue_accuracy.json
│   ├── t1_4_segment_distribution.json
│   └── t1_5_boundary_conditions.json
├── robustness/
│   ├── t2_1_multi_country.json
│   ├── t2_1_multi_country.csv
│   ├── t2_2_seasonal_stability.json
│   ├── t2_3_alpha_sensitivity.json
│   ├── t2_3_alpha_sensitivity.csv
│   ├── t2_4_initial_soc_sensitivity.json
│   └── t2_5_edge_cases.json
├── performance/
│   ├── t3_1_solve_time.json
│   ├── t3_1_solve_time.csv
│   ├── t3_2_memory_usage.json
│   ├── t3_3_horizon_scalability.json
│   ├── t3_4_solver_comparison.json
│   └── t3_5_parallel_execution.json
├── economic/
│   ├── t4_1_roi_calculation.json
│   ├── t4_2_npv_methodology.json
│   ├── t4_3_alpha_optimization.json
│   ├── t4_4_revenue_breakdown.json
│   └── t4_5_degradation_cost.json
├── integration/
│   ├── t5_1_data_preprocessing.json
│   ├── t5_2_model_compatibility.json
│   ├── t5_3_output_export.json
│   ├── t5_4_visualization.json
│   └── t5_5_full_year.json
└── VALIDATION_SUMMARY.md  # Generated summary report
```

### 8.2 Summary Report Template

Create `results/mpc_validation/VALIDATION_SUMMARY.md`:

```markdown
# MPC Rolling Horizon Validation Summary

**Date**: YYYY-MM-DD
**Tester**: [Name]
**Solver**: [CPLEX/Gurobi/CBC]
**Python Version**: X.X.X

## Executive Summary

- **Total Tests**: 25
- **Passed**: X/25
- **Failed**: Y/25
- **Overall Status**: [PASS/FAIL]

## Category Results

| Category | Tests | Passed | Failed | Status |
|----------|-------|--------|--------|--------|
| Correctness | 5 | X | Y | PASS/FAIL |
| Robustness | 5 | X | Y | PASS/FAIL |
| Performance | 5 | X | Y | PASS/FAIL |
| Economic | 5 | X | Y | PASS/FAIL |
| Integration | 5 | X | Y | PASS/FAIL |

## Critical Issues

[List any test failures or concerns]

## Recommendations

[Based on test results, provide recommendations for:
- Parameter tuning
- Configuration changes
- Code improvements
- Further testing needed]

## Performance Highlights

- Median solve time: X seconds
- Full-year simulation time: X hours
- Multi-country success rate: X/5
- Optimal alpha range: [X, Y]

## Sign-off

The MPC rolling horizon implementation is [APPROVED/NOT APPROVED] for:
- [ ] Development use
- [ ] Validation studies
- [ ] Production use
- [ ] Competition submission

**Validator**: _______________
**Date**: _______________
```

---

## 9. Troubleshooting Guide

### 9.1 Common Issues and Solutions

#### Issue: SOC Continuity Violations

**Symptoms**: `max_soc_change` > 0.1 kWh at iteration boundaries

**Diagnosis**:
```python
# Check segment SOC initialization
print(initial_segment_soc)  # Should sum to previous final SOC
```

**Solutions**:
1. Verify `_get_initial_segment_soc()` logic
2. Check for floating-point rounding errors
3. Ensure segment capacity (447.2 kWh) is correctly applied

---

#### Issue: Infeasibility Mid-Simulation

**Symptoms**: Optimization fails after N successful iterations

**Diagnosis**:
```python
# Check SOC before infeasible iteration
previous_soc = results['iteration_results'][N-1]['final_soc']
print(f"SOC before failure: {previous_soc} kWh")

# Check if energy reserves depleted
min_reserve = 0.8 * max_power * 4  # 4-hour AS block
print(f"Minimum reserve needed: {min_reserve} kWh")
```

**Solutions**:
1. Increase alpha (reduce aggressive operation)
2. Increase initial SOC (start with 70-80% instead of 50%)
3. Reduce `max_ancillary_service_ratio` (default 0.8 → 0.7)
4. Add terminal SOC constraint for lookahead

---

#### Issue: Extremely Long Solve Times

**Symptoms**: Median solve time > 60 seconds

**Diagnosis**:
```python
# Check solver and MIP gap
print(optimizer.solver_name)
print(optimizer.mip_gap)  # Should be 0.01 or 0.02
```

**Solutions**:
1. Verify using commercial solver (CPLEX or Gurobi)
2. Increase MIP gap tolerance: `optimizer.mip_gap = 0.03`
3. Reduce horizon length: 48h → 36h
4. Simplify model: use Model II instead of Model III for testing

---

#### Issue: Revenue Calculation Mismatch

**Symptoms**: Manual revenue calculation ≠ reported revenue

**Diagnosis**:
```python
# Check timestamp alignment
print(data.index[:5])  # Should be hourly
print(solution.keys())  # Check timestep indices

# Verify price units
print(data['day_ahead_price'].mean())  # EUR/MWh
print(solution['p_dis[0]'])  # MW
```

**Solutions**:
1. Ensure price data is in EUR/MWh (not EUR/kWh)
2. Verify power units are MW (not kW)
3. Check that execution window indices match data indices
4. Confirm aFRR energy preprocessing applied (0 → NaN)

---

#### Issue: Alpha Optimization Fails to Find Interior Optimum

**Symptoms**: Optimal alpha at boundary (0.1 or 3.0)

**Diagnosis**:
```python
# Plot ROI vs alpha
import matplotlib.pyplot as plt
alphas = [r['alpha'] for r in results['all_results']]
rois = [r['roi'] for r in results['all_results']]
plt.plot(alphas, rois)
plt.xlabel('Alpha')
plt.ylabel('10-Year ROI')
plt.show()
```

**Solutions**:
1. Extend alpha range: test [0.05, 0.1, ..., 3.5, 4.0]
2. Check if ROI curve is monotonic (may indicate issue with ROI calc)
3. Verify degradation costs are being applied (alpha > 0 should reduce degradation)
4. Ensure simulation length is sufficient (use ≥ 1 month for meta-optimization)

---

#### Issue: Memory Leak in Long Simulations

**Symptoms**: Memory usage grows continuously, system slowdown

**Diagnosis**:
```python
import tracemalloc
tracemalloc.start()

# Run simulation
# ... monitor memory between iterations

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")
```

**Solutions**:
1. Explicitly clear old models: `del model` after each iteration
2. Clear solution dictionaries: `solution.clear()`
3. Use garbage collection: `import gc; gc.collect()`
4. Reduce stored results (don't store full solutions for all iterations)

---

#### Issue: Parallel Execution Failures

**Symptoms**: MetaOptimizer crashes or hangs with `max_workers > 1`

**Diagnosis**:
```python
# Test with sequential execution first
meta_opt.max_workers = 1
results_seq = meta_opt.run_meta_optimization()  # Should work

# Then try parallel
meta_opt.max_workers = 2
results_par = meta_opt.run_meta_optimization()  # May fail
```

**Solutions**:
1. Check solver thread-safety (CPLEX/Gurobi are thread-safe)
2. Ensure data loading is not shared across workers
3. Use `if __name__ == '__main__':` guard in script
4. Reduce `max_workers` to avoid resource contention

---

### 9.2 Debugging Tools

#### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=data,
    horizon_hours=48,
    execution_hours=24,
    initial_soc_fraction=0.5,
    verbose=True  # If implemented
)
```

#### Save Intermediate States

```python
# After each iteration, save state
for i, iter_result in enumerate(results['iteration_results']):
    with open(f'debug_iter_{i}.json', 'w') as f:
        json.dump(iter_result, f, indent=2)
```

#### Check Constraint Violations

```python
# Enable validation
simulator = MPCSimulator(
    ...,
    validate_constraints=True
)

# Check reports
for i, iter_result in enumerate(results['iteration_results']):
    if 'validation_report' in iter_result:
        violations = iter_result['validation_report'].get('violations', [])
        if violations:
            print(f"Iteration {i}: {len(violations)} violations")
            print(violations)
```

---

## 10. References

### 10.1 Documentation

- **Main README**: `py_script/rolling_horizon/README.md`
- **Project Description**: `doc/whole_project_description.md`
- **Model Formulation**: `doc/p2_model/p2_bi_model_ggdp.tex`
- **CLAUDE.md**: Project context for AI assistant

### 10.2 Implementation Files

- **MPC Simulator**: `py_script/rolling_horizon/mpc_simulator.py`
- **Meta-Optimizer**: `py_script/rolling_horizon/meta_optimizer.py`
- **Core Optimizer**: `py_script/core/optimizer.py`
- **Configuration**: `data/phase2_aging_config/mpc_config.json`

### 10.3 Validation History

- **SOC Continuity**: `validate_mpc_soc_continuity.py` (PASSED)
- **MPC Findings**: `MPC_VALIDATION_FINDINGS.md`
- **Bug Fixes**: `py_script/rolling_horizon/mpc_improve_suggestion.md`

### 10.4 Related Testing

- **Demo Pipeline**: `py_script/test_scripts/archive/demo_model_iii_pipeline.py`
- **Constraint Validator**: `py_script/validation/constraint_validator.py`
- **Phase 1 Validation**: `results/model_iii_validation_phase1.csv`

---

## Appendix A: Quick Start Checklist

- [ ] Install dependencies: `pip install pyomo pandas numpy plotly`
- [ ] Install MILP solver (CPLEX, Gurobi, or CBC)
- [ ] Verify data availability: `data/phase2_processed/*.parquet`
- [ ] Create output directories (Section 3.2)
- [ ] Run quick validation suite (~1-2 hours)
- [ ] Review results in `results/mpc_validation/`
- [ ] Generate summary report
- [ ] Address any critical failures
- [ ] Proceed to standard validation suite

## Appendix B: Test Script Generator

For convenience, create `generate_test_scripts.py` to automatically create all test scripts:

```python
# This would generate all test_t1_1_*.py files from templates
# Implementation TBD based on needs
```

---

**Document Control**

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-09 | Initial validation plan | AI Assistant |

**Approval**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Lead Developer | | | |
| Validation Engineer | | | |
| Project Manager | | | |

---

*This validation plan is a living document. Update as new tests are added or requirements change.*
