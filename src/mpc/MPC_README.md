# Rolling Horizon MPC Framework - Model III

This module implements the complete three-layer optimization framework for Phase II Model (iii) of the Huawei TechArena 2025 BESS challenge.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OUTER LAYER: MetaOptimizer                 │
│          (Find optimal α for 10-year ROI)                   │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │            MIDDLE LAYER: MPCSimulator                 │ │
│  │     (Rolling horizon for full-year simulation)        │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │      INNER LAYER: BESSOptimizerModelIII        │ │ │
│  │  │  (Single MILP with cyclic + calendar aging)    │ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. BESSOptimizerModelIII (`core/optimizer.py`)
**Inner Layer**: Single-horizon MILP optimization

**Features:**
- Cyclic aging cost (10 SOC segments)
- Calendar aging cost (5 SOS2 breakpoints)
- Day-ahead + aFRR energy markets
- FCR + aFRR capacity markets
- Degradation price parameter α

**Usage:**
```python
from core.optimizer import BESSOptimizerModelIII

# Create optimizer
optimizer = BESSOptimizerModelIII(alpha=1.0)

# Build and solve for 2-day horizon
model = optimizer.build_optimization_model(data, c_rate=0.5)
solution = optimizer.solve_model(model)
```

### 2. MPCSimulator (`rolling_horizon/mpc_simulator.py`)
**Middle Layer**: Rolling horizon simulation

**Features:**
- 30-hour optimization horizon
- 24-hour execution window
- State continuity between windows
- Constraint validation (optional)

**Usage:**
```python
from rolling_horizon import MPCSimulator

# Create simulator
simulator = MPCSimulator(
    optimizer_model=optimizer,
    full_data=country_data,
    horizon_hours=30,
    execution_hours=24
)

# Run full-year simulation
results = simulator.run_full_simulation()
print(f"Net profit: {results['net_profit']:.2f} EUR")
```

### 3. MetaOptimizer (`rolling_horizon/meta_optimizer.py`)
**Outer Layer**: Alpha parameter sweep

**Features:**
- 10-year ROI calculation (NPV with WACC + inflation)
- Parallel execution support
- Comprehensive result export

**Usage:**
```python
from rolling_horizon import MetaOptimizer

# Configure financial parameters
country_config = {
    'wacc': 0.05,
    'inflation': 0.02,
    'investment_eur_per_kwh': 200,
    'capacity_kwh': 4472,
}

# Create meta-optimizer
meta_opt = MetaOptimizer(
    full_data=country_data,
    country_config=country_config,
    alpha_values=[0.5, 1.0, 1.5, 2.0]
)

# Find optimal alpha
results = meta_opt.find_optimal_alpha()
print(f"Best alpha: {results['best_alpha']}")
print(f"Best ROI: {results['best_roi']:.2%}")
```

## Quick Start

### Option 1: Use Demo Script

```bash
# Quick test (1 week, 3 alpha values)
python demo_model_iii_pipeline.py --mode quick --country CH

# Full MPC simulation (custom alpha)
python demo_model_iii_pipeline.py --mode mpc --alpha 1.5 --weeks 4

# Full meta-optimization
python demo_model_iii_pipeline.py --mode meta --country CH --weeks 52
```

### Option 2: Python Code

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.optimizer import BESSOptimizerModelIII
from rolling_horizon import MPCSimulator, MetaOptimizer

# 1. Load data
optimizer = BESSOptimizerModelIII()
data = optimizer.load_and_preprocess_data('data/TechArena2025_data_tidy.jsonl')
country_data = optimizer.extract_country_data(data, 'CH')

# 2. Run MPC simulation
simulator = MPCSimulator(optimizer, country_data, horizon_hours=48)
mpc_results = simulator.run_full_simulation()

# 3. Find optimal alpha
country_config = {
    'wacc': 0.05, 'inflation': 0.02,
    'investment_eur_per_kwh': 200, 'capacity_kwh': 4472
}
meta_opt = MetaOptimizer(country_data, country_config, alpha_values=[0.5, 1.0, 1.5])
best = meta_opt.find_optimal_alpha()

print(f"Optimal alpha: {best['best_alpha']}")
print(f"10-year ROI: {best['best_roi']:.2%}")
```

## Configuration

### MPC Configuration (`mpc_config.json`)

```json
{
  "mpc_parameters": {
    "horizon_hours": 48,
    "execution_hours": 24,
    "initial_soc_fraction": 0.5,
    "validate_constraints": false
  },
  "alpha_sweep": {
    "default_values": [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
  }
}
```

### Aging Configuration (`data/phase2_aging_config/aging_config.json`)

Already configured with:
- **Cyclic aging**: 10 segments, costs from 0.0052 to 0.0990 EUR/kWh
- **Calendar aging**: 5 breakpoints, costs from 1.79 to 10.73 EUR/hr

## Performance Tips

### For Testing
```python
# Use short horizons and limit iterations
simulator = MPCSimulator(optimizer, data, horizon_hours=24, execution_hours=12)
results = simulator.run_full_simulation(max_iterations=7)  # 1 week test
```

### For Production
```python
# Disable validation for speed
simulator = MPCSimulator(optimizer, data, validate_constraints=False)

# Use parallel meta-optimization
meta_opt.find_optimal_alpha(parallel=True, max_workers=4)
```

## Expected Runtime

| Mode | Horizon | Iterations | Time (per alpha) |
|------|---------|------------|------------------|
| Quick test (1 week) | 48h | 7 | ~2-5 minutes |
| 1 month | 48h | 30 | ~8-15 minutes |
| Full year | 48h | 365 | ~1-2 hours |

*Times based on CBC solver. CPLEX/Gurobi are 5-10x faster.*

## Validation

The framework includes post-solve constraint validation:

```python
from validation import validate_solution

# After solving
validation_report = validate_solution(model, solution)

if not validation_report['summary']['all_passed']:
    print(f"Violations found: {validation_report['summary']['total_violations']}")
```

**Validated Constraints:**
- Cst-8: Cross-market mutual exclusivity
- Cst-9: Minimum bid sizes (DA, aFRR-E)
- SOS2 properties (calendar aging)
- Segment SOC ordering (stacked tank)

## Output

### MPC Simulation Results
```python
{
    'total_revenue': 123456.78,         # EUR
    'total_degradation_cost': 12345.67, # EUR
    'net_profit': 111111.11,            # EUR
    'final_soc': 2236.0,                # kWh
    'soc_total_bids_df': [2236, 2100, ...],
    'iteration_results': [...],
    'validation_reports': [...]
}
```

### Meta-Optimization Results
```python
{
    'best_alpha': 1.5,
    'best_roi': 0.45,  # 45%
    'best_result': {...},
    'all_results': [...],
    'summary_df': DataFrame
}
```

## Troubleshooting

### Issue: Solver timeout
**Solution**: Reduce horizon or increase time limit in `market_params['solver_time_limit']`

### Issue: Infeasible solution
**Solution**: Check initial SOC and data quality. Try relaxing binary constraints.

### Issue: Memory error
**Solution**: Process data in smaller chunks or use reduced alpha sweep.

### Issue: Validation violations
**Solution**: Review constraint_validator.py output. Minor violations (<1%) are usually acceptable.

## References

- **Mathematical Model**: `doc/p2_model/p2_bi_model_ggdp.tex`
- **Implementation Guide**: `py_script/rolling_horizon/model_iii_ref.md`
- **Collath et al. (2023)**: Rolling horizon approach for BESS
- **Xu et al. (2017)**: Segmented cyclic aging model

## Support

For questions or issues:
1. Check the demo script: `demo_model_iii_pipeline.py`
2. Review docstrings in source files
3. Check validation reports for constraint violations

---

**Last Updated**: November 2025
**Version**: 1.0
