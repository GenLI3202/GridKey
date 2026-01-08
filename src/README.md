# BESS Optimization Model - Phase II Implementation

This directory contains the Phase II implementation of the Battery Energy Storage System (BESS) optimization model for the Huawei TechArena 2025 challenge.

## Directory Structure

```
py_script/
├── README.md                    # This file
├── requirements.txt             # Python package dependencies
├── ARCHIVE_NOTE.md              # Information about archived Phase I files
│
├── core/                        # Core optimization logic
│   ├── __init__.py
│   ├── optimizer.py             # BESSOptimizerV2 - Main optimization model
│   └── exceptions.py            # Custom exception classes
│
├── data/                        # Data loading and processing
│   ├── __init__.py
│   └── market_data.py           # Market data loading and transformation
│
├── analysis/                    # Analysis modules
│   ├── __init__.py
│   └── investment.py            # Investment analysis and DCF calculations
│
├── visualization/               # Plotting and visualization
│   ├── __init__.py
│   ├── config.py                # McKinsey-style visualization templates
│   └── validation_plots.py      # Validation result plotting functions
│
└── scripts/                     # Executable entry points
    ├── main.py                  # Main CLI for running optimizations
    ├── run_all_scenarios.py     # Run all 45 competition scenarios
    ├── validate_week.py         # Week-scale model validation
    └── process_phase2_data.py   # Phase 2 data preprocessing
```

## Quick Start

### 1. Install Dependencies

```bash
# From the project root directory
pip install -r py_script/requirements.txt
```

### 2. Install a Mixed-Integer Solver

The optimization model requires a MILP solver. Options:

**CBC (Open Source, Recommended)**
```bash
conda install -c conda-forge coincbc
```

**GLPK (Open Source)**
```bash
conda install -c conda-forge glpk
```

**Gurobi or CPLEX (Commercial, Best Performance)**
- Gurobi: https://www.gurobi.com/
- CPLEX: https://www.ibm.com/products/ilog-cplex-optimization-studio

### 3. Test the Implementation

```bash
# From the py_script directory
python scripts/main.py test
```

### 4. Run Optimizations

```bash
# Single scenario
python scripts/main.py single DE 0.5 1.0

# Quick test with subset
python scripts/main.py quick

# Full optimization (all scenarios)
python scripts/main.py full

# Run all 45 scenarios for competition submission
python scripts/run_all_scenarios.py
```

## Usage Examples

### Using the Optimizer Programmatically

```python
import sys
sys.path.insert(0, 'py_script')

from core import BESSOptimizerV2

# Initialize optimizer
optimizer = BESSOptimizerV2()

# Load and preprocess data
data = optimizer.load_and_preprocess_data('../data/market_prices.jsonl')

# Extract country-specific data
country_data = optimizer.extract_country_data(data, 'DE')

# Run optimization
results = optimizer.optimize(country_data)

print(f"Total Revenue: €{results['total_revenue']:.2f}")
```

### Running Week-Scale Validation

```bash
# Run validation for the first week of 2024
python scripts/validate_week.py

# Resume from previous run
python scripts/validate_week.py --resume

# Use specific solver
python scripts/validate_week.py --solver gurobi
```

### Processing Phase 2 Data

```bash
# Convert Excel data to JSON/Parquet format
python scripts/process_phase2_data.py
```

## Model Overview

### Version 2 Improvements (Phase II)

The `BESSOptimizerV2` class incorporates several improvements over Phase I:

**Core Optimizations:**
- Eliminated constraint closure anti-patterns for better solver performance
- Pre-computed block-to-time mappings for O(1) lookup efficiency
- AS prices indexed by block instead of time to reduce memory overhead
- Optimized objective function computation
- Enhanced memory efficiency for full-year optimizations

**Phase II Enhancements:**
- Added reserve duration parameter for accurate energy reserve calculations
- Refined constraints for energy reserve calculations in upward/downward regulation
- Improved representation of activation durations for aFRR and FCR services
- Comprehensive input validation and error handling
- Consistent solver time limits across different solvers

### Objective Function

Maximize total profit from day-ahead energy arbitrage and ancillary service capacity payments:

```
max Z = Σ[(P_DA(t)/1000 · p_dis(t) - P_DA(t)/1000 · p_ch(t)) · Δt]
      + Σ[P_FCR(b)·c_fcr(b) + P_aFRR_pos(b)·c_afrr_pos(b) + P_aFRR_neg(b)·c_afrr_neg(b)] · Δb
```

### Key Constraints

1. **SOC Dynamics** - Energy balance with charging/discharging efficiency
2. **SOC Limits** - Stay within 0-100% operational range
3. **Power Limits** - Respect C-rate configuration limits
4. **Market Co-optimization** - Allocate power between energy and reserves
5. **Daily Cycle Limits** - Limit total daily discharged energy
6. **Energy Reserves** - Maintain sufficient energy for ancillary service delivery
7. **Minimum Bid Sizes** - Respect market minimum bid requirements
8. **Market Exclusivity** - Cannot bid in multiple AS markets simultaneously

### Configuration Scenarios

- **Countries**: DE_LU, AT, CH, HU, CZ (5 options)
- **C-rates**: 0.25, 0.33, 0.5 (3 options)
- **Daily Cycles**: 1.0, 1.5, 2.0 (3 options)
- **Total**: 45 optimization scenarios

### Technical Parameters

- Nominal Capacity: 4,472 kWh
- Charging/Discharging Efficiency: 95%
- Time Resolution: 15 minutes (day-ahead), 4 hours (ancillary services)
- Optimization Horizon: Full year 2024

## Migration from Phase I

Phase I code has been archived to the `r1-static-battery` branch. See `ARCHIVE_NOTE.md` for details.

**Key Changes:**
- `ImprovedBESSOptimizer` → `BESSOptimizerV2`
- `model.py` → `core/optimizer.py`
- `market_da.py` → `data/market_data.py`
- `investment_analysis.py` → `analysis/investment.py`
- `viz_config.py` → `visualization/config.py`
- Scripts moved to `scripts/` subdirectory

**Updating Imports:**

Old (Phase I):
```python
from model import ImprovedBESSOptimizer
```

New (Phase II):
```python
from core import BESSOptimizerV2
```

## Output

The optimization produces:
- Optimal charging/discharging schedules
- Ancillary service capacity bids
- Annual profit calculations
- Performance summary statistics
- CSV files in competition submission format (via `run_all_scenarios.py`)

Results are saved in JSON format for further analysis and Excel export.

## Troubleshooting

### Common Issues

1. **"Solver not available"**
   - Install a solver (see installation section above)
   - Ensure solver executable is in system PATH

2. **"Memory Error" or slow performance**
   - The full year optimization is computationally intensive
   - Consider using commercial solvers (Gurobi/CPLEX) for better performance
   - Set solver time limits in the model

3. **"Module not found" errors**
   - Ensure you're running scripts from the correct directory
   - The scripts add the parent directory to sys.path automatically
   - For imports in other contexts, add `py_script/` to your Python path

4. **"Data loading errors"**
   - Verify data file format matches expected JSONL or Excel structure
   - Check that all required data sources are present
   - Run `process_phase2_data.py` to validate Phase 2 data

### Performance Tips

- Use CBC with time limits for reasonable solve times on open-source solvers
- Consider running scenarios in parallel if computational resources allow
- Monitor memory usage for full-year optimizations
- Use Gurobi or CPLEX for production runs if available

## Model Validation

The validation script (`scripts/validate_week.py`) validates:
1. Data loading and preprocessing
2. Model construction
3. Constraint satisfaction
4. Solver functionality
5. Results consistency

Run validation before attempting full optimization to catch configuration issues early.

## Development

### Running Tests

```bash
# Test model import
python -c "from core import BESSOptimizerV2; print('Import successful!')"

# Test single optimization
python scripts/main.py test
```

### Code Quality

The code follows these principles:
- Type hints for function signatures
- Comprehensive docstrings
- Logging for debugging and monitoring
- Input validation for robustness
- Efficient data structures for performance

## Author

Gen's BESS Optimization Team
Phase II Development: October-November 2025

## License

See LICENSE file in project root.
