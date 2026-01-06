# BESS Optimization Validation Utilities

General-purpose command-line tools for running and comparing BESS optimizations with flexible configurations.

## Overview

This directory contains reusable validation utilities that replace hardcoded test scripts:

| Script | Purpose | Replaces |
|--------|---------|----------|
| `run_optimization.py` | General-purpose optimization runner | `../test/run_36h_hu_winter.py` |
| `compare_optimizations.py` | Compare different configurations | `../test/test_single_32h_vs_mpc.py` |
| `results_exporter.py` | Standardized results saving | (utility module) |

## Quick Start

### Running a Single Optimization

```bash
# Model III: 36h HU winter (equivalent to old run_36h_hu_winter.py)
python py_script/validation/run_optimization.py \
    --model III \
    --country HU \
    --hours 36 \
    --start-step 0 \
    --alpha 0.5 \
    --plots

# Model II: 48h DE with custom parameters
python py_script/validation/run_optimization.py \
    --model II \
    --country DE \
    --hours 48 \
    --c-rate 0.33 \
    --alpha 1.0 \
    --plots

# Quick 12h test
python py_script/validation/run_optimization.py \
    --model I \
    --country AT \
    --hours 12 \
    --c-rate 0.5
```

### Comparing Configurations

```bash
# Compare single vs MPC (equivalent to old test_single_32h_vs_mpc.py)
python py_script/validation/compare_optimizations.py \
    --compare-type single-vs-mpc \
    --hours 32 \
    --country HU

# Compare models I, II, III
python py_script/validation/compare_optimizations.py \
    --compare-type models \
    --models I II III \
    --hours 24 \
    --country DE

# Alpha sensitivity analysis
python py_script/validation/compare_optimizations.py \
    --compare-type alpha \
    --alphas 0.5 1.0 1.5 \
    --hours 36 \
    --country HU \
    --model III

# Compare countries
python py_script/validation/compare_optimizations.py \
    --compare-type countries \
    --countries DE_LU AT CH HU \
    --hours 48 \
    --model II
```

## run_optimization.py

### Usage

```
python py_script/validation/run_optimization.py [OPTIONS]
```

### Required Arguments

| Argument | Description | Choices |
|----------|-------------|---------|
| `--model` | Optimizer model | I, II, III |
| `--country` | Market country | DE_LU, AT, CH, HU, CZ |
| `--hours` | Time horizon (hours) | Any positive integer |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--start-step` | 0 | Starting time step (15-min intervals) |
| `--c-rate` | 0.5 | Battery C-rate |
| `--alpha` | 1.0* | Degradation weight (Model II/III only) |
| `--cycles` | None | Daily cycle limit (Model I only) |
| `--solver` | auto | Solver (cplex, gurobi, highs, cbc, glpk) |
| `--max-as-ratio` | 0.8 | Max ancillary service ratio (0.0-1.0) |
| `--use-ev-weighting` | False | Enable EV weighting for aFRR energy |
| `--plots` | False | Generate visualizations |
| `--output-dir` | validation_results | Base output directory |
| `--run-name` | auto | Custom run name |
| `--data-dir` | data/phase2_processed | Market data directory |

*Default 1.0 for Model II/III, not applicable for Model I

### Output Structure

```
validation_results/
└── YYYYMMDD_HHMMSS_run_name/
    ├── solution_timeseries.csv       # Full solution data
    ├── performance_summary.json      # Metrics and metadata
    └── plots/                        # Visualizations (if --plots)
        ├── da_market_price_bid.html
        ├── afrr_energy_market_price_bid.html
        ├── capacity_markets_price_bid.html
        └── soc_and_power_bids.html
```

### Examples

#### Reproduce Old Scripts

```bash
# Old: py_script/test/run_36h_hu_winter.py
# New:
python py_script/validation/run_optimization.py \
    --model III \
    --country HU \
    --hours 36 \
    --start-step 0 \
    --alpha 0.5 \
    --use-ev-weighting \
    --plots
```

#### Custom Scenarios

```bash
# Week-long optimization
python py_script/validation/run_optimization.py \
    --model III \
    --country CH \
    --hours 168 \
    --alpha 0.75 \
    --plots

# High C-rate test
python py_script/validation/run_optimization.py \
    --model II \
    --country DE_LU \
    --hours 24 \
    --c-rate 1.0 \
    --alpha 1.2

# Different time windows
python py_script/validation/run_optimization.py \
    --model I \
    --country AT \
    --hours 48 \
    --start-step 2000  # Start at step 2000
```

## compare_optimizations.py

### Usage

```
python py_script/validation/compare_optimizations.py [OPTIONS]
```

### Comparison Types

| Type | Description | Required Args |
|------|-------------|---------------|
| `single-vs-mpc` | Single optimization vs MPC | `--hours`, `--country` |
| `models` | Compare I, II, III | `--hours`, `--models` |
| `alpha` | Degradation weight sensitivity | `--hours`, `--alphas` |
| `countries` | Cross-market comparison | `--hours`, `--countries` |
| `c-rates` | C-rate sensitivity | `--hours`, `--c-rates` |

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | III | Baseline model (when not comparing models) |
| `--country` | HU | Baseline country (when not comparing countries) |
| `--alpha` | 1.0 | Baseline alpha (when not comparing alpha) |
| `--c-rate` | 0.5 | Baseline C-rate (when not comparing C-rates) |
| `--start-step` | 0 | Starting time step |
| `--solver` | auto | Solver to use |
| `--output-dir` | validation_results/comparisons | Output directory |

### MPC-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mpc-horizon` | 32 | MPC optimization horizon (hours) |
| `--mpc-update` | 24 | MPC update interval (hours) |

### Examples

#### Single vs MPC

```bash
# Reproduce old test_single_32h_vs_mpc.py
python py_script/validation/compare_optimizations.py \
    --compare-type single-vs-mpc \
    --hours 32 \
    --country HU \
    --model III

# Custom MPC configuration
python py_script/validation/compare_optimizations.py \
    --compare-type single-vs-mpc \
    --hours 48 \
    --country DE_LU \
    --mpc-horizon 24 \
    --mpc-update 12
```

#### Model Comparison

```bash
# Compare all three models
python py_script/validation/compare_optimizations.py \
    --compare-type models \
    --models I II III \
    --hours 24 \
    --country DE

# Model I vs II only
python py_script/validation/compare_optimizations.py \
    --compare-type models \
    --models I II \
    --hours 36 \
    --country HU
```

#### Alpha Sensitivity

```bash
# Test alpha values for Model III
python py_script/validation/compare_optimizations.py \
    --compare-type alpha \
    --alphas 0.25 0.5 0.75 1.0 1.25 1.5 \
    --hours 36 \
    --country HU \
    --model III

# Narrow range for Model II
python py_script/validation/compare_optimizations.py \
    --compare-type alpha \
    --alphas 0.8 1.0 1.2 \
    --hours 24 \
    --country AT \
    --model II
```

#### Country Comparison

```bash
# Compare all markets
python py_script/validation/compare_optimizations.py \
    --compare-type countries \
    --countries DE_LU AT CH HU CZ \
    --hours 48 \
    --model II \
    --alpha 1.0

# Nordic vs Central Europe
python py_script/validation/compare_optimizations.py \
    --compare-type countries \
    --countries AT CH \
    --hours 168 \
    --model III
```

#### C-Rate Sensitivity

```bash
# Test different power ratings
python py_script/validation/compare_optimizations.py \
    --compare-type c-rates \
    --c-rates 0.25 0.33 0.5 0.75 1.0 \
    --hours 24 \
    --country DE_LU \
    --model II
```

### Output

Comparison results are saved as CSV:

```
validation_results/comparisons/
└── YYYYMMDD_HHMMSS_<comparison-type>_comparison.csv
```

Example output for model comparison:

```csv
model,profit_eur,solve_time_sec,status,cyclic_cost_eur,calendar_cost_eur
I,1234.56,5.2,optimal,,
II,1189.32,8.7,optimal,45.24,
III,1178.91,12.3,optimal,38.67,16.98
```

## Migration Guide

### Old Script → New Command

#### run_36h_hu_winter.py

**Old:**
```bash
python py_script/test/run_36h_hu_winter.py
```

**New:**
```bash
python py_script/validation/run_optimization.py \
    --model III \
    --country HU \
    --hours 36 \
    --alpha 0.5 \
    --use-ev-weighting \
    --plots
```

**Advantages:**
- Can easily change to 24h, 48h, or any duration
- Can test different countries without editing code
- Can adjust alpha, c-rate on the fly
- Standardized output format

#### test_single_32h_vs_mpc.py

**Old:**
```bash
python py_script/test/test_single_32h_vs_mpc.py
```

**New:**
```bash
python py_script/validation/compare_optimizations.py \
    --compare-type single-vs-mpc \
    --hours 32 \
    --country HU
```

**Advantages:**
- Can test different time horizons
- Can adjust MPC parameters
- Can run for different countries
- Standardized CSV output for analysis

## Tips and Best Practices

### Organizing Results

Use meaningful run names:

```bash
python py_script/validation/run_optimization.py \
    --model III \
    --country HU \
    --hours 36 \
    --run-name "HU_winter_baseline_alpha0.5" \
    --plots
```

### Batch Testing

Create shell scripts for common scenarios:

```bash
#!/bin/bash
# validate_all_countries.sh

for country in DE_LU AT CH HU CZ; do
    python py_script/validation/run_optimization.py \
        --model III \
        --country $country \
        --hours 48 \
        --alpha 1.0 \
        --plots
done
```

### Solver Selection

Specify solver for consistency:

```bash
# Always use HiGHS for reproducibility
python py_script/validation/run_optimization.py \
    --model II \
    --country DE_LU \
    --hours 24 \
    --solver highs
```

### Performance Testing

Time-limited solves for large problems:

```bash
# Long horizon with specified solver
python py_script/validation/run_optimization.py \
    --model III \
    --country CH \
    --hours 168 \
    --solver cplex \
    --alpha 0.75
```

## Advanced Usage

### Combining with results_exporter

The scripts use `results_exporter.py` for saving. You can load and analyze results:

```python
from py_script.validation.results_exporter import load_optimization_results, list_saved_results

# List all saved results
all_results = list_saved_results("validation_results")
print(all_results)

# Load specific result
solution_df, metrics = load_optimization_results("validation_results/20251112_143000_hu_baseline")
print(f"Profit: {metrics['total_profit_eur']:.2f} EUR")
```

### Custom Analysis

Load comparison results for analysis:

```python
import pandas as pd

# Load comparison
df = pd.read_csv("validation_results/comparisons/20251112_150000_alpha_comparison.csv")

# Find optimal alpha
best_alpha = df.loc[df['profit_eur'].idxmax(), 'alpha']
print(f"Best alpha: {best_alpha}")

# Plot sensitivity
import matplotlib.pyplot as plt
plt.plot(df['alpha'], df['profit_eur'], marker='o')
plt.xlabel('Alpha')
plt.ylabel('Profit (EUR)')
plt.show()
```

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: Market data not found`
- **Solution:** Check `--data-dir` points to correct location with parquet files

**Issue:** `Solver not available`
- **Solution:** Install solver or use `--solver` to specify available one

**Issue:** `MPC simulator not available` (for single-vs-mpc)
- **Solution:** Ensure `py_script/mpc/mpc_simulator.py` exists and imports correctly

**Issue:** Time window exceeds available data
- **Solution:** Reduce `--hours` or adjust `--start-step`

### Getting Help

View full help:

```bash
python py_script/validation/run_optimization.py --help
python py_script/validation/compare_optimizations.py --help
```

## See Also

- `../test/test_optimizer_core.py` - Formal pytest unit tests
- `results_exporter.py` - Results saving/loading utilities
- `../visualization/optimization_analysis.py` - Plotting functions
- `../visualization/aging_analysis.py` - Degradation visualization
