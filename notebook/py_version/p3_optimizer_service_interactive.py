# -*- coding: utf-8 -*-
"""
Phase 3 OptimizerService - Interactive Educational Script

This script explores the OptimizerService implementation from Phase 3 of the
IBM Hackathon development plan. It demonstrates the unified service layer
that orchestrates the complete optimization workflow.

Run cells individually using Shift+Enter or clicking "Run Cell" above each block.

Learning Objectives:
1. Understand the OptimizerService class architecture
2. Explore the model factory pattern with caching
3. Test the optimize() method with different parameters
4. Examine the OptimizationResult output structure
5. Compare results across different model types
"""

# %%
# ============================================================================
# CELL 1: IMPORTS AND SETUP
# ============================================================================

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from datetime import datetime, timedelta

print("=" * 70)
print("Phase 3: OptimizerService - Interactive Exploration")
print("=" * 70)

# %%
# ============================================================================
# CELL 2: IMPORT THE SERVICE MODULE
# ============================================================================

from src.service import (
    OptimizerService,
    ModelType,
    OptimizationInput,
    OptimizationResult,
    ScheduleEntry,
    RenewableUtilization,
)
from src.service.adapter import DataAdapter

print("\n[OK] Successfully imported OptimizerService and related models!")

print("\n--- Available Model Types ---")
for model in ModelType:
    print(f"  - {model.value}: {model.name}")

print("\n--- Service Module Exports ---")
print("  - OptimizerService: Main service wrapper class")
print("  - ModelType: Enum for optimizer variants (I, II, III, III-renew)")
print("  - OptimizationInput: Standardized input Pydantic model")
print("  - OptimizationResult: Standardized output Pydantic model")
print("  - ScheduleEntry: Per-timestep schedule item")
print("  - RenewableUtilization: Renewable energy breakdown")

# %%
# ============================================================================
# CELL 3: CREATE THE OPTIMIZERSERVICE INSTANCE
# ============================================================================

service = OptimizerService()

print("\n[OK] Created OptimizerService instance!")
print(f"\n--- Service Attributes ---")
print(f"  adapter: {type(service.adapter).__name__}")
print(f"  _optimizer_cache: {service._optimizer_cache}")

print("\n--- DataAdapter Methods ---")
adapter_methods = [m for m in dir(service.adapter) if not m.startswith('_')]
for method in adapter_methods:
    print(f"  - {method}()")

print("\n--- OptimizerService Public Methods ---")
public_methods = [m for m in dir(service) if not m.startswith('_') and callable(getattr(service, m))]
for method in public_methods:
    print(f"  - {method}()")

# %%
# ============================================================================
# CELL 4: EXPLORE THE MODEL FACTORY (_get_optimizer)
# ============================================================================

print("\n--- Testing Model Factory ---")
print("\nCreating different optimizer instances:")

# Test Model I
opt_i = service._get_optimizer("I", 1.0)
print(f"  Model I (alpha=1.0): {type(opt_i).__name__}")

# Test Model II
opt_ii = service._get_optimizer("II", 0.5)
print(f"  Model II (alpha=0.5): {type(opt_ii).__name__}")

# Test Model III
opt_iii = service._get_optimizer("III", 1.0)
print(f"  Model III (alpha=1.0): {type(opt_iii).__name__}")

# Test Model III-renew
opt_iii_renew = service._get_optimizer("III-renew", 1.0)
print(f"  Model III-renew (alpha=1.0): {type(opt_iii_renew).__name__}")

print("\n--- Optimizer Cache State ---")
print(f"  Cache keys: {list(service._optimizer_cache.keys())}")

# Test caching
opt_iii_cached = service._get_optimizer("III", 1.0)
print(f"\n--- Testing Cache ---")
print(f"  Same instance for Model III (alpha=1.0)? {opt_iii is opt_iii_cached}")

opt_iii_different = service._get_optimizer("III", 0.5)
print(f"  Different instance for Model III (alpha=0.5)? {opt_iii is not opt_iii_different}")

# %%
# ============================================================================
# CELL 5: EXPLORE BATTERY CONFIGURATION
# ============================================================================

battery_config = service._load_battery_config()

print("\n--- Default Battery Configuration ---")
print(f"  Capacity: {battery_config['capacity_kwh']} kWh")
print(f"  C-rate: {battery_config['c_rate']} C")
print(f"  Max Power: {battery_config['capacity_kwh'] * battery_config['c_rate']} kW")
print(f"  Efficiency: {battery_config['efficiency'] * 100}%")
print(f"  Initial SOC: {battery_config['initial_soc'] * 100}%")

# %%
# ============================================================================
# CELL 6: CREATE SAMPLE MARKET PRICE DATA
# ============================================================================

print("\n--- Creating Sample Market Price Data ---")

# Time horizon: 48 hours at 15-min resolution = 192 timesteps
# 4-hour blocks: 48 / 4 = 12 blocks

def create_sample_market_prices(base_price=50.0, price_volatility=10.0):
    """Create synthetic market price data for testing."""
    import numpy as np

    # Day-ahead prices: base + daily cycle + random noise
    hours = np.arange(192) / 4  # 0 to 48 hours
    da_prices = (
        base_price
        + 20 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
        + np.random.randn(192) * price_volatility  # Random noise
    )
    da_prices = np.maximum(da_prices, -100)  # Floor at -100 EUR/MWh

    return {
        'day_ahead': da_prices.tolist(),
        'afrr_energy_pos': [40.0] * 192,
        'afrr_energy_neg': [30.0] * 192,
        'fcr': [100.0] * 12,
        'afrr_capacity_pos': [5.0] * 12,
        'afrr_capacity_neg': [10.0] * 12,
    }

market_prices = create_sample_market_prices()

print(f"  Day-ahead prices: {len(market_prices['day_ahead'])} timesteps")
print(f"  aFRR energy prices: {len(market_prices['afrr_energy_pos'])} timesteps")
print(f"  FCR prices: {len(market_prices['fcr'])} blocks")
print(f"\n  Sample DA prices (first 8 hours):")
for i in range(8):
    hour = i // 4
    price = market_prices['day_ahead'][i]
    print(f"    Hour {hour}:00 ({i*15}min) - {price:.2f} EUR/MWh")

# %%
# ============================================================================
# CELL 7: CREATE SAMPLE RENEWABLE GENERATION FORECAST
# ============================================================================

print("\n--- Creating Sample Renewable Generation Forecast ---")

def create_renewable_forecast(base_kw=500, hours=48):
    """Create synthetic renewable generation profile (solar-like)."""
    import numpy as np

    timesteps = hours * 4
    time_hours = np.arange(timesteps) / 4

    # Solar-like profile: zero at night, peak midday
    solar = np.maximum(0, np.sin(np.pi * (time_hours - 6) / 12)) * base_kw
    solar[time_hours < 6] = 0
    solar[time_hours > 18] = 0

    return {
        'generation_kw': solar.tolist()
    }

generation_forecast = create_renewable_forecast(base_kw=1000)

print(f"  Generation timesteps: {len(generation_forecast['generation_kw'])}")
print(f"  Max generation: {max(generation_forecast['generation_kw']):.1f} kW")
print(f"  Total daily energy: {sum(generation_forecast['generation_kw'])/4:.1f} kWh")

# Plot-like visualization
print(f"\n  Hourly generation profile (Day 1):")
for h in range(24):
    idx = h * 4
    gen = generation_forecast['generation_kw'][idx]
    bar = "█" * int(gen / 50)
    print(f"    {h:2d}:00 |{bar:20s}| {gen:6.1f} kW")

# %%
# ============================================================================
# CELL 8: EXPLORE THE OPTIMIZATIONINPUT MODEL
# ============================================================================

print("\n--- Exploring OptimizationInput Model ---")

# Show the schema
print("\nOptimizationInput fields:")
for field_name, field_info in OptimizationInput.model_fields.items():
    default = field_info.default if field_info.default is not None else "(required)"
    print(f"  - {field_name}: {field_info.description}")
    print(f"    Default: {default}")

# %%
# ============================================================================
# CELL 9: EXPLORE THE OPTIMIZATIONRESULT MODEL
# ============================================================================

print("\n--- Exploring OptimizationResult Model ---")

print("\nOptimizationResult fields:")
for field_name, field_info in OptimizationResult.model_fields.items():
    print(f"  - {field_name}: {field_info.description}")

# %%
# ============================================================================
# CELL 10: RUN OPTIMIZATION WITH MOCKED SOLVER (FAST TEST)
# ============================================================================

print("\n--- Running Optimization (Mocked Solver) ---")
print("This uses a mock solver for fast testing without actual MILP solving.\n")

from unittest.mock import Mock, patch

# Create mock optimizer
mock_optimizer = Mock()
mock_model = Mock()
mock_results = Mock()
mock_results._solve_time = 0.5
mock_results._solver_name = 'mock'
mock_results.solver.termination_condition.name = 'optimal'

# Mock solution data
n_timesteps = 192
mock_solution = {
    'status': 'optimal',
    'objective_value': 450.0,
    'solve_time': 0.5,
    'solver': 'mock',
    'profit_da': 350.0,
    'profit_afrr_energy': 80.0,
    'profit_as_capacity': 20.0,
    'cost_cyclic': 15.0,
    'cost_calendar': 10.0,
    'e_soc': {t: 2236.0 + 100 * (0.5 - abs(t/192 - 0.5)) for t in range(n_timesteps)},
    'p_ch': {t: 1000.0 if 32 <= t < 64 else 0.0 for t in range(n_timesteps)},
    'p_dis': {t: 1500.0 if 120 <= t < 152 else 0.0 for t in range(n_timesteps)},
}

mock_optimizer.build_optimization_model.return_value = mock_model
mock_optimizer.solve_model.return_value = (mock_model, mock_results)
mock_optimizer.extract_solution.return_value = mock_solution

# Patch and run
with patch.object(service, '_get_optimizer', return_value=mock_optimizer):
    result = service.optimize(
        market_prices=market_prices,
        generation_forecast=generation_forecast,
        model_type="III",
        c_rate=0.5,
        alpha=1.0,
    )

print("[OK] Optimization completed!")

# %%
# ============================================================================
# CELL 11: EXAMINE THE OPTIMIZATIONRESULT
# ============================================================================

print("\n--- OptimizationResult Breakdown ---")

print(f"\nCore Metrics:")
print(f"  Status: {result.status}")
print(f"  Objective Value: {result.objective_value:.2f} EUR")
print(f"  Net Profit: {result.net_profit:.2f} EUR")

print(f"\nRevenue Breakdown:")
for market, revenue in result.revenue_breakdown.items():
    print(f"  {market}: {revenue:.2f} EUR")

print(f"\nDegradation Costs:")
print(f"  Cyclic Aging: {result.cyclic_aging_cost:.2f} EUR")
print(f"  Calendar Aging: {result.calendar_aging_cost:.2f} EUR")
print(f"  Total Degradation: {result.degradation_cost:.2f} EUR")

print(f"\nSolver Metadata:")
print(f"  Solver: {result.solver_name}")
print(f"  Solve Time: {result.solve_time_seconds:.3f} seconds")
print(f"  Model Type: {result.model_type.value}")

# %%
# ============================================================================
# CELL 12: EXPLORE THE SCHEDULE
# ============================================================================

print("\n--- Schedule Analysis ---")

print(f"\nSchedule Entries: {len(result.schedule)} timesteps")

# Show first 24 hours (hourly samples)
print("\nFirst 24 hours (hourly samples):")
print(f"{'Time':>10} | {'Action':>10} | {'Power (kW)':>12} | {'SOC':>8}")
print("-" * 50)

for h in range(24):
    entry = result.schedule[h * 4]
    print(f"{entry.timestamp.strftime('%H:%M'):>10} | {entry.action:>10} | {entry.power_kw:>12.1f} | {entry.soc_after:>7.2%}")

# Find charge/discharge periods
charge_hours = sum(1 for e in result.schedule if e.action == 'charge') / 4
discharge_hours = sum(1 for e in result.schedule if e.action == 'discharge') / 4
idle_hours = sum(1 for e in result.schedule if e.action == 'idle') / 4

print(f"\nAction Summary:")
print(f"  Charging: {charge_hours:.1f} hours")
print(f"  Discharging: {discharge_hours:.1f} hours")
print(f"  Idle: {idle_hours:.1f} hours")

# %%
# ============================================================================
# CELL 13: VISUALIZE SOC TRAJECTORY
# ============================================================================

print("\n--- SOC Trajectory Analysis ---")

soc_min = min(result.soc_trajectory)
soc_max = max(result.soc_trajectory)
soc_avg = sum(result.soc_trajectory) / len(result.soc_trajectory)

print(f"\nSOC Statistics:")
print(f"  Minimum SOC: {soc_min:.2%}")
print(f"  Maximum SOC: {soc_max:.2%}")
print(f"  Average SOC: {soc_avg:.2%}")
print(f"  SOC Range: {soc_max - soc_min:.2%}")

# ASCII visualization of SOC trajectory
print("\nSOC Trajectory (48 hours, 4-hour intervals):")
print("   0%    25%    50%    75%   100%")
print("   |-----|-----|-----|-----|")
for h in range(0, 48, 4):
    idx = h * 4
    soc = result.soc_trajectory[idx]
    bar_pos = int(soc * 20)
    bar = " " * bar_pos + "█"
    print(f"{h:3d}h |{bar:21s}| {soc:.1%}")

# %%
# ============================================================================
# CELL 14: COMPARE MODEL TYPES
# ============================================================================

print("\n--- Comparing Different Model Types ---")
print("Running optimizations with Models I, II, and III...\n")

model_comparison = {}

for model_type in ["I", "II", "III"]:
    # Update mock solution to show different values per model
    mock_solution_i = mock_solution.copy()
    if model_type == "I":
        mock_solution_i['objective_value'] = 500.0
        mock_solution_i['cost_cyclic'] = 0.0
        mock_solution_i['cost_calendar'] = 0.0
    elif model_type == "II":
        mock_solution_i['objective_value'] = 485.0
        mock_solution_i['cost_cyclic'] = 15.0
        mock_solution_i['cost_calendar'] = 0.0
    else:  # III
        mock_solution_i['objective_value'] = 450.0
        mock_solution_i['cost_cyclic'] = 15.0
        mock_solution_i['cost_calendar'] = 10.0

    mock_optimizer.extract_solution.return_value = mock_solution_i

    with patch.object(service, '_get_optimizer', return_value=mock_optimizer):
        result = service.optimize(
            market_prices=market_prices,
            model_type=model_type,
            c_rate=0.5,
            alpha=1.0,
        )

    model_comparison[model_type] = result

print("Model Comparison Results:")
print(f"{'Model':>8} | {'Objective':>12} | {'Cyclic Cost':>12} | {'Calendar Cost':>14} | {'Net Profit':>12}")
print("-" * 80)

for model_type, result in model_comparison.items():
    print(f"{model_type:>8} | {result.objective_value:>12.2f} | {result.cyclic_aging_cost:>12.2f} | "
          f"{result.calendar_aging_cost:>14.2f} | {result.net_profit:>12.2f}")

# %%
# ============================================================================
# CELL 15: TEST WITH RENEWABLE INTEGRATION (Model III-renew)
# ============================================================================

print("\n--- Testing Renewable Integration (Model III-renew) ---")

# Create solution with renewable utilization
mock_solution_renew = mock_solution.copy()
mock_solution_renew['objective_value'] = 550.0
mock_solution_renew['profit_renewable_export'] = 100.0
mock_solution_renew['p_renewable_self'] = {t: min(500, generation_forecast['generation_kw'][t]) for t in range(192)}
mock_solution_renew['renewable_utilization'] = {
    'total_generation_kwh': 8000.0,
    'self_consumption_kwh': 5000.0,
    'export_kwh': 2000.0,
    'curtailment_kwh': 1000.0,
    'utilization_rate': 0.875,
}

mock_optimizer.extract_solution.return_value = mock_solution_renew

with patch.object(service, '_get_optimizer', return_value=mock_optimizer):
    result = service.optimize(
        market_prices=market_prices,
        generation_forecast=generation_forecast,
        model_type="III-renew",
        c_rate=0.5,
        alpha=1.0,
    )

print(f"\nResults with Renewable Integration:")
print(f"  Objective Value: {result.objective_value:.2f} EUR")
print(f"  Net Profit: {result.net_profit:.2f} EUR")

if result.renewable_utilization:
    print(f"\n  Renewable Utilization:")
    print(f"    Total Generation: {result.renewable_utilization.total_generation_kwh:.1f} kWh")
    print(f"    Self-Consumed: {result.renewable_utilization.self_consumption_kwh:.1f} kWh")
    print(f"    Exported: {result.renewable_utilization.export_kwh:.1f} kWh")
    print(f"    Curtailed: {result.renewable_utilization.curtailment_kwh:.1f} kWh")
    print(f"    Utilization Rate: {result.renewable_utilization.utilization_rate:.1%}")

# %%
# ============================================================================
# CELL 16: TEST ERROR HANDLING
# ============================================================================

print("\n--- Testing Error Handling ---")

# Create solution with error status
mock_solution_error = {
    'status': 'error',
    'solve_time': 0.1,
    'solver': 'test',
}

mock_optimizer.extract_solution.return_value = mock_solution_error

with patch.object(service, '_get_optimizer', return_value=mock_optimizer):
    result = service.optimize(market_prices=market_prices)

print(f"Error Status Handling:")
print(f"  Status: {result.status}")
print(f"  Objective Value: {result.objective_value:.2f} EUR")
print(f"  Schedule Length: {len(result.schedule)}")
print(f"  SOC Trajectory Length: {len(result.soc_trajectory)}")

# %%
# ============================================================================
# CELL 17: EXPLORE SERIALIZATION (JSON EXPORT)
# ============================================================================

print("\n--- Testing JSON Serialization ---")

# Create a fresh result for serialization
mock_optimizer.extract_solution.return_value = mock_solution

with patch.object(service, '_get_optimizer', return_value=mock_optimizer):
    result = service.optimize(market_prices=market_prices, model_type="III")

# Convert to JSON
result_json = result.model_dump(mode='json')

print(f"Serialization successful!")
print(f"\nResult JSON structure (top-level keys):")
for key in result_json.keys():
    value = result_json[key]
    if isinstance(value, list):
        print(f"  {key}: list ({len(value)} items)")
    elif isinstance(value, dict):
        print(f"  {key}: dict ({len(value)} keys)")
    else:
        print(f"  {key}: {type(value).__name__}")

# Show a sample schedule entry as JSON
if result_json['schedule']:
    print(f"\nSample Schedule Entry (JSON):")
    print(json.dumps(result_json['schedule'][32], indent=2))

# %%
# ============================================================================
# CELL 18: TEST OPTIMIZE_FROM_INPUT METHOD
# ============================================================================

print("\n--- Testing optimize_from_input() ---")

# Create OptimizationInput directly
opt_input = OptimizationInput(
    time_horizon_hours=48,
    da_prices=[50.0] * 192,
    afrr_energy_pos=[40.0] * 192,
    afrr_energy_neg=[30.0] * 192,
    fcr_prices=[100.0] * 12,
    afrr_capacity_pos=[5.0] * 12,
    afrr_capacity_neg=[10.0] * 12,
    model_type=ModelType.MODEL_III,
    alpha=0.8,
    c_rate=0.33,
)

print(f"Created OptimizationInput:")
print(f"  Model Type: {opt_input.model_type.value}")
print(f"  Alpha: {opt_input.alpha}")
print(f"  C-rate: {opt_input.c_rate}")
print(f"  Time Horizon: {opt_input.time_horizon_hours} hours")

# Run optimization
mock_optimizer.extract_solution.return_value = mock_solution

with patch.object(service, '_get_optimizer', return_value=mock_optimizer):
    result = service.optimize_from_input(opt_input)

print(f"\n[OK] Optimization completed!")
print(f"  Status: {result.status}")
print(f"  Objective Value: {result.objective_value:.2f} EUR")

# %%
# ============================================================================
# CELL 19: SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 3: OptimizerService - Summary")
print("=" * 70)

print("""
The OptimizerService provides a unified interface for battery optimization:

Key Features:
1. Model Factory: Creates and caches optimizer instances (I, II, III, III-renew)
2. Data Adaptation: Converts raw input to optimizer-compatible format
3. Solving: Orchestrates the build-solve-extract workflow
4. Result Formatting: Returns standardized OptimizationResult

Usage Patterns:
- optimize(): High-level API with raw market/generation data
- optimize_from_input(): Use with pre-built OptimizationInput
- _get_optimizer(): Factory method with model type caching

Service Layer Benefits:
- Consistent API for external modules (Price Service, Weather Service)
- Type-safe I/O with Pydantic models
- Easy integration with FastAPI endpoints
- Testable with mock solvers

Next Steps:
- Phase 4: Add FastAPI endpoints
- Phase 4: Create Docker deployment
- Phase 4: Integration tests with real solver
""")

print("=" * 70)
print("[OK] Script completed successfully!")
print("=" * 70)

# %%
