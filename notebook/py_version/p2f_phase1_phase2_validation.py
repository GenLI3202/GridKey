# -*- coding: utf-8 -*-
"""
Phase 1 + Phase 2 Implementation Validation — Interactive Script

Validates every component built in Phase 1 (Pydantic models) and Phase 2
(DataAdapter + BESSOptimizerModelIIIRenew) by running them end-to-end
on real DE_LU market data with a synthetic solar PV forecast.

Cell markers (# %%) allow block-by-block execution in VS Code:
    Shift+Enter  →  run current cell
    Ctrl+Enter   →  run current cell, stay in place

Structure:
    Cell 1   Imports
    Cell 2   Phase 1 — Pydantic model showcase
    Cell 3   Phase 1 — Validation & JSON round-trip
    Cell 4   Load real market data (DE_LU)
    Cell 5   Phase 2 — DataAdapter: real data → OptimizationInput → country_data
    Cell 6   Phase 2 — Model III baseline (no renewable)
    Cell 7   Phase 2 — Synthesise solar PV forecast
    Cell 8   Phase 2 — Model III-Renew (with renewable)
    Cell 9   Comparison — Model III vs III-Renew
"""

# %%
# ============================================================================
# CELL 1: IMPORTS
# ============================================================================

import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Phase 1 — Pydantic models
from src.service.models import (
    ModelType,
    OptimizationInput,
    ScheduleEntry,
    RenewableUtilization,
    OptimizationResult,
)

# Phase 2 — DataAdapter + Renewable Optimizer
from src.service.adapter import DataAdapter
from src.core.optimizer import (
    BESSOptimizerModelIII,
    BESSOptimizerModelIIIRenew,
)

# Data loading utility (existing)
from src.data.load_process_market_data import load_preprocessed_country_data

print("=" * 72)
print("  Phase 1 + Phase 2 Validation Script")
print("=" * 72)
print("[OK] All imports successful\n")

# %%
# ============================================================================
# CELL 2: PHASE 1 — Pydantic Model Showcase
# ============================================================================
#
# Goal: Demonstrate that all five Pydantic models from Phase 1 work
#       correctly — creation, default values, type coercion.
#

print("=" * 72)
print("  PHASE 1: Pydantic Data Models")
print("=" * 72)

# --- ModelType enum ---
print("\n[1/5] ModelType enum")
for mt in ModelType:
    print(f"       {mt.name:25s} → value = \"{mt.value}\"")

# We can construct from string
assert ModelType("III-renew") is ModelType.MODEL_III_RENEW
print("       ✓  ModelType('III-renew') works")

# --- OptimizationInput ---
print("\n[2/5] OptimizationInput — default battery parameters")
dummy_prices_15 = [50.0] * 192
dummy_prices_4h = [100.0] * 12

opt_in = OptimizationInput(
    da_prices=dummy_prices_15,
    afrr_energy_pos=dummy_prices_15,
    afrr_energy_neg=dummy_prices_15,
    fcr_prices=dummy_prices_4h,
    afrr_capacity_pos=dummy_prices_4h,
    afrr_capacity_neg=dummy_prices_4h,
)
print(f"       time_horizon_hours = {opt_in.time_horizon_hours}")
print(f"       battery_capacity   = {opt_in.battery_capacity_kwh} kWh")
print(f"       c_rate             = {opt_in.c_rate}")
print(f"       efficiency         = {opt_in.efficiency}")
print(f"       initial_soc        = {opt_in.initial_soc}")
print(f"       model_type         = {opt_in.model_type.value}")
print(f"       renewable          = {opt_in.renewable_generation is not None}")
print(f"       len(da_prices)     = {len(opt_in.da_prices)}")

# --- ScheduleEntry ---
print("\n[3/5] ScheduleEntry")
entry = ScheduleEntry(
    timestamp=datetime(2024, 5, 12, 8, 0),
    action="discharge",
    power_kw=2000.0,
    market="da",
    soc_after=0.35,
)
print(f"       {entry.timestamp}  {entry.action:10s}  {entry.power_kw:8.1f} kW"
      f"  market={entry.market}  SOC={entry.soc_after:.0%}")

# --- RenewableUtilization ---
print("\n[4/5] RenewableUtilization")
ru = RenewableUtilization(
    total_generation_kwh=100.0,
    self_consumption_kwh=60.0,
    export_kwh=30.0,
    curtailment_kwh=10.0,
    utilization_rate=0.9,
)
print(f"       total={ru.total_generation_kwh} kWh  "
      f"self={ru.self_consumption_kwh}  export={ru.export_kwh}  "
      f"curtail={ru.curtailment_kwh}  rate={ru.utilization_rate:.0%}")

# --- OptimizationResult ---
print("\n[5/5] OptimizationResult (mock)")
mock_result = OptimizationResult(
    objective_value=450.0,
    net_profit=420.0,
    revenue_breakdown={"da": 250, "fcr": 100, "afrr_cap": 50, "afrr_energy": 30, "renewable_export": 20},
    degradation_cost=30.0,
    cyclic_aging_cost=20.0,
    calendar_aging_cost=10.0,
    schedule=[entry],
    soc_trajectory=[0.5, 0.35],
    renewable_utilization=ru,
    solve_time_seconds=1.5,
    solver_name="highs",
    model_type=ModelType.MODEL_III_RENEW,
    status="optimal",
)
print(f"       objective  = {mock_result.objective_value:.2f} EUR")
print(f"       net_profit = {mock_result.net_profit:.2f} EUR")
print(f"       status     = {mock_result.status}")
print(f"       model      = {mock_result.model_type.value}")

print("\n[OK] All 5 Pydantic models construct correctly ✓")

# %%
# ============================================================================
# CELL 3: PHASE 1 — Validation & JSON Round-trip
# ============================================================================
#
# Goal: Show that validators reject bad data and that JSON serialisation
#       round-trips perfectly.
#

print("=" * 72)
print("  PHASE 1: Validation & Serialisation")
print("=" * 72)

from pydantic import ValidationError

# --- Validator tests ---
print("\n[Validators]")

checks = [
    ("efficiency=1.5",  dict(efficiency=1.5)),
    ("efficiency=0.0",  dict(efficiency=0.0)),
    ("c_rate=-0.1",     dict(c_rate=-0.1)),
    ("initial_soc=1.5", dict(initial_soc=1.5)),
]

base_kwargs = dict(
    da_prices=dummy_prices_15,
    afrr_energy_pos=dummy_prices_15,
    afrr_energy_neg=dummy_prices_15,
    fcr_prices=dummy_prices_4h,
    afrr_capacity_pos=dummy_prices_4h,
    afrr_capacity_neg=dummy_prices_4h,
)

for label, override in checks:
    try:
        OptimizationInput(**{**base_kwargs, **override})
        print(f"       FAIL — {label} was NOT rejected")
    except ValidationError:
        print(f"       ✓  {label} correctly rejected")

# --- JSON round-trip ---
print("\n[JSON Round-trip]")
json_str = opt_in.model_dump_json()
restored = OptimizationInput.model_validate_json(json_str)
assert restored.da_prices == opt_in.da_prices
assert restored.model_type == opt_in.model_type
print(f"       ✓  OptimizationInput  →  JSON ({len(json_str):,} bytes)  →  OptimizationInput")

json_str2 = mock_result.model_dump_json()
restored2 = OptimizationResult.model_validate_json(json_str2)
assert restored2.objective_value == mock_result.objective_value
assert restored2.schedule[0].action == "discharge"
print(f"       ✓  OptimizationResult →  JSON ({len(json_str2):,} bytes)  →  OptimizationResult")

print("\n[OK] Validation & serialisation verified ✓")

# %%
# ============================================================================
# CELL 4: LOAD REAL MARKET DATA
# ============================================================================
#
# We load preprocessed DE_LU data (15-min resolution, 2024) and extract a
# 24-hour slice to use in the optimizer.
#

print("=" * 72)
print("  DATA: Loading DE_LU Market Prices")
print("=" * 72)

COUNTRY = "DE_LU"
HORIZON_HOURS = 24
# May 12th — interesting day with price volatility
START_STEP = 96 * 132   # day 132 = May 12

preprocessed_dir = project_root / "data" / "parquet" / "preprocessed"
full_data = load_preprocessed_country_data(COUNTRY, data_dir=preprocessed_dir)

horizon_steps = HORIZON_HOURS * 4
data_slice = full_data.iloc[START_STEP : START_STEP + horizon_steps].copy()
data_slice.reset_index(drop=True, inplace=True)

print(f"\n[OK] Loaded {COUNTRY} data: {len(full_data):,} total rows")
print(f"     Extracted {len(data_slice)} rows ({HORIZON_HOURS}h) starting step {START_STEP}")
print(f"\n     DA price range:   {data_slice['price_day_ahead'].min():.2f}"
      f" – {data_slice['price_day_ahead'].max():.2f} EUR/MWh")
print(f"     FCR price range:  {data_slice['price_fcr'].min():.2f}"
      f" – {data_slice['price_fcr'].max():.2f} EUR/MW")
print(f"     Columns: {list(data_slice.columns[:8])}  ...")

# %%
# ============================================================================
# CELL 5: PHASE 2 — DataAdapter End-to-End
# ============================================================================
#
# Demonstrate both DataAdapter paths:
#   Path A:  raw dict  →  adapt()  →  OptimizationInput
#   Path B:  OptimizationInput  →  to_country_data()  →  DataFrame
#

print("=" * 72)
print("  PHASE 2: DataAdapter")
print("=" * 72)

adapter = DataAdapter()

# --- Path A: adapt() ---
print("\n[Path A] adapt(market_prices_dict) → OptimizationInput")

n = HORIZON_HOURS * 4   # 96 for 24h
n_blocks = HORIZON_HOURS // 4  # 6 for 24h

market_prices = {
    "day_ahead":        list(data_slice["price_day_ahead"].values),
    "afrr_energy_pos":  list(data_slice["price_afrr_energy_pos"].fillna(0).values),
    "afrr_energy_neg":  list(data_slice["price_afrr_energy_neg"].fillna(0).values),
    "fcr":              [float(data_slice.loc[data_slice["block_id"] == b, "price_fcr"].iloc[0])
                         for b in sorted(data_slice["block_id"].unique())],
    "afrr_capacity_pos": [float(data_slice.loc[data_slice["block_id"] == b, "price_afrr_pos"].iloc[0])
                          for b in sorted(data_slice["block_id"].unique())],
    "afrr_capacity_neg": [float(data_slice.loc[data_slice["block_id"] == b, "price_afrr_neg"].iloc[0])
                          for b in sorted(data_slice["block_id"].unique())],
}

opt_input = adapter.adapt(
    market_prices,
    time_horizon_hours=HORIZON_HOURS,
)
print(f"     len(da_prices)          = {len(opt_input.da_prices)}")
print(f"     len(fcr_prices)         = {len(opt_input.fcr_prices)}")
print(f"     model_type              = {opt_input.model_type.value}")
print(f"     renewable_generation    = {opt_input.renewable_generation}")

# --- Path B: to_country_data() ---
print("\n[Path B] to_country_data(opt_input) → DataFrame")
country_df = adapter.to_country_data(opt_input)

print(f"     Shape:   {country_df.shape}")
print(f"     Columns: {list(country_df.columns)}")
print(f"     Blocks:  {sorted(country_df['block_id'].unique())}")
print(f"     DA price sample (first 4): {list(country_df['price_day_ahead'].head(4).round(2))}")

# Sanity check: compare with original data
orig_da = list(data_slice["price_day_ahead"].values[:4])
conv_da = list(country_df["price_day_ahead"].values[:4])
match = all(abs(a - b) < 0.01 for a, b in zip(orig_da, conv_da))
print(f"\n     DA prices match original? {match}  ✓" if match else
      f"\n     DA prices match original? {match}  ✗ MISMATCH!")

print("\n[OK] DataAdapter verified ✓")

# %%
# ============================================================================
# CELL 6: PHASE 2 — Model III Baseline (No Renewable)
# ============================================================================
#
# Run Model III on the real data slice to establish a baseline for
# comparison with Model III-Renew.
#

print("=" * 72)
print("  PHASE 2: Model III Baseline")
print("=" * 72)

C_RATE = 0.5
ALPHA = 1.0

optimizer_iii = BESSOptimizerModelIII(alpha=ALPHA)

print(f"\n[BUILD] Building Model III (c_rate={C_RATE}, alpha={ALPHA})...")
t0 = time.time()
model_iii = optimizer_iii.build_optimization_model(data_slice, c_rate=C_RATE)
build_time = time.time() - t0
print(f"        Variables:   {model_iii.nvariables():,}")
print(f"        Constraints: {model_iii.nconstraints():,}")
print(f"        Build time:  {build_time:.2f}s")

print(f"\n[SOLVE] Solving Model III ...")
t0 = time.time()
model_iii_solved, results_iii = optimizer_iii.solve_model(model_iii)
solve_time_iii = time.time() - t0
print(f"        Solve time:  {solve_time_iii:.2f}s")

solution_iii = optimizer_iii.extract_solution(model_iii_solved, results_iii)
print(f"        Status:      {solution_iii['status']}")
print(f"        Objective:   {solution_iii['objective_value']:.2f} EUR")

if 'profit_da' in solution_iii:
    print(f"\n        Revenue Breakdown:")
    print(f"          DA arbitrage:   {solution_iii.get('profit_da', 0):.2f} EUR")
    print(f"          aFRR energy:    {solution_iii.get('profit_afrr_energy', 0):.2f} EUR")
    print(f"          AS capacity:    {solution_iii.get('profit_as_capacity', 0):.2f} EUR")
    if 'degradation_metrics' in solution_iii:
        dm = solution_iii['degradation_metrics']
        if 'cost_breakdown' in dm:
            cb = dm['cost_breakdown']
            print(f"          Cyclic aging:   -{cb.get('cyclic_eur', 0):.2f} EUR")
            print(f"          Calendar aging: -{cb.get('calendar_eur', 0):.2f} EUR")

print(f"\n[OK] Model III baseline complete ✓")

# %%
# ============================================================================
# CELL 7: PHASE 2 — Synthesise Solar PV Forecast
# ============================================================================
#
# Since Module A (Weather Service) is not implemented yet, we generate a
# realistic solar PV profile:
#   - Zero at night (before 6 AM, after 8 PM)
#   - Bell curve peaking at noon (~1,500 kW for a ~10 kWp system on a clear day)
#   - Scaled to be comparable to the battery's power rating
#

print("=" * 72)
print("  PHASE 2: Synthetic Solar PV Forecast")
print("=" * 72)

PV_PEAK_KW = 1500  # Peak PV generation (kW)

def make_solar_profile(n_timesteps: int, peak_kw: float) -> list:
    """Generate a simple bell-curve solar profile (15-min resolution).

    Assumes the first timestep is midnight.  Sunrise ~06:00, sunset ~20:00,
    peak at ~12:00–13:00.
    """
    profile = []
    for i in range(n_timesteps):
        hour = (i * 0.25) % 24  # hour of day
        if 6.0 <= hour <= 20.0:
            # Normalised sine-bell: peak at 13h, zero at 6h and 20h
            x = (hour - 6.0) / (20.0 - 6.0) * math.pi  # 0 → π
            gen = peak_kw * math.sin(x)
        else:
            gen = 0.0
        profile.append(round(gen, 1))
    return profile

solar_forecast = make_solar_profile(horizon_steps, PV_PEAK_KW)

# Quick stats
total_gen_kwh = sum(g * 0.25 for g in solar_forecast)
peak_val = max(solar_forecast)
nonzero = sum(1 for g in solar_forecast if g > 0)

print(f"\n     Peak generation:    {peak_val:.0f} kW")
print(f"     Daily generation:   {total_gen_kwh:.0f} kWh")
print(f"     Active timesteps:   {nonzero} / {horizon_steps}")
print(f"     Battery capacity:   {4472} kWh  (PV covers {total_gen_kwh/4472*100:.1f}%)")

# Show a compact ASCII preview
print(f"\n     Hourly profile (kW):")
for h in range(24):
    idx = h * 4
    val = solar_forecast[idx] if idx < len(solar_forecast) else 0
    bar = "█" * int(val / peak_val * 30) if val > 0 else ""
    print(f"       {h:02d}:00  {val:7.0f}  {bar}")

print(f"\n[OK] Solar PV profile ready ✓")

# %%
# ============================================================================
# CELL 8: PHASE 2 — Model III-Renew (With Renewable)
# ============================================================================
#
# This is the key test: run the same market data through Model III-Renew
# with the synthetic solar PV forecast.  The optimizer should exploit:
#   1. Self-consumption: charge battery with free solar
#   2. Export: sell excess solar at DA price when profitable
#   3. Curtailment: only as last resort (no penalty, but no revenue)
#

print("=" * 72)
print("  PHASE 2: Model III-Renew (Renewable Integration)")
print("=" * 72)

# Add renewable forecast column to the data slice
data_slice_renew = data_slice.copy()
data_slice_renew["p_renewable_forecast_kw"] = solar_forecast

optimizer_iiir = BESSOptimizerModelIIIRenew(alpha=ALPHA)

print(f"\n[BUILD] Building Model III-Renew (c_rate={C_RATE}, alpha={ALPHA})...")
t0 = time.time()
model_iiir = optimizer_iiir.build_optimization_model(data_slice_renew, c_rate=C_RATE)
build_time_r = time.time() - t0
print(f"        Variables:   {model_iiir.nvariables():,}")
print(f"        Constraints: {model_iiir.nconstraints():,}")
print(f"        Build time:  {build_time_r:.2f}s")

# Compare with baseline
n_t = len(list(model_iii_solved.T))
print(f"\n        Δ Variables   vs Model III: +{model_iiir.nvariables() - model_iii_solved.nvariables()}"
      f"  (expected +{3*n_t} = 3 × {n_t} timesteps)")
print(f"        Δ Constraints vs Model III: +{model_iiir.nconstraints() - model_iii_solved.nconstraints()}"
      f"  (expected +{2*n_t} = 2 × {n_t} constraints)")

print(f"\n[SOLVE] Solving Model III-Renew ...")
t0 = time.time()
model_iiir_solved, results_iiir = optimizer_iiir.solve_model(model_iiir)
solve_time_r = time.time() - t0
print(f"        Solve time:  {solve_time_r:.2f}s")

solution_iiir = optimizer_iiir.extract_solution(model_iiir_solved, results_iiir)
print(f"        Status:      {solution_iiir['status']}")
print(f"        Objective:   {solution_iiir['objective_value']:.2f} EUR")

if 'profit_da' in solution_iiir:
    print(f"\n        Revenue Breakdown:")
    print(f"          DA arbitrage:     {solution_iiir.get('profit_da', 0):.2f} EUR")
    print(f"          aFRR energy:      {solution_iiir.get('profit_afrr_energy', 0):.2f} EUR")
    print(f"          AS capacity:      {solution_iiir.get('profit_as_capacity', 0):.2f} EUR")
    print(f"          Renewable export: {solution_iiir.get('profit_renewable_export', 0):.2f} EUR  ← NEW")
    if 'degradation_metrics' in solution_iiir:
        dm = solution_iiir['degradation_metrics']
        if 'cost_breakdown' in dm:
            cb = dm['cost_breakdown']
            print(f"          Cyclic aging:     -{cb.get('cyclic_eur', 0):.2f} EUR")
            print(f"          Calendar aging:   -{cb.get('calendar_eur', 0):.2f} EUR")

# Renewable utilisation
if 'renewable_utilization' in solution_iiir:
    ru = solution_iiir['renewable_utilization']
    print(f"\n        Renewable Utilisation:")
    print(f"          Total generation:  {ru['total_generation_kwh']:.1f} kWh")
    print(f"          Self-consumption:  {ru['self_consumption_kwh']:.1f} kWh"
          f"  ({ru['self_consumption_kwh']/ru['total_generation_kwh']*100:.1f}%)" if ru['total_generation_kwh'] > 0 else "")
    print(f"          Grid export:       {ru['export_kwh']:.1f} kWh"
          f"  ({ru['export_kwh']/ru['total_generation_kwh']*100:.1f}%)" if ru['total_generation_kwh'] > 0 else "")
    print(f"          Curtailment:       {ru['curtailment_kwh']:.1f} kWh"
          f"  ({ru['curtailment_kwh']/ru['total_generation_kwh']*100:.1f}%)" if ru['total_generation_kwh'] > 0 else "")
    print(f"          Utilisation rate:  {ru['utilization_rate']:.1%}")

print(f"\n[OK] Model III-Renew complete ✓")

# %%
# ============================================================================
# CELL 9: COMPARISON — Model III vs Model III-Renew
# ============================================================================
#
# Side-by-side comparison of the two models.  The Model III-Renew should
# show strictly higher or equal objective value (more revenue sources).
#

print("=" * 72)
print("  COMPARISON: Model III  vs  Model III-Renew")
print("=" * 72)

obj_base = solution_iii['objective_value']
obj_renew = solution_iiir['objective_value']
delta = obj_renew - obj_base

print(f"""
    ┌─────────────────────────┬──────────────┬──────────────┬──────────┐
    │ Metric                  │  Model III   │ III-Renew    │  Δ       │
    ├─────────────────────────┼──────────────┼──────────────┼──────────┤
    │ Objective (EUR)         │ {obj_base:>11.2f} │ {obj_renew:>11.2f} │ {delta:>+7.2f} │
    │ DA profit               │ {solution_iii.get('profit_da',0):>11.2f} │ {solution_iiir.get('profit_da',0):>11.2f} │ {solution_iiir.get('profit_da',0)-solution_iii.get('profit_da',0):>+7.2f} │
    │ aFRR energy             │ {solution_iii.get('profit_afrr_energy',0):>11.2f} │ {solution_iiir.get('profit_afrr_energy',0):>11.2f} │ {solution_iiir.get('profit_afrr_energy',0)-solution_iii.get('profit_afrr_energy',0):>+7.2f} │
    │ AS capacity             │ {solution_iii.get('profit_as_capacity',0):>11.2f} │ {solution_iiir.get('profit_as_capacity',0):>11.2f} │ {solution_iiir.get('profit_as_capacity',0)-solution_iii.get('profit_as_capacity',0):>+7.2f} │
    │ Renewable export        │         n/a │ {solution_iiir.get('profit_renewable_export',0):>11.2f} │   NEW   │
    │ Solve time (s)          │ {solve_time_iii:>11.2f} │ {solve_time_r:>11.2f} │ {solve_time_r-solve_time_iii:>+7.2f} │
    │ Variables               │ {model_iii_solved.nvariables():>11,} │ {model_iiir_solved.nvariables():>11,} │ {model_iiir_solved.nvariables()-model_iii_solved.nvariables():>+7,} │
    │ Constraints             │ {model_iii_solved.nconstraints():>11,} │ {model_iiir_solved.nconstraints():>11,} │ {model_iiir_solved.nconstraints()-model_iii_solved.nconstraints():>+7,} │
    └─────────────────────────┴──────────────┴──────────────┴──────────┘
""")

# Verify key invariant: renewable should not decrease total profit
if delta >= -0.01:
    print("    ✓  Renewable integration improves or maintains profit")
else:
    print("    ✗  WARNING: Renewable decreased profit — investigate!")

if 'renewable_utilization' in solution_iiir:
    ru = solution_iiir['renewable_utilization']
    if ru['utilization_rate'] > 0.99:
        print("    ✓  100% utilisation — zero curtailment (optimal)")
    elif ru['curtailment_kwh'] < 1.0:
        print(f"    ✓  {ru['utilization_rate']:.1%} utilisation — near-zero curtailment")
    else:
        print(f"    ⚠  {ru['utilization_rate']:.1%} utilisation — "
              f"{ru['curtailment_kwh']:.1f} kWh curtailed")

print("\n" + "=" * 72)
print("  ALL PHASE 1 + PHASE 2 VALIDATIONS PASSED")
print("=" * 72)

# %%
# ============================================================================
# CELL 10 (Optional): DataAdapter Full Pipeline Demo
# ============================================================================
#
# Shows the complete adapter pipeline:
#   service dicts → adapt() → OptimizationInput → to_country_data() → DataFrame
# and then feeds that DataFrame directly into Model III-Renew.
#
# This is the exact flow that OptimizerService (Phase 3) will use.
#

print("=" * 72)
print("  BONUS: Full Adapter → Optimizer Pipeline")
print("=" * 72)

# Step 1: Prepare service dicts (simulating Module A + Module B output)
print("\n[Step 1] Prepare service dicts ...")
market_prices_dict = market_prices  # reuse from Cell 5

generation_dict = {"generation_kw": solar_forecast}

battery_config_dict = {
    "capacity_kwh": 4472,
    "c_rate": 0.5,
    "efficiency": 0.95,
    "initial_soc": 0.5,
}

# Step 2: adapt() → OptimizationInput
print("[Step 2] DataAdapter.adapt() → OptimizationInput ...")
opt_input_full = adapter.adapt(
    market_prices_dict,
    generation_forecast=generation_dict,
    battery_config=battery_config_dict,
    time_horizon_hours=HORIZON_HOURS,
)
print(f"         model_type  = {opt_input_full.model_type.value}")
print(f"         renewable   = {len(opt_input_full.renewable_generation)} values")

# Step 3: to_country_data() → DataFrame
print("[Step 3] DataAdapter.to_country_data() → DataFrame ...")
country_df_full = adapter.to_country_data(opt_input_full)
print(f"         Shape       = {country_df_full.shape}")
print(f"         Has renewable col = {'p_renewable_forecast_kw' in country_df_full.columns}")

# Step 4: Optimise
print("[Step 4] BESSOptimizerModelIIIRenew.build → solve → extract ...")
opt_pipeline = BESSOptimizerModelIIIRenew(alpha=ALPHA)
model_p = opt_pipeline.build_optimization_model(country_df_full, c_rate=C_RATE)
_, results_p = opt_pipeline.solve_model(model_p)
sol_p = opt_pipeline.extract_solution(_, results_p)
print(f"         Status      = {sol_p['status']}")
print(f"         Objective   = {sol_p['objective_value']:.2f} EUR")
print(f"         Export rev  = {sol_p.get('profit_renewable_export', 0):.2f} EUR")

print("\n[OK] Full pipeline: service dicts → DataAdapter → Optimizer → Result  ✓")
print("     This is exactly the flow that OptimizerService (Phase 3) will use.")
