# Optimizer Data Loading and Model Building Pipeline

**Document Purpose:** Complete reference for understanding how the BESS optimizer processes market data and constructs the optimization model.

**Date:** 2025-11-13
**Pipeline Version:** Phase 2 Excel-based (Updated)
**Context:** TechArena 2025 Phase II - BESS Optimization

---

## ⚠️ Important Update

**The data pipeline has been modernized for Phase 2:**

- **✅ NEW (Phase 2)**: Excel-based pipeline with preprocessed fast path
  - Primary: `data/TechArena2025_Phase2_data.xlsx` (submission path)
  - Fast path: `data/parquet/preprocessed/{country}.parquet` (validation)

- **❌ DEPRECATED (Phase 1)**: JSONL-based pipeline
  - Still documented below for reference
  - Will be removed in future cleanup

**For current Phase 2 implementation, see the [Phase 2 Data Pipeline](#phase-2-data-pipeline-current) section.**

---

## Table of Contents

1. [Phase 2 Data Pipeline (Current)](#phase-2-data-pipeline-current)
2. [Phase 1 Pipeline (Legacy - For Reference)](#phase-1-pipeline-legacy)
3. [Model Building Flow](#model-building-flow)
4. [Key Takeaways](#key-takeaways)
5. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Phase 2 Data Pipeline (Current)

### Architecture Overview

The Phase 2 pipeline implements a **dual-path system**:

```
┌────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 DATA SOURCES                             │
├────────────────────────────────────────────────────────────────────┤
│  Primary (Submission):                                             │
│    data/TechArena2025_Phase2_data.xlsx                            │
│    └─ All 4 markets in wide format                                │
│                                                                     │
│  Preprocessed (Validation Fast Path):                             │
│    data/parquet/preprocessed/{country}.parquet                    │
│    └─ Country-specific, all markets combined                      │
└────────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
         ┌───────▼──────┐         ┌───────▼──────┐
         │ SUBMISSION   │         │  VALIDATION  │
         │    PATH      │         │  FAST PATH   │
         └──────┬───────┘         └───────┬──────┘
                │                         │
                │ load_and_                │ load_preprocessed_
                │ preprocess_data()        │ country_data()
                │ (~6 seconds)             │ (~0.1 seconds)
                │                          │
                └──────────┬───────────────┘
                           │
                  ┌────────▼────────┐
                  │ COUNTRY DATA    │
                  │ (Ready for      │
                  │  optimization)  │
                  └─────────────────┘
```

### Path 1: Submission Path (Excel → Optimizer)

**Purpose**: Matches Huawei submission requirements

**Code Example**:
```python
from py_script.core.optimizer import BESSOptimizerModelIII

optimizer = BESSOptimizerModelIII()

# Load from Excel
full_data = optimizer.load_and_preprocess_data('data/TechArena2025_Phase2_data.xlsx')

# Extract country-specific data
country_data = optimizer.extract_country_data(full_data, 'DE_LU')
```

**Steps**:
1. **Load Excel** using `load_phase2_market_tables()`:
   - Parses 4 sheets: Day-ahead, FCR, aFRR capacity, aFRR energy
   - Returns wide-format DataFrames

2. **Convert to MultiIndex**:
   - Day-ahead: `(country, 'day_ahead', '')`
   - FCR: `(country, 'fcr', '')`
   - aFRR capacity: `(country, 'afrr', 'positive'/'negative')`
   - aFRR energy: `(country, 'afrr_energy', 'positive'/'negative')`

3. **Extract country data** using `extract_country_data()`:
   - Handles DE/DE_LU naming (DA uses DE_LU, others use DE)
   - Forward-fills 4-hour blocks to 15-min intervals
   - Applies 0→NaN preprocessing for aFRR energy
   - Adds aFRR activation weights from config

**Performance**: ~6 seconds for full-year data

### Path 2: Validation Fast Path (Preprocessed → Optimizer)

**Purpose**: Rapid validation and testing (10-100x faster)

**Code Example**:
```python
from py_script.data.load_process_market_data import load_preprocessed_country_data

# Load preprocessed country data directly
country_data = load_preprocessed_country_data('DE_LU')

# Use with optimizer
optimizer = BESSOptimizerModelIII()
model = optimizer.build_optimization_model(country_data, c_rate=0.5)
```

**Steps**:
1. **Direct load** from preprocessed parquet
2. **No processing needed** - data is ready for optimization

**Performance**: ~0.1 seconds for full-year data

**Preprocessing Script**: `py_script/data/generate_preprocessed_country_data.py`

### Data Processing Details

#### Critical Timestamp Handling

**Issue**: Excel timestamps have fractional seconds (e.g., `2024-01-01 08:00:00.001`)
**Solution**: Round all timestamps to nearest second before reindexing

```python
# In _extract_country_from_wide_tables()
timestamps = pd.to_datetime(market_tables['day_ahead']['timestamp']).dt.round('s')
fcr_df['timestamp'] = pd.to_datetime(fcr_df['timestamp']).dt.round('s')
```

This ensures proper alignment when forward-filling 4-hour block prices to 15-min intervals.

#### Germany Country Code Mapping

```python
if country == 'DE_LU':
    day_ahead_country = 'DE_LU'  # Coupled German-Luxembourg market
    as_country = 'DE'            # Ancillary services (German TSO)
else:
    day_ahead_country = country
    as_country = country
```

#### aFRR Energy Preprocessing

**Critical**: Convert price=0 to NaN (0 means "not activated", not "free energy")

```python
country_df['price_afrr_energy_pos'] = country_df['price_afrr_energy_pos'].replace(0, np.nan)
country_df['price_afrr_energy_neg'] = country_df['price_afrr_energy_neg'].replace(0, np.nan)
```

---

## Phase 1 Pipeline (Legacy)

---

## Data Processing Pipeline

### Current Data Flow (Hybrid Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (Mixed)                         │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1 (Legacy):                                               │
│   - TechArena2025_data_tidy.jsonl (DA, FCR, aFRR capacity)     │
│                                                                  │
│ Phase 2 (New):                                                  │
│   - data/parquet/afrr_energy.parquet (aFRR energy prices)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 1: load_and_preprocess_data()                      │
│         (Lines 260-393 in optimizer.py)                         │
├─────────────────────────────────────────────────────────────────┤
│ Input: JSONL file path + optional aFRR energy parquet          │
│                                                                  │
│ Process:                                                         │
│ 1. Load JSONL → Parse line-by-line → Create DataFrame          │
│ 2. Round timestamps to 15-min intervals (avoid misalignment)    │
│ 3. Pivot each market into MultiIndex columns:                   │
│    - day_ahead: (country, 'day_ahead', '')                      │
│    - fcr: (country, 'fcr', '')                                  │
│    - afrr: (country, 'afrr', 'positive'/'negative')            │
│                                                                  │
│ 4. Load aFRR energy from parquet:                               │
│    - Default path: data/parquet/afrr_energy.parquet             │
│    - Reshape columns: DE_Pos → (DE, 'afrr_energy', 'positive') │
│    - Create MultiIndex format                                   │
│                                                                  │
│ 5. Concatenate all into single DataFrame with MultiIndex cols   │
│ 6. Create complete 15-min timeline + forward fill               │
│                                                                  │
│ Output: Wide-format DataFrame with MultiIndex columns           │
│         Index: timestamp (15-min freq)                          │
│         Columns: (country, market, direction)                   │
│         Example: (HU, 'day_ahead', ''), (HU, 'fcr', ''),       │
│                  (HU, 'afrr_energy', 'positive')                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2: extract_country_data()                          │
│         (Lines 1230-1329 in optimizer.py)                       │
├─────────────────────────────────────────────────────────────────┤
│ Input: Wide MultiIndex DataFrame + country code                │
│                                                                  │
│ Process:                                                         │
│ 1. Handle DE_LU special case:                                   │
│    - Day-ahead: use 'DE_LU' (coupled market)                    │
│    - Ancillary services: use 'DE' (German TSO)                  │
│                                                                  │
│ 2. Extract country columns → Flat DataFrame:                    │
│    ┌──────────────────────────────────────────────┐            │
│    │ price_day_ahead                              │            │
│    │ price_fcr                                    │            │
│    │ price_afrr_pos  (capacity)                   │            │
│    │ price_afrr_neg  (capacity)                   │            │
│    │ price_afrr_energy_pos  ← PHASE 2             │            │
│    │ price_afrr_energy_neg  ← PHASE 2             │            │
│    └──────────────────────────────────────────────┘            │
│                                                                  │
│ 3. ⚠️ CRITICAL PREPROCESSING:                                   │
│    Convert aFRR energy price 0 → NaN                            │
│    Reason: Price=0 means "market NOT activated"                 │
│           NOT "free energy"!                                    │
│    Prevents false arbitrage opportunities                       │
│                                                                  │
│ 4. Add activation weights (w_afrr_pos, w_afrr_neg):            │
│    - If EV weighting enabled: load from config                  │
│    - If disabled: set to 1.0 (deterministic assumption)         │
│                                                                  │
│ 5. Add time-based identifiers:                                  │
│    - hour, day_of_year, month, year                            │
│    - block_id (4-hour blocks, 6 per day)                        │
│    - block_of_day (0-5)                                         │
│    - day_id                                                      │
│                                                                  │
│ Output: Flat country-specific DataFrame ready for optimization  │
│         - Simple column names (no MultiIndex)                   │
│         - Integer index (0, 1, 2, ...)                          │
│         - All market data + time identifiers                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 3: build_optimization_model()                      │
│         Uses the flat country DataFrame                         │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Pipeline Is Confusing

1. **Hybrid Data Sources**:
   - Phase 1 JSONL (DA, FCR, aFRR capacity) - **long format**
   - Phase 2 Parquet (aFRR energy) - **wide format**
   - Mixed in the same pipeline!

2. **Multiple Format Conversions**:
   - JSONL → Long format
   - Pivot → Wide format with MultiIndex
   - Extract → Flat format with simple columns

3. **Hidden Default Paths**:
   - Line 338: `afrr_energy_file = base_dir / 'parquet' / 'afrr_energy.parquet'`
   - Assumes JSONL is in `data/` and parquet is in `data/parquet/`

4. **Scattered Wide-Format Files**:
   - `data/parquet/` has: `day_ahead.parquet`, `fcr.parquet`, `afrr_capacity.parquet`, `afrr_energy.parquet`
   - But optimizer doesn't use them directly!
   - Only uses JSONL + aFRR energy parquet

### The Real Problem

You have **two separate data processing pipelines**:

1. **Data Extraction** (`py_script/data/load_process_market_data.py`):
   - Converts Excel → Parquet/JSON wide format
   - Stored in `data/parquet/` and `data/json/`
   - **Not used by optimizer.py!**

2. **Optimizer Input** (`optimizer.py`):
   - Expects JSONL (long format) + aFRR energy parquet
   - Converts to MultiIndex → Flat format

**They don't connect!** The wide-format parquets in `data/parquet/` are not directly usable by the optimizer without preprocessing into JSONL format first.

---

## Current Data Structure

### Actual Files in Project

```
data/
├── parquet/                    # Wide-format market data (Phase 2)
│   ├── day_ahead.parquet       # All countries, wide format
│   ├── fcr.parquet             # All countries, wide format
│   ├── afrr_capacity.parquet   # All countries, Pos/Neg columns
│   └── afrr_energy.parquet     # All countries, Pos/Neg columns
│
├── json/                       # Wide-format JSON (alternative)
│   ├── day_ahead_wide.json
│   ├── fcr_wide.json
│   ├── afrr_capacity_wide.json
│   └── afrr_energy_wide.json
│
├── p2_config/                  # Configuration files
│   ├── aging_config.json
│   ├── solver_config.json
│   ├── afrr_ev_weights_config.json
│   ├── investment.json
│   └── mpc_config.json
│
└── metadata.json               # Data processing metadata
```

### What the Optimizer Actually Needs

**Option A (Current):**
```
data/TechArena2025_data_tidy.jsonl  ← Phase 1 data (DA, FCR, aFRR capacity)
data/parquet/afrr_energy.parquet    ← Phase 2 data (aFRR energy)
```

**Option B (Recommended):**
```
data/parquet/hu_market_data.parquet    ← Preprocessed for HU
data/parquet/de_lu_market_data.parquet ← Preprocessed for DE_LU
data/parquet/at_market_data.parquet    ← Preprocessed for AT
... (one file per country)
```

---

## Complete Model Building Flow

### Phase 0: Initialization

```
┌══════════════════════════════════════════════════════════════════════┐
║                    INITIALIZATION PHASE                               ║
║              optimizer = BESSOptimizerModelI()                       ║
╠══════════════════════════════════════════════════════════════════════╣
║ __init__() method (Lines 101-146)                                   ║
║                                                                       ║
║ 1. Hardcoded Battery Parameters:                                     ║
║    - capacity_kwh: 4472                                              ║
║    - efficiency: 0.95 (round-trip)                                   ║
║    - soc_min: 0, soc_max: 1 (0-100% allowed)                        ║
║    - initial_soc: 0.5 (start at 50%)                                 ║
║                                                                       ║
║ 2. Hardcoded Market Parameters:                                      ║
║    - min_bid_da: 0.1 MW                                              ║
║    - min_bid_fcr: 1.0 MW                                             ║
║    - min_bid_afrr: 1.0 MW                                            ║
║    - min_bid_afrr_e: 0.1 MW (aFRR energy)                           ║
║    - time_step_hours: 0.25 (15 minutes)                             ║
║    - block_duration_hours: 4.0 (AS markets)                         ║
║    - reserve_duration_hours: 0.25                                    ║
║    - solver_time_limit: 600 seconds                                  ║
║                                                                       ║
║ 3. Configuration Scenarios:                                          ║
║    - countries: ['DE', 'DE_LU', 'AT', 'CH', 'HU', 'CZ']             ║
║    - c_rates: [0.25, 0.33, 0.5]                                     ║
║    - daily_cycles: [1.0, 1.5, 2.0]                                  ║
║                                                                       ║
║ 4. EV Weighting Flag:                                                ║
║    - use_afrr_ev_weighting: bool (default False)                    ║
║    - If True: loads afrr_ev_weights_config.json later               ║
║                                                                       ║
║ OUTPUT: Initialized optimizer object with default parameters         ║
└══════════════════════════════════════════════════════════════════════┘
```

### Phase 1: Model Building

```
┌══════════════════════════════════════════════════════════════════════┐
║                      MODEL BUILDING PHASE                             ║
║      model = optimizer.build_optimization_model(country_data,        ║
║                        c_rate=0.5, daily_cycle_limit=1.0)            ║
╠══════════════════════════════════════════════════════════════════════╣
║ build_optimization_model() method (Lines 395-1227)                  ║
║                                                                       ║
║ INPUT: country_data DataFrame + c_rate + daily_cycle_limit           ║
║                                                                       ║
║ STEP 1: Update Configuration (Lines 416-432)                         ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ - Update battery_params['daily_cycle_limit']                  │   ║
║ │ - Calculate P_max_config = c_rate × capacity_kwh              │   ║
║ │   Example: 0.5 × 4472 = 2236 kW                               │   ║
║ │                                                                 │   ║
║ │ - Extract time range: T_data = [0, 1, 2, ..., n-1]            │   ║
║ │ - Extract unique blocks and days from country_data             │   ║
║ │ - Validate input data                                          │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 2: Pre-compute Mappings (Lines 434-452)                         ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ For EFFICIENCY (avoid nested loops in constraints):            │   ║
║ │                                                                 │   ║
║ │ - block_to_times: {block_id → [t1, t2, ...]}                  │   ║
║ │   Example: {0: [0,1,2...15], 1: [16,17...31], ...}            │   ║
║ │                                                                 │   ║
║ │ - time_to_block: {t → block_id}                               │   ║
║ │   Example: {0:0, 1:0, ..., 16:1, 17:1, ...}                   │   ║
║ │                                                                 │   ║
║ │ - day_to_times: {day_id → [t1, t2, ...]}                      │   ║
║ │   Example: {1: [0...95], 2: [96...191], ...}                  │   ║
║ │                                                                 │   ║
║ │ Stored in: self._block_to_times, self._day_to_times           │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 3: Pre-compute AS Prices by Block (Lines 454-464)               ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ Extract ONE price per block (constant within 4h):              │   ║
║ │                                                                 │   ║
║ │ - fcr_prices_by_block: {block_id → price}                     │   ║
║ │ - afrr_pos_prices_by_block: {block_id → price}                │   ║
║ │ - afrr_neg_prices_by_block: {block_id → price}                │   ║
║ │                                                                 │   ║
║ │ Uses first timestep in block as representative                 │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 4: Create Pyomo Model (Lines 466-468)                           ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ model = pyo.ConcreteModel(name="Improved_BESS_Optimization")  │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 5: Define Sets (Lines 470-472)                                  ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ model.T = Set of time intervals [0, 1, 2, ..., n-1]           │   ║
║ │ model.B = Set of 4h blocks [0, 1, 2, ..., m-1]                │   ║
║ │ model.D = Set of days [1, 2, 3, ..., d]                       │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 6: Define Parameters (Lines 474-544)                            ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ Battery Configuration:                                          │   ║
║ │   - E_nom (capacity)                                           │   ║
║ │   - P_max_config (max power)                                   │   ║
║ │   - eta_ch, eta_dis (efficiencies)                            │   ║
║ │   - SOC_min, SOC_max                                           │   ║
║ │   - E_soc_init                                                 │   ║
║ │   - N_cycles (if daily limit set)                              │   ║
║ │                                                                 │   ║
║ │ Time Parameters:                                                │   ║
║ │   - dt (0.25h = 15min)                                         │   ║
║ │   - tau (reserve duration)                                     │   ║
║ │   - db (block duration = 4h)                                   │   ║
║ │                                                                 │   ║
║ │ Min Bids:                                                       │   ║
║ │   - min_bid_da, min_bid_fcr, min_bid_afrr, min_bid_afrr_e     │   ║
║ │                                                                 │   ║
║ │ Mappings:                                                       │   ║
║ │   - block_map[t] = block_id for time t                        │   ║
║ │                                                                 │   ║
║ │ Prices (extracted from country_data):                           │   ║
║ │   - P_DA[t]: DA price per timestep                            │   ║
║ │   - P_aFRR_E_pos[t], P_aFRR_E_neg[t]: aFRR energy per time    │   ║
║ │   - w_aFRR_pos[t], w_aFRR_neg[t]: activation weights          │   ║
║ │   - P_FCR[b]: FCR price per block                             │   ║
║ │   - P_aFRR_pos[b], P_aFRR_neg[b]: aFRR capacity per block    │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 7: Define Decision Variables (Lines 548-610)                    ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ Continuous Variables:                                           │   ║
║ │   - p_ch[t], p_dis[t]: DA charge/discharge (kW)               │   ║
║ │   - e_soc[t]: State of charge (kWh)                           │   ║
║ │   - p_afrr_pos_e[t], p_afrr_neg_e[t]: aFRR energy (kW)       │   ║
║ │   - p_total_ch[t], p_total_dis[t]: Total power (kW)           │   ║
║ │   - c_fcr[b], c_afrr_pos[b], c_afrr_neg[b]: Capacity (MW)    │   ║
║ │                                                                 │   ║
║ │ Binary Variables (MOSTLY DISABLED for performance):            │   ║
║ │   - Only block-level binaries kept for min bids                │   ║
║ │   - Time-indexed binaries commented out                        │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 8: Define Constraints (Lines 611-1100)                          ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ - SOC dynamics (charge/discharge balance)                      │   ║
║ │ - Power limits (based on c_rate)                               │   ║
║ │ - Daily cycle limits                                            │   ║
║ │ - Reserve energy constraints                                    │   ║
║ │ - Market exclusivity (FCR vs aFRR)                             │   ║
║ │ - Minimum bid sizes                                             │   ║
║ │ - Total power definitions                                       │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ STEP 9: Define Objective Function (Lines 1101-1227)                  ║
║ ┌───────────────────────────────────────────────────────────────┐   ║
║ │ Maximize total profit:                                          │   ║
║ │                                                                 │   ║
║ │ Revenue from:                                                   │   ║
║ │   - Day-ahead market (energy arbitrage)                        │   ║
║ │   - FCR capacity market                                         │   ║
║ │   - aFRR capacity markets (pos + neg)                          │   ║
║ │   - aFRR energy markets (pos + neg) ← Phase 2                 │   ║
║ │     • Weighted by activation probability if EV enabled         │   ║
║ │                                                                 │   ║
║ │ Uses pre-computed block_to_times for efficient summation       │   ║
║ └───────────────────────────────────────────────────────────────┘   ║
║                                                                       ║
║ OUTPUT: Complete Pyomo ConcreteModel ready to solve                  ║
└══════════════════════════════════════════════════════════════════════┘
```

---

## Key Takeaways

### 1. Parameters Are Hardcoded

Battery specifications and market rules are **fixed in the code** in `__init__()` method:
- Battery: 4472 kWh, 95% efficiency, 0-100% SOC
- Market: Min bids, time steps, block durations
- **NOT loaded from config files**

### 2. Data Flows Through 3 Transformations

```
JSONL + Parquet → MultiIndex DataFrame (wide, all countries)
                        ↓
                Flat DataFrame (country-specific)
                        ↓
                Pyomo Parameters (indexed by time/block)
```

### 3. Model Building Is Data-Driven

The `country_data` DataFrame directly populates all price parameters in the Pyomo model:
- Each row becomes a timestep
- Prices are extracted and indexed appropriately (by time or block)

### 4. No External Config for Battery/Market Params

Unlike degradation configs, the core battery and market parameters are **hardcoded** in `optimizer.py`.

### 5. Critical Preprocessing Step

The **`0 → NaN` conversion** for aFRR energy prices happens in `extract_country_data()`:

```python
# Line 1267-1268
country_df['price_afrr_energy_pos'] = country_df['price_afrr_energy_pos'].replace(0, np.nan)
country_df['price_afrr_energy_neg'] = country_df['price_afrr_energy_neg'].replace(0, np.nan)
```

**Why?** Price = 0 means "market NOT activated", not "free energy". This prevents false arbitrage opportunities.

### 6. Performance Optimizations

The optimizer uses several techniques to improve solve time:

1. **Pre-computed mappings** (`block_to_times`, `time_to_block`, `day_to_times`)
   - Avoids O(B×T) loops in constraints
   - Enables O(1) lookup

2. **Block-indexed AS prices**
   - Capacity prices constant within 4h blocks
   - Stored once per block, not per timestep

3. **Disabled time-indexed binaries**
   - Most binary variables commented out
   - Only block-level binaries for min bids
   - Drastically reduces solve time (3-day: 27s vs 7-day: timeout)

---

## Common Issues and Solutions

### Issue 1: "Market data file not found"

**Symptom:** Notebook or script can't find country-specific parquet files

**Cause:** The optimizer expects either:
- JSONL file + aFRR energy parquet (legacy), OR
- Country-specific preprocessed parquet

**Solution:**
- Use JSONL loading path (slower but works)
- OR create country-specific parquet files from wide format

### Issue 2: "Degradation config file not found"

**Symptom:** Model II/III fails with config error

**Cause:** Models II/III look for `data/p2_config/aging_config.json`

**Solution:** Ensure file exists at correct path (fixed in optimizer.py line 1479)

### Issue 3: Solver times out on long horizons

**Symptom:** 7-day optimization times out after 10+ minutes

**Cause:** Too many binary variables

**Solution:** Most time-indexed binaries are already disabled. Consider:
- Shorter time horizons (3-4 days)
- Better solver (CPLEX/Gurobi vs CBC/HiGHS)
- Tighter solver tolerances

### Issue 4: aFRR energy creates false arbitrage

**Symptom:** Unrealistic profits from aFRR energy market

**Cause:** Price = 0 treated as "free energy" instead of "not activated"

**Solution:** Already handled by `0 → NaN` conversion in `extract_country_data()` (line 1267-1268)

### Issue 5: Wide-format parquets not used by optimizer

**Symptom:** Files in `data/parquet/` exist but optimizer still needs JSONL

**Cause:** Optimizer has legacy JSONL loading pathway

**Solution:** Create preprocessing script to generate country-specific files from wide format

---

## Recommended Future Improvements

1. **Unify Data Pipeline**
   - Create preprocessing script: wide parquets → country-specific parquets
   - Eliminate JSONL dependency
   - Single consistent data format

2. **Externalize Battery/Market Parameters**
   - Move hardcoded params to config file
   - Make battery specs configurable per scenario
   - Easier testing and validation

3. **Simplify Data Loading**
   - Direct load from country-specific files
   - Skip MultiIndex intermediate format
   - Fewer format conversions

4. **Better Error Messages**
   - Clearer guidance when data files missing
   - Suggest valid alternative paths
   - Validate data format early

---

## References

- **Optimizer Implementation**: `py_script/core/optimizer.py`
- **Data Loading Utilities**: `py_script/data/load_process_market_data.py`
- **Model Formulation**: `doc/p2_model/p2_bi_model_ggdp.tex`
- **Configuration Files**: `data/p2_config/`
- **Test Notebook**: `notebook/p2b_optimizer.ipynb`

---

**Document Version:** 1.0
**Last Updated:** 2025-11-13
**Maintainer:** Project Team
