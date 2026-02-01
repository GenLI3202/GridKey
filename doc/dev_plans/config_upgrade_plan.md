# Configuration System Upgrade Plan

## 1. Problem Analysis
The current `config/Config.yml` suffers from several issues:
- **Monolithic Structure**: 300+ lines mixing solver settings, physical model parameters, financial assumptions, and test scenarios.
- **Cognitive Load**: Hard to find specific parameters amidst extensive documentation and "notes".
- **Coupling**: Changing a test scenario requires editing the same file used for production solver settings.
- **Redundancy**: Country-specific values are repeated or nested deeply.

## 2. Proposed Architecture
We will transition to a **modular, hierarchical configuration system**.

### 2.1 File Split Strategy
Break `Config.yml` into domain-specific files under `config/`:

| New File | Content |
| :--- | :--- |
| `solver.yaml` | Solver timeouts, tolerances, specific flags (Gurobi/CPLEX) |
| `bess_physical.yaml` | Aging parameters (Cyclic/Calendar), efficiency, degradation curves |
| `market_assumptions.yaml` | aFRR activation weights, price thresholds, forecast/ev_weighting settings |
| `financial.yaml` | ROI parameters, investment costs, WACC, inflation (per country) |
| `mpc_settings.yaml` | Horizon length, execution steps, initial SOC |
| `test_scenarios.yaml` | Test-specific overrides (e.g., "AT_0.5C_Winter") |

### 2.2 Schema Validation (Pydantic)
Instead of raw dictionary access (`config['solver']['gurobi']`), we will implement **Pydantic Settings** in `src/core/config.py`.

```python
from pydantic import BaseModel, Field

class SolverConfig(BaseModel):
    time_limit_sec: int = Field(1200, ge=0)
    gap_tolerance: float = Field(0.01, le=1.0)
    # ...

class BESSConfig(BaseModel):
    capacity_kwh: float = 4472.0
    efficiency: float = 0.95
    aging: AgingConfig
```

**Benefits:**
- **Type Safety**: IDE autocompletion and static checking.
- **Validation**: Immediate error if a required field is missing or invalid (e.g., negative timeout).
- **Defaults**: Clear default values defined in code, overrides in YAML.

### 2.3 Configuration Loading Logic
We will create a `ConfigLoader` that:
1. Loads default values from Pydantic models.
2. Overrides with values from split YAML files.
3. Allows Environment Variable overrides (e.g., `GRIDKEY_SOLVER_TIMEOUT=600`).

## 3. Implementation Steps

### Step 1: Create Validation Models
- Define Pydantic models in `src/core/config_definitions.py` mirroring the proposed file structure.

### Step 2: Split YAML Files
- Create `config/v2/` directory.
- Extract sections from `Config.yml` into individual files.
- Clean up comments (move purely educational notes to documentation, keep functional comments).

### Step 3: Implement Loader
- Update `src/utils/config_loader.py` to support loading the new multi-file structure.
- Ensure backward compatibility (optional) or provide a migration script.

### Step 4: Refactor Codebase
- Search and replace usages:
  - `config['solver_config']['solver_time_limit_sec']` -> `cfg.solver.time_limit`
  - `config['aging_config']['cyclic_aging']` -> `cfg.physical.aging.cyclic`

## 4. Migration Example

**Old (`Config.yml`):**
```yaml
solver_config:
  solver_time_limit_sec: 1200
  solver_options:
    gurobi:
      MIPGap: 0.01
```

**New (`config/v2/solver.yaml`):**
```yaml
time_limit: 1200
gurobi:
  gap: 0.01
```

**Usage:**
```python
# Old
timeout = raw_config['solver_config']['solver_time_limit_sec']

# New
timeout = settings.solver.time_limit
```
