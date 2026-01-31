### Phase 1: Foundation (Parallel Tasks)

These components have **no dependencies** and can be implemented simultaneously.

---

#### 1.1 [NEW] models.py — Pydantic Data Models

Location: `src/service/models.py`

Pydantic-based data models for type safety and serialisation:

```python

from pydantic import BaseModel, Field

from typing import List, Optional, Dict

from datetime import datetime

from enum import Enum


classModelType(str, Enum):

    MODEL_I = "I"

    MODEL_II = "II"

    MODEL_III = "III"

    MODEL_III_RENEW = "III-renew"  # Model III + Renewable Integration


classOptimizationInput(BaseModel):

    """Standardised optimizer input (Section 6.5 of Blueprint)."""

  

    # Time parameters

    time_horizon_hours: int = Field(default=48, description="Optimization horizon")

  

    # Market prices (15-min resolution)

    da_prices: List[float] = Field(..., description="Day-ahead prices (EUR/MWh)")

    afrr_energy_pos: List[float] = Field(..., description="aFRR+ energy prices (EUR/MWh)")

    afrr_energy_neg: List[float] = Field(..., description="aFRR- energy prices (EUR/MWh)")

  

    # Market prices (4-hour blocks)

    fcr_prices: List[float] = Field(..., description="FCR capacity prices (EUR/MW)")

    afrr_capacity_pos: List[float] = Field(..., description="aFRR+ capacity prices (EUR/MW)")

    afrr_capacity_neg: List[float] = Field(..., description="aFRR- capacity prices (EUR/MW)")

  

    # NEW: Renewable generation forecast (15-min resolution)

    renewable_generation: Optional[List[float]] = Field(

        default=None, 

        description="Renewable generation forecast (kW), from Weather Service"

    )

  

    # Battery configuration

    battery_capacity_kwh: float = Field(default=4472)

    c_rate: float = Field(default=0.5)

    efficiency: float = Field(default=0.95)

    initial_soc: float = Field(default=0.5)

  

    # Optimization parameters

    model_type: ModelType = Field(default=ModelType.MODEL_III)

    alpha: float = Field(default=1.0, description="Degradation cost weight")



classScheduleEntry(BaseModel):

    """Single timestep schedule item (Section 6.5 of Blueprint)."""

  

    timestamp: datetime

    action: str = Field(description="charge/discharge/idle")

    power_kw: float

    market: str = Field(description="da/fcr/afrr_cap/afrr_energy")

  

    # NEW: Renewable fields

    renewable_action: Optional[str] = Field(

        default=None, 

        description="self_consume/export/curtail"

    )

    renewable_power_kw: Optional[float] = Field(default=None)

  

    soc_after: float = Field(description="SOC after this timestep (fraction)")



classRenewableUtilization(BaseModel):

    """Renewable energy utilization breakdown."""

  

    total_generation_kwh: float

    self_consumption_kwh: float

    export_kwh: float

    curtailment_kwh: float

    utilization_rate: float = Field(description="(self + export) / total")



classOptimizationResult(BaseModel):

    """Standardised optimizer output (Section 6.5 of Blueprint)."""

  

    # Core metrics

    objective_value: float = Field(description="Total objective value (EUR)")

    net_profit: float = Field(description="Net profit after degradation (EUR)")

  

    # Revenue breakdown

    revenue_breakdown: Dict[str, float] = Field(

        description="Revenue by market: da, fcr, afrr_cap, afrr_energy, renewable_export"

    )

  

    # Degradation costs

    degradation_cost: float

    cyclic_aging_cost: float

    calendar_aging_cost: float

  

    # Schedule

    schedule: List[ScheduleEntry]

    soc_trajectory: List[float]

  

    # NEW: Renewable utilization

    renewable_utilization: Optional[RenewableUtilization] = None

  

    # Solver metadata

    solve_time_seconds: float

    solver_name: str

    model_type: ModelType

    status: str = Field(description="optimal/feasible/infeasible/timeout")

  

    # For debugging

    num_variables: Optional[int] = None

    num_constraints: Optional[int] = None

```

---

#### 1.2 [NEW] Renewable Math Documentation (LaTeX)

Location: `doc/p2_model/p2_renewable_extension.tex`

Document the mathematical formulation to align with existing Model I-III documentation.

##### New Variables (Section 6.4 of Blueprint)

| Variable                   | Type       | Bounds                  | Description               |

| -------------------------- | ---------- | ----------------------- | ------------------------- |

| `p_renewable_self[t]`    | Continuous | `[0, P_renewable[t]]` | kW used to charge battery |

| `p_renewable_export[t]`  | Continuous | `[0, P_renewable[t]]` | kW sold to grid           |

| `p_renewable_curtail[t]` | Continuous | `[0, P_renewable[t]]` | kW curtailed              |

##### New Constraints

**(Cst-R1) Renewable Balance:**

$$
P^{\text{renewable}}_t = P^{\text{self}}_t + P^{\text{export}}_t + P^{\text{curtail}}_t \quad\forall t \in T
$$

**(Cst-R2) Self-consumption Integration (modifies existing total charge definition):**

$$
P^{\text{total,ch}}_t = P^{\text{ch}}_t + P^{\text{aFRR-,E}}_t + P^{\text{self}}_t \quad\forall t \in T
$$

**(Cst-R3) Export Revenue (added to objective):**

$$
R^{\text{export}} = \sum_{t \in T} P^{\text{export}}_t \cdot\pi^{\text{DA}}_t \cdot\Delta t / 1000
$$

##### Objective Function Update

$$
\max\quad R^{\text{DA}} + R^{\text{FCR}} + R^{\text{aFRR,cap}} + R^{\text{aFRR,E}} + R^{\text{export}} - \alpha\cdot (C^{\text{cyclic}} + C^{\text{calendar}})
$$

---

#### 1.3 [NEW] requirements-api.txt — API Dependencies

Location: `GridKey/requirements-api.txt`

```txt

# API & Web Service Dependencies

fastapi>=0.100.0

uvicorn[standard]>=0.22.0

pydantic>=2.0.0

python-multipart>=0.0.6


# Health checks and monitoring

httpx>=0.24.0


# Optional: Rate limiting

slowapi>=0.1.8

```

##### [MODIFY] requirements.txt

Add Pydantic for data validation:

```diff

+ # API Data Models

+ pydantic>=2.0.0

```

---
