"""
Pydantic Data Models for Module D: Optimizer Service
=====================================================

Defines type-safe data models for the API service layer,
based on GridKey WatsonX Blueprint Section 6.5.

Models:
    ModelType           — Optimization model variant enum
    OptimizationInput   — Standardised optimizer input
    ScheduleEntry       — Single timestep schedule item
    RenewableUtilization — Renewable energy utilization breakdown
    OptimizationResult  — Standardised optimizer output
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    """Optimization model variants."""
    MODEL_I = "I"
    MODEL_II = "II"
    MODEL_III = "III"
    MODEL_III_RENEW = "III-renew"  # Model III + Renewable Integration


class OptimizationInput(BaseModel):
    """Standardised optimizer input (Blueprint Section 6.5).

    All price lists use 15-minute resolution unless noted.
    For a 48-hour horizon, 15-min lists have 192 entries,
    and 4-hour block lists have 12 entries.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_horizon_hours": 48,
                "da_prices": [50.0] * 192,
                "afrr_energy_pos": [40.0] * 192,
                "afrr_energy_neg": [30.0] * 192,
                "fcr_prices": [100.0] * 12,
                "afrr_capacity_pos": [5.0] * 12,
                "afrr_capacity_neg": [10.0] * 12,
                "model_type": "III",
            }
        }
    )

    # Time parameters
    time_horizon_hours: int = Field(default=48, description="Optimization horizon")

    # Market prices (15-min resolution)
    da_prices: List[float] = Field(
        ...,
        description="Day-ahead prices (EUR/MWh), 15-min resolution. 48h = 192 values.",
        examples=[[50.0] * 192]
    )
    afrr_energy_pos: List[float] = Field(
        ...,
        description="aFRR+ energy prices (EUR/MWh), 15-min resolution. 48h = 192 values.",
        examples=[[40.0] * 192]
    )
    afrr_energy_neg: List[float] = Field(
        ...,
        description="aFRR- energy prices (EUR/MWh), 15-min resolution. 48h = 192 values.",
        examples=[[30.0] * 192]
    )

    # Market prices (4-hour blocks)
    fcr_prices: List[float] = Field(
        ...,
        description="FCR capacity prices (EUR/MW), 4-hour blocks. 48h = 12 blocks.",
        examples=[[100.0] * 12]
    )
    afrr_capacity_pos: List[float] = Field(
        ...,
        description="aFRR+ capacity prices (EUR/MW), 4-hour blocks. 48h = 12 blocks.",
        examples=[[5.0] * 12]
    )
    afrr_capacity_neg: List[float] = Field(
        ...,
        description="aFRR- capacity prices (EUR/MW), 4-hour blocks. 48h = 12 blocks.",
        examples=[[10.0] * 12]
    )

    # Renewable generation forecast (15-min resolution)
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

    @field_validator('c_rate')
    @classmethod
    def c_rate_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('c_rate must be positive')
        return v

    @field_validator('efficiency')
    @classmethod
    def efficiency_must_be_valid(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError('efficiency must be in (0, 1]')
        return v

    @field_validator('initial_soc')
    @classmethod
    def initial_soc_must_be_valid(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError('initial_soc must be in [0, 1]')
        return v


class ScheduleEntry(BaseModel):
    """Single timestep schedule item (Blueprint Section 6.5)."""
    timestamp: datetime
    action: str = Field(description="charge/discharge/idle")
    power_kw: float
    market: str = Field(description="da/fcr/afrr_cap/afrr_energy")

    # Renewable fields
    renewable_action: Optional[str] = Field(
        default=None,
        description="self_consume/export/curtail"
    )
    renewable_power_kw: Optional[float] = Field(default=None)

    soc_after: float = Field(description="SOC after this timestep (fraction)")

    @field_validator('soc_after')
    @classmethod
    def soc_after_must_be_valid(cls, v: float) -> float:
        # Allow small tolerance for floating point errors
        if not (-1e-10 <= v <= 1.0):
            raise ValueError('soc_after must be in [0, 1]')
        # Clamp to [0, 1] to handle near-zero/near-one floating point issues
        return max(0.0, min(1.0, v))


class RenewableUtilization(BaseModel):
    """Renewable energy utilization breakdown."""
    total_generation_kwh: float
    self_consumption_kwh: float
    export_kwh: float
    curtailment_kwh: float
    utilization_rate: float = Field(description="(self + export) / total")

    @field_validator('utilization_rate')
    @classmethod
    def utilization_rate_must_be_valid(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError('utilization_rate must be in [0, 1]')
        return v


class OptimizationResult(BaseModel):
    """Standardised optimizer output (Blueprint Section 6.5)."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "objective_value": 450.0,
                "net_profit": 420.0,
                "status": "optimal",
            }
        }
    )

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

    # Renewable utilization
    renewable_utilization: Optional[RenewableUtilization] = None

    # Solver metadata
    solve_time_seconds: float
    solver_name: str
    model_type: ModelType
    status: str = Field(description="optimal/feasible/infeasible/timeout")

    # For debugging
    num_variables: Optional[int] = None
    num_constraints: Optional[int] = None
