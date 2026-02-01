"""
GridKey Optimizer Service API
==============================

FastAPI endpoints for BESS optimization with renewable integration.

SOLVER CONFIGURATION:
    - Production uses HiGHS (open-source, no license required)
    - For best performance, use 6-hour rolling horizon
    - Longer horizons (24-48h) require commercial solvers (CPLEX/Gurobi)

ROLLING HORIZON STRATEGY:
    The API uses a 6-hour optimization window by default.
    For continuous operation, call the API every 6 hours with updated forecasts.
    This MPC-style approach provides:
    - Faster solve times (HiGHS handles 6h efficiently)
    - Better responsiveness to changing conditions
    - No license requirements for production deployment
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import logging

from src.service.optimizer_service import OptimizerService
from src.service.models import OptimizationResult

app = FastAPI(
    title="GridKey Optimizer Service",
    description="""
    BESS Optimizer with Renewable Integration.

    **Production Solver**: HiGHS (open-source, no license required)

    **Recommended Usage**: 6-hour rolling horizon. Call the API every 6 hours
    with updated price forecasts for continuous optimization.

    **Why 6-hour horizon?**
    - HiGHS solves 6-hour problems quickly (<30 seconds typical)
    - 24-48 hour horizons require commercial solvers (CPLEX/Gurobi licenses)
    - Rolling 6-hour windows enable real-time responsiveness
    """,
    version="1.0.0"
)

logger = logging.getLogger(__name__)
service = OptimizerService()


# Maximum allowed horizon for HiGHS (prevent excessively long solve times)
MAX_HORIZON_HOURS_HIGHS = 12
MAX_HORIZON_HOURS_COMMERCIAL = 48


class MarketPrices(BaseModel):
    """Market price data structure matching DataAdapter expectations."""
    day_ahead: List[float] = Field(..., description="Day-ahead prices (EUR/MWh), 15-min resolution")
    afrr_energy_pos: List[float] = Field(..., description="aFRR+ energy prices (EUR/MWh), 15-min resolution")
    afrr_energy_neg: List[float] = Field(..., description="aFRR- energy prices (EUR/MWh), 15-min resolution")
    fcr: List[float] = Field(..., description="FCR capacity prices (EUR/MW), 4-hour blocks")
    afrr_capacity_pos: List[float] = Field(..., description="aFRR+ capacity prices (EUR/MW), 4-hour blocks")
    afrr_capacity_neg: List[float] = Field(..., description="aFRR- capacity prices (EUR/MW), 4-hour blocks")


class OptimizeRequest(BaseModel):
    """API request for optimization."""
    location: str = "Munich"
    country: str = "DE_LU"
    model_type: str = Field(default="III", description="Model type: I, II, III, or III-renew")
    c_rate: float = Field(default=0.5, description="Battery C-rate (0.25, 0.33, 0.5)")
    alpha: float = Field(default=1.0, description="Degradation cost weight")
    daily_cycle_limit: float = Field(default=1.0, description="Maximum daily cycles")
    time_horizon_hours: int = Field(
        default=6,
        description="Optimization horizon in hours. Default=6 recommended for HiGHS solver."
    )

    @field_validator('time_horizon_hours')
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        """Validate time horizon is reasonable for the solver."""
        import os
        solver = os.environ.get('GRIDKEY_SOLVER', '').lower()

        max_hours = MAX_HORIZON_HOURS_COMMERCIAL if solver in ['cplex', 'gurobi'] else MAX_HORIZON_HOURS_HIGHS

        if v > max_hours:
            raise ValueError(
                f"time_horizon_hours={v} exceeds maximum of {max_hours} for {solver or 'auto-detected'} solver. "
                f"HiGHS max: {MAX_HORIZON_HOURS_HIGHS}h, Commercial max: {MAX_HORIZON_HOURS_COMMERCIAL}h. "
                f"Use rolling 6-hour windows for continuous operation."
            )
        if v < 1:
            raise ValueError("time_horizon_hours must be at least 1")
        return v

    # Market prices (required)
    market_prices: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Market price data. Keys: day_ahead, afrr_energy_pos, afrr_energy_neg, fcr, afrr_capacity_pos, afrr_capacity_neg"
    )

    # Renewable forecast (optional) - 15-min resolution in kW
    renewable_generation: Optional[List[float]] = Field(
        default=None,
        description="Renewable generation forecast (kW), 15-min resolution"
    )


class OptimizeResponse(BaseModel):
    """API response wrapper."""
    status: str
    data: Dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import os
    solver = os.environ.get('GRIDKEY_SOLVER', 'auto-detected')
    max_horizon = MAX_HORIZON_HOURS_COMMERCIAL if solver in ['cplex', 'gurobi'] else MAX_HORIZON_HOURS_HIGHS

    return {
        "status": "healthy",
        "version": "1.0.0",
        "solver": solver,
        "max_horizon_hours": max_horizon,
        "recommended_horizon_hours": 6,
        "note": "HiGHS (open-source) - use 6-hour rolling horizon for best performance"
    }


@app.post("/api/v1/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Run BESS optimization with optional renewable integration.

    Required: market_prices with all 6 price arrays
    Optional: renewable_generation for Model III-renew
    """
    # Validate required input
    if not request.market_prices:
        raise HTTPException(
            status_code=400,
            detail="market_prices is required. Must include: day_ahead, afrr_energy_pos, afrr_energy_neg, fcr, afrr_capacity_pos, afrr_capacity_neg"
        )

    # Validate required keys
    required_keys = ['day_ahead', 'afrr_energy_pos', 'afrr_energy_neg', 'fcr', 'afrr_capacity_pos', 'afrr_capacity_neg']
    missing_keys = [k for k in required_keys if k not in request.market_prices]
    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required market_prices keys: {missing_keys}"
        )

    try:
        # Prepare generation forecast with correct key for DataAdapter
        generation_forecast = None
        if request.renewable_generation:
            generation_forecast = {"generation_kw": request.renewable_generation}

        result = service.optimize(
            market_prices=request.market_prices,
            generation_forecast=generation_forecast,
            model_type=request.model_type,
            c_rate=request.c_rate,
            alpha=request.alpha,
            daily_cycle_limit=request.daily_cycle_limit,
            time_horizon_hours=request.time_horizon_hours,
        )

        # Use model_dump() for Pydantic v2 (not deprecated .dict())
        return OptimizeResponse(status="success", data=result.model_dump())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
