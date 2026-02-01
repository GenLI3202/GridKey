# MPC Rolling Horizon Implementation Plan (12h Demo)

## Problem Statement

**Requirements (Updated):**

- API provides **12-hour** optimization results (demo scope)
- HiGHS solver cannot solve 12h directly (may be slow)
- Cloud API service **HiGHS only** (no CPLEX license)

**MPC Strategy:**

- Optimization window: 6h (HiGHS ~5 sec)
- Roll step: 4h (commit 4h each iteration)
- Total iterations: 12h / 4h = **3 iterations**

---

## Architecture Design

### MPC Rolling Timeline

```
Input: 12h price forecast (48 values @ 15-min)

Iteration 1: [0h - 6h optimization window] → Commit [0h - 4h] results
Iteration 2: [4h - 10h optimization window] → Commit [4h - 8h] results
Iteration 3: [8h - 12h optimization window] → Commit [8h - 12h] results

Output: 12h complete schedule (48 values @ 15-min)
```

### Performance Estimates

| Metric                   | Value                    |
| ------------------------ | ------------------------ |
| Iterations               | 3                        |
| Per-iteration solve time | ~5 seconds               |
| **Total time**     | **~15-20 seconds** |
| Memory                   | < 200MB                  |

---

## Implementation Plan

### Step 1: Create `src/service/mpc.py` (NEW)

```python
"""
MPC rolling horizon helper for OptimizerService.

Reuses patterns from src/mpc/mpc_simulator:
- _get_initial_segment_soc(): SOC segment distribution
- SOC state management between windows
- Binary variable fixing (LIFO constraints)

But adapted for:
- OptimizationInput slices (not DataFrame)
- Service layer integration
- 12h demo scope
"""

from typing import Dict, List, Any
import logging
from ..service.models import OptimizationInput
from ..service.adapter import DataAdapter

logger = logging.getLogger(__name__)


class MPCRollingHorizon:
    """
    MPC rolling horizon optimization (12h demo).

    Args:
        horizon_hours: Optimization window size (default 6h)
        execution_hours: Commit execution window (default 4h)
    """

    def __init__(
        self,
        optimizer,
        adapter: DataAdapter,
        horizon_hours: int = 6,
        execution_hours: int = 4,
    ):
        self.optimizer = optimizer
        self.adapter = adapter
        self.horizon_hours = horizon_hours
        self.execution_hours = execution_hours

        # Get parameters from optimizer
        self.time_step_hours = self.optimizer.market_params['time_step_hours']
        self.battery_params = self.optimizer.battery_params
        self.degradation_params = getattr(self.optimizer, 'degradation_params', {})

        # Segment parameters (Model II/III)
        self.num_segments = self.degradation_params.get('num_segments', 10)
        self.segment_capacity = self.degradation_params.get(
            'segment_capacity_kwh',
            self.battery_params['capacity_kwh'] / 10
        )

        # Calculate step counts
        self.horizon_steps = int(horizon_hours / self.time_step_hours)  # 24
        self.execution_steps = int(execution_hours / self.time_step_hours)  # 16

    def _get_initial_segment_soc(self, total_soc_kwh: float) -> Dict[int, float]:
        """Convert total SOC to segment-wise distribution (shallow to deep)."""
        segment_soc = {}
        remaining_soc = total_soc_kwh

        for j in range(1, self.num_segments + 1):
            energy = min(remaining_soc, self.segment_capacity)
            segment_soc[j] = energy
            remaining_soc -= energy

        return segment_soc

    def solve_12h(
        self,
        opt_input_12h: OptimizationInput,
        c_rate: float,
    ) -> Dict[str, Any]:
        """
        Execute 12h MPC rolling horizon optimization.

        Returns: Aggregated solution dict (12h schedule)
        """
        total_timesteps = 12 * 4  # 48
        execution_timesteps = self.execution_steps  # 16

        # Initialize state
        current_total_soc = (
            opt_input_12h.initial_soc * self.battery_params['capacity_kwh']
        )

        # Aggregated results storage
        all_solutions = {
            'p_ch': {}, 'p_dis': {},
            'p_afrr_pos_e': {}, 'p_afrr_neg_e': {},
            'c_fcr': {}, 'c_afrr_pos': {}, 'c_afrr_neg': {},
            'e_soc': {},
            'profit_da': 0.0, 'profit_afrr_energy': 0.0,
            'profit_as_capacity': 0.0,
            'cost_cyclic': 0.0, 'cost_calendar': 0.0,
            'solve_time': 0.0, 'solver': 'unknown', 'status': 'optimal',
        }

        # Renewable energy
        has_renewable = opt_input_12h.renewable_generation is not None
        if has_renewable:
            all_solutions.update({
                'p_renewable_self': {}, 'p_renewable_export': {}, 'p_renewable_curtail': {}
            })

        # MPC main loop: 3 iterations (0h, 4h, 8h)
        iteration = 0
        for t_start in range(0, total_timesteps, execution_timesteps):
            iteration += 1
            t_end_horizon = min(t_start + self.horizon_steps, total_timesteps)
            t_end_execute = min(t_start + execution_timesteps, total_timesteps)

            logger.info(f"MPC Iteration {iteration}: Window [{t_start/4}h - {t_end_horizon/4}h], "
                       f"Execute [{t_start/4}h - {t_end_execute/4}h]")

            # 1. Extract window data
            opt_input_window = self._slice_opt_input(
                opt_input_12h, t_start, t_end_horizon
            )

            # 2. Set initial SOC
            initial_soc_fraction = current_total_soc / self.battery_params['capacity_kwh']
            self.optimizer.battery_params['initial_soc'] = initial_soc_fraction

            # 3. Convert to DataFrame and solve
            country_data = self.adapter.to_country_data(opt_input_window)
            model = self.optimizer.build_optimization_model(
                country_data, c_rate, daily_cycle_limit=None
            )

            # 4. Fix initial segment SOC (Model II/III)
            if hasattr(model, 'e_soc_j'):
                initial_segment_soc = self._get_initial_segment_soc(current_total_soc)
                for j in model.J:
                    model.e_soc_j[0, j].setlb(initial_segment_soc[j])
                    model.e_soc_j[0, j].setub(initial_segment_soc[j])

                # Fix binary variables
                if hasattr(model, 'z_segment_active'):
                    for j in model.J:
                        if initial_segment_soc[j] > 1e-6:
                            model.z_segment_active[0, j].fix(1)
                        else:
                            model.z_segment_active[0, j].fix(0)

            # 5. Solve
            model, solver_results = self.optimizer.solve_model(model)
            solution = self.optimizer.extract_solution(model, solver_results)

            if solution['status'] not in ['optimal', 'feasible']:
                all_solutions['status'] = solution['status']
                break

            # 6. Commit execution window results
            execute_length = t_end_execute - t_start
            for t_rel in range(execute_length):
                t_abs = t_start + t_rel

                all_solutions['p_ch'][t_abs] = solution['p_ch'].get(t_rel, 0.0)
                all_solutions['p_dis'][t_abs] = solution['p_dis'].get(t_rel, 0.0)
                all_solutions['p_afrr_pos_e'][t_abs] = solution.get('p_afrr_pos_e', {}).get(t_rel, 0.0)
                all_solutions['p_afrr_neg_e'][t_abs] = solution.get('p_afrr_neg_e', {}).get(t_rel, 0.0)
                all_solutions['e_soc'][t_abs] = solution['e_soc'].get(t_rel, current_total_soc)

                # Capacity bids (block-based)
                rel_block = t_rel // 16
                all_solutions['c_fcr'][t_abs] = solution.get('c_fcr', {}).get(rel_block, 0.0)
                all_solutions['c_afrr_pos'][t_abs] = solution.get('c_afrr_pos', {}).get(rel_block, 0.0)
                all_solutions['c_afrr_neg'][t_abs] = solution.get('c_afrr_neg', {}).get(rel_block, 0.0)

                # Renewable
                if has_renewable:
                    all_solutions['p_renewable_self'][t_abs] = solution.get('p_renewable_self', {}).get(t_rel, 0.0)
                    all_solutions['p_renewable_export'][t_abs] = solution.get('p_renewable_export', {}).get(t_rel, 0.0)
                    all_solutions['p_renewable_curtail'][t_abs] = solution.get('p_renewable_curtail', {}).get(t_rel, 0.0)

            # 7. Update SOC (use last timestep of execution window)
            current_total_soc = all_solutions['e_soc'][t_end_execute - 1]

            # 8. Aggregate metrics (scale by execution window ratio)
            scale = execute_length / len(solution.get('p_ch', {}))
            all_solutions['profit_da'] += solution.get('profit_da', 0.0) * scale
            all_solutions['profit_afrr_energy'] += solution.get('profit_afrr_energy', 0.0) * scale
            all_solutions['profit_as_capacity'] += solution.get('profit_as_capacity', 0.0) * scale
            all_solutions['cost_cyclic'] += solution.get('cost_cyclic', 0.0) * scale
            all_solutions['cost_calendar'] += solution.get('cost_calendar', 0.0) * scale
            all_solutions['solve_time'] += solution.get('solve_time', 0.0)

        all_solutions['solver'] = getattr(solver_results, 'solver', {}).get('name', 'highs')
        return all_solutions

    def _slice_opt_input(
        self, opt_input: OptimizationInput, t_start: int, t_end: int
    ) -> OptimizationInput:
        """Slice OptimizationInput to window."""
        # 15-min prices
        da_prices = opt_input.da_prices[t_start:t_end]
        afrr_ep = opt_input.afrr_energy_pos[t_start:t_end]
        afrr_en = opt_input.afrr_energy_neg[t_start:t_end]

        # 4-h block prices (need to include complete blocks)
        block_start = t_start // 16
        block_end = (t_end + 15) // 16  # Round up to ensure coverage
        fcr_prices = opt_input.fcr_prices[block_start:block_end]
        afrr_cap_pos = opt_input.afrr_capacity_pos[block_start:block_end]
        afrr_cap_neg = opt_input.afrr_capacity_neg[block_start:block_end]

        # Renewable
        renewable = None
        if opt_input.renewable_generation:
            renewable = opt_input.renewable_generation[t_start:t_end]

        return OptimizationInput(
            time_horizon_hours=(t_end - t_start) / 4,
            da_prices=da_prices,
            afrr_energy_pos=afrr_ep,
            afrr_energy_neg=afrr_en,
            fcr_prices=fcr_prices,
            afrr_capacity_pos=afrr_cap_pos,
            afrr_capacity_neg=afrr_cap_neg,
            renewable_generation=renewable,
            battery_capacity_kwh=opt_input.battery_capacity_kwh,
            c_rate=opt_input.c_rate,
            efficiency=opt_input.efficiency,
            initial_soc=opt_input.initial_soc,
            model_type=opt_input.model_type,
            alpha=opt_input.alpha,
        )
```

### Step 2: Update `src/service/optimizer_service.py`

Add new method:

```python
def optimize_12h_mpc(
    self,
    market_prices: Dict[str, List[float]],
    generation_forecast: Optional[Dict] = None,
    model_type: str = "III",
    c_rate: float = 0.5,
    alpha: float = 1.0,
    horizon_hours: int = 6,  # Optimization window
    execution_hours: int = 4,  # Execution window
) -> OptimizationResult:
    """
    Solve 12h problem using MPC rolling horizon.

    Args:
        horizon_hours: MPC optimization window (default 6h)
        execution_hours: Commit execution window (default 4h)

    Returns: 12h complete schedule

    Estimated time: ~15-20 seconds (3 iterations × ~5 sec)
    """
    from .mpc import MPCRollingHorizon

    # 1. Adapt input
    opt_input_12h = self.adapter.adapt(
        market_prices=market_prices,
        generation_forecast=generation_forecast,
        battery_config=self._load_battery_config(),
        time_horizon_hours=12,
    )
    opt_input_12h.model_type = ModelType(model_type)
    opt_input_12h.alpha = alpha

    # 2. Get optimizer
    optimizer = self._get_optimizer(model_type, alpha)

    # 3. Create MPC helper
    mpc = MPCRollingHorizon(
        optimizer=optimizer,
        adapter=self.adapter,
        horizon_hours=horizon_hours,
        execution_hours=execution_hours,
    )

    # 4. Run MPC
    solution = mpc.solve_12h(opt_input_12h, c_rate)

    # 5. Build result
    return self._build_result(solution, opt_input_12h, None)
```

### Step 3: Update `src/api/main.py`

```python
# New request model (12h MPC)
class OptimizeRequestMPC(BaseModel):
    """12h MPC optimization request."""
    location: str = "Munich"
    country: str = "DE_LU"
    model_type: str = "III"
    c_rate: float = 0.5
    alpha: float = 1.0

    # 12h data (48 values @ 15-min)
    market_prices: MarketPrices12h
    renewable_generation: Optional[List[float]] = None  # 48 values

class MarketPrices12h(BaseModel):
    """12h market price validation."""
    day_ahead: List[float] = Field(min_length=48, max_length=48)
    afrr_energy_pos: List[float] = Field(min_length=48, max_length=48)
    afrr_energy_neg: List[float] = Field(min_length=48, max_length=48)
    fcr: List[float] = Field(min_length=3, max_length=3)  # 12h / 4h = 3 blocks
    afrr_capacity_pos: List[float] = Field(min_length=3, max_length=3)
    afrr_capacity_neg: List[float] = Field(min_length=3, max_length=3)

# New endpoint
@app.post("/api/v1/optimize-mpc", response_model=OptimizeResponse)
async def optimize_mpc(request: OptimizeRequestMPC):
    """
    12h MPC rolling horizon optimization.

    Strategy: 6h optimization window, 4h roll step
    Total iterations: 3
    Estimated response time: 15-20 seconds
    """
    generation_forecast = None
    if request.renewable_generation:
        generation_forecast = {"generation_kw": request.renewable_generation}

    result = service.optimize_12h_mpc(
        market_prices=request.market_prices.model_dump(),
        generation_forecast=generation_forecast,
        model_type=request.model_type,
        c_rate=request.c_rate,
        alpha=request.alpha,
    )
    return OptimizeResponse(status="success", data=result.model_dump())
```

---

## File Changes Summary

| File                                 | Action           | Details                                  |
| ------------------------------------ | ---------------- | ---------------------------------------- |
| `src/service/mpc.py`               | **NEW**    | `MPCRollingHorizon` class (~200 lines) |
| `src/service/__init__.py`          | **UPDATE** | Export `MPCRollingHorizon`             |
| `src/service/optimizer_service.py` | **UPDATE** | Add `optimize_12h_mpc()` (~20 lines)   |
| `src/api/main.py`                  | **UPDATE** | Add `/api/v1/optimize-mpc` (~50 lines) |
| `src/test/test_mpc.py`             | **NEW**    | MPC unit tests                           |

---

## Verification Checklist

- [ ] MPC produces 12h schedule (48 timesteps)
- [ ] 3 iterations, ~5 sec each
- [ ] SOC continuous between windows
- [ ] Revenue and costs correctly aggregated
- [ ] Tests pass

---

## API Usage Example

```bash
curl -X POST http://localhost:8000/api/v1/optimize-mpc \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "III",
    "market_prices": {
      "day_ahead": [50.0, 51.0, ...],  # 48 values
      "afrr_energy_pos": [40.0, ...],    # 48 values
      "afrr_energy_neg": [30.0, ...],    # 48 values
      "fcr": [100.0, 105.0, 110.0],     # 3 values
      "afrr_capacity_pos": [5.0, 6.0, 7.0],
      "afrr_capacity_neg": [10.0, 11.0, 12.0]
    },
    "renewable_generation": [0, 0, 10.5, ...]  # 48 values (optional)
  }'
```

Response time: ~15-20 seconds
