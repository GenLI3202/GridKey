### Phase 3: Service Layer (Sequential)

This component is the **critical path** — it depends on both Phase 1 and Phase 2. Details refer to [ibm_hack_dev_phase3.md](ibm_hack_dev_phase3.md).

---

#### 3.1 [NEW] optimizer_service.py — OptimizerService Class

Location: `src/service/optimizer_service.py`

**Depends on:**`models.py` (1.1), `adapter.py` (2.2), `BESSOptimizerModelIII-renewables` (2.1)

##### File Structure

```

src/

├── core/

│   └── optimizer.py              # Existing (extend with renewables)

└── service/                      # NEW directory

    ├── __init__.py

    ├── models.py                 # Data models (Phase 1)

    ├── adapter.py                # DataAdapter (Phase 2)

    └── optimizer_service.py      # OptimizerService (Phase 3)

```

```python

from typing import Optional

import logging


from .models import OptimizationInput, OptimizationResult, ModelType

from .adapter import DataAdapter


logger = logging.getLogger(__name__)


classOptimizerService:

    """

    Unified Optimizer Service interface (Section 6.5 of Blueprint).

  

    Orchestrates the complete optimization workflow:

    1. Data validation and adaptation

    2. Model selection and construction

    3. Solving and result extraction

    4. Result formatting for API/Agent consumption

  

    Usage:

        service = OptimizerService()

        result = service.optimize(

            market_prices=price_service.get_market_prices("DE_LU", 48),

            generation_forecast=weather_service.get_generation_forecast("Munich", 48),

            model_type="III",

            c_rate=0.5,

            alpha=1.0

        )

    """

  

    def__init__(self):

        self.adapter = DataAdapter()

        self._optimizer_cache = {}  # Cache optimizer instances by model type

  

    defoptimize(

        self,

        market_prices: dict,

        generation_forecast: Optional[dict] = None,

        model_type: str = "III",

        c_rate: float = 0.5,

        alpha: float = 1.0,

        time_horizon_hours: int = 48,

    ) -> OptimizationResult:

        """

        Run complete optimization and return structured result.

    

        Args:

            market_prices: Market price data from Price Service

            generation_forecast: Renewable forecast from Weather Service (optional)

            model_type: "I", "II", "III", or "III-renewables"

            c_rate: Battery C-rate (0.25, 0.33, 0.5)

            alpha: Degradation cost weight

            time_horizon_hours: Optimization horizon

        

        Returns:

            OptimizationResult with schedule, metrics, and metadata

        

        Raises:

            ValueError: If input validation fails

            RuntimeError: If solver fails or times out

        """

    

        logger.info(f"Starting optimization: model={model_type}, c_rate={c_rate}, alpha={alpha}")

    

        # 1. Load battery config

        battery_config = self._load_battery_config()

        battery_config["c_rate"] = c_rate

    

        # 2. Adapt input data

        opt_input = self.adapter.adapt(

            market_prices=market_prices,

            generation_forecast=generation_forecast,

            battery_config=battery_config,

            time_horizon_hours=time_horizon_hours,

        )

        opt_input.model_type = ModelType(model_type)

        opt_input.alpha = alpha

    

        # 3. Get or create optimizer

        optimizer = self._get_optimizer(model_type, alpha)

    

        # 4. Convert to legacy format

        country_data = self.adapter.to_country_data(opt_input)

    

        # 5. Build and solve model

        model = optimizer.build_optimization_model(country_data, c_rate)

        solver_results = optimizer.solve_model(model)

    

        # 6. Extract solution

        solution = optimizer.extract_solution(model, solver_results)

    

        # 7. Convert to OptimizationResult

        returnself._build_result(solution, opt_input, solver_results)

  

    defoptimize_from_input(self, opt_input: OptimizationInput) -> OptimizationResult:

        """

        Run optimization from a pre-built OptimizationInput.

        Useful for API endpoints that receive JSON input.

        """

        pass  # TODO: Implement

  

    def_get_optimizer(self, model_type: str, alpha: float):

        """Get or create optimizer instance."""

        from src.core.optimizer import (

            BESSOptimizerModelI,

            BESSOptimizerModelII, 

            BESSOptimizerModelIII,

        )

        # TODO: Add BESSOptimizerModelIIIRenewables when implemented

    

        cache_key = f"{model_type}_{alpha}"

        if cache_key notinself._optimizer_cache:

            if model_type == "I":

                self._optimizer_cache[cache_key] = BESSOptimizerModelI()

            elif model_type == "II":

                self._optimizer_cache[cache_key] = BESSOptimizerModelII(alpha=alpha)

            elif model_type in ("III", "III-renewables"):

                self._optimizer_cache[cache_key] = BESSOptimizerModelIII(alpha=alpha)

            else:

                raiseValueError(f"Unknown model type: {model_type}")

    

        returnself._optimizer_cache[cache_key]

  

    def_load_battery_config(self) -> dict:

        """Load battery configuration from file."""

        from src.utils.config_loader import ConfigLoader

        # TODO: Implement proper config loading

        return {

            "capacity_kwh": 4472,

            "c_rate": 0.5,

            "efficiency": 0.95,

            "initial_soc": 0.5,

        }

  

    def_build_result(

        self, 

        solution: dict, 

        opt_input: OptimizationInput,

        solver_results

    ) -> OptimizationResult:

        """Convert internal solution dict to OptimizationResult."""

        pass  # TODO: Implement

```

---
