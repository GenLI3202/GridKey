### Phase 2: Core Implementation (Parallel After Phase 1)

These components depend on **Phase 1** but can be implemented **in parallel with each other**.

---

#### 2.1 [NEW] BESSOptimizerModelIIIRenew — Renewable Integration

Location: `src/core/optimizer.py` (extend existing file)

**Depends on:** LaTeX formulation (1.2)

> [!IMPORTANT]  
> **Output Format Specification:** The `extract_solution()` output MUST conform to  
> [`data/optimizer_output_template.json`](file:///d:/my_projects/GridPro/GridKey/data/optimizer_output_template.json)  
> New fields: `p_renewable_self_kw`, `p_renewable_export_kw`, `p_renewable_curtail_kw`, `revenue_renewable_export_eur`

Create a new class `BESSOptimizerModelIIIRenew` that extends `BESSOptimizerModelIII`:

```python

class BESSOptimizerModelIIIRenew(BESSOptimizerModelIII):

    """

    Extends Model III with simplified renewable power plant integration.

  

    New Decision Variables:

    - P_renewable_self[t]: Self-consumption power (kW) - used to charge battery

    - P_renewable_export[t]: Grid export power (kW) - sold at DA price

    - P_renewable_curtail[t]: Curtailed power (kW) - wasted, no revenue

  

    New Constraints:

    - Balance: P_renewable[t] = P_self + P_export + P_curtail

    - Self-consumption: P_self[t] <= P_renewable[t]

    - Export: Counted as DA revenue at DA price

  

    New Revenue:

    - R_export = Σ P_renewable_export[t] * DA_price[t] * dt

    """

  

    def__init__(self, alpha: float = 1.0, renewable_forecast: list = None):

        super().__init__(alpha=alpha)

        self.renewable_forecast = renewable_forecast or []

  

    defbuild_optimization_model(self, country_data, c_rate):

        # Call parent to build base model

        model = super().build_optimization_model(country_data, c_rate)

      

        # Add renewable variables

        # TODO: Implement as per Cst-R1, Cst-R2, Cst-R3

      

        return model

```

---

#### 2.2 [NEW] adapter.py — DataAdapter Class

Location: `src/service/adapter.py`

**Depends on:** `models.py` (1.1)

> [!IMPORTANT]  
> **Input Format Specification:** The `DataAdapter.adapt()` input MUST conform to  
> [`data/optimizer_input_template.json`](file:///d:/my_projects/GridPro/GridKey/data/optimizer_input_template.json)  
> Sources: `generation_forecast` (Module A), `market_prices` (Module B), `battery_config` (Config)

```python

import pandas as pd

from typing import Optional

from .models import OptimizationInput


classDataAdapter:

    """

    Converts external service outputs to optimizer-compatible format.

  

    Input Sources:

    - Weather Service (Module A): GenerationForecast

    - Price Service (Module B): MarketPrices

    - Battery Config: Static configuration

  

    Output: OptimizationInput (compatible with existing country_data format)

    """

  

    defadapt(

        self,

        market_prices: dict,           # From Price Service

        generation_forecast: dict,     # From Weather Service (optional)

        battery_config: dict,          # From config file

        time_horizon_hours: int = 48,

    ) -> OptimizationInput:

        """Convert external data to OptimizationInput."""

    

        # 1. Extract and validate market prices

        da_prices = self._extract_15min_prices(market_prices, "day_ahead")

        afrr_e_pos = self._extract_15min_prices(market_prices, "afrr_energy_pos")

        afrr_e_neg = self._extract_15min_prices(market_prices, "afrr_energy_neg")

    

        # 4-hour block prices (need to be replicated or mapped)

        fcr_prices = self._extract_block_prices(market_prices, "fcr")

        afrr_cap_pos = self._extract_block_prices(market_prices, "afrr_capacity_pos")

        afrr_cap_neg = self._extract_block_prices(market_prices, "afrr_capacity_neg")

    

        # 2. Extract renewable generation (if available)

        renewable_gen = None

        if generation_forecast:

            renewable_gen = self._extract_generation(generation_forecast)

    

        # 3. Build OptimizationInput

        return OptimizationInput(

            time_horizon_hours=time_horizon_hours,

            da_prices=da_prices,

            afrr_energy_pos=afrr_e_pos,

            afrr_energy_neg=afrr_e_neg,

            fcr_prices=fcr_prices,

            afrr_capacity_pos=afrr_cap_pos,

            afrr_capacity_neg=afrr_cap_neg,

            renewable_generation=renewable_gen,

            battery_capacity_kwh=battery_config.get("capacity_kwh", 4472),

            c_rate=battery_config.get("c_rate", 0.5),

            efficiency=battery_config.get("efficiency", 0.95),

            initial_soc=battery_config.get("initial_soc", 0.5),

        )

  

    defto_country_data(self, opt_input: OptimizationInput) -> pd.DataFrame:

        """

        Convert OptimizationInput to legacy country_data DataFrame format.

    

        This ensures backward compatibility with existing build_optimization_model().

        """

        # Implementation: Create DataFrame with expected columns

        # price_day_ahead, price_fcr, price_afrr_pos, price_afrr_neg,

        # price_afrr_energy_pos, price_afrr_energy_neg, block_id, day_id

        pass  # TODO: Implement

  

    def_extract_15min_prices(self, market_prices: dict, key: str) -> list:

        """Extract and validate 15-min resolution prices."""

        pass  # TODO: Implement

  

    def_extract_block_prices(self, market_prices: dict, key: str) -> list:

        """Extract 4-hour block prices."""

        pass  # TODO: Implement

  

    def_extract_generation(self, forecast: dict) -> list:

        """Extract PV + Wind generation forecast."""

        pass  # TODO: Implement

```

---
