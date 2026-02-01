"""
DataAdapter — Format conversion between API service layer and optimizer core.

Provides two conversion paths:

1. ``adapt()``  : raw service dicts  →  ``OptimizationInput``   (API layer)
2. ``to_country_data()`` : ``OptimizationInput``  →  ``pd.DataFrame``  (optimizer layer)

The resulting DataFrame has exactly the columns expected by
``BESSOptimizerModelI.build_optimization_model(country_data, ...)``.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from .models import ModelType, OptimizationInput

logger = logging.getLogger(__name__)

# Constants matching the optimizer's time resolution
TIMESTEP_MINUTES = 15
TIMESTEPS_PER_HOUR = 60 // TIMESTEP_MINUTES          # 4
TIMESTEPS_PER_BLOCK = 4 * TIMESTEPS_PER_HOUR          # 16  (4-hour blocks)
BLOCKS_PER_DAY = 6


class DataAdapter:
    """Converts between API service outputs and optimizer-internal formats.

    Input sources (via ``adapt()``):
        - Weather Service (Module A): generation forecast
        - Price Service  (Module B): market prices
        - Battery Config: static configuration

    Output (via ``to_country_data()``):
        pandas DataFrame compatible with ``build_optimization_model()``.
    """

    # Required columns in the output DataFrame
    REQUIRED_COLUMNS = [
        'price_day_ahead',
        'price_afrr_energy_pos',
        'price_afrr_energy_neg',
        'price_fcr',
        'price_afrr_pos',
        'price_afrr_neg',
        'w_afrr_pos',
        'w_afrr_neg',
        'block_id',
        'day_id',
        'block_of_day',
        'hour',
        'day_of_year',
        'month',
        'year',
        'timestamp',
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(
        self,
        market_prices: Dict,
        generation_forecast: Optional[Dict] = None,
        battery_config: Optional[Dict] = None,
        time_horizon_hours: int = 48,
    ) -> OptimizationInput:
        """Convert raw service outputs to ``OptimizationInput``.

        Args:
            market_prices: Dict with keys ``day_ahead``, ``afrr_energy_pos``,
                ``afrr_energy_neg``, ``fcr``, ``afrr_capacity_pos``,
                ``afrr_capacity_neg``.  Each value is a list of floats.
            generation_forecast: Optional dict with key ``generation_kw``
                (list of floats, 15-min resolution) or ``pv_kw`` + ``wind_kw``.
            battery_config: Optional dict overriding default battery params.
            time_horizon_hours: Optimisation horizon in hours.

        Returns:
            Validated ``OptimizationInput`` instance.
        """
        battery_config = battery_config or {}

        # 15-min resolution prices
        da_prices = self._extract_15min_prices(market_prices, 'day_ahead')
        afrr_e_pos = self._extract_15min_prices(market_prices, 'afrr_energy_pos')
        afrr_e_neg = self._extract_15min_prices(market_prices, 'afrr_energy_neg')

        # 4-hour block prices
        fcr_prices = self._extract_block_prices(market_prices, 'fcr')
        afrr_cap_pos = self._extract_block_prices(market_prices, 'afrr_capacity_pos')
        afrr_cap_neg = self._extract_block_prices(market_prices, 'afrr_capacity_neg')

        # Renewable generation (optional)
        renewable_gen = None
        if generation_forecast:
            renewable_gen = self._extract_generation(generation_forecast)

        return OptimizationInput(
            time_horizon_hours=time_horizon_hours,
            da_prices=da_prices,
            afrr_energy_pos=afrr_e_pos,
            afrr_energy_neg=afrr_e_neg,
            fcr_prices=fcr_prices,
            afrr_capacity_pos=afrr_cap_pos,
            afrr_capacity_neg=afrr_cap_neg,
            renewable_generation=renewable_gen,
            battery_capacity_kwh=battery_config.get('capacity_kwh', 4472),
            c_rate=battery_config.get('c_rate', 0.5),
            efficiency=battery_config.get('efficiency', 0.95),
            initial_soc=battery_config.get('initial_soc', 0.5),
        )

    def to_country_data(
        self,
        opt_input: OptimizationInput,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Convert ``OptimizationInput`` to optimizer-compatible DataFrame.

        Produces a DataFrame whose columns exactly match what
        ``BESSOptimizerModelI.build_optimization_model()`` expects.

        Args:
            opt_input: Validated optimisation input.
            start_time: Starting timestamp (default: 2024-01-01 00:00).

        Returns:
            DataFrame with price, time-identifier, and optional renewable
            columns.
        """
        if start_time is None:
            start_time = datetime(2024, 1, 1, 0, 0)

        n_timesteps = opt_input.time_horizon_hours * TIMESTEPS_PER_HOUR

        # Generate timestamps
        timestamps = [
            start_time + timedelta(minutes=TIMESTEP_MINUTES * i)
            for i in range(n_timesteps)
        ]

        # Build DataFrame
        df = pd.DataFrame({'timestamp': timestamps})

        # -- 15-min resolution prices (direct mapping) --------------------
        df['price_day_ahead'] = opt_input.da_prices[:n_timesteps]
        df['price_afrr_energy_pos'] = opt_input.afrr_energy_pos[:n_timesteps]
        df['price_afrr_energy_neg'] = opt_input.afrr_energy_neg[:n_timesteps]

        # -- 4-hour block prices (forward-fill to 15-min) -----------------
        df['price_fcr'] = self._expand_block_prices(
            opt_input.fcr_prices, n_timesteps
        )
        df['price_afrr_pos'] = self._expand_block_prices(
            opt_input.afrr_capacity_pos, n_timesteps
        )
        df['price_afrr_neg'] = self._expand_block_prices(
            opt_input.afrr_capacity_neg, n_timesteps
        )

        # -- aFRR activation weights (deterministic default) --------------
        df['w_afrr_pos'] = 1.0
        df['w_afrr_neg'] = 1.0

        # -- Time identifiers ---------------------------------------------
        df['hour'] = [ts.hour for ts in timestamps]
        df['day_of_year'] = [ts.timetuple().tm_yday for ts in timestamps]
        df['month'] = [ts.month for ts in timestamps]
        df['year'] = [ts.year for ts in timestamps]
        df['block_of_day'] = [ts.hour // 4 for ts in timestamps]
        df['day_id'] = df['day_of_year']
        df['block_id'] = (df['day_of_year'] - df['day_of_year'].iloc[0]) * BLOCKS_PER_DAY + df['block_of_day']

        # -- Renewable generation (optional) ------------------------------
        if opt_input.renewable_generation is not None:
            df['p_renewable_forecast_kw'] = opt_input.renewable_generation[:n_timesteps]

        logger.info(
            "DataAdapter.to_country_data: %d timesteps, %d blocks, renewable=%s",
            n_timesteps,
            df['block_id'].nunique(),
            'yes' if 'p_renewable_forecast_kw' in df.columns else 'no',
        )

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_block_prices(
        block_prices: List[float], n_timesteps: int
    ) -> List[float]:
        """Expand 4-hour block prices to 15-min resolution.

        Each block price is repeated ``TIMESTEPS_PER_BLOCK`` (16) times.
        If the expanded list is shorter than *n_timesteps*, the last price
        is forward-filled; if longer, it is truncated.
        """
        expanded: List[float] = []
        for price in block_prices:
            expanded.extend([price] * TIMESTEPS_PER_BLOCK)

        # Pad or truncate
        if len(expanded) < n_timesteps:
            last = expanded[-1] if expanded else 0.0
            expanded.extend([last] * (n_timesteps - len(expanded)))
        return expanded[:n_timesteps]

    @staticmethod
    def _extract_15min_prices(market_prices: Dict, key: str) -> List[float]:
        """Extract and validate 15-min resolution prices from service dict."""
        prices = market_prices.get(key, [])
        if not prices:
            raise ValueError(f"Missing or empty 15-min prices for key '{key}'")
        return [float(p) for p in prices]

    @staticmethod
    def _extract_block_prices(market_prices: Dict, key: str) -> List[float]:
        """Extract 4-hour block prices from service dict."""
        prices = market_prices.get(key, [])
        if not prices:
            raise ValueError(f"Missing or empty block prices for key '{key}'")
        return [float(p) for p in prices]

    @staticmethod
    def _extract_generation(forecast: Dict) -> List[float]:
        """Extract combined PV + Wind generation forecast.

        Accepts either a single ``generation_kw`` key or separate
        ``pv_kw`` / ``wind_kw`` keys that are summed element-wise.
        """
        if 'generation_kw' in forecast:
            return [float(g) for g in forecast['generation_kw']]

        pv = forecast.get('pv_kw', [])
        wind = forecast.get('wind_kw', [])

        if not pv and not wind:
            return []

        max_len = max(len(pv), len(wind))
        pv_ext = list(pv) + [0.0] * (max_len - len(pv))
        wind_ext = list(wind) + [0.0] * (max_len - len(wind))

        return [float(p) + float(w) for p, w in zip(pv_ext, wind_ext)]
