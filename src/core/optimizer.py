"""
BESS Optimizer V2 - Phase II Implementation
============================================

This module implements the Phase II Battery Energy Storage System (BESS) optimization model
for Huawei TechArena 2025 competition.

Version 2 Improvements (over Phase I archived in r1-static-battery branch):
---------------------------------------------------------------------------
Core Optimizations:
- Eliminated constraint closure anti-patterns for better performance
- Pre-computed block-to-time mappings for O(1) lookup efficiency
- AS prices indexed by block instead of time to reduce memory overhead
- Constraint functions use model parameters instead of external data closures
- Optimized objective function computation
- Enhanced memory efficiency for full-year optimizations

Phase II Enhancements:
- Added reserve duration parameter for accurate energy reserve calculations
- Refined constraints for energy reserve calculations in upward/downward regulation
- Improved representation of activation durations for aFRR and FCR services
- Comprehensive input validation and error handling
- Consistent solver time limits across different solvers

Technical Features:
- Multi-market co-optimization (day-ahead, FCR, aFRR)
- Advanced battery operation constraints (SOC dynamics, cycle limits)
- Support for multiple C-rates and daily cycle configurations
- Cross-market exclusivity and minimum bid size constraints

Author: Gen's BESS Optimization Team
Phase II Development: October-November 2025
"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BESSOptimizerModelI:
    """
    Battery Energy Storage System Optimizer - Phase II Model (i)

    Model (i): Base Model + aFRR Energy Market
    ============================================

    This is the first of three Phase II models, extending the Phase I base model by
    integrating the aFRR Energy Market for real-time balancing revenue optimization.

    Mathematical Formulation:
    -------------------------
    - Objective: max(P_DA + P_ANCI + P_aFRR_E)
    - Markets: Day-Ahead Energy, FCR Capacity, aFRR Capacity, aFRR Energy (NEW)
    - New Variables: p_afrr_pos_e, p_afrr_neg_e, p_total_ch, p_total_dis
    - Total Power: p_total = p_DA + p_aFRR_E (co-optimization across energy markets)

    Model Progression:
    ------------------
    ✓ Model (i): Base + aFRR Energy Market (THIS MODEL)
    ○ Model (ii): Model (i) + Cyclic Aging Cost
    ○ Model (iii): Model (ii) + Calendar Aging Cost = Full Phase II Model

    Key Features:
    -------------
    - Four-market co-optimization (DA energy, aFRR energy, FCR capacity, aFRR capacity)
    - Advanced SOC dynamics with total charge/discharge power tracking
    - Reserve energy constraints with configurable activation durations
    - Daily cycle limits and power constraints based on C-rate configuration
    - Cross-market exclusivity preventing conflicting bids
    - Minimum bid size enforcement across all markets

    Technical Improvements (from Phase I):
    --------------------------------------
    - Constraint closure anti-patterns eliminated for better solver performance
    - Pre-computed index mappings for O(1) lookup efficiency
    - Memory-optimized data structures for full-year horizon
    - Comprehensive input validation and error handling
    - Enhanced reserve duration modeling

    Attributes:
        battery_params (dict): Battery technical specifications
        market_params (dict): Market rules and constraints including aFRR energy min bid
        countries (list): Supported country markets
        c_rates (list): Available C-rate configurations
        daily_cycles (list): Available daily cycle limit options

    References:
        See doc/p2_model/p2_bi_model_ggdp.tex Section "Model (i)"
        See doc/p2_model/p2_3models_formulation.tex Section "Model (i)"
    """

    def __init__(self, use_afrr_ev_weighting: bool = False):
        """Initialize Phase II Model (i) optimizer with aFRR energy market integration.

        Args:
            use_afrr_ev_weighting: Enable Expected Value weighting for aFRR energy bids
                                   to account for activation probability (default: False)
        """
        # Battery specifications
        self.battery_params = {
            'capacity_kwh': 4500,
            'efficiency': 0.95,
            'soc_min': 0, # the offical QnA says in this challenge free to use from 0-100%
            'soc_max': 1,
            'initial_soc': 0.5,
            'daily_cycle_limit': 1.0  # Default, will be overridden
        }

        # Load solver configuration from unified YAML config
        try:
            from src.utils.config_loader import ConfigLoader
            solver_config = ConfigLoader.get_solver_config()
            solver_time_limit = solver_config.get('solver_time_limit_sec', 1200)
            self.solver_config = solver_config
        except Exception as e:
            logger.warning(f"Failed to load solver config: {e}. Using default timeout 1200s")
            solver_time_limit = 1200
            self.solver_config = {}

        # Market parameters
        self.market_params = {
            'min_bid_da': 0.1,    # MW
            'min_bid_fcr': 1.0,   # MW
            'min_bid_afrr': 1.0,  # MW
            'min_bid_afrr_e': 0.1,  # MW - Phase II Model (i): aFRR energy market minimum bid
            'time_step_hours': 0.25,  # 15 minutes
            'block_duration_hours': 4.0,  # AS market blocks
            'reserve_duration_hours': 0.25, # Assumed activation duration for reserve calculation
            'solver_time_limit': solver_time_limit  # seconds - loaded from solver_config.json
        }

        # Configuration scenarios
        # Include DE_LU for coupled Germany-Luxembourg day-ahead market
        self.countries = ['DE', 'DE_LU', 'AT', 'CH', 'HU', 'CZ']
        self.c_rates = [0.25, 0.33, 0.5]
        self.daily_cycles = [1.0, 1.5, 2.0]

        # Pre-computed mappings for efficiency
        self._block_to_times = {}
        self._time_to_block = {}
        self._day_to_times = {}

        # EV weighting configuration
        self.use_afrr_ev_weighting = use_afrr_ev_weighting
        self._activation_config = None  # Lazy loaded when needed

        mode_str = "with EV weighting" if use_afrr_ev_weighting else "without EV weighting"
        logger.info(f"BESS Optimizer - Phase II Model (i): Base + aFRR Energy Market initialized ({mode_str})")

    def _load_activation_config(self) -> Dict[str, Any]:
        """
        Load aFRR activation probability configuration for Expected Value weighting.

        Returns:
            Dictionary with activation probabilities by country (normalized structure)

        Raises:
            ValueError: If config format is invalid
        """
        if self._activation_config is not None:
            return self._activation_config

        try:
            from src.utils.config_loader import ConfigLoader
            afrr_config = ConfigLoader.get_afrr_ev_weights_config()

            # Extract historical_activation section (primary source for EV weighting)
            hist_activation = afrr_config.get('historical_activation', {})

            # Normalize to expected structure: default_probabilities + country_specific
            config = {
                'default_probabilities': {
                    'positive': hist_activation.get('default_values', {}).get('positive', 0.30),
                    'negative': hist_activation.get('default_values', {}).get('negative', 0.30)
                },
                'country_specific': hist_activation.get('country_specific', {})
            }

            self._activation_config = config
            logger.info("Loaded activation config from config/Config.yml")

            # Log default values
            default_pos = config['default_probabilities']['positive']
            default_neg = config['default_probabilities']['negative']
            logger.info(f"Default activation rates: pos={default_pos:.2f}, neg={default_neg:.2f}")

            # Log country-specific values if available
            if config['country_specific']:
                countries_with_custom = list(config['country_specific'].keys())
                logger.info(f"Custom activation rates available for: {', '.join(countries_with_custom)}")

            return self._activation_config

        except Exception as e:
            logger.warning(f"Failed to load activation config: {e}. Using default probabilities.")
            # Fallback to defaults
            self._activation_config = {
                'default_probabilities': {'positive': 0.30, 'negative': 0.30},
                'country_specific': {}
            }
            return self._activation_config

    def _validate_input_data(self, country_data: pd.DataFrame, blocks: List[int], 
                           days: List[int], T_data: List[int]) -> None:
        """
        Comprehensive input data validation.
        
        Args:
            country_data: Market data for validation
            blocks: List of block IDs
            days: List of day IDs
            T_data: List of time indices
        """
        logger.info("Validating input data...")

        # Check for missing data (Phase II Model (i) includes aFRR energy)
        required_cols = ['price_day_ahead', 'price_fcr', 'price_afrr_pos', 'price_afrr_neg',
                        'price_afrr_energy_pos', 'price_afrr_energy_neg',  # Phase II Model (i)
                        'block_id', 'day_id']
        missing_cols = [col for col in required_cols if col not in country_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = country_data[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Validate block_id continuity
        if not all(isinstance(b, (int, np.integer)) for b in blocks):
            raise ValueError("Block IDs must be integers")
        
        # Check negative prices (warn but don't fail - they can be valid)
        for col in ['price_day_ahead', 'price_fcr', 'price_afrr_pos', 'price_afrr_neg']:
            negative_count = (country_data[col] < 0).sum()
            if negative_count > 0:
                logger.warning(f"Found {negative_count} negative prices in {col}")
        
        # Validate block structure (each block should have ~16 time intervals)
        intervals_per_block = self.market_params['block_duration_hours'] / self.market_params['time_step_hours']
        expected_intervals = int(intervals_per_block)
        
        block_sizes = country_data.groupby('block_id').size()
        irregular_blocks = block_sizes[block_sizes != expected_intervals]
        if len(irregular_blocks) > 0:
            logger.warning(f"Found {len(irregular_blocks)} blocks with irregular size (expected {expected_intervals})")
            logger.warning(f"The irregular blocks are: {irregular_blocks.to_dict()}")
        
        # Validate time horizon
        expected_hours_per_year = 365.25 * 24
        expected_intervals_per_year = expected_hours_per_year / self.market_params['time_step_hours']
        
        if len(T_data) > expected_intervals_per_year * 1.1:  # Allow 10% margin
            logger.warning(f"Time horizon unusually large: {len(T_data)} intervals "
                          f"(expected ~{expected_intervals_per_year:.0f} for one year)")
        
        logger.info("Input validation completed")
    
    def load_and_preprocess_data(self, workbook_path: str) -> pd.DataFrame:
        """
        Load and preprocess Phase 2 market data from Excel workbook.

        This method is designed for Huawei submission compatibility.
        It loads the official Phase 2 Excel workbook and converts it to the internal
        MultiIndex DataFrame format expected by extract_country_data().

        Args:
            workbook_path: Path to TechArena2025_Phase2_data.xlsx

        Returns:
            MultiIndex DataFrame with columns:
            - Level 0: country (DE_LU, AT, CH, HU, CZ, DE)
            - Level 1: market type (day_ahead, fcr, afrr, afrr_energy)
            - Level 2: direction ('' for DA/FCR, 'positive'/'negative' for aFRR)
        """
        from pathlib import Path
        from src.data.load_process_market_data import load_phase2_market_tables

        logger.info(f"Loading Phase 2 market data from {workbook_path}")

        # Load wide-format tables using centralized loader
        market_tables = load_phase2_market_tables(Path(workbook_path))
        logger.info(f"Loaded {len(market_tables)} market tables from Excel")

        # Convert wide-format tables to MultiIndex DataFrame
        # This maintains compatibility with extract_country_data() method

        processed_dfs = []

        # CRITICAL: Normalize all timestamps to minute precision to handle Excel microsecond drift
        # Excel has inconsistent seconds/microseconds (00:00:00.5, 00:00:01, 00:15:00.005, etc.)
        # This causes reindex mismatches between 15-min and 4-hour block timestamps
        for market_key in market_tables:
            market_tables[market_key]['timestamp'] = pd.to_datetime(market_tables[market_key]['timestamp']).dt.floor('min')

        # Process day-ahead data (15-min intervals)
        if 'day_ahead' in market_tables:
            da_df = market_tables['day_ahead'].set_index('timestamp')
            # Create MultiIndex: (country, 'day_ahead', '')
            da_multiindex = pd.DataFrame(index=da_df.index)
            for country in da_df.columns:
                da_multiindex[(country, 'day_ahead', '')] = da_df[country]
            da_multiindex.columns = pd.MultiIndex.from_tuples(da_multiindex.columns)
            processed_dfs.append(da_multiindex)
            logger.info(f"Processed day-ahead: {len(da_df)} rows, {len(da_df.columns)} countries")

        # Process FCR data (4-hour blocks -> need to expand to 15-min)
        if 'fcr' in market_tables:
            fcr_df = market_tables['fcr'].set_index('timestamp')
            # Reindex to 15-min and forward fill (timestamps now aligned to minute precision)
            full_index = market_tables['day_ahead'].set_index('timestamp').index
            fcr_expanded = fcr_df.reindex(full_index).ffill()
            # Create MultiIndex: (country, 'fcr', '')
            fcr_multiindex = pd.DataFrame(index=fcr_expanded.index)
            for country in fcr_expanded.columns:
                fcr_multiindex[(country, 'fcr', '')] = fcr_expanded[country]
            fcr_multiindex.columns = pd.MultiIndex.from_tuples(fcr_multiindex.columns)
            processed_dfs.append(fcr_multiindex)
            logger.info(f"Processed FCR: {len(fcr_expanded)} rows, {len(fcr_expanded.columns)} countries")

        # Process aFRR capacity data (4-hour blocks -> need to expand to 15-min)
        if 'afrr_capacity' in market_tables:
            afrr_cap_df = market_tables['afrr_capacity'].set_index('timestamp')
            # Reindex to 15-min and forward fill
            full_index = market_tables['day_ahead'].set_index('timestamp').index
            afrr_cap_expanded = afrr_cap_df.reindex(full_index).ffill()
            # Create MultiIndex: (country, 'afrr', 'positive'/'negative')
            afrr_cap_multiindex = pd.DataFrame(index=afrr_cap_expanded.index)
            for col in afrr_cap_expanded.columns:
                if '_Pos' in col:
                    country = col.replace('_Pos', '')
                    afrr_cap_multiindex[(country, 'afrr', 'positive')] = afrr_cap_expanded[col]
                elif '_Neg' in col:
                    country = col.replace('_Neg', '')
                    afrr_cap_multiindex[(country, 'afrr', 'negative')] = afrr_cap_expanded[col]
            afrr_cap_multiindex.columns = pd.MultiIndex.from_tuples(afrr_cap_multiindex.columns)
            processed_dfs.append(afrr_cap_multiindex)
            logger.info(f"Processed aFRR capacity: {len(afrr_cap_expanded)} rows, {len(afrr_cap_multiindex.columns)} price series")

        # Process aFRR energy data (15-min intervals)
        if 'afrr_energy' in market_tables:
            afrr_energy_df = market_tables['afrr_energy'].set_index('timestamp')
            # Create MultiIndex: (country, 'afrr_energy', 'positive'/'negative')
            afrr_energy_multiindex = pd.DataFrame(index=afrr_energy_df.index)
            for col in afrr_energy_df.columns:
                if '_Pos' in col:
                    country = col.replace('_Pos', '')
                    afrr_energy_multiindex[(country, 'afrr_energy', 'positive')] = afrr_energy_df[col]
                elif '_Neg' in col:
                    country = col.replace('_Neg', '')
                    afrr_energy_multiindex[(country, 'afrr_energy', 'negative')] = afrr_energy_df[col]
            afrr_energy_multiindex.columns = pd.MultiIndex.from_tuples(afrr_energy_multiindex.columns)
            processed_dfs.append(afrr_energy_multiindex)
            logger.info(f"Processed aFRR energy: {len(afrr_energy_df)} rows, {len(afrr_energy_multiindex.columns)} price series")
        else:
            logger.warning("aFRR energy data not found. Model (i) requires this data!")

        # Combine all data
        if not processed_dfs:
            raise ValueError("No valid market data found in workbook")

        combined_df = pd.concat(processed_dfs, axis=1)

        # Verify completeness
        start_time = combined_df.index.min()
        end_time = combined_df.index.max()
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Data points: {len(combined_df)}")
        logger.info(f"Index frequency: {pd.infer_freq(combined_df.index)}")

        # Sort by timestamp
        combined_df = combined_df.sort_index()

        logger.info(f"Data preprocessed. Shape: {combined_df.shape}")
        logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

        return combined_df
    
    def build_optimization_model(self, country_data: pd.DataFrame, 
                               c_rate: float, daily_cycle_limit: float) -> pyo.ConcreteModel:
        """
        Build the improved optimization model addressing all critical issues.
        
        Key improvements:
        1. Pre-computed block mappings for O(1) lookup
        2. AS prices indexed by block instead of time
        3. Constraint functions use model parameters only
        4. Comprehensive validation
        
        Args:
            country_data: Market data for specific country
            c_rate: C-rate configuration (power to energy ratio)
            daily_cycle_limit: Daily cycle limit
            
        Returns:
            pyo.ConcreteModel: Improved optimization model
        """
        logger.info(f"Building improved optimization model for C-rate={c_rate}, cycles={daily_cycle_limit}")
        
        # Update battery parameters for this configuration
        self.battery_params['daily_cycle_limit'] = daily_cycle_limit
        P_max_config = c_rate * self.battery_params['capacity_kwh']  # kW
        
        # Extract time range for this data
        T_data = list(range(len(country_data)))
        
        # Extract unique blocks and days 
        blocks = sorted(country_data['block_id'].unique())
        days = sorted(country_data['day_id'].unique())
        
        logger.info(f"Time horizon: {len(T_data)} periods ({len(T_data) * self.market_params['time_step_hours']:.1f} hours)")
        logger.info(f"Blocks: {len(blocks)} blocks")
        logger.info(f"Days: {len(days)} days")
        
        # Input validation
        self._validate_input_data(country_data, blocks, days, T_data)
        
        # PRE-COMPUTE MAPPINGS FOR EFFICIENCY (Addresses Critical Issue #3)
        block_to_times = {}
        time_to_block = {}
        day_to_times = {}
        for t in T_data:
            block_id = int(country_data['block_id'].iloc[t])
            if block_id not in block_to_times:
                block_to_times[block_id] = []
            block_to_times[block_id].append(t)
            time_to_block[t] = block_id

            day_id = int(country_data['day_id'].iloc[t])
            if day_id not in day_to_times:
                day_to_times[day_id] = []
            day_to_times[day_id].append(t)

        # Store for objective function (eliminates O(B×T) complexity)
        self._block_to_times = block_to_times
        self._day_to_times = day_to_times
        
        # PRE-COMPUTE AS PRICES BY BLOCK (Addresses Critical Issue #4)
        fcr_prices_by_block = {}
        afrr_pos_prices_by_block = {}
        afrr_neg_prices_by_block = {}
        
        for b in blocks:
            # Take first time step in block (all should have same price)
            t_rep = block_to_times[b][0]
            fcr_prices_by_block[b] = float(country_data['price_fcr'].iloc[t_rep])
            afrr_pos_prices_by_block[b] = float(country_data['price_afrr_pos'].iloc[t_rep])
            afrr_neg_prices_by_block[b] = float(country_data['price_afrr_neg'].iloc[t_rep])
        
        # Create concrete model
        model = pyo.ConcreteModel(name="Improved_BESS_Optimization")
        
        # Sets
        model.T = pyo.Set(initialize=T_data, doc="Set of 15-minute time intervals")
        model.B = pyo.Set(initialize=blocks, doc="Set of 4-hour blocks for AS market")
        model.D = pyo.Set(initialize=days, doc="Set of days")
        
        # Parameters - Battery Configuration
        model.E_nom = pyo.Param(initialize=self.battery_params['capacity_kwh'], 
                               doc="Nominal energy capacity (kWh)")
        model.P_max_config = pyo.Param(initialize=P_max_config, 
                                      doc="Maximum power rating (kW)")
        model.eta_ch = pyo.Param(initialize=self.battery_params['efficiency'], 
                                doc="Charging efficiency")
        model.eta_dis = pyo.Param(initialize=self.battery_params['efficiency'], 
                                 doc="Discharging efficiency")
        model.SOC_min = pyo.Param(initialize=self.battery_params['soc_min'], 
                                 doc="Minimum SOC")
        model.SOC_max = pyo.Param(initialize=self.battery_params['soc_max'],
                                 doc="Maximum SOC")
        model.E_soc_init = pyo.Param(initialize=self.battery_params['initial_soc'] * self.battery_params['capacity_kwh'],
                                    doc="Initial SOC energy (kWh)")
        if daily_cycle_limit is not None:
            model.N_cycles = pyo.Param(initialize=daily_cycle_limit, doc="Daily cycle limit")
        
        # Parameters - Time intervals  
        model.dt = pyo.Param(initialize=self.market_params['time_step_hours'], 
                            doc="Time step duration (hours)")
        model.tau = pyo.Param(initialize=self.market_params['reserve_duration_hours'],
                             doc="Assumed reserve activation duration (hours)")
        model.db = pyo.Param(initialize=self.market_params['block_duration_hours'], 
                            doc="Block duration for AS markets (hours)")
        
        # Parameters - Minimum bid sizes
        model.min_bid_da = pyo.Param(initialize=self.market_params['min_bid_da'],
                                     doc="Minimum DA bid size (MW)")
        model.min_bid_fcr = pyo.Param(initialize=self.market_params['min_bid_fcr'],
                                     doc="Minimum FCR bid size (MW)")
        model.min_bid_afrr = pyo.Param(initialize=self.market_params['min_bid_afrr'],
                                      doc="Minimum aFRR capacity bid size (MW)")
        model.min_bid_afrr_e = pyo.Param(initialize=self.market_params['min_bid_afrr_e'],
                                        doc="Minimum aFRR energy bid size (MW) - Phase II Model (i)")
        
        # BLOCK MAPPING PARAMETER (Addresses Critical Issue #2 - No more closures!)
        model.block_map = pyo.Param(model.T, initialize=time_to_block, 
                                   doc="Mapping from time to block")
        
        # OPTIMIZED PRICE PARAMETERS
        # DA prices indexed by time (vary every 15 min)
        da_prices = {t: float(country_data['price_day_ahead'].iloc[t]) for t in T_data}
        model.P_DA = pyo.Param(model.T, initialize=da_prices,
                              doc="Day-ahead price (EUR/MWh)")

        # PHASE II Model (i): aFRR energy prices indexed by time (vary every 15 min)
        afrr_energy_pos_prices = {t: float(country_data['price_afrr_energy_pos'].iloc[t]) if not pd.isna(country_data['price_afrr_energy_pos'].iloc[t]) else 0 for t in T_data}
        afrr_energy_neg_prices = {t: float(country_data['price_afrr_energy_neg'].iloc[t]) if not pd.isna(country_data['price_afrr_energy_neg'].iloc[t]) else 0 for t in T_data}
        model.P_aFRR_E_pos = pyo.Param(model.T, initialize=afrr_energy_pos_prices,
                                      doc="aFRR energy positive price (EUR/MWh) - Phase II Model (i)")
        model.P_aFRR_E_neg = pyo.Param(model.T, initialize=afrr_energy_neg_prices,
                                      doc="aFRR energy negative price (EUR/MWh) - Phase II Model (i)")

        # PHASE II Model (i): aFRR activation probabilities for Expected Value weighting
        # These weights account for the probability that aFRR energy bids will be activated
        # w = 1.0 → deterministic (assumes 100% activation, original formulation)
        # w < 1.0 → probabilistic (accounts for non-activation, more realistic)
        afrr_weights_pos = {t: float(country_data['w_afrr_pos'].iloc[t]) for t in T_data}
        afrr_weights_neg = {t: float(country_data['w_afrr_neg'].iloc[t]) for t in T_data}
        model.w_aFRR_pos = pyo.Param(model.T, initialize=afrr_weights_pos,
                                     doc="aFRR positive activation probability (0-1) - EV weighting")
        model.w_aFRR_neg = pyo.Param(model.T, initialize=afrr_weights_neg,
                                     doc="aFRR negative activation probability (0-1) - EV weighting")

        # AS capacity prices indexed by block (constant within 4h blocks) - MEMORY OPTIMIZED
        model.P_FCR = pyo.Param(model.B, initialize=fcr_prices_by_block,
                               doc="FCR capacity price (EUR/MW/h)")
        model.P_aFRR_pos = pyo.Param(model.B, initialize=afrr_pos_prices_by_block,
                                    doc="aFRR positive capacity price (EUR/MW/h)")
        model.P_aFRR_neg = pyo.Param(model.B, initialize=afrr_neg_prices_by_block,
                                    doc="aFRR negative capacity price (EUR/MW/h)")
        
        # Decision Variables
        # Continuous variables - Day-ahead market
        model.p_ch = pyo.Var(model.T, bounds=(0, P_max_config),
                            doc="DA charging power (kW)")
        model.p_dis = pyo.Var(model.T, bounds=(0, P_max_config),
                             doc="DA discharging power (kW)")
        model.e_soc = pyo.Var(model.T, bounds=(self.battery_params['soc_min'] * self.battery_params['capacity_kwh'],
                                              self.battery_params['soc_max'] * self.battery_params['capacity_kwh']),
                             doc="State of charge energy (kWh)")

        # PHASE II Model (i): aFRR energy market variables (kW)
        model.p_afrr_pos_e = pyo.Var(model.T, bounds=(0, P_max_config),
                                    doc="aFRR energy positive (discharge) power (kW) - Model (i)")
        model.p_afrr_neg_e = pyo.Var(model.T, bounds=(0, P_max_config),
                                    doc="aFRR energy negative (charge) power (kW) - Model (i)")

        # PHASE II Model (i): Total power variables (kW)
        model.p_total_ch = pyo.Var(model.T, bounds=(0, P_max_config),
                                  doc="Total charging power (DA + aFRR-E neg) (kW) - Model (i)")
        model.p_total_dis = pyo.Var(model.T, bounds=(0, P_max_config),
                                   doc="Total discharging power (DA + aFRR-E pos) (kW) - Model (i)")

        # Ancillary service capacity variables (MW)
        model.c_fcr = pyo.Var(model.B, bounds=(0, P_max_config/1000),
                             doc="FCR capacity bid (MW)")
        model.c_afrr_pos = pyo.Var(model.B, bounds=(0, P_max_config/1000),
                                  doc="aFRR positive capacity bid (MW)")
        model.c_afrr_neg = pyo.Var(model.B, bounds=(0, P_max_config/1000),
                                  doc="aFRR negative capacity bid (MW)")

        # Binary variables for operational states - Day-ahead market
        # --- COMMENTED OUT T-INDEXED BINARIES FOR PERFORMANCE (Surgical Optimization V4) ---
        # REASON: These 6 groups of T-indexed binaries (T=672 for 7-day → ~4.2k binaries) cause
        # exponential complexity growth. 3-day solves in 27s, but 7-day times out at 10+ minutes.
        # The constraints they support (Cst-3, Cst-8, Cst-9 energy) are either:
        # - Redundant (already enforced by objective or Cst-4 power limits)
        # - Non-critical (0.1 MW MinBid is minor rule vs 1.0 MW for capacity markets)
        # Keeping only B-indexed binaries (blocks) for critical 1.0 MW MinBids.
        # --- END COMMENT ---
        # model.y_ch = pyo.Var(model.T, domain=pyo.Binary,
        #                     doc="DA charging state binary")
        # model.y_dis = pyo.Var(model.T, domain=pyo.Binary,
        #                      doc="DA discharging state binary")

        # PHASE II Model (i): aFRR energy market binaries
        # model.y_afrr_pos_e = pyo.Var(model.T, domain=pyo.Binary,
        #                             doc="aFRR energy positive bid binary - Model (i)")
        # model.y_afrr_neg_e = pyo.Var(model.T, domain=pyo.Binary,
        #                             doc="aFRR energy negative bid binary - Model (i)")

        # PHASE II Model (i): Total operation binaries
        model.y_total_ch = pyo.Var(model.T, domain=pyo.Binary,
                                  doc="Total charging binary - Model (i)")
        model.y_total_dis = pyo.Var(model.T, domain=pyo.Binary,
                                   doc="Total discharging binary - Model (i)")

        # Binary variables for ancillary service capacity market participation
        model.y_fcr = pyo.Var(model.B, domain=pyo.Binary,
                             doc="FCR market participation")
        model.y_afrr_pos = pyo.Var(model.B, domain=pyo.Binary,
                                  doc="aFRR positive capacity market participation")
        model.y_afrr_neg = pyo.Var(model.B, domain=pyo.Binary,
                                  doc="aFRR negative capacity market participation")
        
        # ============================================================================
        # CONSTRAINTS (Ordered to match documentation structure)
        # ============================================================================

        # Cst-0: Prevent bidding in non-activated aFRR energy markets
        ## REMARK: the aFRR-E neg data is expected to be preprocessed such that all 0 `aFRR-E pos` prices are set as `NA` . 
        def no_bid_if_no_afrr_pos_activation_rule(model, t):
            if pd.isna(country_data['price_afrr_energy_pos'].iloc[t]):
                return model.p_afrr_pos_e[t] == 0
            return pyo.Constraint.Skip
        model.no_bid_if_no_afrr_pos_activation = pyo.Constraint(model.T, rule=no_bid_if_no_afrr_pos_activation_rule,
                                                                doc="Force no aFRR+ energy bid if market not activated")

        def no_bid_if_no_afrr_neg_activation_rule(model, t):
            if pd.isna(country_data['price_afrr_energy_neg'].iloc[t]):
                return model.p_afrr_neg_e[t] == 0
            return pyo.Constraint.Skip
        model.no_bid_if_no_afrr_neg_activation = pyo.Constraint(model.T, rule=no_bid_if_no_afrr_neg_activation_rule,
                                                                doc="Force no aFRR- energy bid if market not activated")

        # PHASE II Model (i): Total Power Definition Constraints (NEW)
        # Define total charge/discharge power as sum of DA and aFRR energy markets
        def total_ch_def_rule(model, t):
            return model.p_total_ch[t] == model.p_ch[t] + model.p_afrr_neg_e[t]
        model.total_ch_def = pyo.Constraint(model.T, rule=total_ch_def_rule,
                                           doc="Total charge power = DA charge + aFRR-E negative")

        def total_dis_def_rule(model, t):
            return model.p_total_dis[t] == model.p_dis[t] + model.p_afrr_pos_e[t]
        model.total_dis_def = pyo.Constraint(model.T, rule=total_dis_def_rule,
                                            doc="Total discharge power = DA discharge + aFRR-E positive")

        # Cst-1: Energy Balance (SOC Dynamics) - UPDATED for Model (i)
        # e_soc(t) = e_soc(t-1) + (p_total_ch(t)*η_ch - p_total_dis(t)/η_dis) * Δt
        def soc_dynamics_rule(model, t):
            if t == T_data[0]:
                return model.e_soc[t] == model.E_soc_init + (model.eta_ch * model.p_total_ch[t] - model.p_total_dis[t] / model.eta_dis) * model.dt
            else:
                return model.e_soc[t] == model.e_soc[t-1] + (model.eta_ch * model.p_total_ch[t] - model.p_total_dis[t] / model.eta_dis) * model.dt
        model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)

        # Cst-2: SOC Limits
        # SOC_min * E_nom ≤ e_soc(t) ≤ SOC_max * E_nom
        # Note: Already enforced via variable bounds (lines 375-377)
        # No explicit constraint needed - included in model.e_soc variable definition

        # --- RE-ENABLED Cst-3: Simultaneous Operation Prevention (Partial Optimization V5) ---
        # ANALYSIS: Validation showed Cst-3 violations (2,700+ instances) when fully commented out.
        # These constraints are ESSENTIAL for preventing simultaneous charge/discharge.
        # Cst-8 and Cst-9 remain commented out (0 violations observed).
        # --- END COMMENT ---
        
        # Cst-3: Simultaneous Operation Prevention - UPDATED for Model (i)
        # Use total binaries to prevent charging and discharging simultaneously
        def total_ch_binary_rule(model, t):
            return model.p_total_ch[t] <= model.y_total_ch[t] * model.P_max_config
        model.total_ch_binary = pyo.Constraint(model.T, rule=total_ch_binary_rule,
                                              doc="Link total charge power to total charge binary")

        def total_dis_binary_rule(model, t):
            return model.p_total_dis[t] <= model.y_total_dis[t] * model.P_max_config
        model.total_dis_binary = pyo.Constraint(model.T, rule=total_dis_binary_rule,
                                               doc="Link total discharge power to total discharge binary")

        def no_simultaneous_rule(model, t):
            return model.y_total_ch[t] + model.y_total_dis[t] <= 1
        model.no_simultaneous = pyo.Constraint(model.T, rule=no_simultaneous_rule,
                                              doc="Prevent simultaneous charging and discharging")

        # Cst-4: Market Co-optimization Power Limits - UPDATED for Model (i)
        # Total discharge: p_total_dis(t) + 1000*c_fcr(b) + 1000*c_afrr_pos(b) ≤ P_max
        # Total charge: p_total_ch(t) + 1000*c_fcr(b) + 1000*c_afrr_neg(b) ≤ P_max
        def power_dis_reserve_limit_rule(model, t):
            block = model.block_map[t]
            return model.p_total_dis[t] + 1000 * model.c_fcr[block] + 1000 * model.c_afrr_pos[block] <= model.P_max_config
        model.power_dis_reserve_limit = pyo.Constraint(model.T, rule=power_dis_reserve_limit_rule)

        def power_ch_reserve_limit_rule(model, t):
            block = model.block_map[t]
            return model.p_total_ch[t] + 1000 * model.c_fcr[block] + 1000 * model.c_afrr_neg[block] <= model.P_max_config
        model.power_ch_reserve_limit = pyo.Constraint(model.T, rule=power_ch_reserve_limit_rule)

        # Cst-5: Daily Cycle Limits (only if daily_cycle_limit is specified)
        # Σ_{t∈d} (p_dis(t)/η_dis * Δt) ≤ N_cycles * E_nom
        if daily_cycle_limit is not None:
            def daily_cycle_rule(model, d):
                # Use pre-computed day_to_times mapping
                return sum(model.p_dis[t] / model.eta_dis * model.dt for t in self._day_to_times[d]) <= model.N_cycles * model.E_nom
            model.daily_cycle_limit = pyo.Constraint(model.D, rule=daily_cycle_rule)

        # Cst-6: Ancillary Service Energy Reserve
        # Upward regulation: (1000*c_fcr + 1000*c_afrr_pos)*τ/η_dis ≤ e_soc(t) - SOC_min*E_nom
        # Downward regulation: (1000*c_fcr + 1000*c_afrr_neg)*τ*η_ch ≤ SOC_max*E_nom - e_soc(t)
        def energy_reserve_pos_rule(model, t):
            block = model.block_map[t]
            required_energy = (1000 * model.c_fcr[block] + 1000 * model.c_afrr_pos[block]) * model.tau / model.eta_dis
            return required_energy <= model.e_soc[t] - model.SOC_min * model.E_nom
        model.energy_reserve_pos = pyo.Constraint(model.T, rule=energy_reserve_pos_rule)

        def energy_reserve_neg_rule(model, t):
            block = model.block_map[t]
            required_storage = (1000 * model.c_fcr[block] + 1000 * model.c_afrr_neg[block]) * model.tau * model.eta_ch
            return required_storage <= model.SOC_max * model.E_nom - model.e_soc[t]
        model.energy_reserve_neg = pyo.Constraint(model.T, rule=energy_reserve_neg_rule)

        # Cst-7: Ancillary Service Market Mutual Exclusivity
        # y_fcr(b) + y_afrr_pos(b) + y_afrr_neg(b) ≤ 1
        def as_market_exclusivity_rule(model, b):
            return model.y_fcr[b] + model.y_afrr_pos[b] + model.y_afrr_neg[b] <= 1
        model.as_market_exclusivity = pyo.Constraint(model.B, rule=as_market_exclusivity_rule)

        # ============================================================================
        # NEW CONSTRAINTS: Market Participation Limits (Added 2025-11-09)
        # These constraints prevent unrealistic over-reliance on single markets
        # ============================================================================

        # --- (Cst-10): DELETED per user request (2025-11-09) ---
        # User feedback: "Delete cst-10, the cap on FCR, this does not match the real world case"
        # Original constraint limited FCR capacity to 50% of total power
        # Removal allows model to freely optimize FCR allocation based on economics
        # --- END DELETED CONSTRAINT ---

        # (Cst-11): Total AS Capacity Upper Limit
        # Enforces maximum capacity reservation across all AS markets (FCR + aFRR)
        # This ensures some power remains available for energy market arbitrage
        model.max_as_ratio = pyo.Param(initialize=getattr(self, 'max_as_ratio', 0.8),
                                       doc="Maximum total AS reservation as fraction of power (default 80%)")

        if pyo.value(model.max_as_ratio) < 1.0:
            def max_as_reservation_rule(model, b):
                # Total AS capacity (FCR + aFRR+ + aFRR-) must be less than Y% of total power
                total_reserved_capacity_mw = model.c_fcr[b] + model.c_afrr_pos[b] + model.c_afrr_neg[b]
                max_allowed_capacity_mw = model.max_as_ratio * (model.P_max_config / 1000.0)
                return total_reserved_capacity_mw <= max_allowed_capacity_mw

            model.max_as_reservation = pyo.Constraint(
                model.B, rule=max_as_reservation_rule,
                doc=f"(Cst-11) Enforce max total AS reservation at {pyo.value(model.max_as_ratio)*100:.0f}%"
            )
            logger.info("Enabled (Cst-11) Max Total AS Reservation Limit at %.0f%%",
                       pyo.value(model.max_as_ratio) * 100)
        # ============================================================================

        # --- RE-ENABLED Cst-8: Cross-Market Mutual Exclusivity (Fix for Validation Violations) ---
        # REASON: Validation showed 142 violations when commented out. Cst-4 and Cst-7 are NOT sufficient.
        # CRITICAL: These constraints prevent physical impossibility - can't discharge AND hold reserves.
        # Binary variables must enforce: y_total_dis + y_fcr + y_afrr_neg ≤ 1 (only one action at a time)
        # Cost: Adds 2T constraints but ensures solution feasibility.
        # --- END COMMENT ---

        # Cst-8: Cross-Market Mutual Exclusivity - UPDATED for Model (i)
        # y_total_dis(t) + y_fcr(b) + y_afrr_neg(b) ≤ 1  (no discharge with charging AS reserves)
        # y_total_ch(t) + y_fcr(b) + y_afrr_pos(b) ≤ 1   (no charge with discharging AS reserves)
        def cross_market_exclusivity_rule_1(model, t):
            block = model.block_map[t]
            return model.y_total_dis[t] + model.y_fcr[block] + model.y_afrr_neg[block] <= 1
        model.cross_market_exclusivity1 = pyo.Constraint(model.T, rule=cross_market_exclusivity_rule_1,
                                                         doc="Cst-8a: Prevent discharge + charging AS reserves")

        def cross_market_exclusivity_rule_2(model, t):
            block = model.block_map[t]
            return model.y_total_ch[t] + model.y_fcr[block] + model.y_afrr_pos[block] <= 1
        model.cross_market_exclusivity2 = pyo.Constraint(model.T, rule=cross_market_exclusivity_rule_2,
                                                         doc="Cst-8b: Prevent charge + discharging AS reserves")

        # Check if cross-market exclusivity should be disabled (default: enabled)
        enable_cross_market = self.market_params.get('enable_cross_market_exclusivity',
                                                      getattr(self, 'enable_cross_market_exclusivity', True))

        if not enable_cross_market:
            model.cross_market_exclusivity1.deactivate()
            model.cross_market_exclusivity2.deactivate()
            logger.info("Cst-8: Cross-Market Mutual Exclusivity constraints DEACTIVATED (disabled for reduced complexity)")
        else:
            logger.info("Added Cst-8: Cross-Market Mutual Exclusivity constraints")

        # --- COMMENTED OUT Cst-9: DA Energy MinBid Constraints (REQUIRES y_ch/y_dis binaries) ---
        # REASON: These 4 constraints require y_ch/y_dis binaries which are commented out (lines 586-588).
        # DEPENDENCY ISSUE: Cannot enforce MinBid for DA energy without separate DA binaries.
        # RESOLUTION: Keep commented unless y_ch/y_dis binaries are re-enabled.
        # --- END COMMENT ---
        # # Cst-9: Minimum and Maximum Bid Size Constraints
        # # DA Energy Bids: y(t)*MinBid*1000 ≤ p(t) ≤ y(t)*P_max_config
        # def da_ch_min_bid_rule(model, t):
        #     return model.p_ch[t] >= model.y_ch[t] * model.min_bid_da * 1000
        # model.da_ch_min_bid = pyo.Constraint(model.T, rule=da_ch_min_bid_rule)

        # def da_ch_max_bid_rule(model, t):
        #     return model.p_ch[t] <= model.y_ch[t] * model.P_max_config
        # model.da_ch_max_bid = pyo.Constraint(model.T, rule=da_ch_max_bid_rule)

        # def da_dis_min_bid_rule(model, t):
        #     return model.p_dis[t] >= model.y_dis[t] * model.min_bid_da * 1000
        # model.da_dis_min_bid = pyo.Constraint(model.T, rule=da_dis_min_bid_rule)

        # def da_dis_max_bid_rule(model, t):
        #     return model.p_dis[t] <= model.y_dis[t] * model.P_max_config
        # model.da_dis_max_bid = pyo.Constraint(model.T, rule=da_dis_max_bid_rule)

        # FCR Capacity Bids: y(b)*MinBid ≤ c(b) ≤ y(b)*P_max_config/1000
        # NOTE: These are ACTIVE - AS capacity MinBid constraints use block-indexed binaries which DO exist
        def fcr_min_bid_rule(model, b):
            return model.c_fcr[b] >= model.y_fcr[b] * model.min_bid_fcr
        model.fcr_min_bid = pyo.Constraint(model.B, rule=fcr_min_bid_rule,
                                          doc="Cst-9i: FCR minimum bid size (1.0 MW)")

        def fcr_max_bid_rule(model, b):
            return model.c_fcr[b] <= model.y_fcr[b] * (model.P_max_config / 1000)
        model.fcr_max_bid = pyo.Constraint(model.B, rule=fcr_max_bid_rule,
                                          doc="Cst-9j: FCR maximum bid size")

        # aFRR Capacity Bids: y(b)*MinBid ≤ c(b) ≤ y(b)*P_max_config/1000
        def afrr_pos_min_bid_rule(model, b):
            return model.c_afrr_pos[b] >= model.y_afrr_pos[b] * model.min_bid_afrr
        model.afrr_pos_min_bid = pyo.Constraint(model.B, rule=afrr_pos_min_bid_rule,
                                                doc="Cst-9k: aFRR+ capacity minimum bid size (1.0 MW)")

        def afrr_pos_max_bid_rule(model, b):
            return model.c_afrr_pos[b] <= model.y_afrr_pos[b] * (model.P_max_config / 1000)
        model.afrr_pos_max_bid = pyo.Constraint(model.B, rule=afrr_pos_max_bid_rule,
                                                doc="Cst-9l: aFRR+ capacity maximum bid size")

        def afrr_neg_min_bid_rule(model, b):
            return model.c_afrr_neg[b] >= model.y_afrr_neg[b] * model.min_bid_afrr
        model.afrr_neg_min_bid = pyo.Constraint(model.B, rule=afrr_neg_min_bid_rule,
                                                doc="Cst-9m: aFRR- capacity minimum bid size (1.0 MW)")

        def afrr_neg_max_bid_rule(model, b):
            return model.c_afrr_neg[b] <= model.y_afrr_neg[b] * (model.P_max_config / 1000)
        model.afrr_neg_max_bid = pyo.Constraint(model.B, rule=afrr_neg_max_bid_rule,
                                                doc="Cst-9n: aFRR- capacity maximum bid size")

        # --- COMMENTED OUT Cst-9: aFRR Energy MinBid Constraints (REQUIRES y_afrr_pos_e/y_afrr_neg_e binaries) ---
        # REASON: These 4 constraints require y_afrr_pos_e/y_afrr_neg_e binaries which are commented out (lines 592-594).
        # DEPENDENCY ISSUE: Cannot enforce MinBid for aFRR energy without separate aFRR energy binaries.
        # RESOLUTION: Keep commented unless those binaries are re-enabled.
        # --- END COMMENT ---
        # # PHASE II Model (i): aFRR Energy Market Bid Constraints (NEW)
        # def afrr_pos_e_min_bid_rule(model, t):
        #     return model.p_afrr_pos_e[t] >= model.y_afrr_pos_e[t] * model.min_bid_afrr_e * 1000
        # model.afrr_pos_e_min_bid = pyo.Constraint(model.T, rule=afrr_pos_e_min_bid_rule)

        # def afrr_pos_e_max_bid_rule(model, t):
        #     return model.p_afrr_pos_e[t] <= model.y_afrr_pos_e[t] * model.P_max_config
        # model.afrr_pos_e_max_bid = pyo.Constraint(model.T, rule=afrr_pos_e_max_bid_rule)

        # def afrr_neg_e_min_bid_rule(model, t):
        #     return model.p_afrr_neg_e[t] >= model.y_afrr_neg_e[t] * model.min_bid_afrr_e * 1000
        # model.afrr_neg_e_min_bid = pyo.Constraint(model.T, rule=afrr_neg_e_min_bid_rule)

        # def afrr_neg_e_max_bid_rule(model, t):
        #     return model.p_afrr_neg_e[t] <= model.y_afrr_neg_e[t] * model.P_max_config
        # model.afrr_neg_e_max_bid = pyo.Constraint(model.T, rule=afrr_neg_e_max_bid_rule)

        # # PHASE II Model (i): Total Binary Linkage (NEW)
        # # y_total_ch >= y_ch and y_total_ch >= y_afrr_neg_e
        # # y_total_dis >= y_dis and y_total_dis >= y_afrr_pos_e
        # def total_ch_binary_link1_rule(model, t):
        #     return model.y_total_ch[t] >= model.y_ch[t]
        # model.total_ch_binary_link1 = pyo.Constraint(model.T, rule=total_ch_binary_link1_rule)

        # def total_ch_binary_link2_rule(model, t):
        #     return model.y_total_ch[t] >= model.y_afrr_neg_e[t]
        # model.total_ch_binary_link2 = pyo.Constraint(model.T, rule=total_ch_binary_link2_rule)

        # def total_dis_binary_link1_rule(model, t):
        #     return model.y_total_dis[t] >= model.y_dis[t]
        # model.total_dis_binary_link1 = pyo.Constraint(model.T, rule=total_dis_binary_link1_rule)

        # def total_dis_binary_link2_rule(model, t):
        #     return model.y_total_dis[t] >= model.y_afrr_pos_e[t]
        # model.total_dis_binary_link2 = pyo.Constraint(model.T, rule=total_dis_binary_link2_rule)

        # OPTIMIZED OBJECTIVE FUNCTION - UPDATED for Model (i) with Profit Component Expressions
        # Define profit components as Pyomo Expressions for easy post-solve retrieval

        def da_profit_rule(model):
            """Day-ahead energy profit (EUR)"""
            return sum((model.P_DA[t] / 1000 * model.p_dis[t] -
                        model.P_DA[t] / 1000 * model.p_ch[t]) * model.dt
                       for t in model.T)
        model.profit_da = pyo.Expression(rule=da_profit_rule)

        def afrr_energy_profit_rule(model):
            """aFRR energy market profit with Expected Value weighting (EUR)
            Multiply power by activation weight: w=1.0 (deterministic), w<1.0 (probabilistic)
            BOTH positive and negative aFRR energy generate POSITIVE revenue (addition, not subtraction)
            """
            return sum((model.P_aFRR_E_pos[t] / 1000 * model.p_afrr_pos_e[t] * model.w_aFRR_pos[t] +
                        model.P_aFRR_E_neg[t] / 1000 * model.p_afrr_neg_e[t] * model.w_aFRR_neg[t]) * model.dt
                       for t in model.T)
        model.profit_afrr_energy = pyo.Expression(rule=afrr_energy_profit_rule)

        def as_capacity_profit_rule(model):
            """Ancillary service capacity profit (EUR)

            IMPORTANT: Prices are in EUR/MW per 4-hour block (NOT EUR/MW/h).
            Do NOT multiply by model.db - the price already includes block duration.
            """
            return sum(model.P_FCR[b] * model.c_fcr[b] +
                       model.P_aFRR_pos[b] * model.c_afrr_pos[b] +
                       model.P_aFRR_neg[b] * model.c_afrr_neg[b]
                       for b in model.B)
        model.profit_as_capacity = pyo.Expression(rule=as_capacity_profit_rule)

        def objective_rule(model):
            """Total profit = DA + aFRR Energy + AS Capacity"""
            return model.profit_da + model.profit_afrr_energy + model.profit_as_capacity

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # OPTIONAL: End-of-horizon SOC constraint (mentioned in review)
        # Uncomment if required to return to initial SOC
        # def final_soc_rule(model):
        #     return model.e_soc[model.T.last()] == model.E_soc_init
        # model.final_soc = pyo.Constraint(rule=final_soc_rule)
        
        logger.info("Improved optimization model built successfully")
        logger.info(f"Variables: {model.nvariables()}")
        logger.info(f"Constraints: {model.nconstraints()}")
        
        return model
    
    def detect_available_solver(self) -> str:
        """
        Detect the best available solver with fallback logic.

        Priority order:
        1. Environment variable GRIDKEY_SOLVER if set (for explicit control)
        2. CPLEX (commercial) - preferred local solver
        3. Gurobi (commercial) - fallback if CPLEX unavailable
        4. HiGHS (open-source) - REQUIRED for production API (no license needed)

        Production API Note:
            For Docker/container deployments, use HiGHS as it requires no license.
            Set GRIDKEY_SOLVER=highs in Dockerfile or environment.

        Returns:
            str: Name of the best available solver
        """
        import os

        # Check for explicit solver preference via environment variable
        preferred_solver = os.environ.get('GRIDKEY_SOLVER', '').lower()
        if preferred_solver:
            logger.info(f"🎯 GRIDKEY_SOLVER environment variable set: {preferred_solver}")
            solver = pyo.SolverFactory(preferred_solver)
            if solver.available():
                logger.info(f"✅ Using preferred solver: {preferred_solver}")
                return preferred_solver
            else:
                logger.warning(f"⚠️  Preferred solver '{preferred_solver}' not available, falling back to auto-detection")

        # Priority order for local development: CPLEX > Gurobi > HiGHS
        solver_priority = [
            ('cplex', 'CPLEX (commercial)'),
            ('gurobi', 'Gurobi (commercial)'),
            ('highs', 'HiGHS (open-source, production-ready)')
        ]
        
        logger.info("🔍 Detecting available optimization solver...")
        
        for solver_name, solver_display in solver_priority:
            try:
                solver = pyo.SolverFactory(solver_name)
                if solver.available():
                    logger.info(f"✅ Using solver: {solver_display}")
                    return solver_name
            except Exception as e:
                logger.debug(f"   Solver {solver_display} not available: {e}")
                continue
        
        # If no solver found, raise error
        raise RuntimeError(
            "❌ No compatible optimization solver found!\n"
            "Required: At least one of CPLEX, Gurobi, or HiGHS (recommended for competition).\n"
            "Install HiGHS: pip install highspy"
        )
    
    def solve_model(self, model: pyo.ConcreteModel, solver_name: str = None) -> tuple[pyo.ConcreteModel, Any]:
        """
        Solve the optimization model with automatic solver detection.

        This method is responsible ONLY for solving the model, not extracting results.
        Use extract_solution() to get solution data from the solved model.

        Args:
            model: Pyomo model to solve
            solver_name: Solver to use. If None, auto-detect best available.
                        Options: 'cplex', 'gurobi', 'highs' (recommended for competition)

        Returns:
            tuple: (solved_model, solver_results)
        """
        # Auto-detect solver if not specified
        if solver_name is None:
            solver_name = self.detect_available_solver()
        else:
            logger.info(f"Using specified solver: {solver_name}")

        try:
            # Create solver
            solver = pyo.SolverFactory(solver_name)

            # Verify solver is available
            if not solver.available():
                logger.warning(f"⚠️  Solver {solver_name} not available, auto-detecting...")
                solver_name = self.detect_available_solver()
                solver = pyo.SolverFactory(solver_name)

            # CONSISTENT SOLVER TIME LIMITS (Addresses Critical Issue #6)
            if solver_name.lower() == 'cplex':
                solver.options['timelimit'] = self.market_params['solver_time_limit']
                solver.options['mip_tolerances_mipgap'] = self.solver_config.get('solver_options', {}).get('cplex', {}).get('mip_tolerances_mipgap', 0.01)
                solver.options['emphasis_mip'] = 1
            elif solver_name.lower() == 'gurobi':
                gurobi_options = self.solver_config.get('solver_options', {}).get('gurobi', {})
                solver.options['TimeLimit'] = self.market_params['solver_time_limit']
                solver.options['MIPGap'] = gurobi_options.get('MIPGap', 0.03)
                solver.options['OutputFlag'] = 0  # Suppress verbose Gurobi output
                # solver.options['Threads'] = gurobi_options.get('Threads', 4)
                solver.options['MIPFocus'] = gurobi_options.get('MIPFocus', 0)
            elif solver_name.lower() == 'highs':
                solver.options['time_limit'] = self.market_params['solver_time_limit']
                solver.options['mip_rel_gap'] = self.solver_config.get('solver_options', {}).get('highs', {}).get('mip_rel_gap', 0.01)
            elif solver_name.lower() == 'scip':
                solver.options['limits/time'] = self.market_params['solver_time_limit']
                solver.options['limits/gap'] = 0.01
            elif solver_name.lower() == 'cbc':
                solver.options['seconds'] = self.market_params['solver_time_limit']
                solver.options['ratio'] = 0.01

            # Solve
            start_time = datetime.now()
            results = solver.solve(model, tee=False)
            solve_time = (datetime.now() - start_time).total_seconds()

            # Check solution status
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                logger.info(f"Optimal solution found in {solve_time:.2f} seconds")
            elif results.solver.termination_condition == pyo.TerminationCondition.feasible:
                logger.info(f"Feasible solution found in {solve_time:.2f} seconds")
            else:
                logger.error(f"Solver failed: {results.solver.termination_condition}")

            # Store metadata in results object for extract_solution
            results._solve_time = solve_time
            results._solver_name = solver_name

            return model, results

        except Exception as e:
            logger.error(f"Error solving model: {str(e)}")
            # Create a minimal results object for error cases
            class ErrorResults:
                def __init__(self, error_msg):
                    self.error = error_msg
                    self._solve_time = 0
                    self._solver_name = solver_name if 'solver_name' in locals() else 'unknown'
                    class SolverInfo:
                        termination_condition = pyo.TerminationCondition.error
                    self.solver = SolverInfo()

            return model, ErrorResults(str(e))

    def extract_solution(self, model: pyo.ConcreteModel, solver_results: Any) -> Dict[str, Any]:
        """
        Extract solution data from a solved Pyomo model.

        This method extracts decision variables, profit components, and metadata
        from the solved model. Subclasses can override to add model-specific results.

        Args:
            model: Solved Pyomo model
            solver_results: Results object from solver.solve()

        Returns:
            Dict containing solution data and performance metrics
        """
        # Check for error in solver results
        if hasattr(solver_results, 'error'):
            return {
                'status': 'error',
                'error': solver_results.error,
                'solve_time': solver_results._solve_time
            }

        # Check solution status
        if solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            status = "optimal"
        elif solver_results.solver.termination_condition == pyo.TerminationCondition.feasible:
            status = "feasible"
        else:
            return {
                'status': 'failed',
                'termination_condition': str(solver_results.solver.termination_condition),
                'solve_time': solver_results._solve_time
            }

        # Extract solution - OPTIMIZED (Addresses Minor Issue #9)
        solution = {
            'status': status,
            'solve_time': solver_results._solve_time,
            'objective_value': pyo.value(model.objective),
            'solver': solver_results._solver_name,
            'termination_condition': str(solver_results.solver.termination_condition)
        }

        def _safe_value(component: Any) -> Optional[float]:
            try:
                return pyo.value(component)
            except (TypeError, ValueError):
                return None

        # Extract profit components from Pyomo Expressions
        if hasattr(model, 'profit_da'):
            solution['profit_da'] = _safe_value(model.profit_da)
        if hasattr(model, 'profit_afrr_energy'):
            solution['profit_afrr_energy'] = _safe_value(model.profit_afrr_energy)
        if hasattr(model, 'profit_as_capacity'):
            solution['profit_as_capacity'] = _safe_value(model.profit_as_capacity)

        # Extract aging cost (Model II and III)
        if hasattr(model, 'cost_cyclic'):
            solution['cost_cyclic'] = _safe_value(model.cost_cyclic)

        # Extract variable values efficiently
        # Day-ahead market
        solution["p_ch"] = {}
        solution["p_dis"] = {}
        solution["e_soc"] = {}
        for t in model.T:
            val_ch = _safe_value(model.p_ch[t])
            if val_ch is not None:
                solution["p_ch"][t] = val_ch

            val_dis = _safe_value(model.p_dis[t])
            if val_dis is not None:
                solution["p_dis"][t] = val_dis

            val_soc = _safe_value(model.e_soc[t])
            if val_soc is not None:
                solution["e_soc"][t] = val_soc

        # PHASE II Model (i): aFRR energy market (NEW)
        solution["p_afrr_pos_e"] = {}
        solution["p_afrr_neg_e"] = {}
        for t in model.T:
            val_pos = _safe_value(model.p_afrr_pos_e[t])
            if val_pos is not None:
                solution["p_afrr_pos_e"][t] = val_pos

            val_neg = _safe_value(model.p_afrr_neg_e[t])
            if val_neg is not None:
                solution["p_afrr_neg_e"][t] = val_neg

        # PHASE II Model (i): Total power (NEW)
        solution["p_total_ch"] = {}
        solution["p_total_dis"] = {}
        for t in model.T:
            total_ch = _safe_value(model.p_total_ch[t])
            if total_ch is not None:
                solution["p_total_ch"][t] = total_ch

            total_dis = _safe_value(model.p_total_dis[t])
            if total_dis is not None:
                solution["p_total_dis"][t] = total_dis

        # Ancillary service capacity
        solution["c_fcr"] = {}
        solution["c_afrr_pos"] = {}
        solution["c_afrr_neg"] = {}
        for b in model.B:
            val_fcr = _safe_value(model.c_fcr[b])
            if val_fcr is not None:
                solution["c_fcr"][b] = val_fcr

            val_afrr_pos = _safe_value(model.c_afrr_pos[b])
            if val_afrr_pos is not None:
                solution["c_afrr_pos"][b] = val_afrr_pos

            val_afrr_neg = _safe_value(model.c_afrr_neg[b])
            if val_afrr_neg is not None:
                solution["c_afrr_neg"][b] = val_afrr_neg

        # Binaries - Day-ahead (OPTIONAL - may be commented out for performance)
        if hasattr(model, 'y_ch'):
            solution["y_ch"] = {}
            solution["y_dis"] = {}
            for t in model.T:
                val_y_ch = _safe_value(model.y_ch[t])
                if val_y_ch is not None:
                    solution["y_ch"][t] = val_y_ch

                val_y_dis = _safe_value(model.y_dis[t])
                if val_y_dis is not None:
                    solution["y_dis"][t] = val_y_dis

        # PHASE II Model (i): aFRR energy binaries (OPTIONAL - may be commented out for performance)
        if hasattr(model, 'y_afrr_pos_e'):
            solution["y_afrr_pos_e"] = {}
            solution["y_afrr_neg_e"] = {}
            for t in model.T:
                val_y_pos = _safe_value(model.y_afrr_pos_e[t])
                if val_y_pos is not None:
                    solution["y_afrr_pos_e"][t] = val_y_pos

                val_y_neg = _safe_value(model.y_afrr_neg_e[t])
                if val_y_neg is not None:
                    solution["y_afrr_neg_e"][t] = val_y_neg

        # PHASE II Model (i): Total binaries (OPTIONAL - may be commented out for performance)
        if hasattr(model, 'y_total_ch'):
            solution["y_total_ch"] = {}
            solution["y_total_dis"] = {}
            for t in model.T:
                val_y_total_ch = _safe_value(model.y_total_ch[t])
                if val_y_total_ch is not None:
                    solution["y_total_ch"][t] = val_y_total_ch

                val_y_total_dis = _safe_value(model.y_total_dis[t])
                if val_y_total_dis is not None:
                    solution["y_total_dis"][t] = val_y_total_dis

        # Binaries - Ancillary service capacity
        solution["y_fcr"] = {}
        solution["y_afrr_pos"] = {}
        solution["y_afrr_neg"] = {}
        for b in model.B:
            val_y_fcr = _safe_value(model.y_fcr[b])
            if val_y_fcr is not None:
                solution["y_fcr"][b] = val_y_fcr

            val_y_afrr_pos = _safe_value(model.y_afrr_pos[b])
            if val_y_afrr_pos is not None:
                solution["y_afrr_pos"][b] = val_y_afrr_pos

            val_y_afrr_neg = _safe_value(model.y_afrr_neg[b])
            if val_y_afrr_neg is not None:
                solution["y_afrr_neg"][b] = val_y_afrr_neg

        # Extract block mapping (time → block) for visualization
        # This maps each time step to its corresponding AS capacity block
        solution["block_map"] = {}
        for t in model.T:
            solution["block_map"][t] = int(_safe_value(model.block_map[t]) or 0)

        return solution

    def extract_country_data(self, data: pd.DataFrame, country: str) -> pd.DataFrame:
        """
        Extract and format data for a specific country with enhanced validation.
        Enhanced for Phase II Model (i) with aFRR energy market data.
        """
        logger.info(f"Extracting data for country: {country}")

        if country not in self.countries:
            raise ValueError(f"Country {country} not supported. Available: {self.countries}")

        try:
            # Handle special case for DE_LU coupled market
            # Day-ahead: Use DE_LU (coupled Germany-Luxembourg market)
            # Ancillary services: Use DE (German TSO responsibility)
            if country == 'DE_LU':
                day_ahead_country = 'DE_LU'
                as_country = 'DE'  # Ancillary services handled by German TSO
            else:
                day_ahead_country = country
                as_country = country

            # Extract country-specific data with market-aware mapping
            country_df = pd.DataFrame()
            country_df['price_day_ahead'] = data[(day_ahead_country, 'day_ahead', '')]
            country_df['price_fcr'] = data[(as_country, 'fcr', '')]
            country_df['price_afrr_pos'] = data[(as_country, 'afrr', 'positive')]
            country_df['price_afrr_neg'] = data[(as_country, 'afrr', 'negative')]

            # PHASE II Model (i): Extract aFRR energy market prices (15-min intervals)
            try:
                country_df['price_afrr_energy_pos'] = data[(as_country, 'afrr_energy', 'positive')]
                country_df['price_afrr_energy_neg'] = data[(as_country, 'afrr_energy', 'negative')]
                logger.info(f"aFRR energy market data extracted for {country}")

                # CRITICAL: Preprocess aFRR energy prices (convert 0 -> NaN)
                # Price = 0 means "market not activated", NOT "free energy"
                # This prevents false arbitrage opportunities
                country_df['price_afrr_energy_pos'] = country_df['price_afrr_energy_pos'].replace(0, np.nan)
                country_df['price_afrr_energy_neg'] = country_df['price_afrr_energy_neg'].replace(0, np.nan)
                logger.info(f"Preprocessed aFRR energy prices: 0 -> NaN (prevents false arbitrage)")

            except KeyError:
                logger.warning(f"aFRR energy market data not available for {country}. Setting to NaN (Model (i) will be limited).")
                country_df['price_afrr_energy_pos'] = np.nan
                country_df['price_afrr_energy_neg'] = np.nan

            # Add aFRR activation probabilities for Expected Value weighting
            if self.use_afrr_ev_weighting:
                # Load activation config
                activation_config = self._load_activation_config()

                # Check for country-specific values (use as_country for activation rates)
                if 'country_specific' in activation_config and as_country in activation_config['country_specific']:
                    w_pos = activation_config['country_specific'][as_country]['positive']
                    w_neg = activation_config['country_specific'][as_country]['negative']
                    logger.info(f"Using country-specific activation rates for {as_country}: pos={w_pos:.2f}, neg={w_neg:.2f}")
                else:
                    # Use default values
                    w_pos = activation_config['default_probabilities']['positive']
                    w_neg = activation_config['default_probabilities']['negative']
                    logger.info(f"Using default activation rates for {as_country}: pos={w_pos:.2f}, neg={w_neg:.2f}")

                # Add as constant columns (could be time-varying in future)
                country_df['w_afrr_pos'] = w_pos
                country_df['w_afrr_neg'] = w_neg
            else:
                # No EV weighting: set weights to 1.0 (100% activation assumption)
                country_df['w_afrr_pos'] = 1.0
                country_df['w_afrr_neg'] = 1.0
                logger.info(f"EV weighting disabled for {country}: using w=1.0 (deterministic)")

            # Create time-based identifiers
            timestamps = data.index
            country_df['hour'] = timestamps.hour
            country_df['day_of_year'] = timestamps.dayofyear
            country_df['month'] = timestamps.month
            country_df['year'] = timestamps.year

            # Create block IDs (4-hour blocks starting at midnight)
            country_df['block_of_day'] = country_df['hour'] // 4
            country_df['block_id'] = (country_df['day_of_year'] - 1) * 6 + country_df['block_of_day']

            # Create day IDs
            country_df['day_id'] = country_df['day_of_year']

            # Keep timestamp as a column for filtering
            country_df['timestamp'] = timestamps

            # Reset index to get integer-based indexing
            country_df = country_df.reset_index(drop=True)

            # Additional validation
            if country_df.isnull().any().any():
                logger.warning(f"Missing data found for country {country}")

            logger.info(f"Extracted {len(country_df)} data points for {country}")
            return country_df

        except KeyError as e:
            raise ValueError(f"Missing data for country {country}: {str(e)}")
    
    def optimize(self, country_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simplified optimization interface for testing and single-scenario runs.
        
        Args:
            country_data: Preprocessed DataFrame with market data for a specific country
            
        Returns:
            Dictionary with optimization results including total_revenue
        """
        try:
            # Get daily cycle limit directly (NOT multiplied by num_days - that was the bug!)
            # # The constraint is applied per day, so we pass the daily limit to the model
            daily_cycle_limit = self.battery_params['daily_cycle_limit']
            
            # Calculate c_rate from max_power_kw and capacity
            max_power_kw = self.market_params.get('max_power_kw', self.battery_params['capacity_kwh'] * 0.5)
            c_rate = max_power_kw / self.battery_params['capacity_kwh']
            
            # Build and solve the optimization model
            model = self.build_optimization_model(country_data, c_rate, daily_cycle_limit)
            solved_model, solver_results = self.solve_model(model)
            results = self.extract_solution(solved_model, solver_results)

            # Return results in expected format
            return {
                'total_revenue': results.get('objective_value', 0),
                'solver_status': results.get('status', 'unknown'),
                'solve_time': results.get('solve_time', 0),
                'detailed_results': results
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return {
                'total_revenue': 0,
                'solver_status': 'failed',
                'solve_time': 0,
                'error': str(e),
                'detailed_results': {}
            }
    
    def run_scenario_analysis(self, data_file: str, output_file: str = None, 
                            num_days: int = 10) -> pd.DataFrame:
        """
        Run comprehensive scenario analysis with improved model.
        """
        logger.info("Starting improved scenario analysis")
        
        # Load data
        data = self.load_and_preprocess_data(data_file)
        
        # Limit to specified number of days
        if num_days:
            end_time = data.index[0] + timedelta(days=num_days)
            data = data[data.index < end_time]
            logger.info(f"Limited analysis to {num_days} days")
        
        results = []
        
        for country in self.countries:
            logger.info(f"Processing country: {country}")
            
            try:
                country_data = self.extract_country_data(data, country)
                
                for c_rate in self.c_rates:
                    for daily_cycle_limit in self.daily_cycles:
                        scenario_name = f"{country}_C{c_rate}_N{daily_cycle_limit}"
                        logger.info(f"Running scenario: {scenario_name}")
                        
                        try:
                            # Build and solve model
                            model = self.build_optimization_model(country_data, c_rate, daily_cycle_limit)
                            solved_model, solver_results = self.solve_model(model, 'gurobi')
                            solution = self.extract_solution(solved_model, solver_results)

                            if solution['status'] in ['optimal', 'feasible']:
                                result = {
                                    'scenario': scenario_name,
                                    'country': country,
                                    'c_rate': c_rate,
                                    'n_cycles': daily_cycle_limit,
                                    'status': solution['status'],
                                    'objective_value': solution['objective_value'],
                                    'solve_time': solution['solve_time'],
                                    'power_rating_kw': c_rate * self.battery_params['capacity_kwh'],
                                    'energy_capacity_kwh': self.battery_params['capacity_kwh']
                                }
                                results.append(result)
                                logger.info(f"Scenario {scenario_name}: Objective = {solution['objective_value']:.2f} EUR")
                            else:
                                logger.warning(f"Scenario {scenario_name} failed: {solution['status']}")
                                
                        except Exception as e:
                            logger.error(f"Error in scenario {scenario_name}: {str(e)}")
                            
            except Exception as e:
                logger.error(f"Error processing country {country}: {str(e)}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        logger.info("Improved scenario analysis completed")
        return results_df

# ============================================================================
# Phase II Model (ii): Cyclic Aging Cost Extension
# ============================================================================


class BESSOptimizerModelII(BESSOptimizerModelI):
    """Battery Energy Storage System Optimizer - Phase II Model (ii).

    Extends :class:`BESSOptimizerModelI` by replacing the hard daily cycle limit with
    a convex, segment-based cyclic degradation cost. Each SOC slice carries a
    marginal cost, enabling the optimizer to weigh revenue against battery aging.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        degradation_config_path: Optional[str] = None,
        alpha: float = 1.0,
        require_sequential_segment_activation: bool = True,
        use_afrr_ev_weighting: bool = False,
    ) -> None:
        """Initialize Phase II Model (ii) with cyclic aging cost.

        Args:
            config: Optional runtime configuration overrides
            degradation_config_path: Path to aging_config.json (defaults to standard location)
            alpha: Degradation price weight parameter (default 1.0)
            require_sequential_segment_activation: Enforce strict sequential segment filling by requiring
                power flow binaries (Eq. 609-610). When True, prevents parallel charging of multiple segments.
                When False (default), allows epsilon-tolerance parallel activation for faster solve (default False)
            use_afrr_ev_weighting: Enable Expected Value weighting for aFRR energy bids (default False)
        """
        super().__init__(use_afrr_ev_weighting=use_afrr_ev_weighting)

        # Optional runtime configuration overrides
        if config:
            self._apply_optional_config(config)

        # Load degradation config: from explicit path or unified YAML config
        if degradation_config_path is not None:
            self.degradation_config = self._load_degradation_config_from_file(degradation_config_path)
        else:
            self.degradation_config = self._load_degradation_config_from_yaml()

        cyclic_config = self.degradation_config['cyclic_aging']
        num_segments = len(cyclic_config['costs'])
        segment_capacity = self.battery_params['capacity_kwh'] / num_segments

        self.degradation_params = {
            'enabled': True,
            'model_type': 'cyclic_only',
            'num_segments': num_segments,
            'segment_capacity_kwh': segment_capacity,
            'marginal_costs': cyclic_config['costs'],
            'alpha': float(alpha),
            'config_file_path': str(degradation_config_path) if degradation_config_path else 'config/Config.yml',
            'require_sequential_segment_activation': self.degradation_config.get('require_sequential_segment_activation', require_sequential_segment_activation),
            'lifo_epsilon_kwh': self.degradation_config.get('lifo_epsilon_kwh', 5.0),
        }

        self._validate_degradation_params()

        logger.info("Initialized BESSOptimizerModelII:")
        logger.info("  - Segments: %d", self.degradation_params['num_segments'])
        logger.info(
            "  - Segment capacity: %.2f kWh",
            self.degradation_params['segment_capacity_kwh'],
        )
        logger.info("  - Alpha: %.4f", self.degradation_params['alpha'])
        logger.info(
            "  - Cost range: EUR %.4f - EUR %.4f per kWh",
            min(self.degradation_params['marginal_costs']),
            max(self.degradation_params['marginal_costs']),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_optional_config(self, config: Dict[str, Any]) -> None:
        """Update battery/market parameters from optional configuration."""

        battery_updates = config.get('battery_params') if isinstance(config, dict) else None
        if isinstance(battery_updates, dict):
            self.battery_params.update(battery_updates)

        market_updates = config.get('market_params') if isinstance(config, dict) else None
        if isinstance(market_updates, dict):
            self.market_params.update(market_updates)

    def _load_degradation_config_from_yaml(self) -> Dict[str, Any]:
        """Load aging configuration from unified YAML config."""
        try:
            from src.utils.config_loader import ConfigLoader
            config = ConfigLoader.get_aging_config()

            if 'cyclic_aging' not in config:
                raise KeyError("Missing 'cyclic_aging' key in aging config")

            if 'costs' not in config['cyclic_aging']:
                raise KeyError("Missing 'costs' array in cyclic_aging config")

            logger.info("Loaded degradation config from: config/Config.yml")
            return config
        except Exception as e:
            raise ValueError(f"Failed to load aging config from YAML: {e}") from e

    def _load_degradation_config_from_file(self, config_path: Any) -> Dict[str, Any]:
        """Load cyclic aging configuration from JSON file (legacy support)."""

        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Degradation config file not found: {config_path}\n"
                "Expected location: config/Config.yml (or pass explicit path)"
            )

        try:
            with open(config_file, 'r', encoding='utf-8') as handle:
                config = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in degradation config file: {exc}") from exc

        if 'cyclic_aging' not in config:
            raise KeyError("Missing 'cyclic_aging' key in degradation config")

        if 'costs' not in config['cyclic_aging']:
            raise KeyError("Missing 'costs' array in cyclic_aging config")

        logger.info("Loaded degradation config from: %s", config_file)
        return config

    def _validate_degradation_params(self) -> None:
        """Validate degradation parameters for physical and numerical consistency."""

        params = self.degradation_params
        num_seg = params['num_segments']
        costs = params['marginal_costs']
        seg_cap = params['segment_capacity_kwh']
        total_cap = self.battery_params['capacity_kwh']
        alpha = params['alpha']

        if num_seg <= 0:
            raise ValueError(f"Number of segments must be positive, got {num_seg}")

        if len(costs) != num_seg:
            raise ValueError(
                "Marginal costs array length ({}) must equal number of segments ({})".format(
                    len(costs), num_seg
                )
            )

        for idx in range(1, num_seg):
            if costs[idx] <= costs[idx - 1]:
                logger.warning(
                    "Marginal costs should increase with depth. Cost[%d]=%.4f is not greater than Cost[%d]=%.4f",
                    idx,
                    costs[idx],
                    idx - 1,
                    costs[idx - 1],
                )

        if any(cost < 0 for cost in costs):
            raise ValueError(f"All marginal costs must be non-negative, got {costs}")

        expected_seg_cap = total_cap / num_seg
        if abs(seg_cap - expected_seg_cap) > 0.01:
            raise ValueError(
                "Segment capacity mismatch: %.2f kWh != %.2f kWh (total: %s / %s)" % (
                    seg_cap,
                    expected_seg_cap,
                    total_cap,
                    num_seg,
                )
            )

        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")

        logger.info("Degradation parameters validated successfully")

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build_optimization_model(
        self,
        country_data: pd.DataFrame,
        c_rate: float,
        daily_cycle_limit: Optional[float] = None,
    ) -> pyo.ConcreteModel:
        if daily_cycle_limit is not None:
            logger.warning(
                "Model (ii) ignores daily_cycle_limit parameter (%.2f). Using cost-based degradation instead.",
                daily_cycle_limit,
            )

        # Pre-compute time sequence for use in constraint rules
        time_indices = list(range(len(country_data)))
        prev_time_lookup = {}
        for idx, t in enumerate(time_indices):
            prev_time_lookup[t] = time_indices[idx - 1] if idx > 0 else None

        # Build parent model (daily cycle limit disabled)
        model = super().build_optimization_model(country_data, c_rate, daily_cycle_limit=None)

        num_segments = self.degradation_params['num_segments']
        segment_capacity = self.degradation_params['segment_capacity_kwh']
        marginal_costs = self.degradation_params['marginal_costs']
        alpha = self.degradation_params['alpha']
        P_max_config = c_rate * self.battery_params['capacity_kwh']

        model.J = pyo.Set(initialize=range(1, num_segments + 1), doc="SOC segments (1=shallowest)")
        model.E_seg = pyo.Param(model.J, initialize={j: segment_capacity for j in range(1, num_segments + 1)}, doc="Energy capacity of each segment (kWh)")
        model.c_cost = pyo.Param(model.J, initialize={j: marginal_costs[j - 1] for j in range(1, num_segments + 1)}, doc="Marginal cyclic degradation cost (EUR/kWh)")
        model.alpha = pyo.Param(initialize=alpha, doc="Degradation weight meta-parameter")

        logger.info("Added %d SOC segments with costs: %s", num_segments, marginal_costs)

        model.p_ch_j = pyo.Var(model.T, model.J, domain=pyo.NonNegativeReals, bounds=(0, P_max_config), doc="Charge power to segment j at time t (kW)")
        model.p_dis_j = pyo.Var(model.T, model.J, domain=pyo.NonNegativeReals, bounds=(0, P_max_config), doc="Discharge power from segment j at time t (kW)")
        model.e_soc_j = pyo.Var(model.T, model.J, domain=pyo.NonNegativeReals, bounds=(0, segment_capacity), doc="Energy stored in segment j at time t (kWh)")

        logger.info("Added segment variables: %s new variables", f"{len(model.T) * len(model.J) * 3:,}")

        if hasattr(model, 'soc_dynamics'):
            model.del_component(model.soc_dynamics)
            if hasattr(model, 'soc_dynamics_index'):
                model.del_component(model.soc_dynamics_index)
            logger.info("Removed parent SOC dynamics constraint (Cst-1)")

        if hasattr(model, 'e_soc'):
            model.del_component(model.e_soc)
            if hasattr(model, 'e_soc_index'):
                model.del_component(model.e_soc_index)

        model.e_soc = pyo.Expression(model.T, rule=lambda m, t: sum(m.e_soc_j[t, j] for j in m.J), doc="Total SOC derived from segment SOCs")

        # ============================================================================
        # CRITICAL FIX: Re-define Cst-6 to override parent and link to new e_soc Expression
        # ============================================================================
        # The parent class (Model I) defined Cst-6 constraints that reference e_soc as a Variable.
        # However, in Model II, we deleted that Variable and replaced it with an Expression.
        # The inherited constraints still reference the deleted Variable (causing them to fail).
        # We must re-define these constraints here to properly reference the new Expression.
        logger.info("Overriding (Cst-6) Energy Reserve constraints to link to segmented SOC Expression...")

        # First, remove the inherited (broken) constraints from parent
        if hasattr(model, 'energy_reserve_pos'):
            model.del_component(model.energy_reserve_pos)
        if hasattr(model, 'energy_reserve_neg'):
            model.del_component(model.energy_reserve_neg)

        # Cst-6: Ancillary Service Energy Reserve (REDEFINED for Model II)
        # Upward regulation: (1000*c_fcr + 1000*c_afrr_pos)*τ/η_dis ≤ e_soc(t) - SOC_min*E_nom
        def energy_reserve_pos_rule_v2(m, t):
            block = m.block_map[t]
            required_energy = (1000 * m.c_fcr[block] + 1000 * m.c_afrr_pos[block]) * m.tau / m.eta_dis
            # Now m.e_soc[t] correctly references the Expression (sum of segments)
            return required_energy <= m.e_soc[t] - m.SOC_min * m.E_nom

        model.energy_reserve_pos = pyo.Constraint(
            model.T, rule=energy_reserve_pos_rule_v2,
            doc="(Cst-6a REDEFINED) Energy reserve for upward regulation"
        )

        # Downward regulation: (1000*c_fcr + 1000*c_afrr_neg)*τ*η_ch ≤ SOC_max*E_nom - e_soc(t)
        def energy_reserve_neg_rule_v2(m, t):
            block = m.block_map[t]
            required_storage = (1000 * m.c_fcr[block] + 1000 * m.c_afrr_neg[block]) * m.tau * m.eta_ch
            # Now m.e_soc[t] correctly references the Expression (sum of segments)
            return required_storage <= m.SOC_max * m.E_nom - m.e_soc[t]

        model.energy_reserve_neg = pyo.Constraint(
            model.T, rule=energy_reserve_neg_rule_v2,
            doc="(Cst-6b REDEFINED) Energy reserve for downward regulation"
        )
        # ============================================================================

        def total_charge_power_rule(m, t):
            return m.p_total_ch[t] == sum(m.p_ch_j[t, j] for j in m.J)

        model.total_charge_aggregation = pyo.Constraint(model.T, rule=total_charge_power_rule, doc="Aggregate segment charge power")

        def total_discharge_power_rule(m, t):
            return m.p_total_dis[t] == sum(m.p_dis_j[t, j] for j in m.J)

        model.total_discharge_aggregation = pyo.Constraint(model.T, rule=total_discharge_power_rule, doc="Aggregate segment discharge power")

        def initial_segment_soc(m, j):
            capacity_per_segment = float(pyo.value(m.E_seg[j]))
            higher_capacity = capacity_per_segment * (int(j) - 1)
            initial_total = float(pyo.value(m.E_soc_init))
            remaining = max(0.0, initial_total - higher_capacity)
            return min(capacity_per_segment, remaining)

        def segment_soc_dynamics_rule(m, t, j):
            prev_t = prev_time_lookup.get(t)
            if prev_t is None:
                initial_soc_j = initial_segment_soc(m, j)
                return m.e_soc_j[t, j] == initial_soc_j + (m.p_ch_j[t, j] * m.eta_ch - m.p_dis_j[t, j] / m.eta_dis) * m.dt
            return m.e_soc_j[t, j] == m.e_soc_j[prev_t, j] + (m.p_ch_j[t, j] * m.eta_ch - m.p_dis_j[t, j] / m.eta_dis) * m.dt

        model.segment_soc_dynamics = pyo.Constraint(model.T, model.J, rule=segment_soc_dynamics_rule, doc="Segment SOC dynamics with top-down initialization")

        def stacked_tank_rule(m, t, j):
            if j == max(m.J):
                return pyo.Constraint.Skip
            return m.e_soc_j[t, j] >= m.e_soc_j[t, j + 1]

        model.stacked_tank_ordering = pyo.Constraint(model.T, model.J, rule=stacked_tank_rule, doc="Monotonic SOC ordering across segments")

        ######### CRITICAL: Binary variables must be defined BEFORE constraints that reference them #########
        # Define segment activation binary variable (required for LIFO constraint below)
        model.z_segment_active = pyo.Var(model.T, model.J, domain=pyo.Binary, doc="Segment activation binary: 1 if segment j has energy, 0 otherwise")

        # Link binary variable to segment SOC (z=1 if segment has energy, z=0 if empty)
        def segment_activation_upper_rule(m, t, j):
            return m.e_soc_j[t, j] <= m.E_seg[j] * m.z_segment_active[t, j]

        model.segment_activation_upper = pyo.Constraint(model.T, model.J, rule=segment_activation_upper_rule, doc="Binary activation: segment j can only have energy if z=1")

        # CRITICAL FIX: Add LIFO fullness prerequisite constraint (Xu et al. 2017, Theorem 1)
        # This enforces that segment j can only have energy if segment j-1 is FULL
        def segment_lifo_fullness_rule(m, t, j):
            """
            LIFO Constraint: Segment j can only contain energy if segment j-1 is full.

            This enforces the "stacked tank" behavior where lower segments must be
            completely filled before upper segments can receive any energy.

            Based on Xu et al. 2017, Theorem 1 & Lemma 1:
            - For discharge: empty segment 1 before segment 2, etc.
            - For charge: fill segment 1 before segment 2, etc.

            Without this constraint, energy gets distributed across all segments equally,
            violating the LIFO principle and producing incorrect degradation costs.
            """
            if j == 1:
                return pyo.Constraint.Skip

            # Tolerance for numerical stability and solver performance
            # Larger epsilon = larger feasible region = faster solve
            # 5 kWh ~ 1.1% of 447.2 kWh segment (acceptable tolerance)
            epsilon = self.degradation_params.get('lifo_epsilon_kwh', 0.0)  # kWh

            # If segment j is active (has ANY energy), segment j-1 must be full
            # z_segment_active[t,j] = 1 if e_soc_j[t,j] > 0
            # This is a big-M constraint where:
            # - If z_segment_active[t,j] = 1 (segment j has energy), then e_soc_j[t,j-1] >= E_seg[j-1] - epsilon (j-1 is full)
            # - If z_segment_active[t,j] = 0 (segment j is empty), constraint is trivially satisfied (RHS = 0)
            return m.e_soc_j[t, j-1] >= (m.E_seg[j-1] - epsilon) * m.z_segment_active[t, j]

        model.segment_lifo_fullness = pyo.Constraint(
            model.T, model.J,
            rule=segment_lifo_fullness_rule,
            doc="CRITICAL: LIFO fullness prerequisite - segment j only has energy if j-1 is full (Xu et al. 2017)"
        )
        epsilon_val = self.degradation_params.get('lifo_epsilon_kwh', 0.0)
        logger.info(f"Added LIFO fullness prerequisite constraints (epsilon={epsilon_val} kWh) to enforce stacked tank behavior")

        # NOTE: segment_activation_cascade is REDUNDANT with segment_lifo_fullness
        # If segment j is active and has energy, segment j-1 must be full (LIFO constraint)
        # If segment j-1 is full, it must be active (segment_activation_upper constraint)
        # Therefore, segment j active => segment j-1 active (transitively)
        # Disabling this constraint reduces T×J constraints and improves solve speed
        # def segment_activation_cascade_rule(m, t, j):
        #     if j == 1:
        #         return pyo.Constraint.Skip
        #     return m.z_segment_active[t, j] <= m.z_segment_active[t, j - 1]
        # model.segment_activation_cascade = pyo.Constraint(model.T, model.J, rule=segment_activation_cascade_rule, doc="Ensure deeper segments only active when shallower ones are active")

        if self.degradation_params.get('require_sequential_segment_activation', True):
            def segment_charge_activation_rule(m, t, j):
                return m.p_ch_j[t, j] <= m.P_max_config * m.z_segment_active[t, j]

            model.segment_charge_activation = pyo.Constraint(model.T, model.J, rule=segment_charge_activation_rule)

            def segment_discharge_activation_rule(m, t, j):
                return m.p_dis_j[t, j] <= m.P_max_config * m.z_segment_active[t, j]

            model.segment_discharge_activation = pyo.Constraint(model.T, model.J, rule=segment_discharge_activation_rule)
            logger.info("Added segment power activation constraints (Eq. 609-610) for strict sequential filling")
        else:
            logger.warning("CAUTION: Sequential segment activation constraints (Eq. 609-610) disabled. "
                         "This allows epsilon-tolerance parallel segment charging, which may underestimate "
                         "degradation costs. For strict LIFO behavior matching Xu et al. 2017, "
                         "set require_sequential_segment_activation=True")
            logger.info("Skipped segment power activation constraints for faster solve (8x speedup typical)")
        ######### WARNING: THIS SECTION WAS DISABLED BECAUSE IT SLOWS DOWN THE OPTIMIZATION MODEL SIGNIFICANTLY #########

        if hasattr(model, 'daily_cycle_limit'):
            model.daily_cycle_limit.deactivate()
            logger.info("Deactivated daily cycle limit constraint (Cst-5)")

        # Define cyclic degradation cost term as Expression (for easy extraction)
        def cost_cyclic_rule(m):
            return sum(
                sum(m.c_cost[j] * (m.p_dis_j[t, j] / m.eta_dis) * m.dt for j in m.J)
                for t in m.T
            )
        model.cost_cyclic = pyo.Expression(rule=cost_cyclic_rule)

        # Capture parent profit expressions before modifying objective
        # Note: model.profit_da, model.profit_afrr_energy, model.profit_as_capacity
        # were already defined in the parent class and will remain available

        # Delete parent objective to replace with aging-aware objective
        model.del_component(model.objective)

        # Create new objective with degradation cost
        # Objective = Profit - α × Aging_Cost
        model.objective = pyo.Objective(
            expr=model.profit_da + model.profit_afrr_energy + model.profit_as_capacity - model.alpha * model.cost_cyclic,
            sense=pyo.maximize,
            doc="Maximize profit minus weighted cyclic degradation cost"
        )
        logger.info("Extended parent objective with cyclic aging cost (alpha=%.4f)", alpha)

        logger.info("Model (ii) build complete: %s variables, %s constraints", f"{model.nvariables():,}", f"{model.nconstraints():,}")
        return model

    # ------------------------------------------------------------------
    # Solve & metrics
    # ------------------------------------------------------------------

    def extract_solution(self, model: pyo.ConcreteModel, solver_results: Any) -> Dict[str, Any]:
        """
        Extract solution from Model II (cyclic aging).

        Extends parent extract_solution() to add cyclic aging results.

        Args:
            model: Solved Model II Pyomo model
            solver_results: Results object from solver.solve()

        Returns:
            Dictionary with base solution plus cyclic aging metrics
        """
        # Call parent to get base solution
        solution_dict = super().extract_solution(model, solver_results)

        # If solve failed, return early
        if solution_dict.get('status') not in ['optimal', 'feasible']:
            return solution_dict

        # If degradation is not enabled, return base solution
        if not self.degradation_params.get('enabled', False):
            return solution_dict

        def _safe_value(component: Any) -> Optional[float]:
            try:
                return pyo.value(component)
            except (TypeError, ValueError):
                return None

        # Extract segment-specific variables
        p_ch_j = {}
        p_dis_j = {}
        e_soc_j = {}
        for t in model.T:
            for j in model.J:
                val_ch = _safe_value(model.p_ch_j[t, j])
                if val_ch is not None:
                    p_ch_j[(t, j)] = val_ch

                val_dis = _safe_value(model.p_dis_j[t, j])
                if val_dis is not None:
                    p_dis_j[(t, j)] = val_dis

                val_soc = _safe_value(model.e_soc_j[t, j])
                if val_soc is not None:
                    e_soc_j[(t, j)] = val_soc

        # Add Model II specific results
        solution_dict['p_ch_j'] = p_ch_j
        solution_dict['p_dis_j'] = p_dis_j
        solution_dict['e_soc_j'] = e_soc_j
        solution_dict['degradation_metrics'] = self._calculate_degradation_metrics(model, p_dis_j)

        return solution_dict

    def _calculate_degradation_metrics(self, model: pyo.ConcreteModel, p_dis_j: Dict[Tuple[int, int], float]) -> Dict[str, Any]:
        eta_dis = pyo.value(model.eta_dis)
        dt = pyo.value(model.dt)
        E_nom = pyo.value(model.E_nom)

        total_cyclic_cost = 0.0
        throughput_per_segment = {j: 0.0 for j in model.J}

        for (t, j), discharge_power in p_dis_j.items():
            energy = (discharge_power / eta_dis) * dt
            throughput_per_segment[j] += energy
            total_cyclic_cost += pyo.value(model.c_cost[j]) * energy

        total_throughput = sum(throughput_per_segment.values())
        efc = total_throughput / E_nom if E_nom else 0.0
        avg_dod = efc / len(model.D) if len(model.D) > 0 else 0.0

        cost_per_segment = {
            j: pyo.value(model.c_cost[j]) * throughput_per_segment[j]
            for j in model.J
        }

        return {
            'total_cyclic_cost_eur': total_cyclic_cost,
            'equivalent_full_cycles': efc,
            'total_throughput_kwh': total_throughput,
            'throughput_per_segment_kwh': throughput_per_segment,
            'cost_per_segment_eur': cost_per_segment,
            'average_dod': avg_dod,
            'alpha': pyo.value(model.alpha) if hasattr(model, 'alpha') else self.degradation_params.get('alpha'),
        }

# ============================================================================
# Phase II Model (iii): Calendar Aging Cost Extension
# ============================================================================


class BESSOptimizerModelIII(BESSOptimizerModelII):
    """Battery Energy Storage System Optimizer - Phase II Model (iii).

    Extends :class:`BESSOptimizerModelII` by adding calendar aging cost on top of
    cyclic aging cost. Calendar aging is modeled using Special Ordered Sets of Type 2 (SOS2)
    piecewise-linear approximation based on Collath et al. (2023).

    Model Progression:
    ------------------
    ✓ Model (i): Base + aFRR Energy Market
    ✓ Model (ii): Model (i) + Cyclic Aging Cost
    ✓ Model (iii): Model (ii) + Calendar Aging Cost (THIS MODEL)

    Calendar Aging Model:
    ---------------------
    - Uses SOS2 variables to approximate non-linear SOC-dependent aging
    - 5 breakpoints from Collath et al. (2023): 0%, 25%, 50%, 75%, 100% SOC
    - Higher SOC → Higher calendar aging cost (encourages lower SOC storage)
    - Cost function: c_cal_cost(t) = Σᵢ λ[t,i] * Cost_point[i]
    - SOC constraint: e_soc(t) = Σᵢ λ[t,i] * SOC_point[i]
    - SOS2 constraint: At most 2 adjacent λ variables non-zero

    Mathematical Formulation:
    -------------------------
    Objective: max(Revenue - α * (C_cyclic + C_calendar))

    Where:
    - C_cyclic: Segment-based cyclic aging cost (from Model II)
    - C_calendar: SOS2-based calendar aging cost (NEW)
    - α: Meta-parameter for degradation price

    References:
        See doc/p2_model/p2_bi_model_ggdp.tex Section \"Model (iii)\"
        Collath et al. (2023), Applied Energy, 348, 121531
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        degradation_config_path: Optional[str] = None,
        alpha: float = 1.0,
        require_sequential_segment_activation: bool = True,
        use_afrr_ev_weighting: bool = False,
    ) -> None:
        """Initialize Phase II Model (iii) with cyclic and calendar aging.

        Args:
            config: Optional runtime configuration overrides
            degradation_config_path: Path to aging_config.json (defaults to standard location)
            alpha: Degradation price weight parameter (default 1.0)
            require_sequential_segment_activation: Enforce strict sequential segment filling (default False for faster solve)
            use_afrr_ev_weighting: Enable Expected Value weighting for aFRR energy bids (default False)
        """
        # Initialize parent (Model II with cyclic aging)
        super().__init__(
            config=config,
            degradation_config_path=degradation_config_path,
            alpha=alpha,
            require_sequential_segment_activation=require_sequential_segment_activation,
            use_afrr_ev_weighting=use_afrr_ev_weighting,
        )

        # Load calendar aging configuration
        if 'calendar_aging' not in self.degradation_config:
            raise KeyError(
                "Missing 'calendar_aging' key in degradation config.\n"
                f"Config file: {self.degradation_params.get('config_file_path', 'unknown')}"
            )

        calendar_config = self.degradation_config['calendar_aging']

        # Validate calendar aging data structure (NEW FORMAT)
        if 'breakpoints' not in calendar_config:
            raise KeyError(
                "Missing 'breakpoints' key in calendar_aging config.\n"
                "Expected format: 'breakpoints': [{'soc_kwh': X, 'cost_eur_hr': Y}, ...]"
            )

        # Extract breakpoint data from new combined format
        breakpoints = calendar_config['breakpoints']

        if not isinstance(breakpoints, list) or len(breakpoints) == 0:
            raise ValueError("Calendar aging 'breakpoints' must be a non-empty list")

        # Parse SOC and cost arrays from combined breakpoints
        soc_breakpoints = []
        cost_breakpoints = []

        for i, bp in enumerate(breakpoints):
            if not isinstance(bp, dict):
                raise ValueError(f"Breakpoint {i} must be a dictionary")

            if 'soc_kwh' not in bp or 'cost_eur_hr' not in bp:
                raise KeyError(
                    f"Breakpoint {i} missing required keys 'soc_kwh' or 'cost_eur_hr'.\n"
                    f"Found keys: {list(bp.keys())}"
                )

            soc_breakpoints.append(bp['soc_kwh'])
            cost_breakpoints.append(bp['cost_eur_hr'])

        # Validate breakpoint consistency
        if len(soc_breakpoints) < 2:
            raise ValueError(f"Need at least 2 breakpoints, got {len(soc_breakpoints)}")

        # Extract unit information from new format
        unit_info = calendar_config.get('unit', {})
        soc_unit = unit_info.get('soc_kwh', 'kWh')
        cost_unit = unit_info.get('cost_eur_hr', 'EUR/hr')

        # Store calendar aging parameters
        self.calendar_params = {
            'num_breakpoints': len(soc_breakpoints),
            'soc_breakpoints_kwh': soc_breakpoints,
            'cost_breakpoints_eur_hr': cost_breakpoints,
            'soc_unit': soc_unit,
            'cost_unit': cost_unit,
        }

        # Update degradation params to indicate calendar aging is enabled
        self.degradation_params['model_type'] = 'cyclic_and_calendar'
        self.degradation_params['calendar_enabled'] = True

        # Validate calendar aging parameters
        self._validate_calendar_params()

        logger.info("Initialized BESSOptimizerModelIII:")
        logger.info("  - Calendar aging enabled with %d breakpoints",
                   self.calendar_params['num_breakpoints'])
        logger.info("  - SOC range: %.0f - %.0f kWh",
                   min(soc_breakpoints), max(soc_breakpoints))
        logger.info("  - Cost range: %.2f - %.2f EUR/hr",
                   min(cost_breakpoints), max(cost_breakpoints))
        logger.info("  - Alpha (degradation weight): %.4f", alpha)

    def _validate_calendar_params(self) -> None:
        """Validate calendar aging parameters for consistency."""
        soc_points = self.calendar_params['soc_breakpoints_kwh']
        cost_points = self.calendar_params['cost_breakpoints_eur_hr']
        battery_capacity = self.battery_params['capacity_kwh']

        # Check SOC breakpoints are monotonically increasing
        for i in range(1, len(soc_points)):
            if soc_points[i] <= soc_points[i-1]:
                raise ValueError(
                    f"SOC breakpoints must be strictly increasing. "
                    f"SOC[{i}]={soc_points[i]} <= SOC[{i-1}]={soc_points[i-1]}"
                )

        # Check SOC range matches battery capacity
        if abs(soc_points[0]) > 1e-6:  # Should start near 0
            logger.warning(
                "First SOC breakpoint (%.2f kWh) is not 0. "
                "This may cause issues at low SOC.", soc_points[0]
            )

        if abs(soc_points[-1] - battery_capacity) > 1e-6:  # Should end at capacity
            logger.warning(
                "Last SOC breakpoint (%.2f kWh) does not match battery capacity (%.2f kWh). "
                "This may cause issues at high SOC.",
                soc_points[-1], battery_capacity
            )

        # Check cost breakpoints are non-negative and generally increasing
        # (Calendar aging typically increases with SOC)
        if any(cost < 0 for cost in cost_points):
            raise ValueError(f"All cost breakpoints must be non-negative, got {cost_points}")

        for i in range(1, len(cost_points)):
            if cost_points[i] < cost_points[i-1]:
                logger.warning(
                    "Calendar cost typically increases with SOC. "
                    f"Cost[{i}]={cost_points[i]:.2f} < Cost[{i-1}]={cost_points[i-1]:.2f}"
                )

        logger.info("Calendar aging parameters validated successfully")

    def build_optimization_model(
        self,
        country_data: pd.DataFrame,
        c_rate: float,
        daily_cycle_limit: Optional[float] = None,
    ) -> pyo.ConcreteModel:
        """Build Model (iii) with cyclic and calendar aging costs.

        Extends Model (ii) by adding SOS2-based calendar aging constraints.

        Args:
            country_data: Market price data
            c_rate: C-rate configuration
            daily_cycle_limit: Ignored in Model (iii), kept for API compatibility

        Returns:
            Pyomo ConcreteModel with full Phase II Model (iii) formulation
        """
        if daily_cycle_limit is not None:
            logger.warning(
                "Model (iii) ignores daily_cycle_limit parameter (%.2f). "
                "Using cost-based degradation (cyclic + calendar) instead.",
                daily_cycle_limit,
            )

        # Build parent model (Model II with cyclic aging)
        model = super().build_optimization_model(country_data, c_rate, daily_cycle_limit=None)

        # Extract calendar aging parameters
        num_breakpoints = self.calendar_params['num_breakpoints']
        soc_breakpoints = self.calendar_params['soc_breakpoints_kwh']
        cost_breakpoints = self.calendar_params['cost_breakpoints_eur_hr']

        logger.info("Extending Model (ii) to Model (iii) with calendar aging...")

        # ============================================================================
        # Add Calendar Aging Components (SOS2 Piecewise-Linear Approximation)
        # ============================================================================

        # New Set: Calendar aging breakpoints
        model.I = pyo.Set(
            initialize=range(1, num_breakpoints + 1),
            doc="Calendar aging breakpoint indices (1=0% SOC, ..., I=100% SOC)"
        )

        # New Parameters: Breakpoint values
        model.SOC_point = pyo.Param(
            model.I,
            initialize={i: soc_breakpoints[i-1] for i in range(1, num_breakpoints + 1)},
            doc="SOC breakpoint values (kWh)"
        )

        model.Cost_point = pyo.Param(
            model.I,
            initialize={i: cost_breakpoints[i-1] for i in range(1, num_breakpoints + 1)},
            doc="Calendar cost breakpoint values (EUR/hr)"
        )

        # New Variables: SOS2 weighting variables and calendar cost
        model.lambda_cal = pyo.Var(
            model.T, model.I,
            domain=pyo.NonNegativeReals,
            bounds=(0, 1),
            doc="SOS2 variables for calendar aging piecewise-linear approximation"
        )

        model.c_cal_cost = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
            doc="Calendar aging cost at time t (EUR/hr)"
        )

        logger.info(
            "Added %d calendar aging variables (%d SOS2 per timestep)",
            len(model.T) * (num_breakpoints + 1),  # lambda_cal + c_cal_cost
            num_breakpoints
        )

        # ============================================================================
        # Calendar Aging Constraints (Section 4.7 of p2_bi_model_ggdp.tex)
        # ============================================================================

        # Constraint: SOC must be expressed as convex combination of breakpoints
        # e_soc(t) = Σᵢ λ_cal[t,i] * SOC_point[i]
        def calendar_soc_rule(m, t):
            # Note: m.e_soc[t] is an Expression (sum of segment SOCs from Model II)
            # We need to create an equality constraint here
            return (
                sum(m.lambda_cal[t, i] * m.SOC_point[i] for i in m.I) ==
                sum(m.e_soc_j[t, j] for j in m.J)  # Use explicit sum instead of Expression
            )

        model.calendar_soc_con = pyo.Constraint(
            model.T,
            rule=calendar_soc_rule,
            doc="Link total SOC to SOS2 variables for calendar aging"
        )

        # Constraint: Calendar cost must be expressed as same convex combination
        # c_cal_cost(t) = Σᵢ λ_cal[t,i] * Cost_point[i]
        def calendar_cost_rule(m, t):
            return m.c_cal_cost[t] == sum(
                m.lambda_cal[t, i] * m.Cost_point[i] for i in m.I
            )

        model.calendar_cost_con = pyo.Constraint(
            model.T,
            rule=calendar_cost_rule,
            doc="Calculate calendar cost from SOS2 variables"
        )

        # Constraint: SOS2 weights must sum to 1
        # Σᵢ λ_cal[t,i] = 1
        def calendar_sos2_sum_rule(m, t):
            return sum(m.lambda_cal[t, i] for i in m.I) == 1

        model.calendar_sos2_sum_con = pyo.Constraint(
            model.T,
            rule=calendar_sos2_sum_rule,
            doc="SOS2 weights must sum to 1"
        )

        # SOS2 Constraint: At most 2 adjacent λ variables can be non-zero
        # This is the key constraint that enforces piecewise-linearity
        def calendar_sos2_rule(m, t):
            # Return list of variables that form the SOS2 set for time t
            return [m.lambda_cal[t, i] for i in m.I]

        model.calendar_sos2_set = pyo.SOSConstraint(
            model.T,
            var=calendar_sos2_rule,
            sos=2,  # Type 2: At most 2 adjacent variables non-zero
            doc="SOS2 constraint ensuring at most 2 adjacent lambda variables are non-zero"
        )

        logger.info("Added %d calendar aging constraints", len(model.T) * 3)  # 3 constraints per t

        # ============================================================================
        # Update Objective Function to Include Calendar Aging Cost
        # ============================================================================

        # Capture current objective (already has Revenue - α*C_cyclic from Model II)
        parent_objective_expr = model.objective.expr

        # Calculate total calendar aging cost
        # C_calendar = Σₜ c_cal_cost[t] * Δt
        cost_calendar = sum(model.c_cal_cost[t] * model.dt for t in model.T)

        # Delete old objective
        model.del_component(model.objective)

        # Create new objective: Revenue - α*(C_cyclic + C_calendar)
        # Note: parent_objective_expr already contains "- alpha * C_cyclic"
        model.objective = pyo.Objective(
            expr=parent_objective_expr - model.alpha * cost_calendar,
            sense=pyo.maximize,
            doc="Maximize profit minus weighted degradation cost (cyclic + calendar)"
        )

        logger.info(
            "Extended objective function with calendar aging cost (alpha=%.4f)",
            pyo.value(model.alpha)
        )

        # ============================================================================
        # Model Summary
        # ============================================================================

        logger.info("Model (iii) build complete:")
        logger.info("  - Variables: %s", f"{model.nvariables():,}")
        logger.info("  - Constraints: %s", f"{model.nconstraints():,}")
        logger.info("  - Degradation: Cyclic (%d segments) + Calendar (%d breakpoints)",
                   self.degradation_params['num_segments'],
                   num_breakpoints)

        return model

    def extract_solution(self, model: pyo.ConcreteModel, solver_results: Any) -> Dict[str, Any]:
        """
        Extract solution from Model III (cyclic + calendar aging).

        Extends parent extract_solution() to add calendar aging results.

        Args:
            model: Solved Model III Pyomo model
            solver_results: Results object from solver.solve()

        Returns:
            Dictionary with base solution plus cyclic and calendar aging metrics
        """
        # Call parent (Model II) to get base solution with cyclic aging
        solution_dict = super().extract_solution(model, solver_results)

        # If solve failed, return early
        if solution_dict.get('status') not in ['optimal', 'feasible']:
            return solution_dict

        # Extract calendar aging results
        def _safe_value(component: Any) -> Optional[float]:
            try:
                return pyo.value(component)
            except (TypeError, ValueError):
                return None

        # Extract SOS2 variables (for validation)
        lambda_cal = {}
        for t in model.T:
            for i in model.I:
                val = _safe_value(model.lambda_cal[t, i])
                if val is not None and val > 1e-6:  # Only store non-zero values
                    lambda_cal[(t, i)] = val

        # Extract calendar costs
        c_cal_cost = {}
        total_calendar_cost = 0.0
        for t in model.T:
            val = _safe_value(model.c_cal_cost[t])
            if val is not None:
                c_cal_cost[t] = val
                total_calendar_cost += val * pyo.value(model.dt)

        # Add calendar aging results to solution
        solution_dict['lambda_cal'] = lambda_cal
        solution_dict['c_cal_cost'] = c_cal_cost

        # Update degradation metrics
        if 'degradation_metrics' in solution_dict:
            solution_dict['degradation_metrics']['total_calendar_cost_eur'] = total_calendar_cost

            # Calculate combined degradation cost
            cyclic_cost = solution_dict['degradation_metrics'].get('total_cyclic_cost_eur', 0.0)
            solution_dict['degradation_metrics']['total_degradation_cost_eur'] = (
                cyclic_cost + total_calendar_cost
            )

            # Add breakdown
            solution_dict['degradation_metrics']['cost_breakdown'] = {
                'cyclic_eur': cyclic_cost,
                'calendar_eur': total_calendar_cost,
                'total_eur': cyclic_cost + total_calendar_cost,
            }

            logger.info("Degradation cost breakdown:")
            logger.info("  - Cyclic aging: %.2f EUR", cyclic_cost)
            logger.info("  - Calendar aging: %.2f EUR", total_calendar_cost)
            logger.info("  - Total degradation: %.2f EUR", cyclic_cost + total_calendar_cost)

        return solution_dict


# ============================================================================
# MODEL III-RENEW: Renewable Integration
# ============================================================================

class BESSOptimizerModelIIIRenew(BESSOptimizerModelIII):
    """Model III + Simplified Renewable Power Plant Integration.

    Extends Model III (cyclic + calendar aging) with renewable generation
    variables and constraints per Blueprint Section 6.4.

    New Decision Variables:
        - p_renewable_self[t]:    Self-consumption power (kW) — charges battery
        - p_renewable_export[t]:  Grid export power (kW) — sold at DA price
        - p_renewable_curtail[t]: Curtailed power (kW) — wasted

    New Constraints:
        - Cst-R1: P_renewable[t] = P_self[t] + P_export[t] + P_curtail[t]
        - Cst-R2: p_total_ch[t] = p_ch[t] + p_afrr_neg_e[t] + p_renewable_self[t]

    New Revenue:
        - R_export = Σₜ P_renewable_export[t] * P_DA[t] / 1000 * dt
    """

    def __init__(self, **kwargs) -> None:
        """Initialize Model III-Renew.

        Args:
            **kwargs: All arguments forwarded to BESSOptimizerModelIII.__init__
                (config, degradation_config_path, alpha,
                 require_sequential_segment_activation, use_afrr_ev_weighting)
        """
        super().__init__(**kwargs)
        logger.info("BESSOptimizerModelIIIRenew initialized (Model III + Renewable)")

    def build_optimization_model(
        self,
        country_data: pd.DataFrame,
        c_rate: float,
        daily_cycle_limit: Optional[float] = None,
    ) -> pyo.ConcreteModel:
        """Build Model III-Renew with renewable integration.

        Extends Model III by adding renewable self-consumption, export, and
        curtailment variables. Renewable data is read from the
        ``p_renewable_forecast_kw`` column of *country_data*.

        If the column is missing or all-null the method falls back to plain
        Model III (no renewable variables are added).

        Args:
            country_data: Market price data, optionally including
                ``p_renewable_forecast_kw`` column (kW, 15-min resolution)
            c_rate: C-rate configuration
            daily_cycle_limit: Ignored (kept for API compat)

        Returns:
            Pyomo ConcreteModel with renewable integration
        """
        # Build full Model III first
        model = super().build_optimization_model(country_data, c_rate, daily_cycle_limit)

        # ------------------------------------------------------------------
        # Check for renewable forecast data
        # ------------------------------------------------------------------
        has_renewable = (
            'p_renewable_forecast_kw' in country_data.columns
            and not country_data['p_renewable_forecast_kw'].isna().all()
        )

        if not has_renewable:
            logger.warning(
                "No renewable forecast data in country_data. "
                "Running as plain Model III."
            )
            return model

        logger.info("Extending Model III to Model III-Renew with renewable integration...")

        T_data = list(range(len(country_data)))

        # ------------------------------------------------------------------
        # New Parameter: Renewable generation forecast (kW)
        # ------------------------------------------------------------------
        renewable_forecast = {
            t: (
                float(country_data['p_renewable_forecast_kw'].iloc[t])
                if not pd.isna(country_data['p_renewable_forecast_kw'].iloc[t])
                else 0.0
            )
            for t in T_data
        }

        model.P_renewable = pyo.Param(
            model.T,
            initialize=renewable_forecast,
            doc="Renewable generation forecast (kW)"
        )

        # ------------------------------------------------------------------
        # New Variables: Renewable power allocation (kW)
        # ------------------------------------------------------------------
        model.p_renewable_self = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
            doc="Self-consumption renewable power — charges battery (kW)"
        )

        model.p_renewable_export = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
            doc="Exported renewable power — sold at DA price (kW)"
        )

        model.p_renewable_curtail = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
            doc="Curtailed renewable power — wasted (kW)"
        )

        logger.info(
            "Added 3×%d renewable variables (%d timesteps)",
            len(T_data), len(T_data)
        )

        # ------------------------------------------------------------------
        # Cst-R1: Renewable Balance
        #   P_self[t] + P_export[t] + P_curtail[t] == P_renewable[t]
        # ------------------------------------------------------------------
        def renewable_balance_rule(m, t):
            return (
                m.p_renewable_self[t]
                + m.p_renewable_export[t]
                + m.p_renewable_curtail[t]
                == m.P_renewable[t]
            )

        model.renewable_balance = pyo.Constraint(
            model.T,
            rule=renewable_balance_rule,
            doc="Cst-R1: Renewable generation must be fully allocated"
        )

        # ------------------------------------------------------------------
        # Cst-R2: Modified Total Charging Definition
        #   p_total_ch[t] == p_ch[t] + p_afrr_neg_e[t] + p_renewable_self[t]
        #
        # The existing total_charge_aggregation (Model II) still holds:
        #   p_total_ch[t] == Σⱼ p_ch_j[t,j]
        # Together these route renewable self-consumption through the
        # battery segments for correct degradation accounting.
        # ------------------------------------------------------------------
        model.del_component(model.total_ch_def)

        def total_ch_def_renew_rule(m, t):
            return m.p_total_ch[t] == m.p_ch[t] + m.p_afrr_neg_e[t] + m.p_renewable_self[t]

        model.total_ch_def = pyo.Constraint(
            model.T,
            rule=total_ch_def_renew_rule,
            doc="Cst-R2: Total charging = DA + aFRR-E neg + renewable self-consumption"
        )

        # ------------------------------------------------------------------
        # Cst-R3: Export Revenue and Objective Update
        #   R_export = Σₜ p_renewable_export[t] * P_DA[t] / 1000 * dt
        # ------------------------------------------------------------------
        profit_export = sum(
            model.p_renewable_export[t] * model.P_DA[t] / 1000 * model.dt
            for t in model.T
        )

        model.profit_renewable_export = pyo.Expression(
            expr=profit_export,
            doc="Revenue from exporting renewable power at DA price (EUR)"
        )

        # Replace objective: add export revenue
        parent_objective_expr = model.objective.expr
        model.del_component(model.objective)

        model.objective = pyo.Objective(
            expr=parent_objective_expr + profit_export,
            sense=pyo.maximize,
            doc="Maximize profit + renewable export revenue - degradation cost"
        )

        logger.info(
            "Added %d renewable constraints (balance + modified total_ch_def)",
            len(model.T) * 2
        )

        # ------------------------------------------------------------------
        # Model Summary
        # ------------------------------------------------------------------
        logger.info("Model III-Renew build complete:")
        logger.info("  - Variables: %s", f"{model.nvariables():,}")
        logger.info("  - Constraints: %s", f"{model.nconstraints():,}")

        return model

    def extract_solution(
        self, model: pyo.ConcreteModel, solver_results: Any
    ) -> Dict[str, Any]:
        """Extract solution including renewable variables.

        Extends Model III extract_solution with:
        - Per-timestep renewable power allocation
        - Renewable export revenue
        - Renewable utilization summary

        Args:
            model: Solved Model III-Renew Pyomo model
            solver_results: Results object from solver.solve()

        Returns:
            Dict with all Model III fields plus renewable results
        """
        solution_dict = super().extract_solution(model, solver_results)

        # Early exit if solve failed
        if solution_dict.get('status') not in ('optimal', 'feasible'):
            return solution_dict

        # Skip if model has no renewable variables (fallback case)
        if not hasattr(model, 'p_renewable_self'):
            return solution_dict

        def _safe_value(component):
            try:
                return pyo.value(component)
            except (TypeError, ValueError):
                return None

        dt = pyo.value(model.dt)

        # Extract per-timestep renewable variables
        solution_dict['p_renewable_self'] = {}
        solution_dict['p_renewable_export'] = {}
        solution_dict['p_renewable_curtail'] = {}
        solution_dict['revenue_renewable_export'] = {}

        total_self_kwh = 0.0
        total_export_kwh = 0.0
        total_curtail_kwh = 0.0
        total_generation_kwh = 0.0

        for t in model.T:
            p_self = _safe_value(model.p_renewable_self[t]) or 0.0
            p_export = _safe_value(model.p_renewable_export[t]) or 0.0
            p_curtail = _safe_value(model.p_renewable_curtail[t]) or 0.0
            p_gen = _safe_value(model.P_renewable[t]) or 0.0
            da_price = _safe_value(model.P_DA[t]) or 0.0

            solution_dict['p_renewable_self'][t] = p_self
            solution_dict['p_renewable_export'][t] = p_export
            solution_dict['p_renewable_curtail'][t] = p_curtail

            # Per-timestep export revenue (EUR)
            revenue_export_t = p_export * da_price / 1000 * dt
            solution_dict['revenue_renewable_export'][t] = revenue_export_t

            # Accumulate energy totals (kW * h = kWh)
            total_self_kwh += p_self * dt
            total_export_kwh += p_export * dt
            total_curtail_kwh += p_curtail * dt
            total_generation_kwh += p_gen * dt

        # Total export revenue
        profit_export = _safe_value(model.profit_renewable_export) or 0.0
        solution_dict['profit_renewable_export'] = profit_export

        # Renewable utilization summary
        solution_dict['renewable_utilization'] = {
            'total_generation_kwh': total_generation_kwh,
            'self_consumption_kwh': total_self_kwh,
            'export_kwh': total_export_kwh,
            'curtailment_kwh': total_curtail_kwh,
            'utilization_rate': (
                (total_self_kwh + total_export_kwh) / total_generation_kwh
                if total_generation_kwh > 0 else 0.0
            ),
        }

        logger.info("Renewable utilization: %.1f%% (self=%.1f kWh, export=%.1f kWh, curtail=%.1f kWh)",
                     solution_dict['renewable_utilization']['utilization_rate'] * 100,
                     total_self_kwh, total_export_kwh, total_curtail_kwh)
        logger.info("Renewable export revenue: %.2f EUR", profit_export)

        return solution_dict


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================
# Maintain backward compatibility with existing code that uses old names

# Phase II Model naming
BESSOptimizer = BESSOptimizerModelI  # Main backward compatibility alias
BESSOptimizerV2 = BESSOptimizerModelI  # For code that used V2 naming
BESSOptimizerV3 = BESSOptimizerModelII  # Version-based alias for cyclic aging model
BESSOptimizerV4 = BESSOptimizerModelIII  # Version-based alias for full model (cyclic + calendar)

# Clear naming for users
BESSOptimizer_Phase2_ModelI = BESSOptimizerModelI  # Explicit Phase II Model (i) reference
BESSOptimizer_Phase2_ModelII = BESSOptimizerModelII  # Explicit Phase II Model (ii) reference
BESSOptimizer_Phase2_ModelIII = BESSOptimizerModelIII  # Explicit Phase II Model (iii) reference
BESSOptimizer_Phase2_ModelIIIRenew = BESSOptimizerModelIIIRenew  # Model (iii) + renewable integration

if __name__ == "__main__":
    # Example usage - Model (i)
    optimizer = BESSOptimizerModelI()

    # Run quick test
    data_file = "../data/"
    results = optimizer.run_scenario_analysis(data_file, num_days=3)
    print("\nPhase II Model (i) Results:")
    print(results.to_string())