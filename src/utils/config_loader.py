"""
Centralized YAML Configuration Loader
=====================================

This module provides a centralized configuration loader for the GridKey BESS optimizer project.
All configuration is stored in a single YAML file (config/Config.yml) and accessed through
the ConfigLoader class.

Usage:
    from src.utils.config_loader import ConfigLoader

    # Get specific config sections
    solver_config = ConfigLoader.get_solver_config()
    aging_config = ConfigLoader.get_aging_config()

    # Or load entire config
    full_config = ConfigLoader.load_config()

Author: GridKey Team
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Centralized configuration loader with caching.

    Loads configuration from config/Config.yml and provides
    section-specific accessor methods for each config domain.

    The config is cached after first load for performance.
    Use clear_cache() to force reload if config file changes.
    """

    _config_cache: Optional[Dict[str, Any]] = None
    _config_path: Optional[Path] = None

    @classmethod
    def _get_config_path(cls) -> Path:
        """Get the path to the config file."""
        if cls._config_path is None:
            # Navigate from src/utils/ to project root, then to config/
            cls._config_path = Path(__file__).resolve().parent.parent.parent / 'config' / 'Config.yml'
        return cls._config_path

    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """
        Load the unified YAML configuration with caching.

        Returns:
            Dict containing all configuration sections.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        if cls._config_cache is None:
            config_path = cls._get_config_path()

            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {config_path}\n"
                    "Expected location: config/Config.yml"
                )

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config_cache = yaml.safe_load(f)
                logger.info(f"Loaded configuration from: {config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in config file: {e}") from e

        return cls._config_cache

    @classmethod
    def get_solver_config(cls) -> Dict[str, Any]:
        """
        Get solver configuration section.

        Returns:
            Dict with keys: default_solver, solver_time_limit_sec, solver_options
        """
        return cls.load_config()['solver_config']

    @classmethod
    def get_aging_config(cls) -> Dict[str, Any]:
        """
        Get aging/degradation configuration section.

        Returns:
            Dict with keys: cyclic_aging, calendar_aging, alpha_sweep,
                           require_sequential_segment_activation, lifo_epsilon_kwh
        """
        return cls.load_config()['aging_config']

    @classmethod
    def get_afrr_ev_weights_config(cls) -> Dict[str, Any]:
        """
        Get aFRR energy activation weights configuration.

        Returns:
            Dict with keys: historical_activation, balanced_activation, usage_notes
        """
        return cls.load_config()['afrr_ev_weights_config']

    @classmethod
    def get_mpc_config(cls) -> Dict[str, Any]:
        """
        Get MPC controller configuration.

        Returns:
            Dict with keys: mpc_parameters, notes
        """
        return cls.load_config()['mpc_config']

    @classmethod
    def get_mpc_test_config(cls) -> Dict[str, Any]:
        """
        Get MPC test scenario configuration.

        Returns:
            Dict with keys: test_scenario, alpha_settings, meta_optimizer,
                           visualization, output, alpha_meta_optimization
        """
        return cls.load_config()['mpc_test_config']

    @classmethod
    def get_investment_config(cls) -> Dict[str, Any]:
        """
        Get investment/financial parameters configuration.

        Returns:
            Dict with keys: investment_eur_per_kwh, project_lifetime_years, country_specific
        """
        return cls.load_config()['investment_config']

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the cached configuration.

        Call this if you need to reload the config file after changes.
        Useful for testing or dynamic config updates.
        """
        cls._config_cache = None
        logger.debug("Configuration cache cleared")

    @classmethod
    def set_config_path(cls, path: Path) -> None:
        """
        Override the default config path.

        Useful for testing with alternative config files.

        Args:
            path: Path to alternative config file.
        """
        cls._config_path = Path(path)
        cls.clear_cache()  # Clear cache to force reload from new path
        logger.info(f"Config path set to: {path}")
