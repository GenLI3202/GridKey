"""
Synthetic data generators for testing.
"""

import numpy as np
from typing import List, Dict


def generate_synthetic_renewable(
    hours: int = 48,
    peak_kw: float = 100.0,
    noise_factor: float = 0.1
) -> List[float]:
    """
    Generate synthetic PV generation profile.

    Args:
        hours: Number of hours to generate
        peak_kw: Peak generation in kW
        noise_factor: Random noise factor (0-1)

    Returns:
        List of 15-min generation values in kW
    """
    timesteps = hours * 4  # 15-min intervals
    generation = []

    for t in range(timesteps):
        hour = (t // 4) % 24

        # Solar curve: peak at noon, zero at night
        if 6 <= hour <= 18:
            base = peak_kw * np.sin(np.pi * (hour - 6) / 12)
        else:
            base = 0.0

        # Add noise
        noise = np.random.normal(0, base * noise_factor) if base > 0 else 0
        generation.append(max(0, base + noise))

    return generation


def generate_synthetic_market_prices(hours: int = 48) -> Dict[str, List[float]]:
    """
    Generate synthetic market price data.

    Returns dict with keys matching DataAdapter expectations:
    - day_ahead, afrr_energy_pos, afrr_energy_neg (15-min resolution)
    - fcr, afrr_capacity_pos, afrr_capacity_neg (4-hour blocks)
    """
    timesteps_15min = hours * 4
    blocks_4h = hours // 4

    return {
        "day_ahead": list(np.random.uniform(20, 80, timesteps_15min)),
        "afrr_energy_pos": list(np.random.uniform(30, 70, timesteps_15min)),
        "afrr_energy_neg": list(np.random.uniform(20, 50, timesteps_15min)),
        "fcr": list(np.random.uniform(50, 150, blocks_4h)),
        "afrr_capacity_pos": list(np.random.uniform(3, 10, blocks_4h)),
        "afrr_capacity_neg": list(np.random.uniform(5, 15, blocks_4h)),
    }
