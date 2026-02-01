"""
Tests for src/service/adapter.py — DataAdapter format conversion.
"""

import pytest
import math
from datetime import datetime

from src.service.adapter import DataAdapter, TIMESTEPS_PER_BLOCK, BLOCKS_PER_DAY
from src.service.models import ModelType, OptimizationInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return DataAdapter()


@pytest.fixture
def sample_15min_prices():
    """192 entries = 48h at 15-min resolution."""
    return [50.0 + i * 0.1 for i in range(192)]


@pytest.fixture
def sample_block_prices():
    """12 entries = 48h at 4-hour blocks."""
    return [100.0 + i for i in range(12)]


@pytest.fixture
def sample_opt_input(sample_15min_prices, sample_block_prices):
    return OptimizationInput(
        da_prices=sample_15min_prices,
        afrr_energy_pos=sample_15min_prices,
        afrr_energy_neg=sample_15min_prices,
        fcr_prices=sample_block_prices,
        afrr_capacity_pos=sample_block_prices,
        afrr_capacity_neg=sample_block_prices,
    )


@pytest.fixture
def sample_market_prices(sample_15min_prices, sample_block_prices):
    return {
        'day_ahead': sample_15min_prices,
        'afrr_energy_pos': sample_15min_prices,
        'afrr_energy_neg': sample_15min_prices,
        'fcr': sample_block_prices,
        'afrr_capacity_pos': sample_block_prices,
        'afrr_capacity_neg': sample_block_prices,
    }


# ---------------------------------------------------------------------------
# to_country_data: basic output
# ---------------------------------------------------------------------------

class TestToCountryDataBasic:
    def test_correct_row_count(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert len(df) == 192  # 48h * 4 per hour

    def test_all_required_columns_present(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        for col in DataAdapter.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_da_prices_mapped_directly(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert list(df['price_day_ahead']) == sample_opt_input.da_prices

    def test_afrr_energy_nonzero_preserved(self, adapter, sample_opt_input):
        """Non-zero aFRR energy prices map through unchanged (sample starts at 50.0)."""
        df = adapter.to_country_data(sample_opt_input)
        assert list(df['price_afrr_energy_pos']) == sample_opt_input.afrr_energy_pos
        assert list(df['price_afrr_energy_neg']) == sample_opt_input.afrr_energy_neg


# ---------------------------------------------------------------------------
# to_country_data: block price forward-fill
# ---------------------------------------------------------------------------

class TestBlockPriceExpansion:
    def test_fcr_forward_fill(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        # First block: rows 0-15 should all have price 100.0
        assert all(df['price_fcr'].iloc[0:16] == 100.0)
        # Second block: rows 16-31 should all have price 101.0
        assert all(df['price_fcr'].iloc[16:32] == 101.0)

    def test_afrr_capacity_forward_fill(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        # Last block (block 11): rows 176-191
        assert all(df['price_afrr_pos'].iloc[176:192] == 111.0)  # 100+11

    def test_each_block_has_16_timesteps(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        for block_id in df['block_id'].unique():
            count = (df['block_id'] == block_id).sum()
            assert count == TIMESTEPS_PER_BLOCK, \
                f"Block {block_id} has {count} timesteps, expected {TIMESTEPS_PER_BLOCK}"

    def test_12_unique_blocks(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert df['block_id'].nunique() == 12


# ---------------------------------------------------------------------------
# to_country_data: time identifiers
# ---------------------------------------------------------------------------

class TestTimeIdentifiers:
    def test_default_start_time(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert df['timestamp'].iloc[0] == datetime(2024, 1, 1, 0, 0)

    def test_custom_start_time(self, adapter, sample_opt_input):
        start = datetime(2024, 6, 15, 8, 0)
        df = adapter.to_country_data(sample_opt_input, start_time=start)
        assert df['timestamp'].iloc[0] == start
        assert df['hour'].iloc[0] == 8

    def test_block_of_day_range(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert df['block_of_day'].min() == 0
        assert df['block_of_day'].max() == 5  # 6 blocks per day (0-5)

    def test_day_id_spans_two_days(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert df['day_id'].nunique() == 2  # 48h = 2 days

    def test_block_id_unique_across_days(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        # Day 1 blocks 0-5, Day 2 blocks 6-11 = 12 total
        assert df['block_id'].nunique() == 12
        assert df['block_id'].min() == 0
        assert df['block_id'].max() == 11


# ---------------------------------------------------------------------------
# to_country_data: aFRR weights
# ---------------------------------------------------------------------------

class TestAfrrWeights:
    def test_default_weights_are_one(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert (df['w_afrr_pos'] == 1.0).all()
        assert (df['w_afrr_neg'] == 1.0).all()


# ---------------------------------------------------------------------------
# to_country_data: renewable generation
# ---------------------------------------------------------------------------

class TestRenewableGeneration:
    def test_with_renewable(self, adapter, sample_15min_prices, sample_block_prices):
        renewable = [10.0] * 192
        opt_input = OptimizationInput(
            da_prices=sample_15min_prices,
            afrr_energy_pos=sample_15min_prices,
            afrr_energy_neg=sample_15min_prices,
            fcr_prices=sample_block_prices,
            afrr_capacity_pos=sample_block_prices,
            afrr_capacity_neg=sample_block_prices,
            renewable_generation=renewable,
        )
        df = adapter.to_country_data(opt_input)
        assert 'p_renewable_forecast_kw' in df.columns
        assert list(df['p_renewable_forecast_kw']) == renewable

    def test_without_renewable(self, adapter, sample_opt_input):
        df = adapter.to_country_data(sample_opt_input)
        assert 'p_renewable_forecast_kw' not in df.columns


# ---------------------------------------------------------------------------
# adapt(): service dict → OptimizationInput
# ---------------------------------------------------------------------------

class TestAdapt:
    def test_basic_adapt(self, adapter, sample_market_prices):
        result = adapter.adapt(sample_market_prices)
        assert isinstance(result, OptimizationInput)
        assert len(result.da_prices) == 192
        assert len(result.fcr_prices) == 12
        assert result.renewable_generation is None

    def test_adapt_with_generation(self, adapter, sample_market_prices):
        forecast = {'generation_kw': [500.0] * 192}
        result = adapter.adapt(sample_market_prices, generation_forecast=forecast)
        assert result.renewable_generation is not None
        assert len(result.renewable_generation) == 192
        assert result.renewable_generation[0] == 500.0

    def test_adapt_with_pv_wind_split(self, adapter, sample_market_prices):
        forecast = {
            'pv_kw': [300.0] * 192,
            'wind_kw': [200.0] * 192,
        }
        result = adapter.adapt(sample_market_prices, generation_forecast=forecast)
        assert result.renewable_generation is not None
        assert result.renewable_generation[0] == 500.0

    def test_adapt_with_battery_config(self, adapter, sample_market_prices):
        battery = {
            'capacity_kwh': 2000,
            'c_rate': 0.25,
            'efficiency': 0.9,
            'initial_soc': 0.8,
        }
        result = adapter.adapt(sample_market_prices, battery_config=battery)
        assert result.battery_capacity_kwh == 2000
        assert result.c_rate == 0.25
        assert result.efficiency == 0.9
        assert result.initial_soc == 0.8

    def test_adapt_missing_prices_raises(self, adapter):
        with pytest.raises(ValueError, match="day_ahead"):
            adapter.adapt({})


# ---------------------------------------------------------------------------
# Round-trip: adapt() → to_country_data()
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_adapt_then_to_country_data(self, adapter, sample_market_prices):
        opt_input = adapter.adapt(sample_market_prices)
        df = adapter.to_country_data(opt_input)
        assert len(df) == 192
        assert 'price_day_ahead' in df.columns
        assert 'block_id' in df.columns
        assert df['block_id'].nunique() == 12

    def test_adapt_with_renewable_then_to_country_data(self, adapter, sample_market_prices):
        forecast = {'generation_kw': [100.0] * 192}
        opt_input = adapter.adapt(sample_market_prices, generation_forecast=forecast)
        df = adapter.to_country_data(opt_input)
        assert 'p_renewable_forecast_kw' in df.columns
        assert (df['p_renewable_forecast_kw'] == 100.0).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAfrrEnergyNanConversion:
    """CRITICAL: aFRR energy price 0 means 'not activated', not 'free energy'.
    The adapter must convert 0 -> NaN to prevent false arbitrage."""

    def test_zero_afrr_energy_becomes_nan(self, adapter, sample_block_prices):
        """Zero aFRR energy prices must be converted to NaN in country_data."""
        # Mix of real prices and zeros (zeros = not activated)
        afrr_pos = [0.0, 120.5, 0.0, 80.0] * 48  # 192 entries
        afrr_neg = [0.0, 0.0, 95.0, 0.0] * 48
        da_prices = [50.0] * 192

        opt_input = OptimizationInput(
            da_prices=da_prices,
            afrr_energy_pos=afrr_pos,
            afrr_energy_neg=afrr_neg,
            fcr_prices=sample_block_prices,
            afrr_capacity_pos=sample_block_prices,
            afrr_capacity_neg=sample_block_prices,
        )
        df = adapter.to_country_data(opt_input)

        # Zeros should be NaN
        import numpy as np
        assert np.isnan(df['price_afrr_energy_pos'].iloc[0])
        assert np.isnan(df['price_afrr_energy_pos'].iloc[2])
        assert np.isnan(df['price_afrr_energy_neg'].iloc[0])
        assert np.isnan(df['price_afrr_energy_neg'].iloc[1])
        assert np.isnan(df['price_afrr_energy_neg'].iloc[3])

        # Non-zero prices should be preserved
        assert df['price_afrr_energy_pos'].iloc[1] == 120.5
        assert df['price_afrr_energy_pos'].iloc[3] == 80.0
        assert df['price_afrr_energy_neg'].iloc[2] == 95.0

    def test_da_prices_not_affected_by_nan_conversion(self, adapter, sample_block_prices):
        """DA price = 0 is valid (e.g. solar surplus). Must NOT be converted."""
        da_prices = [0.0, 50.0, -10.0] * 64  # 192 entries
        afrr = [40.0] * 192

        opt_input = OptimizationInput(
            da_prices=da_prices,
            afrr_energy_pos=afrr,
            afrr_energy_neg=afrr,
            fcr_prices=sample_block_prices,
            afrr_capacity_pos=sample_block_prices,
            afrr_capacity_neg=sample_block_prices,
        )
        df = adapter.to_country_data(opt_input)
        # DA zero prices must stay as 0.0
        assert df['price_day_ahead'].iloc[0] == 0.0

    def test_nan_count_matches_zero_count(self, adapter, sample_block_prices):
        """Number of NaN in output should equal number of zeros in input."""
        import numpy as np
        n_zeros = 50
        afrr = [0.0] * n_zeros + [100.0] * (192 - n_zeros)

        opt_input = OptimizationInput(
            da_prices=[50.0] * 192,
            afrr_energy_pos=afrr,
            afrr_energy_neg=afrr,
            fcr_prices=sample_block_prices,
            afrr_capacity_pos=sample_block_prices,
            afrr_capacity_neg=sample_block_prices,
        )
        df = adapter.to_country_data(opt_input)
        assert df['price_afrr_energy_pos'].isna().sum() == n_zeros
        assert df['price_afrr_energy_neg'].isna().sum() == n_zeros

    def test_round_trip_preserves_nan(self, adapter, sample_block_prices):
        """adapt() then to_country_data() should preserve 0→NaN conversion."""
        market_prices = {
            'day_ahead': [50.0] * 192,
            'afrr_energy_pos': [0.0, 100.0] * 96,  # alternating 0 and 100
            'afrr_energy_neg': [100.0, 0.0] * 96,
            'fcr': sample_block_prices,
            'afrr_capacity_pos': sample_block_prices,
            'afrr_capacity_neg': sample_block_prices,
        }
        import numpy as np
        opt_input = adapter.adapt(market_prices)
        df = adapter.to_country_data(opt_input)
        # Every other value should be NaN
        assert np.isnan(df['price_afrr_energy_pos'].iloc[0])
        assert df['price_afrr_energy_pos'].iloc[1] == 100.0
        assert df['price_afrr_energy_neg'].iloc[0] == 100.0
        assert np.isnan(df['price_afrr_energy_neg'].iloc[1])


class TestExtract15minNoneHandling:
    """_extract_15min_prices must handle None (from JSON null) gracefully."""

    def test_none_becomes_nan(self):
        prices = {'afrr_energy_pos': [None, 50.0, None, 100.0]}
        result = DataAdapter._extract_15min_prices(prices, 'afrr_energy_pos')
        assert math.isnan(result[0])
        assert result[1] == 50.0
        assert math.isnan(result[2])
        assert result[3] == 100.0

    def test_mixed_none_and_zero(self):
        prices = {'key': [None, 0.0, 50.0]}
        result = DataAdapter._extract_15min_prices(prices, 'key')
        assert math.isnan(result[0])  # None → NaN
        assert result[1] == 0.0       # 0 stays 0 (conversion to NaN happens in to_country_data)
        assert result[2] == 50.0


class TestEdgeCases:
    def test_short_horizon(self, adapter, sample_15min_prices, sample_block_prices):
        """4-hour horizon = 16 timesteps, 1 block."""
        opt_input = OptimizationInput(
            time_horizon_hours=4,
            da_prices=sample_15min_prices[:16],
            afrr_energy_pos=sample_15min_prices[:16],
            afrr_energy_neg=sample_15min_prices[:16],
            fcr_prices=sample_block_prices[:1],
            afrr_capacity_pos=sample_block_prices[:1],
            afrr_capacity_neg=sample_block_prices[:1],
        )
        df = adapter.to_country_data(opt_input)
        assert len(df) == 16
        assert df['block_id'].nunique() == 1

    def test_expand_block_prices_padding(self):
        """If fewer block prices than needed, last price is forward-filled."""
        expanded = DataAdapter._expand_block_prices([100.0], n_timesteps=32)
        assert len(expanded) == 32
        # First 16 from the actual block, rest forward-filled
        assert all(v == 100.0 for v in expanded)

    def test_expand_block_prices_truncation(self):
        """If more block prices than needed, result is truncated."""
        expanded = DataAdapter._expand_block_prices([1.0, 2.0, 3.0], n_timesteps=20)
        assert len(expanded) == 20
