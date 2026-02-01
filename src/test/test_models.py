"""
Tests for src/service/models.py — Pydantic data model validation.

Phase 1 verification: ensures all Pydantic models serialize,
validate, and reject invalid inputs correctly.
"""

import pytest
import json
from datetime import datetime
from pydantic import ValidationError

from src.service.models import (
    ModelType,
    OptimizationInput,
    ScheduleEntry,
    RenewableUtilization,
    OptimizationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices_15min():
    """192 entries = 48h at 15-min resolution."""
    return [50.0 + i * 0.1 for i in range(192)]


@pytest.fixture
def sample_prices_4h():
    """12 entries = 48h at 4-hour blocks."""
    return [100.0 + i for i in range(12)]


@pytest.fixture
def valid_optimization_input(sample_prices_15min, sample_prices_4h):
    return OptimizationInput(
        da_prices=sample_prices_15min,
        afrr_energy_pos=sample_prices_15min,
        afrr_energy_neg=sample_prices_15min,
        fcr_prices=sample_prices_4h,
        afrr_capacity_pos=sample_prices_4h,
        afrr_capacity_neg=sample_prices_4h,
    )


@pytest.fixture
def valid_schedule_entry():
    return ScheduleEntry(
        timestamp=datetime(2024, 5, 12, 0, 0),
        action="charge",
        power_kw=1000.0,
        market="da",
        soc_after=0.6,
    )


@pytest.fixture
def valid_renewable_utilization():
    return RenewableUtilization(
        total_generation_kwh=100.0,
        self_consumption_kwh=60.0,
        export_kwh=30.0,
        curtailment_kwh=10.0,
        utilization_rate=0.9,
    )


@pytest.fixture
def valid_optimization_result(valid_schedule_entry, valid_renewable_utilization):
    return OptimizationResult(
        objective_value=450.0,
        net_profit=420.0,
        revenue_breakdown={
            "da": 200.0,
            "fcr": 100.0,
            "afrr_cap": 80.0,
            "afrr_energy": 50.0,
            "renewable_export": 20.0,
        },
        degradation_cost=30.0,
        cyclic_aging_cost=20.0,
        calendar_aging_cost=10.0,
        schedule=[valid_schedule_entry],
        soc_trajectory=[0.5, 0.6],
        renewable_utilization=valid_renewable_utilization,
        solve_time_seconds=1.23,
        solver_name="highs",
        model_type=ModelType.MODEL_III,
        status="optimal",
        num_variables=5000,
        num_constraints=3000,
    )


# ---------------------------------------------------------------------------
# ModelType
# ---------------------------------------------------------------------------

class TestModelType:
    def test_all_values(self):
        assert ModelType.MODEL_I.value == "I"
        assert ModelType.MODEL_II.value == "II"
        assert ModelType.MODEL_III.value == "III"
        assert ModelType.MODEL_III_RENEW.value == "III-renew"

    def test_from_string(self):
        assert ModelType("III") is ModelType.MODEL_III
        assert ModelType("III-renew") is ModelType.MODEL_III_RENEW

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ModelType("IV")

    def test_is_string(self):
        """ModelType inherits from str, so it can be used as a string."""
        assert isinstance(ModelType.MODEL_I, str)
        assert ModelType.MODEL_III == "III"


# ---------------------------------------------------------------------------
# OptimizationInput
# ---------------------------------------------------------------------------

class TestOptimizationInput:
    def test_valid_construction(self, valid_optimization_input):
        obj = valid_optimization_input
        assert obj.time_horizon_hours == 48
        assert len(obj.da_prices) == 192
        assert len(obj.fcr_prices) == 12
        assert obj.model_type == ModelType.MODEL_III

    def test_defaults(self, sample_prices_15min, sample_prices_4h):
        obj = OptimizationInput(
            da_prices=sample_prices_15min,
            afrr_energy_pos=sample_prices_15min,
            afrr_energy_neg=sample_prices_15min,
            fcr_prices=sample_prices_4h,
            afrr_capacity_pos=sample_prices_4h,
            afrr_capacity_neg=sample_prices_4h,
        )
        assert obj.time_horizon_hours == 48
        assert obj.battery_capacity_kwh == 4472
        assert obj.c_rate == 0.5
        assert obj.efficiency == 0.95
        assert obj.initial_soc == 0.5
        assert obj.model_type == ModelType.MODEL_III
        assert obj.alpha == 1.0
        assert obj.renewable_generation is None

    def test_missing_required_field(self, sample_prices_15min, sample_prices_4h):
        with pytest.raises(ValidationError) as exc_info:
            OptimizationInput(
                # da_prices missing
                afrr_energy_pos=sample_prices_15min,
                afrr_energy_neg=sample_prices_15min,
                fcr_prices=sample_prices_4h,
                afrr_capacity_pos=sample_prices_4h,
                afrr_capacity_neg=sample_prices_4h,
            )
        assert "da_prices" in str(exc_info.value)

    def test_wrong_type(self, sample_prices_15min, sample_prices_4h):
        with pytest.raises(ValidationError):
            OptimizationInput(
                da_prices="not a list",
                afrr_energy_pos=sample_prices_15min,
                afrr_energy_neg=sample_prices_15min,
                fcr_prices=sample_prices_4h,
                afrr_capacity_pos=sample_prices_4h,
                afrr_capacity_neg=sample_prices_4h,
            )

    def test_efficiency_too_high(self, sample_prices_15min, sample_prices_4h):
        with pytest.raises(ValidationError, match="efficiency"):
            OptimizationInput(
                da_prices=sample_prices_15min,
                afrr_energy_pos=sample_prices_15min,
                afrr_energy_neg=sample_prices_15min,
                fcr_prices=sample_prices_4h,
                afrr_capacity_pos=sample_prices_4h,
                afrr_capacity_neg=sample_prices_4h,
                efficiency=1.5,
            )

    def test_efficiency_zero(self, sample_prices_15min, sample_prices_4h):
        with pytest.raises(ValidationError, match="efficiency"):
            OptimizationInput(
                da_prices=sample_prices_15min,
                afrr_energy_pos=sample_prices_15min,
                afrr_energy_neg=sample_prices_15min,
                fcr_prices=sample_prices_4h,
                afrr_capacity_pos=sample_prices_4h,
                afrr_capacity_neg=sample_prices_4h,
                efficiency=0.0,
            )

    def test_negative_c_rate(self, sample_prices_15min, sample_prices_4h):
        with pytest.raises(ValidationError, match="c_rate"):
            OptimizationInput(
                da_prices=sample_prices_15min,
                afrr_energy_pos=sample_prices_15min,
                afrr_energy_neg=sample_prices_15min,
                fcr_prices=sample_prices_4h,
                afrr_capacity_pos=sample_prices_4h,
                afrr_capacity_neg=sample_prices_4h,
                c_rate=-0.1,
            )

    def test_initial_soc_out_of_range(self, sample_prices_15min, sample_prices_4h):
        with pytest.raises(ValidationError, match="initial_soc"):
            OptimizationInput(
                da_prices=sample_prices_15min,
                afrr_energy_pos=sample_prices_15min,
                afrr_energy_neg=sample_prices_15min,
                fcr_prices=sample_prices_4h,
                afrr_capacity_pos=sample_prices_4h,
                afrr_capacity_neg=sample_prices_4h,
                initial_soc=1.5,
            )

    def test_with_renewable_generation(self, sample_prices_15min, sample_prices_4h):
        renewable = [10.0] * 192
        obj = OptimizationInput(
            da_prices=sample_prices_15min,
            afrr_energy_pos=sample_prices_15min,
            afrr_energy_neg=sample_prices_15min,
            fcr_prices=sample_prices_4h,
            afrr_capacity_pos=sample_prices_4h,
            afrr_capacity_neg=sample_prices_4h,
            renewable_generation=renewable,
            model_type=ModelType.MODEL_III_RENEW,
        )
        assert obj.renewable_generation is not None
        assert len(obj.renewable_generation) == 192
        assert obj.model_type == ModelType.MODEL_III_RENEW

    def test_custom_battery_config(self, sample_prices_15min, sample_prices_4h):
        obj = OptimizationInput(
            da_prices=sample_prices_15min,
            afrr_energy_pos=sample_prices_15min,
            afrr_energy_neg=sample_prices_15min,
            fcr_prices=sample_prices_4h,
            afrr_capacity_pos=sample_prices_4h,
            afrr_capacity_neg=sample_prices_4h,
            battery_capacity_kwh=2000,
            c_rate=0.25,
            efficiency=0.9,
            initial_soc=0.8,
        )
        assert obj.battery_capacity_kwh == 2000
        assert obj.c_rate == 0.25
        assert obj.efficiency == 0.9
        assert obj.initial_soc == 0.8


# ---------------------------------------------------------------------------
# ScheduleEntry
# ---------------------------------------------------------------------------

class TestScheduleEntry:
    def test_valid_construction(self, valid_schedule_entry):
        entry = valid_schedule_entry
        assert entry.action == "charge"
        assert entry.power_kw == 1000.0
        assert entry.market == "da"
        assert entry.soc_after == 0.6

    def test_optional_renewable_defaults(self, valid_schedule_entry):
        assert valid_schedule_entry.renewable_action is None
        assert valid_schedule_entry.renewable_power_kw is None

    def test_with_renewable_fields(self):
        entry = ScheduleEntry(
            timestamp=datetime(2024, 5, 12, 12, 0),
            action="idle",
            power_kw=0.0,
            market="da",
            renewable_action="self_consume",
            renewable_power_kw=500.0,
            soc_after=0.7,
        )
        assert entry.renewable_action == "self_consume"
        assert entry.renewable_power_kw == 500.0

    def test_soc_after_too_high(self):
        with pytest.raises(ValidationError, match="soc_after"):
            ScheduleEntry(
                timestamp=datetime(2024, 5, 12, 0, 0),
                action="charge",
                power_kw=1000.0,
                market="da",
                soc_after=1.5,
            )

    def test_soc_after_negative(self):
        with pytest.raises(ValidationError, match="soc_after"):
            ScheduleEntry(
                timestamp=datetime(2024, 5, 12, 0, 0),
                action="discharge",
                power_kw=1000.0,
                market="da",
                soc_after=-0.1,
            )

    def test_soc_after_boundary_values(self):
        """SOC of exactly 0 and 1 should be valid."""
        entry_zero = ScheduleEntry(
            timestamp=datetime(2024, 5, 12, 0, 0),
            action="discharge",
            power_kw=1000.0,
            market="da",
            soc_after=0.0,
        )
        assert entry_zero.soc_after == 0.0

        entry_full = ScheduleEntry(
            timestamp=datetime(2024, 5, 12, 0, 0),
            action="charge",
            power_kw=1000.0,
            market="da",
            soc_after=1.0,
        )
        assert entry_full.soc_after == 1.0


# ---------------------------------------------------------------------------
# RenewableUtilization
# ---------------------------------------------------------------------------

class TestRenewableUtilization:
    def test_valid_construction(self, valid_renewable_utilization):
        ru = valid_renewable_utilization
        assert ru.total_generation_kwh == 100.0
        assert ru.self_consumption_kwh == 60.0
        assert ru.export_kwh == 30.0
        assert ru.curtailment_kwh == 10.0
        assert ru.utilization_rate == 0.9

    def test_utilization_rate_too_high(self):
        with pytest.raises(ValidationError, match="utilization_rate"):
            RenewableUtilization(
                total_generation_kwh=100.0,
                self_consumption_kwh=60.0,
                export_kwh=30.0,
                curtailment_kwh=10.0,
                utilization_rate=1.5,
            )

    def test_utilization_rate_negative(self):
        with pytest.raises(ValidationError, match="utilization_rate"):
            RenewableUtilization(
                total_generation_kwh=100.0,
                self_consumption_kwh=60.0,
                export_kwh=30.0,
                curtailment_kwh=10.0,
                utilization_rate=-0.1,
            )

    def test_zero_generation(self):
        """Zero generation with zero utilization should be valid."""
        ru = RenewableUtilization(
            total_generation_kwh=0.0,
            self_consumption_kwh=0.0,
            export_kwh=0.0,
            curtailment_kwh=0.0,
            utilization_rate=0.0,
        )
        assert ru.utilization_rate == 0.0


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

class TestOptimizationResult:
    def test_valid_construction(self, valid_optimization_result):
        result = valid_optimization_result
        assert result.objective_value == 450.0
        assert result.net_profit == 420.0
        assert result.status == "optimal"
        assert result.model_type == ModelType.MODEL_III
        assert len(result.schedule) == 1
        assert result.renewable_utilization is not None

    def test_without_renewable(self, valid_schedule_entry):
        result = OptimizationResult(
            objective_value=300.0,
            net_profit=280.0,
            revenue_breakdown={"da": 200.0, "fcr": 80.0},
            degradation_cost=20.0,
            cyclic_aging_cost=15.0,
            calendar_aging_cost=5.0,
            schedule=[valid_schedule_entry],
            soc_trajectory=[0.5, 0.6],
            solve_time_seconds=0.5,
            solver_name="cplex",
            model_type=ModelType.MODEL_III,
            status="optimal",
        )
        assert result.renewable_utilization is None
        assert result.num_variables is None
        assert result.num_constraints is None

    def test_revenue_breakdown_keys(self, valid_optimization_result):
        breakdown = valid_optimization_result.revenue_breakdown
        assert "da" in breakdown
        assert "fcr" in breakdown
        assert "afrr_cap" in breakdown
        assert "afrr_energy" in breakdown
        assert "renewable_export" in breakdown

    def test_multiple_schedule_entries(self, valid_schedule_entry):
        entries = [
            valid_schedule_entry,
            ScheduleEntry(
                timestamp=datetime(2024, 5, 12, 0, 15),
                action="discharge",
                power_kw=800.0,
                market="afrr_energy",
                soc_after=0.45,
            ),
        ]
        result = OptimizationResult(
            objective_value=300.0,
            net_profit=280.0,
            revenue_breakdown={"da": 200.0},
            degradation_cost=20.0,
            cyclic_aging_cost=15.0,
            calendar_aging_cost=5.0,
            schedule=entries,
            soc_trajectory=[0.5, 0.6, 0.45],
            solve_time_seconds=0.5,
            solver_name="highs",
            model_type=ModelType.MODEL_I,
            status="optimal",
        )
        assert len(result.schedule) == 2
        assert result.schedule[1].action == "discharge"


# ---------------------------------------------------------------------------
# Serialization / Deserialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_optimization_input_roundtrip(self, valid_optimization_input):
        data = valid_optimization_input.model_dump()
        restored = OptimizationInput.model_validate(data)
        assert restored.time_horizon_hours == valid_optimization_input.time_horizon_hours
        assert restored.da_prices == valid_optimization_input.da_prices
        assert restored.model_type == valid_optimization_input.model_type

    def test_optimization_input_json_roundtrip(self, valid_optimization_input):
        json_str = valid_optimization_input.model_dump_json()
        restored = OptimizationInput.model_validate_json(json_str)
        assert restored.da_prices == valid_optimization_input.da_prices

    def test_optimization_result_roundtrip(self, valid_optimization_result):
        data = valid_optimization_result.model_dump()
        restored = OptimizationResult.model_validate(data)
        assert restored.objective_value == valid_optimization_result.objective_value
        assert restored.status == valid_optimization_result.status
        assert len(restored.schedule) == len(valid_optimization_result.schedule)
        assert restored.renewable_utilization is not None

    def test_optimization_result_json_roundtrip(self, valid_optimization_result):
        json_str = valid_optimization_result.model_dump_json()
        parsed = json.loads(json_str)
        assert "objective_value" in parsed
        assert "schedule" in parsed
        assert parsed["model_type"] == "III"

        restored = OptimizationResult.model_validate_json(json_str)
        assert restored.model_type == ModelType.MODEL_III

    def test_schedule_entry_json(self, valid_schedule_entry):
        json_str = valid_schedule_entry.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["action"] == "charge"
        assert parsed["soc_after"] == 0.6
        assert parsed["renewable_action"] is None

    def test_model_type_serialization(self):
        """ModelType should serialize as its string value."""
        data = {"model_type": "III-renew"}
        # Pydantic should accept the string value
        from pydantic import TypeAdapter
        adapter = TypeAdapter(ModelType)
        result = adapter.validate_python("III-renew")
        assert result == ModelType.MODEL_III_RENEW
        assert result.value == "III-renew"
