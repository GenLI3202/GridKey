# src/test/test_integration.py
"""
Integration tests for GridKey Optimizer API.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_market_prices():
    """Generate sample market prices for 6h (24 timesteps, 2 blocks) - API default."""
    return {
        "day_ahead": [50.0 + i * 0.1 for i in range(24)],
        "afrr_energy_pos": [40.0] * 24,
        "afrr_energy_neg": [30.0] * 24,
        "fcr": [100.0] * 2,
        "afrr_capacity_pos": [5.0] * 2,
        "afrr_capacity_neg": [10.0] * 2,
    }


@pytest.fixture
def sample_renewable_generation():
    """Generate sample PV generation for 6h - API default."""
    from src.test.fixtures.generate_synthetic_data import generate_synthetic_renewable
    return generate_synthetic_renewable(hours=6, peak_kw=100.0)


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_healthy(self):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_includes_version(self):
        response = client.get("/health")
        assert "version" in response.json()


# ---------------------------------------------------------------------------
# Optimize Endpoint Tests
# ---------------------------------------------------------------------------

class TestOptimizeEndpoint:
    def test_missing_market_prices_returns_400(self):
        response = client.post("/api/v1/optimize", json={
            "country": "DE_LU",
            "model_type": "III",
            "time_horizon_hours": 6
        })
        assert response.status_code == 400
        assert "market_prices is required" in response.json()["detail"]

    def test_missing_price_keys_returns_400(self):
        response = client.post("/api/v1/optimize", json={
            "model_type": "III",
            "market_prices": {"day_ahead": [50.0] * 96}
        })
        assert response.status_code == 400
        assert "Missing required" in response.json()["detail"]

    def test_valid_request_returns_200(self, sample_market_prices):
        response = client.post("/api/v1/optimize", json={
            "model_type": "III",
            "time_horizon_hours": 6,
            "market_prices": sample_market_prices
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "objective_value" in data["data"]

    def test_with_renewable_integration(self, sample_market_prices, sample_renewable_generation):
        response = client.post("/api/v1/optimize", json={
            "model_type": "III-renew",
            "time_horizon_hours": 6,
            "market_prices": sample_market_prices,
            "renewable_generation": sample_renewable_generation
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # III-renew should include renewable utilization
        if data["data"].get("renewable_utilization"):
            assert "total_generation_kwh" in data["data"]["renewable_utilization"]
