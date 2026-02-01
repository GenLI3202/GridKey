# Phase 4: Deployment

**Parent Plan:** [ibm_hack_dev_plan.md](ibm_hack_dev_plan.md)

**Dependencies:** Phase 3 (`optimizer_service.py`) must be complete

**Parallel Tasks:** All items in this phase can be implemented simultaneously.

---

## 4.1 [NEW] Dockerfile — Container Configuration

Location: `GridKey/Dockerfile`

**Depends on:** `optimizer_service.py` (3.1)

> **Note:** Using solver-optimized base image per design decision.

```dockerfile
# GridKey Optimizer Service - Production Docker Image
# Using solver-optimized image for better MILP performance
FROM python:3.11-slim

# Metadata
LABEL maintainer="GridKey Team"
LABEL version="1.0"
LABEL description="BESS Optimizer Service with Renewable Integration"

# Set working directory
WORKDIR /app

# Install system dependencies for solvers AND curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgmp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt requirements-api.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 4.2 [NEW] FastAPI main.py — API Endpoints

Location: `src/api/main.py`

**Depends on:** `optimizer_service.py` (3.1)

> [!IMPORTANT]
> Also create `src/api/__init__.py` (empty file) for proper Python module structure.

```python
"""
GridKey Optimizer Service API
=============================

FastAPI endpoints for BESS optimization with renewable integration.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from src.service.optimizer_service import OptimizerService
from src.service.models import OptimizationResult

app = FastAPI(
    title="GridKey Optimizer Service",
    description="BESS Optimizer with Renewable Integration",
    version="1.0.0"
)

logger = logging.getLogger(__name__)
service = OptimizerService()


class MarketPrices(BaseModel):
    """Market price data structure matching DataAdapter expectations."""
    day_ahead: List[float] = Field(..., description="Day-ahead prices (EUR/MWh), 15-min resolution")
    afrr_energy_pos: List[float] = Field(..., description="aFRR+ energy prices (EUR/MWh), 15-min resolution")
    afrr_energy_neg: List[float] = Field(..., description="aFRR- energy prices (EUR/MWh), 15-min resolution")
    fcr: List[float] = Field(..., description="FCR capacity prices (EUR/MW), 4-hour blocks")
    afrr_capacity_pos: List[float] = Field(..., description="aFRR+ capacity prices (EUR/MW), 4-hour blocks")
    afrr_capacity_neg: List[float] = Field(..., description="aFRR- capacity prices (EUR/MW), 4-hour blocks")


class OptimizeRequest(BaseModel):
    """API request for optimization."""
    location: str = "Munich"
    country: str = "DE_LU"
    model_type: str = Field(default="III", description="Model type: I, II, III, or III-renew")
    c_rate: float = Field(default=0.5, description="Battery C-rate (0.25, 0.33, 0.5)")
    alpha: float = Field(default=1.0, description="Degradation cost weight")
    daily_cycle_limit: float = Field(default=1.0, description="Maximum daily cycles")
    time_horizon_hours: int = Field(default=48, description="Optimization horizon in hours")
    
    # Market prices (required)
    market_prices: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Market price data. Keys: day_ahead, afrr_energy_pos, afrr_energy_neg, fcr, afrr_capacity_pos, afrr_capacity_neg"
    )
    
    # Renewable forecast (optional) - 15-min resolution in kW
    renewable_generation: Optional[List[float]] = Field(
        default=None,
        description="Renewable generation forecast (kW), 15-min resolution"
    )


class OptimizeResponse(BaseModel):
    """API response wrapper."""
    status: str
    data: Dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "solver_available": "highs"
    }


@app.post("/api/v1/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Run BESS optimization with optional renewable integration.
    
    Required: market_prices with all 6 price arrays
    Optional: renewable_generation for Model III-renew
    """
    # Validate required input
    if not request.market_prices:
        raise HTTPException(
            status_code=400, 
            detail="market_prices is required. Must include: day_ahead, afrr_energy_pos, afrr_energy_neg, fcr, afrr_capacity_pos, afrr_capacity_neg"
        )
    
    # Validate required keys
    required_keys = ['day_ahead', 'afrr_energy_pos', 'afrr_energy_neg', 'fcr', 'afrr_capacity_pos', 'afrr_capacity_neg']
    missing_keys = [k for k in required_keys if k not in request.market_prices]
    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required market_prices keys: {missing_keys}"
        )
    
    try:
        # Prepare generation forecast with correct key for DataAdapter
        generation_forecast = None
        if request.renewable_generation:
            generation_forecast = {"generation_kw": request.renewable_generation}
        
        result = service.optimize(
            market_prices=request.market_prices,
            generation_forecast=generation_forecast,
            model_type=request.model_type,
            c_rate=request.c_rate,
            alpha=request.alpha,
            daily_cycle_limit=request.daily_cycle_limit,
            time_horizon_hours=request.time_horizon_hours,
        )
        
        # Use model_dump() for Pydantic v2 (not deprecated .dict())
        return OptimizeResponse(status="success", data=result.model_dump())
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
```

---

## 4.3 API Input/Output Format Specification

### Optimize Endpoint

**Endpoint:** `POST /api/v1/optimize`

#### Request Body (JSON)

> [!NOTE]
> Market price keys must match DataAdapter expectations exactly.

```json
{
  "location": "Munich",
  "country": "DE_LU",
  "model_type": "III-renew",
  "c_rate": 0.5,
  "alpha": 1.0,
  "daily_cycle_limit": 1.0,
  "time_horizon_hours": 48,
  
  "market_prices": {
    "day_ahead": [39.91, -0.04, -9.01, "... 192 values for 48h"],
    "fcr": [114.8, 104.4, 68.8, "... 12 values for 48h"],
    "afrr_capacity_pos": [6.33, 4.12, "... 12 values"],
    "afrr_capacity_neg": [13.07, 15.02, "... 12 values"],
    "afrr_energy_pos": [50.34, 46.94, "... 192 values"],
    "afrr_energy_neg": [29.70, 40.87, "... 192 values"]
  },
  
  "renewable_generation": [0, 0, 0, 10.5, 25.3, 45.2, "... 192 values in kW"]
}
```

#### Response Body (JSON)

```json
{
  "status": "success",
  "data": {
    "objective_value": 1847.52,
    "net_profit": 1523.18,
  
    "revenue_breakdown": {
      "da": 892.45,
      "fcr": 324.80,
      "afrr_energy": 98.33,
      "renewable_export": 75.82
    },
  
    "degradation_cost": 324.34,
    "cyclic_aging_cost": 287.12,
    "calendar_aging_cost": 37.22,
  
    "renewable_utilization": {
      "total_generation_kwh": 185.5,
      "self_consumption_kwh": 92.3,
      "export_kwh": 85.7,
      "curtailment_kwh": 7.5,
      "utilization_rate": 0.96
    },
  
    "schedule": [
      {
        "timestamp": "2024-01-01T00:00:00",
        "action": "charge",
        "power_kw": 1500.0,
        "market": "da",
        "renewable_action": null,
        "renewable_power_kw": 0,
        "soc_after": 0.58
      }
    ],
  
    "soc_trajectory": [0.50, 0.58, 0.67, "..."],
  
    "solve_time_seconds": 12.45,
    "solver_name": "highs",
    "model_type": "III-renew",
    "status": "optimal",
    "num_variables": 15234,
    "num_constraints": 28456
  }
}
```

### Health Check Endpoint

**Endpoint:** `GET /health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "solver_available": "highs"
}
```

---

## 4.4 [NEW] GitHub Actions CI Workflow

Location: `.github/workflows/test.yml`

```yaml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-api.txt
          pip install pytest pytest-cov
          
      - name: Run unit tests
        run: pytest src/test/ -v --cov=src/service --cov-report=xml
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          
  docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t gridkey-optimizer:test .
        
      - name: Run container health check
        run: |
          docker run -d -p 8000:8000 --name test-container gridkey-optimizer:test
          sleep 15
          curl -f http://localhost:8000/health
          docker stop test-container
```

---

## 4.5 Integration Tests

Location: `src/test/test_integration.py`

### Docker Container Test

```bash
# Build image
docker build -t gridkey-optimizer:latest .

# Run container
docker run -d -p 8000:8000 --name gridkey-opt gridkey-optimizer:latest

# Test health endpoint
curl http://localhost:8000/health

# Test optimize endpoint (with market data)
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "country": "DE_LU",
    "model_type": "III",
    "time_horizon_hours": 24,
    "market_prices": {
      "day_ahead": [50.0, 51.0, 52.0],
      "afrr_energy_pos": [40.0, 41.0, 42.0],
      "afrr_energy_neg": [30.0, 31.0, 32.0],
      "fcr": [100.0],
      "afrr_capacity_pos": [5.0],
      "afrr_capacity_neg": [10.0]
    }
  }'

# Cleanup
docker stop gridkey-opt && docker rm gridkey-opt
```

### API Integration Test (pytest)

```python
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
    """Generate sample market prices for 24h (96 timesteps, 6 blocks)."""
    return {
        "day_ahead": [50.0 + i * 0.1 for i in range(96)],
        "afrr_energy_pos": [40.0] * 96,
        "afrr_energy_neg": [30.0] * 96,
        "fcr": [100.0] * 6,
        "afrr_capacity_pos": [5.0] * 6,
        "afrr_capacity_neg": [10.0] * 6,
    }


@pytest.fixture
def sample_renewable_generation():
    """Generate sample PV generation for 24h."""
    from src.test.fixtures.generate_synthetic_data import generate_synthetic_renewable
    return generate_synthetic_renewable(hours=24, peak_kw=100.0)


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
            "time_horizon_hours": 24
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
            "time_horizon_hours": 24,
            "market_prices": sample_market_prices
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "objective_value" in data["data"]

    def test_with_renewable_integration(self, sample_market_prices, sample_renewable_generation):
        response = client.post("/api/v1/optimize", json={
            "model_type": "III-renew",
            "time_horizon_hours": 24,
            "market_prices": sample_market_prices,
            "renewable_generation": sample_renewable_generation
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        # III-renew should include renewable utilization
        if data["data"].get("renewable_utilization"):
            assert "total_generation_kwh" in data["data"]["renewable_utilization"]
```

---

## 4.6 Synthetic Test Data Generator

Location: `src/test/fixtures/generate_synthetic_data.py`

> **Note:** Using synthetic data per design decision.

```python
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
```

---

## 4.7 Required Files Checklist

Before deployment, ensure these files exist:

| File | Purpose | Status |
|------|---------|--------|
| `src/api/__init__.py` | Python module marker | [NEW] |
| `src/api/main.py` | FastAPI application | [NEW] |
| `src/test/fixtures/__init__.py` | Test fixtures module | [NEW] |
| `src/test/fixtures/generate_synthetic_data.py` | Synthetic data generators | [NEW] |
| `src/test/test_integration.py` | API integration tests | [NEW] |
| `Dockerfile` | Container configuration | [NEW] |
| `.github/workflows/test.yml` | CI pipeline | [NEW] |

---

## Verification Checklist (Phase 4)

- [ ] `src/api/__init__.py` exists
- [ ] Docker image builds successfully (`docker build -t gridkey-optimizer .`)
- [ ] Container health check passes
- [ ] `/health` endpoint returns `{"status": "healthy"}`
- [ ] `/api/v1/optimize` validates missing `market_prices`
- [ ] `/api/v1/optimize` returns valid response with synthetic data
- [ ] GitHub Actions workflow runs without errors
- [ ] Integration tests pass with synthetic data
- [ ] API response matches schema specification
