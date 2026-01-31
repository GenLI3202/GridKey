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

# Install system dependencies for solvers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgmp-dev \
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

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
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


class OptimizeRequest(BaseModel):
    location: str = "Munich"
    country: str = "DE_LU"
    model_type: str = "III-renew"
    c_rate: float = 0.5
    alpha: float = 1.0
    time_horizon_hours: int = 48
    market_prices: Optional[dict] = None
    renewable_generation: Optional[List[float]] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "solver_available": "highs"
    }


@app.post("/api/v1/optimize", response_model=dict)
async def optimize(request: OptimizeRequest):
    """
    Run BESS optimization with optional renewable integration.
    """
    try:
        result = service.optimize(
            market_prices=request.market_prices or {},
            generation_forecast={"pv": request.renewable_generation} if request.renewable_generation else None,
            model_type=request.model_type,
            c_rate=request.c_rate,
            alpha=request.alpha,
            time_horizon_hours=request.time_horizon_hours,
        )
        return {"status": "success", "data": result.dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail="Optimization failed")
```

---

## 4.3 API Input/Output Format Specification

### Optimize Endpoint

**Endpoint:** `POST /api/v1/optimize`

#### Request Body (JSON)

```json
{
  "location": "Munich",
  "country": "DE_LU",
  "model_type": "III-renew",
  "c_rate": 0.5,
  "alpha": 1.0,
  "time_horizon_hours": 48,
  
  "market_prices": {
    "da_prices": [39.91, -0.04, -9.01, "..."],
    "fcr_prices": [114.8, 104.4, 68.8, "..."],
    "afrr_capacity_pos": [6.33, 4.12, "..."],
    "afrr_capacity_neg": [13.07, 15.02, "..."],
    "afrr_energy_pos": [50.34, 46.94, "..."],
    "afrr_energy_neg": [29.70, 40.87, "..."]
  },
  
  "renewable_generation": [0, 0, 0, 10.5, 25.3, 45.2, "..."]
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
      "day_ahead": 892.45,
      "fcr": 324.80,
      "afrr_capacity": 456.12,
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
          sleep 10
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

# Test optimize endpoint
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"country": "DE_LU", "model_type": "III-renew", "time_horizon_hours": 24}'

# Cleanup
docker stop gridkey-opt && docker rm gridkey-opt
```

### API Integration Test (pytest)

```python
# src/test/test_integration.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_optimize_endpoint_minimal():
    response = client.post("/api/v1/optimize", json={
        "country": "DE_LU",
        "model_type": "III-renew",
        "time_horizon_hours": 24
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "objective_value" in data["data"]
```

---

## 4.6 Synthetic Test Data Generator

Location: `src/test/fixtures/generate_synthetic_data.py`

> **Note:** Using synthetic data per design decision.

```python
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_renewable(
    hours: int = 48,
    peak_kw: float = 100.0,
    noise_factor: float = 0.1
) -> list[float]:
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


def generate_synthetic_market_prices(hours: int = 48) -> dict:
    """Generate synthetic market price data."""
    timesteps_15min = hours * 4
    blocks_4h = hours // 4
    
    return {
        "da_prices": list(np.random.uniform(20, 80, timesteps_15min)),
        "fcr_prices": list(np.random.uniform(50, 150, blocks_4h)),
        "afrr_capacity_pos": list(np.random.uniform(3, 10, blocks_4h)),
        "afrr_capacity_neg": list(np.random.uniform(5, 15, blocks_4h)),
        "afrr_energy_pos": list(np.random.uniform(30, 70, timesteps_15min)),
        "afrr_energy_neg": list(np.random.uniform(20, 50, timesteps_15min)),
    }
```

---

## Verification Checklist (Phase 4)

- [ ] Docker image builds successfully
- [ ] Container health check passes
- [ ] `/api/v1/optimize` returns valid response
- [ ] GitHub Actions workflow runs without errors
- [ ] Integration tests pass with synthetic data
- [ ] API response matches schema specification
