# GridKey Optimizer API - User Guide

**Version:** 1.0.0
**Date:** 2026-02-01
**Base URL:** `http://localhost:8000`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Starting the Service](#starting-the-service)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Format](#requestresponse-format)
5. [Usage Examples](#usage-examples)
6. [Data Format Reference](#data-format-reference)
7. [Error Handling](#error-handholding)

---

## Quick Start

### Windows (PowerShell)

```powershell
# Navigate to project directory
cd D:\my_projects\GridPro\GridKey

# Activate conda environment (if needed)
conda activate gridkey

# Start the service
.\startup.ps1 dev

# Or directly with uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Linux/macOS

```bash
cd D:/my_projects/GridPro/GridKey
./startup.sh dev
```

Then open browser: **http://localhost:8000/docs**

---

## Starting the Service

### Windows (PowerShell) - Recommended

```powershell
# Make sure you're in the GridKey directory
cd D:\my_projects\GridPro\GridKey

# Activate conda environment
conda activate gridkey

# Option 1: Use the PowerShell script (dev mode with hot-reload)
.\startup.ps1 dev

# Option 2: Direct uvicorn (simplest)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Production mode (no reload)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Note:** The `.sh` script does **not** work on Windows. Use `startup.ps1` or run uvicorn directly.

### Linux/macOS

```bash
# Production mode
./startup.sh

# Development mode (hot-reload)
./startup.sh dev

# Run tests
./startup.sh test
```

### Docker (Optional)

**Note:** Docker is **not required** to run the service. Only use Docker if you want containerized deployment.

```bash
# Build image
docker build -t gridkey-optimizer .

# Run container
docker run -p 8000:8000 gridkey-optimizer
```

### Service Access

| Resource | URL |
|----------|-----|
| API Documentation (Swagger UI) | http://localhost:8000/docs |
| API Documentation (ReDoc) | http://localhost:8000/redoc |
| Health Check | http://localhost:8000/health |

---

## API Endpoints

### 1. Health Check

Check service status and solver configuration.

```
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "solver": "auto-detected",
    "max_horizon_hours": 12,
    "recommended_horizon_hours": 6,
    "note": "HiGHS (open-source) - use 6-hour rolling horizon for best performance"
}
```

---

### 2. Standard Optimization

Optimize BESS operation for specified time horizon (1-12 hours).

```
POST /api/v1/optimize
```

**Request Body:**
```json
{
    "location": "Munich",
    "country": "DE_LU",
    "model_type": "III",
    "c_rate": 0.5,
    "alpha": 1.0,
    "daily_cycle_limit": 1.0,
    "time_horizon_hours": 6,
    "market_prices": {
        "day_ahead": [50.0, ...],
        "afrr_energy_pos": [40.0, ...],
        "afrr_energy_neg": [30.0, ...],
        "fcr": [100.0, 105.0],
        "afrr_capacity_pos": [5.0, 6.0],
        "afrr_capacity_neg": [10.0, 11.0]
    },
    "renewable_generation": [0.0, ...]  // Optional
}
```

**Response:**
```json
{
    "status": "success",
    "data": {
        "objective_value": 1250.50,
        "net_profit": 1225.50,
        "revenue_breakdown": {
            "da": 1000.0,
            "afrr_energy": 150.0,
            "fcr": 100.5
        },
        "degradation_cost": 25.0,
        "schedule": [
            {
                "timestamp": "2024-01-01T00:00:00",
                "action": "charge",
                "power_kw": 500.0,
                "market": "da",
                "soc_after": 0.55
            }
        ],
        "soc_trajectory": [0.50, 0.55, 0.60, ...],
        "solve_time_seconds": 5.2,
        "solver_name": "highs",
        "model_type": "III",
        "status": "optimal"
    }
}
```

---

### 3. MPC 12h Rolling Horizon

12-hour optimization using Model Predictive Control with 6h window and 4h roll step.

```
POST /api/v1/optimize-mpc
```

**Strategy:**
- Iteration 1: Optimize [0h-6h], commit [0h-4h]
- Iteration 2: Optimize [4h-10h], commit [4h-8h]
- Iteration 3: Optimize [8h-12h], commit [8h-12h]

**Estimated Time:** 15-20 seconds (3 iterations × ~5 sec each)

**Request Body:**
```json
{
    "model_type": "III",
    "c_rate": 0.5,
    "alpha": 1.0,
    "market_prices": {
        "day_ahead": [50.0, ..., 50.0],  // 48 values
        "afrr_energy_pos": [40.0, ..., 40.0],  // 48 values
        "afrr_energy_neg": [30.0, ..., 30.0],  // 48 values
        "fcr": [100.0, 105.0, 110.0],  // 3 blocks
        "afrr_capacity_pos": [5.0, 6.0, 7.0],
        "afrr_capacity_neg": [10.0, 11.0, 12.0]
    },
    "renewable_generation": [0.0, ..., 0.0]  // Optional, 48 values
}
```

---

## Request/Response Format

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `location` | string | "Munich" | Location name |
| `country` | string | "DE_LU" | Country code |
| `model_type` | string | "III" | "I", "II", "III", or "III-renew" |
| `c_rate` | float | 0.5 | Battery C-rate (0.25, 0.33, 0.5) |
| `alpha` | float | 1.0 | Degradation cost weight |
| `daily_cycle_limit` | float | 1.0 | Max daily cycles |
| `time_horizon_hours` | int | 6 | Optimization horizon (1-12) |
| `market_prices` | object | Required | Price data (see below) |
| `renewable_generation` | array | Optional | Generation forecast (kW) |

### Market Prices Structure

| Field | Resolution | Description |
|-------|------------|-------------|
| `day_ahead` | 15-min | Day-ahead energy prices (EUR/MWh) |
| `afrr_energy_pos` | 15-min | aFRR+ energy prices (EUR/MWh) |
| `afrr_energy_neg` | 15-min | aFRR- energy prices (EUR/MWh) |
| `fcr` | 4-hour blocks | FCR capacity prices (EUR/MW) |
| `afrr_capacity_pos` | 4-hour blocks | aFRR+ capacity prices (EUR/MW) |
| `afrr_capacity_neg` | 4-hour blocks | aFRR- capacity prices (EUR/MW) |

### Response Fields

| Field | Description |
|-------|-------------|
| `objective_value` | Raw objective function value |
| `net_profit` | Profit after degradation costs |
| `revenue_breakdown` | Revenue by market (da, afrr_energy, fcr) |
| `degradation_cost` | Total degradation cost |
| `schedule` | Array of per-timestep actions |
| `soc_trajectory` | Normalized SOC values [0, 1] |
| `solve_time_seconds` | Solver execution time |
| `solver_name` | Solver used (highs, cplex, gurobi) |

---

## Usage Examples

### Example 1: Basic 6h Optimization (cURL)

```bash
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "III",
    "c_rate": 0.5,
    "time_horizon_hours": 6,
    "market_prices": {
      "day_ahead": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45],
      "afrr_energy_pos": [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38],
      "afrr_energy_neg": [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28],
      "fcr": [100.0, 105.0],
      "afrr_capacity_pos": [5.0, 6.0],
      "afrr_capacity_neg": [10.0, 11.0]
    }
  }'
```

### Example 2: With Renewable Generation (Python)

```python
import requests

API_URL = "http://localhost:8000"

response = requests.post(f"{API_URL}/api/v1/optimize", json={
    "model_type": "III-renew",
    "c_rate": 0.5,
    "time_horizon_hours": 6,
    "market_prices": {
        "day_ahead": [50.0] * 24,
        "afrr_energy_pos": [40.0] * 24,
        "afrr_energy_neg": [30.0] * 24,
        "fcr": [100.0, 105.0],
        "afrr_capacity_pos": [5.0, 6.0],
        "afrr_capacity_neg": [10.0, 11.0]
    },
    "renewable_generation": [
        0, 0, 0, 50, 150, 300, 450, 500, 480, 400, 250, 100,
        0, 0, 0, 0, 0, 0, 0, 50, 200, 400, 300, 100
    ]
})

result = response.json()
print(f"Status: {result['status']}")
print(f"Net Profit: {result['data']['net_profit']:.2f} EUR")
print(f"Solve Time: {result['data']['solve_time_seconds']:.2f}s")

# Access schedule
for entry in result['data']['schedule'][:5]:
    print(f"{entry['timestamp']}: {entry['action']} {entry['power_kw']}kW -> SOC {entry['soc_after']:.2%}")
```

### Example 3: MPC 12h (Python)

```python
import requests

API_URL = "http://localhost:8000"

# Generate 12h price data (48 values @ 15-min)
day_ahead = [50 + i for i in range(48)]  # Rising price pattern

response = requests.post(f"{API_URL}/api/v1/optimize-mpc", json={
    "model_type": "III",
    "c_rate": 0.5,
    "market_prices": {
        "day_ahead": day_ahead,
        "afrr_energy_pos": [40.0] * 48,
        "afrr_energy_neg": [30.0] * 48,
        "fcr": [100.0, 105.0, 110.0],
        "afrr_capacity_pos": [5.0, 6.0, 7.0],
        "afrr_capacity_neg": [10.0, 11.0, 12.0]
    }
})

result = response.json()
print(f"MPC 12h Complete")
print(f"Net Profit: {result['data']['net_profit']:.2f} EUR")
print(f"Total Time: {result['data']['solve_time_seconds']:.2f}s")
print(f"Schedule Length: {len(result['data']['schedule'])} timesteps")
```

---

## Data Format Reference

### Time Horizon vs Data Length

| Time Horizon | 15-min Prices | 4-hour Block Prices |
|--------------|---------------|---------------------|
| 1 hour | 4 values | 1 block |
| 6 hours | 24 values | 2 blocks |
| 12 hours | 48 values | 3 blocks |

### Model Types

| model_type | Description | Use Case |
|------------|-------------|----------|
| `"I"` | Base 4-market model | Fastest, no degradation |
| `"II"` | + Cyclic aging cost | Cycle-focused degradation |
| `"III"` | + Calendar aging cost | Full degradation (recommended) |
| `"III-renew"` | + Renewable integration | With solar/wind generation |

### C-Rate Options

| C-Rate | Max Power (4472 kWh) | Description |
|--------|---------------------|-------------|
| 0.25 | 1118 kW | Conservative |
| 0.33 | 1492 kW | Moderate |
| 0.5 | 2236 kW | Aggressive (default) |

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (validation error) |
| 500 | Internal Server Error |

### Example Error Response

```json
{
    "detail": "time_horizon_hours=24 exceeds maximum of 12 for auto-detected solver. HiGHS max: 12h, Commercial max: 48h. Use rolling 6-hour windows for continuous operation."
}
```

### Common Errors

| Error | Solution |
|-------|----------|
| Missing `market_prices` | Include all 6 price arrays |
| Wrong array length | Match data length to time_horizon_hours |
| Invalid `model_type` | Use "I", "II", "III", or "III-renew" |
| Horizon too long | Use MPC endpoint for 12h, or reduce horizon |

---

## Performance Benchmarks

| Operation | Estimated Time |
|-----------|----------------|
| 6h optimization (HiGHS) | ~5 seconds |
| 12h MPC (3 iterations) | ~15-20 seconds |
| API overhead | +1-2 seconds |

---

## Interactive Testing

Use Swagger UI for interactive API exploration:

1. **Start service:**
   - Windows: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
   - Or: `.\startup.ps1 dev`

2. **Open browser:** http://localhost:8000/docs

3. Click **Try it out** on any endpoint

4. Fill in request body

5. Click **Execute**

---

## Support

- **Documentation:** `doc/dev_plans/implemented/`
- **Issue Tracker:** GitHub Issues
- **Development Plan:** `doc/dev_plans/ibm_hack_dev_plan.md`
