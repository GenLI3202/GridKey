# GridKey Optimizer API

- **Version:** 1.0.0
- **Base URL:** `http://localhost:8000`
- **Status:** Experimental

The GridKey API is a small HTTP wrapper around the repository's optimization
models. It demonstrates a complete request → optimization → schedule workflow,
but it is not a production EMS interface.

## What the API does

The API accepts caller-provided market-price arrays, builds one of the GridKey
optimization models, solves it, and returns a simplified schedule and summary.

It currently exposes three routes:

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Report the service version and configured solver information |
| `POST /api/v1/optimize` | Run one fixed-horizon optimization |
| `POST /api/v1/optimize-mpc` | Run the fixed 12-hour rolling-horizon experiment |

FastAPI also provides interactive documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Start the service

First complete the virtual-environment setup in the repository
[`README.md`](../README.md), then install the API dependencies:

```bash
python -m pip install -r requirements-api.txt
```

### macOS and Linux

```bash
export GRIDKEY_SOLVER=highs
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Windows PowerShell

```powershell
$env:GRIDKEY_SOLVER = "highs"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Docker is optional. It packages the same experimental API and is not required
to run the optimizer locally.

## `GET /health`

Example response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "solver": "highs",
  "max_horizon_hours": 12,
  "recommended_horizon_hours": 6,
  "note": "HiGHS (open-source) - use 6-hour rolling horizon for best performance"
}
```

This is a liveness-style endpoint. It reports configuration but does not run a
test optimization or verify that the solver can successfully solve a model.

## `POST /api/v1/optimize`

Runs one optimization over a caller-selected horizon.

### Request

| Field | Default | Current behavior |
| --- | --- | --- |
| `model_type` | `"III"` | Accepts `"I"`, `"II"`, `"III"`, or `"III-renew"` |
| `c_rate` | `0.5` | Sets the optimizer power-to-energy ratio |
| `alpha` | `1.0` | Weights degradation cost in Models II and III |
| `daily_cycle_limit` | `1.0` | Used by Model I; Model II/III deactivate this constraint |
| `time_horizon_hours` | `6` | Accepts 1–12 hours with the default/HiGHS configuration |
| `market_prices` | required | Six caller-provided price arrays |
| `renewable_generation` | omitted | Used by `"III-renew"` |
| `location` | `"Munich"` | Accepted but currently not used in the calculation |
| `country` | `"DE_LU"` | Accepted but currently not used in the calculation |

`market_prices` must contain:

| Key | Resolution |
| --- | --- |
| `day_ahead` | 15 minutes |
| `afrr_energy_pos` | 15 minutes |
| `afrr_energy_neg` | 15 minutes |
| `fcr` | 4-hour blocks |
| `afrr_capacity_pos` | 4-hour blocks |
| `afrr_capacity_neg` | 4-hour blocks |

For an `H`-hour horizon, each 15-minute array should contain `4 × H` values.
Capacity arrays should contain enough 4-hour blocks to cover the horizon.

The standard endpoint checks that all arrays exist and are non-empty, but it
does not currently enforce their exact lengths at the API boundary. Extra
15-minute values are truncated; insufficient values can fail later during data
adaptation. If block prices do not cover the horizon, the adapter repeats the
last block value.

### Model variants

| Model | Optimization content |
| --- | --- |
| I | Market co-optimization with SOC, power, reserve, and cycle constraints |
| II | Model I plus a segment-based cyclic aging cost |
| III | Model II plus a piecewise-linear calendar aging cost |
| III-renew | Model III plus simplified renewable self-use, export, and curtailment |

### Fixed battery assumptions

The API does not accept a complete battery configuration. The service currently
uses:

| Parameter | Value |
| --- | --- |
| Nominal capacity | 4,472 kWh |
| Charge/discharge efficiency | 0.95 |
| Initial SOC | 0.50 |

Only `c_rate` is exposed through the request.

### Minimal Python example

```python
import httpx

horizon_hours = 6
n_steps = horizon_hours * 4

response = httpx.post(
    "http://localhost:8000/api/v1/optimize",
    json={
        "model_type": "I",
        "c_rate": 0.5,
        "time_horizon_hours": horizon_hours,
        "market_prices": {
            "day_ahead": [50.0] * n_steps,
            "afrr_energy_pos": [40.0] * n_steps,
            "afrr_energy_neg": [30.0] * n_steps,
            "fcr": [100.0, 100.0],
            "afrr_capacity_pos": [5.0, 5.0],
            "afrr_capacity_neg": [10.0, 10.0]
        }
    },
    timeout=120
)

response.raise_for_status()
print(response.json())
```

### Response

The response contains:

- objective value and the API's derived `net_profit`;
- a partial revenue breakdown;
- reported cyclic and calendar degradation costs;
- a 15-minute charge/discharge/idle schedule;
- normalized SOC trajectory;
- renewable utilization for applicable Model III-renew runs; and
- solver name, status, and reported solve time.

The output is suitable for inspecting an experiment. See
[Known implementation limitations](#known-implementation-limitations) before
interpreting the financial or market-level fields.

## `POST /api/v1/optimize-mpc`

This endpoint runs a fixed 12-hour rolling-horizon experiment:

```text
Iteration 1: optimize 0–6 h, commit 0–4 h
Iteration 2: optimize 4–10 h, commit 4–8 h
Iteration 3: optimize 8–12 h, commit 8–12 h
```

It requires exactly:

- 48 values for each 15-minute market-price array;
- 3 values for each 4-hour capacity-price array; and
- optionally, 48 renewable-generation values.

The endpoint passes SOC between the three optimization windows. The horizon,
execution step, total duration, and starting battery state are not configurable
through this route. It is a demonstration of receding-horizon state continuity,
not a continuously running MPC controller.

## Known implementation limitations

The following limitations reflect the current code, not future design intent:

1. **No market-data retrieval.** The caller must provide every price array.
   `country` and `location` do not select or load data.
2. **No live battery state.** Capacity, efficiency, and initial SOC are fixed in
   the service; the API does not accept BMS telemetry, SOH, temperature, active
   limits, or availability.
3. **Simplified schedule response.** Each schedule entry labels its market as
   `"da"` regardless of the actual allocation. FCR and aFRR bid schedules are
   not exposed separately.
4. **Synthetic timestamps.** Returned timestamps always begin at
   `2024-01-01T00:00:00`; the request has no `start_time`.
5. **Aggregated capacity revenue.** FCR and aFRR capacity revenue are combined
   internally, while the response maps that combined value to the `fcr` key.
6. **Degradation-field mismatch.** Model III records calendar cost inside its
   degradation metrics, while the API result builder looks for a top-level
   `cost_calendar` value. Calendar cost can therefore be reported as zero.
7. **`net_profit` requires caution.** Model II/III objective values already
   subtract weighted degradation costs. The API result builder subtracts
   reported degradation again when deriving `net_profit`, so the field can
   double-count part of the cost.
8. **Weak standard-request length validation.** Exact array lengths are enforced
   for the MPC endpoint but not for the standard endpoint.
9. **Health is not readiness.** `/health` can report `"healthy"` without testing
   solver availability or a complete optimization.
10. **Synchronous solve path.** Optimization runs inside the request lifecycle;
    there is no job queue, progress endpoint, cancellation, or result storage.

Because of these limitations, API results should be treated as experimental
model output, not as operational dispatch instructions or commercial
recommendations.

## What the API does not provide

The current service does not provide:

- BMS/PCS communication or hardware-level power commands;
- execution feedback or closed-loop setpoint tracking;
- telemetry ingestion and time-series persistence;
- protection, safety, alarm, or fault handling;
- user authentication or authorization;
- multi-site or multi-battery management;
- market bidding submission; or
- production concurrency, audit, and observability controls.

## Error behavior

| HTTP status | Meaning |
| --- | --- |
| `200` | The request completed and returned a result |
| `400` | Request validation, missing input, or model-selection error |
| `422` | Pydantic request-schema validation error |
| `500` | Data adaptation, model construction, solver, or response error |

Example horizon error:

```json
{
  "detail": "time_horizon_hours=24 exceeds maximum of 12 for highs solver."
}
```

Commercial-solver environment values (`GRIDKEY_SOLVER=cplex` or
`GRIDKEY_SOLVER=gurobi`) increase the API validator's maximum horizon to 48
hours. This changes the accepted request range; it is not a guarantee of solve
time or feasibility.

## Source of truth

The current behavior is defined in:

- [`src/api/main.py`](../src/api/main.py) — routes and request schemas
- [`src/service/adapter.py`](../src/service/adapter.py) — input conversion
- [`src/service/optimizer_service.py`](../src/service/optimizer_service.py) —
  optimizer orchestration and response mapping
- [`src/service/mpc.py`](../src/service/mpc.py) — fixed 12-hour rolling horizon
- [`src/service/models.py`](../src/service/models.py) — service data models
