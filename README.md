# GridKey

GridKey is a learning project for battery energy storage system (BESS)
optimization. It was originally developed for the Huawei TechArena 2025
challenge and is now kept as a practical testbed for optimization models, data
pipelines, and rolling-horizon experiments.

The repository is useful for studying how battery constraints, electricity
market participation, and simplified degradation costs can be represented in
an optimization problem. It is not a production EMS, a battery digital twin,
or an industrial control system.

## Quick start

Requirements:

- Python 3.11 or newer

HiGHS, the default open-source MILP solver, is installed through
`requirements.txt`. Run the following commands from the repository root.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python src/validation/run_optimization.py --model I --country DE_LU --hours 24 --solver highs
```

If PowerShell blocks the activation script, allow it for the current terminal
session and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

For Command Prompt, activate the same environment with:

```bat
.venv\Scripts\activate.bat
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python src/validation/run_optimization.py \
  --model I \
  --country DE_LU \
  --hours 24 \
  --solver highs
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python src/validation/run_optimization.py \
  --model I \
  --country DE_LU \
  --hours 24 \
  --solver highs
```

The runner reads the included market dataset and writes the schedule and
summary metrics to `validation_results/`. Add `--plots` to generate HTML
visualizations.

To leave the virtual environment:

```text
deactivate
```

## Scope

The core of the project is a set of Pyomo optimization models for scheduling a
simplified BESS across:

- day-ahead energy;
- FCR capacity;
- aFRR capacity; and
- aFRR energy.

Three model variants support incremental study:

| Model | Main addition |
| --- | --- |
| I | Market co-optimization with SOC, power, reserve, and cycle constraints |
| II | Model I plus a segment-based cyclic aging cost |
| III | Model II plus a piecewise-linear calendar aging cost |

The repository also contains market-data processing, validation and
visualization utilities, rolling-horizon/MPC experiments, and a small API
wrapper.

## Important limitations

GridKey intentionally combines optimization, a simplified battery model, and
service/API code in one repository. This made the project convenient for
learning and experimentation, but it does not reflect the separation of
responsibilities normally found in a real BESS project.

In particular:

- the battery is represented by optimization constraints and cost
  approximations, not by a validated electrochemical or plant model;
- the optimizer produces schedules and bids, not hardware-level commands;
- the MPC and API modules are experimental scaffolding, not a complete EMS;
- market rules and degradation assumptions are simplified and tied to the
  original challenge context; and
- the project does not include site integration, telemetry, safety logic,
  protection, availability management, or closed-loop plant control.

Results should therefore be treated as model outputs for learning and
comparison, not as operational or commercial recommendations.

## Repository structure

```text
GridKey/
├── config/             # Model, solver, aging, and MPC configuration
├── data/               # Challenge market data and input/output examples
├── doc/                # Model notes, development records, and API guidance
├── notebook/           # Exploratory analysis
└── src/
    ├── core/           # Pyomo optimization models
    ├── data/           # Data loading and preprocessing
    ├── mpc/            # Rolling-horizon experiments
    ├── service/        # Optimizer adapters and service models
    ├── api/            # FastAPI wrapper
    ├── validation/     # CLI runners and comparison tools
    └── visualization/  # Result analysis and plots
```

## Optional API and Docker

Docker is not required to use the optimizer. The virtual-environment setup
above is the recommended path for local learning and development.

The included Dockerfile packages the experimental FastAPI wrapper for running
the optimizer as an HTTP service. Use it only if you specifically want to test
that service or a container-based deployment.

For the API, model, and CLI details, see:

- [`doc/API_USER_GUIDE.md`](doc/API_USER_GUIDE.md)
- [`src/modeling_guide_pyomo.md`](src/modeling_guide_pyomo.md)
- [`src/validation/README.md`](src/validation/README.md)
- [`src/mpc/MPC_README.md`](src/mpc/MPC_README.md)

## License

The code is available under the [MIT License](LICENSE).
