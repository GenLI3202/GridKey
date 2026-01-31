# GridKey BESS EMS

### BESS Intelligent Management Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](Actions)

A **High-Fidelity Industrial Prototype** for Battery Energy Storage System (BESS) management.

GridKey redefines BESS not just as an energy asset, but as a "financial instrument with physical constraints". The system integrates advanced mathematical optimization, real-time control, and explainable AI to bridge the gap between theoretical algorithms and industrial reality.

---

## Vision & Mission

**The Goal:** Maximize Lifetime Profit.

$$
\text{Profit} = (R_{Energy} + R_{Services}) - C_{Charging} - C_{Aging} - C_{NonTech}
$$

**The Problem:** Traditional EMS often treats batteries as "ideal black boxes," ignoring non-linear degradation and relying on perfect forecasts. This leads to inflated profit projections and reduced asset lifespan.

**The Solution:** GridKey implements a "Glass Box" approach:

1. **Physics-First**: Explicitly models non-linear electrochemical aging (Cyclic & Calendar).
2. **Robust Control**: Decouples long-term planning from real-time execution.
3. **Transparency**: Explains every watt dispatched.

> **Origin**: This project originated as a solution for the **Huawei TechArena 2025** challenge, serving as the foundational testbed for validating these advanced optimization concepts in the European electricity market context.

---

## System Architecture

GridKey is architected in three layers, mimicking biological intelligence:

### 1. Foundation (Physics & Environment)

*The body constraints and external reality.*

- **Non-linear Degradation**: Models capacity fade and resistance increase based on C-rate, DOD, SOC, and temperature.
- **Multi-Market Stacking**: Simultaneous participation in Day-Ahead (Spot), FCR (Primary Reserve), and aFRR (Secondary Reserve) markets.
- **Industrial Reality**: Incorporates grid fees, taxes, and regulatory constraints (e.g., min bid sizes, activation rules).

### 2. Brain (Decision Core)

*The cognitive center for planning and reaction.*

- ✅ **The Planner (Global Optimization)**: A Mixed-Integer Linear Programming (MILP) engine that looks weeks/months ahead to calculate the theoretical global optimum. Used for baselining and long-term strategy.
- ✅ **The Pilot (MPC Controller)**: A Model Predictive Control (MPC) engine that handles forecast uncertainty. It re-optimizes every 15 minutes, adapting to real-time market changes while adhering to long-term health guidelines.

### 3. Face (Interaction)

*The interface for trust and control.*

- 🚧 **Explainable AI Dashboard**: A visualization layer that answers *why* the battery is charging or discharging (e.g., "Charging now to catch a price spike at 18:00").
- 🚧 **Shadow Account**: A real-time ledger comparing "GridKey Performance" vs. "Naïve Strategy" to quantify value add.

---

## Roadmap

### Phase 1: Clean Core & Static Showcase (Partially Completed)

*Goal: Solidify the mathematical core and engineering infrastructure.*

- ✅ **Engineering Refactor**: Replaced ad-hoc scripts with a professional modular architecture; migrated data pipeline from Excel to Parquet/Pandas.
- ✅ **Algorithm Upgrade**: Implemented Models I, II, and III (Base -> +Cyclic Aging -> +Calendar Aging).
- 🚧 **Financial Integration**: Incorporating grid fees, taxes, and levy structures.*(Pending)*
- 🚧 **Web Interface**: Interactive Streamlit dashboard for strategy visualization.*(Pending)*

### Phase 2: Online Simulation & Prediction (🚧 In Progress)

*Goal: Transition from static backtesting to dynamic, real-time emulation.*

- Connect to ENTSO-E Transparency Platform API for live data.
- Develop Machine Learning models (XGBoost/LSTM) for Market Activation Rate prediction.
- Implement rolling-horizon simulation with real-world uncertainty.

### Phase 3: Deep Tech (📅 Future)

*Goal: Push mathematical and physical limits.*

- **Stochastic Optimization**: optimization under uncertainty for Capacity Markets.
- **Multi-Physics Coupling**: Advanced thermal-aging coupled modeling.

---

## Technical Specifications

### Battery System (Reference: Huawei LUNA2000)

- **Capacity**: 4.47 MWh
- **Power**: 2.2 MW (0.5C)
- **Efficiency**: 95% Round-trip

### Market Segments

The system optimizes revenue across four simultaneous revenue streams:

1. **Day-Ahead Energy**: Arbitrage (Buy low, sell high).
2. **FCR Capacity**: Frequency Containment Reserve (Symmetric).
3. **aFRR Capacity**: Automatic Frequency Restoration Reserve (Asymmetric).
4. **aFRR Energy**: Real-time activation payments.

### Optimization Models

- **Model I**: Linear Revenue Stacking.
- **Model II**: + Piecewise-linear Cyclic Aging Cost (Rainflow-counting equivalent).
- **Model III**: + State-dependent Calendar Aging Cost (SOS2 variables).

---

## Repository Structure

```
GridKey/
├── data/                  # Market data (Parquet) & Configs (JSON)
├── doc/                   # Mathematical Models (LaTeX) & Blueprints
├── notebook/              # Analysis & Prototyping (Jupyter)
├── py_script/
│   ├── core/              # MILP Models & Investment Logic
│   ├── mpc/               # Model Predictive Control Simulator
│   ├── visualization/     # Plotting Libraries
│   └── validation/        # CLI Tools for Testing
├── internal/              # Project Internal Docs (Blueprints)
└── README.md              # Project Overview
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- MILP Solver (HiGHS, CBS, Gurobi, or CPLEX)

### Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
2. **Run Optimization (CLI)**

   ```bash
   # Run a 24-hour Model III optimization for Germany
   python py_script/validation/run_optimization.py --model III --country DE_LU --hours 24
   ```
3. **Run Validation**

   ```bash
   python py_script/validation/compare_optimizations.py --compare-type models --country DE
   ```

---

## License

MIT License. See `LICENSE` for details.

## Author

**GridKey Team**
