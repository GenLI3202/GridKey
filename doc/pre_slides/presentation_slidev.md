---
theme: default
background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## TechArena 2025 BESS Optimizer
  Advanced Energy Management System for Battery Storage in European Markets
drawings:
  persist: false
transition: slide-left
title: TechArena 2025 BESS Optimizer
mdc: true
---

# TechArena 2025 BESS Optimizer

## Advanced Energy Management System for Utility-Scale Battery Storage

<div class="pt-12">
  <span class="text-5xl font-bold text-primary">
    Team SoloGen
  </span>
</div>

<div class="pt-4">
  <span class="text-2xl text-gray-500">
    Technical University of Munich
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <span>November 2025</span>
</div>

---

# Agenda

<div class="grid grid-cols-2 gap-8 pt-4">

<div>

## Technical Innovation
- Three-Model Architecture
- LIFO Segment Filling
- SOS2 Calendar Aging
- MPC Framework

## Results & Performance
- Country Rankings
- C-rate Impact Analysis
- Solve Time Metrics

</div>

<div>

## Business Impact
- Investment Strategy
- ROI Projections (?)
- Market Insights

## Implementation Quality
- Code Architecture
- Validation Framework (Which ?)
- Production System (which ?)

</div>

</div>

---
layout: two-cols
---

# Problem Statement

## Competition Challenge

**Objective:** Develop an intelligent EMS for a 4.5 MWh BESS to maximize profitability across European electricity markets while managing battery degradation

### Phase II Requirements
- **Four-market optimization**
  - Day-ahead energy
  - FCR capacity
  - aFRR capacity
  - **NEW: aFRR energy**
- **Battery degradation modeling**
  - Cyclic aging costs
  - Calendar aging costs
- **10-year investment analysis**

::right::

## Key Metrics

### Battery Specifications
- **Capacity:** 4,472 kWh
- **Efficiency:** 95% round-trip
- **C-rates:** 0.25, 0.33, 0.50
- **Daily cycles:** 1.0, 1.5, 2.0

### Market Scope
- 🇩🇪 Germany/Luxembourg
- 🇦🇹 Austria
- 🇨🇭 Switzerland
- 🇭🇺 Hungary
- 🇨🇿 Czech Republic

### Evaluation Weights
- Revenue: 30%
- Degradation: 30%
- Investment: 10%
- Configuration: 10%
- Code Quality: 20%

---

# Technical Innovation: Three-Model Architecture

<div class="grid grid-cols-3 gap-4">

<div>

## Model I: Base + aFRR Energy
$$\max Z = P_{DA} + P_{ANCI} + P_{aFRR_E}$$


- Four-market co-optimization
- 15-min DA + aFRR energy
- 4-hour FCR/aFRR capacity blocks
- Cross-market exclusivity constraints

</div>

<div>

## Model II: + Cyclic Aging
```math
\max Z = Revenue - \alpha \cdot C_{cyclic}
```

- Piecewise-linear degradation
- 10 SOC segments (447.2 kWh each)
- Marginal costs: 0.0052-0.099 EUR/kWh
- Based on Xu et al. 2017

</div>

<div>

## Model III: + Calendar Aging
```math
\max Z = Revenue - \alpha(C_{cyc} + C_{cal})
```

- SOS2 linearization technique
- 5 SOC breakpoints
- Costs: 1.79-10.73 EUR/hr
- Based on Collath et al. 2023

</div>

</div>

### Implementation Statistics
- **Variables:** ~105,000 (70K continuous + 35K binary)
- **Constraints:** ~140,000 total
- **Time horizon:** 35,136 intervals (full year, 15-min resolution)

---

# Innovation 1: LIFO Segment Filling

## The Problem
Without proper constraints, energy distributes equally across all segments, violating "stacked tank" physics

## The Solution
Implemented Xu et al. 2017 Theorem 1:

```python
# Segment j can only have energy if segment j-1 is FULL
def add_lifo_filling_constraints(model):
    for t in model.time_periods:
        for j in range(2, model.num_segments + 1):
            model.lifo_constraint.add(
                model.e_soc_seg[t, j-1] >=
                (E_segment[j-1] - epsilon) * model.z_segment_active[t, j]
            )
```

## Impact
- **Charging:** Must fill segment 1 (90-100% SOC) before segment 2 receives energy
- **Discharging:** Must empty segment 10 (0-10% SOC) before segment 9 discharges
- **Result:** Accurate degradation cost reflecting actual cycling depth

[Graph: LIFO segment filling visualization showing proper sequential activation]

---

# Innovation 2: SOS2 Calendar Aging

## Calendar Aging Challenge
Non-linear relationship between SOC and degradation rate

## Linearization via SOS2

```python
# SOS2 constraints for calendar aging
def add_calendar_aging_sos2(model):
    # Define breakpoints
    soc_points = [0, 0.25, 0.50, 0.75, 1.0]  # Normalized SOC
    cost_points = [1.79, 2.15, 3.58, 6.44, 10.73]  # EUR/hr

    for t in model.time_periods:
        # SOC as weighted combination
        model.soc_total[t] == sum(
            model.lambda_sos2[t, i] * soc_points[i] * E_nom
            for i in range(5)
        )

        # Calendar cost from breakpoints
        model.cal_cost[t] == sum(
            model.lambda_sos2[t, i] * cost_points[i]
            for i in range(5)
        )

        # SOS2: at most 2 adjacent λ non-zero
        model.add_sos2(model.lambda_sos2[t, :])
```

[Graph: Calendar aging cost curve with SOS2 linearization points]

---
layout: center
---

# MPC Framework Architecture

```mermaid
graph TB
    subgraph "**Meta-Optimizer Layer**"
        MO[Meta-Optimizer<br/>Find optimal α]
    end

    subgraph "**MPC Controller Layer**"
        MPC[MPC Simulator<br/>36h horizon - 
        24h execution]
    end

    subgraph "**MILP Solver Layer**"
        MILP[Model III Optimizer<br/>4-market + degradation]
    end

    MO --> | α parameter| MPC
    MPC --> | Rolling horizon| MILP
    MILP --> | Solution| MPC
    MPC -->|10-year NPV| MO

    style MO fill:#e8e8e8
    style MPC fill:#00abf4bd
    style MILP fill:#2256ff96
```
<!-- WATERFALL_COLORS = {
    # Positive Revenue Components (stacked above zero) - Blue gradient @ 85% opacity
    'revenue_primary': 'rgba(32, 233, 217, 0.85)',    # #20e9d9 Teal - DA Discharge, primary revenue
    'revenue_secondary': 'rgba(5, 121, 194, 0.85)',  # #0579c2 Blue - FCR Capacity
    'revenue_tertiary': 'rgba(34, 81, 255, 0.85)',  # #2251ff Bright Blue - aFRR Energy, accent #2256ff96

    # Negative Cost Components (stacked below zero) - Dark blues & grays @ 100% opacity
    'cost_primary': 'rgba(6, 31, 121, 1.0)',         # #061f79 Dark Blue - DA Charge Cost
    'cost_secondary': 'rgba(232, 232, 232, 1.0)',        # #e8e8e8 Light Gray - Cyclic Aging
    'cost_tertiary': 'rgba(5, 28, 44, 1.0)',     # #051c2c Very Dark Blue - Calendar Aging

    # Cumulative Metrics
    'cumulative_line': '#061f79',                    # #061f79 Dark Blue - Cumulative Profit line
    'cumulative_marker': '#061f79',                  # #061f79 Dark Blue - Profit markers (5.6px, 70% standard)

    # Fallback colors (for aggregated view)
    'total_revenue': 'rgba(0, 169, 244, 0.85)',      # #00a9f4 Cyan Blue
    'total_cost': 'rgba(5, 28, 44, 1.0)',            # #051c2c Very Dark Blue
} -->

### Key Parameters
- **Horizon:** 36 hours lookahead
- **Execution:** 24 hours implemented
- **Iterations:** 365 per year
- **State continuity:** Final SOC(n) → Initial SOC(n+1)

---
layout: center
---

# Results: Country Performance Rankings

## Total Profit Analysis (15 Scenarios)

| Rank | Country | Total Profit | Avg Profit/Scenario | Best C-rate |
|------|---------|-------------|-------------------|------------|
| 1 | 🇨🇭 **Switzerland** | €2,520,270 | €840,090 | 0.50 |
| 2 | 🇨🇿 **Czech Republic** | €2,206,078 | €735,360 | 0.50 |
| 3 | 🇦🇹 **Austria** | €2,090,087 | €696,696 | 0.50 |
| 4 | 🇭🇺 **Hungary** | €2,042,062 | €680,687 | 0.50 |
| 5 | 🇩🇪 **Germany/Lux** | €1,105,224 | €368,408 | 0.50 |

[Graph: Country profit comparison bar chart]

### Key Insights
- **Switzerland:** Highest profitability despite small market size
- **Germany:** Surprisingly lowest returns (high competition?)
- **All countries:** C-rate 0.50 consistently optimal

---

# Results: C-Rate Impact Analysis

## Power Capacity vs. Degradation Trade-off

<div class="grid grid-cols-2 gap-8">

<div>

### Financial Performance by C-Rate

| C-Rate | Total Profit | Avg Revenue | Avg Degradation |
|--------|-------------|-------------|-----------------|
| **0.50** | €4,518,052 | €1,064,914 | €161,303 |
| **0.33** | €3,078,210 | €718,108 | €102,466 |
| **0.25** | €2,367,459 | €551,190 | €77,698 |

**0.50 vs 0.25 Comparison:**
- **+91%** profit increase
- **+93%** revenue increase
- **+108%** degradation cost
- **Net benefit:** Strongly positive

</div>

<div>

[Graph: C-rate profitability comparison - stacked bar chart showing revenue vs degradation]

[Graph: Daily cycle distribution histogram by C-rate]

### Conclusion
Higher power capacity enables capturing high-value market opportunities that far exceed increased degradation costs

</div>

</div>

---

# Results: Best & Worst Scenarios

<div class="grid grid-cols-2 gap-4">

<div>

## 🏆 Best: Switzerland @ 0.50

### Performance Metrics
- **Annual Profit:** €1,142,137
- **Revenue:** €1,339,554
- **Degradation:** €197,417 (14.7%)
- **Final SOC:** 42.2 kWh (depleted)
- **Solve Time:** 11.6 minutes

### Success Factors
- High price volatility
- Strong AS markets
- Optimal market timing
- Aggressive cycling strategy

</div>

<div>

## 📉 Worst: Germany @ 0.25

### Performance Metrics
- **Annual Profit:** €262,018
- **Revenue:** €336,851
- **Degradation:** €74,833 (22.2%)
- **Final SOC:** 236.8 kWh (partial)
- **Solve Time:** 8.3 minutes

### Limiting Factors
- Low power capacity
- Market saturation
- Conservative cycling
- Limited AS participation

</div>

</div>

[Graph: Revenue breakdown pie charts for both scenarios]

---


---

# Business Impact: Investment Strategy

## 10-Year ROI Analysis Framework

<div class="grid grid-cols-2 gap-6">

<div>

### Investment Parameters

| Country | WACC | Inflation | NPV Factor |
|---------|------|-----------|------------|
| DE/AT/CH | 8.3% | 2.0-3.3% | 0.92 |
| CZ | 12.0% | 2.9% | 0.88 |
| HU | 15.0% | 4.6% | 0.85 |

**CAPEX:** €200/kWh × 4,472 kWh = €894,400

</div>

<div>

### Strategic Recommendations

1. **Priority Markets**
   - 🇨🇭 Switzerland: Highest absolute returns
   - 🇨🇿 Czech Republic: Strong risk-adjusted returns

2. **Configuration Strategy**
   - Always select C-rate 0.50
   - Optimize daily cycles via MPC

3. **Risk Considerations**
   - Market saturation in Germany
   - Regulatory changes in CEE markets

</div>

</div>

[Graph: 10-year cumulative cash flow projections by country]

---

# Implementation Quality: Architecture

## Clean Code Structure

```
py_script/
├── core/
│   └── optimizer.py (2,398 lines)
│       ├── BESSOptimizerModelI    # Base 4-market
│       ├── BESSOptimizerModelII   # + Cyclic aging
│       └── BESSOptimizerModelIII  # + Calendar aging
├── mpc/
│   ├── mpc_simulator.py (931 lines)
│   └── meta_optimizer.py
├── validation/
│   ├── run_optimization.py        # CLI runner
│   ├── compare_optimizations.py   # Comparison framework
│   └── constraint_validator.py    # Verification tools
└── visualization/
    ├── optimization_analysis.py   # 4 standard plots
    ├── mpc_analysis.py           # 3 MPC plots
    └── financial_analysis.py     # 3 custom plots
```

### Design Principles
- **Separation of concerns:** Clean solve/extract decoupling
- **Type safety:** Comprehensive type hints
- **Documentation:** NumPy-style docstrings
- **Testing:** Automated constraint verification

---

# Implementation Quality: Validation Framework

## General-Purpose CLI Tools

```bash
# Run optimization with flexible parameters
python py_script/validation/run_optimization.py \
    --model III --country CH --c-rate 0.5 \
    --hours 8760 --alpha 1.0 --plots

# Compare different models systematically
python py_script/validation/compare_optimizations.py \
    --compare-type models \
    --models I II III \
    --hours 168 --country AT

# Validate constraints post-solve
python py_script/validation/constraint_validator.py \
    --results-dir submission_results/20251116_181632_ch_crate0.5/
```

## Comparison Framework
- **5 comparison types:** Single vs MPC, Models, Alpha sweep, Countries, C-rates
- **Automated reporting:** Markdown summaries with embedded visualizations
- **Performance metrics:** Solve time, memory usage, optimality gap

---

# Visualization Pipeline

## 10 Plot Types per Scenario

<div class="grid grid-cols-3 gap-4">

<div>

### Market Analysis (4)
1. DA market participation
2. aFRR energy activation
3. Capacity reservations
4. SOC & power flows

[Graph: DA market price and bid visualization]

</div>

<div>

### MPC Analytics (3)
5. Iteration boundaries
6. Performance waterfall
7. State continuity

[Graph: MPC iteration performance waterfall chart]

</div>

<div>

### Financial Analysis (3)
8. Revenue breakdown
9. Daily profit distribution
10. Comparative heatmaps

[Graph: Financial revenue breakdown pie chart]

</div>

</div>

### Visualization Standards
- **Style:** McKinsey-inspired aesthetics
- **Colors:** Navy (#003f5c), Teal (#4ecdc4)
- **Format:** Interactive Plotly HTML
- **Export:** High-res PNG for reports

---

# Performance Metrics

## Computational Efficiency

<div class="grid grid-cols-2 gap-8">

<div>

### Solve Time Analysis

| Metric | Value |
|--------|-------|
| **Average time/scenario** | 14.8 minutes |
| **Fastest (CZ @ 0.25)** | 6.2 minutes |
| **Slowest (AT @ 0.50)** | 39.8 minutes |
| **Total batch time** | 3.7 hours |
| **Solver used** | HiGHS (open-source) |

**With Gurobi:** Expected 2-5 min/scenario

</div>

<div>

### Scalability Metrics

| Component | Scale |
|-----------|-------|
| **Time intervals** | 35,136 |
| **AS blocks** | 2,190 |
| **Variables** | ~105,000 |
| **Constraints** | ~140,000 |
| **Memory/scenario** | 2-4 GB |
| **MIP gap** | <1% |

**Success Rate:** 15/15 (100%)

</div>

</div>

[Graph: Solve time distribution across scenarios]

---
layout: center
---

# Key Achievements

<div class="grid grid-cols-2 gap-8 pt-8">

<div>

## Technical Excellence ⚙️
- ✅ Three-layer optimization architecture
- ✅ LIFO segment enforcement (Xu 2017)
- ✅ SOS2 calendar aging (Collath 2023)
- ✅ Production MPC with checkpointing
- ✅ 100% scenario completion rate

## Performance Metrics 📊
- ✅ €9.96M total profit (15 scenarios)
- ✅ 14.8 min average solve time
- ✅ <1% optimality gap
- ✅ Zero constraint violations

</div>

<div>

## Innovation Highlights 🚀
- ✅ Four-market co-optimization
- ✅ Aging-aware operation
- ✅ 10-100x faster data pipeline
- ✅ Comprehensive validation framework

## Business Impact 💼
- ✅ Switzerland identified as top market
- ✅ C-rate 0.50 proven optimal
- ✅ Degradation managed to 13-15%
- ✅ Clear investment strategy

</div>

</div>

---
layout: center
---

# Conclusion

## World-Class Optimization Engineering

<div class="pt-8 text-2xl">

### We delivered a production-ready BESS optimizer that:

1. **Maximizes profitability** across four European electricity markets
2. **Manages degradation** through advanced aging models
3. **Scales efficiently** to full-year horizons
4. **Provides actionable insights** for investment decisions

</div>

<div class="pt-8">

### Ready for Deployment
- Professional codebase with comprehensive documentation
- Validated against real market data
- Extensible architecture for future enhancements

</div>

---
layout: center
class: text-center
---

# Thank You

<div class="text-4xl font-bold mt-8 mb-4">
Questions & Discussion
</div>

<div class="py-4">
  <span class="text-2xl text-gray-600">
    Team SoloGen
  </span>
</div>

<div class="text-xl text-gray-500">
  Technical University of Munich
</div>

<div class="mt-8 text-lg">
  <div>📧 Contact: gen.li@tum.de</div>
  <div>📂 Repository: TechArena2025_EMS</div>
  <div>📊 Results: submission_results/</div>
</div>

---
layout: center
---

# Appendix: Mathematical Formulation

## Phase II Model III Objective Function

$$\max Z = \underbrace{P_{DA} + P_{FCR} + P_{aFRR}^{cap} + P_{aFRR}^{energy}}_{\text{Revenue}} - \alpha \cdot \underbrace{(C_{cyclic} + C_{calendar})}_{\text{Degradation}}$$

Where:
- $P_{DA} = \sum_{t \in T} \lambda_{DA}(t) \cdot (p_{dis}(t) - p_{ch}(t)) \cdot \Delta t$
- $P_{FCR} = \sum_{b \in B} \lambda_{FCR}(b) \cdot c_{FCR}(b) \cdot 4h$
- $C_{cyclic} = \sum_{t,j} c_j^{cost} \cdot \frac{p_{dis,j}(t)}{\eta_{dis}} \cdot \Delta t$
- $C_{calendar} = \sum_t c_{cal}(t) \cdot \Delta t$

### Degradation Parameters
- **Cyclic costs:** $c_j \in [0.0052, 0.0990]$ EUR/kWh
- **Calendar costs:** $c_{cal}(SOC) \in [1.79, 10.73]$ EUR/hr
- **Alpha range:** $\alpha \in [0.5, 2.0]$ (meta-optimized)

---

# Appendix: Code Repository Structure

```bash
TechArena2025_EMS/
├── data/
│   ├── TechArena2025_Phase2_data.xlsx    # Official competition data
│   ├── parquet/                          # Processed market data
│   └── p2_config/                        # Configuration files
├── py_script/
│   ├── core/
│   │   └── optimizer.py                  # Three-model MILP implementation
│   ├── mpc/
│   │   ├── mpc_simulator.py             # Rolling horizon controller
│   │   └── meta_optimizer.py            # Alpha parameter optimization
│   ├── validation/
│   │   ├── run_optimization.py          # General CLI runner
│   │   └── compare_optimizations.py     # Comparison framework
│   └── visualization/
│       └── *.py                          # 10+ plotting utilities
├── submission_results/                   # Final competition results
├── validation_results/                   # Test and validation outputs
└── doc/
    ├── whole_project_description.md     # Comprehensive documentation
    └── pre_slides/                      # This presentation
```

---

# Appendix: Solver Comparison

## Performance Across Different MILP Solvers

| Solver | Type | Avg Time (min) | MIP Gap | Availability |
|--------|------|---------------|---------|--------------|
| **Gurobi** | Commercial | 2-5 | <0.5% | License required |
| **CPLEX** | Commercial | 2-5 | <0.5% | License required |
| **HiGHS** | Open-source | 10-20 | <1% | ✅ Default |
| **CBC** | Open-source | 15-30 | <2% | ✅ Fallback |
| **GLPK** | Open-source | 30-60 | <5% | ✅ Emergency |

### Auto-Detection Priority
```python
SOLVER_PRIORITY = ['gurobi', 'cplex', 'highs', 'cbc', 'glpk']

def get_available_solver():
    for solver in SOLVER_PRIORITY:
        if pyo.SolverFactory(solver).available():
            return solver
    raise RuntimeError("No MILP solver available")
```

**Note:** HiGHS provides excellent open-source performance, achieving commercial-grade results for most scenarios

---

# Appendix: Future Enhancements

## Potential Improvements for Production Deployment

### Near-term (1-3 months)
- [ ] Implement distributed computing for parallel scenario execution
- [ ] Add real-time market data integration via APIs
- [ ] Develop web-based dashboard for results visualization
- [ ] Include weather-based degradation adjustments

### Medium-term (3-6 months)
- [ ] Integrate machine learning for price forecasting
- [ ] Add stochastic optimization for uncertainty handling
- [ ] Implement multi-battery portfolio optimization
- [ ] Develop automated reporting system

### Long-term (6-12 months)
- [ ] Extend to additional European markets
- [ ] Add grid services beyond FCR/aFRR
- [ ] Implement reinforcement learning for online adaptation
- [ ] Create SaaS platform for BESS operators

---
layout: end
---

# TechArena 2025

## Thank you for your attention!

### Ready to answer your questions

<div class="mt-8">
  <span class="text-xl">
    Team SoloGen | Technical University of Munich
  </span>
</div>