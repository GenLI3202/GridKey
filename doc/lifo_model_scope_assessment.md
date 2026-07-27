# LIFO Model Scope and Trade-off Assessment

## Purpose

This document assesses whether the strict Last-In, First-Out (LIFO) SOC-segment
formulation used by GridKey provides enough decision value to justify its
modeling and computational cost. It also distinguishes its value in a
site-level market optimizer from its possible value in a Rack-level Local
Energy Management System (LEMS).

## Executive decision

The investigation supports three separate conclusions:

1. **Degradation-aware optimization is valuable.** An optimizer should account
   for the cost of battery use when that cost can materially change its
   decisions.
2. **Strict LIFO has a poor trade-off in the current GridKey model.** GridKey
   controls an aggregated site-level BESS, while its LIFO segments are virtual
   bookkeeping states with no direct mapping to Rack or Pack execution.
3. **Rack-level degradation-aware allocation can be valuable in LEMS.** However,
   strict binary LIFO is only one possible degradation formulation and should
   not be the default without evidence that it outperforms a simpler model.

The recommended disposition is:

- retain strict LIFO as an optional research and comparison model;
- do not present it as a physical BESS or EMS behavior;
- use a simpler degradation proxy for the default GridKey optimizer; and
- start Rack-level LEMS with a convex per-Rack degradation cost, promoting
  strict LIFO only if benchmark results justify the additional complexity.

## 1. What the LIFO formulation represents

The current formulation divides the energy of one equivalent battery into
virtual SOC segments:

```text
Equivalent site-level BESS
└── virtual SOC segment j = 1 ... J
```

Each segment has a different marginal discharge cost. Binary activation and
fullness constraints enforce an ordering between those segments so that the
optimizer cannot distribute energy arbitrarily across cheaper and more
expensive portions of the degradation curve.

This is an **optimizer-internal degradation-cost representation**. It is not:

- a physical division of energy among Rack, Pack, or Cell assets;
- a command to charge or discharge physical equipment in LIFO order;
- a representation of BMS balancing or protection behavior; or
- a substitute for actual SOC, SOH, temperature, availability, and power-limit
  feedback.

The phrase “physical stack of tanks” in the Phase II mathematical document
should therefore be understood as a mathematical analogy, not a plant model.

## 2. System ownership boundary

The relevant responsibilities can be separated by the decision each layer
owns:

| Layer | Decision or authority | Degradation role |
| --- | --- | --- |
| Trading/market optimizer | Market bids and site-level power target or advisory profile | May include an aggregate degradation value or cost envelope |
| LEMS | Allocate the site target across available Racks and revise that allocation from current feedback | May optimize per-Rack marginal degradation alongside tracking and balancing |
| BESS/BMS | Execute feasible Rack power, enforce equipment limits, and manage lower-level Pack/Cell behavior | Supplies authoritative state, limits, actual power, and protection behavior |

The normal runtime relationship is:

```text
Market plan / site target
          ↓
LEMS Rack allocation
          ↓
BESS physical execution
          ↓
actual power, state, availability, and active limits
```

LEMS may predict future capability and degradation for optimization, but its
prediction does not replace authoritative BESS execution feedback.

## 3. Value in the current GridKey optimizer

### 3.1 Decision relevance

GridKey optimizes site-level market bids and aggregate charge/discharge power.
It does not decide which Rack or Pack supplies that power. Consequently, strict
station-level LIFO cannot represent the physical allocation that determines
actual degradation in a heterogeneous BESS.

At this level, LIFO can only serve as an aggregate economic proxy for cycle
depth.

### 3.2 Computational cost

The current configuration uses six SOC segments. For the one-year formulation
with 35,040 time steps, the segment model introduces approximately:

| Component | Approximate count |
| --- | ---: |
| Segment charge, discharge, and energy variables | 630,720 continuous variables |
| Segment activation variables | 210,240 binary variables |
| Segment dynamics, ordering, activation, fullness, power-flow, and aggregation constraints | 1,261,440 constraints |

The strict charge/discharge activation logic alone adds:

\[
2 \times T \times J = 420{,}480
\]

constraints to the one-year formulation. Rolling-horizon execution reduces the
size of each individual solve, but pays this formulation overhead repeatedly.

### 3.3 Evidence quality

The model uses a simplified power-law degradation cost, assumed cycle-life
parameters, and an aggregated equivalent battery. The repository does not
currently demonstrate that strict LIFO produces a materially better real-world
decision than a simpler convex degradation proxy.

There are also unresolved semantic issues around segment indexing,
shallow/deep discharge order, and the relationship between cost ordering and
the enforced fullness order. These should be validated before treating strict
LIFO as a reference formulation.

### 3.4 GridKey conclusion

For the current site-level optimizer:

> The marginal decision value of strict LIFO is too low relative to its
> complexity and unresolved modeling uncertainty.

This conclusion applies to the strict binary ordering formulation, not to the
general idea of degradation-aware optimization.

## 4. Value in a Rack-level LEMS

Rack-level LEMS makes a materially different decision:

\[
P_{\mathrm{site}}(t) = \sum_{r \in R} P_r(t)
\]

It can choose which Rack supplies the requested site power. That allocation can
change actual aging when Racks differ in SOC, SOH, temperature, availability,
or power capability.

A degradation-aware LEMS objective can therefore have direct operational value:

\[
\min_{\{P_r\}}
\lambda_{\mathrm{track}}
\left(\sum_r P_r - P_{\mathrm{site}}^{\mathrm{target}}\right)^2
+ \sum_r C_{\mathrm{deg},r}(P_r, SOC_r, SOH_r, T_r)
+ \lambda_{\mathrm{bal}} C_{\mathrm{imbalance}}
\]

In this setting, an SOC-segment model could estimate the marginal cost of
assigning additional power to each Rack. Nevertheless, the virtual segments
remain optimizer states; LEMS does not command a physical Rack to operate an
internal “segment.”

Strict LIFO is most likely to add value when:

- Rack states are materially heterogeneous;
- cycle depth changes the preferred Rack allocation;
- Rack-level SOC and actual execution are observable and updated frequently;
- degradation parameters are calibrated well enough to rank allocation
  alternatives; and
- the solve completes reliably within the LEMS control deadline.

Its value is low when:

- Racks are homogeneous and remain closely balanced;
- the BMS hides the relevant lower-level state;
- target tracking and current limits dominate all other objectives;
- degradation parameters are too uncertain to distinguish Racks reliably; or
- the binary formulation causes slow, unstable, or frequently timed-out solves.

## 5. Adoption gates

Strict LIFO should be adopted only if all five independent gates pass:

| Gate | Required evidence |
| --- | --- |
| Decision relevance | LIFO changes a decision the optimizer can physically implement |
| State observability | Required Rack state is available and synchronized with execution |
| Model validity | Segment ordering and degradation parameters are validated |
| Computational feasibility | Worst-case solve time fits the control deadline with margin |
| Incremental benefit | LIFO materially outperforms a simpler degradation formulation |

Failure of any gate is sufficient to keep strict LIFO out of the default
controller.

## 6. Recommended implementation sequence

### Stage 1: Hard feasibility

Implement Rack power, SOC, availability, and current-limit constraints using
authoritative BESS feedback. Prioritize feasible target tracking.

### Stage 2: Simple per-Rack degradation cost

Add a continuous, interpretable cost based on variables such as:

- energy throughput;
- C-rate;
- high-SOC holding time;
- temperature; and
- SOH-dependent marginal cost.

Avoid strict segment binaries in the baseline.

### Stage 3: Offline evaluation

Evaluate executed Rack trajectories with a more detailed degradation method,
such as rainflow-based cycle counting or a calibrated aging model. Use this
evaluation to determine whether the simple online cost ranks allocations
correctly.

### Stage 4: Optional strict LIFO variant

Implement strict LIFO as a separate research or slower-planning variant. Do not
place it directly in the fast active-controller path.

Compare three formulations on identical scenarios:

1. no degradation cost;
2. simple continuous per-Rack degradation cost; and
3. strict LIFO segment cost.

Measure:

- target-tracking error;
- solve time, timeout rate, and optimality gap;
- Rack SOC and throughput imbalance;
- degradation under the offline evaluation model; and
- economic value after degradation.

Promote strict LIFO only if it produces a meaningful degradation or economic
improvement while meeting the runtime requirement.

## Final recommendation

| Context | Recommendation |
| --- | --- |
| GridKey default site-level optimizer | Do not use strict LIFO by default |
| GridKey research/teaching model | Retain LIFO as an optional formulation |
| Rack-level LEMS MVP | Use simple continuous per-Rack degradation costs |
| Slower LEMS advisory/planning layer | Evaluate strict LIFO as an optional variant |
| BESS/BMS execution | Do not expose or implement virtual LIFO segments as physical commands |

The practical conclusion is:

> Degradation-aware Rack allocation is potentially valuable; strict LIFO must
> earn its place through measured incremental benefit rather than being assumed
> necessary because it is mathematically more detailed.

## Related implementation

- [`p2_model/p2_bi_model_ggdp.tex`](p2_model/p2_bi_model_ggdp.tex) — Phase II
  formulation and strict LIFO description
- [`../src/core/optimizer.py`](../src/core/optimizer.py) — segment variables,
  activation binaries, and degradation objective
- [`../config/Config.yml`](../config/Config.yml) — current segment costs and
  strict-activation configuration
