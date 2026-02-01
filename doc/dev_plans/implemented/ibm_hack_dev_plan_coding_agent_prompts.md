# Module D Implementation - Coding Agent Prompts

## 📋 Master Reference Files

| File                                                  | Purpose                                             |
| ----------------------------------------------------- | --------------------------------------------------- |
| `../wtsx_hack_gridkey/GridKey_WatsonX_Blueprint.md` | **System architecture & module dependencies** |
| `doc/dev_plans/ibm_hack_dev_plan.md`                | Main implementation plan                            |
| `doc/dev_plans/ibm_hack_dev_phase1.md`              | Phase 1 details                                     |
| `doc/dev_plans/ibm_hack_dev_phase2.md`              | Phase 2 details                                     |
| `doc/dev_plans/ibm_hack_dev_phase3.md`              | Phase 3 details                                     |
| `doc/dev_plans/ibm_hack_dev_phase4.md`              | Phase 4 details                                     |
| `data/optimizer_input_template.json`                | Input format spec (from Module A/B)                 |
| `data/optimizer_output_template.json`               | Output format spec (to Module E)                    |
| `GEMINI.md`                                         | Project context                                     |

---

## Phase 1: Foundation

```
Goal: 
Implement Phase 1 according to @ibm_hack_dev_phase1.md

Tasks:
0. Critically review @ibm_hack_dev_phase1.md, make sure it is correct
1. Create `src/service/models.py` — Pydantic data models (ModelType, OptimizationInput, OptimizationResult, ScheduleEntry)
2. Create `src/service/__init__.py`
3. Add `requirements-api.txt` with FastAPI, Pydantic, uvicorn

Reference:
- @ibm_hack_dev_phase1.md for code templates
- `../wtsx_hack_gridkey/GridKey_WatsonX_Blueprint.md` for the big project picture, System architecture & module dependencies
- `doc/dev_plans/ibm_hack_dev_plan.md` for Main implementation plan 

Test: 
Write `src/test/test_models.py` for Pydantic validation
```

---

## Phase 2: Core Implementation

```
Implement Phase 2 according to @ibm_hack_dev_phase2.md

Tasks:
1. Extend `src/core/optimizer.py` — Add BESSOptimizerModelIIIRenew class
2. Create `src/service/adapter.py` — DataAdapter class

Critical:
- Input format: @data/optimizer_input_template.json
- Output format: @data/optimizer_output_template.json
- New fields: p_renewable_self_kw, p_renewable_export_kw, p_renewable_curtail_kw, revenue_renewable_export_eur

Reference:
- @ibm_hack_dev_phase2.md for code templates
- @src/core/optimizer.py for existing BESSOptimizerModelIII

Test: Write `src/test/test_adapter.py` and `src/test/test_renewable_model.py`
```

---

## Phase 3: Service Layer

```
Implement Phase 3 according to @ibm_hack_dev_phase3.md

Tasks:
1. Create `src/service/optimizer_service.py` — OptimizerService class

Dependencies:
- models.py (Phase 1)
- adapter.py (Phase 2)
- BESSOptimizerModelIIIRenew (Phase 2)

Reference:
- @ibm_hack_dev_phase3.md for code templates

Test: Write `src/test/test_optimizer_service.py`
```

---

## Phase 4: Deployment

```
Implement Phase 4 according to @ibm_hack_dev_phase4.md

Tasks:
0. Critically proofread @GridKey\doc\dev_plans\ibm_hack_dev_phase4.md, check its correctness and rigor.
1. Create `Dockerfile`
2. Create `src/api/main.py` — FastAPI endpoints (/health, /api/v1/optimize)
3. Create `.github/workflows/test.yml` — CI pipeline
4. Write integration tests

Reference:
- @ibm_hack_dev_phase4.md for code templates
- @data/optimizer_output_template.json for API response format

Test: Docker build + API integration tests
```

---

## Quick Command for All Phases

```
Read and implement the plan in @doc/dev_plans/ibm_hack_dev_plan.md

Start with Phase 1, then proceed to Phase 2, 3, 4 in order.
Each phase has a detailed plan file in @doc/dev_plans/ibm_hack_dev_phase{1,2,3,4}.md

Critical data specs:
- Input: @data/optimizer_input_template.json
- Output: @data/optimizer_output_template.json

Write tests alongside each phase implementation.
```
