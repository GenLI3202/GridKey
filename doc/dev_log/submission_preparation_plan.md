# Phase 2 Submission Repository Preparation Plan

**Date**: 2025-11-17
**Deadline**: Today (Same-day submission)
**Status**: Planning Complete - Ready for Implementation

---

## Objective
Create a clean `p2_submission_ver` branch with submission-compliant structure that can be exported as a ZIP file for today's deadline.

## Scope
- **Timeline**: Same-day delivery (fast-track)
- **Approach**: Dual-path
  1. `main.py` - Full executable workflow for examiners (loads Excel, runs optimizer, saves results)
  2. Conversion script - Transforms existing `submission_results/` to required Excel format with 15 sheets

---

## Implementation Plan

### Phase 1: Repository Structure Setup (30 min)

**1.1 Create new branch**
```bash
git checkout -b p2_submission_ver
```

**1.2 Create submission directory structure**
```
TechArena2025_EMS/
├── main.py                          # NEW - Main entry point for examiners
├── requirements.txt                ✅ # MODIFY - Remove commercial solvers
├── README.md                       ✅ # MODIFY - Add submission instructions
├── input/                          ✅ # NEW - Input data directory
│   └── TechArena2025_Phase2_data.xlsx
├── output/                         ✅ # NEW - Output results directory    
│   └── demo_results/               ✅ # NEW - save the converted `submission_results` of "20251116_222054_cz_crate0.5" as a demo
├── data/                           ✅ # KEEP - Configuration files
│   └── p2_config/                  ✅  # KEEP - All JSON configs
├── py_script/                      ✅  # KEEP - Core implementation
│   ├── core/
│   ├── data/
│   ├── mpc/
│   ├── visualization/
│   └── submission/                  # NEW - Submission utilities
│       ├── __init__.py
│       ├── convert_results.py       # Converts submission_results to Excel
│       └── format_outputs.py        # Formats optimizer results to submission format
└── doc/                            ✅ # KEEP - Documentation
```

---

### Phase 2: Create main.py (2 hours)

**2.1 Main Script Structure**

Based on `run_submission_batch.py` but submission-compliant:

```python
"""
Huawei TechArena 2025 Phase 2 - BESS Optimization Solution
Main execution script for competition submission

This script demonstrates our Model III BESS optimization approach:
- 4 electricity markets (DA, FCR, aFRR capacity, aFRR energy)
- Full battery degradation modeling (cyclic + calendar aging)
- MPC control with rolling horizon optimization
"""

import os
from pathlib import Path
import sys

# Configuration
TEST_MODE = False  # Set True for 3-day quick test, False for full year

def main():
    """Main execution function"""

    # 1. Load market data from input/TechArena2025_Phase2_data.xlsx
    # 2. Load configuration files from data/p2_config/
    # 3. Run MPC optimization for each scenario
    # 4. Format results to submission format
    # 5. Save to output/ directory

    pass  # Implementation details below

if __name__ == "__main__":
    main()
```

**Key Features**:
- `TEST_MODE = True`: Runs 3-day sample (~5 min) for validation
- `TEST_MODE = False`: Runs full 365-day optimization (~3-6 hours)
- Loads Excel directly (not preprocessed parquet)
- Saves results in submission format with proper headers
- Uses only open-source solvers (HiGHS)

**Scenarios to run**:
- All 5 countries: DE_LU, AT, CH, HU, CZ
- All 3 C-rates: 0.25, 0.33, 0.5
- Model III (no rigid cycle limits, degradation-optimized)

---

### Phase 3: Create Conversion Script (1.5 hours)

**3.1 Script: `py_script/submission/convert_results.py`**

Purpose: Convert existing `submission_results/` (15 scenarios) to submission Excel format

**Functionality**:

#### Output File 1: TechArena_Phase2_Operation.xlsx
- **15 sheets** (one per country-C-rate combination)
- **Sheet names**: "DE_LU_0.25", "AT_0.33", "CH_0.5", etc.
- **Columns**:
  - `Timestamp` (datetime)
  - `Stored energy [MWh]` (from soc_kwh / 1000)
  - `SoC [-]` (from soc_pct / 100, range 0-1)
  - `Charge [MWh]` (from p_ch_kw / 1000 * 0.25)
  - `Discharge [MWh]` (from p_dis_kw / 1000 * 0.25)
  - `Day-ahead buy [MWh]` (p_ch - p_afrr_neg_e, scaled)
  - `Day-ahead sell [MWh]` (p_dis - p_afrr_pos_e, scaled)
  - `FCR Capacity [MW]` (c_fcr_mw)
  - `aFRR Capacity POS [MW]` (c_afrr_pos_mw)
  - `aFRR Capacity NEG [MW]` (c_afrr_neg_mw)

#### Output File 2: TechArena_Phase2_Configuration.xlsx
- **5 sheets** (one per country)
- **Sheet names**: "DE_LU", "AT", "CH", "HU", "CZ"
- **Columns per sheet**:
  - `C-rate` (0.25, 0.33, 0.5)
  - `number of cycles` (calculated from degradation data)
  - `yearly profits [kEUR/MW]` (profit / battery_power_MW / 1000)
  - `levelized ROI [%]` (10-year NPV-based ROI)

#### Output File 3: TechArena_Phase2_Investment.xlsx
- **5 sheets** (one per country)
- **Sheet names**: "DE_LU", "AT", "CH", "HU", "CZ"
- **Content per sheet**:
  - Header rows: WACC, Inflation rate, Discount rate, Yearly profits (2024)
  - Year-by-year table:
    - Year (2023-2033)
    - Initial Investment [kEUR/MWh]
    - Yearly profits [kEUR/MWh] (with degradation factor)
    - NPV contribution
  - Footer: Levelized ROI [%]

**Degradation Model for Investment**:
- Assume 3% annual capacity fade (100% → 70% over 10 years)
- Profit scales proportionally with capacity
- NPV calculation: `Σ(profit_year / (1+WACC)^year)`

**Country-Specific Financial Parameters**:
```python
WACC = {'DE_LU': 0.083, 'AT': 0.083, 'CH': 0.083, 'CZ': 0.12, 'HU': 0.15}
INFLATION = {'DE_LU': 0.02, 'AT': 0.033, 'CH': 0.001, 'CZ': 0.029, 'HU': 0.046}
INVESTMENT_COST = 200  # EUR/kWh
BATTERY_CAPACITY = 4472  # kWh
```

---

### Phase 4: Clean requirements.txt (15 min)

**4.1 Remove commercial solvers**

Current problematic lines:
```
cplex==22.1.1.0
gurobipy==11.0.0
```

**Submission-ready requirements.txt**:
```
# Core optimization
pyomo>=6.7.0
highspy>=1.7.0  # Open-source MILP solver

# Data processing
pandas>=2.1.0
numpy>=1.24.0
openpyxl>=3.1.0  # Excel file support
scipy>=1.11.0

# Visualization (optional)
plotly>=5.18.0
matplotlib>=3.8.0
kaleido>=0.2.1

# Utilities
python-dateutil>=2.8.2
```

---

### Phase 5: Update README.md (30 min)

**5.1 Add submission-specific sections**

```markdown
# Huawei TechArena 2025 - Phase 2 BESS Optimization Solution

## Quick Start (For Examiners)

### Installation
```bash
pip install -r requirements.txt
```

### Run Optimization

**Quick Test (3-day sample, ~5 minutes)**:
```python
# Edit main.py: Set TEST_MODE = True
python main.py
```

**Full Year (365 days, ~3-6 hours)**:
```python
# Edit main.py: Set TEST_MODE = False
python main.py
```

### Output Files

Results are saved in `output/` directory:
- `TechArena_Phase2_Operation.xlsx` - Operational decisions (15 sheets)
- `TechArena_Phase2_Configuration.xlsx` - Configuration analysis (5 sheets)
- `TechArena_Phase2_Investment.xlsx` - Investment ROI analysis (5 sheets)

## Methodology

### Model III: Full Degradation-Aware Optimization

Our solution uses **Model III** with comprehensive battery aging modeling:

**Objective Function**:
```
Maximize: Revenue - α × (Cyclic_Cost + Calendar_Cost)
```

**Markets**: 4 European electricity markets
- Day-ahead (DA): Energy arbitrage
- FCR: Frequency Containment Reserve capacity
- aFRR: Automatic Frequency Restoration Reserve (capacity + energy)

**Degradation Modeling**:
- **Cyclic aging**: Xu et al. 2017 - 10-segment LIFO depth-of-discharge model
- **Calendar aging**: Collath et al. 2023 - SOC-dependent capacity fade

**MPC Control**:
- Planning horizon: 36 hours
- Execution horizon: 24 hours
- Rolling optimization with perfect foresight

### Key Features

✅ No rigid daily cycle limits (replaced with flexible degradation costs)
✅ aFRR energy market integration (Phase 2 requirement)
✅ Calendar aging consideration (Phase 2 requirement)
✅ 10-year investment ROI with capacity fade effects
✅ Open-source solver compatible (HiGHS)

## Technical Details

[... existing README content ...]
```

---

### Phase 6: Clean Up Repository (1 hour)

**6.1 Archive development files**

Create `archive/` directory and move:
- `notebook/` → Keep only essential analysis scripts
- `validation_results/` → Archive (not needed for submission)
- `submission_results/` → Keep for conversion script
- Development scripts → Archive

**6.2 Keep essential files only**

```
TechArena2025_EMS/
├── main.py ✅
├── requirements.txt ✅
├── README.md ✅
├── input/ ✅
├── output/ (generated) ✅
├── data/p2_config/ ✅
├── py_script/ ✅
│   ├── core/ ✅
│   ├── data/ ✅
│   ├── mpc/ ✅
│   ├── visualization/ (optional) ⚠️
│   └── submission/ ✅
├── doc/ (optional) ⚠️
└── submission_results/ (for conversion script) ✅
```

---

### Phase 7: Testing & Validation (1 hour)

**7.1 Test main.py in TEST_MODE**
```bash
# Set TEST_MODE = True in main.py
python main.py
# Expected: Completes in ~5 min, generates 3 Excel files
```

**7.2 Test conversion script**
```bash
python py_script/submission/convert_results.py
# Expected: Generates 3 Excel files from existing submission_results/
```

**7.3 Validate output format**
- Check Excel files open correctly
- Verify 15 sheets in Operation file
- Verify 5 sheets in Configuration and Investment files
- Check column names match submission requirements
- Validate data ranges (SOC 0-1, non-negative powers)

**7.4 Test in clean environment**
```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

### Phase 8: Final Packaging (30 min)

**8.1 Generate submission ZIP**

**Option A: Manual ZIP**
```bash
# On submission branch
zip -r TeamName_TechArena2025_Phase2.zip . \
  -x "*.git*" \
  -x "*__pycache__*" \
  -x "*.pyc" \
  -x "*validation_results*" \
  -x "*archive*"
```

**Option B: Python script**
Create `create_submission_package.py`:
```python
import shutil
from pathlib import Path

def create_submission_zip():
    """Create submission ZIP with required structure"""

    # Files to include
    include = [
        'main.py',
        'requirements.txt',
        'README.md',
        'input/',
        'data/p2_config/',
        'py_script/',
        'doc/whole_project_description.md'
    ]

    # Create ZIP
    shutil.make_archive(
        'TeamName_TechArena2025_Phase2',
        'zip',
        root_dir='.',
        base_dir=include
    )
```

**8.2 Final checklist before submission**

- [ ] `main.py` exists in root directory
- [ ] `main.py` has TEST_MODE flag clearly documented
- [ ] `requirements.txt` contains ONLY open-source packages
- [ ] `input/TechArena2025_Phase2_data.xlsx` exists
- [ ] `README.md` has installation and usage instructions
- [ ] Conversion script generates correct Excel format
- [ ] All 3 output files have correct column names
- [ ] Operation file has 15 sheets (5 countries × 3 C-rates)
- [ ] Configuration file has 5 sheets (one per country)
- [ ] Investment file has 5 sheets (one per country)
- [ ] Test run in clean environment succeeds
- [ ] No absolute file paths in code
- [ ] ZIP file size is reasonable (~50-100 MB without results, ~5-10 MB with only code)

---

## Summary

**Total Estimated Time**: 6-7 hours

**Critical Path**:
1. Create branch and directory structure (30 min)
2. Develop main.py (2 hours)
3. Create conversion script (1.5 hours)
4. Update documentation (30 min)
5. Testing and validation (1 hour)
6. Final packaging (30 min)
7. Buffer time (1 hour)

**Key Deliverables**:
1. ✅ Submission-compliant repository structure
2. ✅ `main.py` that runs full optimization workflow
3. ✅ Conversion script for existing results
4. ✅ Three Excel output files in required format
5. ✅ Clean requirements.txt (open-source only)
6. ✅ Updated README with submission instructions
7. ✅ ZIP package ready for upload

**Risk Mitigation**:
- Conversion script provides backup (uses existing results)
- TEST_MODE allows quick validation
- Clean environment testing catches dependency issues
- Checklist ensures nothing is forgotten

---

## Implementation Status

- [ ] Phase 1: Repository Structure Setup
- [ ] Phase 2: Create main.py
- [ ] Phase 3: Create Conversion Script
- [ ] Phase 4: Clean requirements.txt
- [ ] Phase 5: Update README.md
- [ ] Phase 6: Clean Up Repository
- [ ] Phase 7: Testing & Validation
- [ ] Phase 8: Final Packaging

---

## Notes

**User Requirements Clarifications**:
1. **main.py**: Must have full functionality for examiners (load Excel, run optimizer, save results)
2. **Operation file**: 15 sheets (5 countries × 3 C-rates)
3. **Configuration file**: 5 sheets (one per country), all C-rates per sheet
4. **Investment file**: 5 sheets (one per country), 10-year analysis with capacity fade
5. **Conversion script**: Separate script to convert existing submission_results to Excel format
6. **Submission package**: Only code + documentation (~5-10 MB), no raw results

**Key Technical Decisions**:
- Model III: No rigid cycle limits, degradation-optimized
- Investment: Simplified 3% annual capacity fade approximation
- Solver: HiGHS (open-source, compatible)
- Data loading: Excel directly in main.py (for examiner demonstration)
- Results format: Excel with multiple sheets (cleaner than 15 separate CSV files)
