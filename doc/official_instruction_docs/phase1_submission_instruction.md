# Submission Requirements

## Required Deliverables

Your submission must include **all** of the following components:

1. **Main Script**: `main.py` (in project root directory)
2. **Input Data Processing**: Code to read and process the provided Excel input file
3. **Output Files**: Three result files in CSV or Excel format
4. **Dependencies**: `requirements.txt` file (if using external packages)
5. **Documentation**: `README.md` file explaining your approach

## Output Files Required

Your `main.py` script must generate exactly **three output files** with the following names (CSV or Excel format):

1. **TechArena_Phase1_Configuration.csv** (or `.xlsx`)
   - Optimal BESS configuration parameters
   - Required columns: `C-rate`, `number of cycles`, `yearly profits [kEUR/MW]`, `levelized ROI [%]`

2. **TechArena_Phase1_Investment.csv** (or `.xlsx`)
   - Investment analysis and ROI calculations
   - Must include: WACC, inflation rate, discount rate, yearly profits, year-by-year analysis, levelized ROI

3. **TechArena_Phase1_Operation.csv** (or `.xlsx`)
   - Required columns: `Timestamp`, `Stored energy [MWh]`, `SoC [-]`, `Charge [MWh]`, `Discharge [MWh]`, `Day-ahead buy [MWh]`, `Day-ahead sell [MWh]`, `FCR Capacity [MW]`, `aFRR Capacity POS [MW]`, `aFRR Capacity NEG [MW]`

## Project Structure

Your submission should follow this **exact** directory structure:

```
your_team_submission.zip
├─ main.py                 # Main execution script (REQUIRED)
├─ requirements.txt        # Python dependencies (if needed)
├─ README.md              # Documentation (RECOMMENDED)
├─ input/                 # Input data directory
│  └─ {phase_1_data_name}.xlsx
├─ [additional files/folders] # Your implementation files
└─ output/                # Output data directory
```

## Critical Requirements

- **`main.py` MUST be in the root directory** (not in subdirectories)
- **Input file MUST be in `input/` subdirectory**
- **Output files will be generated in the output directory**
- **No absolute file paths** - use relative paths only