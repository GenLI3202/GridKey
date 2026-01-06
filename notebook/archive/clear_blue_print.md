


# Task: Creat Optimizer test Notebook 

**Objective:**
Open the Jupyter Notebook `notebook\p2b_optimizer.ipynb` in the `./notebook/` directory to serve as a testing and validation harness for the BESS optimization framework.

**Context:**
This notebook will **consume** the components from the previews tasks (descript provided below). It will allow a user to flexibly run single-pass optimization scenarios (no MPC/Meta-Opt) and validate the results.

**Utilities from previous tasks:**

  * `optimizer.py` has been refactored with `solve_model()` and `extract_solution()` 
  * `solver_config.json` exists in `./data/p2_config/` 
  * `results_exporter.py` exists in `./py_script/validation/` 
  * `aging_analysis.py` exists in `./py_script/visualization/`
  * `run_optimization.py` exists in `./py_script/validation/` 
  * `compare_optimizations.py` exists in `./py_script/validation/`

**Notebook Structure:**

1.  **📦 1. Setup & Imports:**

      * Import `BESSOptimizerModelI`, `II`, `III`.
      * Import `load_process_market_data` (or similar).
      * Import `save_optimization_results` from `results_exporter`.
      * Import `plot_stacked_cyclic_soc` and `plot_calendar_aging_curve` from `aging_analysis`.

2.  **⚙️ 2. Configuration:** (Can refer to `run_optimization.py` and `compare_optimizations.py` for examples)

      * Load `solver_config.json`, `aging_config.json`, and `afrr_ev_weights_config.json`.
      * Define scenario parameters (e.g., `TEST_COUNTRY`, `TEST_C_RATE`, `TEST_ALPHA`, `TEST_TIME_HORIZON`).
      * Provide a helper to flexibly select data by time (e.g., "first 7 days").

3.  **🚀 3. Run Scenario:**

      * Show a complete example of:
        1.  Loading and slicing the market data for the chosen time horizon.
        2.  Instantiating `BESSOptimizerModelIII` with the chosen parameters (e.g., `alpha=1.0`).
        3.  Calling `optimizer.build_optimization_model()`.
        4.  Calling `model_solved, solver_status = optimizer.solve_model()`.
        5.  Calling `solution_dict = optimizer.extract_solution()`.
        6.  Creating `solution_df = optimizer.extract_solution_dataframe()`.
        7.  Creating a `summary_metrics` dictionary from the `solution_dict`.

4.  **💾 4. Save Results:**

      * Demonstrate calling `save_optimization_results()` with the `solution_df`, `summary_metrics`, and a descriptive `run_name`.
      * Store the returned `output_directory` path: `validation_results\optimizer_validation`

5.  **📊 5. Validation Plots:**

      * Demonstrate calling the new validation plot functions:
      * Call `plot_stacked_cyclic_soc(solution_df, save_path=output_directory / "plots" / "cyclic_soc.html")`.
      * Call `plot_calendar_aging_curve(solution_df, save_path=output_directory / "plots" / "calendar_curve.html")`.
      * Also include existing plots (from `visualize_market_data` or similar) to show the main BESS scheduling (SOC, power bids, etc.).



# Description (prompts) of previous tasks (REFERENCE ONLY, Already finished, DO NOT EXECUTE THEM AGAIN):

### 1\. `prompt_1_refactor_optimizer.md`

(This prompt focuses *only* on fixing the `optimizer.py` script. It's the most critical prerequisite.)

**Objective:**
Refactor the `optimizer.py` script to decouple the **solving** logic from the **result extraction** logic. The current implementation mixes these two responsibilities, creating a fragile inheritance chain.

The goal is to have one `solve_model` method that *only* solves, and a separate, overridable `extract_solution` method that formats the results.

**Refactoring Plan:**

1.  **Task 1: Refactor `BESSOptimizerModelI.solve_model`**

      * This method must **only** be responsible for calling the Pyomo solver.
      * It must **stop** extracting variable data (`p_ch`, `e_soc`, etc.).
      * It must be modified to return **two** items:
        1.  The solved `model` object.
        2.  The `solver_results` object (from `solver.solve()`).
      * **New Signature:** `def solve_model(self, model: pyo.ConcreteModel, solver_name: str = None) -> tuple[pyo.ConcreteModel, Any]:`

2.  **Task 2: Delete Redundant `solve_model` Methods**

      * Delete the `solve_model` override methods from `BESSOptimizerModelII` and `BESSOptimizerModelIII`. They are no longer needed and will be handled by inheritance from `BESSOptimizerModelI`.

3.  **Task 3: Create the `extract_solution` Chain**

      * **In `BESSOptimizerModelI`:**
          * Create a new method: `def extract_solution(self, model: pyo.ConcreteModel, solver_results: Any) -> Dict[str, Any]:`
          * Move all the extraction logic (for `status`, `solve_time`, `p_ch`, `p_dis`, `e_soc`, `c_fcr`, profit components, etc.) from the *old* `solve_model` into this new method.
          * This method builds and returns the base `solution_dict`.
      * **In `BESSOptimizerModelII`:**
          * Create an `extract_solution` method that **overrides** the parent's.
          * It must first call `solution_dict = super().extract_solution(model, solver_results)`.
          * It then adds its specific results (e.g., `p_ch_j`, `p_dis_j`, `e_soc_j`, `degradation_metrics`) to the `solution_dict` and returns it.
      * **In `BESSOptimizerModelIII`:**
          * Create an `extract_solution` method that **overrides** `ModelII`.
          * It must first call `solution_dict = super().extract_solution(model, solver_results)`.
          * It then adds its specific results (e.g., `lambda_cal`, `c_cal_cost`) and updates the `degradation_metrics` with calendar costs.

-----

-----

### 3\. `prompt_3_create_results_exporter.md`

(This prompt creates the reusable saving utility.)

**Objective:**
Create a new, reusable utility script, `results_exporter.py`, to handle saving all outputs from an optimization run. This script must be callable from any notebook or script.

**File to Create:**

  * `./py_script/validation/results_exporter.py`

**Core Requirements:**

1.  **Main Function:**
      * Create a primary function: `save_optimization_results()`.
      * **Inputs:** It must accept:
        1.  The main solution data (as a `pandas.DataFrame`).
        2.  A dictionary of summary metrics (e.g., `total_profit`, `solve_time`).
        3.  A descriptive `run_name` string.
        4.  An optional `base_output_dir` (defaulting to `"validation_results"`).
      * **Behavior:** The function must:
        1.  Create a unique, timestamped output directory (e.g., `./validation_results/20251112_124500_My_Test_Run/`).
        2.  Create a `plots` subdirectory inside it.
        3.  Save the solution DataFrame as a CSV file (e.g., `solution_timeseries.csv`).
        4.  Save the summary metrics as a JSON file (e.g., `performance_summary.json`).
      * **Return Value:** The function must return the `Path` to the newly created directory.

-----

### 4\. `prompt_4_create_aging_analysis.md`

(This prompt creates the new, specific plotting functions for validation.)

**Objective:**
Create a new visualization script, `aging_analysis.py`, dedicated to plotting and validating the degradation model behaviors from `BESSOptimizerModelIII`.

**File to Create:**

  * `./py_script/visualization/aging_analysis.py`

**Core Requirements:**
This script must contain at least two new plotting functions:

1.  **`plot_stacked_cyclic_soc()`:**

      * **Purpose:** To validate the "stacked tank" logic of the 10 cyclic SOC segments.
      * **Input:** A `pandas.DataFrame` (from `extract_solution_dataframe`).
      * **Plot:** Must generate a stacked area chart showing the energy in each segment (`e_soc_j` for $j=1..10$) over time.
      * **`save_path`:** Must accept an optional `save_path` argument to save the plot as an HTML file.

2.  **`plot_calendar_aging_curve()`:**

      * **Purpose:** To validate the SOS2 piecewise-linear cost function for calendar aging.
      * **Input:** A `pandas.DataFrame` (from `extract_solution_dataframe`).
      * **Plot:** Must generate a 2D scatter plot where:
          * X-axis: Total State of Charge (`e_soc(t)`) [kWh].
          * Y-axis: Calculated Calendar Cost (`c_cal_cost(t)`) [EUR/hr].
      * **Success Criteria:** The plot must visually trace the N-point convex curve defined in the `aging_config.json`.
      * **`save_path`:** Must accept an optional `save_path` argument to save the plot as an HTML file.