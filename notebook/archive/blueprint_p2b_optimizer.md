
# Objective: 
1. Write content in the Jupyter Notebook `p2b_optimizer.ipynb` in the `./notebook/` directory, serving as a testing and validation harness for the BESS optimization framework. 
2. Corresponding components to empower this notebook above, such as config files, visualization python script. 

## Context
This notebook should allow users to flexibly test different BESS optimization models (Model I, II, III) over various time horizons and parameter configurations. It should also include core validation plots to visually confirm the correctness of the degradation models implemented in Model III. BUT, it does NOT involve the MPCSimulator,  MetaOptimizer and Investment Optimization components.

This notebook will majorly based on 
    1. existing scripts from `./py_script/`, esp.:
       - `load_process_market_data.py` to load and preprocess market data. (Refer to `p2a_market_data.ipynb` for examples.), and
       - `optimizer.py` to understand the model implementation, and 
       - other scripts in `./py_script/test` to get useful testing modules
       - `visualize_market_data.py` for extract solutions from solved optimizer and visualize the results.
    2. existing config files in `./data/p2_config`


## Delivery: 
- the updated Jupyter Notebook `p2b_optimizer.ipynb` in the `./notebook/` directory; 
- `solver_config.json` in `./data/p2_config/`; 
- `optimizer.extract_detailed_solution()` in `./py_script/core/optimizer.py` to extract solution (and maybe also save results);
- `aging_analysis.py` under `./py_script/visualization/` for Aging Validation

### Structure of the `p2b_optimizer.ipynb`

#### Task  1: Flexible Model Testing Harness
##### 1. Data Loading:
    - Develop and encapsulate a function to load and preprocess the market data (e.g. refer to the code in `p2a_market_data.ipynb`:

        ```python
        # Load all Phase 2 market tables (day_ahead, fcr, afrr_capacity, afrr_energy)
        p2_market_tables = load_phase2_market_tables(phase2_data_path)

        # Display what was loaded
        print("Loaded tables:")
        for table_name, table_df in p2_market_tables.items():
        print(f"  ✓ {table_name:20s}: {len(table_df):,} rows x {len(table_df.columns)} columns")
        ```
##### 2. Flexible Scenario Execution:
- Create a core testing function that can instantiate and run a single optimization pass. This function must be configurable to test:
    1. Different BESS optimization models: BESSOptimizerModelI, BESSOptimizerModelII, and BESSOptimizerModelIII.
    2. Time Horizon: Allow flexible selection of the data time length (e.g., use timestamps (date_range('2024-01-01')), "D1, D2, ..., D365" "Q1, Q2, Q3", "M1, M2, ..., M12", etc.).
    3. Parameters: Build on the function from (2.b - flexible time horizon) to demonstrate testing of different parameter combinations.
       - c_rate (e.g., 0.25, 0.33, 0.5).
       - daily cycle limit 
       - countries
       - solver options (e.g., solver name 'cplex', 'highs', 'cbc'; time limit, etc.). And to do this, first create a config file `solver_config.json` in `./data/p2_config/` to define default solver parameters.
       - access the aging config file `aging_config.json`, the ev_weighting config `afrr_ev_weights_config.json`, from `./data/aging_config/`.
       - alpha: The degradation price parameter. The notebook must allow setting a fixed alpha (e.g., alpha=1.0) for running Model II and III. (Note: We are not sweeping or tuning alpha in this notebook, as that is handled by the MetaOptimizer).
        As a reference, you can view the following code in `./notebook/p1_final_validation.ipynb` to get what we need,e.g.: 
        ```python
        # Initialize components
        print("🚀 Initializing BESS Optimizer and Investment Analyzer...")
        optimizer = ImprovedBESSOptimizer()
        investment_analyzer = InvestmentAnalyzer()

        # Test configuration parameters
        TEST_CONFIGS = {
            'countries': ['DE_LU', 'AT', 'CH', 'CZ', 'HU'],  # All 5 countries
            'c_rates': [0.25, 0.33, 0.50],              # All C-rate options
            'cycle_limits': [1.0, 1.5, 2.0],            # Different cycle limits
            'test_scenarios': ['quick', 'quick_all_countries', 'medium', 'full'] # Different test depths
        }

        # Define test scenarios
        SCENARIO_CONFIGS = {
            'quick': {
                'countries': ['AT'],                      # Single country
                'c_rates': [0.25, 0.50],                # 2 C-rates
                'cycle_limits': [1.0, 1.5],             # 2 cycle limits
                'data_limit': 672,                       # TODO: HERE REPLACE THIS BY DAY/WEEK/YEAR OR SPECIFIC DATE INTERVAL. 
                'description': 'Quick validation (1 week, 1 country, 4 scenarios)'
            },
            'quick_all_countries': {
                'countries': ['DE_LU', 'AT', 'CH', 'CZ', 'HU'], # All countries
                'c_rates': [0.25],                      # Single C-rate
                'cycle_limits': [1.0],                  # Single cycle limit
                'data_limit': 672,                       # TODO: HERE REPLACE THIS BY DAY/WEEK/YEAR OR SPECIFIC DATE INTERVAL. 
                'description': 'Quick validation all countries (1 week, 5 countries, 1 scenario)'
            },
            'medium': {
                'countries': ['AT', 'DE_LU'],               # 2 countries
                'c_rates': [0.25, 0.33, 0.50],         # All C-rates
                'cycle_limits': [1.0, 1.5],             # 2 cycle limits
                'data_limit': 2016,                     # TODO: HERE REPLACE THIS BY DAY/WEEK/YEAR OR SPECIFIC DATE INTERVAL.
                'description': 'Medium validation (3 weeks, 2 countries, 12 scenarios)'
            },
            
        }
        ################################################################################
        ########### Set test level (change this to test different scenarios) ###########

        TEST_LEVEL = 'full' # 'full'   'one_country_full'  # 'quick_all_countries'  Options: 'quick', 'medium', 'full'
        current_config = SCENARIO_CONFIGS[TEST_LEVEL]

        ########### Set test level (change this to test different scenarios) ###########
        ################################################################################


        print(f"📋 Test Configuration: {TEST_LEVEL.upper()}")
        print(f"   {current_config['description']}")
        print(f"   Countries: {current_config['countries']}")
        print(f"   C-rates: {current_config['c_rates']}")
        print(f"   Cycle limits: {current_config['cycle_limits']}")
        if current_config['data_limit']:
            print(f"   Data limit: {current_config['data_limit']} intervals")
        else:
            print(f"   Data limit: Full year")
        ```
##### 3: Result Retrieval and Validation Plotting
**1. Result Retrieval:** After a model is solved and the `solution_dict` is returned by `solve_model()`, this `solution_dict` will be passed to a new method to be created in `optimizer.py`.
    > * **New Method:** Implement `optimizer.extract_solution_dataframe(self, solution_dict: dict) -> pd.DataFrame` within the `BESSOptimizerModelI` class in `optimizer.py`.
    > * **Function:** This method will take the `solution_dict` returned by `solve_model()` and process it into a time-indexed `pandas.DataFrame` (with columns for `p_ch`, `p_dis`, `e_soc`, `c_fcr`, `e_soc_j`, `c_cal_cost`, etc.).
    > * **Inheritance:** `ModelII` and `ModelIII` should extend this method if necessary to include their specific variables (like `e_soc_j` or `c_cal_cost`). This DataFrame is what the plotting functions in Task 3.2 will use.

2. Core Validation Plots (enable options to save plots as `validation_results/{validation_name}/plots/*.html` (or png)):
    - Call suitable plots from `optimization_analysis.py` in `./py_script/visualization/`, to visually validate the optimal bids (BESS Scheduling) from BESSOptimizers (no addtional plot function for this stage).
    - Create a `aging_analysis.py` under `./py_script/visualization/` for Aging Validation plots:
        1. the cyclic aging: Verify the "stacked tank" logic of the 10 SOC segments.
           - Plot: Create a stacked area chart showing the energy in each segment (e_soc_j for $j=1..10$) over time.
           - Success Criteria: The plot must visually confirm that shallower segments (e.g., j=1) are emptied before deeper segments (e.g., j=2) begin to discharge.
        
        2.  the Calendar Aging (Collath et al. Model) Validation:
            - Objective: Verify the SOS2 piecewise-linear cost function.
            - Plot: Create a 2D scatter plot where:
                - X-axis: Total State of Charge (e_soc(t)) [kWh].
                - Y-axis: Calculated Calendar Cost (c_cal_cost(t)) [EUR/hr].
            - Success Criteria: The resulting plot should clearly trace the N-point convex curve (e.g., in the config `aging_config.json`, it's a, N=5, five-breakpoint example) defined by the breakpoints in aging_config.json (e.g., (0 kWh, 1.79 EUR/hr), (1118 kWh, 2.15 EUR/hr), etc.).
1. Options to save the results (solution DataFrame, performance summary JSON, and plots) to designated output folders for further analysis to achieve the following objectives:
    - Create an output folder structure under `validation_results/{validation_name}/` to save:
   1. `validation_results/{validation_name}/solution_*.csv` (detailed decision variable time series)
   2. `validation_results/{validation_name}/performance_*.json` (Summary statistics, such as total profit including coponents profits, degradation costs, solver status, runtime, etc.)

### Prompt: Create a Reusable Result-Saving Utility

    **Objective:**
    I need a new, reusable utility script, `results_exporter.py`, to handle saving all outputs from my optimization runs. This script must be callable from any notebook or script (like `p2b_optimizer.ipynb` or `mpc_simulator.py`) to standardize how results are saved.

    **File to Create:**

    * `./py_script/visualization/results_exporter.py`

    **Core Requirements:**

    1.  **Main Function:**

        * Create a primary function, `save_optimization_results()`.
        * **Inputs:** This function must accept:
            1.  The main solution data (as a `pandas.DataFrame`).
            2.  A dictionary of summary metrics (like `total_profit`, `solve_time`, etc.).
            3.  A descriptive `run_name` string (e.g., "Model\_III\_C-0.5\_7-days").
            4.  An optional `base_output_dir` (defaulting to `"validation_results"`).
        * **Behavior:** The function must:
            1.  Create a unique, timestamped output directory (e.g., `./validation_results/20251112_124500_Model_III_C-0.5_7-days/`).
            2.  Create a `plots` subdirectory inside it.
            3.  Save the solution DataFrame as a CSV file (e.g., `solution_timeseries.csv`).
            4.  Save the summary metrics as a JSON file (e.g., `performance_summary.json`).
        * **Return Value:** The function must return the `Path` to the newly created directory (e.g., `./validation_results/20251112.../`).

    2.  **Integration with Plotting Functions:**

        * All plotting functions (like those in `aging_analysis.py`) must be modified.
        * They must accept an optional `save_path` argument.
        * **If `save_path` is provided,** the function must save the plot to that path (e.g., as an HTML or PNG file).
        * **If `save_path` is `None`,** the function should just display the plot inline (e.g., `fig.show()`).

    **Example Workflow (Conceptual):**

    This is how I expect to use the new utility in my notebooks:

    ```python
    # --- In a notebook ---
    from py_script.visualization.results_exporter import save_optimization_results
    from py_script.visualization.aging_analysis import plot_stacked_cyclic_soc

    # 1. Run the optimizer and extract data
    solution_df = optimizer.extract_solution_dataframe(solution_dict)
    summary = {"total_profit": 10000, "solve_time": 60}
    run_name = "Model_III_Test_Run"

    # 2. Save the data
    # This creates the directory 'validation_results/TIMESTAMP_Model_III_Test_Run/'
    output_directory = save_optimization_results(
        solution_df=solution_df,
        summary_metrics=summary,
        run_name=run_name
    )

    # 3. Save the plots
    # The plot function now saves its output to the 'plots' subdir
    plot_stacked_cyclic_soc(
        solution_df,
        save_path=output_directory / "plots" / "cyclic_soc.html"
    )
    ```
  
### Reminders
1. Build minimum new wheels. Leverage existing code in `./py_script/` and `./notebook`as much as possible. Focus on integrating components into the notebook to create a flexible testing harness and implement the aging validation plots.
2. Write the jupyter notebook in a concise manner, with clear sections strecurte. Only use emojis to highlight key sections for better readability.


### Refactoring `optimizer.py` to Decouple Solving from Extraction
The current `optimizer.py` script has a design flaw called **tight coupling** (紧耦合). The `solve_model` methods in `ModelII` and `ModelIII` are forced to call `super().solve_model()` just to get the base dictionary of results, which they then add to. This "chain" is fragile and violates the Single Responsibility Principle.

  * `solve_model` should be responsible for **Solving**, period.
  * A different method should be responsible for **Extracting** results.

Here is a precise prompt for your coding agent to refactor this into a much cleaner, more robust, and more professional architecture.

-----

### Prompt for Coding Agent: Refactor `optimizer.py` to Decouple Solving from Extraction

**Objective:**
Refactor the `optimizer.py` script to decouple the *solving* logic from the *result extraction* logic. The current implementation mixes these two responsibilities within the `solve_model` methods, leading to a fragile inheritance chain.

The goal is to have one `solve_model` method that *only* solves, and a separate, overridable `extract_solution` method that formats the results.

### Refactoring Plan

#### Task 1: Refactor the `solve_model` Methods

1.  **Modify `BESSOptimizerModelI.solve_model`:**

      * This method should **only** be responsible for calling the Pyomo solver.
      * It should **no longer** extract any variable data (`p_ch`, `p_dis`, `e_soc`, etc.) into a dictionary.
      * It must be modified to return **two** items:
        1.  The solved `model` object (which now contains the solution values).
        2.  The `solver_results` object (returned by `solver.solve()`).
      * **New Signature:**
        ```python
        # In BESSOptimizerModelI
        def solve_model(self, model: pyo.ConcreteModel, solver_name: str = None) -> tuple[pyo.ConcreteModel, Any]:
            # ... (solver detection and configuration logic stays the same)
            
            solver = pyo.SolverFactory(solver_name)
            # ... (solver option logic stays the same)
            
            start_time = datetime.now()
            solver_results = solver.solve(model, tee=False)
            solve_time = (datetime.now() - start_time).total_seconds()

            # ... (status checking logic stays the same)
            
            logger.info(f"Model solved with status: {solver_results.solver.termination_condition}")
            
            # CRITICAL CHANGE: Return the model and results, NOT the extracted dict
            return model, solver_results
        ```

2.  **Delete `BESSOptimizerModelII.solve_model` and `BESSOptimizerModelIII.solve_model`:**

      * These methods are now completely redundant. Their only purpose was to chain the extraction logic.
      * Delete both `solve_model` override methods from `BESSOptimizerModelII` and `BESSOptimizerModelIII`. The single `solve_model` method in `BESSOptimizerModelI` will now be inherited and used by all child classes, as it is generic.

#### Task 2: Create New `extract_solution` Methods (The "Extractor" Chain)

1.  **Create `BESSOptimizerModelI.extract_solution`:**

      * Create a new method in `BESSOptimizerModelI` with the signature:
        `def extract_solution(self, model: pyo.ConcreteModel, solver_results: Any) -> Dict[str, Any]:`
      * **Move all the extraction logic** from the *old* `BESSOptimizerModelI.solve_model` into this new method.
      * This includes extracting:
          * `status`, `solve_time`, `objective_value`
          * Profit components (`profit_da`, `profit_afrr_energy`, etc.)
          * All base variables (`p_ch`, `p_dis`, `e_soc`, `c_fcr`, `p_afrr_pos_e`, `y_fcr`, etc.)
      * This method builds and returns the base `solution_dict`.

2.  **Create `BESSOptimizerModelII.extract_solution`:**

      * Create a new method in `BESSOptimizerModelII` that **overrides** the parent's.
      * It must first call `super()` to get the base dictionary.
      * Then, it adds the **cyclic aging** results (which were in its old `solve_model` method).
      * **Implementation:**
        ```python
        # In BESSOptimizerModelII
        def extract_solution(self, model: pyo.ConcreteModel, solver_results: Any) -> Dict[str, Any]:
            # 1. Get the base solution dict from Model I
            solution_dict = super().extract_solution(model, solver_results)

            # 2. If solve failed, return immediately
            if solution_dict.get('status') not in ['optimal', 'feasible']:
                return solution_dict

            # 3. Add Model II-specific results (cyclic aging)
            # (This is the logic from the OLD ModelII.solve_model)
            p_ch_j, p_dis_j, e_soc_j = {}, {}, {}
            for t in model.T:
                for j in model.J:
                    # ... (logic to extract p_ch_j, p_dis_j, e_soc_j) ...
            
            solution_dict['p_ch_j'] = p_ch_j
            solution_dict['p_dis_j'] = p_dis_j
            solution_dict['e_soc_j'] = e_soc_j
            solution_dict['degradation_metrics'] = self._calculate_degradation_metrics(model, p_dis_j)

            return solution_dict
        ```

3.  **Create `BESSOptimizerModelIII.extract_solution`:**

      * Create a new method in `BESSOptimizerModelIII` that **overrides** its parent (`ModelII`).
      * It must first call `super()` to get the `solution_dict` (which now includes base *and* cyclic results).
      * Then, it adds the **calendar aging** results (the logic from the user-provided `solve_model`).
      * **Implementation:**
        ```python
        # In BESSOptimizerModelIII
        def extract_solution(self, model: pyo.ConcreteModel, solver_results: Any) -> Dict[str, Any]:
            # 1. Get the base + cyclic solution dict from Model II
            solution_dict = super().extract_solution(model, solver_results)

            # 2. If solve failed, return immediately
            if solution_dict.get('status') not in ['optimal', 'feasible']:
                return solution_dict

            # 3. Add Model III-specific results (calendar aging)
            # (This is the logic from the OLD ModelIII.solve_model)
            lambda_cal = {}
            # ... (logic to extract lambda_cal) ...
            
            c_cal_cost, total_calendar_cost = {}, 0.0
            # ... (logic to extract c_cal_cost and calculate total) ...

            solution_dict['lambda_cal'] = lambda_cal
            solution_dict['c_cal_cost'] = c_cal_cost

            # Update degradation metrics
            if 'degradation_metrics' in solution_dict:
                solution_dict['degradation_metrics']['total_calendar_cost_eur'] = total_calendar_cost
                # ... (logic to add cost_breakdown, etc.) ...
                
            return solution_dict
        ```

#### Task 3: Update User-Facing Methods

Finally, update any internal methods (like `optimizer.optimize`) or external scripts (like `mpc_simulator.py`) that call `solve_model`. They must be changed to use the new two-step process.

**Find this (or similar):**

```python
# BEFORE:
results = self.solve_model(model)
# ... use results ...
```

**Replace with this:**

```python
# AFTER:
# 1. Solve the model
model_solved, solver_status = self.solve_model(model)

# 2. Extract the solution
if solver_status.solver.termination_condition in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible]:
    solution_dict = self.extract_solution(model_solved, solver_status)
    # ... use solution_dict ...
else:
    # Handle solve failure
    solution_dict = {'status': 'failed', 'termination_condition': str(solver_status.solver.termination_condition)}
```

**Summary of Benefits:**
This refactoring perfectly separates "solving" from "extracting." The `solve_model` method is now clean, generic, and stable. The `extract_solution` methods form a clean inheritance chain, where each child adds its own data without modifying the parent's logic. This is far more robust and easier to maintain.







