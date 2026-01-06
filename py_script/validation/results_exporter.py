#!/usr/bin/env python3
"""
Results Exporter for BESS Optimization
========================================

Reusable utility for saving optimization results to structured output directories.
This module provides a standardized way to export solution data, metrics, and plots
from BESS optimization runs.

Key Features:
- Timestamped output directories for reproducibility
- Automatic directory structure creation (main dir + plots subdirectory)
- Solution DataFrame export to CSV
- Summary metrics export to JSON
- Path handling for cross-platform compatibility

Author: SoloGen Team
Date: November 2025
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


def save_optimization_results(
    solution_df: pd.DataFrame,
    summary_metrics: Dict[str, Any],
    run_name: str,
    base_output_dir: Union[str, Path] = "validation_results"
) -> Path:
    """
    Save optimization results to a timestamped directory structure.
    
    This function creates a standardized output directory for storing BESS
    optimization results, including solution time series data, performance
    metrics, and a dedicated subdirectory for plots.
    
    Directory Structure Created:
    ----------------------------
    base_output_dir/
    └── YYYYMMDD_HHMMSS_run_name/
        ├── solution_timeseries.csv       # Full solution DataFrame
        ├── performance_summary.json      # Summary metrics and metadata
        └── plots/                        # Subdirectory for visualization outputs
    
    Parameters
    ----------
    solution_df : pd.DataFrame
        Complete solution data with time-indexed decision variables.
        Expected columns may include:
        - timestamp, hour: time indices
        - p_ch_kw, p_dis_kw: charge/discharge power
        - e_soc_kwh, soc_pct: state of charge
        - c_fcr_mw, c_afrr_pos_mw, c_afrr_neg_mw: capacity bids
        - revenue_da_eur, revenue_fcr_eur: revenue components
        - cost_cyclic_eur, cost_calendar_eur: degradation costs (if applicable)
        
    summary_metrics : dict
        Dictionary of key performance indicators and metadata.
        Recommended keys:
        - 'total_profit_eur': Total profit/revenue (EUR)
        - 'solve_time_sec': Solver execution time (seconds)
        - 'solver_status': Optimization status (optimal/feasible/failed)
        - 'solver_name': Solver used (cplex/gurobi/highs/etc)
        - 'country': Market country code
        - 'c_rate': C-rate configuration
        - 'daily_cycle_limit': Daily cycle limit (if applicable)
        - 'alpha': Degradation weight parameter (Model II/III)
        - 'time_horizon_hours': Total time horizon
        - 'degradation_metrics': Dict with cyclic/calendar costs (Model II/III)
        
    run_name : str
        Descriptive name for this optimization run.
        Will be sanitized (spaces→underscores, lowercase) and appended to
        the timestamp in the directory name.
        Examples: "CH_7day_ModelIII_alpha1.0", "DE_winter_week_baseline"
        
    base_output_dir : str or Path, optional
        Base directory for all validation results.
        Default: "validation_results" (relative to current working directory)
        
    Returns
    -------
    Path
        Absolute path to the created output directory.
        Use this path to save plots or additional files:
        >>> output_dir = save_optimization_results(df, metrics, "my_test")
        >>> fig.write_html(output_dir / "plots" / "soc_trajectory.html")
        
    Raises
    ------
    ValueError
        If solution_df is empty or summary_metrics is not a dictionary
    OSError
        If directory creation or file writing fails due to permissions
        
    Examples
    --------
    >>> # Basic usage after optimization
    >>> solution_df = optimizer.extract_solution_dataframe(model, test_data, 168)
    >>> metrics = {
    ...     'total_profit_eur': 1234.56,
    ...     'solve_time_sec': 12.3,
    ...     'solver_status': 'optimal',
    ...     'country': 'CH',
    ...     'time_horizon_hours': 168
    ... }
    >>> output_dir = save_optimization_results(
    ...     solution_df, 
    ...     metrics, 
    ...     "CH_1week_baseline"
    ... )
    >>> print(f"Results saved to: {output_dir}")
    
    >>> # Save additional plots to the created directory
    >>> fig = plot_soc_trajectory(solution_df)
    >>> fig.write_html(output_dir / "plots" / "soc.html")
    
    Notes
    -----
    - The timestamp format is YYYYMMDD_HHMMSS for chronological sorting
    - Run names are sanitized: spaces→underscores, converted to lowercase
    - The plots/ subdirectory is created empty; populate it with your visualizations
    - CSV is saved with index (time column) for easier reloading
    - JSON is saved with indent=2 for human readability
    """
    # Input validation
    if solution_df is None or solution_df.empty:
        raise ValueError("solution_df cannot be None or empty")
    
    if not isinstance(summary_metrics, dict):
        raise TypeError(f"summary_metrics must be a dict, got {type(summary_metrics)}")
    
    # Sanitize run_name: replace spaces with underscores, lowercase
    sanitized_name = run_name.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    
    # Create timestamped directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{sanitized_name}"
    
    # Construct full output path
    base_path = Path(base_output_dir).resolve()
    output_dir = base_path / dir_name
    plots_dir = output_dir / "plots"
    
    # Create directory structure
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create directory {output_dir}: {e}")
        raise
    
    # Save solution DataFrame as CSV
    csv_path = output_dir / "solution_timeseries.csv"
    try:
        solution_df.to_csv(csv_path, index=True)
        logger.info(f"Saved solution data: {csv_path} ({len(solution_df)} rows)")
    except Exception as e:
        logger.error(f"Failed to save solution CSV: {e}")
        raise
    
    # Enhance summary_metrics with metadata
    enhanced_metrics = {
        'export_timestamp': datetime.now().isoformat(),
        'run_name': run_name,
        'sanitized_name': sanitized_name,
        'solution_rows': len(solution_df),
        'solution_columns': list(solution_df.columns),
        **summary_metrics  # User-provided metrics
    }
    
    # Save summary metrics as JSON
    json_path = output_dir / "performance_summary.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(enhanced_metrics, f, indent=2, default=str)
        logger.info(f"Saved performance summary: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save summary JSON: {e}")
        raise
    
    # Log success summary
    logger.info(f"✅ Results exported successfully:")
    logger.info(f"   📁 Output directory: {output_dir}")
    logger.info(f"   📊 Solution CSV: {csv_path.name}")
    logger.info(f"   📋 Summary JSON: {json_path.name}")
    logger.info(f"   📈 Plots directory: {plots_dir}")
    
    return output_dir


def load_optimization_results(results_dir: Union[str, Path]) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load previously saved optimization results from a directory.
    
    This is the inverse operation of save_optimization_results(), allowing you
    to reload solution data and metrics for post-processing or comparison.
    
    Parameters
    ----------
    results_dir : str or Path
        Path to the results directory (created by save_optimization_results)
        
    Returns
    -------
    solution_df : pd.DataFrame
        Loaded solution DataFrame with time-indexed decision variables
    summary_metrics : dict
        Loaded summary metrics and metadata
        
    Raises
    ------
    FileNotFoundError
        If results_dir doesn't exist or required files are missing
    ValueError
        If files are corrupted or in wrong format
        
    Examples
    --------
    >>> # Load results from a previous run
    >>> results_dir = Path("validation_results/20251112_143000_ch_baseline")
    >>> solution_df, metrics = load_optimization_results(results_dir)
    >>> print(f"Loaded {len(solution_df)} rows")
    >>> print(f"Total profit: {metrics['total_profit_eur']:.2f} EUR")
    """
    results_path = Path(results_dir).resolve()
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # Load CSV
    csv_path = results_path / "solution_timeseries.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Solution CSV not found: {csv_path}")
    
    try:
        solution_df = pd.read_csv(csv_path, index_col=0)
        logger.info(f"Loaded solution data: {len(solution_df)} rows")
    except Exception as e:
        raise ValueError(f"Failed to load solution CSV: {e}")
    
    # Load JSON
    json_path = results_path / "performance_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            summary_metrics = json.load(f)
        logger.info(f"Loaded performance summary with {len(summary_metrics)} metrics")
    except Exception as e:
        raise ValueError(f"Failed to load summary JSON: {e}")
    
    return solution_df, summary_metrics


def list_saved_results(base_output_dir: Union[str, Path] = "validation_results") -> pd.DataFrame:
    """
    List all saved optimization results in the base output directory.
    
    Scans the base directory for timestamped result folders and returns
    a summary DataFrame for easy comparison and selection.
    
    Parameters
    ----------
    base_output_dir : str or Path, optional
        Base directory containing saved results
        Default: "validation_results"
        
    Returns
    -------
    pd.DataFrame
        Summary of all saved results with columns:
        - timestamp: datetime of result creation
        - run_name: descriptive name
        - directory: path to result directory
        - total_profit_eur: total profit (if available)
        - solve_time_sec: solve time (if available)
        - solver_status: optimization status (if available)
        
    Examples
    --------
    >>> # List all saved results
    >>> results_summary = list_saved_results()
    >>> print(results_summary)
    
    >>> # Find best performing run
    >>> best_run = results_summary.loc[results_summary['total_profit_eur'].idxmax()]
    >>> print(f"Best run: {best_run['run_name']} with {best_run['total_profit_eur']:.2f} EUR")
    """
    base_path = Path(base_output_dir).resolve()
    
    if not base_path.exists():
        logger.warning(f"Base directory not found: {base_path}")
        return pd.DataFrame()
    
    results_list = []
    
    # Scan for result directories
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
            
        json_path = item / "performance_summary.json"
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                
            results_list.append({
                'timestamp': pd.to_datetime(metrics.get('export_timestamp')),
                'run_name': metrics.get('run_name', item.name),
                'directory': str(item),
                'total_profit_eur': metrics.get('total_profit_eur'),
                'solve_time_sec': metrics.get('solve_time_sec'),
                'solver_status': metrics.get('solver_status'),
                'country': metrics.get('country'),
                'time_horizon_hours': metrics.get('time_horizon_hours')
            })
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
            continue
    
    if not results_list:
        logger.info(f"No saved results found in {base_path}")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('timestamp', ascending=False)
    
    logger.info(f"Found {len(results_df)} saved result(s) in {base_path}")
    return results_df


if __name__ == "__main__":
    # Demo: Create example results and save them
    print("Results Exporter Demo")
    print("=" * 60)
    
    # Create sample solution DataFrame
    sample_solution = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=96, freq='15min'),
        'p_ch_kw': [100.0] * 96,
        'p_dis_kw': [50.0] * 96,
        'e_soc_kwh': [2236.0] * 96,
        'soc_pct': [50.0] * 96,
        'revenue_da_eur': [5.0] * 96
    })
    
    # Create sample metrics
    sample_metrics = {
        'total_profit_eur': 480.0,
        'solve_time_sec': 10.5,
        'solver_status': 'optimal',
        'solver_name': 'highs',
        'country': 'CH',
        'c_rate': 0.5,
        'time_horizon_hours': 24
    }
    
    # Save results
    output_dir = save_optimization_results(
        sample_solution,
        sample_metrics,
        "Demo Test Run"
    )
    
    print(f"\n✅ Demo results saved to: {output_dir}")
    
    # List all saved results
    print("\n📋 Listing all saved results:")
    all_results = list_saved_results()
    if not all_results.empty:
        print(all_results.to_string())
    
    # Reload the results we just saved
    print("\n🔄 Reloading saved results...")
    loaded_df, loaded_metrics = load_optimization_results(output_dir)
    print(f"   Loaded {len(loaded_df)} rows")
    print(f"   Total profit: {loaded_metrics['total_profit_eur']:.2f} EUR")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
