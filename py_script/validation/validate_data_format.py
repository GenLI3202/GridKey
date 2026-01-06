"""
Data Format and Integrity Validation Script

This script provides tools to validate the structure, format, and integrity of
various data files used in the BESS optimizer project. It is essential for
ensuring that data processing stages produce correct outputs and that the
optimizer receives data in the expected format.

Purpose:
- Validate the schema of processed Parquet/JSON data files.
- Check for missing values, outliers, or inconsistencies in market data.
- Verify the structure of Excel-based submission files.
- Ensure configuration files (e.g., aging_config.json) are valid.

How to Use:
- Run from the command line with the path to the file or directory to validate.
- Example:
  python validate_data_format.py --file data/phase2_processed/parquet/DE_2023.parquet
  python validate_data_format.py --submission-file results/submission.xlsx

Based on the test refactoring plan.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate Data Formats and Integrity")
    parser.add_argument("--file", type=str, help="Path to a single data file (Parquet) to validate.")
    parser.add_argument("--dir", type=str, help="Path to a directory of data files to validate.")
    parser.add_argument("--submission-file", type=str, help="Path to an Excel submission file to validate.")
    parser.add_argument("--aging-config", type=str, help="Path to an aging config JSON to validate.")
    return parser.parse_args()

def validate_market_data_schema(df: pd.DataFrame, file_path: str):
    """Validates the schema of a processed market data DataFrame."""
    print(f"\n--- Validating Market Data Schema for: {file_path} ---")
    is_valid = True
    
    required_columns = [
        'timestamp', 'price_day_ahead', 'price_fcr', 'price_afrr_pos',
        'price_afrr_neg', 'price_afrr_energy_pos', 'price_afrr_energy_neg',
        'hour', 'day_of_year', 'month', 'year', 'block_of_day', 'block_id', 'day_id'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"  [FAIL] Missing required columns: {missing_cols}")
        is_valid = False
    else:
        print("  [PASS] All required columns are present.")

    # Check for null values
    if df.isnull().sum().sum() > 0:
        print(f"  [WARN] Null values detected. Columns with nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        # This might be a warning or a failure depending on strictness
    else:
        print("  [PASS] No null values detected.")

    # Check timestamp continuity
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
         df['timestamp'] = pd.to_datetime(df['timestamp'])
         
    time_diffs = df['timestamp'].diff().dropna()
    if not (time_diffs == pd.Timedelta(minutes=15)).all():
        print("  [WARN] Timestamps are not all 15-minute intervals.")
    else:
        print("  [PASS] Timestamps are continuous 15-minute intervals.")

    print(f"--- Schema Validation Result: {'PASS' if is_valid else 'FAIL'} ---")
    return is_valid

def validate_submission_file(file_path: str):
    """Validates the structure of a competition submission Excel file."""
    print(f"\n--- Validating Submission File: {file_path} ---")
    is_valid = True
    try:
        xls = pd.ExcelFile(file_path)
    except FileNotFoundError:
        print(f"  [FAIL] File not found: {file_path}")
        return False

    required_sheets = ['BESS_Dispatch', 'aFRR_Dispatched', 'Day_Ahead_Dispatched']
    missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]

    if missing_sheets:
        print(f"  [FAIL] Missing required sheets: {missing_sheets}")
        is_valid = False
    else:
        print("  [PASS] All required sheets are present.")
        # Further checks can be added here for columns in each sheet
        # For example, check BESS_Dispatch columns
        bess_df = pd.read_excel(xls, 'BESS_Dispatch')
        bess_cols = ['block_id', 'p_ch', 'p_dis']
        if not all(col in bess_df.columns for col in bess_cols):
            print(f"  [FAIL] BESS_Dispatch sheet is missing required columns.")
            is_valid = False
        else:
            print("  [PASS] BESS_Dispatch sheet has the correct columns.")

    print(f"--- Submission Validation Result: {'PASS' if is_valid else 'FAIL'} ---")
    return is_valid

def validate_aging_config(file_path: str):
    """Validates the structure of the aging_config.json file."""
    print(f"\n--- Validating Aging Config: {file_path} ---")
    is_valid = True
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [FAIL] Could not read or parse JSON file: {e}")
        return False

    required_keys = ['num_segments', 'marginal_costs', 'cost_per_full_cycle_eur']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"  [FAIL] Missing required keys in config: {missing_keys}")
        is_valid = False
    else:
        print("  [PASS] All top-level keys are present.")
        if len(config['marginal_costs']) != config['num_segments']:
            print("  [FAIL] Length of 'marginal_costs' does not match 'num_segments'.")
            is_valid = False
        else:
            print("  [PASS] 'marginal_costs' length matches 'num_segments'.")

    print(f"--- Aging Config Validation Result: {'PASS' if is_valid else 'FAIL'} ---")
    return is_valid


def main():
    """Main execution function."""
    args = get_args()
    
    if not any([args.file, args.dir, args.submission_file, args.aging_config]):
        print("No action specified. Use -h for help.")
        sys.exit(1)

    if args.file:
        file_path = Path(args.file)
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
            validate_market_data_schema(df, args.file)
        else:
            print(f"Unsupported file type for --file: {file_path.suffix}. Only .parquet files are supported.")

    if args.dir:
        # Basic implementation: iterate and validate parquet files
        dir_path = Path(args.dir)
        for parquet_file in dir_path.glob('*.parquet'):
            df = pd.read_parquet(parquet_file)
            validate_market_data_schema(df, str(parquet_file))

    if args.submission_file:
        validate_submission_file(args.submission_file)

    if args.aging_config:
        validate_aging_config(args.aging_config)

if __name__ == '__main__':
    main()
