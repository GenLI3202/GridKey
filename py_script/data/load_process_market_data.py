"""Market data exploration utilities for TechArena 2025 Phase I.

This module loads the official Huawei TechArena 2025 data workbook, reshapes the
Day-Ahead, FCR, and aFRR market tables into wide-format pandas DataFrames, and offers
both Plotly-based visualizations and format conversion utilities.

Data Format
-----------
By default, the module loads data in    melted_df = afrr_raw.melt(
        id_vars=[TIMESTAMP_COL],
        value_vars=country_dir_cols,
        var_name="country_direction",
        value_name=PRICE_COL_MW,
    ).dropna(subset=[PRICE_COL_MW])format:
- Day-ahead & FCR: columns [timestamp, DE_LU/DE, AT, CH, HU, CZ]  
- aFRR: columns [timestamp, DE_Pos, DE_Neg, AT_Pos, AT_Neg, ...]

Format conversion helpers are provided to convert between wide and tidy formats
as needed for different analysis tasks.

Typical usage
-------------
>>> from pathlib import Path
>>> from market_da import (
...     load_market_tables,
...     wide_to_tidy_day_ahead,
...     plot_day_ahead_distribution,
... )
>>> # Load data in wide format (default)
>>> tables = load_market_tables(Path("../SoloGen_TechArena2025_Phase1/input/TechArena2025_data.xlsx"))
>>> print(tables["day_ahead"].columns)  # ['timestamp', 'DE_LU', 'AT', 'CH', 'HU', 'CZ']
>>> 
>>> # Convert to tidy format for specific analyses
>>> tidy_da = wide_to_tidy_day_ahead(tables["day_ahead"]) 
>>> print(tidy_da.columns)  # ['timestamp', 'country', 'price_eur_mwh']
>>>
>>> # Plotting works with both formats
>>> da_fig = plot_day_ahead_distribution(tables["day_ahead"])
>>> da_fig.show()

All helpers return pandas DataFrames or Plotly Figure instances so they can be
embedded into notebooks, dashboards, or downstream reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAY_AHEAD_SHEET = "Day-ahead prices"
FCR_SHEET = "FCR prices"
AFRR_SHEET = "aFRR capacity prices"

TIMESTAMP_COL = "timestamp"
COUNTRY_COL = "country"
PRICE_COL_MWH = "price_eur_mwh"  # For day-ahead (energy prices)
PRICE_COL_MW = "price_eur_mw"    # For FCR and aFRR (capacity prices)
DIRECTION_COL = "direction"

AFRR_DIRECTION_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "up": "positive",
    "+": "positive",
    "upward": "positive",
    "negative": "negative",
    "neg": "negative",
    "down": "negative",
    "-": "negative",
    "downward": "negative",
}


@dataclass(frozen=True)
class MarketTables:
    """Container aggregating tidy DataFrames for each market."""

    day_ahead: pd.DataFrame
    fcr: pd.DataFrame
    afrr: pd.DataFrame

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        """Return the three tables as a dictionary."""
        return {"day_ahead": self.day_ahead, "fcr": self.fcr, "afrr": self.afrr}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_market_tables(workbook_path: Path, *, prefer_csv: bool = False) -> Dict[str, pd.DataFrame]:
    """Load the three market tables as tidy DataFrames.

    Parameters
    ----------
    workbook_path:
        Path to ``TechArena2025_data.xlsx`` or a directory containing CSV exports
        named ``day_ahead.csv``, ``fcr.csv``, and ``afrr.csv``.
    prefer_csv:
        If ``True`` and CSV files exist alongside the workbook, they are used as
        the source of truth. Otherwise the Excel workbook is parsed directly.

    Returns
    -------
    dict
        Keys: ``"day_ahead"``, ``"fcr"``, ``"afrr"``. Values: wide format DataFrames
        with timestamp as first column and countries/directions as separate columns.
        Day-ahead and FCR: timestamp, DE_LU, AT, CH, HU, CZ
        aFRR: timestamp, DE_Pos, DE_Neg, AT_Pos, AT_Neg, CH_Pos, CH_Neg, HU_Pos, HU_Neg, CZ_Pos, CZ_Neg
    """

    workbook_path = workbook_path.expanduser().resolve()

    if workbook_path.is_dir():
        # Interpret the input as a folder containing CSV files.
        directory = workbook_path
        day_ahead_df = _load_csv(directory / "day_ahead.csv", value_name=PRICE_COL_MWH)
        fcr_df = _load_csv(directory / "fcr.csv", value_name=PRICE_COL_MW)
        afrr_df = _load_csv(directory / "afrr.csv", value_name=PRICE_COL_MW)
    else:
        directory = workbook_path.parent
        csv_candidates = {
            "day_ahead": directory / "day_ahead.csv",
            "fcr": directory / "fcr.csv",
            "afrr": directory / "afrr.csv",
        }
        if prefer_csv and all(path.exists() for path in csv_candidates.values()):
            day_ahead_df = _load_csv(csv_candidates["day_ahead"], value_name=PRICE_COL_MWH)
            fcr_df = _load_csv(csv_candidates["fcr"], value_name=PRICE_COL_MW)
            afrr_df = _load_csv(csv_candidates["afrr"], value_name=PRICE_COL_MW)
        else:
            xl = pd.ExcelFile(workbook_path)
            day_ahead_raw = xl.parse(DAY_AHEAD_SHEET)
            fcr_raw = xl.parse(FCR_SHEET)
            afrr_raw = xl.parse(AFRR_SHEET)

            day_ahead_df = _tidy_market_frame(day_ahead_raw, value_name=PRICE_COL_MWH)
            fcr_df = _tidy_market_frame(fcr_raw, value_name=PRICE_COL_MW)
            afrr_df = _tidy_afrr_frame(afrr_raw)
            
            # Convert price columns to numeric for all tables
            for col in day_ahead_df.columns[1:]:
                day_ahead_df[col] = pd.to_numeric(day_ahead_df[col], errors='coerce')
            for col in fcr_df.columns[1:]:
                fcr_df[col] = pd.to_numeric(fcr_df[col], errors='coerce')
            for col in afrr_df.columns[1:]:
                afrr_df[col] = pd.to_numeric(afrr_df[col], errors='coerce')

    return MarketTables(day_ahead_df, fcr_df, afrr_df).as_dict()


def _load_csv(csv_path: Path, *, value_name: str) -> pd.DataFrame:
    """Load CSV exports that follow the tidy schema used in this module."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df



# ===========================================================================
# PHASE 2 EXTENSIONS
# ===========================================================================

# Phase 2 Constants
AFRR_ENERGY_SHEET = "aFRR energy prices"

# Validation Constants
PRICE_BOUNDS = {
    'day_ahead': (-500, 2000),    # EUR/MWh (allow extreme scarcity prices)
    'fcr': (0, 10000),             # EUR/MW (capacity always non-negative)
    'afrr_capacity': (0, 10000),   # EUR/MW
    'afrr_energy': (-500, 2000)    # EUR/MWh (allow extreme scarcity prices)
}

ZERO_THRESHOLD_PCT = 95  # Flag if >95% zeros


def load_phase2_market_tables(workbook_path: Path, *, prefer_csv: bool = False) -> Dict[str, pd.DataFrame]:
    """Load Phase 2 market tables including aFRR energy prices.

    Parameters
    ----------
    workbook_path : Path
        Path to TechArena2025_Phase2_data.xlsx
    prefer_csv : bool, optional
        If True and CSV cache exists, use it instead of Excel

    Returns
    -------
    dict
        Keys: 'day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy' (NEW)
        Values: Wide-format DataFrames

    DataFrame Formats
    -----------------
    day_ahead:
        Columns: [timestamp, DE_LU, AT, CH, HU, CZ]

    fcr:
        Columns: [timestamp, DE, AT, CH, HU, CZ]

    afrr_capacity:
        Columns: [timestamp, DE_Pos, DE_Neg, AT_Pos, AT_Neg, ...]

    afrr_energy (NEW):
        Columns: [timestamp, DE_Pos, DE_Neg, AT_Pos, AT_Neg, ...]
    """
    from py_script.data.exceptions import DataLoadingError
    import logging

    logger = logging.getLogger(__name__)
    workbook_path = workbook_path.expanduser().resolve()

    try:
        xl = pd.ExcelFile(workbook_path)
    except FileNotFoundError:
        raise DataLoadingError(f"Excel file not found: {workbook_path}")
    except Exception as e:
        raise DataLoadingError(f"Failed to open Excel file: {e}")

    tables = {}

    # Load each sheet with individual error handling
    sheet_configs = [
        (DAY_AHEAD_SHEET, 'day_ahead', _tidy_market_frame, PRICE_COL_MWH),
        (FCR_SHEET, 'fcr', _tidy_market_frame, PRICE_COL_MW),
        (AFRR_SHEET, 'afrr_capacity', _tidy_afrr_frame, None),
        (AFRR_ENERGY_SHEET, 'afrr_energy', _tidy_afrr_frame, None)  # NEW
    ]

    for sheet_name, table_key, loader_func, value_name in sheet_configs:
        try:
            raw_df = xl.parse(sheet_name)

            if value_name:
                processed_df = loader_func(raw_df, value_name=value_name)
            else:
                processed_df = loader_func(raw_df)

            # Convert to numeric
            for col in processed_df.columns[1:]:  # Skip timestamp
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            tables[table_key] = processed_df
            logger.info(f"Loaded {table_key}: {len(processed_df)} rows, {len(processed_df.columns)} columns")

        except KeyError:
            logger.warning(f"Sheet '{sheet_name}' not found, skipping...")
            continue  # Allow partial loading
        except Exception as e:
            logger.error(f"Error processing sheet '{sheet_name}': {e}")
            raise DataLoadingError(f"Failed to process sheet '{sheet_name}': {e}")

    # Validate we have minimum required tables
    if 'day_ahead' not in tables:
        raise DataLoadingError("Critical: Day-ahead data missing")

    return tables


# ===========================================================================  

def _coerce_timestamp_column(series: pd.Series) -> pd.Series:
    """Coerce heterogeneous timestamp column into pandas datetime, dropping header artifacts."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"": None, "nan": None, "NaT": None, "Timestep": None, "Timestamp": None})
    return pd.to_datetime(cleaned, errors="coerce")


def _tidy_market_frame(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Skip the first row which contains country names, and use the original wide format
    df_clean = df.iloc[1:].copy()  # Skip header row
    
    # Set proper column names: first column is timestamp, rest are country prices
    original_ts_col = df.columns[0]
    df_clean = df_clean.rename(columns={original_ts_col: TIMESTAMP_COL})
    
    # Clean up the timestamp column
    df_clean[TIMESTAMP_COL] = pd.to_datetime(df_clean[TIMESTAMP_COL])
    
    # Get country names from the first row
    country_names = df.iloc[0, 1:].values  # Skip first column (timestamp header)
    
    # Rename columns to country names
    for i, country in enumerate(country_names, 1):
        if i < len(df.columns):
            df_clean = df_clean.rename(columns={df.columns[i]: str(country).strip()})
    
    # Keep only valid columns (timestamp + countries with actual data)  
    valid_cols = [TIMESTAMP_COL]
    for col in df_clean.columns[1:]:
        # Only check first 6 columns (timestamp + 5 countries) and skip NaN columns  
        if len(valid_cols) >= 6:  # timestamp + 5 countries max
            break
        try:
            has_data = not df_clean[col].isna().all()
            if has_data:
                valid_cols.append(col)
        except:
            continue
    
    df_clean = df_clean[valid_cols]
    
    # Remove rows with invalid timestamps and sort
    df_clean = df_clean.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    
    return df_clean


def _tidy_afrr_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Skip the first two rows which contain country names and pos/neg labels
    df_clean = df.iloc[2:].copy()
    
    # Set timestamp column name
    original_ts_col = df.columns[0]
    df_clean = df_clean.rename(columns={original_ts_col: TIMESTAMP_COL})
    
    # Clean up the timestamp column
    df_clean[TIMESTAMP_COL] = pd.to_datetime(df_clean[TIMESTAMP_COL])
    
    # Get country names and directions from first two rows
    countries_row = df.iloc[0, 1:].values  # Country names
    directions_row = df.iloc[1, 1:].values  # Pos/Neg labels
    
    # Create column names like "DE_Pos", "DE_Neg", "AT_Pos", etc.
    new_column_names = [TIMESTAMP_COL]
    current_country = None
    
    for i, (country, direction) in enumerate(zip(countries_row, directions_row), 1):
        if pd.notna(country) and str(country).strip():
            current_country = str(country).strip()
        
        if pd.notna(direction) and str(direction).strip() and current_country:
            direction_str = str(direction).strip()
            # Standardize direction naming
            if direction_str.lower() in ['pos', 'positive', '+']:
                direction_str = 'Pos'
            elif direction_str.lower() in ['neg', 'negative', '-']:
                direction_str = 'Neg'
            
            col_name = f"{current_country}_{direction_str}"
            new_column_names.append(col_name)
        else:
            new_column_names.append(f"Unknown_{i}")
    
    # Rename columns
    old_columns = df_clean.columns.tolist()
    rename_dict = {old_columns[i]: new_column_names[i] for i in range(min(len(old_columns), len(new_column_names)))}
    df_clean = df_clean.rename(columns=rename_dict)
    
    # Keep only valid columns (timestamp + columns with actual data)
    valid_cols = [TIMESTAMP_COL]
    for col in df_clean.columns[1:]:
        if not df_clean[col].isna().all():
            valid_cols.append(col)
    
    df_clean = df_clean[valid_cols]
    
    # Remove rows with invalid timestamps and sort
    df_clean = df_clean.dropna(subset=[TIMESTAMP_COL]).sort_values(TIMESTAMP_COL).reset_index(drop=True)
    
    return df_clean


def _split_afrr_series_labels(labels: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Split combined labels such as "Germany_positive" into components."""
    country_tokens = []
    direction_tokens = []
    for label in labels:
        parts = str(label).replace("-", "_").replace("/", "_").split("_")
        parts = [token for token in parts if token]
        if not parts:
            country_tokens.append("unknown")
            direction_tokens.append("unspecified")
            continue

        # Assume the last token encodes the direction (positive/negative).
        *country_part, direction_part = parts if len(parts) > 1 else (parts[0], "unspecified")
        direction_normalised = AFRR_DIRECTION_ALIASES.get(direction_part.lower(), "unspecified")
        country_label = " ".join(country_part).strip() or "unknown"
        country_tokens.append(country_label)
        direction_tokens.append(direction_normalised)

    return pd.Series(country_tokens), pd.Series(direction_tokens)


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------


def wide_to_tidy_day_ahead(day_ahead_df: pd.DataFrame) -> pd.DataFrame:
    """Convert day-ahead DataFrame from wide format to tidy format.
    
    Parameters
    ----------
    day_ahead_df : pd.DataFrame
        Wide format DataFrame with columns: timestamp, DE_LU, AT, CH, HU, CZ
        
    Returns
    -------
    pd.DataFrame
        Tidy format DataFrame with columns: timestamp, country, price_eur_mwh
    """
    country_cols = [col for col in day_ahead_df.columns if col != TIMESTAMP_COL]
    
    tidy_df = day_ahead_df.melt(
        id_vars=[TIMESTAMP_COL],
        value_vars=country_cols,
        var_name=COUNTRY_COL,
        value_name=PRICE_COL_MWH
    ).dropna(subset=[PRICE_COL_MWH])
    
    return tidy_df.sort_values([COUNTRY_COL, TIMESTAMP_COL]).reset_index(drop=True)


def wide_to_tidy_fcr(fcr_df: pd.DataFrame) -> pd.DataFrame:
    """Convert FCR DataFrame from wide format to tidy format.
    
    Parameters
    ----------
    fcr_df : pd.DataFrame
        Wide format DataFrame with columns: timestamp, DE, AT, CH, HU, CZ
        
    Returns
    -------
    pd.DataFrame
        Tidy format DataFrame with columns: timestamp, country, price_eur_mwh
    """
    country_cols = [col for col in fcr_df.columns if col != TIMESTAMP_COL]
    
    tidy_df = fcr_df.melt(
        id_vars=[TIMESTAMP_COL],
        value_vars=country_cols,
        var_name=COUNTRY_COL,
        value_name=PRICE_COL_MW
    ).dropna(subset=[PRICE_COL_MW])
    
    return tidy_df.sort_values([COUNTRY_COL, TIMESTAMP_COL]).reset_index(drop=True)


def wide_to_tidy_afrr(afrr_df: pd.DataFrame) -> pd.DataFrame:
    """Convert aFRR DataFrame from wide format to tidy format.
    
    Parameters
    ----------
    afrr_df : pd.DataFrame
        Wide format DataFrame with columns: timestamp, DE_Pos, DE_Neg, AT_Pos, AT_Neg, etc.
        
    Returns
    -------
    pd.DataFrame
        Tidy format DataFrame with columns: timestamp, country, direction, price_eur_mwh
    """
    country_dir_cols = [col for col in afrr_df.columns if col != TIMESTAMP_COL]
    
    # Melt to long format
    melted_df = afrr_df.melt(
        id_vars=[TIMESTAMP_COL],
        value_vars=country_dir_cols,
        var_name="country_direction",
        value_name=PRICE_COL_MW
    ).dropna(subset=[PRICE_COL_MW])
    
    # Split country_direction column into separate country and direction columns
    melted_df[COUNTRY_COL] = melted_df["country_direction"].str.rsplit("_", n=1).str[0]
    melted_df[DIRECTION_COL] = melted_df["country_direction"].str.rsplit("_", n=1).str[1].str.lower()
    
    # Standardize direction labels
    melted_df[DIRECTION_COL] = melted_df[DIRECTION_COL].map({
        'pos': 'positive',
        'neg': 'negative'
    }).fillna(melted_df[DIRECTION_COL])
    
    return (
        melted_df.drop(columns=["country_direction"])
        .sort_values([COUNTRY_COL, DIRECTION_COL, TIMESTAMP_COL])
        .reset_index(drop=True)
    )


def convert_tables_to_tidy(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Convert all market tables from wide format to tidy format.
    
    Parameters
    ----------
    tables : dict
        Dictionary with keys 'day_ahead', 'fcr', 'afrr' and wide format DataFrames as values
        
    Returns
    -------
    dict
        Dictionary with the same keys but tidy format DataFrames as values
    """
    tidy_tables = {}
    
    if 'day_ahead' in tables:
        tidy_tables['day_ahead'] = wide_to_tidy_day_ahead(tables['day_ahead'])
    
    if 'fcr' in tables:
        tidy_tables['fcr'] = wide_to_tidy_fcr(tables['fcr'])
        
    if 'afrr' in tables:
        tidy_tables['afrr'] = wide_to_tidy_afrr(tables['afrr'])
    
    return tidy_tables


# The converters from tidy to wide are just written by Claude Sonnet 4, not so useful 
## as we already loaded the data in wide format.

def tidy_to_wide_day_ahead(day_ahead_tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Convert day-ahead DataFrame from tidy format back to wide format.
    
    Parameters
    ----------
    day_ahead_tidy_df : pd.DataFrame
        Tidy format DataFrame with columns: timestamp, country, price_eur_mwh
        
    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with countries as separate columns
    """
    return day_ahead_tidy_df.pivot(
        index=TIMESTAMP_COL, 
        columns=COUNTRY_COL, 
        values=PRICE_COL_MWH
    ).reset_index()


def tidy_to_wide_fcr(fcr_tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Convert FCR DataFrame from tidy format back to wide format.
    
    Parameters
    ----------
    fcr_tidy_df : pd.DataFrame
        Tidy format DataFrame with columns: timestamp, country, price_eur_mwh
        
    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with countries as separate columns
    """
    return fcr_tidy_df.pivot(
        index=TIMESTAMP_COL, 
        columns=COUNTRY_COL, 
        values=PRICE_COL_MW
    ).reset_index()


def tidy_to_wide_afrr(afrr_tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Convert aFRR DataFrame from tidy format back to wide format.
    
    Parameters
    ----------
    afrr_tidy_df : pd.DataFrame
        Tidy format DataFrame with columns: timestamp, country, direction, price_eur_mwh
        
    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with country_direction as separate columns
    """
    # Create combined country_direction column for pivoting
    afrr_tidy_df = afrr_tidy_df.copy()
    direction_map = {'positive': 'Pos', 'negative': 'Neg'}
    afrr_tidy_df['country_direction'] = (
        afrr_tidy_df[COUNTRY_COL] + '_' + 
        afrr_tidy_df[DIRECTION_COL].map(direction_map).fillna(afrr_tidy_df[DIRECTION_COL])
    )
    
    return afrr_tidy_df.pivot(
        index=TIMESTAMP_COL, 
        columns='country_direction', 
        values=PRICE_COL_MW
    ).reset_index()



# ---------------------------------------------------------------------------
# Analytical summaries
# ---------------------------------------------------------------------------


def summarize_day_ahead(day_ahead_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize day-ahead market data volatility by country.
    """
    summary = day_ahead_df.groupby("country").agg(
        mean_price=("price_eur_mwh", "mean"),
        median_price=("price_eur_mwh", "median"),
        std_dev_price=("price_eur_mwh", "std"),
        # var_price=("price_eur_mwh", "var"),
        min_price=("price_eur_mwh", "min"),
        max_price=("price_eur_mwh", "max"),
        price_range=("price_eur_mwh", lambda x: x.max() - x.min()),
    ).reset_index().sort_values("country")
    return summary

def summarize_fcr(fcr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute average FCR price per country."""
    # Get country columns (all except timestamp)
    summary = fcr_df.groupby("country").agg(
        mean_price_eur_mw=("price_eur_mwh", "mean"),
        median_price_eur_mw=("price_eur_mwh", "median"),
        std_dev_price_eur_mw=("price_eur_mwh", "std"),
        min_price_eur_mw=("price_eur_mwh", "min"),
        max_price_eur_mw=("price_eur_mwh", "max"),
        price_range_eur_mw=("price_eur_mwh", lambda x: x.max() - x.min()),
    ).reset_index().sort_values("country")
    return summary

def summarize_afrr(afrr_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize aFRR prices by country and direction."""
    summary = afrr_df.groupby(["country", "direction"]).agg(
        mean_price_eur_mw=("price_eur_mwh", "mean"),
        median_price_eur_mw=("price_eur_mwh", "median"),
        std_dev_price_eur_mw=("price_eur_mwh", "std"),
        min_price_eur_mw=("price_eur_mwh", "min"),
        max_price_eur_mw=("price_eur_mwh", "max"),
        price_range_eur_mw=("price_eur_mwh", lambda x: x.max() - x.min()),
    ).reset_index().sort_values(["country", "direction"])
    return summary



def ensure_csv_exports(tables: Dict[str, pd.DataFrame], directory: Path) -> None:
    """Persist tidy tables to CSV for faster reloads during experimentation."""
    directory.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        df.to_csv(directory / f"{key}.csv", index=False)



def convert_afrr_energy_zero_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses market data to handle non-activated aFRR energy markets.

    When aFRR energy prices are 0, it signifies not the prices, but that "the market was NOT activated".
    This function replaces these zero prices with NaN to prevent the optimizer
    from incorrectly treating them as free energy.

    Args:
        df: A DataFrame containing market data with columns
            'price_afrr_energy_pos' and 'price_afrr_energy_neg'.

    Returns:
        The DataFrame with zero aFRR energy prices replaced by np.nan.
    """
    df_processed = df.copy()
    
    # Replace 0 with NaN for aFRR energy prices
    if 'price_afrr_energy_pos' in df_processed.columns:
        df_processed['price_afrr_energy_pos'] = df_processed['price_afrr_energy_pos'].replace(0, np.nan)
    
    if 'price_afrr_energy_neg' in df_processed.columns:
        df_processed['price_afrr_energy_neg'] = df_processed['price_afrr_energy_neg'].replace(0, np.nan)
        
    return df_processed


# ===========================================================================
# COUNTRY DATA PREPROCESSING (Phase 2 Pipeline Enhancement)
# ===========================================================================

def _extract_country_from_wide_tables(
    market_tables: Dict[str, pd.DataFrame],
    country: str,
    afrr_ev_weights_config: Optional[Dict] = None
) -> pd.DataFrame:
    """Extract country-specific data from wide-format market tables.

    This function mimics optimizer.extract_country_data() but works directly
    with wide-format tables without requiring MultiIndex conversion.

    Args:
        market_tables: Dict from load_phase2_market_tables() with keys:
                      'day_ahead', 'fcr', 'afrr_capacity', 'afrr_energy'
        country: Country code (DE_LU, AT, CH, HU, CZ)
        afrr_ev_weights_config: Optional config dict for aFRR activation weights

    Returns:
        DataFrame with columns matching optimizer.extract_country_data() output:
        - price_day_ahead, price_fcr, price_afrr_pos, price_afrr_neg
        - price_afrr_energy_pos, price_afrr_energy_neg (with 0→NaN preprocessing)
        - w_afrr_pos, w_afrr_neg (activation weights)
        - hour, day_of_year, month, year, block_of_day, block_id, day_id, timestamp
    """
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Extracting data for country: {country}")

    # Handle Germany special case
    if country == 'DE_LU':
        day_ahead_country = 'DE_LU'  # Coupled market
        as_country = 'DE'  # Ancillary services
    else:
        day_ahead_country = country
        as_country = country

    # Get timestamps from day-ahead (15-min intervals)
    if 'day_ahead' not in market_tables:
        raise ValueError("day_ahead market data not found")

    # Normalize timestamps to minute precision (Excel has inconsistent seconds: 00:00:00.5, 00:00:01, etc.)
    # CRITICAL FIX: Use floor('min') not floor('s') because source data has random +1 second offsets
    timestamps = pd.to_datetime(market_tables['day_ahead'][TIMESTAMP_COL]).dt.floor('min')

    # Initialize output DataFrame
    country_df = pd.DataFrame()

    # Extract day-ahead prices
    try:
        country_df['price_day_ahead'] = market_tables['day_ahead'][day_ahead_country].values
    except KeyError:
        raise ValueError(f"Day-ahead data not found for {day_ahead_country}")

    # Extract FCR prices (need to forward-fill from 4-hour blocks to 15-min)
    try:
        fcr_df = market_tables['fcr'].copy()
        # Normalize to minute precision (match DA timestamps)
        fcr_df[TIMESTAMP_COL] = pd.to_datetime(fcr_df[TIMESTAMP_COL]).dt.floor('min')
        fcr_df = fcr_df.set_index(TIMESTAMP_COL)
        # Reindex to 15-min and forward fill
        fcr_reindexed = fcr_df.reindex(timestamps).ffill()
        country_df['price_fcr'] = fcr_reindexed[as_country].values
    except KeyError:
        raise ValueError(f"FCR data not found for {as_country}")

    # Extract aFRR capacity prices (need to forward-fill from 4-hour blocks to 15-min)
    try:
        afrr_cap_df = market_tables['afrr_capacity'].copy()
        # Normalize to minute precision (match DA timestamps)
        afrr_cap_df[TIMESTAMP_COL] = pd.to_datetime(afrr_cap_df[TIMESTAMP_COL]).dt.floor('min')
        afrr_cap_df = afrr_cap_df.set_index(TIMESTAMP_COL)
        afrr_cap_reindexed = afrr_cap_df.reindex(timestamps).ffill()
        country_df['price_afrr_pos'] = afrr_cap_reindexed[f'{as_country}_Pos'].values
        country_df['price_afrr_neg'] = afrr_cap_reindexed[f'{as_country}_Neg'].values
    except KeyError as e:
        raise ValueError(f"aFRR capacity data not found for {as_country}: {e}")

    # Extract aFRR energy prices (15-min intervals) with CRITICAL 0→NaN preprocessing
    try:
        afrr_energy_df = market_tables['afrr_energy']
        country_df['price_afrr_energy_pos'] = afrr_energy_df[f'{as_country}_Pos'].values
        country_df['price_afrr_energy_neg'] = afrr_energy_df[f'{as_country}_Neg'].values

        # CRITICAL: Convert 0 -> NaN (0 means "not activated", not "free energy")
        country_df['price_afrr_energy_pos'] = country_df['price_afrr_energy_pos'].replace(0, np.nan)
        country_df['price_afrr_energy_neg'] = country_df['price_afrr_energy_neg'].replace(0, np.nan)
        logger.info(f"Preprocessed aFRR energy prices: 0 -> NaN (prevents false arbitrage)")

    except KeyError:
        logger.warning(f"aFRR energy data not available for {as_country}. Setting to NaN.")
        country_df['price_afrr_energy_pos'] = np.nan
        country_df['price_afrr_energy_neg'] = np.nan

    # Add aFRR activation weights
    if afrr_ev_weights_config:
        # Use historical_activation by default (matches optimizer.py behavior)
        config_section = afrr_ev_weights_config.get('historical_activation', afrr_ev_weights_config)

        # Check for country-specific values
        if 'country_specific' in config_section and as_country in config_section['country_specific']:
            w_pos = config_section['country_specific'][as_country]['positive']
            w_neg = config_section['country_specific'][as_country]['negative']
            logger.info(f"Using country-specific activation rates for {as_country}: pos={w_pos:.2f}, neg={w_neg:.2f}")
        else:
            # Use default values if available
            default_section = config_section.get('default_values', {'positive': 1.0, 'negative': 1.0})
            w_pos = default_section['positive']
            w_neg = default_section['negative']
            logger.info(f"Using default activation rates for {as_country}: pos={w_pos:.2f}, neg={w_neg:.2f}")

        country_df['w_afrr_pos'] = w_pos
        country_df['w_afrr_neg'] = w_neg
    else:
        # No EV weighting: set weights to 1.0
        country_df['w_afrr_pos'] = 1.0
        country_df['w_afrr_neg'] = 1.0
        logger.info(f"EV weighting not configured: using w=1.0 (deterministic)")

    # Create time-based identifiers
    country_df['hour'] = timestamps.dt.hour
    country_df['day_of_year'] = timestamps.dt.dayofyear
    country_df['month'] = timestamps.dt.month
    country_df['year'] = timestamps.dt.year

    # Create block IDs (4-hour blocks starting at midnight)
    country_df['block_of_day'] = country_df['hour'] // 4
    country_df['block_id'] = (country_df['day_of_year'] - 1) * 6 + country_df['block_of_day']

    # Create day IDs
    country_df['day_id'] = country_df['day_of_year']

    # Keep timestamp as a column
    country_df['timestamp'] = timestamps.values

    # Reset index
    country_df = country_df.reset_index(drop=True)

    logger.info(f"Extracted {len(country_df)} data points for {country}")
    return country_df


def save_preprocessed_country_data(
    market_tables: Dict[str, pd.DataFrame],
    output_dir: Path = Path("data/parquet/preprocessed"),
    afrr_ev_weights_config_path: Optional[Path] = None
) -> None:
    """Extract and save each country's preprocessed data.

    This generates validation-ready country-specific parquet files that can be
    loaded directly without Excel parsing or MultiIndex conversion.

    Args:
        market_tables: Dict from load_phase2_market_tables()
        output_dir: Output directory for preprocessed files
        afrr_ev_weights_config_path: Optional path to aFRR EV weights config JSON
    """
    import logging
    import json

    logger = logging.getLogger(__name__)

    # Load aFRR EV weights config if provided
    afrr_ev_config = None
    if afrr_ev_weights_config_path and afrr_ev_weights_config_path.exists():
        with open(afrr_ev_weights_config_path, 'r') as f:
            afrr_ev_config = json.load(f)
        logger.info(f"Loaded aFRR EV weights from {afrr_ev_weights_config_path}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process each country
    countries = ['DE_LU', 'AT', 'CH', 'HU', 'CZ']
    for country in countries:
        logger.info(f"\nProcessing {country}...")

        try:
            # Extract country data
            country_df = _extract_country_from_wide_tables(
                market_tables,
                country,
                afrr_ev_config
            )

            # Save to parquet
            output_path = output_dir / f"{country.lower()}.parquet"
            country_df.to_parquet(output_path, index=False)
            logger.info(f"[OK] Saved {country}: {output_path} ({len(country_df)} rows)")

        except Exception as e:
            logger.error(f"[FAIL] Failed to process {country}: {e}")
            continue

    logger.info(f"\n[OK] Preprocessing complete. Files saved to {output_dir}")


def load_preprocessed_country_data(
    country: str,
    data_dir: Path = Path("data/parquet/preprocessed")
) -> pd.DataFrame:
    """Load pre-processed country data for validation testing.

    This is a fast path that bypasses:
    - Excel workbook loading
    - Wide-format to MultiIndex conversion
    - Country-specific extraction

    Use this for rapid validation testing only.
    For submission, use optimizer.load_and_preprocess_data().

    Args:
        country: Country code (DE_LU, AT, CH, HU, CZ)
        data_dir: Directory containing preprocessed parquet files

    Returns:
        DataFrame ready for optimizer.build_optimization_model()

    Raises:
        FileNotFoundError: If preprocessed file doesn't exist
    """
    country_file = Path(data_dir) / f"{country.lower()}.parquet"

    if not country_file.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {country_file}\n"
            f"Run save_preprocessed_country_data() first to generate preprocessed files."
        )

    return pd.read_parquet(country_file)


# ===========================================================================
# DATA VALIDATION
# ===========================================================================

def validate_phase2_data(tables: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """Comprehensive Phase 2 data quality validation.

    Validates:
    - Row count alignment
    - Timestamp continuity and gaps
    - Price bounds (detect outliers)
    - Excessive zeros
    - Data correlations

    Parameters
    ----------
    tables : dict
        Dictionary of market DataFrames from load_phase2_market_tables()

    Returns
    -------
    dict
        Validation report with keys:
        - 'errors': list of error messages
        - 'warnings': list of warning messages
        - 'stats': dict of validation statistics
        - 'passed': bool indicating if validation passed

    Raises
    ------
    DataValidationError
        If critical errors are found (via calling code)
    """
    report = {
        "errors": [],
        "warnings": [],
        "stats": {},
        "passed": True
    }

    # 1. Validate timestamp alignment (15-min data)
    if 'day_ahead' in tables and 'afrr_energy' in tables:
        day_ahead = tables['day_ahead']
        afrr_energy = tables['afrr_energy']

        if len(afrr_energy) != len(day_ahead):
            report["errors"].append(
                f"Row count mismatch: aFRR energy ({len(afrr_energy)}) "
                f"!= day-ahead ({len(day_ahead)})"
            )
            report["passed"] = False

    # 2. Check timestamp continuity and gaps
    for market, df in tables.items():
        if TIMESTAMP_COL not in df.columns:
            continue

        ts = df[TIMESTAMP_COL]
        expected_freq = '15T' if market in ['day_ahead', 'afrr_energy'] else '4H'

        # Check for gaps
        time_diff = ts.diff()[1:]
        expected_delta = pd.Timedelta(expected_freq)
        gaps = time_diff[time_diff > expected_delta]

        if len(gaps) > 0:
            report["warnings"].append(
                f"{market}: Found {len(gaps)} timestamp gaps "
                f"(largest: {gaps.max()})"
            )
            report["stats"][f"{market}_gaps"] = len(gaps)

    # 3. Price bounds validation
    for market, df in tables.items():
        # Get bounds for this market
        bounds_key = market if market in PRICE_BOUNDS else 'day_ahead'
        bounds = PRICE_BOUNDS.get(bounds_key, (-1000, 10000))

        price_cols = [col for col in df.columns if col != TIMESTAMP_COL]

        for col in price_cols:
            min_val = df[col].min()
            max_val = df[col].max()

            if min_val < bounds[0]:
                report["errors"].append(
                    f"{market}.{col}: Min price {min_val:.2f} < lower bound {bounds[0]}"
                )
                report["passed"] = False

            if max_val > bounds[1]:
                report["errors"].append(
                    f"{market}.{col}: Max price {max_val:.2f} > upper bound {bounds[1]}"
                )
                report["passed"] = False

            # Statistics
            report["stats"][f"{market}.{col}_min"] = float(min_val)
            report["stats"][f"{market}.{col}_max"] = float(max_val)
            report["stats"][f"{market}.{col}_mean"] = float(df[col].mean())

    # 4. Check for excessive zeros (may indicate missing data)
    if 'afrr_energy' in tables:
        afrr_energy = tables['afrr_energy']
        for col in afrr_energy.columns[1:]:  # Skip timestamp
            zero_pct = (afrr_energy[col] == 0).sum() / len(afrr_energy) * 100
            report["stats"][f"afrr_energy.{col}_zero_pct"] = float(zero_pct)

            if zero_pct > ZERO_THRESHOLD_PCT:
                report["warnings"].append(
                    f"aFRR energy {col}: {zero_pct:.1f}% zeros "
                    f"(common for activation prices, but verify)"
                )

    # 5. Correlation checks (sanity check: DA prices should correlate across countries)
    if 'day_ahead' in tables:
        day_ahead = tables['day_ahead']
        da_price_cols = [col for col in day_ahead.columns if col != TIMESTAMP_COL]

        if len(da_price_cols) >= 2:
            corr_matrix = day_ahead[da_price_cols].corr()
            min_corr = corr_matrix.min().min()

            if min_corr < 0.3:  # Expect some correlation in European markets
                report["warnings"].append(
                    f"Day-ahead: Low price correlation detected (min={min_corr:.2f}). "
                    f"Verify data integrity."
                )

            report["stats"]["day_ahead_min_correlation"] = float(min_corr)

    return report

# ---------------------------------------------------------------------------
# CLI bootstrap
# ---------------------------------------------------------------------------


def _cli_example(workbook: Path) -> None:
    tables = load_market_tables(workbook)
    volatility = summarize_day_ahead(tables["day_ahead"])
    fcr_summary = summarize_fcr(tables["fcr"])
    afrr_summary = summarize_afrr(tables["afrr"])

    print("Top 5 arbitrage opportunities (Day-Ahead volatility):")
    print(volatility.head())
    print("\nFCR average prices by country:")
    print(fcr_summary.sort_values("mean", ascending=False))
    print("\naFRR summary (positive vs negative):")
    print(afrr_summary)


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[1] / "SoloGen_TechArena2025_Phase1" / "input" / "TechArena2025_data.xlsx"
    if default_path.exists():
        _cli_example(default_path)
    else:
        raise SystemExit(
            "Unable to locate the default workbook. Pass a valid path or adjust the script."
        )
