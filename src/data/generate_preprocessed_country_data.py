"""Generate preprocessed country-specific parquet files from Phase 2 data.

This script loads the Phase 2 Excel workbook and generates validation-ready
country-specific parquet files that can be loaded directly without Excel parsing.

Usage:
    python generate_preprocessed_country_data.py

Output:
    data/parquet/preprocessed/
        de_lu.parquet
        at.parquet
        ch.parquet
        hu.parquet
        cz.parquet
"""

from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.load_process_market_data import (
    load_phase2_market_tables,
    save_preprocessed_country_data
)


def main():
    """Generate preprocessed country data files."""

    logger.info("=" * 70)
    logger.info("Generating Preprocessed Country Data for Phase 2")
    logger.info("=" * 70)

    # Define paths
    excel_path = project_root / "data" / "TechArena2025_Phase2_data.xlsx"
    output_dir = project_root / "data" / "parquet" / "preprocessed"
    config_path = project_root / "data" / "p2_config" / "afrr_ev_weights_config.json"

    # Verify Excel file exists
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.error("Please ensure TechArena2025_Phase2_data.xlsx is in data/ folder")
        return 1

    logger.info(f"Input: {excel_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Config: {config_path if config_path.exists() else 'None (will use defaults)'}")
    logger.info("")

    # Step 1: Load Phase 2 market tables
    logger.info("Step 1: Loading Phase 2 market data from Excel...")
    try:
        market_tables = load_phase2_market_tables(excel_path)
        logger.info(f"[OK] Loaded {len(market_tables)} market tables")
        for market, df in market_tables.items():
            logger.info(f"  - {market}: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"[FAIL] Failed to load Excel data: {e}")
        return 1

    logger.info("")

    # Step 2: Generate preprocessed country files
    logger.info("Step 2: Generating preprocessed country files...")
    try:
        save_preprocessed_country_data(
            market_tables,
            output_dir=output_dir,
            afrr_ev_weights_config_path=config_path if config_path.exists() else None
        )
    except Exception as e:
        logger.error(f"[FAIL] Failed to generate preprocessed files: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info("")

    # Step 3: Verify output files
    logger.info("Step 3: Verifying output files...")
    countries = ['de_lu', 'at', 'ch', 'hu', 'cz']
    all_exist = True

    for country in countries:
        file_path = output_dir / f"{country}.parquet"
        if file_path.exists():
            # Get file size
            file_size = file_path.stat().st_size / 1024  # KB
            logger.info(f"[OK] {country.upper():5s}: {file_path.name:15s} ({file_size:.1f} KB)")
        else:
            logger.error(f"[FAIL] {country.upper():5s}: File not found")
            all_exist = False

    logger.info("")

    if all_exist:
        logger.info("=" * 70)
        logger.info("SUCCESS! All preprocessed files generated.")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Test loading: load_preprocessed_country_data('DE_LU')")
        logger.info("  2. Verify data integrity")
        logger.info("  3. Update optimizer to use new pipeline")
        return 0
    else:
        logger.error("=" * 70)
        logger.error("FAILED: Some files were not generated")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
