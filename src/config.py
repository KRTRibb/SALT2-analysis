"""
config.py

Project-wide configuration file for paths, constants, and API settings.

Defines:
- Base project directories (data, results, notebooks)
- Subdirectories for raw data, processed data, plots, and analysis outputs
- Output directories for SALT2 fits and density analysis
- External API URLs (e.g., Fink broker)
- sncosmo-specific settings such as FID-to-band mapping and default redshift bounds
- Automatically creates all necessary directories if they do not exist
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent  # project_root/
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Subdirectories
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
PLOTS_DIR = RESULTS_DIR / "plots"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis_results"
TNS_DENSITY_PLOTS_OUTPUT_DIR = PLOTS_DIR / "density_plots_by_TNS"
TNS_DENSITY_ANALYSIS_OUTPUT_DIR = ANALYSIS_RESULTS_DIR / "density_plots_by_TNS"
COLOR_DENSITY_PLOTS_OUTPUT_DIR = PLOTS_DIR / "density_plots_by_color"
COLOR_DENSITY_ANALYSIS_OUTPUT_DIR = ANALYSIS_RESULTS_DIR / "density_plots_by_color"


# Directories for SALT2 light-curve fitting outputs
GENERAL_SNCOSMO_PLOTS = PLOTS_DIR / "general_sncosmo_plots"
FIXED_Z_SNCOSMO_PLOTS = PLOTS_DIR / "sncomso_plots_fixed_z"

# External APIs
FINK_API_URL = "https://api.fink-portal.org"

# SNCosmo specific settings
FID_TO_BAND = {1: "ztfg", 2: "ztfr"}
DEFAULT_Z_BOUNDS = (0.01, 0.2) # Used for bounds on z when fitting the SALT2 model without a known redshi

for d in [RAW_DATA, PROCESSED_DATA, PLOTS_DIR, ANALYSIS_RESULTS_DIR, TNS_DENSITY_ANALYSIS_OUTPUT_DIR, FIXED_Z_SNCOSMO_PLOTS,
            TNS_DENSITY_PLOTS_OUTPUT_DIR, COLOR_DENSITY_ANALYSIS_OUTPUT_DIR, COLOR_DENSITY_PLOTS_OUTPUT_DIR, GENERAL_SNCOSMO_PLOTS]:
    os.makedirs(d, exist_ok=True)
