"""
fit_salt2_fixed_z_driver.py

Driver script to fit SALT2 models to early Type Ia candidate light curves
with fixed redshifts.

Workflow:
1. Reads 'flux_fits_data.csv' from the RAW_DATA directory to extract unique
   object IDs along with their TNS classification and redshift.
2. Filters objects to those with non-missing classifications and redshifts.
3. Calls 'run_fitting_fixed_z' from sncosmo_fitting_functions to:
   - Download and process light curves from the Fink broker.
   - Fit SALT2 models with redshift fixed.
   - Generate and save light curve plots.
   - Save CSV files with fit results and any processing errors.

Outputs:
- FIXED_Z_SNCOSMO_PLOTS/: directory containing light curve plots.
- fixed_z_results.csv: CSV of SALT2 fit results with fixed redshift.
- fixed_z_errors.csv: CSV of download or processing errors.

Dependencies:
- config.py for file paths and Fink API URL.
- sncosmo_fitting_functions.py for the fitting workflow.
- flux_fits_data.csv containing precomputed flux fit information.
"""

import sncosmo_fitting_functions as sff
import config
import pandas as pd

def fit_SALT2_fixed_z():
    """
    Run the fixed-redshift SALT2 fitting workflow for all valid objects in flux_fits_data.csv.

    Workflow:
    - Reads 'flux_fits_data.csv' and selects objects with both TNS classification
      and redshift.
    - Calls 'sncosmo_fitting_functions.run_fitting_fixed_z' to process light curves,
      fit SALT2 models with fixed redshift, and save plots and CSV outputs.
    """
    flux_fits_df = pd.read_csv(config.RAW_DATA / "flux_fits_data.csv")
    flux_fits_df_unique = flux_fits_df.drop_duplicates(subset="object id")

    mask = flux_fits_df_unique["TNS classified"].notna() & flux_fits_df["redshift"].notna()

    obj_info = list(
        zip(
            flux_fits_df_unique.loc[mask, "object id"],
            flux_fits_df_unique.loc[mask, "TNS classified"],
            flux_fits_df_unique.loc[mask, "redshift"]
        )
    )

    sff.run_fitting_fixed_z(obj_info)
    print("done")

if __name__ == '__main__':
    fit_SALT2_fixed_z()