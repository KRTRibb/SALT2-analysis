"""
fit_salt2_driver.py

Driver script to fit SALT2 models to early Type Ia candidate light curves
using the functions defined in sncosmo_fitting_functions.py.

Workflow:
1. Reads the precomputed 'flux_fits_data.csv' to get object IDs.
2. Calls 'run_fitting' to download light curves from Fink, fit SALT2 models,
   generate light curve plots, and save fit results to CSV.

Outputs:
- GENERAL_SNCOSMO_PLOTS/: directory containing light curve plots.
- general_results.csv: CSV file containing SALT2 fit results.
- general_errors.csv: CSV file of any download or processing errors.
Dependencies:

- config.py for file paths and Fink API URL.
- sncosmo_fitting_functions.py for the fitting workflow.
- flux_fits_data.csv containing precomputed flux fit information.
"""

import config
import sncosmo_fitting.sncosmo_fitting_functions as sff
import pandas as pd
import time

def fit_SALT2():
    """
    Run the SALT2 fitting workflow for all objects in flux_fits_data.csv.

    Reads 'flux_fits_data.csv' to extract unique object IDs, then calls
    'sncosmo_fitting_functions.run_fitting' to process light curves,
    fit SALT2 models, and save plots and CSV outputs.
    """
    flux_fits_df = pd.read_csv(config.RAW_DATA / "flux_fits_data.csv")

    object_ids = flux_fits_df["object id"].unique()

    sff.run_fitting(object_ids)

if __name__ == '__main__':
    start = time.time()
    fit_SALT2()
    print(f"Time taken to run general sncosmo fits {time.time()-start}")