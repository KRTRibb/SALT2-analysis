"""
sncosmo_fitting_functions.py

This module provides functions to download, process, and fit light curves of 
early Type Ia supernova candidates using the SALT2 model with sncosmo. 

It primarily works with object data from the Fink broker and uses precomputed
flux fit information stored in a CSV file. It supports fitting with either 
a free redshift or a fixed redshift, saving both the fit results and light 
curve plots.

Functions:
- convert_magpsf_to_flux: Converts ZTF PSF magnitudes to flux in mJy.
- get_and_process_light_curve_data: Downloads and processes light curves 
  for given object IDs.
- get_and_process_light_curve_data_fixed_z: Downloads and processes light 
  curves for given objects with fixed redshift.
- fit_and_save_sncosmo_model: Fits SALT2 to light curves and saves plots 
  for free-redshift fits.
- fit_and_save_sncosmo_model_fixed_z: Fits SALT2 to light curves with fixed 
  redshift and saves plots.
- run_fitting: High-level function to run full free-redshift fitting workflow 
  and save CSV outputs.
- run_fitting_fixed_z: High-level function to run full fixed-redshift fitting 
  workflow and save CSV outputs.

Dependencies:
- pandas, numpy, seaborn, matplotlib, scipy, astropy, typing, sncosmo, io, requests, scipy
- config.py for file paths and output directories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import sncosmo
from scipy.stats import chi2
from typing import Dict, List, Tuple
import requests
import io

import config
from core import convert_magpsf_to_flux

APIURL = config.FINK_API_URL
LightcurveDict = Dict[Tuple[str, ...], Table]



def get_and_process_light_curve_data(object_ids: List[str]) -> Tuple[LightcurveDict, pd.DataFrame]:
    """
    Download and process light curve data for a list of object IDs.

    Parameters
    object_ids : List[str]
        List of object IDs to download light curves for.

    Returns:
    light_curve_dict : LightcurveDict
        Dictionary mapping object_id to an astropy Table of light curve data.
    error_pdf : pd.DataFrame
        DataFrame containing object IDs and any errors encountered during download.

    Notes:
    - Hardcoded date range: 2024-07-31 to 2025-08-06.
    - Includes upper limits (`withupperlim=True`) in the download.
    """
    light_curve_dict: LightcurveDict = {}
    error_pdf = pd.DataFrame(columns=['Object id', 'Error message'])
    fid_to_band = config.FID_TO_BAND

    for obj_id in object_ids:
        try: 
            r = requests.post(
                f'{APIURL}/api/v1/objects',
                json={
                    'objectId': obj_id,
                    'output-format': 'json',
                    'withupperlim': 'True',
                    'startdate': '2024-07-31',
                    'stopdate': '2025-08-06',
                }
            )
            
            light_curve = pd.read_json(io.BytesIO(r.content))
            mask_valid = light_curve['d:tag'] == 'valid'
            lc_valid = light_curve[mask_valid]
            mjd = lc_valid["i:jd"].apply(lambda x: x - 2400000.5)
            flux, fluxerr = convert_magpsf_to_flux(
                lc_valid['i:magpsf'], lc_valid['i:sigmapsf']
            )

            data = Table(
                {
                    "mjd": mjd.to_numpy(),
                    "band": np.array([fid_to_band[fid] for fid in lc_valid["i:fid"]]),
                    "flux": flux.to_numpy(),
                    "fluxerr": fluxerr.to_numpy(),
                    "zp": np.full(len(lc_valid), 25.0),
                    "zpsys": np.full(len(lc_valid), "ab"),
                }
            )
            light_curve_dict[obj_id] = data
        except Exception as e:
            error_pdf = pd.concat(
                [error_pdf, pd.DataFrame({"Object id": [obj_id], "Error message": [str(e)]})],
                ignore_index=True
            )

    return light_curve_dict, error_pdf


def get_and_process_light_curve_data_fixed_z(object_info: List[Tuple[str, str, float]]) -> Tuple[LightcurveDict, pd.DataFrame]:
    """
    Download and process light curve data for objects with fixed redshifts.

    Parameters:
    object_info : List[Tuple[str, str, float]]
        List of tuples (object_id, tns_class, redshift).

    Returns:
    light_curve_dict : LightcurveDict
        Dictionary mapping (object_id, tns_class, z) to an astropy Table of light curve data.
    error_pdf : pd.DataFrame
        DataFrame containing object IDs and any errors encountered during download.

    Notes:
    - Hardcoded date range: 2024-07-31 to 2025-08-06.
    - Includes upper limits (`withupperlim=True`) in the download.
    """

    light_curve_dict: LightcurveDict = {}
    error_pdf = pd.DataFrame(columns=['Object id', 'Error message'])
    fid_to_band = config.FID_TO_BAND

    for obj_id, tns_class, z in object_info:
        try: 
            r = requests.post(
                f'{APIURL}/api/v1/objects',
                json={
                    'objectId': obj_id,
                    'output-format': 'json',
                    'withupperlim': 'True',
                    'startdate': '2024-07-31',
                    'stopdate': '2025-08-06',
                }
            )

            light_curve = pd.read_json(io.BytesIO(r.content))
            mask_valid = light_curve['d:tag'] == 'valid'
            lc_valid = light_curve[mask_valid]
            mjd = lc_valid["i:jd"].apply(lambda x: x - 2400000.5)
            flux, fluxerr = convert_magpsf_to_flux(
                lc_valid['i:magpsf'], lc_valid['i:sigmapsf']
            )

            data = Table(
                {
                    "mjd": mjd.to_numpy(),
                    "band": np.array([fid_to_band[fid] for fid in lc_valid["i:fid"]]),
                    "flux": flux.to_numpy(),
                    "fluxerr": fluxerr.to_numpy(),
                    "zp": np.full(len(lc_valid), 25.0),
                    "zpsys": np.full(len(lc_valid), "ab"),
                }
            )
            light_curve_dict[(obj_id, tns_class, z)] = data
        except Exception as e:
            error_pdf = pd.concat(
                [error_pdf, pd.DataFrame({"Object id": [obj_id], "Error message": [str(e)]})],
                ignore_index=True
            )

    return light_curve_dict, error_pdf


def fit_and_save_sncosmo_model(light_curve_dict: LightcurveDict, save_dir: str) -> pd.DataFrame:
    """
    Fit SALT2 model to light curves with free redshift and save plots.

    Parameters:
    light_curve_dict : LightcurveDict
        Dictionary of light curves keyed by (object_id, 0).
    save_dir : str
        Directory where fit plots will be saved.

    Returns:
    results_pdf : pd.DataFrame
        DataFrame with fit results, including:
        'object id', Chi2 calls, ndof, min chi2, p-value, z, t0, x0, x1, c,
        and uncertainties on each parameter.
    """

    model = sncosmo.Model(source="SALT2")
    col_names = [
        'object id', "Chi2 calls", "ndof", "min chi2", "p-val", "z", "t0", "x0", "x1", "c", "sig z", "sig t0", "sig x0", "sig x1", "sig c" 
        ]
    results_pdf = pd.DataFrame(columns=col_names)

    for obj_id, lc_data in light_curve_dict.items():

        result, fitted_model = sncosmo.fit_lc(
            lc_data,
            model,
            ['z', 't0', 'x0', 'x1', 'c'],
            bounds={'z': config.DEFAULT_Z_BOUNDS}
        )

        params_and_errors = np.concatenate((result.parameters, list(result.errors.values())))
        data_to_save = [obj_id, result.ncall, result.ndof, result.chisq, 1 - chi2.cdf(result.chisq, result.ndof), *params_and_errors]
        results_pdf.loc[len(results_pdf)] = data_to_save

        sncosmo.plot_lc(lc_data, model=fitted_model, errors=result.errors)
        plt.savefig(save_dir / f"{obj_id}.png")

    return results_pdf

def fit_and_save_sncosmo_model_fixed_z(light_curve_dict: LightcurveDict, save_dir: str) -> pd.DataFrame:
    """
    Fit SALT2 model to light curves with fixed redshift and save plots.

    Parameters:
    light_curve_dict : LightcurveDict
        Dictionary of light curves keyed by (object_id, tns_class, z).
    save_dir : str
        Directory where fit plots will be saved.

    Returns:
    results_pdf : pd.DataFrame
        DataFrame with fit results, including:
        'object id', Chi2 calls, ndof, min chi2, p-value, tns_class, z, t0, x0, x1, c,
        and uncertainties on each parameter.
    """

    model = sncosmo.Model(source="SALT2")
    
    col_names = [
        'object id', "Chi2 calls", "ndof", "min chi2", "p-val", "tns_class", "z", "t0", "x0", "x1", "c", "sig t0", "sig x0", "sig x1", "sig c" 
        ]
    results_pdf = pd.DataFrame(columns=col_names)

    for (obj_id, tns_class, fixed_z), lc_data in light_curve_dict.items():

        model.set(z=fixed_z)
        result, fitted_model = sncosmo.fit_lc(
            lc_data,
            model,
            ['t0', 'x0', 'x1', 'c'],
        )

        params_and_errors = np.concatenate((result.parameters, list(result.errors.values())))
        data_to_save = [obj_id, result.ncall, result.ndof, result.chisq, 1 - chi2.cdf(result.chisq, result.ndof), tns_class, *params_and_errors]
        results_pdf.loc[len(results_pdf)] = data_to_save

        sncosmo.plot_lc(lc_data, model=fitted_model, errors=result.errors)
        plt.savefig(save_dir / f"{obj_id}_fixed_z.png")

    return results_pdf

def run_fitting(object_ids: List[str]):
    """
    High-level workflow: download light curves, fit SALT2 with free redshift,
    save fit plots and CSVs.

    Parameters:
    object_ids : List[str]
        List of object IDs to process.

    Outputs:
    - general_errors.csv: CSV of download or processing errors.
    - general_results.csv: CSV of SALT2 fit results.
    - Light curve plots saved to GENERAL_SNCOSMO_PLOTS directory.
    """

    light_curve_dict, error_df = get_and_process_light_curve_data(object_ids)

    error_df.to_csv(config.RAW_DATA / "general_errors.csv")

    results_df = fit_and_save_sncosmo_model(light_curve_dict, save_dir=config.GENERAL_SNCOSMO_PLOTS)

    results_df.to_csv(config.RAW_DATA / "general_results.csv")

def run_fitting_fixed_z(object_info: List[Tuple[str, str, float]]):
    """
    High-level workflow: download light curves, fit SALT2 with fixed redshift,
    save fit plots and CSVs.

    Parameters:
    object_info : List[Tuple[str, str, float]]
        List of tuples (object_id, tns_class, redshift) to process.

    Outputs:
    - fixed_z_errors.csv: CSV of download or processing errors.
    - fixed_z_results.csv: CSV of SALT2 fit results.
    - Light curve plots saved to FIXED_Z_SNCOSMO_PLOTS directory.
    """

    light_curve_dict, error_df = get_and_process_light_curve_data_fixed_z(object_info)

    error_df.to_csv(config.RAW_DATA / "fixed_z_errors.csv")

    results_df = fit_and_save_sncosmo_model_fixed_z(light_curve_dict, save_dir=config.FIXED_Z_SNCOSMO_PLOTS)

    results_df.to_csv(config.RAW_DATA / "fixed_z_results.csv")

