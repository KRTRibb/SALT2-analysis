"""
sigmoid_power_fits.py

Functions for fitting early Type Ia candidate light curves from the Fink broker
with simple parametric models (sigmoid and power law).

This module provides tools to:
- Query the Fink broker for candidate SN Ia object IDs in a given date range.
- Fetch and preprocess light curve data for individual objects.
- Fit light curves with sigmoid and power-law functions in flux space.
- Compute chi-squared statistics, reduced chi-squared, and p-values.
- Check g–r color evolution for changes across zero (indicating bumps).
- Generate and save diagnostic light-curve plots with fitted models.
- Save fit results and metadata to CSV files for downstream analysis.

Outputs:
- PNG plots of fitted light curves, saved to ``config.FLUX_PLOTS_DIR``.
- CSV file of fit results with statistics and parameters, saved to
  ``config.RAW_DATA/flux_fits_initial.csv``.

Dependencies:
- pandas, numpy, seaborn, matplotlib, scipy, requests
- config.py for API URL and file paths (``FINK_API_URL``, ``RAW_DATA``,
  ``FLUX_PLOTS_DIR``).
- core.py for helper functions (``convert_magpsf_to_flux``,
  ``fetch_valid_object_data_fink``).
"""

import requests
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as op
import sys, os
from scipy.stats import chi2
import csv
import math
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from core import convert_magpsf_to_flux, fetch_valid_object_data_fink
sns.set_context('talk')

def get_object_ids(start_date: str, end_date: str) -> np.array:
    """
    Query the Fink API for candidate SN Ia object IDs within a date range.

    Parameters:
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns:
    object_ids : np.array
        Unique object IDs of candidates returned by Fink.
    """
    r = requests.post(
        f'{config.FINK_API_URL}/api/v1/latests',
        json={
            'class': 'Early SN Ia candidate',
            'n': '50000',
            'color': 'True',
            'startdate': start_date,
            'stopdate': end_date,
            'columns': 'i:objectId, d:rf_snia_vs_nonia'
        
        }
    )

    pdf = pd.read_json(io.BytesIO(r.content))

    object_ids = pdf["i:objectId"].unique()

    return object_ids

def sigmoid(t, a, b, c):
    """
    Sigmoid model function for light-curve fitting.

    Parameters:
    t : array-like
        Time since first observation (days).
    a : float
        Amplitude parameter.
    b : float
        Midpoint (inflection point) parameter.
    c : float
        Growth rate parameter.

    Returns:
    flux : array-like
        Modeled flux values.
    """
    return a / (1 + np.exp(-c * (t - b)))

def power_law(t, a, b, c):
    """
    Power-law model function for light-curve fitting.

    Parameters:
    t : array-like
        Time since first observation (days).
    a : float
        Normalization parameter.
    b : float
        Time offset parameter.
    c : float
        Power-law index.

    Returns:
    flux : array-like
        Modeled flux values.
    """
    return a*(t-b)**c

def compute_chi_squared_and_reduced(time, magpsf, sigmapsf, func, params) -> Tuple[float, float]:
    """
    Compute chi-squared and reduced chi-squared for a fitted model.

    Parameters:
    time : array-like
        Time values of observations.
    magpsf : array-like
        Observed flux values (converted from magnitudes).
    sigmapsf : array-like
        Uncertainties on the observed flux values.
    func : callable
        Model function (e.g., sigmoid or power_law).
    params : list
        Best-fit model parameters.

    Returns:
    chi_squared : float
        Chi-squared statistic.
    reduced_chi_squared : float
        Chi-squared per degree of freedom.
    """
    chi_squared = np.sum(((magpsf - func(time, *params)) / sigmapsf)**2)
    dof = len(time) - len(params)
    reduced_chi_squared = chi_squared / dof
    return chi_squared, reduced_chi_squared

def chi2_p_val(chi_square_val, time, params) -> float:
    """
    Compute p-value for a chi-squared statistic.

    Parameters:
    chi_square_val : float
        Chi-squared statistic.
    time : array-like
        Time values of observations.
    params : list
        Best-fit model parameters.

    Returns:
    p_value : float
        Probability of obtaining a chi-squared at least this large.
    """
    dof = len(time) - len(params)
    return 1 - chi2.cdf(chi_square_val, dof)

def check_color_change(color_evolution) -> str:
    """
    Check for zero-crossings in g–r color evolution.

    Parameters:
    color_evolution : list
        Sequence of g–r color values (NaNs ignored).

    Returns:
    category : str
        One of {'none', 'single', 'bump'}, indicating the number of color
        changes. Here nump means two
    """
    color_evolution = [x for x in color_evolution if not math.isnan(x)]
    num_changes = 0
    for a, b in zip(color_evolution, color_evolution[1:]):
        if a * b <= 0 and not (a == 0 and b == 0):
            num_changes += 1

    if num_changes == 0: return 'none'
    elif num_changes == 1: return 'single'
    else: return 'bump'

def plot_lc_and_fit_sigmoid_power_and_save(object_ids, start_date, end_date) -> None:
    """
    Fit light curves for a list of objects with sigmoid and power-law models.

    Parameters:
    object_ids : list
        List of object IDs to process.
    start_date : str
        Start date for data fetching (YYYY-MM-DD).
    end_date : str
        End date for data fetching (YYYY-MM-DD).

    Returns:
    None
        Results and plots are saved to disk.
    """
    attribute_list = []
    for object_id in object_ids:
        try:
            result = process_single_object(object_id, start_date, end_date)
            if result:
                attribute_list.extend(result)
        except Exception as e:
            print(f"Error processing object {object_id}: {e}")

    save_fit_results(attribute_list, config.RAW_DATA / "flux_fits_initial.csv")


def process_single_object(object_id, start_date, end_date) -> Dict:
    """
    Fetch, preprocess, and fit light curves for a single object.

    Parameters:
    object_id : str
        Identifier of the object.
    start_date : str
        Start date for data fetching (YYYY-MM-DD).
    end_date : str
        End date for data fetching (YYYY-MM-DD).

    Returns:
    attributes : list of dict
        Fitted parameter dictionaries for each model and filter.
    """
    single_object = fetch_valid_object_data_fink(object_id, start_date, end_date)
    single_object = preprocess_object(single_object)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    attributes = []

    for i, model in enumerate(["sigmoid", "power"]):
        ax = axes[i]
        model_attributes = fit_and_plot_model(single_object, object_id, model, ax)
        attributes.extend(model_attributes)

    finalize_plot(fig, axes, object_id)
    return attributes


def preprocess_object(df) -> pd.DataFrame:
    """
    Preprocess object dataframe before fitting.

    - Replaces missing TNS classifications with 'nonSNIa'.
    - Converts g–r color to float with NaN for missing values.

    Parameters:
    df : pandas.DataFrame
        Raw object dataframe from Fink.

    Returns:
    df : pandas.DataFrame
        Preprocessed dataframe ready for fitting.
    """
    df['d:tns'] = df['d:tns'].replace('', 'nonSNIa').fillna('nonSNIa')
    df['v:g-r'] = df['v:g-r'].replace('', np.nan).astype(float)
    return df


def fit_and_plot_model(single_object, object_id, model, ax) -> List[Dict]:
    """
    Fit a light-curve model (sigmoid or power-law) for both filters.

    Parameters:
    single_object : pandas.DataFrame
        Preprocessed object data.
    object_id : str
        Identifier of the object.
    model : str
        Model name ('sigmoid' or 'power').
    ax : matplotlib.Axes
        Axis on which to plot fits.

    Returns:
    attributes : list of dict
        Fitted parameter dictionaries for each filter.
    """
    colordic = {1: 'C0', 2: 'C1'}
    filtdic = {1: 'g', 2: 'r'}
    attributes = []

    maskValid = single_object['d:tag'] == 'valid'
    min_time = single_object[maskValid]['i:jd'].min() - 2400000.5
    rise_time = (
        single_object[maskValid].loc[single_object[maskValid]['i:magpsf'].idxmin(), 'i:jd']
        - single_object[maskValid]['i:jd'].min()
    )

    for filt in [1, 2]:
        try:
            attr = fit_single_filter(
                single_object, object_id, model, filt, ax,
                min_time, rise_time, colordic, filtdic
            )
            attributes.append(attr)
        except Exception as e:
            print(f"Fit failed for object {object_id} filter {filtdic[filt]}: {e}")
            continue

    ax.set_title(f"{model.capitalize()} Fit")
    ax.set_xlabel("Time since first observation (JD)")
    ax.grid(True)
    ax.legend()
    return attributes


def fit_single_filter(single_object, object_id, model, filt, ax,
                      min_time, rise_time, colordic, filtdic) -> Dict:
    """
    Fit a light-curve model for a single filter.

    Parameters:
    single_object : pandas.DataFrame
        Preprocessed object data.
    object_id : str
        Identifier of the object.
    model : str
        Model name ('sigmoid' or 'power').
    filt : int
        Filter ID (1 = g, 2 = r).
    ax : matplotlib.Axes
        Axis on which to plot fits.
    min_time : float
        Time of first valid observation.
    rise_time : float
        Rise time to maximum light.
    colordic : dict
        Mapping of filter IDs to plot colors.
    filtdic : dict
        Mapping of filter IDs to filter names.

    Returns:
    attributes : dict
        Dictionary of fit results and metadata for this filter.
    """
                      
    maskValid = single_object['d:tag'] == 'valid'
    maskFilt = single_object['i:fid'] == filt
    maskBeforemax = single_object.index >= single_object[maskFilt & maskValid]['i:magpsf'].idxmin()
    mask = maskValid & maskFilt & maskBeforemax

    time = single_object[mask]["i:jd"].apply(lambda x: x - 2400000.5)
    magpsf = single_object[mask]["i:magpsf"]
    sigmapsf = single_object[mask]["i:sigmapsf"]

    flux, sigmaflux = convert_magpsf_to_flux(magpsf, sigmapsf)
    adjusted_time = time - min_time
    num_points = len(time)

    if model == "sigmoid":
        params, pcov, t_vals, fit_vals = fit_sigmoid(adjusted_time, flux, sigmaflux)
    else:
        params, pcov, t_vals, fit_vals = fit_power(adjusted_time, flux, sigmaflux)

    ax.errorbar(adjusted_time, flux*1e3, sigmaflux*1e3,
                ls="", marker="o", color=colordic[filt], label=filtdic[filt])
    ax.plot(t_vals, fit_vals, color=colordic[filt], linestyle="--")

    return build_attribute_dict(
        object_id, model, filt, filtdic, params, pcov,
        adjusted_time, flux, sigmaflux, num_points,
        rise_time, single_object
    )


def fit_sigmoid(time, flux, sigmaflux):
    """
    Fit a sigmoid model to flux data.

    Parameters:
    time : array-like
        Time since first observation (days).
    flux : array-like
        Flux values.
    sigmaflux : array-like
        Uncertainties on flux values.

    Returns:
    params : list
        Best-fit sigmoid parameters [a, b, c].
    pcov : 2D array
        Covariance matrix of fitted parameters.
    t_vals : array
        Dense time grid for plotting.
    fit_vals : array
        Model flux values on the dense grid.
    """
    a0 = flux.max()
    c0 = (flux.max() - flux.min()) / (time.max() - time.min())
    b0 = np.log(a0 / flux.min() - 1) / c0
    p0 = [a0, b0, c0]

    params, pcov = op.curve_fit(sigmoid, time, flux, sigma=sigmaflux, p0=p0, maxfev=10000)
    t_vals = np.linspace(0, time.max(), 1000)
    fit_vals = sigmoid(t_vals, *params) * 1e3
    return params, pcov, t_vals, fit_vals


def fit_power(time, flux, sigmaflux):
    """
    Fit a power-law model to flux data.

    Parameters:
    time : array-like
        Time since first observation (days).
    flux : array-like
        Flux values.
    sigmaflux : array-like
        Uncertainties on flux values.

    Returns:
    params : list
        Best-fit power-law parameters [a, b, c].
    pcov : 2D array
        Covariance matrix of fitted parameters.
    t_vals : array
        Dense time grid for plotting.
    fit_vals : array
        Model flux values on the dense grid.
    """
    a0 = (flux.max() - flux.min()) / (time.max() - time.min())
    p0 = [a0, -1, 0.7]
    bounds = [[-np.inf, -np.inf, 0], [np.inf, 0, 1]]

    params, pcov = op.curve_fit(power_law, time, flux, sigma=sigmaflux,
                                p0=p0, bounds=bounds, maxfev=10000)
    t_vals = np.linspace(0, time.max(), 1000)
    fit_vals = power_law(t_vals, *params) * 1e3
    return params, pcov, t_vals, fit_vals


def build_attribute_dict(object_id, model, filt, filtdic, params, pcov,
                         time, flux, sigmaflux, num_points, rise_time, df):
    """
    Construct a dictionary of fit results and metadata.

    Parameters:
    object_id : str
        Identifier of the object.
    model : str
        Model name ('sigmoid' or 'power').
    filt : int
        Filter ID.
    filtdic : dict
        Mapping of filter IDs to filter names.
    params : list
        Best-fit parameters.
    pcov : 2D array
        Covariance matrix of fitted parameters.
    time : array-like
        Time values used for fitting.
    flux : array-like
        Flux values used for fitting.
    sigmaflux : array-like
        Flux uncertainties used for fitting.
    num_points : int
        Number of data points fitted.
    rise_time : float
        Rise time to maximum light.
    df : pandas.DataFrame
        Object data.

    Returns:
    attributes : dict
        Dictionary of statistics, fitted parameters, and metadata.
    """
    fit_func = sigmoid if model == "sigmoid" else power_law
    chi_2_val, reduced_chi_2_val = compute_chi_squared_and_reduced(time, flux, sigmaflux, fit_func, params)
    p_val = chi2_p_val(chi_2_val, time, params)

    sig_params = np.sqrt(np.diag(pcov))
    return {
        "object id": object_id,
        "model": model,
        "filter": filtdic[filt],
        "chi2": chi_2_val,
        "reduced chi2": reduced_chi_2_val,
        "p-value": p_val,
        "a": params[0]*1e3,
        "b": params[1],
        "c": params[2],
        "sig a": sig_params[0],
        "sig b": sig_params[1],
        "sig c": sig_params[2],
        "num points": num_points,
        "TNS classified": (df['d:tns'] == 'SN Ia').any(),
        "color change": check_color_change(df.loc[time.index, 'v:g-r']),
        "snia confidence": max(df['d:rf_snia_vs_nonia']),
        "rise_time": rise_time
    }


def finalize_plot(fig, axes, object_id):
    """
    Finalize and save the plot of fitted light curves.

    Parameters:
    fig : matplotlib.Figure
        Figure object.
    axes : list of matplotlib.Axes
        Axes containing the plots.
    object_id : str
        Identifier of the object.

    Returns:
    None
        Plot is saved to file.
    """
    for ax in axes:
        ax.set_ylabel("Flux (mJy)")
    fig.suptitle(f"Object {object_id}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show(block=False)
    plt.savefig(config.FLUX_PLOTS_DIR / f"{object_id}_fit.png", dpi=300)
    plt.close()


def save_fit_results(attribute_list, filepath):
    """
    Save fit results to a CSV file.

    Parameters:
    attribute_list : list of dict
        List of dictionaries containing fit results.
    filepath : str or Path
        Output path for the CSV file.

    Returns:
    None
    """

    fieldnames = [
        "object id", "model", "filter", "chi2", "reduced chi2", "p-value",
        "a", "b", "c", "sig a", "sig b", "sig c",
        "num points", "TNS classified", "color change", "snia confidence", "rise_time"
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attribute_list)