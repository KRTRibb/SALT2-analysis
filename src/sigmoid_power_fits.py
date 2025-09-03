import requests
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as op
import os
from scipy.stats import chi2
import csv
import math

import config
from core import convert_magpsf_to_flux, fetch_valid_object_data_fink
sns.set_context('talk')

def get_object_ids(start_date: str, end_date: str) -> np.array:
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
    return a / (1 + np.exp(-c * (t - b)))

def power_law(t, a, b, c):
    return a*(t-b)**c

def compute_chi_squared_and_reduced(time, magpsf, sigmapsf, func, params):
    chi_squared = np.sum(((magpsf - func(time, *params)) / sigmapsf)**2)
    dof = len(time) - len(params)
    reduced_chi_squared = chi_squared / dof
    return chi_squared, reduced_chi_squared

def chi2_p_val(chi_square_val, time, params):
    dof = len(time) - len(params)
    return 1 - chi2.cdf(chi_square_val, dof)

def check_color_change(color_evolution):
    color_evolution = [x for x in color_evolution if not math.isnan(x)]
    num_changes = 0
    for a, b in zip(color_evolution, color_evolution[1:]):
        if a * b <= 0 and not (a == 0 and b == 0):
            num_changes += 1

    if num_changes == 0: return 'none'
    elif num_changes == 1: return 'single'
    else: return 'bump'

def plot_lc_and_fit_sigmoid_power_and_save(object_ids, start_date, end_date):
    attribute_list = []
    for object_id in object_ids:
        try:
            result = process_single_object(object_id, start_date, end_date)
            if result:
                attribute_list.extend(result)
        except Exception as e:
            print(f"Error processing object {object_id}: {e}")

    save_fit_results(attribute_list, config.RAW_DATA / "flux_fits_initial.csv")


def process_single_object(object_id, start_date, end_date):
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


def preprocess_object(df):
    df['d:tns'] = df['d:tns'].replace('', 'nonSNIa').fillna('nonSNIa')
    df['v:g-r'] = df['v:g-r'].replace('', np.nan).astype(float)
    return df


def fit_and_plot_model(single_object, object_id, model, ax):
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
                      min_time, rise_time, colordic, filtdic):
                      
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
    a0 = flux.max()
    c0 = (flux.max() - flux.min()) / (time.max() - time.min())
    b0 = np.log(a0 / flux.min() - 1) / c0
    p0 = [a0, b0, c0]

    params, pcov = op.curve_fit(sigmoid, time, flux, sigma=sigmaflux, p0=p0, maxfev=10000)
    t_vals = np.linspace(0, time.max(), 1000)
    fit_vals = sigmoid(t_vals, *params) * 1e3
    return params, pcov, t_vals, fit_vals


def fit_power(time, flux, sigmaflux):
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
        "rise time": rise_time
    }


def finalize_plot(fig, axes, object_id):
    for ax in axes:
        ax.set_ylabel("Flux (mJy)")
    fig.suptitle(f"Object {object_id}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show(block=False)
    plt.savefig(config.FLUX_PLOTS_DIR / f"{object_id}_fit.png", dpi=300)
    plt.close()


def save_fit_results(attribute_list, filepath):
    fieldnames = [
        "object id", "model", "filter", "chi2", "reduced chi2", "p-value",
        "a", "b", "c", "sig a", "sig b", "sig c",
        "num points", "TNS classified", "color change", "snia confidence", "rise time"
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attribute_list)