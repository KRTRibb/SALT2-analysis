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
from core import convert_magpsf_to_flux
sns.set_context('talk')

def get_object_ids(start_date: str, end_date: str) -> np.array:
    r = requests.post(
        f'{config.APIURL}/api/v1/latests',
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
            r = requests.post(
                f'{config.APIURL}/api/v1/objects',
                json={
                    'objectId': object_id,
                    'output-format': 'json',
                    'withupperlim': 'True',
                    'startdate': start_date,
                    'stopdate': end_date,
                }
            )
            single_object = pd.read_json(io.BytesIO(r.content))
            single_object['d:tns'] = single_object['d:tns'].replace('', 'nonSNIa').fillna('nonSNIa')
            tns_classification = (single_object['d:tns'] == 'SN Ia').any()
            maskValid = single_object['d:tag'] == 'valid'
            maskBeforemax_combined = single_object.index >= single_object[maskValid]['i:magpsf'].idxmin()
            single_object['v:g-r'] = single_object['v:g-r'].replace('', np.nan).astype(float) 
            color_change = check_color_change(single_object.loc[maskBeforemax_combined, 'v:g-r'])

            rf_snia_confidence = max(single_object['d:rf_snia_vs_nonia'])

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            colordic = {1: 'C0', 2: 'C1'}
            filtdic = {1: 'g', 2: 'r'}

            maskValid = single_object['d:tag'] == 'valid'
            min_time = single_object[maskValid]['i:jd'].min() - 2400000.5
            rise_time = single_object[maskValid].loc[single_object[maskValid]['i:magpsf'].idxmin(), 'i:jd'] - single_object[maskValid]['i:jd'].min()

            for i, model in enumerate(['sigmoid', 'power']):
                ax = axes[i]
                for filt in [1, 2]:
                    maskFilt = single_object['i:fid'] == filt
                    maskValid = single_object['d:tag'] == 'valid'
                    maskBeforemax = single_object.index >= single_object[maskFilt & maskValid]['i:magpsf'].idxmin()

                    mask = maskValid & maskFilt & maskBeforemax
                    time = single_object[mask]["i:jd"].apply(lambda x: x - 2400000.5)
                    magpsf = single_object[mask]["i:magpsf"]
                    sigmapsf = single_object[mask]["i:sigmapsf"]

                    flux, sigmaflux = convert_magpsf_to_flux(magpsf, sigmapsf)
                    adjusted_time = time - min_time

                    num_points = len(time)

                    ax.errorbar(adjusted_time, flux*10**3, sigmaflux*10**3, ls="", marker="o", color=colordic[filt], label=filtdic[filt])

                    try:
                        if model == 'sigmoid':
                            a0 = flux.max()
                            c0 = (flux.max() - flux.min()) / (adjusted_time.max() - adjusted_time.min())
                            b0 = np.log(a0 / (flux.min()) - 1) / c0
                            p0 = [a0, b0, c0]
                            params, pcov = op.curve_fit(
                                sigmoid, adjusted_time, flux, sigma=sigmaflux, p0=p0, maxfev=10000
                            )
                            t_vals = np.linspace(0, adjusted_time.max(), 1000)
                            a, b, c = params
                            sig_a, sig_b, sig_c = np.sqrt(np.diag(pcov))
                            fit_vals = sigmoid(t_vals, a, b, c) * 10**3
                            a *= 10**3 # Adjust for correction from J to mJ
                            
                            chi_2_val, reduced_chi_2_val = compute_chi_squared_and_reduced(adjusted_time, flux, sigmaflux, sigmoid, params)
                            p_val = chi2_p_val(chi_2_val, time, params)

                            label = fr"{filtdic[filt]}: $y = \frac{{{params[0]*1E3:.2f}}}{{1 + e^{{-{params[2]:.2f} (t - {params[1]:.2f})}}}}$"

                            attribute_list.append(
                                {
                                "object id": object_id,
                                "model": model,
                                "filter": filtdic[filt],
                                "chi2": chi_2_val,
                                "reduced chi2": reduced_chi_2_val,
                                "p-value": p_val,
                                "a": a,
                                "b": b,
                                "c": c,
                                "sig a": sig_a,
                                "sig b": sig_b,
                                "sig c": sig_c,
                                "num points": num_points,
                                "TNS classified": tns_classification,
                                "color change": color_change,
                                "snia confidence": rf_snia_confidence,
                                "rise time": rise_time
                                }
                            )

                        else:
                            a0 = (flux.max() - flux.min()) / (adjusted_time.max() - adjusted_time.min())
                            b0 = -1
                            c0 = 0.7
                            p0 = [a0, b0, c0]
                            bounds = [[-np.inf, -np.inf, 0], [np.inf, 0, 1]]
                            params, pcov = op.curve_fit(
                                power_law, adjusted_time, flux, sigma=sigmaflux, p0=p0, maxfev=10000, bounds=bounds
                            )
                            t_vals = np.linspace(0, adjusted_time.max(), 1000)
                            a, b, c = params
                            sig_a, sig_b, sig_c = np.sqrt(np.diag(pcov))
                            fit_vals = power_law(t_vals, a, b, c) * 10**3
                            a *= 10**3 # Adjust for correction from J to mJ

                            chi_2_val, reduced_chi_2_val = compute_chi_squared_and_reduced(adjusted_time, flux, sigmaflux, power_law, params)
                            p_val = chi2_p_val(chi_2_val, time, params)

                            label = fr"{filtdic[filt]}: $y = {params[0]*1E3:.2f}(t {params[1]:.2f})^{{{params[2]:.2f}}}$"

                            attribute_list.append(
                                {
                                "object id": object_id,
                                "model": model,
                                "filter": filtdic[filt],
                                "chi2": chi_2_val,
                                "reduced chi2": reduced_chi_2_val,
                                "p-value": p_val,
                                "a": a,
                                "b": b,
                                "c": c,
                                "sig a": sig_a,
                                "sig b": sig_b,
                                "sig c": sig_c,
                                "num points": num_points,
                                "TNS classified": tns_classification,
                                "color change": color_change,
                                "snia confidence": rf_snia_confidence,
                                "rise time": rise_time
                                }
                            )
                        ax.plot(t_vals, fit_vals, color=colordic[filt], linestyle='--', label=label)
                    except Exception as e:
                        print(f"Fit failed for object {object_id} filter {filtdic[filt]}: {e}")
                        continue

                ax.set_title(f"{model.capitalize()} Fit")
                ax.set_xlabel("time")
                ax.grid(True)
                ax.legend()

            axes[0].set_ylabel("Flux (mJy)")
            axes[0].set_xlabel("Time since first observation (JD)")
            axes[1].set_ylabel("Flux (mJy)")
            axes[1].set_xlabel("Time since first observation (JD)")
            fig.suptitle(f"Object {object_id}", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            plt.show(block=False)
            plt.savefig(config.FLUX_PLOTS_DIR / f"{object_id}_fit.png", dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error processing object {object_id}: {e}")

    with open(config.RAW_DATA_DIR / "flux_fits_initial.csv", 'w', newline='') as f:
        fieldnames = [
            "object id", "model", "filter", "chi2", "reduced chi2", "p-value", "a", "b", "c", 
            "sig a", "sig b", "sig c", "num points", "TNS classified", "color change", "snia confidence", "rise time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attribute_list)
