import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = 'notebook'
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import linregress, t, norm, laplace, gennorm, skewnorm
from sklearn.mixture import GaussianMixture


def remove_outside_range(params, lower_bound, upper_bound, parameter):
    col_map = {"z": 0, "x1": 1, "c": 2, "sig x1": 3, "sig c": 4}
    
    array = np.array(params, dtype=float)
    col_idx = col_map[parameter]
    
    mask = (array[:, col_idx] >= lower_bound) & (array[:, col_idx] <= upper_bound)
    return array[mask]

# data format [data1, data2, [names], title]
def plot_sncosmo_param_hist(data, width, height, nrows=1, ncols=5, 
                            share_y=False, data_groups=False, 
                            colors=('C0', 'C1'), alpha=0.7, linewidth=2, 
                            show_stats=True):
                            
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height), sharey=share_y)
    axs = np.array(axs).reshape(-1)

    if not data_groups:
        for i, (data, name, xlog, ylog, xlim) in enumerate(data):

            if xlim is not None:
                axs[i].set_xlim((xlim))

            if ylog:
                axs[i].set_yscale('log')
            
            if xlog:
                data_min = np.min(data)
                data_max = np.max(data)
    
                axs[i].set_xscale("log")
                bins = np.logspace(np.log10(data_min), np.log10(data_max), num=50)
                
            else:
                q25, q75 = np.percentile(data, [25, 75])
                iqr = q75 - q25
                bin_width = 2 * iqr / (len(data) ** (1/3))
                bin_width = bin_width if bin_width > 0 else 1e-3

                data_min, data_max = np.min(data), np.max(data)
                bins = np.arange(data_min, data_max + bin_width, bin_width)

            axs[i].hist(data, bins=bins, histtype='step', color=colors[0], 
                        alpha=alpha, linewidth=linewidth, label=name)

            if show_stats:
                median = np.median(data)
                mean = np.mean(data)
                stdev = np.std(data)
                axs[i].set_xlabel(f"\nmed = {median:.2e}, mean = {mean:.2e}, \nstd = {stdev:.2e}")

            axs[i].set_title(name)
            axs[i].legend()

    else:
        for i, (data1, data2, names, title, xlog, ylog, xlim) in enumerate(data):
            
            if xlim is not None:
                axs[i].set_xlim((xlim))
                    
            if ylog:
                axs[i].set_yscale('log')

            if xlog:
                data_min = min(np.min(data1), np.min(data2))
                data_max = max(np.max(data1), np.max(data2))
    
                axs[i].set_xscale("log")
                bins = np.logspace(np.log10(data_min), np.log10(data_max), num=50)
            else:
                combined = np.concatenate((data1, data2))
                q25, q75 = np.percentile(combined, [25, 75])
                iqr = q75 - q25
                bin_width = 2 * iqr / (len(combined) ** (1/3))
                bin_width = bin_width if bin_width > 0 else 1e-3

                data_min, data_max = np.min(combined), np.max(combined)
                bins = np.arange(data_min, data_max + bin_width, bin_width)

            axs[i].hist(data1, bins=bins, histtype='step', color=colors[0], alpha=alpha, linewidth=linewidth, label=names[0])
            axs[i].hist(data2, bins=bins, histtype='step', color=colors[1], alpha=alpha, linewidth=linewidth, label=names[1])

            if show_stats:
                median1 = np.median(data1)
                mean1 = np.mean(data1)
                stdev1 = np.std(data1)
                median2 = np.median(data2)
                mean2 = np.mean(data2)
                stdev2 = np.std(data2)
                axs[i].set_xlabel(f"\n{names[0]}:\n med={median1:.2e}, mean={mean1:.2e},  std={stdev1:.2e}\n{names[1]}:\n med={median2:.2e}, mean={mean2:.2e},  std={stdev2:.2e}")

            axs[i].set_title(title)
            axs[i].legend()
    plt.tight_layout()
    plt.show()

def plot_param_hist_with_models(data, width, height, nrows=1, ncols=5, 
                    share_y=False, colors=('C0', 'C1'), alpha=0.7, linewidth=2, 
                    show_stats=True):
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height), sharey=share_y)
    axs = np.array(axs).reshape(-1)

    for i, (data_vals, name, xlog, ylog, dist_type) in enumerate(data):
        data_vals = np.array(data_vals)

        if ylog:
                axs[i].set_yscale('log')

        if xlog:
            data_min, data_max = np.min(data_vals), np.max(data_vals)
            axs[i].set_xscale("log")
            bins = np.logspace(np.log10(data_min), np.log10(data_max), num=50)
        else:
            q25, q75 = np.percentile(data_vals, [25, 75])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(data_vals) ** (1/3))
            bin_width = bin_width if bin_width > 0 else 1e-3
            data_min, data_max = np.min(data_vals), np.max(data_vals)
            bins = np.arange(data_min, data_max + bin_width, bin_width)

        axs[i].hist(data_vals, bins=bins, histtype='step', color=colors[0], 
                    alpha=alpha, linewidth=linewidth, label=name, density=True)

        x = np.linspace(data_min, data_max, 500)
        pdf = None

        if dist_type == "skewed":
            a, loc, scale = skewnorm.fit(data_vals)
            pdf = skewnorm.pdf(x, a, loc, scale)

        elif dist_type == "gennorm":
            beta, loc, scale = gennorm.fit(data_vals)
            pdf = gennorm.pdf(x, beta, loc, scale)

        elif dist_type == "laplace":
            loc, scale = laplace.fit(data_vals)
            pdf = laplace.pdf(x, loc, scale)

        elif dist_type == "t":
            df, loc, scale = t.fit(data_vals)
            pdf = t.pdf(x, df, loc, scale)

        elif dist_type == "2 sum":
            data_array = data_vals.reshape(-1, 1)
            gm = GaussianMixture(n_components=2)
            gm.fit(data_array)
            x_array = x.reshape(-1, 1)
            pdf = np.exp(gm.score_samples(x_array))

        if pdf is not None:
            axs[i].plot(x, pdf, 'r', lw=2, label=f"{dist_type} fit")

        if show_stats:
            median, mean, stdev = np.median(data_vals), np.mean(data_vals), np.std(data_vals)
            axs[i].set_xlabel(f"\nmed = {median:.2e}, mean = {mean:.2e}, \nstd = {stdev:.2e}")

        axs[i].set_title(name)
        axs[i].legend()

    plt.tight_layout()
    plt.show()

def remove_outliers(params, lower_percentile=0, upper_percentile=95):
    array = np.array(params, dtype=float) 

    lower_bounds = np.percentile(array, lower_percentile, axis=0)
    upper_bounds = np.percentile(array, upper_percentile, axis=0)

    mask = (array >= lower_bounds) & (array <= upper_bounds)
    mask = np.all(mask, axis=1)

    return array[mask]

def merge_by_object_id(
    sncosmo_df, fluxfit_df, 
    sncosmo_col, fluxfit_col, include_errors=False, 
    sncosmo_mask=None, fluxfit_mask=None
):
    if sncosmo_mask is not None:
        sncosmo_df = sncosmo_df[sncosmo_mask]
    if fluxfit_mask is not None:
        fluxfit_df = fluxfit_df[fluxfit_mask]

    sncosmo_cols = ['object id', sncosmo_col]
    fluxfit_cols = ['object id', fluxfit_col]

    if include_errors:
        sncosmo_err_col = f"sig {sncosmo_col}"
        fluxfit_err_col = f"sig {fluxfit_col}"

        if sncosmo_err_col in sncosmo_df.columns:
            sncosmo_cols.append(sncosmo_err_col)
        if fluxfit_err_col in fluxfit_df.columns:
            fluxfit_cols.append(fluxfit_err_col)

    sncosmo_sel = sncosmo_df[sncosmo_cols].rename(
        columns={sncosmo_col: f"sncosmo_{sncosmo_col}"}
    )
    fluxfit_sel = fluxfit_df[fluxfit_cols].rename(
        columns={fluxfit_col: f"fluxfit_{fluxfit_col}"}
    )

    if include_errors:
        if f"sig {sncosmo_col}" in sncosmo_cols:
            sncosmo_sel = sncosmo_sel.rename(columns={f"sig {sncosmo_col}": f"sncosmo_sig_{sncosmo_col}"})
        if f"sig {fluxfit_col}" in fluxfit_cols:
            fluxfit_sel = fluxfit_sel.rename(columns={f"sig {fluxfit_col}": f"fluxfit_sig_{fluxfit_col}"})

    merged_df = pd.merge(sncosmo_sel, fluxfit_sel, on="object id", how="inner")

    return merged_df

def get_bump_nb_pairs(sncosmo_df, fluxfit_df, sncosmo_col, fluxfit_col, model_name, include_errors=False, extra_fluxfitmask=None, extra_sncosmomask=None):
    # g band
    mask_bump_g = (
        (fluxfit_df["color change"] == "bump") &
        (fluxfit_df["model"] == model_name) &
        (fluxfit_df["filter"] == "g")
    )

    if extra_fluxfitmask is not None:
        fluxfit_mask = mask_bump_g & extra_fluxfitmask
    else:
        fluxfit_mask = mask_bump_g

    bump_g = merge_by_object_id(sncosmo_df, fluxfit_df, sncosmo_col, fluxfit_col, fluxfit_mask=fluxfit_mask, include_errors=include_errors, sncosmo_mask=extra_sncosmomask)

    mask_nb_g = (
        (fluxfit_df["color change"] != "bump") &
        (fluxfit_df["model"] == model_name) &
        (fluxfit_df["filter"] == "g")
    )

    if extra_fluxfitmask is not None:
        fluxfit_mask = mask_nb_g & extra_fluxfitmask
    else:
        fluxfit_mask = mask_nb_g

    nb_g = merge_by_object_id(sncosmo_df, fluxfit_df, sncosmo_col, fluxfit_col, fluxfit_mask=fluxfit_mask, include_errors=include_errors, sncosmo_mask=extra_sncosmomask)

    # r band
    mask_bump_r = (
        (fluxfit_df["color change"] == "bump") &
        (fluxfit_df["model"] == model_name) &
        (fluxfit_df["filter"] == "r")
    )

    if extra_fluxfitmask is not None:
        fluxfit_mask = mask_bump_r & extra_fluxfitmask
    else:
        fluxfit_mask = mask_bump_r

    bump_r = merge_by_object_id(sncosmo_df, fluxfit_df, sncosmo_col, fluxfit_col, fluxfit_mask=fluxfit_mask, include_errors=include_errors, sncosmo_mask=extra_sncosmomask)

    mask_nb_r = (
        (fluxfit_df["color change"] != "bump") &
        (fluxfit_df["model"] == model_name) &
        (fluxfit_df["filter"] == "r")
    )

    if extra_fluxfitmask is not None:
        fluxfit_mask = mask_nb_r & extra_fluxfitmask
    else:
        fluxfit_mask = mask_nb_r

    nb_r = merge_by_object_id(sncosmo_df, fluxfit_df, sncosmo_col, fluxfit_col, fluxfit_mask=fluxfit_mask, include_errors=include_errors, sncosmo_mask=extra_sncosmomask)

    return bump_g, nb_g, bump_r, nb_r

# data format[[set1, set2, label], ...]
def scatterplot_2x4_pairs(
    data1_list, data2_list, xlabel, ylabel, titles=('Plot 1', 'Plot 2'), 
    width=18, height=6, xlims=(None, None), ylims=(None, None)): 

    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    colors = ['red', 'orange', 'blue', 'green']
    
    for ax, dataset, title, xlim, ylim in zip(axes, [data1_list, data2_list], titles, xlims, ylims):
        for i, (x, y, label) in enumerate(dataset):
            ax.scatter(x, y, color=colors[i % len(colors)], label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
    
    plt.tight_layout()
    plt.show()

def scatterplot_2x4_pairs_interactive(
    data1_list, data2_list, object_ids_list1, object_ids_list2,
    xlabel, ylabel, titles=('Plot 1', 'Plot 2'),
    xlims=(None, None), ylims=(None, None),
    colors=['red', 'orange', 'blue', 'green']
):
    fig = make_subplots(rows=1, cols=2, subplot_titles=titles)
    
    for i, ((x, y, label), obj_ids) in enumerate(zip(data1_list, object_ids_list1)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(color=colors[i % len(colors)]),
                name=label,
                text=obj_ids,
                hovertemplate='object id: %{text}<br>' + xlabel + ': %{x}<br>' + ylabel + ': %{y}<extra></extra>'
            ),
            row=1, col=1
        )
    
    for i, ((x, y, label), obj_ids) in enumerate(zip(data2_list, object_ids_list2)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                marker=dict(color=colors[i % len(colors)]),
                name=label,
                text=obj_ids,
                hovertemplate='object id: %{text}<br>' + xlabel + ': %{x}<br>' + ylabel + ': %{y}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text=xlabel, range=xlims[0], row=1, col=1)
    fig.update_yaxes(title_text=ylabel, range=ylims[0], row=1, col=1)
    fig.update_xaxes(title_text=xlabel, range=xlims[1], row=1, col=2)
    fig.update_yaxes(title_text=ylabel, range=ylims[1], row=1, col=2)
    
    fig.update_layout(width=1200, height=500, legend=dict(itemsizing='constant'))
    fig.show()

def add_regression_line(x, y, row, col, color, label, fig):
        result = linregress(x, y)
        slope, intercept, r_value, slope_err, intercept_err = result.slope, result.intercept, result.rvalue, result.stderr, result.intercept_stderr
        R2 = r_value**2
        N = len(x)
        t_val = r_value * np.sqrt((N-2)/(1-R2)) if R2 < 1 else np.inf
        p_value = 2 * t.sf(abs(t_val), N-2)
        signifigance = norm.isf(p_value / 2)
        
        print(f"{label}: slope={slope:.4f}, intercept={intercept:.4f}, R²={R2:.4f}, p={p_value:.4e}, t={t_val:.4f}, N={N}, sig={signifigance:.2f}σ\n∆slope = {slope_err:.4e}, ∆intercept={intercept_err:.4e}")
        x_fit = np.array([min(x), max(x)])
        y_fit = intercept + slope * x_fit
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode='lines',
                line=dict(color=color, dash='dash'),
                name=label,
                showlegend=True
            ),
            row=row, col=col
        )
        
def scatterplot_2x2_pairs_interactive(
    sig_data, pow_data, sig_obj_ids, pow_obj_ids,
    xlabel, ylabel, colors=['blue', 'orange']
):
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Sigmoid g-band", "Power law g-band",
            "Sigmoid r-band", "Power law r-band"
        ]
    )
    
    for row_idx, (sig_pair, pow_pair, sig_ids_pair, pow_ids_pair) in enumerate(
        zip(sig_data, pow_data, sig_obj_ids, pow_obj_ids), start=1
    ):
        for i, ((y, x, label), obj_ids) in enumerate(zip(sig_pair, sig_ids_pair)):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(color=color),
                    name=label,
                    text=obj_ids,
                    hovertemplate='object id: %{text}<br>' + xlabel + ': %{x}<br>' + ylabel + ': %{y}<extra></extra>'
                ),
                row=row_idx, col=1
            )
            add_regression_line(x, y, row_idx, 1, color, label, fig)
        
        for i, ((y, x, label), obj_ids) in enumerate(zip(pow_pair, pow_ids_pair)):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(color=color),
                    name=label,
                    text=obj_ids,
                    hovertemplate='object id: %{text}<br>' + xlabel + ': %{x}<br>' + ylabel + ': %{y}<extra></extra>'
                ),
                row=row_idx, col=2
            )
            add_regression_line(x, y, row_idx, 2, color, label, fig)
    
    for r in [1,2]:
        for c in [1,2]:
            fig.update_xaxes(title_text=xlabel, row=r, col=c)
            fig.update_yaxes(title_text=ylabel, row=r, col=c)
    
    fig.update_layout(width=1200, height=800, legend=dict(itemsizing='constant'))
    fig.show()


def remove_outliers_with_object_ids(x, y, obj_ids, lower_percentile=10, upper_percentile=90):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    obj_ids = np.array(obj_ids)

    array = np.column_stack((x, y))

    lower_bounds = np.percentile(array, lower_percentile, axis=0)
    upper_bounds = np.percentile(array, upper_percentile, axis=0)

    mask = (array >= lower_bounds) & (array <= upper_bounds)
    mask = np.all(mask, axis=1)

    return x[mask], y[mask], obj_ids[mask]

def clean_triplets(arr):
    mask = np.isfinite(arr).all(axis=1)
    cleaned = arr[mask]
    if cleaned.size == 0:
        return np.array([]), np.array([]), np.array([])
    return cleaned[:, 0], cleaned[:, 1], cleaned[:, 2]