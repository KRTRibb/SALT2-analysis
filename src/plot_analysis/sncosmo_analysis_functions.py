"""
sncosmo_analysis_functions.py

Functions for analyzing SALT2 and flux-fit data of early Type Ia candidate light curves.

This module provides tools to:
- Load and merge flux-fit and SALT2 results.
- Compute derived features such as rise time, slope, curvature, and early g-r color.
- Assign Ia/non-Ia grouping based on TNS classifications.
- Plot joint distributions of features vs. SALT2 parameters with KDE and scatter plots.
- Compute energy distances between groups with permutation testing.
- Automate a full analysis workflow that generates plots and CSV summaries.

Outputs:
- KDE plots saved to configurable output directories.
- CSV summaries of plots and permutation-test results.
- CSV of merged data with derived features.

Dependencies:
- pandas, numpy, seaborn, matplotlib, scipy, joblib
- config.py for file paths and output directories.
"""


import os, sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def load_flux_and_sncosmo(
    flux_path: str,
    sncosmo_path: str,
    stratify_col: str
) -> pd.DataFrame:
    """
    Load flux-fit and SALT2 results, merge into a single DataFrame, and assign Ia/non-Ia groups.

    Parameters:
    flux_path : str
        Path to CSV file with flux-fit parameters.
    sncosmo_path : str
        Path to CSV file with SALT2 fit results.
    stratify_col : str, optional
        Column used for stratification/filtering (must be "TNS classified" or "color change").

    Returns:
    df : pd.DataFrame
        Merged DataFrame with derived metadata and 'tns_group' column ('Ia' or 'non-Ia').
    """
    flux_fits_pdf = pd.read_csv(flux_path)
    sncosmo_pdf = pd.read_csv(sncosmo_path)

    param_cols = ['a', 'b', 'c', 'sig a', 'sig b', 'sig c',
                  'reduced chi2', 'p-value', 'num points']
    flux_fits_pdf['model_filter'] = (
        flux_fits_pdf['model'] + '_' + flux_fits_pdf['filter']
    )

    df_melted = flux_fits_pdf.melt(
        id_vars=['object id', 'model_filter',
                 'TNS classified', 'color change', 'snia confidence'],
        value_vars=param_cols,
        var_name='parameter',
        value_name='value'
    )

    flux_wide = df_melted.pivot_table(
        index='object id',
        columns=['model_filter', 'parameter'],
        values='value'
    )

    flux_wide.columns = ['_'.join(col).strip() for col in flux_wide.columns.values]
    flux_wide.reset_index(inplace=True)

    metadata_cols = ['object id', 'TNS classified', 'color change', 'snia confidence']
    metadata = flux_fits_pdf[metadata_cols].drop_duplicates(subset='object id')
    df = flux_wide.merge(metadata, on='object id', how='left')

    df = df.merge(
        sncosmo_pdf[['object id', 't0', 'x1', 'c', 'z', 'x0', 'sig c', 'sig x1']],
        on='object id',
        how='left'
    )

    ia_types = {'SN Ia', 'SN Ia-91T-like', 'SN Iax[02cx-like]', 'SN Ia-pec'}
    df['tns_group'] = np.where(
        df['TNS classified'].isin(ia_types), 'Ia', 'non-Ia'
    )

    if stratify_col == "TNS classified":
        df = df[df['TNS classified'].notna() & (df['TNS classified'].str.strip() != '')].copy()

    return df

def add_rise_slope_curvature_color(df: pd.DataFrame, t_offset: int = 3) -> pd.DataFrame:
    """
    Compute derived features from flux-fit parameters: rise time, slope, curvature,
    and early g-r color at specified phases.

    Parameters:
    df : pd.DataFrame
        Input DataFrame with flux-fit and SALT2 parameters.
    t_offset : int, optional
        Time offset for slope/curvature calculations of power-law fits (default: 3).

    Returns:
    df : pd.DataFrame
        DataFrame with additional derived features.
    """
    df = df.copy()

    for band in ['g','r']:
        df[f"sigmoid_{band}_slope"] = np.where(
            df[f"sigmoid_{band}_a"].notna() & df[f"sigmoid_{band}_c"].notna(),
        df[f"sigmoid_{band}_a"] * df[f"sigmoid_{band}_c"] / 4,
            np.nan
        )
        mask = df[f"power_{band}_a"].notna() & df[f"power_{band}_c"].notna()
        df.loc[mask, f"power_{band}_slope"] = df.loc[mask, f"power_{band}_a"] * df.loc[mask, f"power_{band}_c"] * t_offset**(df.loc[mask, f"power_{band}_c"]-1)

    for band in ['g','r']:
        df[f"sigmoid_{band}_curvature"] = np.where(
            df[f"sigmoid_{band}_a"].notna() & df[f"sigmoid_{band}_c"].notna(),
            df[f"sigmoid_{band}_a"] * df[f"sigmoid_{band}_c"]**3 / 4,
            np.nan
        )
        mask = df[f"power_{band}_a"].notna() & df[f"power_{band}_c"].notna()
        df.loc[mask, f"power_{band}_curvature"] = (
            df.loc[mask, f"power_{band}_a"] *
            df.loc[mask, f"power_{band}_c"] *
            (df.loc[mask, f"power_{band}_c"]-1) *
            t_offset**(df.loc[mask, f"power_{band}_c"]-2)
        )   

    if "TNS classified" in df.columns:
        df["tns_group"] = df["TNS classified"].apply(
            lambda x: "Ia" if isinstance(x, str) and "Ia" in x else "non-Ia"
        )

    for phase in config.PHASES:
        t_sample = df['t0'] + phase
        df[f"sigmoid_gr_mag_{phase}"] = -2.5 * np.log10(
            (cleaned_sigmoid_flux(df['sigmoid_g_a'], df['sigmoid_g_b'], df['sigmoid_g_c'], t_sample)+25) /
            (cleaned_sigmoid_flux(df['sigmoid_r_a'], df['sigmoid_r_b'], df['sigmoid_r_c'], t_sample)+25)
        )
        df[f"power_gr_mag_{phase}"] = -2.5 * np.log10(
            (cleaned_power_flux(df['power_g_a'], df['power_g_b'], df['power_g_c'], t_sample)+25) /
            (cleaned_power_flux(df['power_r_a'], df['power_r_b'], df['power_r_c'], t_sample)+25)
        )

    return df

def cleaned_sigmoid_flux(a,b,c,t):
    """
    Safely compute flux from sigmoid fit parameters, returning NaN if inputs are missing.
    """
    return np.where(a.notna() & b.notna() & c.notna(), a / (1 + np.exp(-c*(t-b))), np.nan)

def cleaned_power_flux(a,b,c,t):
    """
    Safely compute flux from power-law fit parameters, returning NaN if inputs are missing
    or if t-b <= 0.
    """
    return np.where(a.notna() & b.notna() & c.notna() & (t-b>0), a * (t-b)**c, np.nan)

def freedman_diaconis_bins(data: np.ndarray) -> int:
    """Compute number of bins using Freedman-Diaconis rule."""
    data = np.asarray(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr * (len(data) ** (-1/3))
    if bin_width <= 0:
        return 10
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(1, bins)

def plot_joint_distribution(df: pd.DataFrame, x_col: str, y_col: str, stratify_col: str,
    save_path: str = None, groups: list = None,lower_percentile: float = 0.1,upper_percentile: float = 0.9):
    """
    Plot joint distribution (2D KDE + scatter + marginal histograms) of two features,
    stratified by a column.
    """

    if groups is None:
        groups = sorted(df[stratify_col].dropna().unique().tolist())
    if "combined" not in groups:
        groups.append("combined")

    n_groups = len(groups)
    ncols = 2 if n_groups <= 3 else 3
    nrows = int(np.ceil(n_groups / ncols))

    fig = plt.figure(figsize=(6 * ncols, 6 * nrows))

    x_lower, x_upper = df[x_col].quantile([lower_percentile, upper_percentile])
    y_lower, y_upper = df[y_col].quantile([lower_percentile, upper_percentile])

    for i, group in enumerate(groups):
        row, col = divmod(i, ncols)
        outer_gs = GridSpec(nrows, ncols, figure=fig)[row, col]
        inner_gs = GridSpec(4, 4, figure=fig,
                            left=outer_gs.get_position(fig).xmin,
                            right=outer_gs.get_position(fig).xmax,
                            bottom=outer_gs.get_position(fig).ymin,
                            top=outer_gs.get_position(fig).ymax,
                            hspace=0.05, wspace=0.05)

        ax_main = fig.add_subplot(inner_gs[1:, :-1])
        ax_xdist = fig.add_subplot(inner_gs[0, :-1], sharex=ax_main)
        ax_ydist = fig.add_subplot(inner_gs[1:, -1], sharey=ax_main)

        subset = df.copy() if group == "combined" else df[df[stratify_col] == group]
        mask = subset[x_col].between(x_lower, x_upper) & subset[y_col].between(y_lower, y_upper)
        subset = subset.loc[mask, [x_col, y_col]].dropna()

        if len(subset) > 1:
            x = subset[x_col].to_numpy()
            y = subset[y_col].to_numpy()

            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            ax_main.contourf(xx, yy, zz, levels=20, cmap="Blues")
            ax_main.scatter(x, y, s=10, color="red", alpha=0.6)

            xbins = freedman_diaconis_bins(x)
            ybins = freedman_diaconis_bins(y)
            ax_xdist.hist(x, bins=xbins, color="gray", alpha=0.7)
            ax_ydist.hist(y, bins=ybins, orientation="horizontal", color="gray", alpha=0.7)

        ax_main.set_xlabel(x_col)
        ax_main.set_ylabel(y_col)
        ax_main.set_title(f"{group} ({len(subset)})")
        plt.setp(ax_xdist.get_xticklabels(), visible=False)
        plt.setp(ax_ydist.get_yticklabels(), visible=False)

    plt.suptitle(f"Joint Distribution: {x_col} vs {y_col}", fontsize=18)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_median(x, y, ax, color):
    median_x, median_y = np.median(x), np.median(y)
    ax.scatter(median_x, median_y, color=color, marker='x', s=80, zorder=5)

def plot_joint_distribution_overlayed(df, x_col, y_col, stratify_col,
    save_path: str = None, groups: list = None,
    lower_percentile: float = 0.1, upper_percentile: float = 0.9):

    if groups is None:
        groups = sorted(df[stratify_col].dropna().unique().tolist())

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_xdist = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_ydist = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    x_lower, x_upper = df[x_col].quantile([lower_percentile, upper_percentile])
    y_lower, y_upper = df[y_col].quantile([lower_percentile, upper_percentile])

    base_colors = plt.cm.tab10.colors

    for i, group in enumerate(groups):
        subset = df[df[stratify_col] == group].copy()
        mask = subset[x_col].between(x_lower, x_upper) & subset[y_col].between(y_lower, y_upper)
        subset = subset.loc[mask, [x_col, y_col]].dropna()

        if len(subset) > 1:
            x = subset[x_col].to_numpy()
            y = subset[y_col].to_numpy()

            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xx, yy = np.meshgrid(np.linspace(x_lower, x_upper, 200), np.linspace(y_lower, y_upper, 200))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            z_sorted = np.sort(zz.ravel())[::-1]
            cdf = np.cumsum(z_sorted) / np.sum(z_sorted)
            level_50 = z_sorted[np.searchsorted(cdf, 0.6)]

            ax_main.contourf(
                xx, yy, zz,
                levels=[level_50, zz.max()],
                colors=[base_colors[i % len(base_colors)]],
                alpha=0.3
            )

            ax_xdist.hist(x, bins="fd", histtype="step", color=base_colors[i % len(base_colors)], label=group)
            ax_ydist.hist(y, bins="fd", histtype="step", orientation="horizontal",
                          color=base_colors[i % len(base_colors)])

            plot_median(x, y, ax_main, color=base_colors[i % len(base_colors)])

    ax_main.set_xlabel(x_col)
    ax_main.set_ylabel(y_col)
    plt.setp(ax_xdist.get_xticklabels(), visible=False)
    plt.setp(ax_ydist.get_yticklabels(), visible=False)

    ax_xdist.legend(loc="upper right", fontsize=10)
    plt.suptitle(f"Joint Distribution: {x_col} vs {y_col}", fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def energy_distance_2d(X, Y):
    """
    Compute 2D energy distance between two arrays of points: X, Y.
    """
    XY = cdist(X, Y, metric='euclidean')
    XX = cdist(X, X, metric='euclidean')
    YY = cdist(Y, Y, metric='euclidean')
    return 2*XY.mean() - XX.mean() - YY.mean()

def permutation_test(data1, data2, n_permutations=1000):
    """
    Compute permutation-based p-value for 2D energy distance using parallel execution.

    Parameters:
    data1, data2 : np.ndarray
        Two arrays of shape (n_samples, 2) to compare.
    n_permutations : int
        Number of permutations to perform (default: 1000).

    Returns:
    obs_dist : float
        Observed energy distance between data1 and data2.
    p_value : float
        Permutation test p-value.

    Notes:
        Computes the permutations in parallel using all available cores (n_jobs=-1)
    """
    obs_dist = energy_distance_2d(data1, data2)
    combined = np.vstack([data1, data2])
    n1 = len(data1)
    
    def one_perm(seed):
        np.random.seed(seed)
        np.random.shuffle(combined)
        return energy_distance_2d(combined[:n1], combined[n1:])
    
    perm_dists = Parallel(n_jobs=-1)(delayed(one_perm)(seed) for seed in range(n_permutations))
    p_value = 1 - np.mean(np.array(perm_dists) >= obs_dist)
    return obs_dist, p_value

def run_energy_distance(df, x_col, y_col, group1, group2, group_name: str, n_permutations=1000):
    """
    Compute energy distance and permutation-test p-value between two groups for selected features.

    Parameters:
    df : pd.DataFrame
        DataFrame containing features and group labels.
    x_col, y_col : str
        Feature columns to compute distance on.
    group1, group2 : str
        Names of the groups to compare.
    group_name : str
        Column specifying group labels.
    n_permutations : int
        Number of permutation iterations (default: 1000).

    Returns:
    obs_dist : float
        Observed energy distance between the groups.
    p_value : float
        Permutation-test p-value.
    """
    data1 = df[df[group_name]==group1][[x_col, y_col]].dropna().values
    data2 = df[df[group_name]==group2][[x_col, y_col]].dropna().values
    if len(data1)<2 or len(data2)<2:
        return np.nan, np.nan
    return permutation_test(data1, data2, n_permutations=n_permutations)

def get_feature_groups(
) -> dict:
    """
    Return a dictionary mapping analysis feature groups to feature lists and targets.

    Returns:
    feature_groups : dict
        Mapping from feature group names to tuples (features_list, target_column).
    """

    slope_features = [f"{model}_{band}_slope" for model in ['sigmoid','power'] for band in ['g','r']]
    curvature_features = [f"{model}_{band}_curvature" for model in ['sigmoid','power'] for band in ['g','r']]
    early_color_features = [f"{model}_gr_mag_{phase}" for model in ['sigmoid','power'] for phase in config.PHASES]
    salt2_features = ['x1','c','z','x0','sig c','sig x1']

    feature_groups = {
        'Slope vs x1': (slope_features, 'x1'),
        'Curvature vs x1': (curvature_features, 'x1'),
        'Early g-r color vs x1': (early_color_features, 'x1'),

        'Slope vs c': (slope_features, 'c'),
        'Curvature vs c': (curvature_features, 'c'),
        'Early g-r color vs c': (early_color_features, 'c'),

        'Slope vs z': (slope_features, 'z'),
        'Curvature vs z': (curvature_features, 'z'),
        'Early g-r Color vs z': (early_color_features, 'z'),

        'Slope vs x0': (slope_features, 'x0'),
        'Curvature vs x0': (curvature_features, 'x0'),
        'Early g-r Color vs x0': (early_color_features, 'x0'),

        'Slope vs sig c': (slope_features, 'sig c'),
        'Curvature vs sig c': (curvature_features, 'sig c'),
        'Early g-r Color vs sig c': (early_color_features, 'sig c'),

        'Slope vs sig x1': (slope_features, 'sig x1'),
        'Curvature vs sig x1': (curvature_features, 'sig x1'),
        'Early g-r Color vs sig x1': (early_color_features, 'sig x1'),
    }   

    return feature_groups

def get_output_dirs(stratify_col):
    if stratify_col == "TNS classified":
        wide_df_output_dir = config.PROCESSED_DATA / "density_plots_by_TNS"
        plot_data_output_dir = config.TNS_DENSITY_ANALYSIS_OUTPUT_DIR
        plot_output_dir = config.TNS_DENSITY_PLOTS_OUTPUT_DIR
    else:
        wide_df_output_dir = config.PROCESSED_DATA / "density_plots_by_color"
        plot_data_output_dir = config.COLOR_DENSITY_ANALYSIS_OUTPUT_DIR
        plot_output_dir = config.COLOR_DENSITY_PLOTS_OUTPUT_DIR

    return wide_df_output_dir, plot_data_output_dir, plot_output_dir

def run_full_analysis(
    df: pd.DataFrame,
    feature_groups: dict,
    stratify_col: str,
    overlayed: bool = False
):
    """
    Run full density-based analysis pipeline:
    - Compute derived features
    - Generate KDE joint plots (overlayed or separate depending on flag)
    - Compute pairwise energy distances with permutation testing
    - Save plots and CSV summaries

    Parameters
    df : pd.DataFrame
        Input merged DataFrame with flux-fit and SALT2 parameters.
    feature_groups : dict
        Feature-target mappings from get_feature_groups.
    stratify_col : str
        Column to stratify by (must be 'TNS classified' or 'color change').
    overlayed : bool, optional
        If True, plots all stratified groups overlayed on a single chart.
        If False, plots each group separately (default: False).
    """
    final_df = add_rise_slope_curvature_color(df)
    plot_records = []
    pairwise_results = []

    stratify_colname = "tns_group" if stratify_col == "TNS classified" else "color change"
    groups = sorted(final_df[stratify_colname].dropna().unique().tolist())

    wide_df_output_dir, plot_data_output_dir, plot_output_dir = get_output_dirs(stratify_col)

    for fg_name, (features, target) in feature_groups.items():
        target_path = target + "_overlayed" if overlayed else target
        target_dir = os.path.join(plot_output_dir, target_path)
        os.makedirs(target_dir, exist_ok=True)

        for feat in features:
            filename = f"{fg_name.replace(' ', '_')}_{feat}_vs_{target}_KDE_overlayed.png"
            plot_save_path = os.path.join(target_dir, filename)

            if overlayed:
                plot_joint_distribution_overlayed(final_df, feat, target, stratify_col=stratify_colname, save_path=plot_save_path,)
            else:
                plot_joint_distribution(final_df, feat, target, stratify_col=stratify_colname, save_path=plot_save_path,)

            plot_records.append(
                {
                    "feature_group": fg_name,
                    "feature": feat,
                    "target": target,
                    "plot_path": os.path.join(target, filename),
                }
            )

            for group1, group2 in itertools.combinations(groups, 2):
                dist, pval = run_energy_distance(final_df, feat, target, group1, group2, group_name=stratify_colname, n_permutations=10000,)
                pairwise_results.append(
                    {
                        "feature_group": fg_name,
                        "feature": feat,
                        "target": target,
                        "group1": group1,
                        "group2": group2,
                        "energy_distance": dist,
                        "p_value": pval,
                    }
                )

    pd.DataFrame(plot_records).to_csv(
        os.path.join(plot_data_output_dir, "KDE_plot_log.csv"), index=False
    )
    pd.DataFrame(pairwise_results).to_csv(
        os.path.join(plot_data_output_dir, "energy_distance_permutation_results.csv"),
        index=False,
    )
    
    output_file_name = "TNS_stratified_wide" if stratify_col == "TNS classified" else "color_stratified_wide"
    final_df.to_csv(
        os.path.join(wide_df_output_dir, output_file_name),
        index=False,
    )