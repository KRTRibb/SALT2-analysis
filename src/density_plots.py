"""
run_sncosmo_analysis_driver.py

Driver script to perform density-based analysis of SALT2 and flux-fit results
for early Type Ia candidate light curves.

Workflow:
1. Loads flux-fit and SALT2 results from CSV files.
2. Computes derived features (rise time, slope, curvature, early g-r color).
3. Runs analysis stratified by:
   - TNS classification (Ia vs non-Ia)
   - Early color change
4. Generates KDE joint plots for feature-target pairs.
5. Computes pairwise energy distances between groups with permutation testing.
6. Saves plots and CSV summaries of results.

Inputs:
- general_results.csv: SALT2 fit results
- flux_fits_data.csv: precomputed flux-fit parameters

Outputs:
- KDE plots per feature and target, organized by stratification.
- CSV logs of plots and energy distance permutation-test results.
- CSV of merged data with derived features.
"""

import sncosmo_analysis_functions as saf


def main():
    """
    Run the full sncosmo analysis workflow for both TNS classification and color-change stratifications.

    Workflow:
    - Loads the relevant CSV files with flux-fit and SALT2 data.
    - Computes derived features for each object.
    - Runs 'run_full_analysis' for TNS classification stratification.
    - Runs 'run_full_analysis' for color-change stratification.
    - Saves all plots and CSV summaries to configured directories.
    """
    sncosmo_dir = "data/raw/general_results.csv"
    flux_dir = "data/raw/flux_fits_data.csv"

    df_tns = saf.load_flux_and_sncosmo(flux_dir, sncosmo_dir, stratify_col="TNS classified")

    feature_groups = saf.get_feature_groups()

    print("Running analysis stratified by TNS classification")
    saf.run_full_analysis(df_tns, feature_groups=feature_groups, stratify_col="TNS classified")
    print("Done with TNS stratification.\n")

    df_color = saf.load_flux_and_sncosmo(flux_dir, sncosmo_dir, stratify_col="color change")

    print("Running analysis stratified by color change")
    saf.run_full_analysis(df_color, feature_groups=feature_groups, stratify_col="color change")
    print("Done with color change stratification.\n")


if __name__ == "__main__":
    main()
