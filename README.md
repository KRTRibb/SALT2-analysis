Early SNe Ia Lightcurve Analysis

Author: Kieran Tribble
Institution: Imperial College London
Role: Student
Email: [kieran.tribble24@imperial.ac.uk](mailto:kieran.tribble24@imperial.ac.uk)

---

Overview

This repository contains code for fitting and analyzing SALT2 models to the lightcurves of early Type Ia supernova (SNe Ia) candidates classified by the Fink data stream using the sncosmo library. The SALT2 fits are compared to sigmoid and power-law fits applied to the rising part of these lightcurves, and features derived from these fits are used for further analysis.

---

Workflow

1. SALT2 Fits

* Use the driver script:
  python src/sncosmo\_fits.py
* Generates plots and saves parameters for all objects listed in data/raw/flux\_fits\_data.csv.
* Functions used for fitting are defined in src/sncosmo\_fitting\_functions.py
* Can also fit objects classified by the Transient Name Server (TNS) using known redshifts.

2. Exploratory Analysis

* Jupyter notebooks in the notebooks/ folder are used for exploratory data analysis.
* Some plots are already provided.
* Additional helper functions are available in src/exploratory\_analysis\_functions.py

3. Feature Comparison and Density Plots

* Driver script:
  python src/density\_plots.py
* Generates distribution plots of features derived from early flux sigmoid and power-law fits.
* Comparisons are made using energy distance.
* Functions for analysis are provided in src/sncosmo\_analysis\_functions.py

---

Project Structure

data/             # Raw and processed datasets
notebooks/        # Jupyter notebooks for exploratory analysis
results/          # Generated plots and fitted parameters
src/              # Python scripts and function libraries
.gitignore        # Git ignore rules
README.md         # This file

---

Requirements

* Python 3.x
* sncosmo library
* Standard scientific Python libraries (numpy, scipy, matplotlib, etc.)

---

Contact

For questions or suggestions, please contact: [kieran.tribble24@imperial.ac.uk](mailto:kieran.tribble24@imperial.ac.uk)
