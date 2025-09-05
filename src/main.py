from flux_fits.plot_flux_fits import plot_flux_fits
from sncosmo_fitting.sncosmo_fits import fit_SALT2
from sncosmo_fitting.sncosmo_fits_fixed_z import fit_SALT2_fixed_z
from plot_analysis.density_plots import density_plots
from process_tns_search import tns_search_process
import time

def main():
    print("Doing flux fits")
    plot_flux_fits()
    print("Finished with flux fits\n")

    print("Processing TNS search")
    tns_search_process()
    print("Finished with TNS search")

    print("Doing sncosmo general fits")
    fit_SALT2()
    print("Finished with sncosmo general fits\n")

    print("Doing sncosmo fits with fixed z")
    fit_SALT2_fixed_z()
    print("Finished sncosmo fits with fixed z")

    print("Starting density plots")
    density_plots()
    print("Finished density plots")

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Time taken to run main {time.time() - start}")