import config
from plot_flux_fits import plot_flux_fits
from sncosmo_fits import fit_SALT2
from sncosmo_fits_fixed_z import fit_SALT2_fixed_z
from density_plots import density_plots

def main():
    print("Doing flux fits")
    plot_flux_fits()
    print("Finished with flux fits\n")

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
    main()