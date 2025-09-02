import config
from plot_flux_fits import plot_flux_fits
from sncosmo_fits import fit_SALT2
from sncosmo_fits_fixed_z import fit_SALT2_fixed_z
from density_plots import density_plots

def main():
    plot_flux_fits()
    fit_SALT2()
    fit_SALT2_fixed_z()
    density_plots()


if __name__ == '__main__':
    main()