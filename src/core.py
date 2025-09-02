import numpy as np
from typing import Tuple


def convert_magpsf_to_flux(magpsf: np.ndarray, sigmapsf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ZTF PSF magnitudes to flux in milliJanskys (mJy).

    Parameters:
    magpsf : np.ndarray
        Array of ZTF PSF magnitudes.
    sigmapsf : np.ndarray
        Array of magnitude uncertainties.

    Returns:
    flux : np.ndarray
        Flux values in mJy.
    sigmaflux : np.ndarray
        Uncertainties on flux values in mJy.
    """
    flux = 3631 * 10**(-0.4 * magpsf) * 1e3  # Factor of 10^3 for mJ conversion
    sigmaflux = 0.4 * np.log(10) * sigmapsf * flux
    return flux, sigmaflux