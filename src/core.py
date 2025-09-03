import numpy as np
from typing import Tuple
import config
import requests
import pandas as pd
import io

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

def fetch_valid_object_data_fink(object_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch the light curve data for a single object and filter out all invalid poins

    Parameters:
    object_id : string
        The ZTF object id of the desired object
    start_date, end_date : string, string
        The range of dates for which to search for this object, by default they are the global start and end dates from congif.py

    Returns:
    lc_valid : pd.DataFrame:
        The pandas dataframe of the object information with all the bad quality points masked out

    Notes:
        - start_date and end_date must be in the yyyy-mm-dd format
        - Includes upper limits (`withupperlim=True`) in the download.

    """

    r = requests.post(
            f'{config.FINK_API_URL}/api/v1/objects',
            json={
                'objectId': object_id,
                'output-format': 'json',
                'withupperlim': 'True',
                'startdate': start_date,
                'stopdate': end_date,
            }
    )

    single_object = pd.read_json(io.BytesIO(r.content))
    maskValid = single_object['d:tag'] == 'valid'
    lc_valid = single_object[maskValid]
    return lc_valid


