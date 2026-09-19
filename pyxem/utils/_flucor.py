"""Utils for Extracting Differential Scattering Signals."""

import numpy as np
from scipy.ndimage import uniform_filter



def _spatial_local_bkg_calc(
    data: np.ndarray,
    inner_cutoff: int = 3,
    outer_cutoff: int = 7,
    mode: str = "mirror",
    **kwargs,
):
    """Calculate the spatial local background as the mean of a square annulus.

    The annulus is the region within ``outer_cutoff`` pixels of the centre
    pixel but outside ``inner_cutoff`` pixels, with the distance measured
    along each axis (a square window of side ``2 * cutoff + 1``).

    Parameters
    ----------
    data : np.ndarray
        The data array from which to calculate the local background,
        typically a 2D array representing spatial data.
    inner_cutoff : int, optional
        The inner cutoff radius in pixels, by default 3. Pixels within this
        radius are excluded from the background.
    outer_cutoff : int, optional
        The outer cutoff radius in pixels, by default 7. Pixels beyond this
        radius are excluded from the background.
    mode : str, optional
        How the input array is extended when the filter overlaps a border.
        Options include 'reflect', 'constant', 'nearest', 'mirror' and
        'wrap'. By default, it is set to 'mirror'.
    **kwargs : dict
        Additional keyword arguments passed to the filtering function.

    Returns
    -------
    np.ndarray
        The calculated spatial local background array.
    """
    n_dim = data.ndim
    inner_size = 2 * inner_cutoff + 1
    outer_size = 2 * outer_cutoff + 1
    inner = uniform_filter(data, size=inner_size, mode=mode, **kwargs)
    outer = uniform_filter(data, size=outer_size, mode=mode, **kwargs)
    n_inner = inner_size**n_dim
    n_outer = outer_size**n_dim
    return (outer * n_outer - inner * n_inner) / (n_outer - n_inner)

def _log_bkg_removal(data: np.ndarray, bkg: np.ndarray) -> np.ndarray:
    return np.log(data / bkg)

def _get_reference_vdf(data, ref_mask) -> np.ndarray:
    return np.nanmean(data[ref_mask])


def _get_corr(data, ref_vdf_data) -> np.ndarray:
    return np.corrcoef(data.flatten(), ref_vdf_data.flatten())[0, 1]

