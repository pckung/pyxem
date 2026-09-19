"""Utils for Extracting Differential Scattering Signals."""

import numpy as np
from scipy.ndimage import uniform_filter



def _spatial_local_bkg_calc(
        data: np.ndarray,
        inner_cutoff: int = 3,
        outer_cutoff: int = 7,
        mode = 'mirror',
        **kwargs,
):
    """Calculate the spatial local background using inner and outer square filters.
    
    Parameters
    ----------
    data : np.ndarray
        The data array from which to calculate the local background, typically a 2D array representing spatial data.
    inner_cutoff : int, optional
        The size of the inner square filter, by default 3.
    outer_cutoff : int, optional
        The size of the outer square filter, by default 7.
    mode : str, optional
        The mode parameter determines how the input array is extended when the filter overlaps a border. Options include 'reflect', 'constant', 'nearest', 'mirror', and 'wrap'. By default, it is set to 'mirror'.
    **kwargs : dict
        Additional keyword arguments passed to the filtering function.

    Returns
    -------
    np.ndarray
        The calculated spatial local background array.
    """
    n_dim = data.ndim
    inner = uniform_filter(data, size=inner_cutoff, mode=mode, **kwargs)
    outer = uniform_filter(data, size=outer_cutoff, mode=mode, **kwargs)
    return (outer*outer_cutoff**n_dim - inner*inner_cutoff**n_dim) / (outer_cutoff**n_dim - inner_cutoff**n_dim)

def _log_bkg_removal(data: np.ndarray, bkg: np.ndarray) -> np.ndarray:
    return np.log(data / bkg)

def _get_reference_vdf(data, ref_mask) -> np.ndarray:
    return np.nanmean(data[ref_mask])


def _get_corr(data, ref_vdf_data) -> np.ndarray:
    return np.corrcoef(data.flatten(), ref_vdf_data.flatten())[0, 1]

