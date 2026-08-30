"""Utils for Extracting Differential Scattering Signals."""

import numpy as np
from scipy.ndimage import uniform_filter



def _spatial_local_bkg_calc(
        data: np.ndarray,
        inner_size: int = 3,
        outer_size: int = 7,
        **kwargs,
):
    """Calculate the spatial local background using inner and outer square filters.
    
    Parameters
    ----------
    data : np.ndarray
        The data array from which to calculate the local background, typically a 2D array representing spatial data.
    inner_size : int, optional
        The size of the inner square filter, by default 3.
    outer_size : int, optional
        The size of the outer square filter, by default 7.
    **kwargs : dict
        Additional keyword arguments passed to the filtering function.

    Returns
    -------
    np.ndarray
        The calculated spatial local background array.
    """
    n_dim = data.ndim
    inner = uniform_filter(data, size=inner_size, **kwargs)
    outer = uniform_filter(data, size=outer_size, **kwargs)
    return (outer*outer_size**n_dim - inner*inner_size**n_dim) / (outer_size**n_dim - inner_size**n_dim)
    

