# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""Utils for operating on insitu signals."""

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as ss
from scipy.ndimage import uniform_filter
from typing import Literal


def _register_drift_5d(data, shifts1, shifts2, order=1):
    """
    Register 5D data set using affine transformation

     Parameters
    ----------
    data: ~numpy.ndarray or dask.array
        Input image in 5D array (time * rx * ry * kx * ky)
    shifts1: ~numpy.ndarray
        1D array for shifts in 1st real space direction or x in hyperspy indexing.
    shifts2: ~numpy.ndarray
        1D array for shifts in 2nd real space direction or y in hyperspy indexing.
    order: int
        The order of the spline interpolation for affine transformation. Default
        is 1, has to be in the range 0-5

    Returns
    -------
    data_t: ~numpy.ndarray
        5D array after translation according to shift vectors

    """
    data_t = np.zeros_like(data)
    time_size = len(shifts1)
    for i in range(time_size):
        data_t[i, :, :, :, :] = ndi.affine_transform(
            data[i, :, :, :, :],
            np.identity(4),
            offset=(shifts1[i][0, 0, 0, 0], shifts2[i][0, 0, 0, 0], 0, 0),
            order=order,
        )
    return data_t


def _register_drift_2d(data, shift1, shift2, order=1):
    """
    Register 2D data set using affine transformation

     Parameters
    ----------
    data: ~numpy.ndarray or dask.array
        Input image in 2D array (ry * rx)
    shift1: float
        shifts in 1st real space direction or x in hyperspy indexing.
    shift2: float
        shifts in 2nd real space direction or y in hyperspy indexing.
    order: int
        The order of the spline interpolation for affine transformation. Default
        is 1, has to be in the range 0-5

    Returns
    -------
    data_t: ~numpy.ndarray
        2D array after translation according to shift vectors

    """
    data_t = ndi.affine_transform(
        data, np.identity(2), offset=(shift1, shift2), order=order
    )
    return data_t


def _g2_2d(data, normalization="split", k1bin=1, k2bin=1, tbin=1):
    """
    Calculate k resolved g2(k,t) from I(t,k_r,k_phi)

    Parameters
    ----------
    data: 3D np.array
        Time series for I(t,k_r,k_phi)
    normalization: string
        Normalization format for time autocorrelation, 'split' or 'self'
    k1bin: int
            Binning factor for k1 axis
    k2bin: int
        Binning factor for k2 axis
    tbin: int
        Binning factor for t axis

    Returns
    -------
    g2: 3D np.array
        Time correlation function g2(t,k_r,k_phi)
    """
    data = data.T
    data = (
        data.reshape(
            (data.shape[0] // k1bin),
            k1bin,
            (data.shape[1] // k2bin),
            k2bin,
            (data.shape[2] // tbin),
            tbin,
        )
        .sum(5)
        .sum(3)
        .sum(1)
    )

    # Calculate autocorrelation along time axis
    autocorr = ss.fftconvolve(data, data[:, :, ::-1], mode="full", axes=[-1])
    norm = ss.fftconvolve(np.ones(data.shape), data[:, :, ::-1], mode="full", axes=[-1])
    if normalization == "self":
        overlap_factor = np.expand_dims(
            np.linspace(data.shape[-1], 1, data.shape[-1]), axis=(0, 1)
        )
        norm_factor = norm[:, :, data.shape[-1] : 0 : -1] ** 2
        g2 = autocorr[:, :, data.shape[-1] - 1 :] / norm_factor * overlap_factor
    elif normalization == "split":
        overlap_factor = np.expand_dims(
            np.linspace(data.shape[-1], 1, data.shape[-1]), axis=(0, 1)
        )
        norm_factor = (
            norm[:, :, data.shape[-1] - 1 :]
            * norm[:, :, ::-1][:, :, data.shape[-1] - 1 :]
        )
        g2 = autocorr[:, :, data.shape[-1] - 1 :] / norm_factor * overlap_factor
    else:
        raise ValueError(
            normalization
            + " not recognize, normalization must be chosen 'split' or 'self'"
        )

    return g2.T


def _get_resample_time(t_size, dt, t_rs_size):
    """
    Return log linear resampled time array based on time step and sampling points

    Parameters
    ----------
    t_size: int
        Size of original time array
    dt: float
        Time interval for original time array
    t_rs_size: int
        Size of resampled time array

    Returns
    -------
    t_rs: 1D np.array
        Resampled time array
    """
    t_rs = np.zeros(t_rs_size, dtype=float)

    for i in range(t_rs_size):
        t_rs[i] = np.power(
            10, np.log10(dt) + np.log10(t_size - 1) * i / (t_rs_size - 1)
        )

    return t_rs


def _interpolate_g2_2d(g2, t_rs, dt):
    """
    Interpolate k resolved g2(k,t) based on resampled time

    Parameters
    ----------
    g2: 3D np.array
        Time correlation function g2(t,k_r,k_phi)
    t_rs: 1D np.array
        Resampled time axis array
    dt: float
        Time interval for original g2 function

    Returns
    -------
    g2rs: 3D np.array
        Resampled time correlation function g2(t,k_r,k_phi)
    """
    t = np.round(t_rs / dt, 8)
    g2_l = g2[np.floor(t).astype(int)]
    g2_h = g2[np.ceil(t).astype(int)]
    g2_rs = (
        g2_l + (g2_h - g2_l) * (t - np.floor(t).astype(int))[:, np.newaxis, np.newaxis]
    )
    return g2_rs

def _interpolate_g2_1d(g2, t_rs, dt):
    """
    Interpolate 1D g2(t) based on resampled time

    Parameters
    ----------
    g2: 1D np.array
        Time correlation function g2(t)
    t_rs: 1D np.array
        Resampled time axis array
    dt: float
        Time interval for original g2 function

    Returns
    -------
    g2rs: 1D np.array
        Resampled time correlation function g2(t)
    """
    t = np.round(t_rs / dt, 8)
    g2_l = g2[np.floor(t).astype(int)]
    g2_h = g2[np.ceil(t).astype(int)]
    g2_rs = g2_l + (g2_h - g2_l) * (t - np.floor(t).astype(int))
    return g2_rs

def _bkg_calc(
    data: np.ndarray,
    axis: list | int,
    local_axis: list | int | None = None,
    local_size: list | int | None = None,
    center: Literal["mean", "median"] = "mean",
    **kwargs,
):
    """Calculate the background by taking the mean along the specified axis and expend to original shape.
    
    Parameters
    ----------
    data : np.ndarray
        The data array from which to calculate the background.
    axis : list | int
        The axis or axes along which to calculate the mean background.
    local_axis : list | int | None, optional
        The axis or axes along which to apply a local uniform filter, by default None.
    local_size : int, optional
        The size of the local uniform filter, by default 3.
    **kwargs : dict
        Additional keyword arguments passed to the filtering function.

    Returns
    -------
    np.ndarray
        The calculated background array, broadcasted to the shape of the input data.

    """
    if axis is not None:
        if isinstance(axis, int):
            axis = [axis]
        if center == "median":
            mean = np.nanmedian(data, axis=tuple(axis), keepdims=True)
        else:
            mean = np.nanmean(data, axis=tuple(axis), keepdims=True)
        bkg = np.broadcast_to(mean, data.shape)
    else:
        bkg = data.copy()
    if local_axis is not None:
        n_dim = data.ndim
        if isinstance(local_axis, int):
            local_axis = [local_axis]
        filter_size = [1] * n_dim
        if local_size is None:
            local_sizes = [3] * len(local_axis)
        elif isinstance(local_size, int):
            local_sizes = [local_size] * len(local_axis)
        elif isinstance(local_size, list):
            local_sizes = local_size
        else:
            raise TypeError("'local_size' must be int, list, or None")
        for ax, sz in zip(local_axis, local_sizes):
            filter_size[ax] = sz
        bkg = uniform_filter(bkg, size=filter_size, **kwargs)
    return bkg

def _find_time_axis(signal, nav=True) -> int:
        time_axis_names = ["time", "t", "Time", "T"]
        axes = signal.axes_manager.navigation_axes if nav else signal.axes_manager.signal_axes
        for i, ax in enumerate(axes):
            if ax.name in time_axis_names:
                time_axis_index = i
                break
        else:
            raise ValueError(f"Time axis not found, name should be one of {time_axis_names}")
        return time_axis_index

def _ttcf_2_g2(data: np.ndarray) -> np.ndarray:
    """Fast, NaN-aware 1D g2 calculation using numpy.nanmean along diagonals."""
    nt = data.shape[0]
    g2 = np.empty(nt)
    
    # Loop over time-lags (diagonals)
    for offset in range(nt):
        # Extract the k-th upper diagonal (representing a constant time lag)
        diag = np.diagonal(data, offset=offset)
        
        # Calculate mean ignoring NaNs
        # np.nanmean handles summing and count tracking in C
        g2[offset] = np.nanmean(diag)
        
    return g2

def _ttcf_2_c2(data, window: int = 100, size: int | None = None) -> np.ndarray:
    """Convert a two-time correlation function (ttcf) to a series of c2 windows.
    
    Parameters
    ----------
    data : np.ndarray
        The two-time correlation function data.
    window : int, optional
        The size of the delay time window to extract from the ttcf, by default 100. 
    size : int, optional
        The size of the uniform filter to apply to the c2 windows along the wait time, by default None.

    Returns
    -------
    np.ndarray
        The extracted c2 windows, optionally smoothed with a uniform filter.
    """
    c2_windows = np.zeros((len(data) - window + 1, window))
    for i in range(len(data) - window + 1):
        c2_windows[i] = data[i, i : i + window]

    if size is not None:
        c2_windows = uniform_filter(c2_windows, size=(size, 0), mode="mirror")
    return c2_windows