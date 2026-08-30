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

from typing import TYPE_CHECKING, Optional, Union

from hyperspy.signals import Signal1D, Signal2D, BaseSignal
from .diffraction1d import Diffraction1D
from hyperspy._signals.lazy import LazySignal
from pyxem.utils._insitu import _get_resample_time, _interpolate_g2_1d
from hyperspy.axes import DataAxis
import hyperspy.api as hs

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def _KWW_curve(t, tau, beta, C):
    return C * np.exp(-2 * np.power((t / tau), beta))

def _g2_fit(g2_rs, t_rs, tau_guess, mask=None, y_err=None):
    if mask is not None and mask:
        g20 = g2_rs[0]
        p0 = [tau_guess, 1, g20]
        try:
            popt, pcov = curve_fit(
                _KWW_curve, 
                t_rs[1:-10], 
                g2_rs[1:-10], 
                sigma = y_err[1:-10] if y_err is not None else None,
                p0=p0,
                bounds = ([0.1, 0.1, 0.5*g20], [np.inf, 2, 2*g20]),
                maxfev=1000000
            )
            perr = np.sqrt(np.diag(pcov))
            return np.concatenate([popt, perr])
        except Exception as e:
            return np.full(6, np.nan)
    else:
        return np.full(6, np.nan)

def _popt_to_g2(popt, t_rs=None):
    if t_rs is None:
        raise ValueError("t_rs must be provided")
    return _KWW_curve(t_rs, *popt[:3])

class OneTimeCorrelationFunction(Diffraction1D):
    """Signal class for one-time correlation functions of in-situ 4D-STEM data.

    Parameters
    ----------
    *args:
        See :class:`hyperspy.api.signals.Signal2D`.
    **kwargs:
        See :class:`hyperspy.api.signals.Signal2D`
    """

    _signal_type = "otcf"
    _signal_dimension = 1
    _resampled = False
    _fitted_result = False

    @property
    def _dt(self):
        return self.axes_manager.signal_axes[0].scale
    @property
    def _time_steps(self):
        return self.axes_manager.signal_axes[0].size

    def resample_g2(self, t_rs_size: Optional[int] = 100, dt: Optional[float] = None) -> "OneTimeCorrelationFunction":
        """
        Resample the g2 function along the time axis.

        Parameters
        ----------
        t_rs_size: int, optional
            Size of the resampled time array. Default is 100.
        dt: float, optional
            Time interval for the original g2 function. If None, it will use the signal's time axis scale.

        Returns
        -------
        OneTimeCorrelationFunction
            Resampled one-time correlation function
        """
        if dt is None:
            dt = self._dt
        t_rs = _get_resample_time(self._time_steps, dt, t_rs_size)
        g2_rs = self.map(_interpolate_g2_1d, t_rs=t_rs, dt=dt, inplace=False)
        new_axis = DataAxis(
            name="resampled_time",
            units=g2_rs.axes_manager.signal_axes[0].units,
            axis=t_rs
            )
        g2_rs.axes_manager._axes[-1] = new_axis
        g2_rs._resampled = True
        g2_rs._t_rs = t_rs
        return g2_rs

    def plot_log(self):
        """
        Plot the one-time correlation function on a logarithmic time axis.
        """
        self.plot()
        fig = plt.gcf()
        axes = fig.get_axes()
        axes[-1].set_xscale('log')
        axes[-1].axhline(y=0, color='black', linestyle=':')
        return fig, axes

    def fit_g2_decay(self, tau_int: float, mask: np.ndarray | None = None, y_err: Optional["OneTimeCorrelationFunction"] = None) -> "OneTimeCorrelationFunction":
        """
        Fit the one-time correlation function using the G2 decay model. Recommend normalizing the signal before fitting.

        Parameters
        ----------
        tau_int: float
            Initial guess for the decay time constant.
        mask: np.ndarray or None, optional
            Mask to apply during fitting. If None, no mask is applied. False region is not fitted. Need to have the same shape as the navigation shape of the signal.
        y_err: OneTimeCorrelationFunction or None, optional
            Error in the y-data for weighted fitting. If None, no weighting is applied. Need to have the same navigation shape as the signal.
            
        Returns
        -------
        OneTimeCorrelationFunction
            Fitted Results ([tau, beta, C, tau_err, beta_err, C_err])
        """
        if not self._resampled:
            raise RuntimeError("Signal recommended to be resampled before fitting. Call `resample_g2` before fitting. To force non-resampled fitting, set the `_resampled` attribute to True.")

        if mask is not None:
            nav_shape = list(self.axes_manager.navigation_shape)
            mask = mask.astype(bool)
            mask = np.broadcast_to(mask, nav_shape[::-1])
            mask = BaseSignal(mask).T
        else:
            mask = True
        if y_err is not None:
            if not isinstance(y_err, OneTimeCorrelationFunction):
                raise TypeError("y_err must be an instance of OneTimeCorrelationFunction.")
            if y_err.axes_manager.navigation_shape != self.axes_manager.navigation_shape:
                raise ValueError("y_err must have the same navigation shape as the signal.")
        kww_fit = self.map(_g2_fit, t_rs=self._t_rs, tau_guess=tau_int, mask=mask, y_err=y_err, inplace=False, output_signal_size=(6,))
        kww_fit = OneTimeCorrelationFunction(kww_fit.data)
        if self._lazy:
            kww_fit = kww_fit.as_lazy()
        for i, ax in enumerate(kww_fit.axes_manager.navigation_axes):
            ax.name = self.axes_manager.navigation_axes[i].name
            ax.units = self.axes_manager.navigation_axes[i].units
            ax.scale = self.axes_manager.navigation_axes[i].scale
            ax.offset = self.axes_manager.navigation_axes[i].offset
        kww_fit._fitted_result = True
        kww_fit._t_rs = self._t_rs
        kww_fit.axes_manager._axes[-1].name = "fitted_results"
        kww_fit.axes_manager._axes[-1].size = 6
        return kww_fit

    def get_fitted_g2(self) -> "OneTimeCorrelationFunction":
        """
        Retrieve the fitted one-time correlation function.

        Returns
        -------
        OneTimeCorrelationFunction
            Fitted one-time correlation function.
        """
        if not self._fitted_result:
            raise RuntimeError("The one-time correlation function has not been fitted yet. Call `fit_g2_decay` first.")
        fitted_g2 = self.map(_popt_to_g2, t_rs=self._t_rs, inplace=False)
        fitted_g2._fitted_result = False
        fitted_g2._t_rs = self._t_rs
        return fitted_g2

        
    
    

    

class LazyOneTimeCorrelationFunction(LazySignal, OneTimeCorrelationFunction):
    pass
