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

from typing import TYPE_CHECKING, Optional, Union, Literal

from hyperspy.signals import Signal1D, Signal2D
from pyxem.signals import PolarDiffraction2D, InSituDiffraction2D
from hyperspy._signals.lazy import LazySignal


if TYPE_CHECKING:
    from pyxem.signals.correlation2d import Correlation2D
    from pyxem.signals.ttcf import TwoTimeCorrelationFunction

import numpy as np
from hyperspy.roi import RectangularROI
from hyperspy.roi import BaseROI

import dask.array as da
from dask.graph_manipulation import clone

from pyxem.utils._dask import _get_dask_array, _get_chunking
from pyxem.utils._insitu import (
    _register_drift_5d,
    _register_drift_2d,
    _g2_2d,
    _interpolate_g2_2d,
    _get_resample_time,
)
import pyxem.utils._pixelated_stem_tools as pst
from pyxem.utils._differential_scattering import (
    _bkg_calc,
    _find_time_axis,
    _find_azim_axis,
)



class InSituPolarDiffraction2D(PolarDiffraction2D, InSituDiffraction2D):
    """Signal class for in-situ 4D-STEM data.

    Parameters
    ----------
    *args:
        See :class:`hyperspy.api.signals.Signal2D`.
    **kwargs:
        See :class:`hyperspy.api.signals.Signal2D`
    """

    _signal_type = "insitu_polar"

    def expected_polar_intensity(
            self,
            method: Literal["tazi_local", "tazi", "t", "k"] = "tazi_local",
            local_azi_angle: float = 60.0,
            chunk_optimize: bool = False
            ) -> "InSituPolarDiffraction2D":
        """
        Calculate the expected intensity in the time series using the specified method. Signal is transposed so that time axis is the last signal axis (first axis of the signal ndarray) and the azimuthal axis is the first signal axis (last axis of the signal ndarray).
        
        Parameters
        ----------
        method : Literal["tazi_local", "tazi", "t", "k"], default="tazi_local"
            The method to use for calculating the expected intensity.
            The "tazi_local" method calculates the expected intensity using the time axis and a local azimuthal angle, "tazi" uses the full azimuthal axis, "t" uses the time axis, and "k" uses all non-time axes.
        local_azi_angle : float, default=60.0
            The local azimuthal angle in degrees for the "tazi_local" method.
        chunk_optimize : bool, default=False
            Whether to optimize the chunking of the signal for computation after initial transpose of the signal. This can improve performance for large lazy datasets if the chunking is not optimize to iterate over the spatial axes.
        
        Returns
        -------
        InSituPolarDiffraction2D
            The expected intensity signal.
        """
        custom_axes = None
        if method == "tazi_local":
            _method = "custom"
            custom_axes = {}
            custom_axes['full_axis'] = [0]
            custom_axes['local_axis'] = [2]
            azi_size = int(local_azi_angle / 360.0 * self.axes_manager.signal_axes[0].size)
            if azi_size % 2 == 0:
                azi_size += 1
            custom_axes['local_size'] = azi_size
            
        elif method == 'tazi':
            _method = "custom"
            custom_axes = {}
            custom_axes['full_axis'] = [0]
            custom_axes['local_axis'] = None
            custom_axes['local_size'] = None
        elif method == 't':
            _method = "t"
        elif method == 'k':
            _method = "k"
        else:
            raise ValueError("Method must be one of 'tazi_local', 'tazi', 't', 'k'")

        return self.expected_intensity(method=_method, custom_axes=custom_axes, chunk_optimize=chunk_optimize)
        



class LazyInSituPolarDiffraction2D(LazySignal, InSituPolarDiffraction2D):
    pass
