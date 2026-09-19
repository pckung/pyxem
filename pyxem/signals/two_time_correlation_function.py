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

from hyperspy.signals import Signal1D, Signal2D

if TYPE_CHECKING:
    from .one_time_correlation_function import OneTimeCorrelationFunction
from .diffraction2d import Diffraction2D
from hyperspy._signals.lazy import LazySignal
from pyxem.utils._insitu import _ttcf_2_g2, _ttcf_2_c2
import numpy as np



class TwoTimeCorrelationFunction(Diffraction2D):
    """Signal class for two-time correlation functions of in-situ 4D-STEM data.

    Parameters
    ----------
    *args:
        See :class:`hyperspy.api.signals.Signal2D`.
    **kwargs:
        See :class:`hyperspy.api.signals.Signal2D`
    """

    _signal_type = "ttcf"
    _signal_dimension = 2

    def get_g2(self) -> "OneTimeCorrelationFunction":
        """Return the g2 one-time correlation function as a 1D signal."""
        g2_signal = self.map(_ttcf_2_g2, inplace=False)
        g2_signal.set_signal_type("otcf")
        return g2_signal

    def get_c2(self, window: int = 100, size: Optional[int] = None) -> "OneTimeCorrelationFunction":
        """Return the c2 two-time correlation function as a 1D signal along the Delta t axis.
        
        Parameters
        ----------
        window : int, optional
            The size of the delay time window to extract from the ttcf, by default 100.
        size : int, optional
            The size of the uniform filter to apply to the c2 windows along the wait time, by default None.

        Returns
        -------
        "OneTimeCorrelationFunction"
            The extracted c2 windows as a 1D signal.
        """
        c2_signal = self.map(_ttcf_2_c2, inplace=False, window=window, size=size)
        tax = self.axes_manager.signal_axes[0]
        ax_name = ["Delay Time", "Wait Time"]
        for i, ax in enumerate(c2_signal.axes_manager.signal_axes):
            ax.name = ax_name[i]
            ax.units = tax.units
            ax.scale = tax.scale
            ax.offset = tax.offset

        navigation_shape = self.axes_manager.navigation_shape
        navigation_axes_len = len(navigation_shape)
        if navigation_axes_len > 0:
            c2_signal = c2_signal.transpose(navigation_axes=[0,1,3])
        else:
            c2_signal = c2_signal.transpose(navigation_axes=[1])
        if size is not None:
            edge = size // 2
            if navigation_axes_len > 0:
                c2_signal = c2_signal.inav[:,:,edge:-edge]
            else:
                c2_signal = c2_signal.inav[edge:-edge]
        c2_signal.set_signal_type("otcf")
        return c2_signal



class LazyTwoTimeCorrelationFunction(LazySignal, TwoTimeCorrelationFunction):
    pass
