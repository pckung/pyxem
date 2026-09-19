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
from pyxem.signals import Diffraction2D
from hyperspy._signals.lazy import LazySignal


import numpy as np

from pyxem.utils._flucor import (
    _spatial_local_bkg_calc,
    _get_reference_vdf,
    _get_corr
)



class DifferentialDiffraction2D(Diffraction2D):
    """Signal class for differential diffuse-scattering 4D-STEM data.

    Parameters
    ----------
    *args:
        See :class:`hyperspy.api.signals.Signal2D`.
    **kwargs:
        See :class:`hyperspy.api.signals.Signal2D`
    """

    _signal_type = "differential_diffraction"


    def get_fluctuation_map(self) -> Diffraction2D:
        """Calculate the fluctuation map of the signal.

        Returns
        -------
        Diffraction2D
            The calculated fluctuation map.
        """

        fluctuation_map = self.var(axis='nav')
        return fluctuation_map

    def get_correlation_map(self, ref_mask: np.ndarray) -> Diffraction2D:
        """Calculate the correlation map of the signal with respect to a reference mask.

        Parameters
        ----------
        ref_mask : np.ndarray
            The reference mask in diffraction space to correlate with. True values in the mask indicate the regions to consider for correlation.

        Returns
        -------
        Diffraction2D
            The calculated correlation map.
        """
        ref_vdf = self.map(_get_reference_vdf, ref_mask=ref_mask, inplace=False)
        ref_vdf = ref_vdf.T
        s = self.T
        corr_map = s.map(_get_corr, ref_vdf_data=ref_vdf.data, inplace=False)
        return corr_map.T




class LazyDifferentialDiffraction2D(LazySignal, DifferentialDiffraction2D):
    pass
