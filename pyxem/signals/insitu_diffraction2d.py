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


if TYPE_CHECKING:
    from pyxem.signals.correlation2d import Correlation2D
    from pyxem.signals.ttcf import TwoTimeCorrelationFunction
    from pyxem.signals.insitu_polar_diffraction2d import InSituPolarDiffraction2D, LazyInSituPolarDiffraction2D

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
    _bkg_calc,
    _find_time_axis,
)
import pyxem.utils._pixelated_stem_tools as pst

def _compute_ttcf(data):
    nt = data.shape[0]
    data = data.reshape(nt, -1)
    ttcf = data @ data.T / data.shape[1]
    return ttcf


class InSituDiffraction2D(Diffraction2D):
    """Signal class for in-situ 4D-STEM data.

    Parameters
    ----------
    *args:
        See :class:`hyperspy.api.signals.Signal2D`.
    **kwargs:
        See :class:`hyperspy.api.signals.Signal2D`
    """

    _signal_type = "insitu_diffraction"

    def roll_time_axis(self, time_axis: int) -> "InSituDiffraction2D":
        """Roll time axis to default index (2)"""
        return self.rollaxis(time_axis, 2)

    def get_time_series(self, roi: BaseROI = None, time_axis: int = 2) -> Signal2D:
        """Create a intensity time series from virtual aperture defined by roi.

        Parameters
        ----------
        roi: :obj:`~hyperspy.roi.BaseInteractiveROI`
            Roi for virtual detector. If None, full roi of diffraction plane is used
        time_axis: int
            Index of time axis. Default is 2

        Returns
        ---------
        virtual_series: ~hyperspy.api.signals.Signal2D
            Time series of virtual detector images
        """
        out_axes = [0, 1, 2]
        out_axes.remove(time_axis)

        if roi is None:
            roi = RectangularROI(
                self.axes_manager.signal_extent[0],
                self.axes_manager.signal_extent[2],
                self.axes_manager.signal_extent[1],
                self.axes_manager.signal_extent[3],
            )

        virtual_series = self.get_integrated_intensity(roi, out_signal_axes=out_axes)
        virtual_series.metadata.General.title = "Integrated intensity time series"

        return virtual_series

    def get_drift_vectors(
        self,
        time_axis: int = 2,
        reference: str = "cascade",
        sub_pixel_factor: float = 10,
        **kwargs,
    ) -> Signal1D:
        """Calculate real space drift vectors from time series of images

        Parameters
        ----------
        s: :class:`hyperspy.api.signals.Signal2D`
            Time series of reconstructed images
        reference: 'current', 'cascade', or 'stat'
            reference argument passed to :meth:`~hyperspy.api.signals.Signal2D.estimate_shift2D`
            function. Default is 'cascade'
        sub_pixel_factor: float
            sub_pixel_factor passed to :meth:`~hyperspy.api.signals.Signal2D.estimate_shift2D`
            function. Default is 10
        **kwargs:
            Passed to the :meth:`~pyxem.signals.InSituDiffraction2D.get_time_series` function

        Returns
        -------
        shift_vectors : ~hyperspy.api.signals.Signal1D
        """
        roi = kwargs.pop("roi", None)
        ref = self.get_time_series(roi=roi, time_axis=time_axis)

        s = ref.estimate_shift2D(
            reference=reference, sub_pixel_factor=sub_pixel_factor, **kwargs
        )
        shift_vectors = Signal1D(s)

        pst._copy_axes_object_metadata(
            self.axes_manager.navigation_axes[time_axis],
            shift_vectors.axes_manager.navigation_axes[0],
        )

        return shift_vectors

    def correct_real_space_drift(
        self,
        shifts: Signal1D = None,
        time_axis: int = 2,
        order: int = 1,
        lazy_result: bool = True,
    ) -> "InSituDiffraction2D":
        """
        Perform real space drift registration on the dataset.

        Parameters
        ----------
        shifts: ~hyperspy.api.signals.Signal1D
            shift vectors to register, must be in the shape of <N_time | 2>.
            If None, shift vectors will be calculated automatically
        time_axis: int
            Index of time axis. Default is 2
        lazy_result: bool, default True
            Whether to return lazy result.
        order: int
           The order of the spline interpolation for registration. Default is 1

        Returns
        ---------
        registered_data: InSituDiffraction2D
            Real space drift corrected version of the original dataset
        """
        if shifts is None:
            shifts = self.get_drift_vectors(time_axis=time_axis)

        if time_axis != 2:
            s_ = self.roll_time_axis(time_axis)
        else:
            s_ = self
        dask_data = _get_dask_array(s_)

        if self._lazy:
            time_chunks = s_.get_chunk_size()[0][0]
        else:
            time_chunks = _get_chunking(s_)[0][0]
        xdrift = shifts.data[:, 0]
        ydrift = shifts.data[:, 1]
        xdrift_dask = da.from_array(
            xdrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
            chunks=(time_chunks, 1, 1, 1, 1),
        )
        ydrift_dask = da.from_array(
            ydrift[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
            chunks=(time_chunks, 1, 1, 1, 1),
        )
        depthx = np.ceil(np.max(np.abs(xdrift))).astype(int)
        depthy = np.ceil(np.max(np.abs(ydrift))).astype(int)
        overlapped_depth = {0: 0, 1: depthx, 2: depthy, 3: 0, 4: 0}

        data_overlapped = da.overlap.overlap(
            dask_data, depth=overlapped_depth, boundary={a: "none" for a in range(5)}
        )

        # Clone original overlap dask array to work around memory release issue in map_overlap
        data_clones = da.concatenate(
            [clone(b, omit=data_overlapped) for b in data_overlapped.blocks]
        )

        mapped = data_clones.map_blocks(
            _register_drift_5d,
            shifts1=xdrift_dask,
            shifts2=ydrift_dask,
            order=order,
            dtype="float32",
        )

        registered_data = InSituDiffraction2D(
            da.overlap.trim_internal(mapped, overlapped_depth)
        ).as_lazy()

        # Set axes info for registered signal
        for nav_axis_old, nav_axis_new in zip(
            s_.axes_manager.navigation_axes,
            registered_data.axes_manager.navigation_axes,
        ):
            pst._copy_axes_object_metadata(nav_axis_old, nav_axis_new)
        for sig_axis_old, sig_axis_new in zip(
            s_.axes_manager.signal_axes, registered_data.axes_manager.signal_axes
        ):
            pst._copy_axes_object_metadata(sig_axis_old, sig_axis_new)

        if not lazy_result:
            registered_data.compute()

        return registered_data

    def correct_real_space_drift_fast(
        self, shifts: Signal1D = None, time_axis: int = 2, order: int = 1, **kwargs
    ) -> "InSituDiffraction2D":
        """
        Perform real space drift registration on the dataset with fast performance
        over spatial axes. If signal is lazy, spatial axes must not be chunked

        Parameters
        ----------
        shifts: ~hyperspy.api.signals.Signal1D
            shift vectors to register, must be in the shape of <N_time | 2>.
            If None, shift vectors will be calculated automatically
        time_axis: int
            Index of time axis. Default is 2
        order: int
           The order of the spline interpolation for registration. Default is 1
        **kwargs:
            Passed to :meth:`~hyperspy.api.signals.BaseSignal.map`

        Returns
        ---------
        registered_data: InSituDiffraction2D
            Real space drift corrected version of the original dataset
        """
        if self._lazy:
            nav_axes = [0, 1, 2]
            nav_axes.remove(2 - time_axis)
            chunkings = self.get_chunk_size()
            if len(chunkings[nav_axes[0]]) != 1 or len(chunkings[nav_axes[1]]) != 1:
                raise Exception(
                    "Spatial axes are chunked. Please rechunk signal or use 'correct_real_space_drift' "
                    "instead"
                )

        if shifts is None:
            shifts = self.get_drift_vectors(time_axis=time_axis)

        if time_axis != 2:
            s_ = self.roll_time_axis(time_axis=time_axis)
        else:
            s_ = self
        s_transposed = s_.transpose(signal_axes=(0, 1))

        xdrift = shifts.data[:, 0]
        ydrift = shifts.data[:, 1]
        xs = Signal1D(
            np.repeat(
                np.repeat(
                    xdrift[:, np.newaxis, np.newaxis],
                    repeats=s_transposed.axes_manager.navigation_axes[0].size,
                    axis=-1,
                ),
                repeats=s_transposed.axes_manager.navigation_axes[1].size,
                axis=1,
            )[:, :, :, np.newaxis]
        )

        ys = Signal1D(
            np.repeat(
                np.repeat(
                    ydrift[:, np.newaxis, np.newaxis],
                    repeats=s_transposed.axes_manager.navigation_axes[0].size,
                    axis=-1,
                ),
                repeats=s_transposed.axes_manager.navigation_axes[1].size,
                axis=1,
            )[:, :, :, np.newaxis]
        )

        registered_data = s_transposed.map(
            _register_drift_2d,
            shift1=xs,
            shift2=ys,
            order=order,
            inplace=False,
            **kwargs,
        )

        registered_data_t = registered_data.transpose(navigation_axes=[-2, -1, -3])
        registered_data_t.set_signal_type("insitu_diffraction")

        return registered_data_t

    def get_g2_2d_kresolved(
        self,
        time_axis: int = 2,
        normalization: str = "split",
        k1bin: int = 1,
        k2bin: int = 1,
        tbin: int = 1,
        resample_time: Optional[Union[int, np.ndarray]] = None,
    ) -> "Correlation2D":
        """
        Calculate k resolved g2 from in situ diffraction signal

        Parameters
        ----------
        time_axis: int
            Index of time axis. Default is 2
        normalization: string, Default is 'split'
            Normalization format for time autocorrelation, 'split' or 'self'
        k1bin: int
            Binning factor for k1 axis
        k2bin: int
            Binning factor for k2 axis
        tbin: int
            Binning factor for t axis
        resample_time: int or np.array, Default is None
            If int, time is resample into log linear with resample_time as
            number of sampling. If array, it is used as resampled time axis
            instead. No resampling is performed if None

        Returns
        ---------
        g2kt : ~hyperspy.api.signals.Signal2D or ~pyxem.signals.Correlation2D
            k resolved time correlation signal
        """
        if time_axis != 2:
            transposed_signal = self.roll_time_axis(time_axis).transpose(
                navigation_axes=[0, 1]
            )
        else:
            transposed_signal = self.transpose(navigation_axes=[0, 1])

        g2kt = transposed_signal.map(
            _g2_2d,
            normalization=normalization,
            k1bin=k1bin,
            k2bin=k2bin,
            tbin=tbin,
            inplace=False,
        )

        if resample_time is not None:
            if isinstance(resample_time, int):
                trs = _get_resample_time(
                    t_size=transposed_signal.axes_manager.signal_axes[-1].size / tbin,
                    dt=transposed_signal.axes_manager.signal_axes[-1].scale * tbin,
                    t_rs_size=resample_time,
                )
                g2rs = g2kt.map(
                    _interpolate_g2_2d,
                    t_rs=trs,
                    dt=transposed_signal.axes_manager.signal_axes[-1].scale * tbin,
                    inplace=False,
                )
                g2rs.set_signal_type("correlation")
                return g2rs
            if (
                isinstance(resample_time, (list, tuple, np.ndarray))
                and len(np.shape(resample_time)) == 1
            ):
                g2rs = g2kt.map(
                    _interpolate_g2_2d,
                    t_rs=resample_time / tbin,
                    dt=transposed_signal.axes_manager.signal_axes[-1].scale * tbin,
                    inplace=False,
                )
                g2rs.set_signal_type("correlation")
                return g2rs
            else:
                raise TypeError("'resample_time' must be int or 1d array")

        g2kt.set_signal_type("correlation")

        return g2kt

    

    def expected_intensity(
            self,
            method: Literal["t", "k", "custom"] = "t",
            custom_axes: Optional[dict] = None,
            chunk_optimize: bool = False,
            center: Literal["mean", "median"] = "mean",
            **kwargs
            ) -> "InSituDiffraction2D":
        """
        Calculate the expected intensity in the time series using the specified method. Signal is transposed so that time axis is the last signal axis (first axis of the signal ndarray)
        
        Parameters
        ----------
        method : Literal["t", "k", "custom"], default="t"
            The method to use for calculating the expected intensity.
            The "t" method calculates the expected intensity at each time and scattering position using the time axis, "k" uses all non-time axes, and "custom" uses the user-specified axes.
        custom_axes : dict | None, default=None
            Custom axes specification for the "custom" method. Should be a dictionary with keys 'full_axis', 'local_axis', and 'local_size' specifying the axes and local size for the background calculation.
            The values for 'full_axis' and 'local_axis' should be integers or lists of integers specifying the axes along which to calculate the background and apply the local filter, respectively. The indexing is the ndarray indexing after the intial transpose of the signal to have only the spatial axes in the navigation axes.
        chunk_optimize : bool, default=False
            Whether to optimize the chunking of the signal for computation after initial transpose of the signal. This can improve performance for large lazy datasets if the chunking is not optimize to iterate over the spatial axes.
        center : Literal["mean", "median"], default="mean"
            The method to use for calculating the expected intensity. "mean" uses the mean, while "median" uses the median. This is passed to the `_bkg_calc` function for background calculation.
        **kwargs : dict
            Additional keyword arguments passed to the scipy.ndimage.uniform_filter function when applying the local uniform filter for background calculation.
            
        Returns
        -------
        InSituDiffraction2D
            The expected intensity signal.
        """
        time_axis = _find_time_axis(self)
        if time_axis != 2:
            s_ = self.roll_time_axis(time_axis)
        else:
            s_ = self

        signal_T = s_.transpose(navigation_axes=[0,1], optimize=chunk_optimize)
        
        if method == 't':
            bkg_T = signal_T.map(_bkg_calc, axis=0, local_axis=None, inplace=False, center=center)

        elif method == 'k':
            bkg_T = signal_T.map(_bkg_calc, axis=[1,2], local_axis=None, inplace=False, center=center)
        elif method == 'custom':
            if custom_axes is None:
                raise ValueError("For 'custom' method, 'custom_axes' must be provided")
            bkg_T = signal_T.map(_bkg_calc, axis=custom_axes.get('full_axis', None), local_axis=custom_axes.get('local_axis', None), local_size=custom_axes.get('local_size', None), inplace=False, center=center, **kwargs)
        else:
            raise ValueError("Method must be one of 't', 'k', or 'custom'")

        bkg = bkg_T.transpose(navigation_axes=[0,1,4])

        return bkg

    def get_TTCF(self) -> "TwoTimeCorrelationFunction":
        """
        Compute the two-time correlation function (TTCF) of the signal.

        Returns
        -------
        TwoTimeCorrelationFunction
            The computed TTCF signal.
        """
        time_axis = _find_time_axis(self)
        tax = self.axes_manager.navigation_axes[time_axis]

        if time_axis != 2:
            signal_T = self.roll_time_axis(time_axis).transpose(navigation_axes=[0,1], optimize=False)
        else:
            signal_T = self.transpose(navigation_axes=[0,1], optimize=False)


        ttcf_signal = signal_T.map(_compute_ttcf, inplace=False)
        for ax in ttcf_signal.axes_manager.signal_axes:
            ax.name = tax.name
            ax.units = tax.units
            ax.offset = tax.offset
            ax.scale = tax.scale
        ttcf_signal.set_signal_type("ttcf")
        return ttcf_signal

    def get_azimuthal_integral2d(self, *args, **kwargs)-> "InSituPolarDiffraction2D":
        """Perform 2D azimuthal integration and return an InSituPolarDiffraction2D signal."""
        # 1. Execute parent Diffraction2D azimuthal integration logic
        res = super().get_azimuthal_integral2d(*args, **kwargs)

        # 2. Handle 'inplace=True' vs returned signal
        if kwargs.get("inplace", False):
            self.set_signal_type("insitu_polar")
            return None

        # 3. Handle returned transformed signal
        if res is not None:
            
            res.set_signal_type("insitu_polar")

        return res
        


class LazyInSituDiffraction2D(LazySignal, InSituDiffraction2D):
    pass
