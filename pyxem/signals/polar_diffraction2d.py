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


from typing import TYPE_CHECKING, Optional, Sequence, Union

from pyxem.utils._typing import Number

import numpy as np
from hyperspy.signals import Signal2D
from hyperspy._signals.lazy import LazySignal
from numpy import rad2deg

from pyxem.signals.common_diffraction import CommonDiffraction

if TYPE_CHECKING:
    from diffsims.simulations import DiffractionSimulation
    from pyxem.signals.correlation1d import Correlation1D
    from pyxem.signals.correlation2d import Correlation2D
    from pyxem.signals.power2d import Power2D
from pyxem.utils._correlations import _correlation, _power, _pearson_correlation
from pyxem.utils._deprecated import deprecated
from pyxem.utils.indexation_utils import (
    _mixed_matching_lib_to_polar,
    _get_integrated_polar_templates,
    _norm_rows,
    _get_max_n,
)

from pyxem.utils._background_subtraction import (
    _polar_subtract_radial_median,
    _polar_subtract_radial_percentile,
)


def _circular_span(columns: np.ndarray) -> tuple[int, int]:
    """Find the shortest circular interval that contains every True element.

    Parameters
    ----------
    columns : np.ndarray
        A non-empty 1D boolean array, treated as periodic.

    Returns
    -------
    start, width : int
        The first index of the interval and its length. The interval covers
        ``start, start + 1, ..., start + width - 1``, modulo ``columns.size``.
    """
    n = columns.size
    idx = np.flatnonzero(columns)
    # the number of False elements between each True element and the next one
    gaps = np.diff(np.append(idx, idx[0] + n)) - 1
    largest = np.argmax(gaps)
    start = idx[(largest + 1) % idx.size]
    return int(start), int(n - gaps[largest])


def _take_azimuthal(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Select ``indices`` along the last (azimuthal) axis."""
    return data[..., indices]


class PolarDiffraction2D(CommonDiffraction, Signal2D):
    """Signal class for two-dimensional diffraction data in polar coordinates.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`
    """

    _signal_type = "polar_diffraction"

    def get_angular_correlation(
        self,
        mask: Optional[Union[np.ndarray, Signal2D]] = None,
        normalize: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> Optional["Correlation2D"]:
        r"""Calculate the angular auto-correlation function in the form of a Signal2D class.

        The angular correlation measures the angular symmetry by computing the self or auto
        correlation. The equation being calculated is
        $ C(\phi,k,n)= \frac{ <I(\theta,k,n)*I(\theta+\phi,k,n)>_\theta-<I(\theta,k,n)>^2}{<I(\theta,k,n)>^2}$

        Parameters
        ----------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If the mask
            is a BaseSignal than it is iterated with the polar signal
        normalize: bool
            Normalize the radial correlation by the average value at some radius.
        kwargs: dict
            Any additional options for the :meth:`hyperspy.api.signals.BaseSignal.map` function
        inplace: bool
            From :meth:`hyperspy.api.signals.BaseSignal.map`. inplace=True means the signal is
            overwritten.

        Returns
        -------
        correlation: ~hyperspy.api.signals.Signal2D
            The radial correlation for the signal2D, when inplace is False,
            otherwise None

        """
        correlation = self.map(
            _correlation,
            axis=1,
            mask=mask,
            normalize=normalize,
            inplace=inplace,
            silence_warnings=True,
            **kwargs,
        )
        s = self if inplace else correlation
        theta_axis = s.axes_manager.signal_axes[0]

        theta_axis.name = r"Angular Correlation, $ \Delta \Theta$"
        theta_axis.offset = 0

        s.set_signal_type("correlation")
        return correlation

    def get_angular_power(
        self,
        mask: Optional[Union[np.ndarray, Signal2D]] = None,
        normalize: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> Optional["Power2D"]:
        """Calculate the power spectrum of the angular auto-correlation function
        in the form of a Signal2D class.

        This gives the fourier decomposition of the radial correlation. Due to
        nyquist sampling the number of fourier coefficients will be equal to the
        angular range.

        Parameters
        ----------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If the mask
            is a BaseSignal than it is iterated with the polar signal
         normalize: bool
             Normalize the radial correlation by the average value at some radius.
        inplace: bool
            From :meth:`hyperspy.api.signals.BaseSignal.map` inplace=True means the signal is
            overwritten.
        kwargs: dict
            Any additional options for the :meth:`hyperspy.api.signals.BaseSignal.map` function

        Returns
        -------
        power: ~hyperspy.api.signals.Signal2D
            The power spectrum of the Signal2D, when inplace is False, otherwise
            return None
        """
        power = self.map(
            _power,
            axis=1,
            mask=mask,
            normalize=normalize,
            inplace=inplace,
            silence_warnings=True,
            **kwargs,
        )

        s = self if inplace else power
        s.set_signal_type("power")
        fourier_axis = s.axes_manager.signal_axes[0]

        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.scale = 1

        return power

    def get_full_pearson_correlation(
        self,
        mask: Optional[np.ndarray] = None,
        krange: Optional[Sequence[Number]] = None,
        inplace: bool = False,
        **kwargs,
    ) -> Optional["Correlation1D"]:
        """Calculate the fully convolved pearson rotational correlation in the
        form of a Signal1D class.

        Parameters
        ----------
        mask: ~numpy.ndarray
            A bool mask of values to ignore of shape equal to the signal shape.
            True for elements masked, False for elements unmasked
        krange: tuple of int or float
            The range of k values for segment correlation. If type is ``int``,
            the value is taken as the axis index. If type is ``float`` the
            value is in corresponding unit.
            If None (default), use the entire pattern .
        inplace: bool
            From :meth:`~hyperspy.api.signals.BaseSignal.map` inplace=True means the signal is
            overwritten.
        kwargs: dict
            Any additional options for the :meth:`~hyperspy.api.signals.BaseSignal.map` function.
        Returns
        -------
        correlation: ~hyperspy.api.signals.Signal1D,
            The pearson rotational correlation when inplace is False, otherwise
            return None
        """
        # placeholder to handle inplace playing well with cropping and mapping
        s_ = self
        if krange is not None:
            if inplace:
                s_.crop(-1, start=krange[0], end=krange[1])
            else:
                s_ = self.isig[:, krange[0] : krange[1]]

            if mask is not None:
                mask = Signal2D(mask)
                # When float krange are used, axis calibration is required
                mask.axes_manager[-1].scale = self.axes_manager[-1].scale
                mask.axes_manager[-1].offset = self.axes_manager[-1].offset
                mask.crop(-1, start=krange[0], end=krange[1])

        correlation = s_.map(
            _pearson_correlation,
            mask=mask,
            inplace=inplace,
            silence_warnings=True,
            **kwargs,
        )

        s = s_ if inplace else correlation
        s.set_signal_type("correlation")

        rho_axis = s.axes_manager.signal_axes[0]
        rho_axis.name = r"Correlation Angle, $ \Delta \Theta$"
        rho_axis.offset = 0
        rho_axis.units = "rad"
        rho_axis.scale = self.axes_manager[-2].scale

        return correlation

    @deprecated(
        since="0.15",
        removal="1.0.0",
        alternative="pyxem.signals.PolarDiffraction2D.get_pearson_correlation",
    )
    def get_pearson_correlation(self, **kwargs):
        return self.get_full_pearson_correlation(**kwargs)

    def get_resolved_pearson_correlation(
        self,
        mask: Optional[np.ndarray] = None,
        krange: Optional[Sequence[Number]] = None,
        inplace: bool = False,
        **kwargs,
    ) -> Optional["Correlation2D"]:
        """
        Calculate the pearson rotational correlation with k resolution in
        the form of a Signal2D class.

        Parameters
        ----------
        mask: Numpy array
            A bool mask of values to ignore of shape equal to the signal shape.
            True for elements masked, False for elements unmasked
        krange: tuple of int or float
            The range of k values for segment correlation. If type is ``int``,
            the value is taken as the axis index. If type is ``float`` the
            value is in corresponding unit.
            If None (default), use the entire pattern .
        inplace: bool
            From :meth:`~hyperspy.api.signals.BaseSignal.map` inplace=True means the signal is
            overwritten.
        kwargs: dict
            Any additional options for the :meth:`~hyperspy.api.signals.BaseSignal.map` function


        Returns
        -------
        correlation: hyperspy.signals.Signal2D
            The pearson rotational correlation when inplace is False, otherwise
            return None
        """
        # placeholder to handle inplace playing well with cropping and mapping
        s_ = self
        if krange is not None:
            if inplace:
                s_.crop(-1, start=krange[0], end=krange[1])
            else:
                s_ = self.isig[:, krange[0] : krange[1]]

            if mask is not None:
                mask = Signal2D(mask)
                # When float krange are used, axis calibration is required
                mask.axes_manager[-1].scale = self.axes_manager[-1].scale
                mask.axes_manager[-1].offset = self.axes_manager[-1].offset
                mask.crop(-1, start=krange[0], end=krange[1])

        correlation = s_.map(
            _pearson_correlation,
            mask=mask,
            mode="kresolved",
            inplace=inplace,
            silence_warnings=True,
            **kwargs,
        )

        s = s_ if inplace else correlation
        s.set_signal_type("correlation")

        rho_axis = s.axes_manager.signal_axes[0]
        rho_axis.name = r"Correlation Angle, $ \Delta \Theta$"
        rho_axis.offset = 0
        rho_axis.units = "rad"
        rho_axis.scale = self.axes_manager[-2].scale

        k_axis = s.axes_manager.signal_axes[1]
        k_axis.name = "k"
        if krange is not None:
            k_axis.offset = krange[0]
        else:
            k_axis.offset = self.axes_manager[-1].offset
        k_axis.units = r"$\AA^{-1}$"
        k_axis.scale = self.axes_manager[-1].scale

        return correlation

    def subtract_diffraction_background(
        self, method: str = "radial median", inplace: bool = False, **kwargs
    ) -> Optional["PolarDiffraction2D"]:
        """
        Background subtraction of the diffraction data.

        Parameters
        ----------
        method : str, optional
            'radial median', 'radial percentile'
            Default 'radial median'.
            For 'radial median' no extra parameters are necessary.
            For 'radial percentile' the 'percentile' argument decides
            which percentile to substract.
        **kwargs :
                To be passed to the chosen method.

        Returns
        -------
        s : pyxem.signals.PolarDiffraction2D, pyxem.signals.LazyPolarDiffraction2D

        """
        method_dict = {
            "radial median": _polar_subtract_radial_median,
            "radial percentile": _polar_subtract_radial_percentile,
        }
        if method not in method_dict:
            raise NotImplementedError(
                f"The method specified, '{method}',"
                f" is not implemented.  The different methods are:  "
                f"{', '.join(method_dict.keys())}."
            )
        subtraction_function = method_dict[method]

        return self.map(
            subtraction_function,
            inplace=inplace,
            output_dtype=self.data.dtype,
            output_signal_size=self.axes_manager._signal_shape_in_array,
            silence_warnings=True,
            **kwargs,
        )

    def get_orientation(
        self,
        simulation: "DiffractionSimulation",
        n_keep: Optional[int] = None,
        frac_keep: float = 0.1,
        n_best: int = 1,
        normalize_templates: bool = True,
        **kwargs,
    ) -> Signal2D:
        """
        Match the orientation with some simulated diffraction patterns using
        an accelerated orientation mapping algorithm.
        The details of the algorithm are described in the paper:
        "Free, flexible and fast: Orientation mapping using the multi-core and
        GPU-accelerated template matching capabilities in the python-based open
        source 4D-STEM analysis toolbox Pyxem"

        Parameters
        ----------
        simulation : DiffractionSimulation
            The diffraction simulation object to use for indexing.
        n_keep : int
            The number of orientations to keep for each diffraction pattern.
        frac_keep : float
            The fraction of the best matching orientations to keep.
        n_best : int
            The number of best matching orientations to return. If n_best == -1 all of the
            orientations and correlations are returned.
        normalize_templates : bool
            Normalize the templates to the same intensity.
        kwargs : dict
            Any additional options for the :meth:`~hyperspy.api.signals.BaseSignal.map` function.

        Returns
        -------
        orientation : ~hyperspy.api.signals.BaseSignal
            A signal with the orientation at each navigation position.

        Notes
        -----
            If :code:`n_best` exceeds :code:`n_keep` or :code:`frac_keep * N` for :code:`N` simulations,
            then full correlation is performed on :code:`n_best` simulations instead. This ensures the
            output contains :code:`n_best` simulations.

            A gamma correction is often applied to the diffraction patterns. A good value
            to start with is the square root (gamma=0.5) of the diffraction patterns to
            increase the intensity of the low intensity reflections and decrease the
            intensity of the high intensity reflections. This can be applied via:

            `s_gamma = s**0.5`

            In most cases gamma < 1 See :cite:`pyxemorientationmapping2022` for more information.
            Additionally, subtracting a small value can sometimes be helpful as it penalizes
            diffraction patterns which do not have the full compliment of simulated diffraction
            vectors.

        """
        (
            r_templates,
            theta_templates,
            intensities_templates,
        ) = simulation.polar_flatten_simulations(
            radial_axes=self.axes_manager.signal_axes[1].axis,
            azimuthal_axes=self.axes_manager.signal_axes[0].axis,
        )
        radius = self.axes_manager.signal_axes[1].size  # number radial pixels
        integrated_templates = _get_integrated_polar_templates(
            radius, r_templates, intensities_templates, normalize_templates
        )
        if normalize_templates:
            intensities_templates = _norm_rows(intensities_templates)

        if n_best == -1:
            n_best = r_templates.shape[0]

        max_n = _get_max_n(N=r_templates.shape[0], n_keep=n_keep, frac_keep=frac_keep)
        if max_n < n_best:
            # n_keep takes precedence over frac_keep, so this ensures we get n_best simulations in the result
            n_keep = n_best
        orientation = self.map(
            _mixed_matching_lib_to_polar,
            integrated_templates=integrated_templates,
            r_templates=r_templates,
            theta_templates=theta_templates,
            intensities_templates=intensities_templates,
            n_keep=n_keep,
            frac_keep=frac_keep,
            n_best=n_best,
            inplace=False,
            transpose=True,
            output_signal_size=(n_best, 4),
            output_dtype=float,
            silence_warnings=True,
            **kwargs,
        )

        # Translate in-plane rotation from index to degrees
        # by using the calibration of the axis
        def rotation_index_to_degrees(data, axis):
            data = data.copy()
            ind = data[:, 2].astype(int)
            data[:, 2] = rad2deg(axis[ind])
            return data

        orientation.axes_manager.signal_axes[0].name = "n-best"
        orientation.axes_manager.signal_axes[1].name = "columns"

        orientation.map(
            rotation_index_to_degrees, axis=self.axes_manager.signal_axes[0].axis
        )

        orientation.set_signal_type("orientation_map")
        orientation.simulation = simulation
        orientation.column_names = ["index", "correlation", "rotation", "factor"]
        orientation.units = ["a.u.", "a.u.", "deg", "a.u."]
        return orientation

    def get_beam_stop(
        self,
        start_point: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Retrieve the beam stop mask for the averaged diffraction pattern.

        The averaged pattern is edge-filtered and thresholded, and the beam stop
        is the region enclosed by those edges that contains ``start_point``. The
        azimuthal axis is treated as periodic, so a beam stop that crosses the
        0/360 degree seam is returned as a single region.

        Parameters
        ----------
        start_point : list[int], optional
            The ``[radial, azimuthal]`` pixel index of a point inside the beam
            stop. It must not lie on an edge of the averaged pattern. Defaults
            to the centre of the pattern.

        Returns
        -------
        np.ndarray
            A boolean array with the shape of the signal, ``(radial, azimuthal)``,
            marking the beam stop region.

        Raises
        ------
        ValueError
            If ``start_point`` is outside the pattern, or lies on an edge and so
            does not belong to any region.
        """
        from scipy.ndimage import binary_dilation, label
        from skimage.filters import sobel, threshold_otsu
        from skimage.morphology import disk

        avg_dp = self.mean(axis="nav").data
        # check if avg_dp is dask array and compute if necessary
        if hasattr(avg_dp, "compute"):
            avg_dp = avg_dp.compute()
        n_rad, n_azi = avg_dp.shape

        if start_point is None:
            start_point = [n_rad // 2, n_azi // 2]
        start_rad, start_azi = start_point
        if not (0 <= start_rad < n_rad and 0 <= start_azi < n_azi):
            raise ValueError(
                f"`start_point` {list(start_point)} is outside the pattern, "
                f"which has shape {(n_rad, n_azi)}."
            )

        # The azimuthal axis is periodic. Rolling the start point to the centre
        # keeps a beam stop that crosses the 0/360 degree seam in one piece.
        shift = n_azi // 2 - start_azi
        avg_dp = np.roll(avg_dp, shift, axis=1)

        edges = sobel(avg_dp)
        thresh = threshold_otsu(edges)
        binary = binary_dilation(edges > thresh, structure=disk(3))
        labeled_inv, _ = label(np.logical_not(binary))
        start_label = labeled_inv[start_rad, n_azi // 2]
        if start_label == 0:
            raise ValueError(
                f"`start_point` {list(start_point)} lies on an edge of the averaged "
                "pattern, so it does not belong to a region. Choose a point inside "
                "the beam stop."
            )
        beam_stop_mask = binary_dilation(labeled_inv == start_label, structure=disk(3))
        return np.roll(beam_stop_mask, -shift, axis=1)

    def remove_beam_stop_area(
        self,
        beam_stop_mask: Optional[np.ndarray] = None,
        start_point: Optional[list[int]] = None,
        exclude_angle: Optional[float] = None,
    ) -> "PolarDiffraction2D":
        """
        Remove the beam stop area from the diffraction pattern using the provided mask.

        The azimuthal axis is treated as periodic, so the beam stop and the
        excluded margin around it may cross the 0/360 degree seam. The remaining
        data starts at the first azimuthal pixel after the removed area, so the
        azimuthal axis is shortened and also rolled.

        Parameters
        ----------
        beam_stop_mask : np.ndarray, optional
            The mask indicating the beam stop region, with the shape of the
            signal ``(radial, azimuthal)``. If None, it is computed with
            :meth:`get_beam_stop`.
        start_point : list[int], optional
            The ``[radial, azimuthal]`` starting point for beam stop detection if
            the mask needs to be computed.
        exclude_angle : float, optional
            The total angular range to exclude around the beam stop, in degrees.
            Data on each side of the beam stop within half of this angular range
            will be excluded. 8 extra degrees are added to the total angular
            range to ensure sufficient exclusion. Defaults to None, meaning only
            the default 8 degrees will be excluded.

        Returns
        -------
        PolarDiffraction2D
            The polar diffraction signal with the beam stop area removed.

        Raises
        ------
        ValueError
            If ``beam_stop_mask`` has the wrong shape or no beam stop pixels, or
            if the area to remove covers the whole azimuthal range.
        """
        if exclude_angle is None:
            exclude_angle = 8.0
        else:
            exclude_angle += 8.0

        azi_size = self.axes_manager.signal_axes[0].size
        exclude_pix = int(exclude_angle / 360.0 * azi_size)
        half_exclude_pix = (exclude_pix + 1) // 2

        if beam_stop_mask is None:
            beam_stop_mask = self.get_beam_stop(start_point=start_point)
        beam_stop_mask = np.asarray(beam_stop_mask)
        if beam_stop_mask.shape != self.data.shape[-2:]:
            raise ValueError(
                f"`beam_stop_mask` has shape {beam_stop_mask.shape} but must have "
                f"the shape of the signal, {self.data.shape[-2:]}."
            )
        bs_columns = np.any(beam_stop_mask, axis=0)
        if not bs_columns.any():
            raise ValueError("`beam_stop_mask` does not contain any beam stop pixels.")

        bs_start, bs_width = _circular_span(bs_columns)
        n_excluded = bs_width + 2 * half_exclude_pix
        if n_excluded >= azi_size:
            raise ValueError(
                "The beam stop and the excluded angle cover the whole azimuthal range."
            )
        first_kept = bs_start + bs_width + half_exclude_pix
        keep = (first_kept + np.arange(azi_size - n_excluded)) % azi_size

        return self.map(_take_azimuthal, indices=keep, inplace=False)


class LazyPolarDiffraction2D(LazySignal, PolarDiffraction2D):
    pass
