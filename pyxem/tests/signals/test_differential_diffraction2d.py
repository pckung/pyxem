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

import pytest
import numpy as np
from numpy.random import default_rng

from pyxem.signals import (
    Diffraction2D,
    DifferentialDiffraction2D,
    LazyDifferentialDiffraction2D,
)


class TestGetDifferentialDiffraction:
    @pytest.fixture()
    def signal(self):
        # strictly positive, so the log of the ratio to the background is defined
        data = default_rng(seed=0).random((7, 8, 5, 6)) + 1.0
        return Diffraction2D(data)

    @staticmethod
    def _expected(data, inner, outer):
        """Brute-force log(data / annulus mean) over the navigation axes.

        The annulus is every pixel whose largest per-axis distance from the
        centre is in (inner, outer], with zero padding outside the scan.
        """
        ny, nx = data.shape[:2]
        padded = np.pad(data, ((outer, outer), (outer, outer), (0, 0), (0, 0)))
        dy, dx = np.mgrid[-outer : outer + 1, -outer : outer + 1]
        in_annulus = (np.maximum(abs(dy), abs(dx)) > inner) & (
            np.maximum(abs(dy), abs(dx)) <= outer
        )
        expected = np.empty_like(data)
        for y in range(ny):
            for x in range(nx):
                window = padded[y : y + 2 * outer + 1, x : x + 2 * outer + 1]
                bkg = window[in_annulus].mean(axis=0)
                expected[y, x] = np.log(data[y, x] / bkg)
        return expected

    def test_output_type_and_axes(self, signal):
        out = signal.get_differential_diffraction()
        assert isinstance(out, DifferentialDiffraction2D)
        assert out.metadata.Signal.signal_type == "differential_diffraction"
        assert out.data.shape == signal.data.shape
        assert out.axes_manager.navigation_shape == signal.axes_manager.navigation_shape
        assert out.axes_manager.signal_shape == signal.axes_manager.signal_shape
        assert np.isfinite(out.data).all()

    def test_input_not_modified(self, signal):
        original = signal.data.copy()
        signal.get_differential_diffraction()
        np.testing.assert_array_equal(signal.data, original)
        assert type(signal) is Diffraction2D

    def test_constant_signal_gives_zero(self):
        # the background of a constant signal is that constant, so log(1) == 0
        s = Diffraction2D(np.full((6, 7, 4, 5), 3.0))
        out = s.get_differential_diffraction(inner_cutoff=1, outer_cutoff=2)
        np.testing.assert_allclose(out.data, 0.0, atol=1e-12)

    @pytest.mark.parametrize("inner, outer", [(1, 2), (2, 3), (0, 2)])
    def test_cutoffs_are_radii(self, signal, inner, outer):
        # "constant" mode zero-pads, which the brute-force reference reproduces
        out = signal.get_differential_diffraction(
            inner_cutoff=inner, outer_cutoff=outer, mode="constant"
        )
        np.testing.assert_allclose(
            out.data, self._expected(signal.data, inner, outer), rtol=1e-8
        )

    def test_mode_is_used(self, signal):
        mirror = signal.get_differential_diffraction(1, 2, mode="mirror")
        constant = signal.get_differential_diffraction(1, 2, mode="constant")
        assert not np.allclose(mirror.data, constant.data)
        # away from the scan edges the border handling does not matter
        np.testing.assert_allclose(
            mirror.data[2:-2, 2:-2], constant.data[2:-2, 2:-2], rtol=1e-8
        )

    def test_default_arguments(self, signal):
        default = signal.get_differential_diffraction()
        explicit = signal.get_differential_diffraction(
            inner_cutoff=3, outer_cutoff=7, mode="mirror"
        )
        np.testing.assert_array_equal(default.data, explicit.data)

    def test_lazy(self, signal):
        eager = signal.get_differential_diffraction(inner_cutoff=1, outer_cutoff=2)
        lazy = signal.as_lazy().get_differential_diffraction(
            inner_cutoff=1, outer_cutoff=2
        )
        assert isinstance(lazy, LazyDifferentialDiffraction2D)
        lazy.compute()
        np.testing.assert_allclose(lazy.data, eager.data)


class TestDifferentialDiffraction2DFluctuationAndCorrelationMaps:
    @pytest.fixture()
    def signal(self):
        data = default_rng(seed=0).random((4, 5, 3, 3))
        return DifferentialDiffraction2D(data)

    def test_fluctuation_map_output_type_and_axes(self, signal):
        out = signal.get_fluctuation_map()
        assert isinstance(out, DifferentialDiffraction2D)
        assert out.axes_manager.navigation_shape == ()
        assert out.axes_manager.signal_shape == signal.axes_manager.signal_shape

    def test_fluctuation_map_values(self, signal):
        out = signal.get_fluctuation_map()
        expected = np.var(signal.data, axis=(0, 1))
        np.testing.assert_allclose(out.data, expected)

    def test_fluctuation_map_input_not_modified(self, signal):
        original = signal.data.copy()
        signal.get_fluctuation_map()
        np.testing.assert_array_equal(signal.data, original)

    def test_fluctuation_map_lazy(self, signal):
        eager = signal.get_fluctuation_map()
        lazy = signal.as_lazy().get_fluctuation_map()
        assert isinstance(lazy, LazyDifferentialDiffraction2D)
        lazy.compute()
        np.testing.assert_allclose(lazy.data, eager.data)

    @staticmethod
    def _expected_correlation_map(data, ref_mask):
        ref_vdf = np.nanmean(data[:, :, ref_mask], axis=-1)
        ny, nx = data.shape[-2:]
        expected = np.empty((ny, nx))
        for i in range(ny):
            for j in range(nx):
                expected[i, j] = np.corrcoef(
                    data[:, :, i, j].flatten(), ref_vdf.flatten()
                )[0, 1]
        return expected

    def test_correlation_map_output_type_and_axes(self, signal):
        ref_mask = np.zeros(signal.axes_manager.signal_shape, dtype=bool)
        ref_mask[0, 0] = True
        out = signal.get_correlation_map(ref_mask)
        assert isinstance(out, DifferentialDiffraction2D)
        assert out.axes_manager.navigation_shape == ()
        assert out.axes_manager.signal_shape == signal.axes_manager.signal_shape

    def test_correlation_map_values(self, signal):
        ref_mask = np.zeros(signal.axes_manager.signal_shape, dtype=bool)
        ref_mask[0, 0] = True
        ref_mask[1, 1] = True
        out = signal.get_correlation_map(ref_mask)
        expected = self._expected_correlation_map(signal.data, ref_mask)
        np.testing.assert_allclose(out.data, expected)

    def test_correlation_map_perfect_self_correlation(self, signal):
        # correlating the mask's own pixel against itself gives 1.0
        ref_mask = np.zeros(signal.axes_manager.signal_shape, dtype=bool)
        ref_mask[0, 0] = True
        out = signal.get_correlation_map(ref_mask)
        np.testing.assert_allclose(out.data[0, 0], 1.0)

    def test_correlation_map_input_not_modified(self, signal):
        ref_mask = np.zeros(signal.axes_manager.signal_shape, dtype=bool)
        ref_mask[0, 0] = True
        original = signal.data.copy()
        signal.get_correlation_map(ref_mask)
        np.testing.assert_array_equal(signal.data, original)

    def test_correlation_map_lazy(self, signal):
        ref_mask = np.zeros(signal.axes_manager.signal_shape, dtype=bool)
        ref_mask[0, 0] = True
        eager = signal.get_correlation_map(ref_mask)
        lazy = signal.as_lazy().get_correlation_map(ref_mask)
        assert isinstance(lazy, LazyDifferentialDiffraction2D)
        lazy.compute()
        np.testing.assert_allclose(lazy.data, eager.data)
