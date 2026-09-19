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

import numpy as np
import pytest

from pyxem.signals import InSituPolarDiffraction2D, LazyInSituPolarDiffraction2D

ALL_METHODS = ["tazi_local", "tazi", "azi", "azi_local", "t", "k"]


class TestExpectedPolarIntensity:
    n_azi = 36  # 10 degrees per azimuthal pixel

    @pytest.fixture
    def insitu_data(self):
        # (time, y, x, k, azimuth), with non-default time axis metadata
        data = np.random.default_rng(seed=0).random((5, 2, 3, 4, self.n_azi)) + 1.0
        s = InSituPolarDiffraction2D(data)
        time_axis = s.axes_manager.navigation_axes[2]
        time_axis.name, time_axis.scale, time_axis.units = "Time", 0.5, "s"
        return s

    @staticmethod
    def _sliding_mean(x, window):
        """Mean over `window` neighbouring azimuthal pixels.

        Only the pixels whose whole window lies inside the array are returned, so
        the result does not depend on how the azimuthal boundary is handled.
        """
        return np.stack(
            [
                x[..., i : i + window].mean(axis=-1)
                for i in range(x.shape[-1] - window + 1)
            ],
            axis=-1,
        )

    def test_output_type_and_axes(self, insitu_data):
        out = insitu_data.expected_polar_intensity(method="t")
        assert isinstance(out, InSituPolarDiffraction2D)
        assert out._signal_type == "insitu_polar"
        assert out.data.shape == insitu_data.data.shape
        assert out.axes_manager.navigation_shape == (
            insitu_data.axes_manager.navigation_shape
        )
        assert out.axes_manager.signal_shape == insitu_data.axes_manager.signal_shape
        time_axis = out.axes_manager.navigation_axes[2]
        assert (time_axis.name, time_axis.scale, time_axis.units) == ("Time", 0.5, "s")

    @pytest.mark.parametrize(
        "method, axes",
        [
            ("t", (0,)),  # time
            ("k", (-2, -1)),  # radial and azimuthal
            ("tazi", (0, -1)),  # time and azimuthal
            ("azi", (-1,)),  # azimuthal
        ],
    )
    def test_averaged_methods(self, insitu_data, method, axes):
        # the expected intensity is the mean over `axes`, repeated along them
        data = insitu_data.data
        expected = np.broadcast_to(data.mean(axis=axes, keepdims=True), data.shape)
        out = insitu_data.expected_polar_intensity(method=method)
        np.testing.assert_allclose(out.data, expected)

    @pytest.mark.parametrize(
        "local_azi_angle, window",
        [
            (35.0, 3),  # 3.5 pixels -> 3
            (60.0, 7),  # 6 pixels is even -> 7
            (65.0, 7),  # 6.5 pixels -> 6, even -> 7
            (125.0, 13),  # 12.5 pixels -> 12, even -> 13
        ],
    )
    @pytest.mark.parametrize("method", ["tazi_local", "azi_local"])
    def test_local_methods(self, insitu_data, method, local_azi_angle, window):
        data = insitu_data.data
        # "tazi_local" averages over time first, "azi_local" does not
        base = data.mean(axis=0, keepdims=True) if method == "tazi_local" else data
        base = np.broadcast_to(base, data.shape)
        out = insitu_data.expected_polar_intensity(
            method=method, local_azi_angle=local_azi_angle
        )
        half = window // 2
        np.testing.assert_allclose(
            out.data[..., half:-half], self._sliding_mean(base, window)
        )

    @pytest.mark.parametrize("local_azi_angle, window", [(35.0, 3), (65.0, 7)])
    @pytest.mark.parametrize("method", ["tazi_local", "azi_local"])
    def test_local_methods_are_periodic(
        self, insitu_data, method, local_azi_angle, window
    ):
        # every azimuthal pixel, including the first and last, is averaged with its
        # neighbours on both sides of the 0/360 degree seam
        data = insitu_data.data
        base = data.mean(axis=0, keepdims=True) if method == "tazi_local" else data
        base = np.broadcast_to(base, data.shape)
        half = window // 2
        padded = np.concatenate([base[..., -half:], base, base[..., :half]], axis=-1)
        out = insitu_data.expected_polar_intensity(
            method=method, local_azi_angle=local_azi_angle
        )
        np.testing.assert_allclose(out.data, self._sliding_mean(padded, window))

    @pytest.mark.parametrize("method", ["tazi_local", "azi_local"])
    def test_local_methods_spread_an_impulse_across_the_seam(self, method):
        data = np.zeros((5, 2, 3, 4, self.n_azi))
        data[..., 0] = 1.0
        s = InSituPolarDiffraction2D(data)
        s.axes_manager.navigation_axes[2].name = "Time"
        out = s.expected_polar_intensity(method=method, local_azi_angle=65.0)
        # a window of 7 pixels: 3 either side of azimuthal pixel 0, wrapping round
        expected = np.zeros(self.n_azi)
        expected[[-3, -2, -1, 0, 1, 2, 3]] = 1 / 7
        np.testing.assert_allclose(out.data, np.broadcast_to(expected, out.data.shape))

    @pytest.mark.parametrize("shift", [1, 5, 17])
    @pytest.mark.parametrize("method", ["tazi_local", "azi_local"])
    def test_local_methods_commute_with_azimuthal_rotation(
        self, insitu_data, method, shift
    ):
        # rotating the pattern rotates the result, wherever the seam falls
        rotated = insitu_data.deepcopy()
        rotated.data = np.roll(insitu_data.data, shift, axis=-1)
        np.testing.assert_allclose(
            rotated.expected_polar_intensity(method=method).data,
            np.roll(
                insitu_data.expected_polar_intensity(method=method).data,
                shift,
                axis=-1,
            ),
        )

    def test_default_arguments(self, insitu_data):
        default = insitu_data.expected_polar_intensity()
        explicit = insitu_data.expected_polar_intensity(
            method="tazi_local",
            local_azi_angle=60.0,
            chunk_optimize=False,
            center="mean",
        )
        np.testing.assert_array_equal(default.data, explicit.data)

    @pytest.mark.parametrize("method", ["t", "tazi", "tazi_local"])
    def test_center_median(self, method):
        # one outlier in time moves the mean but not the median
        data = np.zeros((5, 2, 3, 4, self.n_azi))
        data[4] = 10.0
        s = InSituPolarDiffraction2D(data)
        s.axes_manager.navigation_axes[2].name = "Time"
        mean = s.expected_polar_intensity(method=method)
        median = s.expected_polar_intensity(method=method, center="median")
        np.testing.assert_allclose(mean.data, 2.0)
        np.testing.assert_allclose(median.data, 0.0)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_time_axis_not_at_index_2(self, insitu_data, method):
        rolled = insitu_data.rollaxis(2, 0)
        assert rolled.axes_manager.navigation_axes[0].name == "Time"
        np.testing.assert_allclose(
            rolled.expected_polar_intensity(method=method).data,
            insitu_data.expected_polar_intensity(method=method).data,
        )

    @pytest.mark.parametrize("chunk_optimize", [False, True])
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_lazy_matches_eager(self, insitu_data, method, chunk_optimize):
        eager = insitu_data.expected_polar_intensity(method=method)
        lazy = insitu_data.as_lazy().expected_polar_intensity(
            method=method, chunk_optimize=chunk_optimize
        )
        assert isinstance(lazy, LazyInSituPolarDiffraction2D)
        lazy.compute()
        np.testing.assert_allclose(lazy.data, eager.data)

    def test_chunk_optimize_is_forwarded(self, insitu_data):
        lazy = insitu_data.as_lazy()
        lazy.rechunk((1, 1, 1, -1, -1))
        plain = lazy.expected_polar_intensity(method="t", chunk_optimize=False)
        optimized = lazy.expected_polar_intensity(method="t", chunk_optimize=True)
        # optimizing joins the real-space chunks (y: 2 pixels, x: 3 pixels)
        assert plain.data.chunks[1:3] == ((1, 1), (1, 1, 1))
        assert optimized.data.chunks[1:3] == ((2,), (3,))

    def test_input_not_modified(self, insitu_data):
        original = insitu_data.data.copy()
        for method in ALL_METHODS:
            insitu_data.expected_polar_intensity(method=method)
        np.testing.assert_array_equal(insitu_data.data, original)
        assert type(insitu_data) is InSituPolarDiffraction2D

    def test_invalid_method(self, insitu_data):
        with pytest.raises(ValueError, match="Method must be one of"):
            insitu_data.expected_polar_intensity(method="not_a_method")

    def test_missing_time_axis(self):
        s = InSituPolarDiffraction2D(np.ones((5, 2, 3, 4, self.n_azi)))
        with pytest.raises(ValueError, match="Time axis not found"):
            s.expected_polar_intensity()
