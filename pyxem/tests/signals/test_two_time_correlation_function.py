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

from pyxem.signals import (
    LazyOneTimeCorrelationFunction,
    OneTimeCorrelationFunction,
    TwoTimeCorrelationFunction,
)
from pyxem.utils._insitu import _ttcf_2_c2, _ttcf_2_g2

NT = 12  # number of times
WINDOW = 5  # delay times in a c2 window


def _base():
    """ttcf[i, j] = i**2 + j, so every diagonal and every c2 window is distinct."""
    i, j = np.mgrid[0:NT, 0:NT]
    return (i * i + j).astype(float)


def _with_time_axes(signal):
    for axis in signal.axes_manager.signal_axes:
        axis.name, axis.scale, axis.units = "Time", 0.5, "s"
    return signal


@pytest.fixture
def ttcf():
    # (y, x, t1, t2), with a different offset at each of the 2 x 3 real-space positions
    offsets = 1000.0 * np.arange(6).reshape(2, 3, 1, 1)
    return _with_time_axes(TwoTimeCorrelationFunction(_base() + offsets))


@pytest.fixture
def ttcf_no_navigation():
    return _with_time_axes(TwoTimeCorrelationFunction(_base()))


class TestGetG2:
    def test_values(self, ttcf):
        # g2(k) is the mean of the k-th diagonal, at each real-space position
        g2 = ttcf.get_g2()
        data = ttcf.data
        expected = np.array(
            [
                [
                    [
                        np.mean([data[y, x, i, i + k] for i in range(NT - k)])
                        for k in range(NT)
                    ]
                    for x in range(3)
                ]
                for y in range(2)
            ]
        )
        assert g2.data.shape == (2, 3, NT)
        np.testing.assert_allclose(g2.data, expected)

    def test_type_and_axes(self, ttcf):
        g2 = ttcf.get_g2()
        assert isinstance(g2, OneTimeCorrelationFunction)
        assert g2._signal_type == "otcf"
        assert g2.axes_manager.navigation_shape == ttcf.axes_manager.navigation_shape
        time_axis = g2.axes_manager.signal_axes[0]
        assert (time_axis.name, time_axis.scale, time_axis.units) == ("Time", 0.5, "s")

    def test_ignores_nan(self, ttcf):
        # ttcf[3, 5] lies on the k = 2 diagonal
        ttcf.data[..., 3, 5] = np.nan
        g2 = ttcf.get_g2()
        assert np.isfinite(g2.data).all()
        expected = np.mean([_base()[i, i + 2] for i in range(NT - 2) if i != 3])
        np.testing.assert_allclose(g2.data[0, 0, 2], expected)

    def test_no_navigation_axes(self, ttcf_no_navigation):
        g2 = ttcf_no_navigation.get_g2()
        assert g2.data.shape == (NT,)
        np.testing.assert_allclose(
            g2.data,
            [np.mean([_base()[i, i + k] for i in range(NT - k)]) for k in range(NT)],
        )

    def test_input_not_modified(self, ttcf):
        original = ttcf.data.copy()
        ttcf.get_g2()
        np.testing.assert_array_equal(ttcf.data, original)
        assert type(ttcf) is TwoTimeCorrelationFunction

    def test_lazy(self, ttcf):
        lazy = ttcf.as_lazy().get_g2()
        assert isinstance(lazy, LazyOneTimeCorrelationFunction)
        lazy.compute()
        np.testing.assert_allclose(lazy.data, ttcf.get_g2().data)


class TestGetC2:
    n_wait = NT - WINDOW + 1

    def _unfiltered(self, ttcf):
        """c2[wait i, y, x, delay k] = ttcf[y, x, i, i + k]"""
        data = ttcf.data
        return np.array(
            [
                [
                    [[data[y, x, i, i + k] for k in range(WINDOW)] for x in range(3)]
                    for y in range(2)
                ]
                for i in range(self.n_wait)
            ]
        )

    def test_values(self, ttcf):
        c2 = ttcf.get_c2(window=WINDOW)
        assert c2.data.shape == (self.n_wait, 2, 3, WINDOW)
        np.testing.assert_allclose(c2.data, self._unfiltered(ttcf))

    def test_type_and_axes(self, ttcf):
        c2 = ttcf.get_c2(window=WINDOW)
        assert isinstance(c2, OneTimeCorrelationFunction)
        assert c2._signal_type == "otcf"
        assert c2.axes_manager.navigation_shape == (3, 2, self.n_wait)
        delay = c2.axes_manager.signal_axes[0]
        wait = c2.axes_manager.navigation_axes[2]
        assert (delay.name, delay.size, delay.scale, delay.units) == (
            "Delay Time",
            WINDOW,
            0.5,
            "s",
        )
        assert (wait.name, wait.size, wait.scale, wait.units) == (
            "Wait Time",
            self.n_wait,
            0.5,
            "s",
        )

    @pytest.mark.parametrize("size", [None, 1, 3, 5])
    def test_size_smooths_and_trims(self, ttcf, size):
        # A uniform filter of `size` points over i**2 adds (size**2 - 1) / 12 to each
        # point. Its first and last size // 2 wait times reach beyond the data and
        # are removed, so a size of 1 (no filtering) keeps every wait time.
        edge = 0 if size is None else size // 2
        shift = 0.0 if size is None else (size**2 - 1) / 12
        c2 = ttcf.get_c2(window=WINDOW, size=size)
        expected = self._unfiltered(ttcf)[edge : self.n_wait - edge] + shift
        assert c2.data.shape[0] == self.n_wait - 2 * edge
        assert c2.axes_manager.navigation_axes[2].size == self.n_wait - 2 * edge
        np.testing.assert_allclose(c2.data, expected)

    def test_size_1_is_the_same_as_no_filter(self, ttcf):
        # regression: `edge:-edge` with edge = 0 used to leave no wait times at all
        unfiltered = ttcf.get_c2(window=WINDOW)
        size_1 = ttcf.get_c2(window=WINDOW, size=1)
        assert size_1.data.shape == unfiltered.data.shape
        np.testing.assert_allclose(size_1.data, unfiltered.data)

    def test_input_not_modified(self, ttcf):
        original = ttcf.data.copy()
        ttcf.get_c2(window=WINDOW, size=3)
        np.testing.assert_array_equal(ttcf.data, original)

    @pytest.mark.parametrize("size", [None, 3])
    def test_lazy(self, ttcf, size):
        eager = ttcf.get_c2(window=WINDOW, size=size)
        lazy = ttcf.as_lazy().get_c2(window=WINDOW, size=size)
        assert isinstance(lazy, LazyOneTimeCorrelationFunction)
        lazy.compute()
        np.testing.assert_allclose(lazy.data, eager.data)


class TestGetC2NavigationShapes:
    """get_c2 must not assume how many navigation axes the ttcf has."""

    n_wait = NT - WINDOW + 1

    @staticmethod
    def _ttcf(nav_shape):
        # a different offset at every real-space position
        n = int(np.prod(nav_shape))
        offsets = 1000.0 * np.arange(n).reshape(nav_shape + (1, 1))
        return _with_time_axes(TwoTimeCorrelationFunction(_base() + offsets))

    def _unfiltered(self, ttcf):
        """c2[wait i, *position, delay k] = ttcf[*position, i, i + k]"""
        nav_shape = ttcf.data.shape[:-2]
        c2 = np.empty((self.n_wait,) + nav_shape + (WINDOW,))
        for position in np.ndindex(*nav_shape):
            for i in range(self.n_wait):
                c2[(i,) + position] = ttcf.data[position][i, i : i + WINDOW]
        return c2

    @pytest.mark.parametrize("nav_shape", [(3,), (2, 3), (2, 2, 3)])
    def test_values_and_axes(self, nav_shape):
        ttcf = self._ttcf(nav_shape)
        c2 = ttcf.get_c2(window=WINDOW)
        np.testing.assert_allclose(c2.data, self._unfiltered(ttcf))
        assert isinstance(c2, OneTimeCorrelationFunction)
        # the wait time joins the real-space axes as the last navigation axis
        assert len(c2.axes_manager.navigation_axes) == len(nav_shape) + 1
        assert c2.axes_manager.navigation_axes[-1].name == "Wait Time"
        assert c2.axes_manager.navigation_axes[-1].size == self.n_wait
        assert (
            c2.axes_manager.navigation_shape[:-1] == ttcf.axes_manager.navigation_shape
        )
        assert c2.axes_manager.signal_axes[0].name == "Delay Time"
        assert c2.axes_manager.signal_axes[0].size == WINDOW

    @pytest.mark.parametrize("size", [1, 3, 5])
    @pytest.mark.parametrize("nav_shape", [(3,), (2, 3), (2, 2, 3)])
    def test_size_smooths_and_trims(self, nav_shape, size):
        ttcf = self._ttcf(nav_shape)
        edge = size // 2
        shift = (size**2 - 1) / 12
        c2 = ttcf.get_c2(window=WINDOW, size=size)
        expected = self._unfiltered(ttcf)[edge : self.n_wait - edge] + shift
        assert c2.data.shape[0] == self.n_wait - 2 * edge
        np.testing.assert_allclose(c2.data, expected)

    @pytest.mark.parametrize("nav_shape", [(3,), (2, 2, 3)])
    def test_lazy(self, nav_shape):
        ttcf = self._ttcf(nav_shape)
        lazy = ttcf.as_lazy().get_c2(window=WINDOW, size=3)
        assert isinstance(lazy, LazyOneTimeCorrelationFunction)
        lazy.compute()
        np.testing.assert_allclose(lazy.data, ttcf.get_c2(window=WINDOW, size=3).data)


class TestGetC2NoNavigation:
    n_wait = NT - WINDOW + 1

    def test_values_and_axes(self, ttcf_no_navigation):
        c2 = ttcf_no_navigation.get_c2(window=WINDOW)
        expected = np.array(
            [[_base()[i, i + k] for k in range(WINDOW)] for i in range(self.n_wait)]
        )
        np.testing.assert_allclose(c2.data, expected)
        assert isinstance(c2, OneTimeCorrelationFunction)
        assert c2.axes_manager.navigation_shape == (self.n_wait,)
        assert c2.axes_manager.navigation_axes[0].name == "Wait Time"
        assert c2.axes_manager.signal_axes[0].name == "Delay Time"

    @pytest.mark.parametrize("size", [1, 3, 5])
    def test_size_smooths_and_trims(self, ttcf_no_navigation, size):
        edge = size // 2
        shift = (size**2 - 1) / 12
        unfiltered = ttcf_no_navigation.get_c2(window=WINDOW).data
        c2 = ttcf_no_navigation.get_c2(window=WINDOW, size=size)
        np.testing.assert_allclose(
            c2.data, unfiltered[edge : self.n_wait - edge] + shift
        )


class TestHelpers:
    def test_ttcf_2_g2(self):
        data = _base()
        expected = [np.mean([data[i, i + k] for i in range(NT - k)]) for k in range(NT)]
        np.testing.assert_allclose(_ttcf_2_g2(data), expected)

    def test_ttcf_2_c2(self):
        data = _base()
        c2 = _ttcf_2_c2(data, window=4)
        assert c2.shape == (NT - 3, 4)
        for i in range(NT - 3):
            np.testing.assert_allclose(c2[i], data[i, i : i + 4])

    def test_ttcf_2_c2_size_does_not_filter_the_delay_axis(self):
        # the uniform filter runs along the wait time only
        data = _base()
        filtered = _ttcf_2_c2(data, window=4, size=3)
        unfiltered = _ttcf_2_c2(data, window=4)
        # interior wait times: i**2 + i + k plus the filter's (3**2 - 1) / 12
        np.testing.assert_allclose(filtered[1:-1], unfiltered[1:-1] + 2 / 3)
