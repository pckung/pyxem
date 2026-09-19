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
import dask.array as da

from hyperspy.signals import Signal2D, Signal1D

from pyxem.signals import (
    PolarDiffraction2D,
    LazyPolarDiffraction2D,
    Correlation2D,
    Power2D,
    Correlation1D,
)


class TestComputeAndAsLazy2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyPolarDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == PolarDiffraction2D
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyPolarDiffraction2D(dask_array)
        s.compute()
        assert s.__class__ == PolarDiffraction2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = PolarDiffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyPolarDiffraction2D
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = PolarDiffraction2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyPolarDiffraction2D
        assert data.shape == s_lazy.data.shape


class TestCorrelations:
    @pytest.fixture
    def flat_pattern(self):
        pd = PolarDiffraction2D(data=np.ones(shape=(2, 2, 5, 5)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    def test_correlation_signal(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation()
        assert isinstance(ac, Correlation2D)

    def test_axes_transfer(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation()
        assert (
            ac.axes_manager.signal_axes[0].scale
            == flat_pattern.axes_manager.signal_axes[0].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].scale
            == flat_pattern.axes_manager.signal_axes[1].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].name
            == flat_pattern.axes_manager.signal_axes[1].name
        )

    @pytest.mark.parametrize(
        "mask", [None, np.zeros(shape=(5, 5)), Signal2D(np.zeros(shape=(2, 2, 5, 5)))]
    )
    def test_masking_correlation(self, flat_pattern, mask):
        ap = flat_pattern.get_angular_correlation(mask=mask)
        assert isinstance(ap, Correlation2D)

    def test_correlation_inplace(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation(inplace=True)
        assert ac is None
        assert isinstance(flat_pattern, Correlation2D)

    @pytest.mark.parametrize(
        "mask", [None, np.zeros(shape=(5, 5)), Signal2D(np.zeros(shape=(2, 2, 5, 5)))]
    )
    def test_masking_angular_power(self, flat_pattern, mask):
        ap = flat_pattern.get_angular_power(mask=mask)
        print(ap)
        assert isinstance(ap, Power2D)

    def test_axes_transfer_power(self, flat_pattern):
        ac = flat_pattern.get_angular_power()
        assert ac.axes_manager.signal_axes[0].scale == 1
        assert (
            ac.axes_manager.signal_axes[1].scale
            == flat_pattern.axes_manager.signal_axes[1].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].name
            == flat_pattern.axes_manager.signal_axes[1].name
        )

    def test_power_inplace(self, flat_pattern):
        ac = flat_pattern.get_angular_power(inplace=True)
        assert ac is None
        assert isinstance(flat_pattern, Power2D)


class TestPearsonCorrelation:
    @pytest.fixture
    def flat_pattern(self):
        rng = np.random.default_rng(seed=1)
        pd = PolarDiffraction2D(data=rng.random((2, 2, 15, 50)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 0.1
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    @pytest.mark.parametrize("krange", [None, (0, 4), (0.5, 1.4)])
    def test_pearson_correlation_signal(self, flat_pattern, krange):
        rho = flat_pattern.get_full_pearson_correlation(krange=krange)
        assert isinstance(rho, Signal1D)
        rhok = flat_pattern.get_resolved_pearson_correlation(krange=krange)
        assert isinstance(rhok, Signal2D)

    @pytest.mark.parametrize("inplace", (True, False))
    @pytest.mark.parametrize("krange", [None, (0, 10), (0.5, 1.4)])
    def test_full_pearson_correlation_results(self, flat_pattern, krange, inplace):
        out = flat_pattern.get_full_pearson_correlation(
            krange=krange,
            inplace=inplace,
        )
        if inplace:
            assert out is None
            out = flat_pattern
        else:
            # check the original signal is not changed
            assert flat_pattern.axes_manager[-1].size == 15

        assert isinstance(out, Correlation1D)
        np.testing.assert_allclose(np.zeros((2, 2, 49)), out.data[..., 1:], atol=0.2)

    @pytest.mark.parametrize("inplace", (True, False))
    @pytest.mark.parametrize("krange", [None, (0, 10), (0.5, 1.4)])
    def test_resolved_pearson_correlation_results(self, flat_pattern, krange, inplace):
        out = flat_pattern.get_resolved_pearson_correlation(
            krange=krange,
            inplace=inplace,
        )
        if inplace:
            assert out is None
            out = flat_pattern
        else:
            # check the original signal is not changed
            assert flat_pattern.axes_manager[-1].size == 15
            assert flat_pattern.axes_manager[-2].size == 50

        assert isinstance(out, Correlation2D)
        np.testing.assert_allclose(
            np.zeros((2, 2, 49)), np.mean(out.data[..., 1:], axis=-2), atol=0.2
        )

    def test_full_pearson_correlation_inplace(self, flat_pattern):
        rho = flat_pattern.get_full_pearson_correlation(inplace=True)
        assert rho is None
        assert isinstance(flat_pattern, Correlation1D)

    def test_resolved_pearson_correlation_inplace(self, flat_pattern):
        rho = flat_pattern.get_resolved_pearson_correlation(inplace=True)
        assert rho is None
        assert isinstance(flat_pattern, Correlation2D)

    def test_axes_transfer(self, flat_pattern):
        rho = flat_pattern.get_full_pearson_correlation()
        assert (
            rho.axes_manager.signal_axes[0].scale
            == flat_pattern.axes_manager.signal_axes[0].scale
        )

        rhok = flat_pattern.get_resolved_pearson_correlation()
        assert (
            rhok.axes_manager.signal_axes[0].scale
            == flat_pattern.axes_manager.signal_axes[0].scale
        )
        assert (
            rhok.axes_manager.signal_axes[1].scale
            == flat_pattern.axes_manager.signal_axes[1].scale
        )

    @pytest.mark.parametrize("mask", [None, np.zeros(shape=(15, 50))])
    def test_masking_pearson_correlation(self, flat_pattern, mask):
        rho_0 = flat_pattern.get_full_pearson_correlation(mask=mask)
        assert isinstance(rho_0, Correlation1D)
        rho = flat_pattern.get_full_pearson_correlation(mask=mask, krange=(0, 4))
        assert isinstance(rho, Correlation1D)

        rhok_0 = flat_pattern.get_resolved_pearson_correlation(mask=mask)
        assert isinstance(rhok_0, Correlation2D)
        rhok = flat_pattern.get_resolved_pearson_correlation(mask=mask, krange=(0, 4))
        assert isinstance(rhok, Correlation2D)


class TestDecomposition:
    def test_decomposition_is_performed(self, diffraction_pattern):
        s = PolarDiffraction2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = PolarDiffraction2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, PolarDiffraction2D)


class TestSubtractingDiffractionBackground:
    @pytest.fixture
    def noisy_data(self):
        data = np.random.rand(3, 2, 20, 15)
        data[:, :, 10:12, 7:9] = 100
        dp = PolarDiffraction2D(data)
        return dp

    @pytest.mark.parametrize(
        ["method", "kwargs"],
        [
            ("radial median", {}),
            ("radial percentile", {"percentile": 40}),
            pytest.param(
                "this method does not exist",
                {},
                marks=pytest.mark.xfail(raises=NotImplementedError),
            ),
        ],
    )
    def test_subtract_backgrounds(self, method, kwargs, noisy_data):
        kwargs["inplace"] = False

        subtracted = noisy_data.subtract_diffraction_background(method=method, **kwargs)
        assert isinstance(subtracted, PolarDiffraction2D)
        assert subtracted.data.shape == noisy_data.data.shape


class TestBeamStop:
    n_k = 30
    n_azi = 180  # 2 degrees per azimuthal pixel
    stripe = slice(80, 100)  # azimuthal columns hidden behind the beam stop

    @pytest.fixture
    def signal(self):
        data = 100 + np.random.default_rng(seed=0).random((3, 4, self.n_k, self.n_azi))
        data[..., self.stripe] = 0.0
        return PolarDiffraction2D(data)

    @pytest.fixture
    def index_signal(self):
        # every value is its own azimuthal index, so the columns that are kept,
        # and their order, can be read directly from the output
        data = np.broadcast_to(
            np.arange(self.n_azi, dtype=float), (2, 3, self.n_k, self.n_azi)
        ).copy()
        return PolarDiffraction2D(data)

    @pytest.fixture
    def seam_signal(self):
        # the beam stop crosses the 0/360 degree seam: columns 170-179 and 0-9
        data = 100 + np.random.default_rng(seed=1).random((3, 4, self.n_k, self.n_azi))
        data[..., 170:] = 0.0
        data[..., :10] = 0.0
        return PolarDiffraction2D(data)

    def _mask(self, start, stop):
        mask = np.zeros((self.n_k, self.n_azi), dtype=bool)
        mask[:, start:stop] = True
        return mask

    def test_get_beam_stop_type_and_shape(self, signal):
        mask = signal.get_beam_stop()
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (self.n_k, self.n_azi)

    def test_get_beam_stop_finds_stripe(self, signal):
        mask = signal.get_beam_stop()
        assert mask[:, 85:95].all()
        assert not mask[:, :70].any()
        assert not mask[:, 110:].any()
        # a beam stop hides whole azimuthal columns, at every radius
        np.testing.assert_array_equal(mask.any(axis=0), mask.all(axis=0))

    def test_get_beam_stop_default_start_point_is_centre(self, signal):
        default = signal.get_beam_stop()
        explicit = signal.get_beam_stop(start_point=[self.n_k // 2, self.n_azi // 2])
        np.testing.assert_array_equal(default, explicit)

    def test_get_beam_stop_start_point_selects_region(self, signal):
        # starting in the background selects the background, not the stripe
        mask = signal.get_beam_stop(start_point=[15, 40])
        assert mask[15, 40]
        assert not mask[:, self.stripe].any()

    @pytest.mark.parametrize("start_point", [[5, 86], [25, 94], [0, 90]])
    def test_get_beam_stop_same_from_any_point_inside(self, signal, start_point):
        np.testing.assert_array_equal(
            signal.get_beam_stop(start_point=start_point), signal.get_beam_stop()
        )

    @pytest.mark.parametrize("start_point", [[15, 3], [15, 175]])
    def test_get_beam_stop_across_seam(self, seam_signal, start_point):
        mask = seam_signal.get_beam_stop(start_point=start_point)
        # both halves of the beam stop are found, whichever side is picked
        assert mask[:, 175:].all()
        assert mask[:, :5].all()
        assert not mask[:, 20:160].any()
        np.testing.assert_array_equal(mask.any(axis=0), mask.all(axis=0))

    def test_get_beam_stop_across_seam_same_from_either_side(self, seam_signal):
        np.testing.assert_array_equal(
            seam_signal.get_beam_stop(start_point=[15, 3]),
            seam_signal.get_beam_stop(start_point=[15, 175]),
        )

    @pytest.mark.parametrize("start_point", [[-1, 90], [30, 90], [15, -1], [15, 180]])
    def test_get_beam_stop_start_point_outside(self, signal, start_point):
        with pytest.raises(ValueError, match="outside the pattern"):
            signal.get_beam_stop(start_point=start_point)

    def test_get_beam_stop_start_point_on_edge(self, signal):
        # column 80 is the edge between the background and the beam stop
        with pytest.raises(ValueError, match="edge"):
            signal.get_beam_stop(start_point=[15, 80])

    def test_get_beam_stop_lazy(self, signal):
        mask = signal.get_beam_stop()
        lazy_mask = signal.as_lazy().get_beam_stop()
        assert isinstance(lazy_mask, np.ndarray)
        np.testing.assert_array_equal(lazy_mask, mask)

    @pytest.mark.parametrize(
        "exclude_angle, half_exclude_pix",
        [
            (None, 2),  # 8 degrees -> 4 pixels
            (20.0, 7),  # 20 + 8 degrees -> 14 pixels
        ],
    )
    def test_remove_beam_stop_area_columns(
        self, index_signal, exclude_angle, half_exclude_pix
    ):
        mask = self._mask(80, 100)
        out = index_signal.remove_beam_stop_area(
            beam_stop_mask=mask, exclude_angle=exclude_angle
        )
        left = 80 - half_exclude_pix
        right = 99 + half_exclude_pix + 1
        # the data now starts just after the beam stop and wraps round to it
        expected = np.concatenate([np.arange(right, self.n_azi), np.arange(0, left)])
        assert out.data.shape[-1] == self.n_azi - (right - left)
        np.testing.assert_array_equal(
            out.data, np.broadcast_to(expected, out.data.shape)
        )

    def test_remove_beam_stop_area_type_and_axes(self, index_signal):
        out = index_signal.remove_beam_stop_area(beam_stop_mask=self._mask(80, 100))
        assert isinstance(out, PolarDiffraction2D)
        assert out.axes_manager.navigation_shape == (
            index_signal.axes_manager.navigation_shape
        )
        # only the azimuthal axis shrinks
        assert out.axes_manager.signal_axes[1].size == self.n_k
        assert out.axes_manager.signal_axes[0].size < self.n_azi

    def test_remove_beam_stop_area_does_not_modify_inputs(self, index_signal):
        data = index_signal.data.copy()
        mask = self._mask(80, 100)
        index_signal.remove_beam_stop_area(beam_stop_mask=mask)
        np.testing.assert_array_equal(index_signal.data, data)
        np.testing.assert_array_equal(mask, self._mask(80, 100))

    def test_remove_beam_stop_area_computes_mask(self, signal):
        out = signal.remove_beam_stop_area(exclude_angle=10.0)
        assert not (out.data == 0).any()
        explicit = signal.remove_beam_stop_area(
            beam_stop_mask=signal.get_beam_stop(), exclude_angle=10.0
        )
        np.testing.assert_array_equal(out.data, explicit.data)

    def test_remove_beam_stop_area_forwards_start_point(self, signal):
        out = signal.remove_beam_stop_area(start_point=[15, 40])
        explicit = signal.remove_beam_stop_area(
            beam_stop_mask=signal.get_beam_stop(start_point=[15, 40])
        )
        np.testing.assert_array_equal(out.data, explicit.data)
        # a different start point selects a different region
        default = signal.remove_beam_stop_area()
        assert out.data.shape != default.data.shape

    def test_remove_beam_stop_area_lazy(self, signal):
        eager = signal.remove_beam_stop_area(exclude_angle=10.0)
        lazy = signal.as_lazy().remove_beam_stop_area(exclude_angle=10.0)
        assert isinstance(lazy, LazyPolarDiffraction2D)
        lazy.compute()
        np.testing.assert_array_equal(lazy.data, eager.data)

    @pytest.mark.parametrize(
        "mask_columns, expected",
        [
            # each side of the beam stop loses 2 pixels (8 degrees) of margin
            (np.r_[80:100], np.r_[102:180, 0:78]),
            (np.r_[0:10], np.r_[12:178]),  # margin wraps to the end of the axis
            (np.r_[170:180], np.r_[2:168]),  # margin wraps to the start of the axis
            (np.r_[175:180, 0:5], np.r_[7:173]),  # beam stop crosses the seam
        ],
        ids=["middle", "start", "end", "seam"],
    )
    def test_remove_beam_stop_area_periodic(
        self, index_signal, mask_columns, expected
    ):
        mask = np.zeros((self.n_k, self.n_azi), dtype=bool)
        mask[:, mask_columns] = True
        out = index_signal.remove_beam_stop_area(beam_stop_mask=mask)
        assert out.data.shape[-1] <= self.n_azi
        np.testing.assert_array_equal(
            out.data, np.broadcast_to(expected, out.data.shape)
        )

    def test_remove_beam_stop_area_computes_mask_across_seam(self, seam_signal):
        out = seam_signal.remove_beam_stop_area(start_point=[15, 3], exclude_angle=10.0)
        assert not (out.data == 0).any()
        assert out.data.shape[-1] < self.n_azi - 20

    def test_remove_beam_stop_area_keeps_one_column(self, index_signal):
        # 175 beam stop columns and 2 pixels of margin either side leaves column 177
        out = index_signal.remove_beam_stop_area(beam_stop_mask=self._mask(0, 175))
        np.testing.assert_array_equal(out.data, np.full_like(out.data, 177))

    def test_remove_beam_stop_area_covers_everything(self, index_signal):
        with pytest.raises(ValueError, match="whole azimuthal range"):
            index_signal.remove_beam_stop_area(beam_stop_mask=self._mask(0, 176))

    def test_remove_beam_stop_area_empty_mask(self, index_signal):
        with pytest.raises(ValueError, match="does not contain any"):
            index_signal.remove_beam_stop_area(beam_stop_mask=self._mask(0, 0))

    def test_remove_beam_stop_area_wrong_mask_shape(self, index_signal):
        mask = np.ones((self.n_k, self.n_azi + 1), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            index_signal.remove_beam_stop_area(beam_stop_mask=mask)
