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
import hyperspy.api as hs

from hyperspy.signals import Signal1D
from pyxem.signals import InSituDiffraction2D, TwoTimeCorrelationFunction, InSituPolarDiffraction2D, LazyInSituPolarDiffraction2D


class TestTimeSeriesReconstruction:
    @pytest.fixture
    def insitu_data(self):
        dc = InSituDiffraction2D(data=np.ones(shape=(5, 2, 2, 25, 25)))
        dc.axes_manager.signal_axes[0].scale = 0.1
        dc.axes_manager.signal_axes[0].name = "kx"
        dc.axes_manager.signal_axes[1].scale = 0.1
        dc.axes_manager.signal_axes[1].name = "ky"
        return dc

    def test_roll_time_axis(self, insitu_data):
        rolled_data = insitu_data.roll_time_axis(0)
        assert (
            rolled_data.axes_manager.navigation_axes[2].size
            == insitu_data.axes_manager.navigation_axes[0].size
        )

    @pytest.mark.parametrize(
        "roi", [hs.roi.CircleROI(1, 1, 0.5), hs.roi.RectangularROI(0, 1, 1, 2), None]
    )
    def test_different_roi(self, insitu_data, roi):
        series = insitu_data.get_time_series(roi=roi)
        assert len(series.axes_manager.navigation_axes) == 1
        assert (
            series.axes_manager.navigation_axes[0].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        assert (
            series.axes_manager.signal_axes[0].size
            == insitu_data.axes_manager.navigation_axes[0].size
        )
        assert (
            series.axes_manager.signal_axes[1].size
            == insitu_data.axes_manager.navigation_axes[1].size
        )

    def test_time_axis(self, insitu_data):
        series = insitu_data.get_time_series(
            roi=hs.roi.CircleROI(1, 1, 0.5), time_axis=0
        )
        assert (
            series.axes_manager.navigation_axes[0].size
            == insitu_data.axes_manager.navigation_axes[0].size
        )
    
    def test_time_axis_identification(self, insitu_data):
        from pyxem.utils._insitu import _find_time_axis
        import re
        time_axis_names = ["time", "t", "Time", "T"]
        expected_msg = f"Time axis not found, name should be one of {time_axis_names}"
        with pytest.raises(ValueError, match=re.escape(expected_msg)):
            _find_time_axis(insitu_data)
        insitu_data.axes_manager.navigation_axes[2].name = "Time"
        time_axis = _find_time_axis(insitu_data)
        assert time_axis == 2

class TestDriftCorrection:
    @pytest.fixture(params=[
        np.random.randint(0, 5, (10, 2)),
        np.random.randint(0, 5, (10, 2))
    ])
    def insitu_data_with_shifts(self, request):
        shift_schedule = request.param
        dc = InSituDiffraction2D(data=np.zeros((10, 20, 20, 2, 2)))
        dc.axes_manager.navigation_axes[2].name = 'Time'

        shift_schedule[0] = [0, 0]
        for i, shift in enumerate(shift_schedule):
            shift_x, shift_y = shift
            dc.data[i, shift_x:shift_x+10, shift_y:shift_y+10] = np.ones((2, 2))

        return { "data": dc, "expected_shifts": shift_schedule }
    

    def test_drift(self, insitu_data_with_shifts):
        insitu_data = insitu_data_with_shifts["data"]
        expected_shifts = insitu_data_with_shifts["expected_shifts"]
        shifts = insitu_data.get_drift_vectors()
        assert (
            shifts.axes_manager.navigation_axes[0].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        assert shifts.axes_manager.signal_axes[0].size == 2

        assert isinstance(shifts, Signal1D)

        np.testing.assert_allclose(
            shifts.data, 
            expected_shifts,
            atol=0.5
        )

    @pytest.mark.parametrize(
        "lazy",
        [True, False]
    )
    def test_correct_real_space_drift(self, insitu_data_with_shifts, lazy):
        insitu_data = insitu_data_with_shifts["data"]
        expected_shifts = insitu_data_with_shifts["expected_shifts"]
        shifts = Signal1D(expected_shifts)
        if lazy:
            insitu_data = insitu_data.as_lazy()
            assert insitu_data._lazy
        
        shifted_data = insitu_data.correct_real_space_drift(shifts=shifts, lazy_result=lazy)
        if lazy:
            shifted_data.compute()
        assert isinstance(shifted_data, InSituDiffraction2D)
        desire_results = np.zeros((10, 20, 20, 2, 2))
        desire_results[0:10, 0:10, 0:10] = np.ones((2, 2))
        np.testing.assert_almost_equal(
            shifted_data.data, 
            desire_results
        )
    @pytest.mark.parametrize(
        "lazy",
        [True, False]
    )
    def test_correct_real_space_drift_fast(self, insitu_data_with_shifts, lazy):
        insitu_data = insitu_data_with_shifts["data"]
        expected_shifts = insitu_data_with_shifts["expected_shifts"]
        shifts = Signal1D(expected_shifts)
        if lazy:
            insitu_data = insitu_data.as_lazy()
            assert insitu_data._lazy
            insitu_data.rechunk((-1, 1, 1, -1, -1))
            with pytest.raises(Exception) as exc_info:
                insitu_data.correct_real_space_drift_fast(shifts=shifts)
            assert exc_info.match(
                "Spatial axes are chunked. Please rechunk signal or use 'correct_real_space_drift' "
                "instead"
            )
            insitu_data.rechunk((1, -1, -1, 1, 1))
        shifted_data = insitu_data.correct_real_space_drift_fast(shifts=shifts)
        if lazy:
            shifted_data.compute()
        assert isinstance(shifted_data, InSituDiffraction2D)
        desire_results = np.zeros((10, 20, 20, 2, 2))
        desire_results[0:10, 0:10, 0:10] = np.ones((2, 2))
        np.testing.assert_almost_equal(
            shifted_data.data, 
            desire_results
        )

        
    



class TestCorrelation:
    @pytest.fixture
    def insitu_data(self):
        dc = InSituDiffraction2D(data=np.random.rand(50, 10, 10, 4, 4))
        dc.axes_manager.signal_axes[0].scale = 0.1
        dc.axes_manager.signal_axes[0].name = "kx"
        dc.axes_manager.signal_axes[1].scale = 0.1
        dc.axes_manager.signal_axes[1].name = "ky"
        dc.axes_manager.navigation_axes[2].scale = 1.0
        dc.axes_manager.navigation_axes[2].name = "Time"
        return dc

    

    @pytest.mark.parametrize("normalization", ["self", "split"])
    def test_g2_normalization(self, insitu_data, normalization):
        g2 = insitu_data.get_g2_2d_kresolved(normalization=normalization)
        mean_g2 = g2.isig[:, :, 1:-1].mean(axis=[-1, -2, -3]).data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((10, 10))[num_index], mean_g2[num_index], atol=0.1
        )

    @pytest.mark.parametrize("trs", [np.linspace(0, 10, 25), 10])
    @pytest.mark.parametrize("bins", [(2, 2, 5), (1, 4, 1)])
    def test_g2_bin_resample_time(self, insitu_data, trs, bins):
        g2 = insitu_data.get_g2_2d_kresolved(
            k1bin=bins[0], k2bin=bins[1], tbin=bins[2], resample_time=trs
        )
        assert g2.axes_manager.signal_axes[0].size == int(4 / bins[0])
        assert g2.axes_manager.signal_axes[1].size == int(4 / bins[1])
        mean_g2 = g2.isig[:, :, 1:-1].mean(axis=[-1, -2, -3]).data
        num_index = ~np.isreal(mean_g2)
        np.testing.assert_allclose(
            np.ones((10, 10))[num_index], mean_g2[num_index], atol=0.1
        )

    def test_unrolled_time_axes(self, insitu_data):
        rolled_data = insitu_data.roll_time_axis(0)
        shifted_data = rolled_data.correct_real_space_drift(
            shifts=Signal1D(np.zeros((50, 2))), time_axis=1
        )
        assert (
            shifted_data.axes_manager.navigation_axes[2].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        shifted_data_fast = rolled_data.correct_real_space_drift_fast(
            shifts=Signal1D(np.zeros((50, 2))), time_axis=1
        )
        assert (
            shifted_data_fast.axes_manager.navigation_axes[2].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
        g2 = rolled_data.get_g2_2d_kresolved(time_axis=1)
        assert (
            g2.axes_manager.signal_axes[-1].size
            == insitu_data.axes_manager.navigation_axes[2].size
        )
    
    def test_get_TTCF(self, insitu_data):
        normalized_data = (insitu_data-0.5)/0.5
        beta = np.var(normalized_data.data)
        ttcf = normalized_data.get_TTCF()
        assert isinstance(ttcf, TwoTimeCorrelationFunction)
        np.testing.assert_allclose(
            np.diagonal(ttcf.data.mean(axis=(0, 1))), np.ones(ttcf.data.shape[2])*beta, atol=0.1
        )
        masked_ttcf = ttcf.data.mean(axis=(0, 1))
        masked_ttcf[np.diag_indices_from(masked_ttcf)] = 0.0
        np.testing.assert_allclose(
            masked_ttcf, np.zeros_like(masked_ttcf), atol=0.1
        )
        assert ttcf.axes_manager.signal_axes[0].size == ttcf.axes_manager.signal_axes[1].size
        assert ttcf.axes_manager.signal_axes[0].name == ttcf.axes_manager.signal_axes[1].name
        assert ttcf.axes_manager.signal_axes[0].scale == ttcf.axes_manager.signal_axes[1].scale
        assert ttcf.axes_manager.signal_axes[0].units == ttcf.axes_manager.signal_axes[1].units
        assert ttcf.axes_manager.signal_axes[0].offset == insitu_data.axes_manager.navigation_axes[2].offset

class TestExpectedIntensity:
    @pytest.fixture
    def insitu_data(self):
        dc = InSituDiffraction2D(data=np.ones((5,2,2,3,3)))
        dc.axes_manager.navigation_axes[2].name = "Time"
        for i in range(5):
            dc.data[i] += i
        gradient = np.zeros((3,3))
        for i in range(3):
            gradient[i] += i-1
        dc.data += gradient
        return dc

    def test_expected_intensity_invalid_inputs(self, insitu_data):
        # Test missing custom_axes for 'custom' method
        with pytest.raises(ValueError, match="For 'custom' method, 'custom_axes' must be provided"):
            insitu_data.expected_intensity(method="custom")
            
        # Test invalid method name
        with pytest.raises(ValueError, match="Method must be one of 't', 'k', or 'custom'"):
            insitu_data.expected_intensity(method="invalid_method")

    def test_expected_intensity_output_type(self, insitu_data):
        exp_I_data = insitu_data.expected_intensity(method="t")
        assert isinstance(exp_I_data, InSituDiffraction2D)
        assert exp_I_data._signal_type == "insitu_diffraction"

    def test_lazy_chunk_optimize_behavior(self, insitu_data):
        lazy_data = insitu_data.as_lazy()
        lazy_data.rechunk((1, 1, 1, -1, -1))
        exp_I_data = lazy_data.expected_intensity(method="t", chunk_optimize=True)
        chunk_sizes = exp_I_data.data.chunks
        assert chunk_sizes[0] == (exp_I_data.axes_manager.navigation_axes[2].size,)

    def test_expected_intensity_method_t(self, insitu_data):
        exp_I_data = insitu_data.expected_intensity(method="t")
        desired_data = np.ones((5, 2, 2, 3, 3)) * np.mean(insitu_data.data, axis=0)
        np.testing.assert_almost_equal(exp_I_data.data, desired_data)

    def test_expected_intensity_method_k(self, insitu_data):
        exp_I_data = insitu_data.expected_intensity(method="k")
        desired_data = np.arange(1,6).reshape(5, 1, 1, 1, 1) + np.zeros((5, 2, 2, 3, 3))
        np.testing.assert_almost_equal(exp_I_data.data, desired_data)

    def test_expected_intensity_method_custom_full_axis(self, insitu_data):
        custom_axes = {"full_axis": [1]}
        exp_I_data = insitu_data.expected_intensity(method="custom", custom_axes=custom_axes)
        np.testing.assert_almost_equal(exp_I_data.data, np.arange(1,6).reshape(5, 1, 1, 1, 1) + np.zeros((5, 2, 2, 3, 3)))

    @pytest.mark.parametrize("mode", ["wrap", "mirror"])
    def test_expected_intensity_method_custom_local_axis(self, insitu_data, mode):
        custom_axes = {"local_axis": [1], "local_size": 3}
        exp_I_data = insitu_data.expected_intensity(method="custom", custom_axes=custom_axes, mode=mode)
        if mode == "wrap":
            desired_data = np.arange(1,6).reshape(5, 1, 1, 1, 1) + np.zeros((5, 2, 2, 3, 3)) 
        elif mode == "mirror":
            gradient = np.zeros((3,3))
            for i in range(3):
                gradient[i] += (1-i)*2/3
            desired_data = insitu_data.data + gradient
        np.testing.assert_almost_equal(exp_I_data.data, desired_data) 



class TestAzimuthalIntegration:
    @pytest.fixture
    def insitu_data(self):
        dc = InSituDiffraction2D(data=np.random.rand(50, 10, 10, 4, 4))
        dc.axes_manager.signal_axes[0].scale = 0.1
        dc.axes_manager.signal_axes[0].name = "kx"
        dc.axes_manager.signal_axes[1].scale = 0.1
        dc.axes_manager.signal_axes[1].name = "ky"
        dc.axes_manager.navigation_axes[2].scale = 1.0
        dc.axes_manager.navigation_axes[2].name = "Time"
        return dc
    
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_azimuthal_integral2d_type_casting(self, insitu_data, inplace, lazy):
        data = insitu_data.as_lazy() if lazy else insitu_data
        
        # Call the method
        res = data.get_azimuthal_integral2d(inplace=inplace, npt=2, radial_range=(0, 0.2))
        
        # Determine which object to inspect
        target_obj = data if inplace else res

        print("Class name:", type(target_obj).__name__)
        
        if inplace:
            assert res is None
        
        # Verify correct subclass assignment
        if lazy:
            assert isinstance(target_obj, LazyInSituPolarDiffraction2D)
        else:
            assert isinstance(target_obj, InSituPolarDiffraction2D)
            
        assert target_obj._signal_type == "insitu_polar"