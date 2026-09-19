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

import matplotlib.pyplot as plt
import numpy as np
import pytest
from hyperspy.axes import DataAxis

from pyxem.signals import LazyOneTimeCorrelationFunction, OneTimeCorrelationFunction
from pyxem.signals.one_time_correlation_function import (
    _KWW_curve,
    _g2_fit,
    _popt_to_g2,
)
from pyxem.utils._insitu import _get_resample_time, _interpolate_g2_1d

DT = 0.1  # time step of the measured g2
N = 120  # number of time steps
TRUE = (1.5, 1.0, 1.4)  # tau, beta, C


def _with_real_space_axes(signal):
    """Give the two navigation axes non-default metadata."""
    for axis, name in zip(signal.axes_manager.navigation_axes, ["x", "y"]):
        axis.name, axis.scale, axis.units, axis.offset = name, 2.0, "nm", 1.0
    return signal


@pytest.fixture
def ramp():
    # g2[i] = i is linear in time, so linear interpolation reproduces it exactly
    s = OneTimeCorrelationFunction(np.tile(np.arange(N, dtype=float), (2, 3, 1)))
    axis = s.axes_manager.signal_axes[0]
    axis.name, axis.scale, axis.units = "Delay Time", DT, "s"
    return _with_real_space_axes(s)


@pytest.fixture
def resampled_kww():
    # The exact model on a resampled time grid, flagged as resampled, so the fit
    # can be checked without the interpolation error of resample_g2
    t_rs = _get_resample_time(N, DT, 60)
    s = OneTimeCorrelationFunction(np.tile(_KWW_curve(t_rs, *TRUE), (2, 3, 1)))
    s._resampled = True
    s._t_rs = t_rs
    return _with_real_space_axes(s)


class TestModel:
    def test_kww_curve(self):
        t = np.array([0.0, 1.5, 3.0])
        # C at t = 0, and a simple exponential exp(-2 t / tau) when beta = 1
        np.testing.assert_allclose(
            _KWW_curve(t, 1.5, 1.0, 1.4), 1.4 * np.exp(-2 * t / 1.5)
        )
        assert _KWW_curve(0.0, 3.0, 0.5, 2.0) == 2.0
        # a smaller beta gives a slower, stretched decay at late times
        assert _KWW_curve(6.0, 1.5, 0.5, 1.0) > _KWW_curve(6.0, 1.5, 1.0, 1.0)

    def test_popt_to_g2(self):
        t_rs = np.array([0.1, 1.0, 5.0])
        popt = np.array([1.5, 0.8, 1.4, 0.1, 0.1, 0.1])  # the errors are ignored
        np.testing.assert_allclose(
            _popt_to_g2(popt, t_rs), _KWW_curve(t_rs, 1.5, 0.8, 1.4)
        )

    def test_popt_to_g2_needs_time(self):
        with pytest.raises(ValueError, match="t_rs must be provided"):
            _popt_to_g2(np.array([1.5, 1.0, 1.4]))

    def test_g2_fit_recovers_the_parameters(self):
        t_rs = _get_resample_time(N, DT, 60)
        result = _g2_fit(_KWW_curve(t_rs, *TRUE), t_rs, tau_guess=1.0, mask=True)
        assert result.shape == (6,)
        np.testing.assert_allclose(result[:3], TRUE, rtol=1e-6)
        np.testing.assert_allclose(result[3:], 0.0, atol=1e-6)

    def test_g2_fit_masked_out(self):
        t_rs = _get_resample_time(N, DT, 60)
        result = _g2_fit(_KWW_curve(t_rs, *TRUE), t_rs, tau_guess=1.0, mask=False)
        assert np.isnan(result).all() and result.shape == (6,)

    def test_g2_fit_failure_gives_nan(self):
        # a signal that starts at zero has empty bounds on C, which curve_fit rejects
        t_rs = _get_resample_time(N, DT, 60)
        result = _g2_fit(np.zeros(60), t_rs, tau_guess=1.0, mask=True)
        assert np.isnan(result).all() and result.shape == (6,)


class TestTimeHelpers:
    def test_get_resample_time_is_log_spaced(self):
        t_rs = _get_resample_time(N, DT, 40)
        assert t_rs.shape == (40,)
        np.testing.assert_allclose([t_rs[0], t_rs[-1]], [DT, (N - 1) * DT])
        # a constant ratio between neighbours, so equally spaced in log time
        np.testing.assert_allclose(np.diff(np.log(t_rs)), np.diff(np.log(t_rs))[0])

    def test_interpolate_g2_1d(self):
        g2 = np.arange(10, dtype=float)  # linear, so the interpolation is exact
        t_rs = np.array([0.25, 0.5, 1.2, 4.5])
        np.testing.assert_allclose(_interpolate_g2_1d(g2, t_rs, dt=0.5), t_rs / 0.5)


class TestResampleG2:
    def test_time_axis_properties(self, ramp):
        assert ramp._dt == DT
        assert ramp._time_steps == N

    def test_resampled_signal(self, ramp):
        rs = ramp.resample_g2(t_rs_size=40)
        t_rs = _get_resample_time(N, DT, 40)
        assert isinstance(rs, OneTimeCorrelationFunction)
        assert rs.data.shape == (2, 3, 40)
        assert rs._resampled
        assert not ramp._resampled
        np.testing.assert_allclose(rs._t_rs, t_rs)
        time_axis = rs.axes_manager.signal_axes[0]
        assert isinstance(time_axis, DataAxis)
        assert (time_axis.name, time_axis.units) == ("resampled_time", "s")
        np.testing.assert_allclose(time_axis.axis, t_rs)

    def test_values(self, ramp):
        rs = ramp.resample_g2(t_rs_size=40)
        expected = _get_resample_time(N, DT, 40) / DT
        np.testing.assert_allclose(rs.data, np.broadcast_to(expected, rs.data.shape))

    def test_default_size(self, ramp):
        assert ramp.resample_g2().data.shape == (2, 3, 100)

    def test_explicit_dt(self, ramp):
        rs = ramp.resample_g2(t_rs_size=40, dt=0.2)
        np.testing.assert_allclose(rs._t_rs, _get_resample_time(N, 0.2, 40))
        np.testing.assert_allclose(rs.data[0, 0], rs._t_rs / 0.2)

    def test_navigation_axes_preserved(self, ramp):
        rs = ramp.resample_g2(t_rs_size=40)
        assert rs.axes_manager.navigation_shape == ramp.axes_manager.navigation_shape
        for axis in rs.axes_manager.navigation_axes:
            assert (axis.scale, axis.units, axis.offset) == (2.0, "nm", 1.0)
        assert [a.name for a in rs.axes_manager.navigation_axes] == ["x", "y"]

    def test_input_not_modified(self, ramp):
        original = ramp.data.copy()
        ramp.resample_g2(t_rs_size=40)
        np.testing.assert_array_equal(ramp.data, original)
        assert ramp.data.shape == (2, 3, N)

    def test_lazy(self, ramp):
        lazy = ramp.as_lazy().resample_g2(t_rs_size=40)
        assert isinstance(lazy, LazyOneTimeCorrelationFunction)
        assert lazy._resampled
        lazy.compute()
        np.testing.assert_allclose(lazy.data, ramp.resample_g2(t_rs_size=40).data)


class TestFitG2Decay:
    def test_requires_resampling(self, ramp):
        with pytest.raises(RuntimeError, match="resampled"):
            ramp.fit_g2_decay(tau_int=1.0)

    def test_recovers_the_parameters(self, resampled_kww):
        fit = resampled_kww.fit_g2_decay(tau_int=1.0)
        assert fit.data.shape == (2, 3, 6)
        np.testing.assert_allclose(
            fit.data[..., :3], np.broadcast_to(TRUE, (2, 3, 3)), rtol=1e-6
        )
        np.testing.assert_allclose(fit.data[..., 3:], 0.0, atol=1e-6)

    def test_resample_then_fit(self):
        # the whole pipeline on data measured on the linear time grid
        t = np.arange(N) * DT
        s = OneTimeCorrelationFunction(np.tile(_KWW_curve(t, *TRUE), (2, 3, 1)))
        s.axes_manager.signal_axes[0].scale = DT
        fit = s.resample_g2(t_rs_size=100).fit_g2_decay(tau_int=1.0)
        np.testing.assert_allclose(
            fit.data[..., :3], np.broadcast_to(TRUE, (2, 3, 3)), rtol=0.02
        )

    def test_axes_and_flags(self, resampled_kww):
        fit = resampled_kww.fit_g2_decay(tau_int=1.0)
        assert isinstance(fit, OneTimeCorrelationFunction)
        fit_axis = fit.axes_manager.signal_axes[0]
        assert (fit_axis.name, fit_axis.size) == ("fitted_results", 6)
        for axis, name in zip(fit.axes_manager.navigation_axes, ["x", "y"]):
            assert (axis.name, axis.scale, axis.units, axis.offset) == (
                name,
                2.0,
                "nm",
                1.0,
            )
        assert fit._fitted_result
        np.testing.assert_allclose(fit._t_rs, resampled_kww._t_rs)

    def test_mask(self, resampled_kww):
        # the mask has the shape of the data without its time axis, (y, x)
        mask = np.ones(resampled_kww.data.shape[:-1], dtype=int)
        mask[0, 1] = 0
        fit = resampled_kww.fit_g2_decay(tau_int=1.0, mask=mask)
        assert np.isnan(fit.data[0, 1]).all()
        others = np.delete(fit.data.reshape(-1, 6), 1, axis=0)
        np.testing.assert_allclose(
            others[:, :3], np.broadcast_to(TRUE, (5, 3)), rtol=1e-6
        )

    def test_failed_pixel_gives_nan(self, resampled_kww):
        resampled_kww.data[1, 2] = 0.0
        fit = resampled_kww.fit_g2_decay(tau_int=1.0)
        assert np.isnan(fit.data[1, 2]).all()
        np.testing.assert_allclose(fit.data[0, 0, :3], TRUE, rtol=1e-6)

    def test_y_err(self, resampled_kww):
        y_err = OneTimeCorrelationFunction(np.full(resampled_kww.data.shape, 0.01))
        weighted = resampled_kww.fit_g2_decay(tau_int=1.0, y_err=y_err)
        unweighted = resampled_kww.fit_g2_decay(tau_int=1.0)
        # a constant error does not change the least-squares fit
        np.testing.assert_allclose(
            weighted.data[..., :3], unweighted.data[..., :3], rtol=1e-4
        )

    def test_y_err_is_used_as_weights(self, resampled_kww):
        rng = np.random.default_rng(seed=0)
        resampled_kww.data *= 1 + 0.05 * rng.standard_normal(resampled_kww.data.shape)
        y_err = OneTimeCorrelationFunction(1.0 / np.abs(resampled_kww.data))
        weighted = resampled_kww.fit_g2_decay(tau_int=1.0, y_err=y_err)
        unweighted = resampled_kww.fit_g2_decay(tau_int=1.0)
        assert not np.allclose(weighted.data[..., :3], unweighted.data[..., :3])

    def test_y_err_must_be_a_correlation_function(self, resampled_kww):
        with pytest.raises(TypeError, match="OneTimeCorrelationFunction"):
            resampled_kww.fit_g2_decay(tau_int=1.0, y_err=np.ones((2, 3, 60)))

    def test_y_err_navigation_shape(self, resampled_kww):
        y_err = OneTimeCorrelationFunction(np.ones((4, 5, 60)))
        with pytest.raises(ValueError, match="navigation shape"):
            resampled_kww.fit_g2_decay(tau_int=1.0, y_err=y_err)

    def test_lazy(self, resampled_kww):
        eager = resampled_kww.fit_g2_decay(tau_int=1.0)
        lazy = resampled_kww.as_lazy()
        lazy._resampled = True
        lazy._t_rs = resampled_kww._t_rs
        result = lazy.fit_g2_decay(tau_int=1.0)
        assert isinstance(result, LazyOneTimeCorrelationFunction)
        result.compute()
        np.testing.assert_allclose(result.data, eager.data, atol=1e-6)


class TestGetFittedG2:
    def test_requires_a_fit(self, resampled_kww):
        with pytest.raises(RuntimeError, match="fit_g2_decay"):
            resampled_kww.get_fitted_g2()

    def test_reproduces_the_curve(self, resampled_kww):
        fitted = resampled_kww.fit_g2_decay(tau_int=1.0).get_fitted_g2()
        assert isinstance(fitted, OneTimeCorrelationFunction)
        assert fitted.data.shape == resampled_kww.data.shape
        np.testing.assert_allclose(fitted.data, resampled_kww.data, rtol=1e-5)
        # the result is a curve again, not a set of fit parameters
        assert not fitted._fitted_result
        np.testing.assert_allclose(fitted._t_rs, resampled_kww._t_rs)

    def test_failed_fit_gives_a_nan_curve(self, resampled_kww):
        mask = np.ones(resampled_kww.data.shape[:-1], dtype=bool)
        mask[0, 0] = False
        fitted = resampled_kww.fit_g2_decay(tau_int=1.0, mask=mask).get_fitted_g2()
        assert np.isnan(fitted.data[0, 0]).all()
        assert np.isfinite(fitted.data[1, 1]).all()


class TestPlotLog:
    def test_log_time_axis(self, resampled_kww):
        fig, axes = resampled_kww.plot_log()
        try:
            assert axes[-1].get_xscale() == "log"
            # a dotted line at zero marks the baseline
            baselines = [
                line
                for line in axes[-1].lines
                if line.get_linestyle() == ":"
                and np.all(np.asarray(line.get_ydata()) == 0)
            ]
            assert len(baselines) == 1
        finally:
            plt.close("all")
