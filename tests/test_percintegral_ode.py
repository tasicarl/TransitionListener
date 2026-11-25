"""Tests for the ODE-based percolation integral.

Compares ``percIntegralODE`` against the reference nested-trapezoidal
``percIntegral`` on synthetic data.
"""

import math

import numpy as np
import pytest

from scipy import integrate

from transitionlistener import errors
from transitionlistener.compiled import percolation as compiled_percolation
from transitionlistener.compiled.splines import PchipInterpolator
from transitionlistener.bubbledynamics import (
    _LOG_GAMMA_SOURCE_FLOOR,
    _PERC_INTEGRAL_STOP,
    Gamma,
    HubbleParameter,
    integrate_broken_temperature,
    logGamma,
    percIntegral,
    percIntegralODE,
    percIntegralODE_full_sweep,
    _log_bag_gamma_source_array,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_grid(n: int = 80, *, seed: int = 42):
    """Build a synthetic (T, H, S) grid mimicking a conformal U(1) with fermions transition.

    T is in internal units (O(1)), S ~ O(100) * T at the hot end (large S/T,
    exponentially suppressed Gamma) decreasing to S ~ O(5) * T near
    percolation.
    """
    rng = np.random.default_rng(seed)
    # Logarithmically spaced temperatures, descending
    T = np.geomspace(50.0, 0.5, n)
    # S3/T profile: large at hot end, decreasing
    s_over_t = 120.0 * (T / T[0]) ** 0.5 + 5.0
    S = s_over_t * T
    # Hubble: radiation-dominated approximation H ~ T^2 / Mpl_internal
    Mpl_internal = 1e6  # some large number
    H = T ** 2 / Mpl_internal
    return T, H, S


# ---------------------------------------------------------------------------
# Tests for _log_bag_gamma_source_array
# ---------------------------------------------------------------------------

class TestLogBagGammaSourceArray:
    def test_basic_finite(self):
        T = np.array([10.0, 5.0, 1.0])
        S = np.array([100.0, 30.0, 8.0])
        H = np.array([1e-4, 5e-5, 1e-5])
        result = _log_bag_gamma_source_array(T, S, H)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_zero_action_gives_neg_inf(self):
        T = np.array([10.0, 5.0])
        S = np.array([0.0, 30.0])
        H = np.array([1e-4, 5e-5])
        result = _log_bag_gamma_source_array(T, S, H)
        assert result[0] == -np.inf
        assert np.isfinite(result[1])

    def test_inf_action_gives_neg_inf(self):
        T = np.array([10.0, 5.0])
        S = np.array([np.inf, 30.0])
        H = np.array([1e-4, 5e-5])
        result = _log_bag_gamma_source_array(T, S, H)
        assert result[0] == -np.inf
        assert np.isfinite(result[1])

    def test_consistency_with_logGamma(self):
        """Bag-limit ln gamma = logGamma - 4 ln T - ln H."""
        T = np.array([10.0, 5.0, 2.0])
        S = np.array([50.0, 20.0, 8.0])
        H = np.array([1e-3, 5e-4, 2e-4])
        log_gamma_source = _log_bag_gamma_source_array(T, S, H)
        log_gamma = logGamma(T, S)
        expected = log_gamma - 4.0 * np.log(T) - np.log(H)
        np.testing.assert_allclose(log_gamma_source, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tests for percIntegralODE vs percIntegral (reference)
# ---------------------------------------------------------------------------

class TestPercIntegralODE:
    """Compare ODE-based I(T) against the reference trapezoidal implementation."""

    def test_single_point(self):
        """With a single grid point, I should be zero."""
        T = np.array([10.0])
        H = np.array([1e-4])
        S = np.array([100.0])
        I_ode = percIntegralODE(T, H, S, vw=1.0)
        assert I_ode.shape == (1,)
        assert I_ode[0] == 0.0

    def test_two_points(self):
        """Minimal non-trivial case — both methods are crude with 2 points,
        but the ODE should still give a finite non-negative result."""
        T = np.array([10.0, 9.0])
        H = np.array([1e-4, 9e-5])
        S = np.array([50.0, 40.0])
        I_ode = percIntegralODE(T, H, S, vw=1.0)
        assert I_ode.shape == (2,)
        assert I_ode[0] == pytest.approx(0.0, abs=1e-20)
        assert I_ode[-1] >= 0.0
        assert np.isfinite(I_ode[-1])

    def test_synthetic_grid_endpoint(self):
        """On a well-resolved synthetic grid, the ODE endpoint should be
        *more accurate* than the trapezoidal reference (converges faster).
        Compare both to a high-resolution trapezoid truth."""
        T, H, S = _synthetic_grid(n=100)
        I_ode = percIntegralODE(T, H, S, vw=1.0)
        I_ref = percIntegral(T, H, S, vw=1.0)
        # Build high-res reference
        T_hr, H_hr, S_hr = _synthetic_grid(n=2000)
        I_truth = percIntegral(T_hr, H_hr, S_hr, vw=1.0)
        if I_truth >= _PERC_INTEGRAL_STOP:
            # The sweep stops once the false vacuum is gone (exp(-I) < exp(-stop));
            # colder points hold the stop value instead of the exact integral.
            assert I_ode[-1] == pytest.approx(_PERC_INTEGRAL_STOP, rel=1e-6)
            return
        err_ode = abs(I_ode[-1] - I_truth) / I_truth
        err_trap = abs(I_ref - I_truth) / I_truth
        assert err_ode < 0.001, f"ODE error {err_ode:.4e} > 0.1%"
        assert err_ode < err_trap, (
            f"ODE ({err_ode:.4e}) should be more accurate than trapezoid ({err_trap:.4e})"
        )

    def test_synthetic_grid_monotonic(self):
        """I(T) must be monotonically non-decreasing as T decreases.
        Allow tiny floating-point noise at the hot end where I ~ 1e-40."""
        T, H, S = _synthetic_grid(n=80)
        I_ode = percIntegralODE(T, H, S, vw=1.0)
        diffs = np.diff(I_ode)
        # Physically relevant region: where I > 1e-20
        mask = I_ode[1:] > 1e-20
        if np.any(mask):
            assert np.all(diffs[mask] >= -1e-15), "I(T) decreased during cooling in the relevant region"

    def test_synthetic_grid_all_points(self):
        """Compare I(T_i) at every grid point against a high-res reference.
        The ODE on a coarse grid should achieve < 0.1% error at all
        physically relevant points (I > 1e-10)."""
        T, H, S = _synthetic_grid(n=60)
        I_ode = percIntegralODE(T, H, S, vw=1.0)

        # Build per-point high-res reference
        T_hr, H_hr, S_hr = _synthetic_grid(n=2000)
        I_hr_all = np.zeros(2000)
        for i in range(2000):
            I_hr_all[i] = percIntegral(T_hr[: i + 1], H_hr[: i + 1], S_hr[: i + 1], vw=1.0)

        # Interpolate the high-res reference to the coarse grid points
        from scipy.interpolate import PchipInterpolator
        interp_hr = PchipInterpolator(T_hr[::-1], I_hr_all[::-1], extrapolate=True)
        I_ref_at_coarse = interp_hr(T[::-1])[::-1]

        # Exact below the stop value (with margin for the coarse-grid reference);
        # at and beyond it the sweep holds the stop value.
        beyond = I_ref_at_coarse > 1.05 * _PERC_INTEGRAL_STOP
        if np.any(beyond):
            assert np.all(I_ode[beyond] >= _PERC_INTEGRAL_STOP * (1.0 - 1e-6))
        mask = (I_ref_at_coarse > 1e-10) & (I_ref_at_coarse < 0.95 * _PERC_INTEGRAL_STOP)
        if np.any(mask):
            rel_err = np.abs(I_ode[mask] - I_ref_at_coarse[mask]) / I_ref_at_coarse[mask]
            assert np.max(rel_err) < 0.005, (
                f"Max relative error {np.max(rel_err):.4e} at index "
                f"{np.argmax(rel_err)}"
            )

    def test_vw_scaling(self):
        """I scales as vw^3."""
        T, H, S = _synthetic_grid(n=40)
        I_1 = percIntegralODE(T, H, S, vw=1.0)
        I_half = percIntegralODE(T, H, S, vw=0.5)
        # The scaling holds wherever neither sweep has reached its stop value.
        mask = (I_1 > 1e-30) & (I_1 < 0.95 * _PERC_INTEGRAL_STOP)
        if np.any(mask):
            np.testing.assert_allclose(
                I_half[mask], I_1[mask] * 0.5**3, rtol=1e-10,
            )

    def test_full_sweep_consistency(self):
        """percIntegralODE_full_sweep should give P = 1 - exp(-I)."""
        T, H, S = _synthetic_grid(n=50)
        I_ode, P_ode = percIntegralODE_full_sweep(T, H, S, vw=1.0)
        P_expected = 1.0 - np.exp(-I_ode)
        np.testing.assert_allclose(P_ode, P_expected, atol=1e-15)

    def test_hot_end_outlier_robustness(self):
        """Regression: a single far-hot temperature causes a huge u-space
        interval that previously crashed DOP853.  The ODE must succeed and
        return an array of the same length as the input."""
        T_body, H_body, S_body = _synthetic_grid(n=49)
        # Prepend one isolated hot-end temperature far above the action peak
        T_outlier = np.array([T_body[0] * 100.0])  # 100x hotter, A ≈ 0 there
        S_outlier = np.array([T_outlier[0] * 400.0])  # S/T = 400 gives negligible gamma source
        H_outlier = T_outlier ** 2 / 1e6
        T = np.concatenate([T_outlier, T_body])
        H = np.concatenate([H_outlier, H_body])
        S = np.concatenate([S_outlier, S_body])
        # Must not crash and must return the same length as input
        I_ode = percIntegralODE(T, H, S, vw=1.0)
        assert I_ode.shape == T.shape, (
            f"Shape mismatch: got {I_ode.shape}, expected {T.shape}"
        )
        # The hot-end outlier should contribute I ≈ 0
        assert I_ode[0] == pytest.approx(0.0, abs=1e-200)
        # The cold-end integral should match a run without the outlier
        I_no_outlier = percIntegralODE(T_body, H_body, S_body, vw=1.0)
        if I_no_outlier[-1] > 1e-20:
            np.testing.assert_allclose(
                I_ode[-1], I_no_outlier[-1], rtol=1e-6,
                err_msg="Hot-end outlier trimming changed the integral value",
            )


class TestPercIntegralTargetEvent:
    """The percolation temperature can be taken from the integrator itself."""

    def test_event_leaves_values_unchanged(self):
        T, H, S = _synthetic_grid(n=60)
        I_plain = percIntegralODE(T, H, S, vw=1.0)
        target = 0.5 * float(I_plain[-1])
        I_event, crossing = percIntegralODE(T, H, S, vw=1.0, i_target=target)
        np.testing.assert_array_equal(I_event, I_plain)
        assert crossing is not None

    def test_crossing_agrees_with_dense_reference(self):
        T, H, S = _synthetic_grid(n=60)
        I_coarse = percIntegralODE(T, H, S, vw=1.0)
        target = 0.3 * float(I_coarse[-1])
        _, crossing = percIntegralODE(T, H, S, vw=1.0, i_target=target)
        # Bracketed by the grid points on either side of the target ...
        above = T[I_coarse < target]
        below = T[I_coarse >= target]
        assert below.max() <= crossing <= above.min()
        # ... and consistent with a finely resolved integral.
        T_hr, H_hr, S_hr = _synthetic_grid(n=2000)
        I_hr = percIntegralODE(T_hr, H_hr, S_hr, vw=1.0)
        order = np.argsort(I_hr)
        T_ref = float(np.interp(np.log(target), np.log(I_hr[order][I_hr[order] > 0]),
                                T_hr[order][I_hr[order] > 0]))
        assert crossing == pytest.approx(T_ref, rel=1e-3)

    def test_unreached_target_gives_none(self):
        T, H, S = _synthetic_grid(n=40)
        I_plain = percIntegralODE(T, H, S, vw=1.0)
        _, crossing = percIntegralODE(T, H, S, vw=1.0, i_target=10.0 * float(I_plain[-1]) + 1.0)
        assert crossing is None

    def test_degenerate_grid_keeps_pair_contract(self):
        for n in (0, 1):
            T = np.linspace(2.0, 1.0, n)
            H = np.ones(n)
            S = np.full(n, 100.0)
            assert isinstance(percIntegralODE(T, H, S), np.ndarray)
            I_values, crossing = percIntegralODE(T, H, S, i_target=1.0)
            assert I_values.shape == (n,)
            assert crossing is None


class TestIntegrateBrokenTemperatureGuards:
    """Inputs the reheating integration cannot work with return None, not raise."""

    def test_too_few_samples(self):
        T = np.array([2.0, 1.5, 1.0])
        P = np.array([0.0, 0.3, 0.9])
        assert integrate_broken_temperature(None, None, None, T, P, 1.2) is None

    def test_target_outside_profile(self):
        T = np.linspace(2.0, 1.0, 10)
        P = np.linspace(0.0, 0.99, 10)
        assert integrate_broken_temperature(None, None, None, T, P, 0.5) is None
        assert integrate_broken_temperature(None, None, None, T, P, 2.5) is None


def _solve_ivp_chain(T, log_source, log_transport, prefactor, target):
    """The chain of J_n with scipy's DOP853, its events and interpolants, at the tolerances of percIntegralODE."""
    u = np.log(T)
    source = PchipInterpolator(u[::-1], log_source[::-1], extrapolate=True)
    transport = PchipInterpolator(u[::-1], log_transport[::-1], extrapolate=True)

    def rhs(v, y):
        t = np.exp(v)
        a, b = float(source(v)), float(transport(v))
        gamma = np.exp(a) if a > _LOG_GAMMA_SOURCE_FLOOR else 0.0
        nu = np.exp(b) if b > _LOG_GAMMA_SOURCE_FLOOR else 0.0
        return [-gamma * t, -y[0] * nu * t, -2.0 * y[1] * nu * t, -3.0 * y[2] * nu * t]

    def reaches(v, y):
        return prefactor * y[3] - target

    def gone(v, y):
        return prefactor * y[3] - _PERC_INTEGRAL_STOP

    gone.terminal = True
    gone.direction = 1.0
    return integrate.solve_ivp(rhs, (u[0], u[-1]), [0.0] * 4, method="DOP853", t_eval=u, rtol=1e-10, atol=1e-300,
                               events=[reaches, gone]), source, transport


def _compiled_chain(T, source, transport, prefactor, target):
    J3 = np.empty(T.size)
    status, reached, crossing, j3_stop = compiled_percolation.chain_integral(
        np.ascontiguousarray(source.x), np.ascontiguousarray(source.c), np.ascontiguousarray(transport.c),
        _LOG_GAMMA_SOURCE_FLOOR, np.log(T), prefactor, target, _PERC_INTEGRAL_STOP, 1e-10, 1e-300, J3)
    return status, reached, crossing, j3_stop, J3


class TestCompiledChain:
    """chain_integral against solve_ivp, on the two ways it ends and on a chain with a closed form."""

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("n, top", [(60, 0.3), (250, 0.3), (250, 40.0), (250, 200.0)])
    def test_it_is_solve_ivp(self, n, top, seed):
        rng = np.random.default_rng(seed)
        T = np.geomspace(150.0, rng.uniform(30.0, 60.0), n)
        H = rng.uniform(0.5, 2.0) * 1e-16 * T**2
        S = T * (140.0 + rng.uniform(800.0, 3000.0) * (T / 100.0 - 1.0))
        S = np.maximum(S, 3.0 * T)
        log_source = np.maximum(_log_bag_gamma_source_array(T, S, H), _LOG_GAMMA_SOURCE_FLOOR)
        log_transport = -np.log(H)
        prefactor = rng.uniform(0.2, 4.0)
        scale = top / max(abs(float(percIntegralODE(T, H, S, vw=1.0)[-1])), 1e-300)
        log_source = np.maximum(log_source + np.log(scale) * (top < _PERC_INTEGRAL_STOP), _LOG_GAMMA_SOURCE_FLOOR)
        target = 0.1 * top * prefactor
        sol, source, transport = _solve_ivp_chain(T, log_source, log_transport, prefactor, target)
        status, reached, crossing, j3_stop, J3 = _compiled_chain(T, source, transport, prefactor, target)
        assert status == sol.status
        assert reached == sol.y.shape[1]
        reference = sol.y[3]
        assert np.all(np.abs(J3[:reached] - reference) <= 1e-7 * np.abs(reference) + 1e-50 * np.max(reference))
        if sol.t_events[0].size:
            assert crossing == pytest.approx(sol.t_events[0][-1], abs=1e-10)
        else:
            assert np.isnan(crossing)
        if status == compiled_percolation.CHAIN_STOPPED:
            assert prefactor * j3_stop == pytest.approx(_PERC_INTEGRAL_STOP, rel=1e-7)
            assert j3_stop == pytest.approx(sol.y_events[1][0][3], rel=1e-7)

    def test_the_closed_form(self):
        """gamma = T^a and nu = T^b: with u(T) = (T_0^p - T^p)/p, J3 = 6 * integral of the nested powers."""
        T = np.geomspace(2.0, 1.0, 40)
        a, b = 3.0, -2.0
        u = np.log(T)
        source = PchipInterpolator(u[::-1], a * u[::-1], extrapolate=True)
        transport = PchipInterpolator(u[::-1], b * u[::-1], extrapolate=True)
        status, reached, crossing, _, J3 = _compiled_chain(T, source, transport, 1.0, np.nan)
        assert status == compiled_percolation.CHAIN_FINISHED and reached == T.size and np.isnan(crossing)
        T0 = T[0]
        # With nu T^-1 dT = T^-2 dT and r = 1/T - 1/T0: J0' = T^3, Jn' = n J_(n-1) in r.
        grid = np.geomspace(T0, T[-1], 400001)
        r = 1.0 / grid - 1.0 / T0
        J0 = (T0**4 - grid**4) / 4.0
        J1 = integrate.cumulative_trapezoid(J0, r, initial=0.0)
        J2 = 2.0 * integrate.cumulative_trapezoid(J1, r, initial=0.0)
        J3_exact = 3.0 * integrate.cumulative_trapezoid(J2, r, initial=0.0)
        assert np.max(np.abs(J3[1:] / np.interp(-T[1:], -grid, J3_exact) - 1.0)) < 1e-6

    def test_a_target_after_the_stop_is_not_reported(self):
        T, H, S = _synthetic_grid(n=200)
        S = S * 0.4
        assert percIntegralODE(T, H, S, vw=1.0)[-1] == pytest.approx(_PERC_INTEGRAL_STOP, rel=1e-7)
        _, below = percIntegralODE(T, H, S, vw=1.0, i_target=_PERC_INTEGRAL_STOP * 0.99)
        _, above = percIntegralODE(T, H, S, vw=1.0, i_target=_PERC_INTEGRAL_STOP * 1.01)
        assert below is not None
        assert above is None

    def test_an_overflowing_rate_is_a_typed_failure(self):
        T = np.geomspace(100.0, 50.0, 30)
        S = 100.0 * T
        with pytest.raises(errors.PercolationError):
            percIntegralODE(T, np.full_like(T, 1e-300), S, vw=1.0)
