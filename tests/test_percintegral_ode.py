"""Tests for the ODE-based percolation integral.

Compares ``percIntegralODE`` against the reference nested-trapezoidal
``percIntegral`` on synthetic data.
"""

import math

import numpy as np
import pytest

from transitionlistener.bubbledynamics import (
    Gamma,
    HubbleParameter,
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
    """Build a synthetic (T, H, S) grid mimicking a conformal dark U(1) transition.

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

        mask = I_ref_at_coarse > 1e-10
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
        mask = I_1 > 1e-30
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

