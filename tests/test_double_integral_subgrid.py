"""The double integral must not depend on how coarsely the support grid samples the rate.

The adaptive solver places its support points where the rate is large and leaves gaps of up to
several e-folds of temperature on the hot shoulder. Bubbles nucleated there are few but large,
and the percolation integral weighs them with the cube of their radius, so the quadrature
across such a gap matters at the level of tens of per cent. These tests compare percIntegral
and percIntegralODE on gappy grids with the percolation integral of an exponential rate in de
Sitter space, which is known in closed form, and of a rate with a Gaussian turnover, computed
by adaptive quadrature.
"""
import math
import unittest
import warnings

import numpy as np
from scipy import integrate

from transitionlistener import bubbledynamics as bd

H0 = 1e-3


def action_for(T, log_gamma):
    """S with T^4 (S/2 pi T)^{3/2} exp(-S/T) = exp(log_gamma), by Newton iteration in S/T."""
    x = np.full_like(T, 50.0)
    target = log_gamma - 4.0 * np.log(T)
    for _ in range(60):
        f = 1.5 * np.log(x / (2 * np.pi)) - x - target
        x = x - f / (1.5 / x - 1.0)
    return x * T


def de_sitter_grid(u_nodes):
    """Descending temperatures T = exp(-N) and a constant Hubble rate (vacuum domination)."""
    T = np.exp(-np.asarray(u_nodes, dtype=float))
    return T, np.full_like(T, H0)


def gappy_nodes(N_end, beta, gap_efolds=0.7):
    """Dense near the evaluation point, one gap of gap_efolds on the hot shoulder, sparse above."""
    near = np.linspace(N_end - 3.0 / beta, N_end, 25)
    below_gap = near[0] - gap_efolds
    shoulder = np.linspace(below_gap - 1.5, below_gap, 6)
    return np.concatenate((shoulder, near))


def exponential_I(N, N_start, beta):
    """I(N) for Gamma = H0^4 exp(beta N) from N_start on, in de Sitter space, v_w = 1:
    4 pi/3 exp(beta N) int_0^X dx exp(-beta x) (1 - exp(-x))^3, X = N - N_start."""
    X = N - N_start
    s = sum(math.comb(3, k) * (-1) ** k * (1.0 - math.exp(-(beta + k) * X)) / (beta + k) for k in range(4))
    return 4 * np.pi / 3 * math.exp(beta * N) * s


def quadrature_I(N, N_start, log_rate):
    """I(N) for Gamma = H0^4 exp(log_rate(N')) in de Sitter space, by adaptive quadrature."""
    f = lambda Np: math.exp(log_rate(Np)) * (1.0 - math.exp(Np - N)) ** 3
    return 4 * np.pi / 3 * integrate.quad(f, N_start, N, epsabs=0, epsrel=1e-11, limit=400)[0]


class ExponentialRateTests(unittest.TestCase):
    def setUp(self):
        self.beta = 13.0
        # the evaluation point is where I ~ 0.34, i.e. where Tperc is fixed
        self.N_end = math.log(0.34 * self.beta**4 / (8 * np.pi)) / self.beta

    def _grid(self, nodes):
        T, H = de_sitter_grid(nodes)
        S = action_for(T, 4 * math.log(H0) + self.beta * np.asarray(nodes))
        return T, H, S

    def test_the_double_integral_on_a_gappy_grid(self):
        nodes = gappy_nodes(self.N_end, self.beta)
        T, H, S = self._grid(nodes)
        expected = exponential_I(self.N_end, nodes[0], self.beta)
        self.assertLess(abs(bd.percIntegral(T, H, S) / expected - 1), 5e-3)

    def test_the_ode_on_a_gappy_grid(self):
        nodes = gappy_nodes(self.N_end, self.beta)
        T, H, S = self._grid(nodes)
        expected = exponential_I(self.N_end, nodes[0], self.beta)
        self.assertLess(abs(bd.percIntegralODE(T, H, S)[-1] / expected - 1), 5e-3)

    def test_the_double_integral_on_a_dense_grid(self):
        nodes = np.linspace(self.N_end - 3.0, self.N_end, 3001)
        T, H, S = self._grid(nodes)
        expected = exponential_I(self.N_end, nodes[0], self.beta)
        self.assertLess(abs(bd.percIntegral(T, H, S) / expected - 1), 1e-4)

    def test_the_same_history_through_the_generalised_factors(self):
        # a ~ 1/T and 3 c_s^2 = 1 passed explicitly must give the same answer as the bag path
        nodes = gappy_nodes(self.N_end, self.beta)
        T, H, S = self._grid(nodes)
        bag = bd.percIntegral(T, H, S)
        general = bd.percIntegral(T, H, S, scale_factor=T[0] / T, cooling_factor=np.ones_like(T))
        self.assertTrue(np.isclose(general, bag, rtol=1e-12, atol=0))


class GaussianTurnoverTests(unittest.TestCase):
    """ln Gamma = ln H0^4 + beta N - (gamma N)^2 / 2, the shape of a rate near its maximum."""

    def test_double_integral_and_ode_on_a_gappy_grid(self):
        for beta, gamma_over_beta in ((13.0, 0.15), (40.0, 0.1), (8.0, 0.3)):
            gamma = gamma_over_beta * beta
            log_rate = lambda N, b=beta, g=gamma: b * N - 0.5 * (g * N) ** 2
            # evaluation point where I ~ 0.34 (Newton on the exponential estimate is enough)
            N_end = math.log(0.34 * beta**4 / (8 * np.pi)) / beta
            for _ in range(20):
                slope = beta - gamma**2 * N_end
                N_end += (math.log(0.34 * slope**4 / (8 * np.pi)) - log_rate(N_end)) / slope
            nodes = gappy_nodes(N_end, beta)
            T, H = de_sitter_grid(nodes)
            S = action_for(T, 4 * math.log(H0) + np.array([log_rate(n) for n in nodes]))
            expected = quadrature_I(N_end, nodes[0], log_rate)
            with self.subTest(beta=beta, gamma_over_beta=gamma_over_beta):
                self.assertLess(abs(bd.percIntegral(T, H, S) / expected - 1), 5e-3)
                # The ODE interpolates ln(source) with PCHIP, which is off by a few per cent
                # in I across such a gap; with I ~ exp(beta N) that moves Tperc by about
                # 0.05/beta, below 0.5 % for these rates.
                self.assertLess(abs(bd.percIntegralODE(T, H, S)[-1] / expected - 1), 5e-2)


class ResolutionConstantTests(unittest.TestCase):
    """The two sub-grid constants must be converged, not merely tuned to these cases.

    If one of them is later loosened for speed, these tests say so: refining ten times further
    must not move the integral, and the cap on sub-steps per interval must not be what limits
    the resolution actually achieved.
    """

    def _gappy(self, beta, gap):
        N_end = math.log(0.34 * beta**4 / (8 * np.pi)) / beta
        nodes = gappy_nodes(N_end, beta, gap_efolds=gap)
        T, H = de_sitter_grid(nodes)
        S = action_for(T, 4 * math.log(H0) + beta * np.asarray(nodes))
        return T, H, S, np.ones_like(T), T[0] / T

    def test_refining_ten_times_further_does_not_move_the_integral(self):
        for gap in (0.7, 2.5):
            T, H, S, cool, scale = self._gappy(13.0, gap)
            kw = dict(cooling_factor=cool, scale_factor=scale)
            coarse = bd.percIntegral(T, H, S, **kw)
            step = bd._SUBGRID_MAX_LOG_STEP
            try:
                bd._SUBGRID_MAX_LOG_STEP = step / 10.0
                # At a tenth of the production step the cap on sub-steps per interval starts to
                # bind and warns about it, delivering about 0.02 rather than 0.01 on the widest
                # interval. That is still five times finer than production, and the point of the
                # test is that the integral does not move even so.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fine = bd.percIntegral(T, H, S, **kw)
            finally:
                bd._SUBGRID_MAX_LOG_STEP = step
            with self.subTest(gap=gap):
                self.assertLess(abs(coarse / fine - 1), 1e-3)

    def test_the_negligibility_threshold_does_not_bind(self):
        # switching it off may only add sub-points where they do not matter
        for gap in (0.7, 2.5):
            T, H, S, cool, scale = self._gappy(13.0, gap)
            kw = dict(cooling_factor=cool, scale_factor=scale)
            default = bd.percIntegral(T, H, S, **kw)
            negligible = bd._SUBGRID_NEGLIGIBLE
            try:
                bd._SUBGRID_NEGLIGIBLE = np.inf
                everywhere = bd.percIntegral(T, H, S, **kw)
            finally:
                bd._SUBGRID_NEGLIGIBLE = negligible
            with self.subTest(gap=gap):
                self.assertLess(abs(default / everywhere - 1), 1e-3)

    def test_the_cap_on_sub_steps_is_not_what_limits_the_resolution(self):
        for gap in (0.7, 2.5):
            T, H, S, cool, scale = self._gappy(13.0, gap)
            sub = bd._double_integral_subgrid(T, H, S, cool, scale)
            log_source = np.asarray(bd.logGamma(sub[0], sub[2]), dtype=float) \
                + 3.0 * np.log(sub[4]) - np.log(sub[3] * sub[1] * sub[0])
            # only where the source is not negligible: elsewhere the intervals stay whole by design
            relevant = np.maximum(log_source[:-1], log_source[1:]) > \
                np.max(log_source) - bd._SUBGRID_NEGLIGIBLE
            achieved = np.max(np.abs(np.diff(log_source))[relevant])
            with self.subTest(gap=gap):
                self.assertLess(achieved, 1.5 * bd._SUBGRID_MAX_LOG_STEP)


class CapWarningTests(unittest.TestCase):
    """The cap on sub-steps per interval bounds memory, and may not degrade the resolution silently."""

    def test_no_warning_on_the_grids_the_solver_produces(self):
        beta = 13.0
        N_end = math.log(0.34 * beta**4 / (8 * np.pi)) / beta
        for gap in (0.3, 0.7, 1.5, 2.5):
            nodes = gappy_nodes(N_end, beta, gap_efolds=gap)
            T, H = de_sitter_grid(nodes)
            S = action_for(T, 4 * math.log(H0) + beta * np.asarray(nodes))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                bd.percIntegral(T, H, S)
            capped = [w for w in caught if "_SUBGRID_MAX_STEPS" in str(w.message)]
            with self.subTest(gap=gap):
                self.assertEqual(capped, [])

    def test_a_warning_names_the_resolution_actually_achieved(self):
        # one support interval wide enough that the requested step cannot be reached
        beta = 13.0
        N_end = math.log(0.34 * beta**4 / (8 * np.pi)) / beta
        span = 1.2 * bd._SUBGRID_MAX_STEPS * bd._SUBGRID_MAX_LOG_STEP / beta
        nodes = np.concatenate(([N_end - span], np.linspace(N_end - 3.0 / beta, N_end, 25)))
        T, H = de_sitter_grid(nodes)
        S = action_for(T, 4 * math.log(H0) + beta * np.asarray(nodes))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bd.percIntegral(T, H, S)
        messages = [str(w.message) for w in caught if "_SUBGRID_MAX_STEPS" in str(w.message)]
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("instead of the requested", messages[0])


class DeepHistoryTests(unittest.TestCase):
    def test_hundreds_of_efolds_before_the_rate_switches_on(self):
        # a support grid that starts 600 e-folds before the transition; the scale factor
        # spans exp(600) and must not overflow into the radius
        beta = 13.0
        N_end = math.log(0.34 * beta**4 / (8 * np.pi)) / beta
        nodes = np.concatenate((np.linspace(N_end - 600.0, N_end - 4.0, 30), gappy_nodes(N_end, beta)[1:]))
        T, H = de_sitter_grid(nodes)
        S = action_for(T, 4 * math.log(H0) + beta * nodes)
        a = np.exp(nodes - nodes[0])
        I = bd.percIntegral(T, H, S, scale_factor=a, cooling_factor=np.ones_like(T))
        self.assertTrue(np.isfinite(I))
        self.assertLess(abs(I / exponential_I(N_end, nodes[0], beta) - 1), 5e-3)


if __name__ == "__main__":
    unittest.main()
