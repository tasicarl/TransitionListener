"""T_p, T_reh and (beta/H)_RH must not depend on where the adaptive support points landed.

Four mechanisms can make them step at fixed parameters, and each is excluded here:

1. interpolating ``ln gamma`` between support points with a shape-preserving spline flattens
   the peak of the rate whenever no sample sits on it (`percIntegralODE` evaluates the rate on
   a lattice in ``ln T`` from a cubic spline of ``S/T``, and the action interpolant of the
   observables reads the same spline);
2. step 3 must not return the profile of a sweep that followed a support rebuild in its last
   iteration, computed from a zeroed P(T), i.e. with no reheating in H(T);
3. the hot end of the support must not be accepted on ``Gamma/H^4 < 1`` there alone, since
   bubbles nucleated above it can fill percent of I(T_p) after several e-folds of supercooling;
4. the reheated broken-phase temperature must not add each interval's converted energy at its
   cold end, a first-order error in the width of the interval.

Also the convergence measure of T_p and T_f: the difference between the crossing on the full
and on the halved support, not the width of the bracket, which is a grid spacing.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import types
import unittest

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import config
from transitionlistener import percolation_adaptivestepsize as ad
from transitionlistener.transitionObservables import _BoundedPchipInterpolator, build_action_spline

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
F_PERC = config.PercolationConf.f_perc
TARGET = -math.log(1.0 - F_PERC)
U_PEAK = math.log(95.0)


def _exponent(u):
    """S/T with its minimum, and so the peak of the rate at constant H, at T = 95."""
    return 187.7 + 60.0 * (np.asarray(u, dtype=float) - U_PEAK) ** 2


def _support(gap_point: bool, hot_step: float = 0.03):
    """A support shaped like the dark U(1) one: coarse above the peak, a 5 % gap across it."""
    u = np.concatenate((
        np.arange(math.log(214.0), math.log(97.5), -hot_step),
        [U_PEAK + 0.006] if gap_point else [],
        np.arange(math.log(92.7), math.log(11.4), -0.008),
    ))
    T = np.exp(u)
    return T, np.full_like(T, 1.15e-18), _exponent(u) * T


def _tp(T, H, S):
    return bd.percIntegralODE(T, H, S, vw=1.0, i_target=TARGET)[1]


class RateBetweenSupportPointsTests(unittest.TestCase):
    def test_a_point_in_the_gap_across_the_rate_peak_moves_tp_only_at_the_spline_error(self):
        """With the old interpolant of ln gamma the one point moved T_p by 1.7e-3, here 1.4e-5."""
        without = _tp(*_support(False))
        with_point = _tp(*_support(True))
        self.assertLess(abs(with_point / without - 1.0), 1e-4)

    def test_tp_is_stable_under_halving_the_support(self):
        T, H, S = _support(True)
        full = _tp(T, H, S)
        keep = ad._halved_support_mask(T.size)
        self.assertLess(abs(_tp(T[keep], H[keep], S[keep]) / full - 1.0), 2e-4)

    def test_the_lattice_is_fixed_in_ln_t(self):
        """Two ranges share every interior node, whatever their ends."""
        a = bd._rate_lattice(1.00013, 2.5)[1:-1]
        b = bd._rate_lattice(0.9, 2.4991)[1:-1]
        du = bd._RATE_LATTICE_DU
        np.testing.assert_allclose(a / du, np.round(a / du), rtol=0, atol=1e-6)
        shared = np.intersect1d(a, b)
        self.assertGreater(shared.size, 0.9 * min(a.size, b.size))

    def test_a_floored_source_keeps_the_old_interpolant(self):
        """Samples that carry no rate are not read through the spline."""
        u = np.linspace(0.0, 1.0, 12)
        S = np.exp(u) * 100.0
        source = np.full(12, -5.0)
        source[:4] = bd._LOG_GAMMA_SOURCE_FLOOR
        nodes, values, _ = bd._rate_on_log_lattice(u, S, source, np.zeros(12))
        cold = nodes < u[3]
        self.assertTrue(np.all(values[cold] <= -5.0 + 1e-9))


class ActionInterpolantTests(unittest.TestCase):
    def test_the_action_between_samples_follows_the_spline_of_s_over_t(self):
        T, _, S = _support(False)
        spline = build_action_spline(T, S)
        pchip = _BoundedPchipInterpolator(T, S, fill_low=float(S[-1]), fill_high=np.inf)
        probe = np.exp(np.linspace(math.log(97.0), math.log(93.0), 21))
        exact = _exponent(np.log(probe))
        error = np.max(np.abs(spline(probe) / probe - exact))
        self.assertLess(error, 1e-3, error)
        # The shape-preserving spline is what made (beta/H)_RH step: it is off by far more.
        self.assertGreater(np.max(np.abs(pchip(probe) / probe - exact)), 10 * error)

    def test_the_samples_are_returned_at_the_support_temperatures(self):
        T, _, S = _support(True)
        np.testing.assert_array_equal(build_action_spline(T, S)(T), S)

    def test_the_bubble_separation_does_not_depend_on_one_point_in_the_gap(self):
        def separation(gap_point):
            T, H, S = _support(gap_point)
            I = bd.percIntegralODE(T, H, S, vw=1.0)
            P = 1.0 - np.exp(-I)
            Pint = _BoundedPchipInterpolator(T, P, fill_low=1.0, fill_high=0.0, clip=(0.0, 1.0),
                                             transform="log_false_fraction")
            Hint = _BoundedPchipInterpolator(T, H, fill_low=float(H[-1]), fill_high=float(H[0]))
            tp = _tp(T, H, S)
            return bd.calcMeanBubbleSeparation(tp, float(T[0]), build_action_spline(T, S), Pint, Hint)

        self.assertLess(abs(separation(True) / separation(False) - 1.0), 2e-4)

    def test_a_noisy_near_coincident_pair_does_not_set_the_slope(self):
        """The adaptive support has samples 1e-6 T apart, with the noise of the bounce solver."""
        u = np.linspace(0.0, 1.0, 11)
        u = np.sort(np.concatenate((u, [0.5 + 1e-6])))
        x = 100.0 + 5.0 * u
        x[np.argmin(np.abs(u - (0.5 + 1e-6)))] += 1e-2
        spline = bd.ActionExponentSpline(u, x * np.exp(u))
        probe = np.linspace(0.32, 0.68, 13)
        self.assertLess(np.max(np.abs(spline(probe) - (100.0 + 5.0 * probe))), 2e-2)


class HotTailTests(unittest.TestCase):
    @staticmethod
    def _supercooled(t_hot):
        """Vacuum domination (constant H) with a rate rising slowly as T falls, from t_hot down."""
        u = np.arange(math.log(t_hot), math.log(1e-8), -0.02)
        T = np.exp(u)
        x = 91.0 + 4.0 * (u - math.log(5e-8))
        return T, np.full_like(T, 1.7e-17), x * T

    def test_a_support_starting_inside_the_nucleation_region_is_measured(self):
        settings = config.PercolationConf()
        truncated = ad._hot_tail_share(*self._supercooled(6e-7), settings)
        self.assertGreater(truncated, 1e-3)

    def test_a_support_starting_where_the_rate_is_dead_is_not(self):
        settings = config.PercolationConf()
        T, H, S = _support(False)
        self.assertLess(abs(ad._hot_tail_share(T, H, S, settings)), ad._HOT_TAIL_SHARE_LIMIT)

    def test_the_controller_promotes_hotter_cached_actions_for_it(self):
        settings = config.PercolationConf()
        T, H, S = self._supercooled(6e-7)
        P = 1.0 - np.exp(-bd.percIntegralODE(T, H, S, vw=1.0))
        calls = []

        def promote_hot():
            calls.append("hot")
            return True

        decision, *_ = ad._dynamiczoomwindow_post_sweep_controller(
            T, H, P, S, settings, None, stage="step 3", iterator=1, remaining_budget=10,
            previous_estimate=None, vw=1.0, hot_bound=1.0,
            apply_candidate_grid=lambda *a, **k: False,
            promote_cached_action_points=lambda *a, **k: False,
            promote_hot_head_cached_points=promote_hot,
        )
        self.assertEqual(calls, ["hot"])
        self.assertEqual(decision, "retry")


def _smooth_profile():
    """The support above with a Hubble rate low enough for the transition to complete."""
    T, H, S = _support(True)
    H = 0.7 * H
    I = bd.percIntegralODE(T, H, S, vw=1.0)
    return T, H, 1.0 - np.exp(-I), S


def _pot(settings):
    return types.SimpleNamespace(config=types.SimpleNamespace(
        percolationConf=types.SimpleNamespace(f_final=settings.f_final)))


class TpConvergenceMeasureTests(unittest.TestCase):
    def test_the_uncertainty_measures_the_support_not_its_spacing(self):
        settings = config.PercolationConf()
        T, H, P, S = _smooth_profile()
        estimate = ad._estimate_dynamiczoomwindow_accuracy(T, H, P, S, settings, _pot(settings))
        bracket = estimate["Tperc_bracket_width"] / estimate["Tperc"]
        self.assertGreater(bracket, 5e-3)
        self.assertLess(estimate["Tperc_uncertainty"], 1e-3)
        self.assertLess(estimate["Tperc_uncertainty"], settings.acc_tperc)
        self.assertTrue(np.isfinite(estimate["Tf_half_support"]))
        self.assertLess(estimate["Tf_uncertainty"], settings.acc_tfinal)

    def test_the_crossings_are_the_integrators(self):
        settings = config.PercolationConf()
        T, H, P, S = _smooth_profile()
        estimate = ad._estimate_dynamiczoomwindow_accuracy(T, H, P, S, settings, _pot(settings))
        self.assertAlmostEqual(estimate["Tperc_ode"] / _tp(T, H, S), 1.0, places=12)

    def test_the_drift_since_the_previous_iteration_compares_the_integrators_crossings(self):
        """The linear crossing moves with the support; a drift of the integrator's one is real."""
        settings = config.PercolationConf()
        T, H, P, S = _smooth_profile()
        first = ad._estimate_dynamiczoomwindow_accuracy(T, H, P, S, settings, _pot(settings))
        previous = dict(first, Tperc_ode=first["Tperc_ode"] * (1.0 + 3e-3))
        second = ad._estimate_dynamiczoomwindow_accuracy(
            T, H, P, S, settings, _pot(settings), previous=previous)
        self.assertAlmostEqual(second["Tperc_uncertainty"] * second["Tperc"] / first["Tperc_ode"], 3e-3, places=6)

    def _completed(self, acc):
        settings = config.PercolationConf()
        settings.acc_tperc = settings.acc_tfinal = acc
        T, H, P, S = _smooth_profile()
        reasons = []

        def apply(grid, reason):
            reasons.append(reason)
            return reason == "accuracy_refine"

        previous = {"Tperc": math.nan, "Tf": math.nan, "RH": math.nan}
        retry, _ = ad._try_dynamiczoomwindow_completed_refinement(
            T, H, P, S, settings, _pot(settings), remaining_budget=10, previous_estimate=previous,
            iterator=settings.maxit - 1, stage="step 3", vw=1.0,
            apply_candidate_grid=apply, promote_cached_action_points=lambda reason: False)
        return retry, reasons

    def test_a_measured_miss_refines_the_transition_band(self):
        retry, reasons = self._completed(1e-12)
        self.assertTrue(retry)
        self.assertIn("accuracy_refine", reasons)

    def test_a_met_target_does_not(self):
        _, reasons = self._completed(1e-2)
        self.assertNotIn("accuracy_refine", reasons)

    def test_the_refinement_grid_lies_in_the_band(self):
        settings = config.PercolationConf()
        T, _, P, _ = _smooth_profile()
        grid = ad._build_accuracy_refine_grid(T, P, settings, max_new_points=5)
        self.assertEqual(grid.size, 5)
        band = T[(P >= settings.f_start) & (P <= settings.f_final)]
        self.assertTrue(np.all((grid < band.max() * 1.2) & (grid > band.min() / 1.2)))


_POINT = r"""
import json, os, sys
os.chdir(sys.argv[1]); sys.path.insert(0, "models")
from transitionlistener.interface import _compute_single_point
from transitionlistener.helper_functions import load_potential
model, params = sys.argv[2], json.loads(sys.argv[3])
base = load_potential(model, "specific_potential")
def factory(p, *a, **kw):
    pot = base(p, *a, **kw)
    if "dark_U1" in model:
        pot.config.gwConf.wall_velocity = "c"
    return pot
ctx, err = _compute_single_point(params, factory, timeout=-1, verbose=False, include_smbhb=False, max_stage=3)
obs = ctx["strongest_transition_observables"]
print("RESULT " + json.dumps({k: float(obs[k]) for k in ("Tperc", "betaH_RH", "Treh")}))
"""


def _run_point(model, params):
    """One point per process: several runs in one interpreter leak state."""
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-c", _POINT, REPO, model, json.dumps(params)],
                         capture_output=True, text=True, env=env, timeout=900)
    lines = [line for line in out.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        raise AssertionError(out.stderr[-2000:])
    return json.loads(lines[-1][len("RESULT "):])


class EndToEndTests(unittest.TestCase):
    """Points whose values one of the four mechanisms would move.

    The references are converged values: for the dark U(1) point the step-3 profile recomputed
    on the support with one and two rounds of midpoints inserted agrees to 5e-6 in T_p; the
    end-point conversion of the reheating or the interpolation of ``ln gamma`` would each move
    it by 3.0e-3. For the two conformal nodes, scans in the neighbouring parameters are smooth
    to 4e-5 in T_p around them; the unreheated profile of the last sweep would move one by
    -6.4 %, and a support starting inside the nucleation region the other by -3.7 %. Internal
    units throughout.
    """

    def test_dark_u1_next_to_the_stall_edge(self):
        r = _run_point("models/TL_dark_U1_g_parameterization.py", {"g": 1.0, "lambda": 0.0180296, "v_GeV": 0.1})
        self.assertAlmostEqual(r["Tperc"] / 42.5055758 - 1.0, 0.0, delta=5e-4)
        self.assertAlmostEqual(r["betaH_RH"] / 1.7061010 - 1.0, 0.0, delta=2e-3)

    def test_conformal_u1_node_whose_last_sweep_followed_a_rebuild(self):
        r = _run_point("models/TL_conformal_u1_no_fermions_3d.py", {"g": 0.41, "v_GeV": 10 ** 0.95625})
        self.assertAlmostEqual(r["Tperc"] / 4.757429e-08 - 1.0, 0.0, delta=2e-3)
        self.assertAlmostEqual(r["betaH_RH"] / 2.668433 - 1.0, 0.0, delta=5e-3)

    def test_conformal_u1_node_whose_support_started_inside_the_nucleation_region(self):
        r = _run_point("models/TL_conformal_u1_no_fermions_3d.py", {"g": 0.41, "v_GeV": 10 ** 0.909375})
        self.assertAlmostEqual(r["Tperc"] / 6.018715e-08 - 1.0, 0.0, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
