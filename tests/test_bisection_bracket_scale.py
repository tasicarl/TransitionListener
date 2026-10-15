"""The shooting bisection converges to a fraction of the `x` it brackets, not to an absolute one.

`findProfile` varies `x`, where the trial profile starts at
``phi(0) = phi_absMin + exp(-x) * delta_phi``, so for a small excursion `x` is the fraction of
the gap between the minima that the bounce covers:
``|phi(0) - phi_metaMin| / |delta_phi| = |exp(-x) - 1| ~ x``. On a supercooled transition that
fraction shrinks with the temperature while an absolute `xtol` does not, so the bisection used
to deliver `x` to a relative precision of `xtol / x`, which degrades decade by decade.

Part of TransitionListener v2
"""
import unittest

import numpy as np

import models.TL_conformal_dark_u1 as conformal_u1
from transitionlistener import pathDeformation
from transitionlistener import tunneling1D


def x_of_every_trial(pot, T_int):
    """Every raw bisection variable `x` the shooting tried, read where `findProfile` sets it.

    `findProfile` hands `delta_phi0 = exp(-x) * delta_phi` to :func:`initialConditions`, so
    ``x = -ln|delta_phi0 / delta_phi|`` recovers it with nothing inverted.

    Reading it instead from the `y0` that :func:`integrateProfile` receives would be wrong:
    `initialConditions` returns the profile at a radius `r0`, not at `r = 0`. The two agree
    while `r0` is still `rmin`, which holds for ``x < -ln(thinCutoff) = 4.605`` at the shipped
    ``thinCutoff = 0.01``. Above that the returned field is pinned to
    ``thinCutoff * delta_phi``, and a `y0`-based reading sticks at 4.605 whatever `x` is, which
    understates a large `x` instead of overstating it. Both branches are measured in
    `scans/v211_validation/shooting_evidence/spy_saturation.log`.

    A trial is still one `integrateProfile` call, so the count here is the count of profiles
    integrated. `findProfile` can call `initialConditions` once more than that, on the
    iteration whose `r0` comes back non-finite and breaks the loop, and that `x` belongs to no
    integrated trial.
    """
    seen = []
    pending = {}
    real_ic = tunneling1D.SingleFieldInstanton.initialConditions
    real_ip = tunneling1D.SingleFieldInstanton.integrateProfile

    def ic_spy(self, delta_phi0, *a, **k):
        delta_phi = float(np.squeeze(self.phi_metaMin - self.phi_absMin))
        ratio = abs(float(np.squeeze(delta_phi0)) / delta_phi)
        pending["x"] = -np.log(max(ratio, 1e-300))
        return real_ic(self, delta_phi0, *a, **k)

    def ip_spy(self, r0, y0, *a, **k):
        if "x" in pending:
            seen.append(pending.pop("x"))
        return real_ip(self, r0, y0, *a, **k)

    tunneling1D.SingleFieldInstanton.initialConditions = ic_spy
    tunneling1D.SingleFieldInstanton.integrateProfile = ip_spy
    try:
        pathDeformation.bounceAction(
            T_int, lambda X, T: pot.Vtot(X, T), lambda X, T: pot.gradV(X, T), {},
            dict(low_vev=np.array([float(pot.v_stable)]), high_vev=np.array([0.0]),
                 Tnuc=T_int, low_phase=1, high_phase=0),
            verbose=False, conversionFactor=pot.conversionFactor,
            **pot.config.tracingConf.tunneling_params)
    finally:
        tunneling1D.SingleFieldInstanton.initialConditions = real_ic
        tunneling1D.SingleFieldInstanton.integrateProfile = real_ip
    return seen


class TheBracketClosesToAFractionOfX(unittest.TestCase):
    """The last two trials must differ by a fraction of `x`, not by an absolute window.

    Measured on the shipped conformal dark U(1) at g = 0.46, y = 0.1, v_GeV = 1.0, internal
    T = 1e-7, through `pathDeformation.bounceAction` with the model's own configured tunnelling
    parameters, which is the entry a scan takes.
    """

    T_INT = 1e-7

    def _trials(self):
        pot = conformal_u1.specific_potential(
            {"g": 0.46, "y": 0.1, "v_GeV": 1.0}, verbose=False)
        xs = x_of_every_trial(pot, self.T_INT)
        xtol = float(pot.config.tracingConf.tunneling_params
                     .get("tunneling_findProfile_params", {}).get("xtol", 1e-4))
        return xs, xtol

    def test_the_relative_window_is_far_tighter_here_than_the_absolute_one(self):
        """The premise: without it the two rules agree and the test below is empty.

        What matters is not `x` against `xtol` but `x` against one, since the relative window
        is `xtol * x` and the absolute one is `xtol`. At this configuration `x` is about
        4.2e-10, so the relative window is ten orders of magnitude tighter.
        """
        xs, xtol = self._trials()
        self.assertTrue(xs, "integrateProfile was never reached")
        self.assertLess(
            xs[-1], 1e-3,
            msg=f"x at the accepted trial is {xs[-1]:.6g}, which is not far below one, so an "
                f"absolute bracket and a relative one would close at a similar width here")

    def test_the_last_two_trials_differ_by_a_fraction_of_x(self):
        xs, xtol = self._trials()
        self.assertGreater(len(xs), 2, f"only {len(xs)} trials: the bisection did not run")
        width = abs(xs[-1] - xs[-2])
        self.assertLessEqual(
            width, 4.0 * xtol * xs[-1],
            msg=f"the bisection stopped with the last two trials {width:.6g} apart, which is "
                f"{width / xs[-1]:.3g} of x = {xs[-1]:.6g}, against xtol = {xtol:g}")


class TheRelativeWindowIsTheTighterOfTheTwoRules(unittest.TestCase):
    """The precondition the change rests on, measured where it has the least room.

    The stopping window is ``xtol * max(|x|, |xmin|)``, so it is tighter than the old absolute
    ``xtol`` only while that factor stays below one. Above one the change would loosen the
    bisection and could leave the answer less converged than an absolute window would.

    `x` is the raw bisection variable, ``phi(0) = phi_absMin + exp(-x) * delta_phi``, so
    ``x > 1`` means the bounce starts more than 63 per cent of the way from the false vacuum to
    the true one. `x` grows as the wall thickens, so the supercooled configuration the class
    above uses is the wrong place to check it: there `x` is 6.8e-10 and the bound has nine
    orders of magnitude of headroom. This uses the thick-walled ``g = 0.7``, ``y = 0.24``
    configuration of `tests/test_1d_tunneling.py`, where `x` reaches 0.144 and the margin is a
    factor of 6.9. It costs about 4 s.
    """

    T_INT = 15.0

    def test_x_stays_below_one(self):
        pot = conformal_u1.specific_potential(
            {"g": 0.7, "y": 0.24, "v_GeV": 1.0}, verbose=False)
        seen = x_of_every_trial(pot, self.T_INT)
        self.assertTrue(seen, "integrateProfile was never reached")
        worst = max(seen)
        self.assertLess(
            worst, 1.0,
            msg=f"the largest x the bisection tried is {worst:.6g}, at or above one, so here "
                f"xtol*x is the looser of the two rules and this change may leave the "
                f"bisection less converged than an absolute xtol would")


if __name__ == "__main__":
    unittest.main()
