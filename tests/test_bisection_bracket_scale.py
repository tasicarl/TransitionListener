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
        seen = []
        real = tunneling1D.SingleFieldInstanton.integrateProfile

        def spy(self, r0, y0, *a, **k):
            delta_phi = float(self.phi_metaMin - self.phi_absMin)
            seen.append(dict(x=abs(float(y0[0]) - float(self.phi_metaMin)) / abs(delta_phi),
                             xtol=None))
            return real(self, r0, y0, *a, **k)

        tunneling1D.SingleFieldInstanton.integrateProfile = spy
        try:
            pathDeformation.bounceAction(
                self.T_INT, lambda X, T: pot.Vtot(X, T), lambda X, T: pot.gradV(X, T), {},
                dict(low_vev=np.array([float(pot.v_stable)]), high_vev=np.array([0.0]),
                     Tnuc=self.T_INT, low_phase=1, high_phase=0),
                verbose=False, conversionFactor=pot.conversionFactor,
                **pot.config.tracingConf.tunneling_params)
        finally:
            tunneling1D.SingleFieldInstanton.integrateProfile = real
        xtol = float(pot.config.tracingConf.tunneling_params
                     .get("tunneling_findProfile_params", {}).get("xtol", 1e-4))
        return [t["x"] for t in seen], xtol

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


if __name__ == "__main__":
    unittest.main()
