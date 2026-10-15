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
    understates a large `x` instead of overstating it. Both branches were measured on a hand-made
    instanton with ``phi_absMin = 1``, ``phi_metaMin = 0``: the two readings agree to 1.8e-09
    relative at every `x` from 0 to 4.6, and at `x` = 4.61, 5, 8 and 15 the `y0` one stays at
    4.60517.

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

    def test_the_last_two_trials_differ_by_a_fraction_of_x(self):
        pot = conformal_u1.specific_potential(
            {"g": 0.46, "y": 0.1, "v_GeV": 1.0}, verbose=False)
        xs = x_of_every_trial(pot, self.T_INT)
        self.assertGreater(len(xs), 2, f"only {len(xs)} trials: the bisection did not run")
        xtol = float(pot.config.tracingConf.tunneling_params
                     .get("tunneling_findProfile_params", {}).get("xtol", 1e-4))
        # The premise, asserted first because without it the two rules agree and the assertion
        # below is empty: what decides is not `x` against `xtol` but `x` against one, since the
        # relative window is `xtol * x` and the absolute one is `xtol`. `x` at the accepted
        # trial is about 4.2e-10 here, so the relative window is ten orders of magnitude the
        # tighter of the two.
        self.assertLess(
            xs[-1], 1e-3,
            msg=f"x at the accepted trial is {xs[-1]:.6g}, which is not far below one, so an "
                f"absolute bracket and a relative one would close at a similar width here")
        # The factor 4 is slack the assertion does not need here: the relative window closes
        # the bracket until the last two trials agree in `x` to double precision, so the gap is
        # exactly 0 at this configuration. It is there for a configuration that stops one
        # halving earlier. With the stopping line put back to `(xmax-xmin) < xtol` the gap is
        # 6.84e-11 against an allowed 1.92e-19, which is what makes this assertion bite.
        width = abs(xs[-1] - xs[-2])
        self.assertLessEqual(
            width, 4.0 * xtol * xs[-1],
            msg=f"the bisection stopped with the last two trials {width:.6g} apart, which is "
                f"{width / xs[-1]:.3g} of x = {xs[-1]:.6g}, against xtol = {xtol:g}")


class TheWindowIsNeverWiderThanTheAbsoluteOne(unittest.TestCase):
    """The scale is capped at one, so the bisection never stops looser than `xtol` alone.

    `xtol * max(|x|, |xmin|)` is tighter than the absolute `xtol` only while that factor stays
    below one. Above one it is the looser of the two, and `x` passes one on a model shipped with
    the code: on the conformal dark U(1) at g = 0.7, y = 0.24, v_GeV = 1.0 the accepted trial
    sits at `x` = 0.98 at T = 95 and 2.02 at T = 135. Without the cap the bisection there
    stopped with the last two trials 1.0114 to 1.1598 times `xtol` apart, at the five
    temperatures from T = 125 to 135. The ``min(..., 1)`` costs one further trial, 37 against
    36, and brings that gap back to 0.5057 to 0.5799 of `xtol`.

    T = 130 is the middle of those five: `x` at the accepted trial is 1.85, so the uncapped
    window would be 1.85 `xtol`, and the uncapped rule stopped 1.0833 `xtol` apart against
    0.5416 `xtol` with the cap. Without the cap the gap therefore exceeds the bound the
    assertion below applies by 8 per cent, and at T = 135, the widest of the five, by 16 per
    cent. T = 130 costs about 2 s.

    The gap between the last two integrated trials is the bracket `(xmax-xmin)` that the
    stopping test read. Both of that bracket's endpoints are trials that were integrated, and
    the trial the loop computes after the test is their midpoint, which the ``break`` leaves
    unintegrated: at T = 130 the gap and the bracket are the same 5.416445070238751e-11 at
    `xtol` = 1e-10.
    """

    T_INT = 130.0

    def test_the_last_two_trials_are_never_more_than_xtol_apart(self):
        pot = conformal_u1.specific_potential(
            {"g": 0.7, "y": 0.24, "v_GeV": 1.0}, verbose=False)
        xs = x_of_every_trial(pot, self.T_INT)
        self.assertGreater(len(xs), 2, f"only {len(xs)} trials: the bisection did not run")
        xtol = float(pot.config.tracingConf.tunneling_params
                     .get("tunneling_findProfile_params", {}).get("xtol", 1e-4))
        # The premise, on the accepted trial rather than on the largest `x` tried: the scale
        # the stopping test reads is `max(|x|, |xmin|)` at the midpoint of the final bracket,
        # so the cap binds at the step that stops the loop only if `x` is above one there. A
        # configuration can try an `x` above one and still stop below it, and then the
        # assertion below would hold whether or not the cap is in the source: at g = 0.50,
        # y = 0.01, v_GeV = 0.14, internal T = 55.43 the largest `x` tried is 1.014 while the
        # accepted one is 0.669, and the two rules give the same action to every digit printed.
        self.assertGreater(
            xs[-1], 1.0,
            msg=f"x at the accepted trial is {xs[-1]:.6g}, not above one, so the cap at one "
                f"cannot bind where the bisection stops and this test is empty")
        width = abs(xs[-1] - xs[-2])
        self.assertLessEqual(
            width, xtol,
            msg=f"the bisection stopped with the last two trials {width:.6g} apart, which is "
                f"{width / xtol:.4g} times xtol = {xtol:g}, so the relative window was the "
                f"looser of the two rules here and the cap at one is not holding")


if __name__ == "__main__":
    unittest.main()
