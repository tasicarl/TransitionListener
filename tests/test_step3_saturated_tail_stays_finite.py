"""The saturated tail of the step-3 profile must stay finite, or the integral stops accumulating.

Percolation step 3 stops computing the action once step 2's fraction has passed `f_final`, on
the grounds that nothing new nucleates below there. That is right: on the 2HDM benchmark
`S_3/T` is already 133 at the saturation point and only grows colder, so the rate is
`exp(-133)` or less whatever the action is.

What it missed is that the region carries the growth of the bubbles already nucleated, and
`percIntegralODE` needs valid samples across it to integrate that growth. Writing `nan` there
fails its validity test, floors the source and transport terms, and the integral stops
accumulating: the benchmark coupling `lambda3 = 5.7918389018` had `I` fall from 50 to 9.7e-38,
so `P = 1 - exp(-I)` rounded to exactly 0 at all 174 support points and a transition whose
percolation temperature step 2 had already found was reported as never percolating.

Note on what this file does not test. A minimal synthetic case does not reproduce the collapse:
with a constant action and a smooth scale factor the integral reaches the same value whether
the tail is finite or not, because the warm region alone already saturates it. The collapse
needs the real `S_3(T)` and the real `a(T)` of a strongly supercooled transition, where the
warm region alone reaches only a fifth of the total. The evidence for the mechanism is
therefore the recorded arrays of that benchmark run, replayed with the inputs crossed, not a
constructed case here. Tuning a synthetic case until it agreed would be fitting the test to the
answer.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptivestepsize as pas


class SaturatedTailTests(unittest.TestCase):
    """Drive the step-3 profile with a fraction that saturates partway down the support."""

    phases = (types.SimpleNamespace(name="sym"), types.SimpleNamespace(name="bro"))

    def profile(self):
        sym, bro = self.phases

        def energy_density(pot, phase, T, include_decoupled=True):
            return (1.0 if phase is sym else 0.0) + T**4

        # an action that varies with temperature, so "the last computed one" is identifiable
        def action(pot, T, ps, pb, outdict, *a, **k):
            return 100.0 + float(np.squeeze(T))

        TSYM = np.linspace(1.0, 0.5, 40)
        # P passes f_final = 0.99 at index 20 and stays there, so the walk saturates
        P = np.where(np.arange(TSYM.size) < 20, 0.5, 0.999)
        n = TSYM.size
        state = bd.PercolationState(TSYM=TSYM, Sr=np.zeros(n), Hr=np.zeros(n),
                                    Pr=np.zeros(n), Tb=np.zeros(n))
        settings = types.SimpleNamespace(f_final=0.99, time_temperature_mode="sound_speed",
                                         integral_method="ode")
        with mock.patch.object(bd, "energyDensity", energy_density), \
                mock.patch.object(bd, "calcAction", action), \
                mock.patch.object(bd, "h_eff_DS", lambda T, pot, phase: 5.0), \
                mock.patch.object(bd, "h_eff_coupled_radiation", lambda T, pot: 0.0), \
                mock.patch.object(bd, "percIntegralODE_full_sweep",
                                  lambda TSYM, *a, **k: (None, P)):
            pas._compute_step3_profile(
                state, TSYM, P, settings, {}, types.SimpleNamespace(conversionFactor=1.0),
                sym, bro, 1.0, 1.0, TBROmin=0.1, TBROmax=5.0, verbose=False)
        return TSYM, state

    def test_the_saturated_tail_carries_a_finite_action(self):
        _, state = self.profile()
        self.assertTrue(np.all(np.isfinite(state.Sr)),
                        "the saturated branch left a non-finite action in Sr, which fails "
                        "percIntegralODE's validity test and floors the integrand")

    def test_the_tail_carries_the_last_computed_action(self):
        TSYM, state = self.profile()
        computed = np.isfinite(state.Sr)
        self.assertTrue(computed.any())
        # the walk saturates partway down, so some entries are carried rather than computed
        carried = state.Sr[22:]
        self.assertTrue(np.allclose(carried, carried[0]),
                        "the carried tail should hold one value, the last computed action")
        self.assertGreater(carried[0], 100.0,
                           "the carried value should be an action, not a placeholder")


if __name__ == "__main__":
    unittest.main()
