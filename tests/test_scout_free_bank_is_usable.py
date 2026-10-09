"""The step-1 scout's free support bank must hold only usable support points.

``_build_step1_approx_profile`` returns two temperature sets, and they mean different
things:

* ``sampled_temperatures`` records every temperature the scout *probed*, failures
  included, so the scout does not probe the same dead end twice;
* ``temperatures`` holds those whose action came back usable, and it is the set that
  carries actions, Hubble rates and probabilities.

``_initial_temperature_grid`` seeds step 2's ``free_support_bank`` from one of them. The
free bank means "support that costs no new action budget", so the controller activates
those points freely. Seeding it from the *probed* set hands step 2 the scout's failed
cold probe as free support, and the active grid's cold boundary is then planted there.

Traced at lambda3 = 5.8123128723 (clean main): the scout probes 3.8032e-06 GeV, and the
very next rebuild is

    reason=cached_action_points, old=[2.2235, 62.279] -> new=[3.8032e-06, 68.772]

after which ``rate_step_refine`` fills the span in one step (29 -> 75 points). The final
bank holds 75 finite-action points down to 25.4 GeV and 56 non-finite ones below, a grid
running 6.7e+06 times colder than the last usable action, against a transition at
20-40 GeV.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders


def settings(**overrides):
    values = dict(f_start=1.0e-3, f_final=0.99, maxit=20, rel_increment=0.1,
                  max_action_temperatures=100, n_action_increment=5,
                  time_temperature_mode="sound_speed", integral_method="ode",
                  entropy_definition=None)
    values.update(overrides)
    return types.SimpleNamespace(**values)


# finite actions in a band, nothing computable far colder -- the measured structure
USABLE = [76.4, 62.0, 41.1, 31.7, 25.4]
DEAD_PROBE = 3.8032e-06


class ScoutProfileSetsTests(unittest.TestCase):
    """The two temperature sets must keep their distinct meanings."""

    def _profile(self):
        from transitionlistener import bubbledynamics as bd
        outdict = {float(T): {"action": 150.0 * T, "trantype": 1} for T in USABLE}
        outdict[DEAD_PROBE] = {"action": np.nan, "trantype": 1}

        def cached(od, ps, pb, *, tmin=None, tmax=None, require_finite_action=True):
            out = [float(k) for k, v in od.items() if isinstance(v, dict) and "action" in v
                   and (not require_finite_action or np.isfinite(v["action"]))]
            return np.asarray(sorted(out, reverse=True), dtype=float)

        def action(pot, T, ps, pb, od, *a, **k):
            return float(od[float(T)]["action"])

        pot = types.SimpleNamespace(conversionFactor=1.0)
        with mock.patch.object(bd, "_transition_action_outdict_temperatures", cached), \
             mock.patch.object(bd, "calcAction", action), \
             mock.patch.object(bd, "HubbleParameter", lambda *a, **k: 1.0e-16), \
             mock.patch.object(bd, "energyDensity", lambda *a, **k: 1.0), \
             mock.patch.object(bd, "Gamma", lambda T, S: np.full(np.size(T), 1.0e-40)), \
             mock.patch.object(bd, "logGamma", lambda T, S: np.full(np.size(T), -92.0)), \
             mock.patch.object(bd, "percIntegralODE_full_sweep",
                               lambda *a, **k: (np.zeros(len(a[0])), np.zeros(len(a[0])))):
            return gridbuilders._build_step1_approx_profile(
                outdict, pot, None, None, 1.0,
                tmin=1.0e-13, tmax=615.0, tperc_approx=35.0, settings=settings(),
                sampled_temperatures=np.asarray(USABLE + [DEAD_PROBE], dtype=float),
            )

    def test_the_probed_set_remembers_the_dead_end(self):
        """Control: the scout must not forget it already tried that temperature."""
        profile = self._profile()
        self.assertIsNotNone(profile, "the profile could not be built at all")
        self.assertTrue(
            np.any(np.isclose(profile.sampled_temperatures, DEAD_PROBE, rtol=0.0, atol=1e-12)),
            "the probed set dropped a failed probe, so the scout could retry it",
        )

    def test_the_usable_set_excludes_the_dead_end(self):
        """The invariant the free bank relies on: no unusable point is 'free support'."""
        profile = self._profile()
        self.assertIsNotNone(profile)
        self.assertFalse(
            np.any(np.isclose(profile.temperatures, DEAD_PROBE, rtol=0.0, atol=1e-12)),
            f"a temperature whose action is non-finite appeared in the usable set; "
            f"seeding the free support bank from it plants the active grid's cold "
            f"boundary at {DEAD_PROBE:g} GeV",
        )
        self.assertEqual(
            sorted(float(t) for t in profile.temperatures), sorted(USABLE),
            "the usable set must be exactly the finite-action temperatures",
        )


def _scout_profile(usable, probed):
    """A step-1 scout result whose two temperature sets deliberately disagree."""
    usable = np.asarray(usable, dtype=float)
    return gridbuilders._Step1ApproxProfile(
        sampled_temperatures=np.asarray(probed, dtype=float),
        temperatures=usable,
        actions=150.0 * usable,
        gamma=np.full(usable.size, 1.0e-40),
        hubble_sym=np.full(usable.size, 1.0e-16),
        log_gamma_h4_sym=np.full(usable.size, -92.0),
        i_approx=np.zeros(usable.size),
        probabilities=np.zeros(usable.size),
        tperc_approx=35.0,
        startup_center=35.0,
        window_high=76.4,
        window_low=25.4,
        window_source="test",
    )


class FreeSupportBankWiringTests(unittest.TestCase):
    """The wiring: whichever set ``_initial_temperature_grid`` hands over is what step 2 gets.

    Testing the two sets alone cannot catch this. Swapping the argument back to
    ``sampled_temperatures`` leaves every assertion above passing, because the free bank is
    built one function further on.
    """

    def _grid(self, active_grid):
        from transitionlistener import bubbledynamics as bd

        profile = _scout_profile(USABLE, USABLE + [DEAD_PROBE])
        pot = types.SimpleNamespace(conversionFactor=1.0)
        phase = types.SimpleNamespace(Tmin=1.0e-13, Tmax=615.0)
        with mock.patch.object(gridbuilders, "_highest_tsym_with_tb_root",
                               lambda *a, **k: 615.0), \
             mock.patch.object(gridbuilders, "_extend_step1_scout_until_bracketed",
                               lambda *a, **k: profile), \
             mock.patch.object(gridbuilders, "_build_step1_active_grid",
                               lambda *a, **k: active_grid), \
             mock.patch.object(bd, "calcApproxPercolation", lambda *a, **k: 35.0):
            return gridbuilders._initial_temperature_grid(
                {}, pot, 615.0, phase, phase, 1.0, 20,
                settings(n_action_min=10, weight=0.5),
                1.0e-13, 615.0, False,
            )

    def test_the_compact_startup_grid_hands_over_only_usable_support(self):
        grid = self._grid(np.asarray([62.0, 41.1, 31.7], dtype=float))
        bank = np.asarray(grid.free_support_bank, dtype=float)
        self.assertFalse(
            np.any(np.isclose(bank, DEAD_PROBE, rtol=0.0, atol=1e-12)),
            f"step 2 was handed {DEAD_PROBE:g} GeV as free support, so the active grid's "
            "cold boundary can be planted there at no apparent cost",
        )
        self.assertEqual(sorted(float(t) for t in bank), sorted(USABLE))

    def test_the_broad_fallback_grid_hands_over_only_usable_support(self):
        """The same choice is made a second time where no compact window was found."""
        grid = self._grid(None)
        bank = np.asarray(grid.free_support_bank, dtype=float)
        self.assertFalse(
            np.any(np.isclose(bank, DEAD_PROBE, rtol=0.0, atol=1e-12)),
            "the fallback return still seeds the free bank from the probed set",
        )
        self.assertEqual(sorted(float(t) for t in bank), sorted(USABLE))


if __name__ == "__main__":
    unittest.main()
