"""Change 24, the producer: the controller must compute the verdict, not assert it.

The companion file tests the dataclass hop, which a stub that hardcoded
``estimate["targets_met"] = True`` would survive. This one drives the real branch in
``_try_dynamiczoomwindow_completed_refinement`` and checks the verdict tracks the targets.

What it must NOT assert: that the verdict gates acceptance. It does not -- both exits of that
function return False, which the caller reads as "accept" -- so the targets are advisory and
the verdict is only a report. A test asserting otherwise would encode a falsehood.
"""
import math
import unittest

import numpy as np


def _call(estimate, *, previous=None, acc=1e-2):
    """Run the producer with the accuracy estimate stubbed to a known dict."""
    from transitionlistener import percolation_adaptivestepsize as ad
    from transitionlistener import config
    settings = config.PercolationConf()
    settings.acc_tperc = settings.acc_tfinal = settings.acc_rh = acc
    # `_build_percolation_settings` returns the conf itself after attaching this one
    # runtime-only value, so the real object plus that attribute is exactly what the
    # controller is handed in production.
    settings.max_boundary_n = int(settings.max_boundary_ratio * 100)

    original = ad._estimate_dynamiczoomwindow_accuracy
    ad._estimate_dynamiczoomwindow_accuracy = lambda *a, **k: dict(estimate)
    try:
        n = 8
        return ad._try_dynamiczoomwindow_completed_refinement(
            np.linspace(90.0, 40.0, n), np.full(n, 1e-16),
            np.linspace(0.02, 0.95, n), np.linspace(150.0, 120.0, n),
            settings, None,
            remaining_budget=0, previous_estimate=previous, iterator=5,
            stage="step 3", vw=1.0,
            apply_candidate_grid=lambda *a, **k: False,
            promote_cached_action_points=lambda *a, **k: False,
        )
    finally:
        ad._estimate_dynamiczoomwindow_accuracy = original


def _estimate(rh_unc, *, tf_unc=1e-3, tperc_unc=1e-3, rh=0.05):
    return {"Tperc": 40.0, "Tf": 30.0, "RH": rh,
            "Tperc_uncertainty": tperc_unc, "Tf_uncertainty": tf_unc,
            "RH_uncertainty": rh_unc,
            "Tperc_bracket_width": 1e-3, "Tf_bracket_width": 1e-3}


class AccuracyVerdictProducerTests(unittest.TestCase):
    def test_verdict_is_true_when_every_target_is_met(self):
        _retry, est = _call(_estimate(1e-3), previous=_estimate(1e-3))
        self.assertIn("targets_met", est)
        self.assertIs(est["targets_met"], True)

    def test_verdict_is_false_when_rh_misses_its_target(self):
        """acc_rh = 1% against an RH uncertainty of 27.6%, the measured p90 spread."""
        _retry, est = _call(_estimate(0.276), previous=_estimate(0.276))
        self.assertIs(est["targets_met"], False)

    def test_verdict_is_undecided_when_only_rh_was_never_measured(self):
        """RH is the one target with no reference-free measure, so an infinite RH uncertainty
        beside two met targets is undecided rather than missed.

        This test asserted False until 2026-10-16. False makes the warning fire, and because
        the controller normally reaches this point with nothing to difference RH against, the
        warning fired on every point of every scan: 127 of 127 resolved couplings of the 2HDM
        broad arm, 36 of 36 of its edge arm, and the shipped example point. A column that says
        the same about every point separates none of them.

        None does not weaken the guard this field exists for. The target is still not
        certified, `DIAG:accuracy_rh_uncertainty` is still infinite and says so, and
        `targets_met` is still never True without a comparison. What changed is that "never
        measured" stopped being reported as "measured and missed"."""
        _retry, est = _call(_estimate(math.inf), previous=None)
        self.assertIsNone(est["targets_met"])

    def test_a_measured_miss_is_still_false_without_a_previous_estimate(self):
        """The case a blanket None would have thrown away.

        Tperc and Tf fall back to their own bracket widths, so their uncertainties are
        measured from a single profile and a miss on either is a real miss whether or not RH
        could be compared. On the release scans 21 of the 163 warned points are exactly this:
        10 of the broad arm and 11 of the edge arm, with a worst Tf uncertainty of 14.2% and a
        worst Tperc uncertainty of 4.8% against targets of 1%. The shipped example point is
        one of them, at a Tf uncertainty of 1.03%."""
        _retry, est = _call(_estimate(math.inf, tf_unc=0.142), previous=None)
        self.assertIs(est["targets_met"], False)
        _retry, est = _call(_estimate(math.inf, tperc_unc=0.048), previous=None)
        self.assertIs(est["targets_met"], False)

    def test_a_required_observable_that_is_absent_is_missed_not_undecided(self):
        """An absent value is not an absent reference, for any of the three.

        `None` is reserved for "`acc_rh` could not be judged". A `Tperc`, `Tf` or `RH` that
        came back non-finite is a different matter: nothing about that profile was determined,
        so it is reported as missed. Only `RH` was covered here until the review pointed out
        that the documented meaning of `False` left this case out.
        """
        for observable in ("Tperc", "Tf", "RH"):
            with self.subTest(observable=observable):
                est = _estimate(math.inf)
                est[observable] = math.nan
                _retry, out = _call(est, previous=None)
                self.assertIs(out["targets_met"], False,
                              f"a non-finite {observable} is a missed target, not undecided")

    def test_the_two_verdict_functions_agree_where_rh_was_measured(self):
        """`_dynamiczoomwindow_accuracy_verdict` must not drift from its sibling.

        `_dynamiczoomwindow_accuracy_reached` decides whether the controller may stop
        refining and reads the same three settings a few lines above. The two are allowed to
        differ in exactly one place, an infinite `RH_uncertainty`, so everywhere else they
        have to give the same answer; a threshold edited in one copy only shows up here.

        This also covers the verdict's True return, which no other test reaches: the
        all-targets-met case returns True from the `reached` branch in the caller before the
        verdict is ever called.
        """
        from transitionlistener import percolation_adaptivestepsize as ad
        from transitionlistener import config
        settings = config.PercolationConf()
        settings.acc_tperc = settings.acc_tfinal = settings.acc_rh = 1e-2
        for tperc_unc, tf_unc, rh_unc in (
            (1e-3, 1e-3, 1e-3),      # all met
            (1e-3, 1e-3, 2e-2),      # RH missed
            (2e-2, 1e-3, 1e-3),      # Tperc missed
            (1e-3, 2e-2, 1e-3),      # Tf missed
            (1e-2, 1e-2, 1e-2),      # all exactly on target
        ):
            with self.subTest(tperc=tperc_unc, tf=tf_unc, rh=rh_unc):
                est = _estimate(rh_unc, tf_unc=tf_unc, tperc_unc=tperc_unc)
                verdict = ad._dynamiczoomwindow_accuracy_verdict(est, settings)
                reached = ad._dynamiczoomwindow_accuracy_reached(est, settings)
                self.assertIs(verdict, bool(reached))
        # and the True return really is exercised above
        self.assertIs(
            ad._dynamiczoomwindow_accuracy_verdict(_estimate(1e-3), settings), True)

    def test_a_single_profile_measures_tperc_and_tf_but_not_rh(self):
        """The claim the whole asymmetry rests on, taken from the real estimator.

        Every other test here stubs `_estimate_dynamiczoomwindow_accuracy`, so nothing
        checked that a lone profile really does measure Tperc and Tf. It does, through the
        bracket-width fallback, and the references below are the linearised crossing and the
        sample spacing of the grid this test writes down, not values read off a previous run.
        RH has no such fallback, which is why an infinite `RH_uncertainty` must not be read
        as a miss: `_dynamiczoomwindow_accuracy_reached` is False on this estimate however
        loose the targets are, while the verdict is None.
        """
        import types
        from transitionlistener import percolation_adaptivestepsize as ad
        from transitionlistener import config
        settings = config.PercolationConf()
        pot = types.SimpleNamespace(config=types.SimpleNamespace(
            percolationConf=types.SimpleNamespace(f_final=settings.f_final)))

        temperatures = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0])
        probabilities = np.array([0.0, 0.05, 0.20, 0.40, 0.70, 0.90, 0.98, 0.995])
        spacing = 10.0
        self.assertTrue(0.20 < settings.f_perc < 0.40,
                        "the grid brackets f_perc between its third and fourth sample")
        self.assertTrue(0.98 < settings.f_final < 0.995,
                        "the grid brackets f_final between its seventh and eighth sample")

        est = ad._estimate_dynamiczoomwindow_accuracy(
            temperatures, np.full(temperatures.size, 1e-16), probabilities,
            np.linspace(150.0, 120.0, temperatures.size), settings, pot, previous=None)

        expected_Tperc = 80.0 + (settings.f_perc - 0.20) / (0.40 - 0.20) * (70.0 - 80.0)
        self.assertAlmostEqual(est["Tperc"], expected_Tperc, places=10)
        self.assertAlmostEqual(est["Tperc_uncertainty"], spacing / expected_Tperc, places=12)

        # Tf is solved on the interpolant rather than linearised, so it is pinned to its
        # bracket and to the relation the fallback asserts, not to a number.
        self.assertTrue(30.0 <= est["Tf"] <= 40.0, est["Tf"])
        self.assertAlmostEqual(est["Tf_uncertainty"] * est["Tf"], spacing, places=10)

        self.assertTrue(np.isfinite(est["RH"]), "RH itself was determined")
        self.assertEqual(est["RH_uncertainty"], math.inf,
                         "RH has no bracket measure, so one profile cannot bound it")

        loose = types.SimpleNamespace(acc_tperc=1.0, acc_tfinal=1.0, acc_rh=1.0)
        self.assertFalse(ad._dynamiczoomwindow_accuracy_reached(est, loose),
                         "the conjunction is false on any single profile, which is what made "
                         "the warning fire everywhere")
        self.assertIsNone(ad._dynamiczoomwindow_accuracy_verdict(est, loose))
        self.assertIs(ad._dynamiczoomwindow_accuracy_verdict(est, settings), False,
                      "against the 1% defaults this profile is a measured miss on both Tperc "
                      "and Tf, and must stay one")

    def test_the_verdict_does_not_gate_acceptance(self):
        """Both exits accept. Pinned so a future change cannot quietly make it binding."""
        retry_met, _ = _call(_estimate(1e-3), previous=_estimate(1e-3))
        retry_missed, _ = _call(_estimate(0.276), previous=_estimate(0.276))
        self.assertIs(retry_met, False)
        self.assertIs(retry_missed, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
