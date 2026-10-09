"""Change 24: the percolation accuracy verdict must survive the solve.

The controller's accuracy estimate used to be a loop-local that the solver dropped at
`return state`, so a profile that missed `acc_rh` was indistinguishable in the output from one
that met it. Measured consequence: RH differed by a 90th percentile of 27.6% between two
accepted profiles, both nominally held to acc_rh = 1%, while Tperc agreed to better than 1%.

The targets do not gate acceptance -- both exits of
`_try_dynamiczoomwindow_completed_refinement` return False, which the caller reads as "accept"
-- so these fields are the only record of whether they were honoured. A test that only checked
"True propagates" would pass against a stub that hardcoded True, so the three states that must
be distinguishable are checked separately: met, missed, and never assessed.
"""
import math
import unittest

import numpy as np


def _state(estimate):
    from transitionlistener.bubbledynamics import PercolationState
    T = np.array([90.0, 80.0, 70.0])
    s = PercolationState(TSYM=T, Sr=np.array([150.0, 140.0, 130.0]),
                         Hr=np.array([1e-16, 1e-16, 1e-16]),
                         Pr=np.array([0.01, 0.3, 0.9]), Tb=T.copy())
    s.support_bank = T.copy()
    s.action_temperatures = T.copy()
    s.accuracy_estimate = estimate
    return s


def _grid():
    from transitionlistener.bubbledynamics import PercolationGrid
    T = np.array([90.0, 80.0, 70.0])
    return PercolationGrid(TSYM=T, Tstart=90.0, TpercApprox=80.0, tmin=70.0, tmax=90.0,
                           TBROmin=70.0, TBROmax=90.0)


def _build(estimate):
    from transitionlistener.bubbledynamics import PercolationDiagnostics
    return PercolationDiagnostics.from_state(_state(estimate), _grid(), None, {})


class AccuracyVerdictTests(unittest.TestCase):
    def test_targets_met_is_reported_true(self):
        d = _build({"targets_met": True, "Tperc_uncertainty": 1e-3,
                    "Tf_uncertainty": 2e-3, "RH_uncertainty": 3e-3})
        self.assertIs(d.accuracy_targets_met, True)

    def test_targets_missed_is_reported_false_not_none(self):
        """A profile that missed its targets must be distinguishable from one never judged."""
        d = _build({"targets_met": False, "Tperc_uncertainty": 0.5,
                    "Tf_uncertainty": 0.5, "RH_uncertainty": 0.5})
        self.assertIs(d.accuracy_targets_met, False)

    def test_no_estimate_means_no_verdict(self):
        """None means the controller never reached a comparable second estimate."""
        d = _build(None)
        self.assertIsNone(d.accuracy_targets_met)

    def test_estimate_without_the_key_means_no_verdict(self):
        """A verdict must not be invented when the key is absent."""
        d = _build({"Tperc_uncertainty": 1e-3, "Tf_uncertainty": 1e-3, "RH_uncertainty": 1e-3})
        self.assertIsNone(d.accuracy_targets_met)

    def test_the_three_uncertainties_are_carried(self):
        """The verdict alone does not say how badly a target was missed."""
        d = _build({"targets_met": False, "Tperc_uncertainty": 0.011,
                    "Tf_uncertainty": 0.022, "RH_uncertainty": 0.276})
        self.assertAlmostEqual(d.accuracy_tperc_uncertainty, 0.011)
        self.assertAlmostEqual(d.accuracy_tfinal_uncertainty, 0.022)
        self.assertAlmostEqual(d.accuracy_rh_uncertainty, 0.276)

    def test_uncertainties_are_nan_when_absent(self):
        d = _build(None)
        for v in (d.accuracy_tperc_uncertainty, d.accuracy_tfinal_uncertainty,
                  d.accuracy_rh_uncertainty):
            self.assertTrue(math.isnan(v), f"expected nan, got {v}")

    def test_the_rest_of_the_snapshot_still_works(self):
        """The new fields must not disturb what the diagnostics already reported."""
        d = _build({"targets_met": True, "Tperc_uncertainty": 0.0,
                    "Tf_uncertainty": 0.0, "RH_uncertainty": 0.0})
        self.assertEqual(d.start_temperature, 90.0)
        self.assertEqual(d.final_active_support_points, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
