"""Change 24: the percolation accuracy verdict must survive the solve.

The controller's accuracy estimate used to be a loop-local that the solver dropped at
`return state`, so a profile that missed `acc_rh` was indistinguishable in the output from one
that met it. Measured consequence: RH differed by a 90th percentile of 27.6% between two
accepted profiles, both nominally held to acc_rh = 1%, while Tperc agreed to better than 1%.

The targets do not gate acceptance -- both exits of
`_try_dynamiczoomwindow_completed_refinement` return False, which the caller reads as "accept"
-- so these fields are the only record of whether they were honoured. A test that only checked
"True propagates" would pass against a stub that hardcoded True, so the three states that must
be distinguishable are checked separately: met, not certified, and no verdict at all.
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
        """An uncertified profile must be distinguishable from one that was never judged."""
        d = _build({"targets_met": False, "Tperc_uncertainty": 0.5,
                    "Tf_uncertainty": 0.5, "RH_uncertainty": 0.5})
        self.assertIs(d.accuracy_targets_met, False)

    def test_no_estimate_means_no_verdict(self):
        """None means no verdict was formed at all, not that a target was missed."""
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


def _observables_context(metadata):
    """A context carrying only the observable that gates the verdict block."""
    import types
    derived = {
        "WARNING:accuracy_targets_missed": False,
        "DIAG:accuracy_tperc_uncertainty": float("nan"),
        "DIAG:accuracy_tfinal_uncertainty": float("nan"),
        "DIAG:accuracy_rh_uncertainty": float("nan"),
        "WARNING:action_rescue_attempted": False,
        "WARNING:action_rescue_failed": False,
        "DIAG:action_rescue_attempts": 0,
    }
    return types.SimpleNamespace(
        derived_param_names={"Tnuc_SM_GeV"},
        derived_params=derived,
        pot=types.SimpleNamespace(conversionFactor=1.0),
        tr=types.SimpleNamespace(Tnuc=42.0),
        verbose=False,
    ), derived


def _populate(metadata):
    """Drive the observables consumer, not the dataclass, with one metadata snapshot."""
    import types
    from transitionlistener import transitionObservables as tobs

    ctx, derived = _observables_context(metadata)
    percolation = types.SimpleNamespace(Tperc=40.0, metadata=metadata)
    tobs.TransitionObservables._populate_temperature_observables(None, ctx, percolation)
    return derived


class AccuracyVerdictReachesTheObservablesTests(unittest.TestCase):
    """The hop the dataclass tests above cannot see: metadata -> derived parameters.

    `PercolationDiagnostics` carrying the verdict is not the same statement as the scan
    table carrying it. Driving only `from_state` leaves
    `_populate_temperature_observables` free to drop the verdict on the floor, which is
    what it did before these four names were registered, and a test of the dataclass alone
    cannot tell the difference.
    """

    def test_a_missed_verdict_raises_the_warning(self):
        derived = _populate(_build({"targets_met": False, "Tperc_uncertainty": 0.011,
                                    "Tf_uncertainty": 0.022, "RH_uncertainty": 0.276}))
        self.assertIs(derived["WARNING:accuracy_targets_missed"], True)

    def test_a_met_verdict_leaves_the_warning_down(self):
        derived = _populate(_build({"targets_met": True, "Tperc_uncertainty": 1e-3,
                                    "Tf_uncertainty": 1e-3, "RH_uncertainty": 1e-3}))
        self.assertIs(derived["WARNING:accuracy_targets_missed"], False)

    def test_no_verdict_leaves_the_warning_down_and_the_uncertainties_nan(self):
        """The third state: false beside nan, which is what an unassessed point reads as."""
        derived = _populate(_build(None))
        self.assertIs(derived["WARNING:accuracy_targets_missed"], False)
        for name in ("DIAG:accuracy_tperc_uncertainty", "DIAG:accuracy_tfinal_uncertainty",
                     "DIAG:accuracy_rh_uncertainty"):
            self.assertTrue(math.isnan(float(derived[name])), f"{name} = {derived[name]}")

    def test_the_three_uncertainties_reach_the_columns(self):
        """Without these the warning alone cannot separate 'met' from 'never judged'."""
        derived = _populate(_build({"targets_met": False, "Tperc_uncertainty": 0.011,
                                    "Tf_uncertainty": 0.022, "RH_uncertainty": 0.276}))
        self.assertAlmostEqual(derived["DIAG:accuracy_tperc_uncertainty"], 0.011)
        self.assertAlmostEqual(derived["DIAG:accuracy_tfinal_uncertainty"], 0.022)
        self.assertAlmostEqual(derived["DIAG:accuracy_rh_uncertainty"], 0.276)

    def test_a_metadata_object_without_the_fields_is_tolerated(self):
        """An older metadata snapshot must not turn a solved point into a failed one."""
        import types
        derived = _populate(types.SimpleNamespace())
        self.assertIs(derived["WARNING:accuracy_targets_missed"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
