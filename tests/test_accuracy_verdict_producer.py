"""Change 24, the producer: the controller must compute the verdict, not assert it.

The companion file tests the dataclass hop, which a stub that hardcoded
``estimate["targets_met"] = True`` would survive. This one drives the real branch in
``_try_dynamiczoomwindow_completed_refinement`` and checks the verdict tracks the targets.

What it must NOT assert: that the verdict gates acceptance. It does not -- both exits of that
function return False, which the caller reads as "accept" -- so the targets are advisory and
the verdict is only a report. A test asserting otherwise would encode a falsehood.
"""
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


def _estimate(rh_unc):
    return {"Tperc": 40.0, "Tf": 30.0, "RH": 0.05,
            "Tperc_uncertainty": 1e-3, "Tf_uncertainty": 1e-3, "RH_uncertainty": rh_unc,
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

    def test_verdict_is_false_without_a_comparable_previous_estimate(self):
        """One sample carries no information about its own convergence."""
        _retry, est = _call(_estimate(1e-3), previous=None)
        self.assertIs(est["targets_met"], False)

    def test_the_verdict_does_not_gate_acceptance(self):
        """Both exits accept. Pinned so a future change cannot quietly make it binding."""
        retry_met, _ = _call(_estimate(1e-3), previous=_estimate(1e-3))
        retry_missed, _ = _call(_estimate(0.276), previous=_estimate(0.276))
        self.assertIs(retry_met, False)
        self.assertIs(retry_missed, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
