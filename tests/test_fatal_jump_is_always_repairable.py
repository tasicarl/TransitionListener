"""A jump the acceptance test will reject must be repairable.

``_raise_if_large_delta_p_underresolved`` rejects a profile whose neighbouring ``Delta P``
exceeds the strict threshold in the ``P < 0.2`` region.
``_build_dynamiczoomwindow_unresolved_jump_grid`` builds the support that repairs such a
jump - but it used to return ``None`` for any
interval that ``_rate_interval_is_post_peak_inert`` declared inert, i.e. any interval
colder than the nucleation-rate peak.

Those two rules contradict each other. Measured on the 2HDM edge line at
lambda3 = 5.8177725978: the offending interval [28.745, 30.453] GeV lies below the rate
peak at 41.05 GeV, so the controller drove max ``Delta P`` from 0.85 down to 0.12, found
nothing further it was permitted to do, and accepted the profile with **119 budget
unspent** - whereupon the gate rejected it. Four separate budget- and ordering-based
fixes produced bit-identical runs, because the budget was never the constraint.

"Post-peak" cannot excuse a coarse ``P(T)`` in any case: ``R_H`` is sensitive to ``P(T)``
for ``T > Tperc`` and ``T_final`` to ``P(T)`` for ``T > T_final``, and the interval above
spans from 0.05 GeV below Tperc to 1.66 GeV above it, so about 97% of it lies inside
``R_H``'s integration range. Small ``Delta P`` at *all* temperatures is the requirement,
which is why the gate is the right handle and the repair must not be vetoed.

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
    values = dict(large_delta_p_refine_threshold=0.1, large_delta_p_success_threshold=0.2,
                  n_action_increment=5, f_start=1.0e-3, f_final=0.99, maxit=25)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def post_peak_profile():
    """The measured profile, as a 29-point slice of the real support bank.

    Taken verbatim from the all+8 run at lambda3 = 5.8177725978
    (``diag/targets_all8/state_58177725978.json``), trimmed to the smallest contiguous
    slice that still reproduces both conditions the bug needs: the acceptance criterion
    selects [28.745, 30.453] GeV in strict ``P < 0.2`` mode, and
    ``_rate_interval_is_post_peak_inert`` calls that interval inert because the rate peaks
    at 41.05 GeV. A synthetic profile was tried first and did not satisfy the second
    condition, so it could not have exercised the bug at all.
    """
    T = np.array([61.758439, 61.708759, 61.653040, 61.647112, 61.597322, 61.591449,
                  61.591449, 61.591449, 61.591449, 61.591449, 61.591449, 55.139201,
                  49.362884, 44.191687, 42.591271, 41.048814, 39.562218, 39.439019,
                  39.076923, 38.960668, 37.809584, 37.029875, 36.769022, 35.417727,
                  34.080605, 32.743483, 31.707408, 30.452490, 28.745093])
    S = np.array([15708.972798, 15668.818157, 15624.306710, 15619.629031, 15579.516823,
                  15574.912933, 15574.673704, 15575.099584, 15575.072306, 15574.765014,
                  15575.073560, 11389.958613, 8807.980336, 7061.410106, 6591.999463,
                  6155.649102, 5969.047633, 5965.084770, 5949.940065, 5918.753109,
                  5999.325313, 5967.200588, 5956.533481, 5912.134077, 5882.576219,
                  5848.707062, 5811.726261, 5816.099510, 5798.298636])
    H = np.array([5.473785e-15, 5.465509e-15, 5.456236e-15, 5.455250e-15, 5.446973e-15,
                  5.445997e-15, 5.445997e-15, 5.445997e-15, 5.445997e-15, 5.445997e-15,
                  5.445997e-15, 4.439133e-15, 3.658817e-15, 3.065893e-15, 2.902673e-15,
                  2.754552e-15, 2.620446e-15, 2.609716e-15, 2.578519e-15, 2.568610e-15,
                  2.473359e-15, 2.411745e-15, 2.391632e-15, 2.291788e-15, 2.199392e-15,
                  2.112447e-15, 2.047498e-15, 1.970974e-15, 1.868523e-15])
    P = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  0.00000003, 0.00000515, 0.00030067, 0.00037743, 0.00069125, 0.00082492,
                  0.00340382, 0.00701343, 0.00867585, 0.02209118, 0.04580787, 0.08337835,
                  0.12401933, 0.18837080, 0.30316421])
    return T, P, S, H


class FatalJumpIsRepairableTests(unittest.TestCase):

    def setUp(self):
        self.T, self.P, self.S, self.H = post_peak_profile()
        self.settings = settings()

    def test_the_interval_is_the_one_the_gate_would_reject_on(self):
        """Premise: the acceptance criterion selects the post-peak interval."""
        interval, threshold, mode = gridbuilders._preferred_probability_jump_interval(
            self.T, self.P, self.settings, for_acceptance=True)
        self.assertIsNotNone(interval, "the test profile has no acceptance-fatal jump")
        self.assertEqual(mode, "strict P<0.2",
                         "the jump must fall in the strict region for this test to mean anything")
        self.assertGreater(interval[4], threshold)

    def test_the_interval_really_is_post_peak(self):
        """Premise: without the fix, the inert veto is what fires."""
        interval, _t, _m = gridbuilders._preferred_probability_jump_interval(
            self.T, self.P, self.settings, for_acceptance=True)
        jump_high, jump_low = interval[0], interval[1]
        self.assertTrue(
            gridbuilders._rate_interval_is_post_peak_inert(
                self.T, self.S, self.H, jump_high, jump_low, self.settings, vw=1.0),
            "the test profile's fatal jump is not post-peak, so it cannot exercise the bug",
        )

    def test_an_acceptance_fatal_jump_gets_a_repair_grid(self):
        """The fix: refinement must offer support for a jump that would be fatal."""
        grid = gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
            self.T, self.P, self.settings, max_new_points=40,
            actions=self.S, hubble=self.H, vw=1.0)
        self.assertIsNotNone(
            grid,
            "refinement declined to repair the interval the acceptance test rejects on; "
            "the controller can then only accept a profile that will be thrown away",
        )
        pts = np.asarray(grid, dtype=float)
        self.assertGreater(pts.size, 0)
        self.assertTrue(
            np.all((pts > 28.7451) & (pts < 30.4525)),
            f"the repair points {pts} fall outside the offending interval",
        )

    def test_the_exemption_survives_moving_the_strict_region(self):
        """The exemption must key on *which criterion selected the jump*, not on a label.

        ``_STRICT_JUMP_P_MAX`` and ``_STRICT_JUMP_APPLY_GLOBALLY`` exist to be moved in
        threshold studies, and the mode string is built from them. A comparison against the
        rendered ``"strict P<0.2"`` therefore matches only at today's values, and the first
        study that moves either would silently restore the contradiction above with nothing
        failing to say so.
        """
        with mock.patch.object(gridbuilders, "_STRICT_JUMP_P_MAX", 0.3):
            _interval, _threshold, mode = gridbuilders._preferred_probability_jump_interval(
                self.T, self.P, self.settings, for_acceptance=True)
            self.assertEqual(mode, "strict P<0.3",
                             "the label did not follow the constant, so this test is inert")
            grid = gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
                self.T, self.P, self.settings, max_new_points=40,
                actions=self.S, hubble=self.H, vw=1.0)
        self.assertIsNotNone(
            grid,
            "moving the strict region turned the acceptance-fatal jump back into an "
            "unrepairable one, so the exemption is keyed on a string rather than on the "
            "criterion that selected the interval",
        )

    def test_a_merely_refinable_post_peak_jump_is_still_declined(self):
        """Control: the inert veto must keep working where the jump is not fatal.

        Same geometry, but ``P`` stays above 0.2 across the jump so the strict mode does
        not apply and the jump is only a refinement target, not an acceptance failure.
        """
        # Shift the whole profile above 0.2 so the strict region does not apply, and widen
        # the final jump past the generic success threshold so it is still selected.
        P = np.clip(self.P + 0.25, 0.0, 1.0)
        P[-1] = 0.95
        interval, _t, mode = gridbuilders._preferred_probability_jump_interval(
            self.T, P, self.settings, for_acceptance=True)
        self.assertIsNotNone(interval, "the control profile has no selectable jump")
        self.assertNotEqual(mode, "strict P<0.2",
                            "the control must exercise the non-strict path")
        self.assertTrue(
            gridbuilders._rate_interval_is_post_peak_inert(
                self.T, self.S, self.H, interval[0], interval[1], self.settings, vw=1.0),
            "the control jump must still be post-peak for the veto to be under test",
        )
        grid = gridbuilders._build_dynamiczoomwindow_unresolved_jump_grid(
            self.T, P, self.settings, max_new_points=40,
            actions=self.S, hubble=self.H, vw=1.0)
        self.assertIsNone(
            grid,
            "the inert veto stopped applying to a post-peak jump that is not fatal, which "
            "would spend budget where it cannot change the verdict",
        )


if __name__ == "__main__":
    unittest.main()
