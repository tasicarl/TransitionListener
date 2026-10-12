"""A completed action evaluation must count as completed, whatever its value.

The controller's refinement budget is

    remaining_budget = n_action_max - _controller_added_support_points(bank, free_bank)

where the free bank holds the support temperatures whose action is already paid for.
It is built from ``_transition_action_outdict_temperatures``, which drops every entry
with a non-finite action. A temperature whose bounce solve ran and returned a
non-finite action is therefore never free: it stays on the pending side of that
subtraction for the rest of the run, even though its cost is sunk and re-asking for it
only re-reads the cache.

Below the transition the bounce stops existing, so a cold-tail walk that oversteps
collects such temperatures in bulk. Measured on the 2HDM line scan at
lambda3 = 5.8123128723: 60 of 106 support points carried a non-finite action, which is
``n_action_max`` exactly, so ``remaining_budget`` was zero and every refinement path
was closed. The profile was then rejected with error 10 for an unresolved
Delta P = 0.15 inside the transition window -- a window that held 6 of the 106 points.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import bubbledynamics as bd
from transitionlistener import percolation_adaptivestepsize as pas


def settings(**overrides):
    values = dict(n_action_max=60, n_action_increment=5, maxit=20, f_start=1.0e-3,
                  f_final=0.99, rel_increment=0.1)
    values.update(overrides)
    return types.SimpleNamespace(**values)


# Temperatures where the solve succeeded, and cold ones where it ran and returned NaN.
# The payloads carry no vevs, so the phase-matching branch of the helper is not reached
# and no phase objects are needed.
SOLVED = [76.4, 62.0, 41.1, 31.7, 25.4, 20.3]
FAILED = [16.3, 8.1, 4.1, 0.18, 7.9e-3, 3.8e-6]


def outdict_with(solved=SOLVED, failed=FAILED):
    out = {float(T): {"action": 150.0 + i, "trantype": 1} for i, T in enumerate(solved)}
    out.update({float(T): {"action": np.nan, "trantype": 1} for T in failed})
    return out


class SettledActionAccountingTests(unittest.TestCase):

    def test_a_settled_non_finite_action_is_reported_as_settled(self):
        """The helper must be able to answer 'has this temperature been attempted?'."""
        out = outdict_with()
        settled = bd._transition_action_outdict_temperatures(
            out, None, None, tmin=1.0e-9, tmax=200.0, require_finite_action=False,
        )
        for T in FAILED:
            self.assertTrue(
                np.any(np.isclose(settled, T, rtol=0.0, atol=1e-9)),
                f"the settled-action list left out T = {T} GeV, whose solve already ran",
            )

    def test_the_usable_action_list_still_holds_only_finite_actions(self):
        """Control: everything that consumes actions must keep seeing finite ones only."""
        out = outdict_with()
        usable = bd._transition_action_outdict_temperatures(
            out, None, None, tmin=1.0e-9, tmax=200.0,
        )
        self.assertEqual(sorted(np.asarray(usable, dtype=float)), sorted(SOLVED))

    def test_a_cold_tail_of_failed_solves_leaves_refinement_budget(self):
        """The intent: failed cold solves must not close every refinement path.

        This is the production expression, driven through the function that refreshes
        the free bank, so it fails if either half of the accounting regresses.
        """
        out = outdict_with()
        active = np.asarray(SOLVED + FAILED, dtype=float)
        bank = active.copy()
        # The measured run had 60 failed cold solves against n_action_max = 60, so the
        # budget went to exactly zero. Scale the limit to the same ratio here.
        conf = settings(n_action_max=len(FAILED))

        _, _, bank_after, free_after = pas._apply_dynamiczoomwindow_candidate_grid(
            active,
            np.asarray([], dtype=float),
            bank,
            np.asarray([], dtype=float),
            conf,
            out,
            None,
            None,
            tmin=1.0e-9,
            tmax=200.0,
            stage="step 3",
            reset_arrays=lambda *a, **k: None,
            note_rebuild=lambda *a, **k: None,
        )
        pending = pas._controller_added_support_points(bank_after, free_after)
        remaining = max(int(conf.n_action_max) - int(pending), 0)
        self.assertGreater(
            remaining, 0,
            f"{len(FAILED)} cold temperatures whose solve had already run and failed "
            f"were still counted as pending, leaving remaining_budget = {remaining}; "
            "the controller cannot refine the transition window in that state",
        )

    def test_points_with_no_action_attempt_are_still_pending(self):
        """Control: a point that was never handed to the solver must stay pending."""
        out = outdict_with(failed=[])
        never_tried = [16.3, 8.1, 4.1]
        bank = np.asarray(SOLVED + never_tried, dtype=float)
        free = bd._transition_action_outdict_temperatures(
            out, None, None, tmin=1.0e-9, tmax=200.0, require_finite_action=False,
        )
        self.assertEqual(pas._controller_added_support_points(bank, free), len(never_tried))


if __name__ == "__main__":
    unittest.main()
