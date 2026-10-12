"""Reaching the support-update limit is only worth reporting if something was outstanding.

``_try_dynamiczoomwindow_completed_refinement`` is entered for every profile that brackets
the transition, resolved or not. At ``iterator >= maxit`` both repair paths are skipped, the
accuracy estimate is taken on whatever grid is in hand and the caller is told "accept", with
nothing in the output to say that the controller had stopped being allowed to refine. That is
worth a warning -- but only where a repair really was waiting. A profile that reaches the
limit with nothing left to repair is converged, and warning on it would fire on every run
that happens to use all its iterations, which is the same defect as a rejection message
asserting a budget was spent when it was not (see
``tests/test_rejection_names_the_real_limit.py``).

So the message is built from a probe: the same builders are asked, in the same order, with
an apply hook that records the offer and refuses it. Nothing is spent, and the message cannot
drift out of step with the order actually used.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

from transitionlistener import percolation_adaptivestepsize as adaptive

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The same measured 29-point bank slice the jump tests use, so the two agree about what an
# unresolved profile looks like.
from test_fatal_jump_is_always_repairable import post_peak_profile, settings  # noqa: E402


class MaxitWarningTests(unittest.TestCase):

    def setUp(self):
        self.T, self.P, self.S, self.H = post_peak_profile()
        self.settings = settings(f_perc=0.28957, n_action_max=60, max_log10_rate_step=12.0,
                                 maxit=10)

    def _run(self, iterator, suppress_repairs=False):
        """Return (printed lines, reasons actually applied) for one refinement pass."""
        applied: list[str] = []
        saved: dict[str, object] = {}

        def apply_candidate_grid(grid, reason):
            applied.append(reason)
            return True

        if suppress_repairs:
            saved = {
                "_try_dynamiczoomwindow_rate_refinement":
                    adaptive._try_dynamiczoomwindow_rate_refinement,
                "_build_dynamiczoomwindow_hot_shoulder_grid":
                    adaptive._build_dynamiczoomwindow_hot_shoulder_grid,
                "_build_dynamiczoomwindow_unresolved_jump_grid":
                    adaptive._build_dynamiczoomwindow_unresolved_jump_grid,
                "_build_dynamiczoomwindow_anchor_remesh_grid":
                    adaptive._build_dynamiczoomwindow_anchor_remesh_grid,
            }
            adaptive._try_dynamiczoomwindow_rate_refinement = lambda *a, **k: False
            adaptive._build_dynamiczoomwindow_hot_shoulder_grid = lambda *a, **k: (None, None)
            adaptive._build_dynamiczoomwindow_unresolved_jump_grid = lambda *a, **k: None
            adaptive._build_dynamiczoomwindow_anchor_remesh_grid = lambda *a, **k: (None, None)

        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):
                adaptive._try_dynamiczoomwindow_completed_refinement(
                    self.T, self.H, self.P, self.S, self.settings, None,
                    remaining_budget=40, previous_estimate=None, iterator=iterator,
                    stage="step 3", vw=1.0,
                    apply_candidate_grid=apply_candidate_grid,
                    promote_cached_action_points=lambda *a, **k: False,
                )
        finally:
            for name, value in saved.items():
                setattr(adaptive, name, value)
        printed = stream.getvalue().splitlines()
        return printed, applied

    def test_an_outstanding_repair_at_the_limit_is_reported_and_named(self):
        printed, applied = self._run(iterator=10)
        warnings = [line for line in printed if "support updates with" in line]
        self.assertEqual(
            len(warnings), 1,
            f"the iteration limit was reached with repairs waiting and nothing said so; "
            f"the output was {printed}",
        )
        self.assertIn("still outstanding", warnings[0])
        self.assertIn("limit of 10 support updates", warnings[0])
        self.assertTrue(
            any(reason in warnings[0]
                for reason in ("rate_step_refine", "hot_shoulder_refine", "jump_refine",
                               "anchor_remesh")),
            f"the warning named no repair at all: {warnings[0]!r}",
        )
        self.assertEqual(
            applied, [],
            "the probe spent support instead of only recording what was on offer",
        )

    def test_nothing_outstanding_at_the_limit_is_not_reported(self):
        """The defect: a converged profile that used all its iterations is not a problem."""
        printed, applied = self._run(iterator=10, suppress_repairs=True)
        self.assertEqual(
            [line for line in printed if "support updates with" in line], [],
            f"the limit was reported although no repair was available; output {printed}",
        )
        self.assertEqual(applied, [])

    def test_below_the_limit_nothing_is_reported(self):
        """Control: with iterations left there is no limit to report, only refinement."""
        printed, applied = self._run(iterator=1)
        self.assertEqual(
            [line for line in printed if "support updates with" in line], [],
            f"the limit was reported while the controller still had iterations: {printed}",
        )
        self.assertTrue(applied, "the refinement did nothing at all below the limit")


if __name__ == "__main__":
    unittest.main()
