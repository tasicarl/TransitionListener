"""An exempted post-peak repair must not take the anchor remesh's turn.

``_build_dynamiczoomwindow_unresolved_jump_grid`` exempts an *acceptance-fatal* jump from
the post-peak inert veto, because leaving such a jump unrepaired guarantees rejection
(see ``test_fatal_jump_is_always_repairable``). That exemption introduced a second
problem: ``_try_dynamiczoomwindow_completed_refinement`` offers the direct jump repair
*before* the anchor remesh, so the newly permitted repair consumes the refinement slot
the remesh needed.

Measured on the 2HDM edge line. At lambda3 = 5.8177725978 the exempted repair is the only
thing that resolves the profile. At lambda3 = 5.8136778037 it is the opposite: the repair
preempted an ``anchor_remesh`` (2 remeshes became 1), the run ended one support update
short, and the profile was rejected at ``Delta P = 0.057785`` over the 0.05 gate at
T = [35.8, 36.169] GeV - a point that resolved before the exemption existed. The two
couplings looked like an either/or for three configurations until the ordering was found
to be the whole difference.

So the exemption is kept and the *priority* is corrected: where the repair exists only by
exemption, the remesh is offered first and the direct repair stays available as the
fallback. Both couplings then resolve.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest

import sys
from pathlib import Path

import numpy as np

from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders
from transitionlistener import percolation_adaptivestepsize as adaptive

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The measured 29-point bank slice lives with the test that established it; both tests
# need exactly the same profile, so it is shared rather than copied out of step.
from test_fatal_jump_is_always_repairable import post_peak_profile, settings  # noqa: E402


class ExemptedRepairOrderingTests(unittest.TestCase):

    def setUp(self):
        self.T, self.P, self.S, self.H = post_peak_profile()
        # The shared helper carries only what the jump builder asks for. The remesh
        # builder also works from f_perc, and the rate repair needs its own budget and
        # step limit, so those are supplied here at their shipped defaults.
        self.settings = settings(f_perc=0.28957, n_action_max=60, max_log10_rate_step=12.0)

    # ---- the predicate identifies exactly the exempted case ----

    def test_the_exempted_case_is_recognised(self):
        """Premise: this profile's fatal jump is refinable only by the exemption."""
        self.assertTrue(
            gridbuilders._acceptance_jump_is_exempted_inert(
                self.T, self.P, self.settings, actions=self.S, hubble=self.H, vw=1.0),
            "the measured profile is the exempted-inert case; if this is False the "
            "ordering fix can never be reached and the test is vacuous",
        )

    def test_a_non_post_peak_jump_is_not_treated_as_exempted(self):
        """Control: a fatal jump that the veto never touched must keep its priority.

        The direct repair is the right first move for a jump the controller can actually
        fix; only the exempted ones must yield to the remesh. Here the rate rises
        monotonically towards the cold end, so no interval is post-peak.
        """
        S = np.linspace(20000.0, 4000.0, self.T.size)
        self.assertFalse(
            gridbuilders._acceptance_jump_is_exempted_inert(
                self.T, self.P, self.settings, actions=S, hubble=self.H, vw=1.0),
            "a jump that is not post-peak was reported as exempted, which would hand the "
            "remesh priority over repairs that the veto never blocked",
        )

    def test_a_profile_without_a_fatal_jump_is_not_exempted(self):
        """Control: no acceptance-fatal jump means nothing to exempt."""
        P = np.linspace(0.0, 0.30316421, self.T.size)
        self.assertFalse(
            gridbuilders._acceptance_jump_is_exempted_inert(
                self.T, P, self.settings, actions=self.S, hubble=self.H, vw=1.0),
            "a smooth profile was reported as having an exempted fatal jump",
        )

    # ---- the ordering itself ----

    def _first_applied_reason(self, remesh_available: bool):
        """Run the refinement once and report which grid it spent its slot on.

        The rate-step repair comes before the jump step in the refinement order and has
        work to do on this profile, so it is suppressed here: the question under test is
        which of the *jump repair* and the *anchor remesh* gets the slot, and leaving the
        rate repair in place would answer a different one. That the rate repair keeps its
        priority is the subject of ``test_the_rate_repair_keeps_its_priority``.
        """
        applied: list[str] = []

        def apply_candidate_grid(grid, reason):
            applied.append(reason)
            return True

        real_remesh = gridbuilders._build_dynamiczoomwindow_anchor_remesh_grid

        def remesh(*args, **kwargs):
            if not remesh_available:
                return None, None
            grid, reason = real_remesh(*args, **kwargs)
            if grid is None:
                # Stand in for a remesh that has work to do, so that the ordering is what
                # this test measures rather than the remesh builder's own verdict.
                return np.array([36.0, 35.9]), "anchor_remesh"
            return grid, reason

        original = adaptive._build_dynamiczoomwindow_anchor_remesh_grid
        original_rate = adaptive._try_dynamiczoomwindow_rate_refinement
        adaptive._build_dynamiczoomwindow_anchor_remesh_grid = remesh
        adaptive._try_dynamiczoomwindow_rate_refinement = lambda *a, **k: False
        try:
            adaptive._try_dynamiczoomwindow_completed_refinement(
                self.T, self.H, self.P, self.S, self.settings, None,
                remaining_budget=40, previous_estimate=None, iterator=1,
                stage="step 2", vw=1.0,
                apply_candidate_grid=apply_candidate_grid,
                promote_cached_action_points=lambda: False,
            )
        finally:
            adaptive._build_dynamiczoomwindow_anchor_remesh_grid = original
            adaptive._try_dynamiczoomwindow_rate_refinement = original_rate
        return applied[0] if applied else None

    def test_the_rate_repair_keeps_its_priority(self):
        """The reordering applies to the jump step only.

        A coarse nucleation rate invalidates every later verdict, so its repair outranks
        both the remesh and the jump repair. An earlier form of this fix offered the
        remesh before the whole refinement order and so outranked the rate repair too;
        on the measured couplings that made no difference, because the rate repairs are
        finished long before any acceptance-fatal jump exists, but nothing guarantees
        that in general.
        """
        applied: list[str] = []
        original = adaptive._build_dynamiczoomwindow_anchor_remesh_grid
        adaptive._build_dynamiczoomwindow_anchor_remesh_grid = (
            lambda *a, **k: (np.array([36.0, 35.9]), "anchor_remesh"))
        try:
            adaptive._try_dynamiczoomwindow_completed_refinement(
                self.T, self.H, self.P, self.S, self.settings, None,
                remaining_budget=40, previous_estimate=None, iterator=1,
                stage="step 2", vw=1.0,
                apply_candidate_grid=lambda g, r: applied.append(r) or True,
                promote_cached_action_points=lambda: False,
            )
        finally:
            adaptive._build_dynamiczoomwindow_anchor_remesh_grid = original
        self.assertEqual(
            applied[:1], ["rate_step_refine"],
            "the remesh was spent before the rate repair, which the reordering must not "
            f"outrank; the refinement applied {applied} instead",
        )

    def test_the_remesh_gets_the_slot_when_the_repair_is_only_exempted(self):
        """The fix: the remesh is spent first, so it is not preempted."""
        self.assertEqual(
            self._first_applied_reason(remesh_available=True), "anchor_remesh",
            "the exempted direct repair took the refinement slot; this is what ended the "
            "lambda3 = 5.8136778037 run one support update short of its gate",
        )

    def test_the_direct_repair_is_still_the_fallback(self):
        """The exemption must survive: with no remesh to do, the repair still happens.

        This is the half that lambda3 = 5.8177725978 depends on - if the ordering fix
        suppressed the repair instead of deferring it, that point would go back to
        accepting a profile the gate then rejects.
        """
        self.assertEqual(
            self._first_applied_reason(remesh_available=False), "jump_refine",
            "with no remesh available the exempted repair was not attempted either, so "
            "the fatal jump is once again unrepairable",
        )


if __name__ == "__main__":
    unittest.main()
